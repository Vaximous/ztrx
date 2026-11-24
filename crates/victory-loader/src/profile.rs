//! A sampling profiler for the whole process, for finding where a frame's
//! CPU time goes: `VICTORY_PROFILE=<start s>,<seconds>[,<interval ms>]`
//! samples every thread's stack that often between those times, then prints
//! per thread its CPU time and the functions it ran in (leaf and inclusive)
//! and the stacks it waited in. Functions are named by dbghelp where it
//! can (system DLLs' exports), else `module+rva`; our own image's
//! addresses can be named afterwards with `llvm-symbolizer --obj=victory.exe`
//! (image base 0x140000000).

use std::collections::HashMap;
use std::ptr::null_mut;

use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    RtlLookupFunctionEntry, SYMBOL_INFOW, SYMOPT_DEFERRED_LOADS, SYMOPT_UNDNAME, SymFromAddrW,
    SymInitializeW, SymSetOptions,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId, GetThreadTimes, OpenThread,
    THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::locate;
use crate::stacks::{frames_of, name_of, start_of};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryThreadCycleTime(thread: HANDLE, cycles: *mut u64) -> i32;
}

/// Frames of a waiting stack kept as its signature.
const WAIT_FRAMES: usize = 8;
/// Frames of a running stack kept as its signature.
const RUN_FRAMES: usize = 6;
/// Cycles between samples above which a thread counts as running.
const RAN_CYCLES: u64 = 500_000;

/// One sampled thread.
struct Sampled {
    handle: HANDLE,
    name: String,
    start: usize,
    cycles: u64,
    cpu_at_start: u64,
    samples: u32,
    running: u32,
    /// Running samples by the function they were in (its start address).
    leaf: HashMap<u64, u32>,
    /// Running samples by every function on the stack, once per sample.
    inclusive: HashMap<u64, u32>,
    /// Running samples by their innermost few functions.
    run_stacks: HashMap<Vec<u64>, u32>,
    /// Waiting samples by their innermost few functions.
    wait_stacks: HashMap<Vec<u64>, u32>,
}

/// Reads `VICTORY_PROFILE` and, if set, starts the sampling thread.
pub fn profile_on_schedule() {
    let Ok(spec) = std::env::var("VICTORY_PROFILE") else {
        return;
    };
    let Some((start, length, interval)) = parse(&spec) else {
        eprintln!("VIctory: VICTORY_PROFILE wants <start s>,<seconds>[,<interval ms>]");
        return;
    };
    let _ = std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(move || {
            std::thread::sleep(start);
            run(length, interval);
        });
}

fn parse(
    spec: &str,
) -> Option<(
    std::time::Duration,
    std::time::Duration,
    std::time::Duration,
)> {
    let mut parts = spec.split(',').map(|p| p.trim().parse::<f64>().ok());
    let start = parts.next()??;
    let length = parts.next()??;
    let interval = parts.next().flatten().unwrap_or(5.0);
    let secs = |s: f64| std::time::Duration::try_from_secs_f64(s.max(0.0)).ok();
    Some((secs(start)?, secs(length)?, secs(interval / 1000.0)?))
}

fn thread_ids() -> Vec<u32> {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    let mut ids = Vec::new();
    if snapshot == INVALID_HANDLE_VALUE {
        return ids;
    }
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    ids
}

/// User plus kernel time, in 100 ns units.
fn cpu_time(thread: HANDLE) -> u64 {
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut a, mut b, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: writes four FILETIMEs.
    if unsafe { GetThreadTimes(thread, &mut a, &mut b, &mut kernel, &mut user) } == 0 {
        return 0;
    }
    let t = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32) | u64::from(f.dwLowDateTime);
    t(kernel) + t(user)
}

fn cycles(thread: HANDLE) -> u64 {
    let mut n = 0;
    // SAFETY: writes one u64.
    unsafe { QueryThreadCycleTime(thread, &mut n) };
    n
}

/// The start of the function holding `pc`, or `pc` itself when no unwind
/// table covers it.
fn function_of(pc: u64) -> u64 {
    let mut image = 0;
    // SAFETY: a lookup in the process's function tables.
    let entry = unsafe { RtlLookupFunctionEntry(pc, &mut image, null_mut()) };
    if entry.is_null() {
        return pc;
    }
    // SAFETY: a live entry of `image`'s table.
    image + u64::from(unsafe { (*entry).BeginAddress })
}

fn run(length: std::time::Duration, interval: std::time::Duration) {
    let began = std::time::Instant::now();
    let mut threads: HashMap<u32, Sampled> = HashMap::new();
    let mut listed = began - std::time::Duration::from_secs(10);
    let mut ticks = 0u64;
    eprintln!("VIctory: profiling for {length:?} every {interval:?}");
    while began.elapsed() < length {
        let tick = std::time::Instant::now();
        if listed.elapsed() >= std::time::Duration::from_secs(1) {
            listed = tick;
            for id in thread_ids() {
                threads.entry(id).or_insert_with(|| {
                    // SAFETY: opens a thread of our own process.
                    let handle = unsafe {
                        OpenThread(
                            THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                            0,
                            id,
                        )
                    };
                    Sampled {
                        handle,
                        name: if handle.is_null() {
                            String::new()
                        } else {
                            name_of(handle)
                        },
                        start: if handle.is_null() {
                            0
                        } else {
                            start_of(handle)
                        },
                        cycles: if handle.is_null() { 0 } else { cycles(handle) },
                        cpu_at_start: if handle.is_null() {
                            0
                        } else {
                            cpu_time(handle)
                        },
                        samples: 0,
                        running: 0,
                        leaf: HashMap::new(),
                        inclusive: HashMap::new(),
                        run_stacks: HashMap::new(),
                        wait_stacks: HashMap::new(),
                    }
                });
            }
        }
        ticks += 1;
        for t in threads.values_mut() {
            if t.handle.is_null() {
                continue;
            }
            let now = cycles(t.handle);
            // Suspending a waiting thread to sample it costs it some cycles
            // too (tens of thousands); a thread that ran for even a tenth
            // of a millisecond has far more.
            let ran = now.wrapping_sub(t.cycles) > RAN_CYCLES;
            t.cycles = now;
            let (frames, _) = frames_of(t.handle);
            if frames.is_empty() {
                continue;
            }
            t.samples += 1;
            let functions: Vec<u64> = frames.iter().map(|&pc| function_of(pc)).collect();
            if ran {
                t.running += 1;
                *t.leaf.entry(functions[0]).or_default() += 1;
                let mut seen = functions.clone();
                seen.sort_unstable();
                seen.dedup();
                for f in seen {
                    *t.inclusive.entry(f).or_default() += 1;
                }
                let key = functions.iter().take(RUN_FRAMES).copied().collect();
                *t.run_stacks.entry(key).or_default() += 1;
            } else {
                let key = functions.iter().take(WAIT_FRAMES).copied().collect();
                *t.wait_stacks.entry(key).or_default() += 1;
            }
        }
        std::thread::sleep(interval.saturating_sub(tick.elapsed()));
    }
    let seconds = began.elapsed().as_secs_f64();
    report(&mut threads, seconds, ticks);
}

/// Names `function` from the loaded PDBs, falling back to `module+rva`.
fn name(function: u64, cache: &mut HashMap<u64, String>) -> String {
    if let Some(n) = cache.get(&function) {
        return n.clone();
    }
    #[repr(C, align(8))]
    struct Buffer(SYMBOL_INFOW, [u16; 512]);
    // SAFETY: zeroed is valid for both parts.
    let mut buffer: Buffer = unsafe { std::mem::zeroed() };
    buffer.0.SizeOfStruct = size_of::<SYMBOL_INFOW>() as u32;
    buffer.0.MaxNameLen = 512;
    let mut displacement = 0;
    // SAFETY: the buffer holds MaxNameLen characters after the struct.
    let found = unsafe {
        SymFromAddrW(
            GetCurrentProcess(),
            function,
            &mut displacement,
            &mut buffer.0,
        )
    } != 0;
    let module = locate(function as usize);
    let text = if found {
        let len = (buffer.0.NameLen as usize).min(512);
        // SAFETY: NameLen characters were written from `Name` on.
        let chars = unsafe { std::slice::from_raw_parts(buffer.0.Name.as_ptr(), len) };
        let module = module.split('+').next().unwrap_or("").to_string();
        format!("{module}!{}", String::from_utf16_lossy(chars))
    } else {
        module
    };
    cache.insert(function, text.clone());
    text
}

fn top<K: Clone>(map: &HashMap<K, u32>, n: usize) -> Vec<(K, u32)> {
    let mut v: Vec<_> = map.iter().map(|(k, c)| (k.clone(), *c)).collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1));
    v.truncate(n);
    v
}

fn report(threads: &mut HashMap<u32, Sampled>, seconds: f64, ticks: u64) {
    // SAFETY: initialises dbghelp for this process, loading modules' symbols
    // on first use.
    unsafe {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        SymInitializeW(GetCurrentProcess(), std::ptr::null(), 1);
    }
    let mut names = HashMap::new();
    let mut order: Vec<(u32, f64)> = threads
        .iter()
        .map(|(&id, t)| {
            let cpu = if t.handle.is_null() {
                0.0
            } else {
                cpu_time(t.handle).saturating_sub(t.cpu_at_start) as f64 / 1e7
            };
            (id, cpu)
        })
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = order.iter().map(|o| o.1).sum();
    eprintln!(
        "VIctory profile: {seconds:.1} s, {ticks} ticks, {} threads, {total:.1} CPU s in all",
        threads.len()
    );
    for &(id, cpu) in &order {
        let t = &threads[&id];
        eprintln!(
            "profile thread {id} \"{}\" start {}: {cpu:.2} CPU s ({:.0}% of a core), {} samples, {} running",
            t.name,
            name(t.start as u64, &mut names),
            100.0 * cpu / seconds,
            t.samples,
            t.running
        );
    }
    for &(id, cpu) in order.iter().take(16) {
        let t = &threads[&id];
        if t.samples == 0 {
            continue;
        }
        let share = |c: u32, of: u32| 100.0 * f64::from(c) / f64::from(of.max(1));
        eprintln!(
            "profile detail thread {id} \"{}\": {cpu:.2} CPU s, running in {:.0}% of samples",
            t.name,
            share(t.running, t.samples)
        );
        for (f, c) in top(&t.leaf, 25) {
            eprintln!(
                "  leaf {:5.1}% {}",
                share(c, t.running),
                name(f, &mut names)
            );
        }
        for (f, c) in top(&t.inclusive, 40) {
            eprintln!(
                "  incl {:5.1}% {}",
                share(c, t.running),
                name(f, &mut names)
            );
        }
        for (stack, c) in top(&t.run_stacks, 12) {
            let text: Vec<String> = stack.iter().map(|&f| name(f, &mut names)).collect();
            eprintln!("  run  {:5.1}% {}", share(c, t.running), text.join(" < "));
        }
        for (stack, c) in top(&t.wait_stacks, 8) {
            let text: Vec<String> = stack.iter().map(|&f| name(f, &mut names)).collect();
            eprintln!("  wait {:5.1}% {}", share(c, t.samples), text.join(" < "));
        }
    }
    for t in threads.values() {
        if !t.handle.is_null() {
            // SAFETY: we opened it.
            unsafe { CloseHandle(t.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_with_a_default_interval() {
        let (s, l, i) = parse("60, 30").unwrap();
        assert_eq!((s.as_secs(), l.as_secs(), i.as_millis()), (60, 30, 5));
        let (_, _, i) = parse("1,2,10").unwrap();
        assert_eq!(i.as_millis(), 10);
        assert!(parse("x").is_none());
    }
}
