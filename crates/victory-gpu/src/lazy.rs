//! Copy-back left on the card until the CPU reads it
//! (`VICTORY_LAZY_BACK=1`, with the relay). First step: the probe.
//!
//! Every landing fetches the pieces of the copies in the card's memory
//! that the compare found changed and merges them into title memory: 66-76
//! MB a frame in the world (run zp1). Much of it is never read by the
//! title's CPU code. The probe (`VICTORY_READBACK_CHECK=1`) measures how
//! much is: after a landing has merged as before, the whole 4 KiB pages of
//! each merged piece are made `PAGE_NOACCESS`. The first touch of such a
//! page faults; the handler gives it its protection back and counts the
//! touch, by whether our code or another module (the title's) made it, and
//! the piece counts as read by the CPU for a while ([`decay`]). A piece the
//! next landing merges again while it is still untouched is one whose
//! fetch could have been left out. Title memory is merged as before, so
//! nothing the title reads changes: the probe only costs the faults and
//! the protection calls.
//!
//! Left alone: copies whose memory is not write-watched (the title's views
//! of physical pages, whose 64 KiB blocks `crate::watch` guards with
//! protections of its own) and pieces with no whole page.
//!
//! Kernel paths into title memory do not fault into a handler: our writes
//! through the kernel and the title's file reads (`crate::watch::writing`)
//! open the pages first ([`opening`]); a remapped range is forgotten
//! ([`forget`]). Reads of title memory through the kernel are not hooked:
//! the switch needs the guarded reads (`crate::guarded`), which fault.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect};

const PAGE: usize = 4096;

/// Whether the lazy copy-back is on (`VICTORY_LAZY_BACK=1`; needs the relay
/// and the guarded reads). Only its probe ([`probe`]) exists so far.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_LAZY_BACK").is_ok_and(|v| v == "1");
        if !on {
            return false;
        }
        if crate::shadow::relay_stage() < 2 || !crate::guarded::on() {
            tracing::warn!("VICTORY_LAZY_BACK=1 needs the relay (VICTORY_RELAY=2) and the guarded reads: left off");
            return false;
        }
        if !probe() {
            tracing::warn!("VICTORY_LAZY_BACK=1 has only its probe so far (VICTORY_READBACK_CHECK=1): left off");
            return false;
        }
        install();
        true
    })
}

/// `VICTORY_READBACK_CHECK=1`: everything is fetched and merged as before;
/// the pages are protected afterwards only to see what the CPU touches.
fn probe() -> bool {
    std::env::var("VICTORY_READBACK_CHECK").is_ok_and(|v| v == "1")
}

/// Frames after a touch during which a piece stays read by the CPU and is
/// not protected again (`VICTORY_LAZY_DECAY`, default 120).
fn decay() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_LAZY_DECAY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120))
}

/// Protected pages of one piece.
struct Run {
    end: usize,
    /// The protection the pages had.
    old: u32,
    /// The copy's number and the piece's.
    key: (u64, u32),
}

#[derive(Default)]
struct State {
    /// By first address.
    runs: BTreeMap<usize, Run>,
    /// Pieces the CPU touched, and the frame it last did.
    touched: HashMap<(u64, u32), u64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(s.get_or_insert_with(State::default))
}

/// Since the last frame line (bytes unless said):
const PROTECTED: usize = 0; // pages protected after a merge
const PROTECTED_N: usize = 1; // their pieces
const AGAIN: usize = 2; // merged again while untouched: fetches to spare
const AGAIN_N: usize = 3;
const OURS: usize = 4; // touched by our code
const OURS_N: usize = 5;
const TITLE: usize = 6; // touched by other code (the title's)
const TITLE_N: usize = 7;
const KERNEL: usize = 8; // opened for the kernel
const KERNEL_N: usize = 9;
const READ_LATELY: usize = 10; // not protected: touched within `decay`
const UNWATCHED: usize = 11; // not protected: physical-page views
const NO_PAGE: usize = 12; // not protected: no whole page
const FAILED: usize = 13; // protections that failed or found another protection
static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

fn add(i: usize, n: usize) {
    STATS[i].fetch_add(n as u64, Ordering::Relaxed);
}

/// A piece merged at a landing: the copy, offsets `[lo, hi)` in it, the
/// piece's number.
pub(crate) type Piece = (std::sync::Arc<crate::shadow::Shadow>, usize, usize, u32);

/// The whole pages of `[from, to)`.
fn inner(from: usize, to: usize) -> Option<(usize, usize)> {
    let (a, b) = (from.next_multiple_of(PAGE), to / PAGE * PAGE);
    (a < b).then_some((a, b))
}

fn title_range(piece: &Piece) -> (usize, usize) {
    let start = piece.0.title_start() as usize;
    (start + piece.1, start + piece.2)
}

/// Gives the runs overlapping `[from, to)` their protection back and takes
/// them out; calls `each` with each run's key and bytes.
fn open_in(s: &mut State, from: usize, to: usize, mut each: impl FnMut(&mut State, (u64, u32), usize)) {
    let starts: Vec<usize> = s
        .runs
        .range(..to)
        .rev()
        .take_while(|(_, r)| r.end > from)
        .map(|(&a, _)| a)
        .collect();
    for a in starts {
        if let Some(r) = s.runs.remove(&a) {
            let mut old = 0;
            // SAFETY: title pages we protected; only their protection changes.
            unsafe { VirtualProtect(a as *const _, r.end - a, r.old, &mut old) };
            each(s, r.key, r.end - a);
        }
    }
}

/// What a landing merges, held from before its merges until after them
/// ([`Landing::new`]); dropped, it protects the pieces.
pub(crate) struct Landing(Vec<Piece>);

impl Landing {
    /// Before the merges of a landing: pieces of `pieces` still protected
    /// are merged again untouched (counted), and opened silently so that
    /// the merge reads them.
    pub(crate) fn new(pieces: Vec<Piece>) -> Option<Landing> {
        if !on() || pieces.is_empty() {
            return None;
        }
        with(|s| {
            for piece in &pieces {
                let (from, to) = title_range(piece);
                let Some((a, b)) = inner(from, to) else {
                    continue;
                };
                let key = (piece.0.id(), piece.3);
                open_in(s, a, b, |_, k, n| {
                    if k == key {
                        add(AGAIN, n);
                        add(AGAIN_N, 1);
                    }
                });
            }
        });
        Some(Landing(pieces))
    }
}

impl Drop for Landing {
    fn drop(&mut self) {
        let frame = crate::queue::presented();
        let lately = decay();
        with(|s| {
            s.touched.retain(|_, &mut at| frame.saturating_sub(at) <= lately);
            for piece in &self.0 {
                let (from, to) = title_range(piece);
                let Some((a, b)) = inner(from, to) else {
                    add(NO_PAGE, to - from);
                    continue;
                };
                if piece.0.unwatched() {
                    add(UNWATCHED, b - a);
                    continue;
                }
                let key = (piece.0.id(), piece.3);
                if s.touched.contains_key(&key) {
                    add(READ_LATELY, b - a);
                    continue;
                }
                if s.runs.range(..b).next_back().is_some_and(|(_, r)| r.end > a) {
                    // Another copy's piece over the same pages.
                    continue;
                }
                let mut old = 0;
                // SAFETY: title pages of a copy's piece; only their
                // protection changes, and the handler gives it back.
                if unsafe { VirtualProtect(a as *const _, b - a, PAGE_NOACCESS, &mut old) } == 0 {
                    add(FAILED, b - a);
                    continue;
                }
                if old != PAGE_READWRITE {
                    // SAFETY: as above, back as it was.
                    unsafe { VirtualProtect(a as *const _, b - a, old, &mut old) };
                    add(FAILED, b - a);
                    continue;
                }
                s.runs.insert(a, Run { end: b, old: PAGE_READWRITE, key });
                add(PROTECTED, b - a);
                add(PROTECTED_N, 1);
            }
        });
    }
}

/// Before the kernel writes title memory `[at, at + len)` for us or for
/// the title: its protected pages are opened, as touched.
pub(crate) fn opening(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let frame = crate::queue::presented();
    with(|s| {
        open_in(s, at, at.saturating_add(len), |s, key, n| {
            s.touched.insert(key, frame);
            add(KERNEL, n);
            add(KERNEL_N, 1);
        });
    });
}

/// Title memory `[at, at + len)` shows something new: its runs are
/// forgotten (the new mapping has protections of its own).
pub(crate) fn forget(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    with(|s| {
        let starts: Vec<usize> = s.runs.range(..end).rev().take_while(|(_, r)| r.end > at).map(|(&a, _)| a).collect();
        for a in starts {
            s.runs.remove(&a);
        }
    });
}

/// Our code's image: a touch from inside it is ours.
fn ours(rip: u64) -> bool {
    static IMAGE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (base, end) = *IMAGE.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        // SAFETY: the process's own module, mapped for its lifetime; the
        // headers are read where the PE format puts them.
        unsafe {
            let base = GetModuleHandleW(std::ptr::null()) as u64;
            if base == 0 {
                return (0, 0);
            }
            let nt = base + u64::from((base as *const u8).add(0x3c).cast::<u32>().read_unaligned());
            let size = (nt as *const u8).add(0x50).cast::<u32>().read_unaligned();
            (base, base + u64::from(size))
        }
    });
    rip >= base && rip < end
}

/// For the vectored handlers: whether an access violation at `at`, by the
/// instruction at `rip`, was a touch of a protected page; it is opened
/// then, and the access can be made again.
pub(crate) fn fault(at: usize, rip: u64) -> bool {
    if !ACTIVE.load(Ordering::Acquire) {
        return false;
    }
    let frame = crate::queue::presented();
    let by_us = ours(rip);
    let mut hit = false;
    with(|s| {
        open_in(s, at, at + 1, |s, key, n| {
            hit = true;
            s.touched.insert(key, frame);
            if by_us {
                add(OURS, n);
                add(OURS_N, 1);
            } else {
                add(TITLE, n);
                add(TITLE_N, 1);
            }
        });
    });
    hit
}

/// The access violation `info` describes, if it is one: the address and
/// the faulting instruction.
///
/// # Safety
///
/// `info` must be what the system passed a vectored handler.
pub(crate) unsafe fn violation(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> Option<(usize, u64)> {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    // SAFETY: as the caller vouches.
    let (r, c) = unsafe { (&*(*info).ExceptionRecord, &*(*info).ContextRecord) };
    (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS && r.NumberParameters >= 2)
        .then(|| (r.ExceptionInformation[1], c.Rip))
}

/// Set once the handler is in ([`install`]).
static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Our handler, first in line. The guarded reads' handler and the write
/// guards' handler ask [`fault`] first as well, whichever was installed
/// last.
fn install() {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        match unsafe { violation(info) } {
            Some((at, rip)) if fault(at, rip) => EXCEPTION_CONTINUE_EXECUTION,
            _ => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    // SAFETY: the handler only opens pages this module protected.
    unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    }
    ACTIVE.store(true, Ordering::Release);
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let mb = |b: f64| b / (1 << 20) as f64;
    let (held, held_bytes) = with(|st| (st.runs.len(), st.runs.iter().map(|(a, r)| r.end - a).sum::<usize>()));
    tracing::info!(
        "lazy copy-back probe per frame (VICTORY_LAZY_BACK=1, VICTORY_READBACK_CHECK=1): {:.1} MB protected after merges \
         ({:.0} pieces); merged again untouched (fetches to spare) {:.1} MB ({:.0}); touched by our code {:.1} MB ({:.0}), \
         by other code (the title's) {:.1} MB ({:.0}), opened for the kernel {:.1} MB ({:.0}); not protected: read lately \
         {:.1} MB, views of physical pages {:.1} MB, no whole page {:.2} MB, failed {:.2} MB; {held} runs {:.0} MB protected now",
        mb(s[PROTECTED]),
        s[PROTECTED_N],
        mb(s[AGAIN]),
        s[AGAIN_N],
        mb(s[OURS]),
        s[OURS_N],
        mb(s[TITLE]),
        s[TITLE_N],
        mb(s[KERNEL]),
        s[KERNEL_N],
        mb(s[READ_LATELY]),
        mb(s[UNWATCHED]),
        mb(s[NO_PAGE]),
        mb(s[FAILED]),
        mb(held_bytes as f64),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A touch of a protected page faults into the handler, which opens
    /// it and counts it; the read then gets the page's bytes.
    #[test]
    fn a_touch_opens_the_page_and_is_counted() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, VirtualAlloc, VirtualFree};
        install();
        // SAFETY: a fresh allocation of ours, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(42) };
        let mut old = 0;
        // SAFETY: as above.
        assert_ne!(unsafe { VirtualProtect((p + PAGE) as *const _, 2 * PAGE, PAGE_NOACCESS, &mut old) }, 0);
        with(|s| s.runs.insert(p + PAGE, Run { end: p + 3 * PAGE, old, key: (u64::MAX, 7) }));
        let before = STATS[OURS].load(Ordering::Relaxed);
        // SAFETY: the page faults once; the handler opens it.
        let read = unsafe { std::ptr::read_volatile((p + PAGE + 8) as *const u8) };
        assert_eq!(read, 42);
        assert!(STATS[OURS].load(Ordering::Relaxed) >= before + (2 * PAGE) as u64);
        assert!(with(|s| s.touched.contains_key(&(u64::MAX, 7)) && !s.runs.contains_key(&(p + PAGE))));
        // A fault elsewhere is not ours.
        assert!(!fault(p, 0));
        // SAFETY: ours, unused now.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Only whole pages inside a piece are protected.
    #[test]
    fn inner_pages_are_whole() {
        assert_eq!(inner(0x1000, 0x3000), Some((0x1000, 0x3000)));
        assert_eq!(inner(0x1010, 0x3010), Some((0x2000, 0x3000)));
        assert_eq!(inner(0x1010, 0x2010), None);
        assert_eq!(inner(0x1000, 0x1fff), None);
    }
}
