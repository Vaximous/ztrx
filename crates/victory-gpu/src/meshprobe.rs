//! `VICTORY_MESH_READBACK=1` (diagnostics): at the frames listed for
//! capture (`VICTORY_CAPTURE_FRAMES`), every indirect mesh dispatch's
//! records are read back as the card ran them (after our clamp) and as the
//! title's buffer holds them by then, to tell a draw the card was given no
//! groups from one whose geometry was stale: walls missing in a frame
//! whose draws were all recorded.
//!
//! Per pipeline: calls, records, records of no groups (and of them, those
//! whose title record asks for groups by then), and the groups run. For a
//! record of no groups: the compute shader that last declared a write
//! over the title's record, in the same host list if one did. Lines go to
//! `frame-<n>-mesh.txt` in the capture folder; the totals to the log.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::HostList;

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MESH_READBACK").is_some_and(|v| v == "1"))
}

/// Whether the frame being recorded is probed.
pub(crate) fn now() -> bool {
    on() && crate::capture::listed(crate::queue::presented() + 1)
}

thread_local! {
    /// The compute shader being bound (hash, its number in the frame), 0
    /// while anything else is.
    static WRITER: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A pipeline is being bound: `hash` its compute shader, 0 if not compute.
pub(crate) fn binding(hash: u64) {
    if !on() {
        return;
    }
    let seq = if hash != 0 && now() { SEQ.fetch_add(1, Ordering::Relaxed) + 1 } else { 0 };
    WRITER.set(if seq != 0 { (hash, seq) } else { (0, 0) });
}

struct Write {
    lo: u64,
    hi: u64,
    hash: u64,
    seq: u64,
    list: usize,
}

static WRITES: Mutex<Vec<Write>> = Mutex::new(Vec::new());

/// Shaders of the bound pipeline are declared to write title memory
/// `[address, address + len)`.
pub(crate) fn wrote(list: &HostList, address: u64, len: u64) {
    if !on() {
        return;
    }
    let (hash, seq) = WRITER.get();
    if hash == 0 {
        return;
    }
    WRITES.lock().unwrap_or_else(|e| e.into_inner()).push(Write {
        lo: address,
        hi: address + len,
        hash,
        seq,
        list: std::ptr::from_ref(list) as usize,
    });
}

/// The last compute dispatch recorded so far declared to write over
/// `[lo, hi)`: one of `list` if any, else of any list.
fn last_writer(list: usize, lo: u64, hi: u64) -> String {
    let writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let over = |w: &&Write| w.lo < hi && w.hi > lo;
    if let Some(w) = writes.iter().rev().filter(over).find(|w| w.list == list) {
        return format!("compute {:016x} (dispatch #{} of the frame, same list)", w.hash, w.seq);
    }
    match writes.iter().rev().find(over) {
        Some(w) => format!("compute {:016x} (dispatch #{} of the frame, list {:#x})", w.hash, w.seq, w.list),
        None => "no compute dispatch declared a write over it this frame".into(),
    }
}

/// One indirect mesh call, as recorded.
pub(crate) struct Call {
    pub(crate) pipeline: String,
    /// How its records were clamped (`crate::draw`'s `WINDOWED` order).
    pub(crate) how: usize,
    /// For a record read from a window clamped before: calls that may
    /// store recorded in the list since that clamp.
    pub(crate) since: Option<u64>,
    /// Whether shader writes were unwaited for when it was recorded.
    pub(crate) unwaited: bool,
    pub(crate) arguments: u64,
    pub(crate) records: u32,
    pub(crate) stride: u32,
}

const HOW: [&str; 7] = ["window read", "window clamped", "alone: dispatch", "alone: records", "alone: count", "alone: stores", "alone: no step"];

#[derive(Default)]
struct Tally {
    calls: u64,
    records: u64,
    empty: u64,
    /// Empty, the title's record asking for groups.
    emptied: u64,
    groups: u64,
}

#[derive(Default)]
struct Frame {
    frame: u64,
    by_pipeline: BTreeMap<String, Tally>,
    /// Records of no groups by the last writer of the title's record.
    writers: BTreeMap<String, u64>,
    /// Records of no groups by how they were clamped.
    hows: BTreeMap<&'static str, u64>,
    lines: Vec<String>,
}

static FRAME: Mutex<Option<Frame>> = Mutex::new(None);

/// After `list`'s indirect mesh call: copies its records from `scratch`
/// (at `at`) and from the title's `source` into a buffer read once the list
/// has run.
pub(crate) fn probe(list: &HostList, call: Call, scratch: &ID3D12Resource, at: u64, source: (&ID3D12Resource, u64)) {
    let len = u64::from(call.stride) * u64::from(call.records.clamp(1, 64));
    let Some((buffer, data)) = crate::indirect::copy_args_out(list, scratch, at, source, len) else {
        return;
    };
    let frame = crate::queue::presented() + 1;
    let writer = last_writer(std::ptr::from_ref(list) as usize, call.arguments, call.arguments + len);
    let data = data as usize;
    list.defer_probe(Box::new(move || {
        let _keep = &buffer;
        let words = |at: u64| -> [u32; 3] {
            // SAFETY: the mapped read-back buffer of `2 * len` bytes the
            // list has written; records are 4-byte aligned.
            std::array::from_fn(|k| unsafe { (data as *const u32).add(at as usize / 4 + k).read_volatile() })
        };
        let mut f = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        let f = f.get_or_insert_with(|| Frame { frame, ..Default::default() });
        if f.frame != frame {
            return;
        }
        let how = HOW.get(call.how).copied().unwrap_or("?");
        let t = f.by_pipeline.entry(call.pipeline.clone()).or_default();
        t.calls += 1;
        for i in 0..u64::from(call.records.clamp(1, 64)) {
            let ran = words(i * u64::from(call.stride));
            let asked = words(len + i * u64::from(call.stride));
            let groups = ran.iter().map(|&g| u64::from(g)).product::<u64>();
            let wanted = asked.iter().map(|&g| u64::from(g)).product::<u64>();
            t.records += 1;
            t.groups += groups;
            if groups == 0 {
                t.empty += 1;
                if wanted != 0 {
                    t.emptied += 1;
                }
                *f.writers.entry(writer.clone()).or_default() += 1;
                *f.hows.entry(how).or_default() += 1;
                f.lines.push(format!(
                    "{} record {i} at {:#x}: ran {ran:?}, the title's {asked:?}; {how}{}{}; last written by {writer}",
                    call.pipeline,
                    call.arguments + i * u64::from(call.stride),
                    call.since.map(|n| format!(" ({n} storing calls since its clamp)")).unwrap_or_default(),
                    if call.unwaited { ", shader writes unwaited" } else { "" },
                ));
            } else if ran != asked {
                f.lines.push(format!(
                    "{} record {i} at {:#x}: ran {ran:?}, the title's now {asked:?}; {how}",
                    call.pipeline,
                    call.arguments + i * u64::from(call.stride),
                ));
            }
        }
    }));
}

/// At a frame-rate line: the probed frame's totals, once every list of it
/// has run.
pub(crate) fn report() {
    if !on() {
        return;
    }
    let mut slot = FRAME.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_none_or(|f| crate::queue::presented() < f.frame + 4) {
        return;
    }
    let Some(f) = slot.take() else {
        return;
    };
    WRITES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    let all = f.by_pipeline.values().fold(Tally::default(), |a, t| Tally {
        calls: a.calls + t.calls,
        records: a.records + t.records,
        empty: a.empty + t.empty,
        emptied: a.emptied + t.emptied,
        groups: a.groups + t.groups,
    });
    tracing::info!(
        "mesh read-back, frame {}: {} calls, {} records, {} of no groups ({} whose title record asks for groups by then), {} groups run, {} pipelines",
        f.frame,
        all.calls,
        all.records,
        all.empty,
        all.emptied,
        all.groups,
        f.by_pipeline.len()
    );
    tracing::info!("mesh read-back, records of no groups by how clamped: {:?}", f.hows);
    let mut writers: Vec<_> = f.writers.iter().collect();
    writers.sort_by_key(|w| std::cmp::Reverse(*w.1));
    for (w, n) in writers.iter().take(12) {
        tracing::info!("mesh read-back, records of no groups last written by {w}: {n}");
    }
    let mut pipes: Vec<_> = f.by_pipeline.iter().filter(|t| t.1.empty > 0).collect();
    pipes.sort_by_key(|t| std::cmp::Reverse(t.1.empty));
    for (p, t) in pipes.iter().take(20) {
        tracing::info!(
            "mesh read-back, {p}: {} calls, {} records, {} of no groups ({} asking), {} groups run",
            t.calls,
            t.records,
            t.empty,
            t.emptied,
            t.groups
        );
    }
    if let Some(folder) = crate::capture::folder() {
        let mut text: Vec<String> = f
            .by_pipeline
            .iter()
            .map(|(p, t)| format!("{p}: {} calls, {} records, {} empty ({} asking), {} groups", t.calls, t.records, t.empty, t.emptied, t.groups))
            .collect();
        text.push(String::new());
        text.extend(f.lines);
        let _ = std::fs::write(folder.join(format!("frame-{}-mesh.txt", f.frame)), text.join("\n"));
    }
}
