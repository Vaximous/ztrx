//! Title buffer copies done by the card, in their list's order
//! (`VICTORY_CARD_COPIES=1`).
//!
//! A title buffer copy is done on the CPU when its list is submitted. When
//! its source is memory a flight in the air may write (a shader's results
//! the title copies on), the submission waits for that flight to run and
//! land first (in the world twice a frame, 30-90 ms each: the card then
//! idles while the next submission is prepared). Done by the card between
//! the host copies of both buffers, the copy comes after the flight on the
//! one host queue, as the title's own copy comes after its shaders; the
//! destination is declared written like a shader's view, so it reaches
//! title memory with the list's copy-back, and a read of it waits for that
//! as for any shader write.
//!
//! Which copies: those whose source once made a submission wait
//! ([`blocked`]): the same ranges are copied every frame.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::HostList;

/// Whether copies whose source made a submission wait are done by the card.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_COPIES").is_ok_and(|v| v == "1"))
}

/// `VICTORY_CARD_COPIES_CHECK=1`: the first copies done by the card are
/// compared with what the CPU's copy would have written (the source as the
/// card held it right after the copy, against the destination in title
/// memory once the list has landed).
fn check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_COPIES_CHECK").is_ok_and(|v| v == "1"))
}

/// The most copies compared in a run.
const CHECKS_MOST: u64 = 2000;

/// Source ranges (start, end) of title buffer copies that made a submission
/// wait for flights, joined.
static LEARNED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: copies done by the card, their bytes,
/// sources learned; and of all time: compared, alike, different.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A title buffer copy from `[src, src + len)` made a submission wait for
/// flights: later copies from there are done by the card.
pub(crate) fn blocked(src: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let mut learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    if learned.iter().any(|&(a, b)| a <= src && src + len <= b) {
        return;
    }
    STATS[2].fetch_add(1, Ordering::Relaxed);
    learn(&mut learned, src, len);
}

/// `ranges` (in order, joined) with `[src, src + len)`.
fn learn(ranges: &mut Vec<(usize, usize)>, src: usize, len: usize) {
    ranges.push((src, src + len));
    ranges.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(a, b) in ranges.iter() {
        match joined.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => joined.push((a, b)),
        }
    }
    *ranges = joined;
}

/// Whether `[src, src + len)` overlaps one of `ranges` (in order, joined).
fn overlaps(ranges: &[(usize, usize)], src: usize, len: usize) -> bool {
    let at = ranges.partition_point(|r| r.1 <= src);
    ranges.get(at).is_some_and(|&(a, _)| a < src + len)
}

fn learned(src: usize, len: usize) -> bool {
    overlaps(&LEARNED.lock().unwrap_or_else(|e| e.into_inner()), src, len)
}

/// A copy compared ([`check_on`]): the destination, and the source's bytes
/// as the card read them back.
struct Compared {
    dst: usize,
    src: Box<[u8]>,
    done: Box<std::sync::atomic::AtomicU32>,
    age: u32,
}

static COMPARED: Mutex<Vec<Compared>> = Mutex::new(Vec::new());

fn uav_barrier() -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::*;
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// The title's copy of `len` bytes from `src` to `dst` (title memory),
/// recorded into `list`: done by the card if its source is one that made a
/// submission wait and both ends have host copies. Whether it was; if not
/// the caller defers it to the CPU as before.
pub(crate) fn try_copy(list: &HostList, dst: usize, src: usize, len: usize) -> bool {
    if !on() || len == 0 || !learned(src, len) {
        return false;
    }
    let (Some(from), Some(to)) =
        (crate::shadow::find(src as u64, len as u64), crate::shadow::find(dst as u64, len as u64))
    else {
        return false;
    };
    if from.left(src as u64) < len as u64 || to.left(dst as u64) < len as u64 {
        return false;
    }
    // Both copies up to date with the title's bytes at submission (where
    // flights write the source, the refresh leaves their bytes alone), then
    // the destination is written by the card.
    from.refresh(list, src as u64, len as u64);
    to.refresh(list, dst as u64, len as u64);
    to.gpu_writes(list, dst as u64, len as u64, crate::shadow::Declarer::View);
    // What the card copies there is not what a constant buffer read at a
    // later bind finds: such constants bound no store (`crate::extent`).
    crate::extent::card_wrote(dst, len);
    let l = &list.list;
    // SAFETY: an open list; live host copies (the list's steps hold them
    // until it is reset), the ranges inside them (checked above).
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.CopyBufferRegion(&to.buffer, to.offset(dst as u64), &from.buffer, from.offset(src as u64), len as u64);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(len as u64, Ordering::Relaxed);
    if check_on() && STATS[3].load(Ordering::Relaxed) < CHECKS_MOST {
        compare_later(list, &from, src, dst, len);
    }
    true
}

/// [`check_on`]: the source's bytes on the card, read back right after
/// the copy, to compare with the destination once the list has landed.
fn compare_later(list: &HostList, from: &crate::shadow::Shadow, src: usize, dst: usize, len: usize) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let Some(back) = host.read_back_buffer(len as u64) else {
        return;
    };
    // SAFETY: as in `try_copy`; the read-back buffer is fresh and `len`
    // bytes long.
    unsafe {
        list.list.CopyBufferRegion(&back, 0, &from.buffer, from.offset(src as u64), len as u64);
    }
    let mut record = Compared {
        dst,
        src: vec![0u8; len].into_boxed_slice(),
        done: Box::new(std::sync::atomic::AtomicU32::new(0)),
        age: 0,
    };
    list.defer_read_back(back, record.src.as_mut_ptr(), [len, 1, len]);
    list.defer_write(record.done.as_ptr().cast(), crate::queue::Written::Value(1), 4);
    STATS[3].fetch_add(1, Ordering::Relaxed);
    COMPARED.lock().unwrap_or_else(|e| e.into_inner()).push(record);
}

/// After flights landed (their copy-backs and bottom-of-pipe writes done):
/// the copies compared whose lists have run, before a later copy to the
/// same place (the same copies are made every frame) changes it.
pub(crate) fn landed() {
    if fills_on() && fills_check_on() {
        fills_landed();
    }
    if !on() || !check_on() {
        return;
    }
    let mut compared = COMPARED.lock().unwrap_or_else(|e| e.into_inner());
    if compared.is_empty() {
        return;
    }
    compared.retain_mut(|c| {
        if c.done.load(Ordering::Acquire) == 0 {
            c.age += 1;
            if c.age > 300 {
                // A list may still write them: never freed.
                std::mem::forget(std::mem::take(&mut c.src));
                std::mem::forget(std::mem::replace(&mut c.done, Box::new(std::sync::atomic::AtomicU32::new(0))));
                return false;
            }
            return true;
        }
        let mut now = vec![0u8; c.src.len()];
        let same = crate::host::read_into(c.dst as *const u8, &mut now) && *now == *c.src;
        STATS[if same { 4 } else { 5 }].fetch_add(1, Ordering::Relaxed);
        if !same {
            let first = now.iter().zip(c.src.iter()).position(|(a, b)| a != b);
            tracing::warn!(
                dst = format_args!("{:#x}", c.dst),
                len = c.src.len(),
                ?first,
                "a buffer copy done by the card left other bytes than its source"
            );
        }
        false
    });
}

/// Whether constants filled into upload memory from title memory a
/// flight's shaders may write are copied on the card instead
/// (`VICTORY_CARD_FILLS`; on by default, `=0` off). On: the fills compared
/// against the CPU's were all alike (run zl3cck, 2,000 of 2,000), and
/// off, the fills' waits were most of the submitting thread's waiting for
/// "title memory read": 1.0 a frame, 31 ms (run zall7b).
///
/// A fill of upload memory is a read of title memory at submission, for
/// the card: where a flight in the air declared that memory written (a
/// table view of 174 MB declared whole for an append buffer, run fxprod2),
/// the submission waits for the flight to land, about twice a frame, 30-90
/// ms each. When the memory has a host copy, the card can take the
/// constants from there itself, in the list's order, after the flight on
/// the one host queue: a copy into a stand-in of the list's own that the
/// view shows instead. No CPU step is left to wait. Any host copy will do,
/// in the card's memory or the PC's: with [`on`] the fills that wait read
/// the small destinations of title buffer copies done on the card (run
/// zl3cc: 3.8 a frame, 97 ms), which are copies in the PC's memory.
///
/// Which fills: those whose source once made a submission wait
/// ([`fill_blocked`]).
pub(crate) fn fills_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_FILLS").is_none_or(|v| v != "0"))
}

/// `VICTORY_CARD_FILLS_CHECK=1`: the first constants copied on the card are
/// also filled by the CPU, as before, and the two compared once the list
/// has landed.
fn fills_check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_FILLS_CHECK").is_ok_and(|v| v == "1"))
}

/// Source ranges of fills that made a submission wait, joined.
static LEARNED_FILLS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills done by the card, their bytes;
/// of all time: sources learned, compared, alike, different.
static FILL_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A fill of upload memory from `[src, src + len)` made a submission wait.
pub(crate) fn fill_blocked(src: usize, len: usize) {
    if !fills_on() || len == 0 {
        return;
    }
    let mut learned = LEARNED_FILLS.lock().unwrap_or_else(|e| e.into_inner());
    if learned.iter().any(|&(a, b)| a <= src && src + len <= b) {
        return;
    }
    FILL_STATS[2].fetch_add(1, Ordering::Relaxed);
    learn(&mut learned, src, len);
}

/// A fill compared ([`fills_check_on`]): what the CPU filled and what the
/// card copied.
struct FillCompared {
    at: usize,
    cpu: Box<[u8]>,
    card: Box<[u8]>,
    done: Box<std::sync::atomic::AtomicU32>,
    age: u32,
}

static FILLS_COMPARED: Mutex<Vec<FillCompared>> = Mutex::new(Vec::new());

/// The `len` bytes of title memory at `src` a list's draw reads as
/// constants, copied on the card from the memory's card copy into a
/// stand-in of the list's, at this place in the list: the stand-in's
/// address, for the view. None: the caller fills upload memory as before.
pub(crate) fn try_fill(list: &HostList, src: u64, len: u64) -> Option<u64> {
    use windows::Win32::Graphics::Direct3D12::*;
    if !fills_on() || len == 0 || !overlaps(&LEARNED_FILLS.lock().unwrap_or_else(|e| e.into_inner()), src as usize, len as usize)
    {
        return None;
    }
    let from = crate::shadow::find(src, len)?;
    if from.left(src) < len {
        return None;
    }
    let (stand_in, at, first) = list.card_scratch(len)?;
    // The title's bytes sent up before the submission (where flights write
    // the memory, the refresh leaves their bytes alone).
    from.refresh(list, src, len);
    let l = &list.list;
    let transition = |before, after| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer the barrier never releases.
                pResource: unsafe { std::mem::transmute_copy(&stand_in) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: an open list; live buffers (the list keeps the stand-in, its
    // steps the host copy, until it is reset), the ranges inside them. The
    // stand-in's first copy in the recording takes it from the common state
    // (a buffer's at the start of every execution); a later one from the
    // constant reads the list left it in.
    unsafe {
        if first {
            l.ResourceBarrier(&[uav_barrier()]);
        } else {
            l.ResourceBarrier(&[
                uav_barrier(),
                transition(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST),
            ]);
        }
        l.CopyBufferRegion(&stand_in, at, &from.buffer, from.offset(src), len);
        l.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)]);
    }
    FILL_STATS[0].fetch_add(1, Ordering::Relaxed);
    FILL_STATS[1].fetch_add(len, Ordering::Relaxed);
    if fills_check_on() && FILL_STATS[3].load(Ordering::Relaxed) < CHECKS_MOST {
        compare_fill(list, &stand_in, at, src, len);
    }
    // SAFETY: a live buffer.
    Some(unsafe { stand_in.GetGPUVirtualAddress() } + at)
}

/// [`fills_check_on`]: the stand-in read back after the draws that read it,
/// and the CPU's fill of the same bytes at submission, compared once the
/// list has landed.
fn compare_fill(list: &HostList, stand_in: &windows::Win32::Graphics::Direct3D12::ID3D12Resource, at: u64, src: u64, len: u64) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let Some(back) = host.read_back_buffer(len) else {
        return;
    };
    let n = len as usize;
    let mut record = FillCompared {
        at: src as usize,
        cpu: vec![0u8; n].into_boxed_slice(),
        card: vec![0u8; n].into_boxed_slice(),
        done: Box::new(std::sync::atomic::AtomicU32::new(0)),
        age: 0,
    };
    // SAFETY: as in `try_fill`; the read-back buffer is fresh and `len`
    // bytes long. The stand-in is read as constants now; a copy source is
    // a read state too, but a buffer's states do not combine with constant
    // reads in one barrier: through the common state.
    unsafe {
        use windows::Win32::Graphics::Direct3D12::*;
        let t = |before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(stand_in),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        list.list.ResourceBarrier(&[t(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.list.CopyBufferRegion(&back, 0, stand_in, at, len);
        list.list.ResourceBarrier(&[t(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)]);
    }
    list.defer_fill(record.cpu.as_mut_ptr(), src as *const u8, n);
    list.defer_read_back(back, record.card.as_mut_ptr(), [n, 1, n]);
    list.defer_write(record.done.as_ptr().cast(), crate::queue::Written::Value(1), 4);
    FILL_STATS[3].fetch_add(1, Ordering::Relaxed);
    FILLS_COMPARED.lock().unwrap_or_else(|e| e.into_inner()).push(record);
}

/// After flights landed: the fills compared whose lists have run.
fn fills_landed() {
    let mut compared = FILLS_COMPARED.lock().unwrap_or_else(|e| e.into_inner());
    compared.retain_mut(|c| {
        if c.done.load(Ordering::Acquire) == 0 {
            c.age += 1;
            if c.age > 300 {
                // A list may still write them: never freed.
                std::mem::forget(std::mem::take(&mut c.cpu));
                std::mem::forget(std::mem::take(&mut c.card));
                std::mem::forget(std::mem::replace(&mut c.done, Box::new(std::sync::atomic::AtomicU32::new(0))));
                return false;
            }
            return true;
        }
        let same = c.cpu == c.card;
        FILL_STATS[if same { 4 } else { 5 }].fetch_add(1, Ordering::Relaxed);
        if !same {
            let first = c.cpu.iter().zip(c.card.iter()).position(|(a, b)| a != b);
            tracing::warn!(
                src = format_args!("{:#x}", c.at),
                len = c.cpu.len(),
                ?first,
                "constants copied on the card differ from the CPU's fill"
            );
        }
        false
    });
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if fills_on() {
        let made = FILL_STATS[0].swap(0, Ordering::Relaxed) as f64 / frames;
        let bytes = FILL_STATS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "constants copied on the card instead of filled per frame (VICTORY_CARD_FILLS=1): {made:.1}, {:.1} KB;              sources learned {}; compared {} (alike {}, different {})",
            bytes / 1024.0,
            FILL_STATS[2].load(Ordering::Relaxed),
            FILL_STATS[3].load(Ordering::Relaxed),
            FILL_STATS[4].load(Ordering::Relaxed),
            FILL_STATS[5].load(Ordering::Relaxed),
        );
    }
    if !on() {
        return;
    }
    let made = STATS[0].swap(0, Ordering::Relaxed) as f64 / frames;
    let bytes = STATS[1].swap(0, Ordering::Relaxed) as f64 / frames;
    tracing::info!(
        "title buffer copies done by the card per frame (VICTORY_CARD_COPIES=1): {made:.1}, {:.1} KB; sources learned \
         {}; compared {} (alike {}, different {})",
        bytes / 1024.0,
        STATS[2].load(Ordering::Relaxed),
        STATS[3].load(Ordering::Relaxed),
        STATS[4].load(Ordering::Relaxed),
        STATS[5].load(Ordering::Relaxed),
    );
}

#[cfg(test)]
mod tests {
    use super::{learn, overlaps};

    /// Learned sources join where they touch; a copy is done by the card if
    /// it reads any of them.
    #[test]
    fn learned_sources_join_and_match_overlaps() {
        let mut r = Vec::new();
        learn(&mut r, 0x1000, 0x800);
        learn(&mut r, 0x1800, 0x800);
        learn(&mut r, 0x4000, 0x100);
        assert_eq!(r, [(0x1000, 0x2000), (0x4000, 0x4100)]);
        assert!(overlaps(&r, 0x1f00, 0x200));
        assert!(overlaps(&r, 0x3f00, 0x101));
        assert!(!overlaps(&r, 0x2000, 0x2000));
        assert!(!overlaps(&r, 0x4100, 0x10));
    }
}
