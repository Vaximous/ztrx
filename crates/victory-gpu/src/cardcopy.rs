//! Constants copied on the card where their fill would wait for a flight
//! (`VICTORY_CARD_FILLS=1`).
//!
//! A fill of upload memory reads title memory at submission for a draw's
//! constants. Where a flight in the air declared that memory written, the
//! submission waits for it to land. When the memory has a copy in the
//! card's memory, the card copies the constants from it into a stand-in of
//! the list's instead, in the list's order, after the flight on the one
//! host queue.
//!
//! Which fills: those whose source once made a submission wait
//! ([`fill_blocked`]): the same ranges are filled every frame.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::HostList;

/// How many fills [`fills_check_on`] compares at most.
const CHECKS_MOST: u64 = 2000;

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

/// Whether constants filled into upload memory from title memory a
/// flight's shaders may write are copied on the card instead
/// (`VICTORY_CARD_FILLS=1`; off by default).
///
/// A fill of upload memory is a read of title memory at submission, for
/// the card: where a flight in the air declared that memory written (a
/// table view of 174 MB declared whole for an append buffer, run fxprod2),
/// the submission waits for the flight to land, about twice a frame, 30-90
/// ms each. When the memory has a copy in the card's memory, the card can
/// take the constants from there itself, in the list's order, after the
/// flight on the one host queue: a copy into a stand-in of the list's own
/// that the view shows instead. No CPU step is left to wait.
///
/// Which fills: those whose source once made a submission wait
/// ([`fill_blocked`]).
pub(crate) fn fills_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_FILLS").is_ok_and(|v| v == "1"))
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
    if !from.carded() || from.left(src) < len {
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

/// After flights landed: the fills compared ([`fills_check_on`]) whose
/// lists have run.
pub(crate) fn landed() {
    if fills_on() && fills_check_on() {
        fills_landed();
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if fills_on() {
        let made = FILL_STATS[0].swap(0, Ordering::Relaxed) as f64 / frames;
        let bytes = FILL_STATS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "constants copied on the card instead of filled per frame (VICTORY_CARD_FILLS=1): {made:.1}, {:.1} KB; sources learned {}; compared {} (alike {}, different {})",
            bytes / 1024.0,
            FILL_STATS[2].load(Ordering::Relaxed),
            FILL_STATS[3].load(Ordering::Relaxed),
            FILL_STATS[4].load(Ordering::Relaxed),
            FILL_STATS[5].load(Ordering::Relaxed),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{learn, overlaps};

    /// Learned sources join where they touch; a fill is copied on the card
    /// if it reads any of them.
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
