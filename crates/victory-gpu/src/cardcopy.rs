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

/// Whether copies whose source made a submission wait are done by the card
/// (`VICTORY_CARD_COPIES`; on by default, `=0` off). Checked in run
/// zall8cc: of 2,000 compared, 1,885 alike and 115 alike a later copy to
/// the same place that landed with them, none different (run zl3cck's 90
/// "different" were those).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_COPIES").is_none_or(|v| v != "0"))
}

/// The range learned when a read of `[src, src + len)` made a submission
/// wait: the range read (`VICTORY_CARD_LEARN_WHOLE=1`: the whole title
/// buffer holding it; off by default). The title reads what flights wrote from
/// another place of the same buffers every frame (ring buffers): learned
/// by the range read, fills went on learning about 7 new sources a frame
/// and waiting for each (run zall8cc: still 1.0 settle a frame, 55-69 ms),
/// and with card copies off the same flights' outputs made a buffer copy
/// wait instead (run zall8a, 1.0 a frame, 60 ms).
///
/// Learned whole, 10-13k fills a frame were done by the card (run zall10:
/// preparing 67 to 103 ms a frame, the copy-back 57 to 82, median frame
/// 137 to 173-178 ms). `VICTORY_CARD_LEARN_BLOCK=<KB>` (takes the place of
/// the whole buffer) learns the range read widened to blocks of that many
/// KB, aligned in title memory and cut to the buffer: a ring buffer's next
/// reads nearby are caught, the rest of a big buffer is not.
///
/// Off by default for speed: learned by the range read, frames ran 132-136
/// ms (runs zall11lw, zall12lw0) against 185-190 ms learned whole (zall11a,
/// zall11fb), one settle of about 64 ms a frame left. Learned narrowly,
/// frame 2400 mostly shows blue blocks along its bottom edge (zall11blk,
/// zall12lw0; zall11lw did not), not seen learned whole (zall9b).
fn learned_range(src: usize, len: usize) -> (usize, usize) {
    static WHOLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let block = *BLOCK.get_or_init(|| {
        std::env::var("VICTORY_CARD_LEARN_BLOCK").ok().and_then(|v| v.parse::<usize>().ok()).map_or(0, |kb| kb << 10)
    });
    let whole = *WHOLE.get_or_init(|| std::env::var_os("VICTORY_CARD_LEARN_WHOLE").is_some_and(|v| v == "1"));
    if (whole || block != 0)
        && let Some(copy) = crate::shadow::find(src as u64, len as u64)
    {
        let start = copy.title_start() as usize;
        let buffer = (start, start + copy.left(start as u64) as usize);
        if buffer.0 <= src && src + len <= buffer.1 {
            return in_blocks(src, len, if block != 0 { block } else { usize::MAX }, buffer);
        }
    }
    (src, len)
}

/// `[src, src + len)` widened to `block`-aligned blocks (`usize::MAX`: the
/// whole buffer), cut to `buffer` (start, end): start and length.
fn in_blocks(src: usize, len: usize, block: usize, buffer: (usize, usize)) -> (usize, usize) {
    if block == usize::MAX {
        return (buffer.0, buffer.1 - buffer.0);
    }
    let from = (src / block * block).max(buffer.0);
    let to = (src + len).next_multiple_of(block).min(buffer.1);
    (from, to - from)
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
/// sources learned; and of all time: compared, alike, different, alike a
/// later card copy to the same place that landed with it, of the different
/// those whose destination on the card was not the source right after the
/// copy.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A title buffer copy from `[src, src + len)` made a submission wait for
/// flights: later copies from there are done by the card.
pub(crate) fn blocked(src: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    if learned.iter().any(|&(a, b)| a <= src && src + len <= b) {
        return;
    }
    STATS[2].fetch_add(1, Ordering::Relaxed);
    drop(learned);
    let (src, len) = learned_range(src, len);
    learn(&mut LEARNED.lock().unwrap_or_else(|e| e.into_inner()), src, len);
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
    /// The destination on the card right after the copy.
    card_dst: Box<[u8]>,
    done: Box<std::sync::atomic::AtomicU32>,
    age: u32,
}

static COMPARED: Mutex<Vec<Compared>> = Mutex::new(Vec::new());

pub(crate) fn uav_barrier() -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
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
    let chosen = || {
        if read_on_card() {
            let under = crate::host::flights_write(src, len) || crate::host::flights_write(dst, len);
            if under {
                READ_ON_CARD_STATS[1].fetch_add(1, Ordering::Relaxed);
            }
            under
        } else {
            learned(src, len)
        }
    };
    // The list's own card writes earlier (`after_write_on`): a CPU copy
    // runs at submission, before them, so it would read the bytes from
    // before, or be overwritten by their copy-back.
    let after_write = || after_write_on() && (list.gpu_written(src, len) || list.gpu_written(dst, len));
    if !on() || len == 0 || !chosen() && !after_write() {
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
        compare_later(list, &from, &to, src, dst, len);
    }
    true
}

/// [`check_on`]: the source's bytes on the card, read back right after
/// the copy, to compare with the destination once the list has landed.
fn compare_later(
    list: &HostList,
    from: &crate::shadow::Shadow,
    to: &crate::shadow::Shadow,
    src: usize,
    dst: usize,
    len: usize,
) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let (Some(back), Some(back_dst)) = (host.read_back_buffer(len as u64), host.read_back_buffer(len as u64)) else {
        return;
    };
    // SAFETY: as in `try_copy`; the read-back buffers are fresh and `len`
    // bytes long.
    unsafe {
        list.list.CopyBufferRegion(&back, 0, &from.buffer, from.offset(src as u64), len as u64);
        list.list.CopyBufferRegion(&back_dst, 0, &to.buffer, to.offset(dst as u64), len as u64);
    }
    let mut record = Compared {
        dst,
        src: vec![0u8; len].into_boxed_slice(),
        card_dst: vec![0u8; len].into_boxed_slice(),
        done: Box::new(std::sync::atomic::AtomicU32::new(0)),
        age: 0,
    };
    list.defer_read_back(back, record.src.as_mut_ptr(), [len, 1, len]);
    list.defer_read_back(back_dst, record.card_dst.as_mut_ptr(), [len, 1, len]);
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
    // Several flights land together: a later copy to the same place may
    // have landed with this one, and title memory then holds its source
    // (run zl3cck's 90 differences, all at one 32-byte destination copied
    // every frame, were looked at only after that). Which source the
    // title's bytes are is decided over all copies landed now.
    let landed: Vec<bool> = compared.iter().map(|c| c.done.load(Ordering::Acquire) != 0).collect();
    let mut verdicts: Vec<Option<Verdict>> = (0..compared.len())
        .map(|i| {
            if !landed[i] {
                return None;
            }
            let c = &compared[i];
            let mut now = vec![0u8; c.src.len()];
            if !crate::host::read_into(c.dst as *const u8, &mut now) {
                return Some(Verdict::Different(None, false));
            }
            let later = compared
                .iter()
                .zip(&landed)
                .skip(i + 1)
                .filter(|(o, l)| **l && o.dst == c.dst && o.src.len() == c.src.len())
                .map(|(o, _)| &*o.src);
            Some(verdict(&now, &c.src, &c.card_dst, later))
        })
        .collect();
    let mut i = 0;
    compared.retain_mut(|c| {
        let v = verdicts[i].take();
        i += 1;
        let Some(v) = v else {
            c.age += 1;
            if c.age > 300 {
                // A list may still write them: never freed.
                std::mem::forget(std::mem::take(&mut c.src));
                std::mem::forget(std::mem::take(&mut c.card_dst));
                std::mem::forget(std::mem::replace(&mut c.done, Box::new(std::sync::atomic::AtomicU32::new(0))));
                return false;
            }
            return true;
        };
        match v {
            Verdict::Alike => STATS[4].fetch_add(1, Ordering::Relaxed),
            Verdict::Superseded => STATS[6].fetch_add(1, Ordering::Relaxed),
            Verdict::Different(first, copy_wrong) => {
                STATS[5].fetch_add(1, Ordering::Relaxed);
                if copy_wrong {
                    STATS[7].fetch_add(1, Ordering::Relaxed);
                }
                tracing::warn!(
                    dst = format_args!("{:#x}", c.dst),
                    len = c.src.len(),
                    ?first,
                    copy_wrong,
                    "a buffer copy done by the card left other bytes than its source (copy_wrong: the card's \
                     destination right after the copy was not the source either)"
                );
                0
            }
        };
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
/// has landed. `=2`: only those `VICTORY_READ_ON_CARD` sent to the card
/// because flights may write them (the CPU fill then makes the submission
/// wait for them, so the compare is against the landed bytes). `=3`: only
/// those sent to the card because the list itself may write them earlier
/// ([`after_write_on`]; the CPU fill reads title memory before that write,
/// so a difference there is the staleness the card fill closes).
fn fills_check_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_CARD_FILLS_CHECK").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("3") => 3,
        _ => 0,
    })
}

pub(crate) fn fills_check_on() -> bool {
    fills_check_mode() != 0
}

/// Source ranges of fills that made a submission wait, joined.
static LEARNED_FILLS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills done by the card, their bytes;
/// of all time: sources learned, compared, alike, different; since the
/// last line: copies recorded for the fills, barrier calls recorded for
/// them.
static FILL_STATS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// Since the last frame-rate line, where the fills' cost may land:
/// [`try_fill`]'s own time (us, recording), its fills from host copies
/// flights' shaders may be writing, and the new refreshes they added to
/// their lists ([`HostList::defer_refresh`]).
const FILL_US: usize = 8;
const FILL_UNDER_FLIGHTS: usize = 9;
const FILL_REFRESHES: usize = 10;
/// Since the last line: fills shown a list's [`FillGroup`] made before, and
/// groups made.
const FILL_GROUP_USES: usize = 11;
const FILL_GROUPS: usize = 12;

/// Since the last frame-rate line: time (us) and calls of the submission's
/// check for refreshes of memory flights may write (`host`'s
/// `stale_under_flights`, per copy over the span of its refreshed ranges:
/// fills spread over a buffer widen the span).
pub(crate) static STALE_CHECK: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a draw's or dispatch's fills on the card ([`try_fill`]) are
/// recorded together when its binding is done ([`batched`]): one wait for
/// earlier shader writes, one transition into a copy destination per
/// stand-in buffer, the copies (fills that follow on in both buffers made
/// one), one transition back per buffer (`VICTORY_FILL_BATCH`; on by
/// default, `=0` off: each fill its own wait, transitions and copy, about
/// 10,600 a frame in run zall10).
///
/// The same work in the list: a binding records no other GPU work between
/// its fills (root arguments, and descriptors written by the CPU), so the
/// copies may move to its end: each reads a host copy no command between
/// writes, and writes a fresh range of a stand-in that only the draw after
/// the binding reads. A barrier for all unordered access orders the same
/// earlier work before every copy as one before each did. A stand-in
/// buffer starts the batch in the common state (its first use in the
/// recording) or a constant buffer, ends it a constant buffer, as before;
/// a fill of the same bytes as an earlier one in the batch is shown the
/// earlier one's stand-in, which holds the same bytes.
pub(crate) fn fill_batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_BATCH").is_none_or(|v| v != "0"))
}

/// `VICTORY_FILL_UAV_SKIP=1` (off by default): a batch of fills records no
/// barrier for earlier writes when no shader that may write buffers was
/// recorded since the list's last such barrier ([`HostList::writes_unwaited`]).
/// Not known equivalent: copies of ours that write host copies on the card
/// (title texture copies, whose memory a buffer's host copy may share) are
/// not counted as writes there.
fn fill_uav_skip_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_FILL_UAV_SKIP").is_ok_and(|v| v == "1"))
}

/// `VICTORY_FILL_AFTER_WRITE=1` (on by default with [`read_on_card`],
/// `=0` turns it off): a draw's constants in
/// title memory that the list's own shaders or card copies may write
/// earlier in it are copied on the card ([`try_fill`]) whether or not
/// their source was learned.
///
/// The staleness it closes: a fill of upload memory reads title memory when
/// the list is submitted, before the card runs any of it, and waits only
/// for flights in the air (earlier submissions). A range the same list
/// writes on the card (a title buffer copy done by the card, whose
/// destination is declared written; a shader's view) and then reads as
/// constants is filled with the bytes from before the write. With whole
/// buffers learned (`VICTORY_CARD_LEARN_WHOLE`) such a destination buffer
/// was learned whole the first time a fill of it waited for a flight, so
/// every later fill of it read the host copy in the list's order. Learned
/// by range or block, the other places of the buffer (a ring) stay CPU
/// fills; written and read within one submission they never wait, so they
/// are never learned (runs zall10lw and zall11blk: the strips at the
/// bottom of the picture, not in zall9b).
///
/// With `VICTORY_READ_ON_CARD` every title buffer copy whose ends flights
/// may write is a card copy, its destination declared written; the title
/// often copies a buffer's constants and draws with them in the same list.
/// Done by the CPU (before), the copy came first in the list's steps; a
/// CPU fill of them now reads title memory before the card's copy (run
/// zall15d: the hair black).
fn after_write_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_FILL_AFTER_WRITE").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => read_on_card(),
    })
}

/// Whether a draw's constants and a title buffer copy are read on the card
/// exactly when what they read is in what flights in the air may write
/// (`crate::host::flights_write`, the test a submission's reads wait by),
/// instead of when their source made a submission wait before (learned):
/// `VICTORY_READ_ON_CARD`, on by default; `=0` learned, as before.
///
/// Learned, the sources chased the title's moving reads: a ring read in a
/// new place each frame made one submission wait each frame (runs zall13g,
/// zall14r: 1.0 settle a frame, 61-69 ms), and learning whole buffers sent
/// 11-13k fills a frame through the card. Asked at the read, a read under a
/// flight's writes goes to the card, behind the flight on the one queue,
/// and everything else stays a fill of upload memory.
///
/// As the wait: the settle lands the flights and copies back what their
/// shaders wrote (the declared ranges), then the CPU reads title memory.
/// The card reads the host copy after the flights' shaders, which holds
/// what they wrote (beyond the declared ranges too, where the copy-back
/// takes nothing), and the title's bytes the submission sends up; where
/// the title wrote under a flight's writes, the submission settles first
/// as for any refresh there (`stale_under_flights`). The same card fills and
/// copies were compared with the CPU's (`VICTORY_CARD_FILLS_CHECK`,
/// `VICTORY_CARD_COPIES_CHECK`: all alike). A flight sent after the read
/// was recorded is not known then: that read stays on the CPU and waits as
/// before.
pub(crate) fn read_on_card() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_READ_ON_CARD").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line ([`read_on_card`]): fills and title buffer
/// copies sent to the card because flights may write what they read.
static READ_ON_CARD_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A fill waiting in a [`batched`] binding to be recorded.
struct Fill {
    stand_in: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    at: u64,
    from: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    from_at: u64,
    len: u64,
    /// The stand-in's first use in the recording ([`HostList::card_scratch`]).
    first: bool,
    /// Title memory the fill reads.
    src: u64,
    /// Compared with the CPU's fill ([`fills_check_on`]).
    check: bool,
}

thread_local! {
    /// The list whose binding is [`batched`] on this thread, and its fills so far.
    static BATCH: std::cell::RefCell<Option<(usize, Vec<Fill>)>> = const { std::cell::RefCell::new(None) };
    /// The list's steps when that binding began ([`HostList::steps`]).
    static BIND_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The size of a [`FillGroup`]'s block of title memory
/// (`VICTORY_FILL_GROUP_KB`, at most 64, by default; 0 turns it off: each
/// fill copies its own bytes, about 11,700 a frame and 4.3 us of CPU each
/// in run zall11a, nearly all one constant buffer per binding).
///
/// The same bytes as a fill of its own: a stand-in copied at a binding
/// holds the host copy's bytes at that place in the list's order. A later
/// binding of the same list reads the same bytes there as a copy of its own
/// would, unless the card wrote the host copy in between: the list's shaders
/// and its title buffer copies done on the card, which declare every range
/// they may write ([`crate::shadow::Shadow::gpu_writes`], what the copy-back
/// takes from them too). A group is used only while no call whose shaders
/// may write buffers was recorded since it was made ([`group_good`]: the
/// binding's own draw is recorded after the copy, so it counts) and no range
/// declared since its binding began meets its block (card copies); copies
/// in place are not grouped. Flights run before the list; title writes reach the host
/// copy at submission, before the list, for every fill alike; each fill's
/// own range is still refreshed.
///
/// On (64) by default since runs zall15g, zall15d (300 s, defaults
/// otherwise): 126 ms a frame with it against 128 without, and without it
/// the hair at frame 2400 drew as black blocks; with it the picture was
/// no worse than before.
///
/// Once off by default: run zall12w (64 KiB) broke. Each block took a 64 KiB
/// chunk of its list's, and lists kept their chunks across resets, so every
/// list the title records into grew to its largest recording's (51 GB
/// committed, a buffer made for most blocks: 827 ms a frame "recording"
/// fills, draws dropped); the chunks now go back to a pool at the reset.
/// And a range refreshed for a group before a title copy done at
/// submission was not refreshed again after it (each fill's own refresh is
/// made again after one, `HostList::title_writes`): the host copy kept the
/// bytes from before the copy where the per-fill path took those after.
fn group_bytes() -> u64 {
    static KB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *KB.get_or_init(|| {
        std::env::var("VICTORY_FILL_GROUP_KB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(64).min(64) << 10
    })
}

/// A block of title memory a list's bindings read constants from, copied
/// once into a stand-in so that addresses `phase` past a multiple of 256
/// are 256-byte aligned there ([`group_bytes`]).
pub(crate) struct FillGroup {
    /// Copied: its block is in a stand-in. Not yet: a fill was seen there
    /// ([`group_second`]), filled on its own.
    copied: bool,
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// The stand-in's address of `lo`.
    gpu: u64,
    /// The title memory copied.
    lo: u64,
    hi: u64,
    /// The list's steps up to which no declared write meets `[lo, hi)`.
    checked: usize,
    /// Ranges of it refreshed for the list (joined), since the list's title
    /// copy `epoch` ([`HostList::title_writes`]).
    refreshed: Vec<(usize, usize)>,
    epoch: u64,
    /// The list's calls whose shaders may write buffers when it was made
    /// ([`HostList::write_count`]).
    writes: u64,
}

/// Whether a group still holds what a fill of its own would copy now: no
/// call whose shaders may write buffers recorded since it was made
/// (`writes`, `writes_now`), and no write declared since that meets it
/// (`declared`: card copies, which count as no such call).
///
/// Declared ranges alone (d4c8446b) were not enough (run zall12lw: the
/// window's fence and glass, the people in the yard gone, strips at the
/// bottom): what a shader declares is its view, narrowed by its store
/// extent, and what the list's later draws read of it on the card, not in
/// title memory, a copy of its own would see whatever the extent left out.
fn group_good(writes: u64, writes_now: u64, declared: bool) -> bool {
    writes == writes_now && !declared
}

/// Whether a block is copied only for the second fill from it with nothing
/// written in between (`VICTORY_FILL_GROUP_SECOND`; on by default, `=0`
/// the first fill copies it). With sources learned whole most blocks served
/// one fill before a writing call dropped them (run zall13wg: 1,236 blocks
/// of 64 KiB copied a frame for 2,900 fills, 0.47 ms each): one fill's own
/// copy is 256 bytes, a block's a chunk of 64 KiB.
fn group_second() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_GROUP_SECOND").is_none_or(|v| v != "0"))
}

/// What a fill does with the list's entry for its block ([`group_fill`]).
#[derive(Debug, PartialEq)]
enum GroupStep {
    /// Show the block's stand-in.
    Use,
    /// Copy the block now.
    Copy,
    /// Note the block seen; fill on its own.
    Note,
}

/// [`GroupStep`] for an entry (`None`: none for this block and copy;
/// `Some(copied)`) that is still good or not ([`group_good`]), with
/// [`group_second`] `second`.
fn group_step(entry: Option<bool>, good: bool, second: bool) -> GroupStep {
    match entry {
        Some(true) if good => GroupStep::Use,
        Some(false) if good => GroupStep::Copy,
        _ if second => GroupStep::Note,
        _ => GroupStep::Copy,
    }
}

/// Whether `[src, src + len)` is to be refreshed for a group (noted so if
/// it is): not refreshed for it since title copy `now` (`refreshed` and
/// `epoch`, the group's; a later title copy starts them over).
fn refresh_due(refreshed: &mut Vec<(usize, usize)>, epoch: &mut u64, now: u64, src: usize, len: usize) -> bool {
    if *epoch != now {
        refreshed.clear();
        *epoch = now;
    }
    if inside(refreshed, src, len) {
        return false;
    }
    learn(refreshed, src, len);
    true
}

/// A group's first title address: the first at or after `from` that is
/// `phase` past a multiple of 256.
fn group_start(from: u64, phase: u64) -> u64 {
    from + (phase + 256 - from % 256) % 256
}

/// The constants at `[src, src + len)` of `from`, in a [`FillGroup`] of the
/// list's: its stand-in's address of `src`, the group made (its copy in the
/// binding's batch) if there is none still good. None: filled on its own.
fn group_fill(list: &HostList, from: &std::sync::Arc<crate::shadow::Shadow>, src: u64, len: u64) -> Option<u64> {
    let block = group_bytes();
    if block == 0 || from.in_place() || fills_check_on() {
        return None;
    }
    let me = std::ptr::from_ref(list) as usize;
    if !BATCH.with_borrow(|b| b.as_ref().is_some_and(|b| b.0 == me)) {
        return None;
    }
    let (phase, bs) = (src % 256, src / block * block);
    let start = from.title_start();
    let be = (bs + block).min(start + from.left(start));
    if src + len > be {
        return None;
    }
    let mut groups = list.fill_groups();
    let key = (bs, phase as u16);
    let lo = group_start(bs.max(start), phase);
    let (entry, good) = match groups.get(&key) {
        Some(g) if std::sync::Arc::ptr_eq(&g.shadow, from) => {
            let (hit, _) = list.declared_writes_overlap(g.checked, g.lo as usize, g.hi as usize);
            (Some(g.copied), group_good(g.writes, list.write_count(), hit))
        }
        _ => (None, false),
    };
    let step = group_step(entry, good, group_second());
    if step == GroupStep::Note {
        groups.insert(
            key,
            FillGroup {
                copied: false,
                shadow: from.clone(),
                gpu: 0,
                lo,
                hi: be,
                checked: BIND_STEPS.get(),
                refreshed: Vec::new(),
                epoch: 0,
                writes: list.write_count(),
            },
        );
        return None;
    }
    if step == GroupStep::Use
        && let Some(g) = groups.get_mut(&key)
    {
        g.checked = list.steps();
        if refresh_due(&mut g.refreshed, &mut g.epoch, list.title_writes(), src as usize, len as usize)
            && from.refresh(list, src, len)
        {
            FILL_STATS[FILL_REFRESHES].fetch_add(1, Ordering::Relaxed);
        }
        FILL_STATS[FILL_GROUP_USES].fetch_add(1, Ordering::Relaxed);
        return Some(g.gpu + (src - g.lo));
    }
    let (stand_in, at, first) = list.card_scratch(be - lo)?;
    if from.refresh(list, src, len) {
        FILL_STATS[FILL_REFRESHES].fetch_add(1, Ordering::Relaxed);
    }
    if from.writes_pending() {
        FILL_STATS[FILL_UNDER_FLIGHTS].fetch_add(1, Ordering::Relaxed);
    }
    // SAFETY: a live buffer.
    let gpu = unsafe { stand_in.GetGPUVirtualAddress() } + at;
    let fill = Fill { stand_in, at, from: from.buffer.clone(), from_at: from.offset(lo), len: be - lo, first, src: lo, check: false };
    BATCH.with_borrow_mut(|b| b.as_mut().map(|b| b.1.push(fill)))?;
    groups.insert(
        key,
        FillGroup {
            copied: true,
            shadow: from.clone(),
            gpu,
            lo,
            hi: be,
            checked: BIND_STEPS.get(),
            refreshed: vec![(src as usize, (src + len) as usize)],
            epoch: list.title_writes(),
            writes: list.write_count(),
        },
    );
    FILL_STATS[FILL_GROUPS].fetch_add(1, Ordering::Relaxed);
    Some(gpu + (src - lo))
}

/// Whether `[src, src + len)` lies inside one of `ranges` (in order, joined).
fn inside(ranges: &[(usize, usize)], src: usize, len: usize) -> bool {
    let at = ranges.partition_point(|r| r.1 <= src);
    ranges.get(at).is_some_and(|&(a, b)| a <= src && src + len <= b)
}

/// Runs `bind`, the binding of a draw or dispatch into `list`, with its
/// fills on the card recorded together at its end ([`fill_batch_on`]).
pub(crate) fn batched<T>(list: &HostList, bind: impl FnOnce() -> T) -> T {
    struct Flush<'a>(&'a HostList);
    impl Drop for Flush<'_> {
        fn drop(&mut self) {
            if let Some((_, fills)) = BATCH.with_borrow_mut(Option::take) {
                record_fills(self.0, &fills);
            }
        }
    }
    if !fills_on() || !fill_batch_on() || BATCH.with_borrow(Option::is_some) {
        return bind();
    }
    BATCH.with_borrow_mut(|b| *b = Some((std::ptr::from_ref(list) as usize, Vec::new())));
    BIND_STEPS.set(list.steps());
    let _flush = Flush(list);
    bind()
}

/// How a batch of fills is recorded: the stand-ins (by index of their
/// first fill) to take from constant reads to a copy destination first,
/// the copies (index of the fill, bytes from there), and the stand-ins to
/// take back to constant reads after. `fills`: stand-in, offset in it,
/// source buffer, offset in it, bytes, first use of the stand-in.
fn plan_fills(fills: &[(usize, u64, usize, u64, u64, bool)]) -> (Vec<usize>, Vec<(usize, u64)>, Vec<usize>) {
    let mut buffers: Vec<usize> = Vec::new();
    let mut before = Vec::new();
    for (i, f) in fills.iter().enumerate() {
        if fills[..i].iter().any(|g| g.0 == f.0) {
            continue;
        }
        buffers.push(i);
        // A stand-in first used in the batch starts it in the common state.
        if !fills.iter().any(|g| g.0 == f.0 && g.5) {
            before.push(i);
        }
    }
    let mut copies: Vec<(usize, u64)> = Vec::new();
    for (i, f) in fills.iter().enumerate() {
        if let Some((j, len)) = copies.last_mut() {
            let g = &fills[*j];
            if g.0 == f.0 && g.2 == f.2 && g.1 + *len == f.1 && g.3 + *len == f.3 {
                *len += f.4;
                continue;
            }
        }
        copies.push((i, f.4));
    }
    (before, copies, buffers)
}

fn transition(
    resource: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::*;
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer the barrier never releases.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Records `fills` into `list` ([`fill_batch_on`]).
fn record_fills(list: &HostList, fills: &[Fill]) {
    use windows::Win32::Graphics::Direct3D12::*;
    if fills.is_empty() {
        return;
    }
    let key: Vec<_> = fills
        .iter()
        .map(|f| (windows::core::Interface::as_raw(&f.stand_in) as usize, f.at, windows::core::Interface::as_raw(&f.from) as usize, f.from_at, f.len, f.first))
        .collect();
    let (before, copies, after) = plan_fills(&key);
    let mut barriers = Vec::with_capacity(before.len() + 1);
    let wait = !fill_uav_skip_on() || list.writes_unwaited();
    if wait {
        barriers.push(uav_barrier());
    }
    barriers.extend(before.iter().map(|&i| {
        transition(&fills[i].stand_in, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST)
    }));
    let back: Vec<_> = after
        .iter()
        .map(|&i| {
            transition(&fills[i].stand_in, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)
        })
        .collect();
    let l = &list.list;
    // SAFETY: an open list; live buffers (the list keeps the stand-ins,
    // its steps the host copies, until it is reset), the ranges inside
    // them (a merged copy covers two fills' ranges that meet in both).
    unsafe {
        if !barriers.is_empty() {
            l.ResourceBarrier(&barriers);
        }
        for &(i, len) in &copies {
            let f = &fills[i];
            l.CopyBufferRegion(&f.stand_in, f.at, &f.from, f.from_at, len);
        }
        l.ResourceBarrier(&back);
    }
    if wait {
        // A barrier for all unordered access: every earlier shader write waited for.
        list.waited();
    }
    FILL_STATS[6].fetch_add(copies.len() as u64, Ordering::Relaxed);
    FILL_STATS[7].fetch_add(1 + u64::from(!barriers.is_empty()), Ordering::Relaxed);
    for f in fills.iter().filter(|f| f.check) {
        compare_fill(list, &f.stand_in, f.at, f.src, f.len);
    }
}

/// A fill of upload memory from `[src, src + len)` made a submission wait.
pub(crate) fn fill_blocked(src: usize, len: usize) {
    if !fills_on() || len == 0 {
        return;
    }
    let learned = LEARNED_FILLS.lock().unwrap_or_else(|e| e.into_inner());
    if learned.iter().any(|&(a, b)| a <= src && src + len <= b) {
        return;
    }
    FILL_STATS[2].fetch_add(1, Ordering::Relaxed);
    drop(learned);
    let (src, len) = match repeat_whole(src, len) {
        Some(whole) => whole,
        None => learned_range(src, len),
    };
    learn(&mut LEARNED_FILLS.lock().unwrap_or_else(|e| e.into_inner()), src, len);
}

/// `VICTORY_CARD_LEARN_REPEAT=<n>` (0, off, by default): with sources
/// learned by range or block, the title buffer of a fill that made a
/// submission wait is learned whole once its fills did so `n` times (each
/// at a place not learned yet). A ring the title reads a new place of every
/// frame, which a flight wrote, made a submission wait every frame (run
/// zall13g: 1.0 settles a frame, 61 ms); whole learning for every source
/// sent 10-13k fills a frame through the card (run zall10).
fn learn_repeat() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_CARD_LEARN_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Title buffers (start) whose fills made submissions wait, and how often.
static WAITED_BY_BUFFER: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

/// [`learn_repeat`]: the whole title buffer to learn for a wait of a fill
/// at `[src, src + len)`, if its buffer's fills waited often enough.
fn repeat_whole(src: usize, len: usize) -> Option<(usize, usize)> {
    let n = learn_repeat();
    if n == 0 {
        return None;
    }
    let copy = crate::shadow::find(src as u64, len as u64)?;
    let start = copy.title_start();
    let count = {
        let mut waited = WAITED_BY_BUFFER.lock().unwrap_or_else(|e| e.into_inner());
        count_wait(&mut waited, start)
    };
    (count >= n).then(|| (start as usize, copy.left(start) as usize))
}

/// Counts a wait for title buffer `start` in `waited`: how many now.
fn count_wait(waited: &mut Vec<(u64, u32)>, start: u64) -> u32 {
    match waited.iter_mut().find(|w| w.0 == start) {
        Some(w) => {
            w.1 += 1;
            w.1
        }
        None => {
            waited.push((start, 1));
            1
        }
    }
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
    let start = std::time::Instant::now();
    let filled = fill(list, src, len);
    if filled.is_some() {
        FILL_STATS[FILL_US].fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
    filled
}

/// [`try_fill`].
fn fill(list: &HostList, src: u64, len: u64) -> Option<u64> {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut under = false;
    let mut chosen = || {
        if read_on_card() {
            under = crate::host::flights_write(src as usize, len as usize);
            if under {
                READ_ON_CARD_STATS[0].fetch_add(1, Ordering::Relaxed);
            }
            under
        } else {
            overlaps(&LEARNED_FILLS.lock().unwrap_or_else(|e| e.into_inner()), src as usize, len as usize)
        }
    };
    if !fills_on() || len == 0 || !chosen() && !(after_write_on() && list.gpu_written(src as usize, len as usize)) {
        return None;
    }
    let from = crate::shadow::find(src, len)?;
    if from.left(src) < len {
        return None;
    }
    if let Some(gpu) = group_fill(list, &from, src, len) {
        FILL_STATS[0].fetch_add(1, Ordering::Relaxed);
        FILL_STATS[1].fetch_add(len, Ordering::Relaxed);
        return Some(gpu);
    }
    let me = std::ptr::from_ref(list) as usize;
    // In a batch: the same bytes filled already are shown there.
    let same = BATCH.with_borrow(|b| {
        let (_, fills) = b.as_ref().filter(|b| b.0 == me)?;
        let f = fills.iter().find(|f| f.src == src && f.len == len && f.from == from.buffer)?;
        // SAFETY: a live buffer.
        Some(unsafe { f.stand_in.GetGPUVirtualAddress() } + f.at)
    });
    if let Some(gpu) = same {
        FILL_STATS[0].fetch_add(1, Ordering::Relaxed);
        FILL_STATS[1].fetch_add(len, Ordering::Relaxed);
        return Some(gpu);
    }
    let (stand_in, at, first) = list.card_scratch(len)?;
    // The title's bytes sent up before the submission (where flights write
    // the memory, the refresh leaves their bytes alone).
    if from.refresh(list, src, len) {
        FILL_STATS[FILL_REFRESHES].fetch_add(1, Ordering::Relaxed);
    }
    if from.writes_pending() {
        FILL_STATS[FILL_UNDER_FLIGHTS].fetch_add(1, Ordering::Relaxed);
    }
    let check = match fills_check_mode() {
        1 => true,
        2 => under,
        3 => !under,
        _ => false,
    } && FILL_STATS[3].load(Ordering::Relaxed) < CHECKS_MOST;
    // SAFETY: a live buffer.
    let gpu = unsafe { stand_in.GetGPUVirtualAddress() } + at;
    let fill = Fill { stand_in, at, from: from.buffer.clone(), from_at: from.offset(src), len, first, src, check };
    let Some(Fill { stand_in, at, first, .. }) = BATCH.with_borrow_mut(|b| match b {
        Some((of, fills)) if *of == me => {
            fills.push(fill);
            None
        }
        _ => Some(fill),
    }) else {
        FILL_STATS[0].fetch_add(1, Ordering::Relaxed);
        FILL_STATS[1].fetch_add(len, Ordering::Relaxed);
        return Some(gpu);
    };
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
    FILL_STATS[6].fetch_add(1, Ordering::Relaxed);
    FILL_STATS[7].fetch_add(2, Ordering::Relaxed);
    if check {
        compare_fill(list, &stand_in, at, src, len);
    }
    Some(gpu)
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
        let copies = FILL_STATS[6].swap(0, Ordering::Relaxed) as f64 / frames;
        let barriers = FILL_STATS[7].swap(0, Ordering::Relaxed) as f64 / frames;
        let [us, under, refreshes, uses, groups] = [FILL_US, FILL_UNDER_FLIGHTS, FILL_REFRESHES, FILL_GROUP_USES, FILL_GROUPS]
            .map(|i| FILL_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        let [stale_us, stale_calls] = [0, 1].map(|i| STALE_CHECK[i].swap(0, Ordering::Relaxed) as f64 / frames);
        let [under_fills, under_copies] = [0, 1].map(|i| READ_ON_CARD_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        if read_on_card() {
            tracing::info!(
                "reads on the card under flights' writes per frame (VICTORY_READ_ON_CARD): {under_fills:.0} fills, \
                 {under_copies:.1} title buffer copies"
            );
        }
        tracing::info!(
            "card fills' cost per frame: {:.2} ms recording them; {under:.0} from host copies flights may write; \
             {refreshes:.0} refreshes they added; submissions' check of refreshes under flights {:.2} ms in \
             {stale_calls:.1} calls; {groups:.0} blocks copied for the lists, {uses:.0} fills shown one copied \
             before (VICTORY_FILL_GROUP_KB)",
            us / 1e3,
            stale_us / 1e3,
        );
        tracing::info!(
            "constants copied on the card instead of filled per frame (VICTORY_CARD_FILLS=1): {made:.1}, {:.1} KB, in \
             {copies:.1} copies and {barriers:.1} barrier calls (VICTORY_FILL_BATCH); sources learned {}; compared {} \
             (alike {}, different {})",
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
         {}; compared {} (alike {}, alike a later copy to the same place landed with it {}, different {}, of them \
         wrong on the card right after the copy {})",
        bytes / 1024.0,
        STATS[2].load(Ordering::Relaxed),
        STATS[3].load(Ordering::Relaxed),
        STATS[4].load(Ordering::Relaxed),
        STATS[6].load(Ordering::Relaxed),
        STATS[5].load(Ordering::Relaxed),
        STATS[7].load(Ordering::Relaxed),
    );
}

/// What a compared card copy left ([`landed`]).
#[derive(Debug, PartialEq)]
enum Verdict {
    /// Title memory holds the source.
    Alike,
    /// It holds the source of a later copy to the same place that landed
    /// with it.
    Superseded,
    /// Neither: the first byte that differs, and whether the card's
    /// destination right after the copy was not the source either.
    Different(Option<usize>, bool),
}

fn verdict<'a>(now: &[u8], src: &[u8], card_dst: &[u8], mut later: impl Iterator<Item = &'a [u8]>) -> Verdict {
    if now == src {
        return Verdict::Alike;
    }
    if later.any(|s| s == now) {
        return Verdict::Superseded;
    }
    let first = now.iter().zip(src).position(|(a, b)| a != b);
    Verdict::Different(first, card_dst != src)
}

#[cfg(test)]
mod tests {
    use super::{
        GroupStep, Verdict, count_wait, group_good, group_start, group_step, in_blocks, inside, learn, overlaps, plan_fills,
        refresh_due, verdict,
    };

    /// With the second fill rule a block is noted first, copied for the
    /// next fill while nothing was written, used after; anything written
    /// starts it over.
    #[test]
    fn a_block_is_copied_for_its_second_fill_and_used_after() {
        assert_eq!(group_step(None, false, true), GroupStep::Note);
        assert_eq!(group_step(Some(false), true, true), GroupStep::Copy);
        assert_eq!(group_step(Some(false), false, true), GroupStep::Note);
        assert_eq!(group_step(Some(true), true, true), GroupStep::Use);
        assert_eq!(group_step(Some(true), false, true), GroupStep::Note);
        // Without it, the first fill copies.
        assert_eq!(group_step(None, false, false), GroupStep::Copy);
        assert_eq!(group_step(Some(true), false, false), GroupStep::Copy);
        assert_eq!(group_step(Some(true), true, false), GroupStep::Use);
    }

    /// Waits are counted per title buffer.
    #[test]
    fn waits_are_counted_per_buffer() {
        let mut w = Vec::new();
        assert_eq!(count_wait(&mut w, 0x1000), 1);
        assert_eq!(count_wait(&mut w, 0x2000), 1);
        assert_eq!(count_wait(&mut w, 0x1000), 2);
    }

    /// A group is used again only while no shader that may write was
    /// recorded since it was made and no declared write meets it.
    #[test]
    fn a_group_is_dropped_after_any_writing_call_or_a_declared_write() {
        assert!(group_good(7, 7, false));
        assert!(!group_good(7, 8, false));
        assert!(!group_good(7, 7, true));
    }

    /// A group's range is refreshed once, and again after a title copy done
    /// at submission (run zall12w: refreshed only before such a copy, the
    /// group's constants were the bytes from before it).
    #[test]
    fn a_groups_ranges_are_refreshed_again_after_a_title_copy() {
        let (mut refreshed, mut epoch) = (Vec::new(), 3);
        assert!(refresh_due(&mut refreshed, &mut epoch, 3, 0x1000, 0x100));
        assert!(!refresh_due(&mut refreshed, &mut epoch, 3, 0x1000, 0x100));
        assert!(!refresh_due(&mut refreshed, &mut epoch, 3, 0x1040, 0x40));
        assert!(refresh_due(&mut refreshed, &mut epoch, 3, 0x10f0, 0x20));
        // A title copy recorded since: everything again.
        assert!(refresh_due(&mut refreshed, &mut epoch, 4, 0x1040, 0x40));
        assert!(refresh_due(&mut refreshed, &mut epoch, 4, 0x1000, 0x100));
        assert!(!refresh_due(&mut refreshed, &mut epoch, 4, 0x1080, 0x10));
    }

    /// A group starts at the first address of its phase in its block, so
    /// every fill of that phase is 256-byte aligned in the stand-in.
    #[test]
    fn a_groups_fills_of_one_phase_are_aligned_in_its_stand_in() {
        assert_eq!(group_start(0x1_0000, 0x40), 0x1_0040);
        assert_eq!(group_start(0x1_0080, 0x40), 0x1_0140);
        assert_eq!(group_start(0x1_0040, 0x40), 0x1_0040);
        assert_eq!(group_start(0x1_0000, 0), 0x1_0000);
        for src in [0x1_0140u64, 0x1_8f40, 0x1_ff40] {
            let lo = group_start(0x1_0000, src % 256);
            assert!(lo <= src && (src - lo).is_multiple_of(256));
        }
        let r = [(0x100, 0x200), (0x400, 0x500)];
        assert!(inside(&r, 0x100, 0x100) && inside(&r, 0x480, 0x10));
        assert!(!inside(&r, 0x1f0, 0x20) && !inside(&r, 0x300, 0x10) && !inside(&r, 0x500, 1));
    }

    /// A learned range in blocks: aligned in title memory, cut to its
    /// buffer; without a block size the whole buffer.
    #[test]
    fn learned_blocks_are_aligned_and_cut_to_the_buffer() {
        let buffer = (0x1_0800, 0x5_0000);
        assert_eq!(in_blocks(0x2_1000, 0x100, 0x1_0000, buffer), (0x2_0000, 0x1_0000));
        assert_eq!(in_blocks(0x2_ff00, 0x200, 0x1_0000, buffer), (0x2_0000, 0x2_0000));
        assert_eq!(in_blocks(0x1_0900, 0x10, 0x1_0000, buffer), (0x1_0800, 0xf800));
        assert_eq!(in_blocks(0x4_ff00, 0x100, 0x1_0000, buffer), (0x4_0000, 0x1_0000));
        assert_eq!(in_blocks(0x4_0000, 0x100, usize::MAX, buffer), (0x1_0800, 0x3_f800));
    }

    /// A batch of fills: each stand-in buffer taken to a copy destination
    /// once (not at all where the batch has its first use: the common
    /// state) and back once; fills that follow on in both buffers one copy.
    #[test]
    fn a_batch_of_fills_transitions_each_stand_in_once_and_joins_fills_that_follow_on() {
        // (stand-in, at, source, at, bytes, first use)
        let fills = [
            (1, 0x100, 7, 0x1000, 0x100, false),
            (1, 0x200, 7, 0x1100, 0x100, false),
            (1, 0x300, 7, 0x1300, 0x40, false),
            (2, 0x0, 7, 0x1340, 0x40, true),
            (2, 0x100, 8, 0x0, 0x100, false),
        ];
        let (before, copies, after) = plan_fills(&fills);
        assert_eq!(before, [0]);
        assert_eq!(copies, [(0, 0x200), (2, 0x40), (3, 0x40), (4, 0x100)]);
        assert_eq!(after, [0, 3]);
        // One stand-in used before, another first used in the batch (common state).
        let (before, _, after) = plan_fills(&[(3, 0x100, 7, 0, 0x100, false), (4, 0, 7, 0x100, 0x100, true)]);
        assert_eq!((before, after), (vec![0], vec![0, 1]));
        assert_eq!(plan_fills(&[]), (vec![], vec![], vec![]));
    }

    /// A copy whose place a later landed copy wrote is not counted
    /// different; one wrong on the card already is told apart.
    #[test]
    fn verdicts_tell_later_copies_and_wrong_copies_apart() {
        let (a, b, c) = ([1u8, 2, 3, 4], [1u8, 2, 9, 4], [7u8, 7, 7, 7]);
        assert_eq!(verdict(&a, &a, &a, std::iter::empty()), Verdict::Alike);
        assert_eq!(verdict(&b, &a, &a, [&c[..], &b[..]].into_iter()), Verdict::Superseded);
        assert_eq!(verdict(&b, &a, &a, [&c[..]].into_iter()), Verdict::Different(Some(2), false));
        assert_eq!(verdict(&b, &a, &c, std::iter::empty()), Verdict::Different(Some(2), true));
    }

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
