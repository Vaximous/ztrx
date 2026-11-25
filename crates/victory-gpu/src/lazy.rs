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
    if early_on() && len > 0 {
        early_open(at, at.saturating_add(len), EARLY_KERNEL);
    }
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
    if early_on() && len > 0 {
        let end = at.saturating_add(len);
        with_early(|s| {
            let starts: Vec<usize> =
                s.runs.range(..end).rev().take_while(|(_, r)| r.end > at).map(|(&a, _)| a).collect();
            for a in starts {
                s.runs.remove(&a);
            }
        });
    }
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
    if early_on() && early_fault(at, rip) {
        return true;
    }
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
    if r.NumberParameters >= 1 {
        KIND.with(|k| k.set(r.ExceptionInformation[0]));
    }
    (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS && r.NumberParameters >= 2)
        .then(|| (r.ExceptionInformation[1], c.Rip))
}

thread_local! {
    /// The kind of the access [`violation`] last saw on this thread: 0 a
    /// read, 1 a write, 8 an instruction fetch.
    static KIND: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

// The early-read check (`VICTORY_EARLY_READ_CHECK=1`).
//
// Title memory a flight in the air declares its shaders write must not be
// read by the CPU before the flight's landing merged what they wrote: the
// title waits for the flight's fence, and our steps that read it settle
// first. If something reads it earlier it takes the bytes from before the
// flight, which would show as stale blocks. With the check, when a flight
// goes to the card the whole 4 KiB pages of each piece it declares (of
// copies in the card's memory whose memory is write-watched) are made
// `PAGE_NOACCESS`; the landing of the newest flight that declares the
// piece gives them their protection back before it merges. A touch in
// between faults: the page is opened, and the touch noted (read or write,
// our code or the title's, where, by which instruction). At the landing,
// a noted read of a piece the flight's compare found changed is an early
// read that took stale bytes, and is logged. Nothing read changes: the
// check costs the faults and the protection calls.
//
// The kernel does not fault: our writes through it open the pages first
// (counted). Our reads through it need the guarded reads (which fault).

/// `VICTORY_EARLY_READ_CHECK=1` (needs the relay and the guarded reads;
/// not with `VICTORY_LAZY_BACK`).
pub(crate) fn early_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if !std::env::var("VICTORY_EARLY_READ_CHECK").is_ok_and(|v| v == "1") {
            return false;
        }
        if crate::shadow::relay_stage() < 2 || !crate::guarded::on() || on() {
            tracing::warn!(
                "VICTORY_EARLY_READ_CHECK=1 needs the relay and the guarded reads, and not VICTORY_LAZY_BACK: left off"
            );
            return false;
        }
        install();
        true
    })
}

struct EarlyRun {
    end: usize,
    /// The copy (its address) and the piece's number.
    key: (usize, u32),
    /// The place in the queue's order of the newest flight declaring it.
    flight: u64,
}

/// A touch of a protected piece before its landing.
#[derive(Clone, Copy)]
struct Touch {
    read: bool,
    ours: bool,
    at: usize,
    rip: u64,
    frame: u64,
}

#[derive(Default)]
struct EarlyState {
    runs: BTreeMap<usize, EarlyRun>,
    /// Pieces touched while protected, until their landing.
    touched: HashMap<(usize, u32), Touch>,
    /// Instructions that read pieces their flight then changed, and how
    /// often, since the start.
    rips: HashMap<(u64, bool), u64>,
}

static EARLY: Mutex<Option<EarlyState>> = Mutex::new(None);

fn with_early<R>(f: impl FnOnce(&mut EarlyState) -> R) -> R {
    let mut s = EARLY.lock().unwrap_or_else(|e| e.into_inner());
    f(s.get_or_insert_with(EarlyState::default))
}

/// Since the last frame line:
const EARLY_PROTECTED: usize = 0; // pieces protected at sends
const EARLY_PROTECTED_B: usize = 1; // their bytes
const EARLY_TITLE_READ: usize = 2; // touches before landing, by kind and by whom
const EARLY_TITLE_WRITE: usize = 3;
const EARLY_OURS_READ: usize = 4;
const EARLY_OURS_WRITE: usize = 5;
const EARLY_KERNEL: usize = 6; // opened for our writes through the kernel
const EARLY_STALE_TITLE: usize = 7; // reads of pieces the flight changed: stale
const EARLY_STALE_OURS: usize = 8;
const EARLY_SKIPPED: usize = 9; // not protected: no whole page, unwatched, failed
const EARLY_HELD: usize = 10; // declared again while protected
static EARLY_STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

fn early_open(from: usize, to: usize, why: usize) {
    with_early(|s| {
        let starts: Vec<usize> =
            s.runs.range(..to).rev().take_while(|(_, r)| r.end > from).map(|(&a, _)| a).collect();
        for a in starts {
            if let Some(r) = s.runs.remove(&a) {
                let mut old = 0;
                // SAFETY: title pages we protected; only their protection changes.
                unsafe { VirtualProtect(a as *const _, r.end - a, PAGE_READWRITE, &mut old) };
                EARLY_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
        }
    });
}

/// A flight went to the card at `flight` in the queue's order, declaring
/// `pieces` written.
pub(crate) fn early_sent(flight: u64, pieces: &[Piece]) {
    with_early(|s| {
        for piece in pieces {
            let (from, to) = title_range(piece);
            let key = (std::sync::Arc::as_ptr(&piece.0) as usize, piece.3);
            let Some((a, b)) = inner(from, to) else {
                EARLY_STATS[EARLY_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            if piece.0.unwatched() {
                EARLY_STATS[EARLY_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let mut held = false;
            for (_, r) in s.runs.range_mut(..b).rev().take_while(|(_, r)| r.end > a) {
                // Declared again by a newer flight (or another copy's piece
                // over the same pages): held until the newer landing.
                r.flight = r.flight.max(flight);
                held = true;
            }
            if held {
                EARLY_STATS[EARLY_HELD].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let mut old = 0;
            // SAFETY: title pages of a copy's piece; only their protection
            // changes, and the handler or the landing gives it back.
            if unsafe { VirtualProtect(a as *const _, b - a, PAGE_NOACCESS, &mut old) } == 0 {
                EARLY_STATS[EARLY_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if old != PAGE_READWRITE {
                // SAFETY: as above, back as it was.
                unsafe { VirtualProtect(a as *const _, b - a, old, &mut old) };
                EARLY_STATS[EARLY_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            s.runs.insert(a, EarlyRun { end: b, key, flight });
            EARLY_STATS[EARLY_PROTECTED].fetch_add(1, Ordering::Relaxed);
            EARLY_STATS[EARLY_PROTECTED_B].fetch_add((b - a) as u64, Ordering::Relaxed);
        }
    });
}

/// The landing of the flights up to `floor` in the queue's order, before
/// its merges: the pieces no newer flight declares are opened, and the
/// touches of the pieces the landed flights declared (`declared`: copy
/// address, piece, whether the compare found it changed) are judged.
pub(crate) fn early_landing(floor: u64, declared: &[(usize, u32, bool)]) {
    let frame = crate::queue::presented();
    let mut stale: Vec<((usize, u32), Touch)> = Vec::new();
    with_early(|s| {
        let due: Vec<usize> = s.runs.iter().filter(|(_, r)| r.flight <= floor).map(|(&a, _)| a).collect();
        for a in due {
            if let Some(r) = s.runs.remove(&a) {
                let mut old = 0;
                // SAFETY: title pages we protected; only their protection changes.
                unsafe { VirtualProtect(a as *const _, r.end - a, PAGE_READWRITE, &mut old) };
            }
        }
        for &(copy, piece, changed) in declared {
            let Some(t) = s.touched.remove(&(copy, piece)) else {
                continue;
            };
            if changed && t.read {
                *s.rips.entry((t.rip, t.ours)).or_default() += 1;
                stale.push(((copy, piece), t));
            }
        }
        // Touches of pieces no landing judged within some seconds.
        s.touched.retain(|_, t| frame.saturating_sub(t.frame) < 600);
    });
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    for ((_, piece), t) in stale {
        EARLY_STATS[if t.ours { EARLY_STALE_OURS } else { EARLY_STALE_TITLE }].fetch_add(1, Ordering::Relaxed);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 200 {
            tracing::warn!(
                at = format_args!("{:#x}", t.at),
                rip = format_args!("{:#x}", t.rip),
                ours = t.ours,
                piece,
                read_frame = t.frame,
                landing_frame = frame,
                "memory a flight in the air changed was read before its landing: the read took stale bytes \
                 (VICTORY_EARLY_READ_CHECK)"
            );
        }
    }
}

fn early_fault(at: usize, rip: u64) -> bool {
    let read = KIND.with(std::cell::Cell::get) == 0;
    let by_us = ours(rip);
    let frame = crate::queue::presented();
    let mut hit = false;
    with_early(|s| {
        let starts: Vec<usize> =
            s.runs.range(..at + 1).rev().take_while(|(_, r)| r.end > at).map(|(&a, _)| a).collect();
        for a in starts {
            if let Some(r) = s.runs.remove(&a) {
                let mut old = 0;
                // SAFETY: title pages we protected; only their protection changes.
                unsafe { VirtualProtect(a as *const _, r.end - a, PAGE_READWRITE, &mut old) };
                hit = true;
                let touch = Touch { read, ours: by_us, at, rip, frame };
                // A read is what matters: it replaces a write noted first.
                let t = s.touched.entry(r.key).or_insert(touch);
                if read && !t.read {
                    *t = touch;
                }
            }
        }
    });
    if hit {
        let i = match (by_us, read) {
            (false, true) => EARLY_TITLE_READ,
            (false, false) => EARLY_TITLE_WRITE,
            (true, true) => EARLY_OURS_READ,
            (true, false) => EARLY_OURS_WRITE,
        };
        EARLY_STATS[i].fetch_add(1, Ordering::Relaxed);
    }
    hit
}

/// For the frame-rate line.
pub(crate) fn early_line(frames: f64) {
    if !early_on() {
        return;
    }
    let s: Vec<f64> = EARLY_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let (held, top) = with_early(|st| {
        let mut top: Vec<((u64, bool), u64)> = st.rips.iter().map(|(&k, &n)| (k, n)).collect();
        top.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
        top.truncate(6);
        (st.runs.len(), top)
    });
    let top: Vec<String> =
        top.iter().map(|((rip, us), n)| format!("{rip:#x}{} {n}", if *us { " (ours)" } else { "" })).collect();
    tracing::info!(
        "early-read check per frame (VICTORY_EARLY_READ_CHECK=1): {:.0} pieces {:.1} MB protected at sends ({:.0} not, \
         {:.0} declared again while held); touched before their landing: by the title {:.1} reads {:.1} writes, by our \
         code {:.1} reads {:.1} writes; opened for our kernel writes {:.1}; reads of pieces the flight changed (stale) \
         by the title {:.2}, by our code {:.2}; {held} runs held; the instructions of stale reads since the start: [{}]",
        s[EARLY_PROTECTED],
        s[EARLY_PROTECTED_B] / (1 << 20) as f64,
        s[EARLY_SKIPPED],
        s[EARLY_HELD],
        s[EARLY_TITLE_READ],
        s[EARLY_TITLE_WRITE],
        s[EARLY_OURS_READ],
        s[EARLY_OURS_WRITE],
        s[EARLY_KERNEL],
        s[EARLY_STALE_TITLE],
        s[EARLY_STALE_OURS],
        top.join(", "),
    );
}
