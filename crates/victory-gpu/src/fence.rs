//! Fences: a 64-bit value and the events waiting for it. Three things move
//! a value: the title's CPU signals, our queues' signals and bottom-of-pipe
//! writes a command list makes at the fence's GPU address. All three wake
//! the waits, so an event or a blocking wait ends once the value it asked
//! for is reached, whichever moved it.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    /// Wakes blocking waits (`SetEventOnCompletion` with a null event)
    /// whenever the value moves.
    moved: Condvar,
}

impl Drop for Fence {
    fn drop(&mut self) {
        live().remove(&((&raw const self.value) as usize));
    }
}

/// Live fences by the address of their value (their GPU address), for
/// bottom-of-pipe writes that land on one. Locked before a fence's
/// `waiters`, never after.
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// How often a blocking wait looks at the value without being woken (a
/// value moved by a path we do not see, or work a queue holds that has
/// become free to run), and how long it waits before saying it is stuck.
const RECHECK: Duration = Duration::from_millis(100);
const STALL: Duration = Duration::from_secs(5);

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        moved: Condvar::new(),
    });
    let at = (&raw const fence.value) as usize;
    live().insert(at, (&raw const *fence) as usize);
    // Dropped (and so unregistered) here if the title asked for an
    // interface a fence does not have.
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if !event.is_null() {
        waiters.push((value, event as usize));
        return S_OK;
    }
    // A null event: block until the value is reached, as D3D12 does (this
    // answered E_NOTIMPL at once, so a title that waits this way went on
    // before the value was reached).
    drop(waiters);
    // With submissions sent to the card and not waited for
    // (`crate::host::late`) the signal may be behind them: they are
    // settled, which sets it.
    #[cfg(windows)]
    if crate::host::late() {
        crate::host::settle_for_fence();
    }
    let started = Instant::now();
    let mut warned = Duration::ZERO;
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    while fence.value.load(Ordering::Acquire) < value {
        let (guard, timed_out) = fence
            .moved
            .wait_timeout(waiters, RECHECK)
            .unwrap_or_else(|e| e.into_inner());
        waiters = guard;
        if !timed_out.timed_out() {
            continue;
        }
        // What a queue holds may have become free to run (its own signal
        // may be what this waits for).
        #[cfg(windows)]
        {
            drop(waiters);
            crate::queue::release_held(false);
            waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
        }
        let waited = started.elapsed();
        if waited >= warned + STALL {
            warned = waited;
            tracing::warn!(
                value,
                completed = fence.value.load(Ordering::Acquire),
                seconds = waited.as_secs(),
                "blocking fence wait still waiting"
            );
        }
    }
    S_OK
}
/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    let waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    wake(fence, waiters);
}

/// Sets the events whose value the fence has reached and wakes the
/// blocking waits.
fn wake(fence: &Fence, mut waiters: MutexGuard<'_, Vec<(u64, usize)>>) {
    let value = fence.value.load(Ordering::Acquire);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    fence.moved.notify_all();
}

/// After a bottom-of-pipe write of `bytes` at `at`: wakes the waits of a
/// fence whose value the write touched.
pub(crate) fn written(at: usize, bytes: usize) {
    let end = at.saturating_add(bytes);
    let live = live();
    let Some((&value_at, &fence)) = live.range(..end).next_back() else {
        return;
    };
    if value_at + size_of::<u64>() <= at {
        return;
    }
    // SAFETY: registered fences are alive until their drop, which takes
    // the registry lock we hold.
    let fence = unsafe { &*(fence as *const Fence) };
    wake(
        fence,
        fence.waiters.lock().unwrap_or_else(|e| e.into_inner()),
    );
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    type Wait = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
    type Signal = extern "system" fn(This, u64) -> HRESULT;

    fn new_fence(initial: u64) -> This {
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), initial, 0, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        f
    }

    fn release_fence(f: This) {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    fn address(f: This) -> usize {
        slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as usize
    }

    /// A fence pointer that may cross threads (the fence is `Sync`).
    #[derive(Clone, Copy)]
    struct Shared(This);
    // SAFETY: fences are made for use from many threads.
    unsafe impl Send for Shared {}

    /// Runs a blocking wait for `value` on another thread; the receiver
    /// gets its result once it returns.
    fn blocking_wait(f: This, value: u64) -> std::sync::mpsc::Receiver<HRESULT> {
        let (done, result) = std::sync::mpsc::channel();
        let f = Shared(f);
        std::thread::spawn(move || {
            let f = f;
            let wait = slot::<Wait>(f.0, 0x48);
            let _ = done.send(wait(f.0, value, std::ptr::null_mut()));
        });
        result
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(1);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<Wait>(f, 0x48);
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // The title reads the value at the fence's address.
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(address(f) as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[test]
    fn blocking_wait_returns_at_once_when_reached() {
        let f = new_fence(7);
        let wait = slot::<Wait>(f, 0x48);
        assert_eq!(wait(f, 7, std::ptr::null_mut()), S_OK);
        assert_eq!(wait(f, 3, std::ptr::null_mut()), S_OK);
        release_fence(f);
    }

    /// Used to answer `E_NOTIMPL` at once, so the title went on before the
    /// value was reached.
    #[test]
    fn blocking_wait_blocks_until_signalled() {
        let f = new_fence(0);
        let result = blocking_wait(f, 2);
        let short = Duration::from_millis(200);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned before any signal"
        );
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(signal(f, 1), S_OK);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned below its value"
        );
        assert_eq!(signal(f, 2), S_OK);
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[test]
    fn bottom_of_pipe_write_releases_blocking_wait() {
        let f = new_fence(0);
        let result = blocking_wait(f, 0x1_0000_0003);
        std::thread::sleep(Duration::from_millis(50));
        // The low half alone does not reach the value.
        crate::queue::Written::Value(3).write(address(f) as *mut u8, 4);
        assert!(result.recv_timeout(Duration::from_millis(200)).is_err());
        crate::queue::Written::Value(0x1_0000_0003).write(address(f) as *mut u8, 8);
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn bottom_of_pipe_write_sets_waiting_event() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(0);
        // SAFETY: plain manual-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        assert_eq!(slot::<Wait>(f, 0x48)(f, 9, event), S_OK);
        crate::queue::Written::Value(9).write(address(f) as *mut u8, 8);
        // SAFETY: a valid event.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[test]
    fn released_fences_leave_the_registry() {
        let f = new_fence(0);
        let at = address(f);
        assert!(live().contains_key(&at));
        release_fence(f);
        assert!(!live().contains_key(&at));
        // A write next to (not on) a live fence's value wakes nothing and
        // does not fault.
        let g = new_fence(0);
        let mut other = [0u64; 2];
        written(other.as_mut_ptr() as usize, 16);
        written(address(g) + 8, 8);
        release_fence(g);
    }
}
