//! Cheaper ways of the copy path, ported from victory-perf (c3faf2a8,
//! branch fx-mem-copy there), each behind a switch, on by default:
//!
//! - `VICTORY_FAST_HASH`: the change hash of texture memory is
//!   [`lane_hash`], not BLAKE3 (no attacker here, only a change to see).
//!   `=0` hashes with BLAKE3, as before.
//! - `VICTORY_TEXTURE_OVERLAP`: a submission's texture refreshes run on
//!   threads of their own while the submitting thread brings buffer copies
//!   up to date; they are waited for before any step that writes title
//!   memory and before the lists go to the card. `=0` refreshes them on
//!   the submitting thread, as before.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

fn flag_on(name: &str) -> bool {
    std::env::var_os(name).is_none_or(|v| v != "0")
}

/// `VICTORY_TEXTURE_OVERLAP` (on by default; `=0` disables).
pub(crate) fn texture_overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag_on("VICTORY_TEXTURE_OVERLAP"))
}

/// `VICTORY_FAST_HASH` (on by default; `=0` disables).
pub(crate) fn fast_hash() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag_on("VICTORY_FAST_HASH"))
}

/// Since the last line: texture hashes (count, bytes, microseconds on all
/// threads); texture refreshes run beside the submitting thread, their
/// time on those threads, and the time the submitting thread waited.
static COUNT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const HASHES: usize = 0;
const HASH_BYTES: usize = 1;
const HASH_US: usize = 2;
pub(crate) const OVERLAPPED: usize = 3;
pub(crate) const OVERLAP_US: usize = 4;
pub(crate) const OVERLAP_WAIT_US: usize = 5;

pub(crate) fn add(i: usize, n: u64) {
    COUNT[i].fetch_add(n, Relaxed);
}

/// Logs the counters per frame (after the frame-rate line).
pub(crate) fn line(frames: f64) {
    let c: Vec<f64> = COUNT.iter().map(|a| a.swap(0, Relaxed) as f64 / frames.max(1.0)).collect();
    tracing::info!(
        "memcopy per frame: texture hashes {:.0} {:.1} MB {:.2} ms (VICTORY_FAST_HASH={}); texture refreshes beside \
         the submitting thread {:.0}, {:.2} ms on their threads, {:.2} ms waited for (VICTORY_TEXTURE_OVERLAP={})",
        c[HASHES],
        c[HASH_BYTES] / 1e6,
        c[HASH_US] / 1e3,
        u8::from(fast_hash()),
        c[OVERLAPPED],
        c[OVERLAP_US] / 1e3,
        c[OVERLAP_WAIT_US] / 1e3,
        u8::from(texture_overlap()),
    );
}

/// The change hash of texture memory: BLAKE3, or [`lane_hash`] with
/// `VICTORY_FAST_HASH`. Only ever compared with hashes made the same way
/// in the same run.
pub(crate) fn content_hash(bytes: &[u8]) -> blake3::Hash {
    let started = std::time::Instant::now();
    let hash = if fast_hash() { blake3::Hash::from(lane_hash(bytes)) } else { blake3::hash(bytes) };
    add(HASHES, 1);
    add(HASH_BYTES, bytes.len() as u64);
    add(HASH_US, started.elapsed().as_micros() as u64);
    hash
}

const LANES: usize = 8;
const K: [u64; LANES] = [
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
    0x1656_67b1_9e37_79f9,
    0x85eb_ca77_c2b2_ae63,
    0x27d4_eb2f_1656_67c5,
    0xff51_afd7_ed55_8ccd,
    0xc4ce_b9fe_1a85_ec53,
    0xd6e8_feb8_6659_fd93,
];

/// A 256-bit hash of `bytes` for telling whether memory changed, at memory
/// speed: eight independent lanes of 64-bit words, each step of a lane
/// `x -> (rotl(x, 23) ^ w) * k` with `k` odd. For a given state the step is
/// a bijection of the word and of the state, so a change of any one word
/// always changes its lane's final state; a change of several words is
/// missed only if their effects cancel exactly (about 2^-64 for data not
/// made to do so). The length and the tail bytes are mixed in last.
pub(crate) fn lane_hash(bytes: &[u8]) -> [u8; 32] {
    let mut lane = K;
    let mut blocks = bytes.chunks_exact(8 * LANES);
    for block in &mut blocks {
        for (i, l) in lane.iter_mut().enumerate() {
            let w = u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap_or([0; 8]));
            *l = (l.rotate_left(23) ^ w).wrapping_mul(K[i] | 1);
        }
    }
    let rest = blocks.remainder();
    for (j, word) in rest.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[..word.len()].copy_from_slice(word);
        let w = u64::from_le_bytes(w);
        lane[j] = (lane[j].rotate_left(23) ^ w).wrapping_mul(K[j] | 1);
    }
    lane[0] ^= bytes.len() as u64;
    // Finish: every lane into every output word.
    let mut out = [0u8; 32];
    for o in 0..4 {
        let mut x = (o as u64).wrapping_mul(K[7]) ^ bytes.len() as u64;
        for (i, l) in lane.iter().enumerate() {
            x = (x ^ l.rotate_left((7 * i + 13 * o) as u32 % 64)).wrapping_mul(K[(i + o) % LANES] | 1);
            x ^= x >> 29;
        }
        out[8 * o..8 * o + 8].copy_from_slice(&x.to_le_bytes());
    }
    out
}

/// Work run beside the submitting thread and waited for as a group
/// ([`texture_overlap`]). Its threads are its own: a job may itself spread
/// work over `crate::shadow`'s threads (big textures are detiled in bands).
pub(crate) struct Group {
    left: Arc<(Mutex<usize>, Condvar)>,
}

struct Pool {
    jobs: Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>,
    more: Condvar,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let pool = POOL.get_or_init(|| Pool { jobs: Mutex::new(std::collections::VecDeque::new()), more: Condvar::new() });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| n.get() / 6).clamp(1, 3);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-texprep-{i}")).spawn(move || {
                let pool = POOL.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = pool.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = pool.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    // A job's panic is its own: the thread goes on.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
            });
        }
    });
    pool
}

impl Default for Group {
    fn default() -> Group {
        Group { left: Arc::new((Mutex::new(0), Condvar::new())) }
    }
}

impl Group {
    /// Runs `job` on the pool's threads.
    pub(crate) fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        *self.left.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        let left = self.left.clone();
        let pool = pool();
        pool.jobs.lock().unwrap_or_else(|e| e.into_inner()).push_back(Box::new(move || {
            // Counted down even if the job panics.
            struct Done(Arc<(Mutex<usize>, Condvar)>);
            impl Drop for Done {
                fn drop(&mut self) {
                    let mut n = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
                    *n -= 1;
                    if *n == 0 {
                        self.0.1.notify_all();
                    }
                }
            }
            let _done = Done(left);
            job();
        }));
        pool.more.notify_one();
    }

    /// Whether jobs are still to finish.
    pub(crate) fn busy(&self) -> bool {
        *self.left.0.lock().unwrap_or_else(|e| e.into_inner()) > 0
    }

    /// Returns when every job spawned so far has run. While waiting this
    /// thread runs jobs still queued (any group's: they are all
    /// independent and all waited for by someone).
    pub(crate) fn join(&self) {
        loop {
            if !self.busy() {
                return;
            }
            let job = pool().jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
            match job {
                Some(job) => {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
                None => break,
            }
        }
        let mut n = self.left.0.lock().unwrap_or_else(|e| e.into_inner());
        while *n > 0 {
            n = self.left.1.wait(n).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lane_hash_sees_every_one_byte_change() {
        let mut data: Vec<u8> = (0..4099u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let base = lane_hash(&data);
        assert_eq!(base, lane_hash(&data), "the same bytes hash the same");
        for at in [0, 1, 7, 8, 63, 64, 65, 511, 2048, 4095, 4096, 4098] {
            let was = data[at];
            for flip in [1u8, 0x80, 0xff] {
                data[at] = was ^ flip;
                assert_ne!(lane_hash(&data), base, "byte {at} ^ {flip:#x}");
            }
            data[at] = was;
        }
        assert_eq!(lane_hash(&data), base);
    }

    #[test]
    fn the_lane_hash_tells_lengths_and_zeros_apart() {
        let zeros = vec![0u8; 256];
        let mut seen = std::collections::HashSet::new();
        for n in 0..=256 {
            assert!(seen.insert(lane_hash(&zeros[..n])), "length {n}");
        }
    }

    #[test]
    fn the_lane_hash_sees_swapped_words() {
        // Two lanes' words exchanged, and two words of one lane exchanged.
        let mut a = vec![0u8; 1024];
        a[0] = 1;
        a[8] = 2;
        let mut b = a.clone();
        b.swap(0, 8);
        assert_ne!(lane_hash(&a), lane_hash(&b));
        let mut c = a.clone();
        c[64] = 1;
        c[0] = 0;
        assert_ne!(lane_hash(&a), lane_hash(&c));
    }

    #[test]
    fn a_group_waits_for_all_its_jobs() {
        let group = Group::default();
        let sum = Arc::new(AtomicU64::new(0));
        for i in 1..=100u64 {
            let sum = sum.clone();
            group.spawn(move || {
                std::thread::sleep(std::time::Duration::from_micros(50));
                sum.fetch_add(i, Relaxed);
            });
        }
        group.join();
        assert_eq!(sum.load(Relaxed), 5050);
        assert!(!group.busy());
        // Joined again with nothing spawned: returns at once.
        group.join();
    }

    #[test]
    fn a_job_that_panics_is_still_counted_done() {
        let group = Group::default();
        group.spawn(|| panic!("a job's panic"));
        group.join();
        assert!(!group.busy());
    }
}
