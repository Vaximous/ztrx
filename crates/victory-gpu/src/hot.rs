//! `VICTORY_HOT_DISPATCH`: what the dispatches that take the card longest
//! bind, logged for one run. A timing run (`VICTORY_GPU_TIMING`, zall16t)
//! found single dispatches of small compute programs taking 3-7 ms of the
//! card a frame (5cf00dc3 7.4 ms, cb6adcc6 5.5, 14264103 3.4, 2f03a2df
//! 1.7, 0a5ecfee 1.6): too short a program for its arithmetic to cost
//! that. What such a dispatch costs is likely its memory: a host copy in
//! the PC's memory read or written over the bus (only copies of
//! `VICTORY_CARD_BUFFERS_MIN_MB` and more live in the card's,
//! `crate::shadow::cards`), many groups, or atomics on one address.
//!
//! For each such dispatch (the first few times): its group counts, each
//! buffer view it binds (read or written, title address, bytes, the host
//! copy's size and where it lives), its constant buffers, and the buffer
//! atomics in its code, those on one address (no lane's own index or
//! offset) counted apart.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

/// The shader hash beginnings followed: `VICTORY_HOT_DISPATCH` (by commas;
/// `1`: the five of zall16t; unset: none).
fn followed() -> &'static [String] {
    static HASHES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    HASHES.get_or_init(|| match std::env::var("VICTORY_HOT_DISPATCH") {
        Ok(v) if v == "1" => ["5cf00dc3", "cb6adcc6", "14264103", "2f03a2df", "0a5ecfee"].map(str::to_owned).to_vec(),
        Ok(v) => v.split(',').map(str::trim).filter(|h| !h.is_empty() && *h != "0").map(str::to_owned).collect(),
        Err(_) => Vec::new(),
    })
}

/// Whether the dispatch of compute program `hash` is followed.
pub(crate) fn wanted(hash: &str) -> bool {
    let hashes = followed();
    !hashes.is_empty() && hashes.iter().any(|h| hash.starts_with(h.as_str()))
}

struct Seen {
    /// What is bound, in words.
    views: Vec<String>,
    /// Bytes bound by where they live: the card's memory, the PC's (a
    /// host copy, or title memory in place).
    card: u64,
    system: u64,
}

thread_local! {
    static SEEN: RefCell<Option<Seen>> = const { RefCell::new(None) };
}

/// The binding of a followed dispatch begins.
pub(crate) fn start() {
    SEEN.with_borrow_mut(|s| *s = Some(Seen { views: Vec::new(), card: 0, system: 0 }));
}

/// A buffer view bound ([`crate::skip::note_buffer`]'s callers).
pub(crate) fn note_buffer(shadow: &crate::shadow::Shadow, address: u64, len: u64, uav: bool) {
    SEEN.with_borrow_mut(|s| {
        let Some(s) = s else {
            return;
        };
        let (place, card) = if shadow.carded() {
            ("the card's memory", true)
        } else if shadow.in_place() {
            ("title memory in place (the PC's)", false)
        } else {
            ("a host copy in the PC's memory", false)
        };
        if card {
            s.card += len;
        } else {
            s.system += len;
        }
        s.views.push(format!(
            "{} {address:#x}+{len:#x} of a copy of {:#x} bytes at {:#x}, in {place}",
            if uav { "writes" } else { "reads" },
            shadow.title_len(),
            shadow.title_start(),
        ));
    });
}

/// A constant buffer bound: `card` whether it is read from the card's
/// memory (a card fill or a copy there), else from upload memory.
pub(crate) fn note_constant(address: u64, size: u64, card: bool) {
    SEEN.with_borrow_mut(|s| {
        if let Some(s) = s {
            s.views.push(format!(
                "constants {address:#x}+{size:#x} from {}",
                if card { "the card's memory" } else { "upload memory (the PC's)" }
            ));
        }
    });
}

/// The binding of a followed dispatch of `shader` (`groups` of it; None:
/// indirect) ended: logged, the first few times per program.
pub(crate) fn finish(shader: &crate::pipeline::Shader, groups: Option<[u32; 3]>) {
    let Some(seen) = SEEN.with_borrow_mut(Option::take) else {
        return;
    };
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) >= 8 * followed().len() as u32 {
        return;
    }
    let (atomics, one) = atomics(&shader.container);
    let groups = groups.map_or_else(
        || "indirect".to_owned(),
        |[x, y, z]| format!("{x}x{y}x{z} = {} groups", u64::from(x) * u64::from(y) * u64::from(z)),
    );
    tracing::info!(
        "VICTORY_HOT_DISPATCH {}: {groups}; buffer atomics in its code {atomics}, {one} of them on one address; \
         {} MB bound in the card's memory, {} MB in the PC's: {}",
        shader.hash,
        seen.card >> 20,
        seen.system >> 20,
        seen.views.join("; ")
    );
}

/// The buffer atomics of a compute program's native code, and those on
/// one address (neither a lane's index nor its offset in the address).
fn atomics(container: &[u8]) -> (usize, usize) {
    use victory_shader::gfx10::Encoding;
    use victory_shader::{Container, FourCC};
    let Ok(c) = Container::parse(container) else {
        return (0, 0);
    };
    let Some(programs) = c.part(FourCC::DXIL).and_then(|p| victory_shader::native_programs(p.data).ok()) else {
        return (0, 0);
    };
    let (mut all, mut one) = (0, 0);
    for p in &programs {
        let Ok(code) = victory_shader::gfx10::decode(&p.code) else {
            continue;
        };
        for i in code.iter().filter(|i| i.encoding == Encoding::Mubuf) {
            if !i.name().is_some_and(|n| n.starts_with("buffer_atomic")) {
                continue;
            }
            all += 1;
            // MUBUF: offen bit 12, idxen bit 13.
            if i.words[0] & (3 << 12) == 0 {
                one += 1;
            }
        }
    }
    (all, one)
}
