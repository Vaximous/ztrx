//! Loops a console wave64 program's two host waves of 32 can each run on their own lanes
//! (`VICTORY_HALF_LOCAL=1`; off by default).
//!
//! A loop whose branches the wave takes together ([`Structured::wide`]) has its two host
//! waves meet at every such branch (and at every first-lane read) to agree on the console
//! wave's decision. A serial walk that steps through a list one element at a time, with
//! each lane leaving after its own number of steps, meets twice a step that way. In the
//! loops found here each host wave decides on its own lanes instead:
//!
//! - nothing in the loop needs the other host wave but its branch decisions and first-lane
//!   reads (no store, atomic, LDS access, barrier, lane read, lane mask read as data);
//! - no scalar it writes is read after it (a host wave that left early holds the scalars of
//!   its last step, not the console's);
//! - its exits come together in one block, past blocks of no such work, so that both host
//!   waves reach the meetings after the loop the same way.
//!
//! shortcut: the branch decisions are the host wave's own, which is the console's where the
//! values deciding them are the same in every lane still walking (a list walked in step by
//! every lane, a count of steps), and a host wave with no lane left leaves (its exit
//! condition holds of no lane, or the walk's mask has no lane of it). A loop deciding on
//! values that differ by lane would go its own way per host wave; the switch is off by
//! default until the cases it is on for are checked.

use std::collections::HashSet;

use crate::ir::{Exit, Program, Reg, Value};
use crate::structure::Structured;

/// `VICTORY_HALF_LOCAL=1`: wide loops that qualify run per host wave ([`find`]).
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_HALF_LOCAL.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HALF_LOCAL").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's switch, for its thread only.
    pub(super) static TEST_HALF_LOCAL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether instruction `i` needs nothing from the other host wave but what the loop's
/// own lowering gives per host wave (first-lane reads, branch decisions).
fn alone(st: &Structured, i: &crate::ir::Inst) -> bool {
    let n = i.name();
    if super::lower::reads_mask_as_data(st, i) {
        return false;
    }
    if let crate::gfx10::Fields::Valu(v) = &i.fields
        && (v.dpp.is_some() || v.sdwa.is_some())
    {
        return false;
    }
    if n.starts_with("v_") {
        return n == "v_readfirstlane_b32" || !(n.contains("lane") || n.starts_with("v_mbcnt"));
    }
    if n.starts_with("buffer_load") || n.starts_with("s_load") || n.starts_with("s_buffer_load") {
        return true;
    }
    n.starts_with("s_")
        && !(n.contains("store")
            || n.contains("atomic")
            || n.starts_with("s_barrier")
            || n.starts_with("s_sendmsg")
            || n.starts_with("s_setpc")
            || n.starts_with("s_swappc")
            || n.starts_with("s_getpc")
            || n.starts_with("s_dcache")
            || n.starts_with("s_memtime")
            || n.starts_with("s_memrealtime"))
}

/// The IR blocks of the loops whose host waves can each run them on their own lanes, with
/// the first block of each such loop.
pub(super) fn find(p: &Program, st: &Structured) -> (HashSet<usize>, Vec<usize>) {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut place = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        place[b] = n;
    }
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut all = HashSet::new();
    let mut heads = Vec::new();
    for &h in &order {
        // the loop of header `h`: the blocks that reach one of its back edges' sources
        // without passing `h`
        let tails: Vec<usize> = p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&t| place[t] != usize::MAX && place[t] >= place[h])
            .collect();
        if tails.is_empty() {
            continue;
        }
        let mut body: HashSet<usize> = HashSet::from([h]);
        let mut stack = tails.clone();
        while let Some(b) = stack.pop() {
            if body.insert(b) {
                stack.extend(p.blocks[b].preds.iter().copied().filter(|&q| place[q] != usize::MAX));
            }
        }
        if !body.iter().any(|b| st.wide.contains(b)) {
            continue;
        }
        if !body.iter().all(|&b| p.blocks[b].insts.iter().all(|i| alone(st, i))) {
            continue;
        }
        // the exits: to one block, or through blocks of no such work that jump to it
        let mut targets: Vec<usize> = Vec::new();
        for &b in &body {
            for s in p.blocks[b].exit.successors() {
                if !body.contains(&s) && !targets.contains(&s) {
                    targets.push(s);
                }
            }
        }
        let join = match targets.as_slice() {
            [x] => Some(*x),
            _ => targets.iter().copied().find(|&x| {
                targets.iter().all(|&t| {
                    t == x
                        || p.blocks[t].exit == Exit::Jump(x)
                            && p.blocks[t].phis.is_empty()
                            && p.blocks[t].preds.iter().all(|q| body.contains(q))
                            && p.blocks[t].insts.iter().all(|i| alone(st, i))
                })
            }),
        };
        let Some(join) = join else { continue };
        let passed: HashSet<usize> = targets.iter().copied().filter(|&t| t != join).collect();
        // no scalar the loop writes is read after it (phis outside the loop carry it on)
        let mut written: HashSet<Value> = body
            .iter()
            .flat_map(|&b| {
                let blk = &p.blocks[b];
                blk.phis
                    .iter()
                    .map(|ph| (ph.reg, ph.value))
                    .chain(blk.insts.iter().flat_map(|i| i.writes.iter().copied()))
                    .collect::<Vec<_>>()
            })
            .filter(|&(r, v)| !matches!(r, Reg::V(_)) && !is_mask(v))
            .map(|(_, v)| v)
            .collect();
        let mut grew = true;
        while grew {
            grew = false;
            for (b, blk) in p.blocks.iter().enumerate() {
                if body.contains(&b) {
                    continue;
                }
                for ph in &blk.phis {
                    if !matches!(ph.reg, Reg::V(_))
                        && ph.args.iter().any(|a| written.contains(a))
                        && written.insert(ph.value)
                    {
                        grew = true;
                    }
                }
            }
        }
        let read_after = p.blocks.iter().enumerate().any(|(b, blk)| {
            !body.contains(&b)
                && !passed.contains(&b)
                && blk.insts.iter().any(|i| i.reads.iter().any(|r| written.contains(&r.1)))
        }) || passed
            .iter()
            .any(|&b| p.blocks[b].insts.iter().any(|i| i.reads.iter().any(|r| written.contains(&r.1))));
        if read_after {
            continue;
        }
        heads.push(h);
        all.extend(body);
    }
    (all, heads)
}
