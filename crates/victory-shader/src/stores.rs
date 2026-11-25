//! What a program's buffer stores address with, for the host's account of
//! what shaders may write. A store through a view of a whole heap is taken
//! to write the whole view (174 MB in the world, about 150 GB of declared
//! ranges a frame); where the per-lane part of a store's address depends
//! only on the lane's place in the dispatch, its extent is bounded by the
//! dispatch's size, and the uniform part (a constant, user data) can be
//! read when the call is recorded.
//!
//! This first pass only classifies: per store, the worst term its address
//! depends on ([`Term`]).

use crate::gfx10::{Encoding, Fields, Operand};
use crate::ir::{Program, Reg, Value};
use std::collections::HashMap;

/// What the address of a store depends on, from best to worst.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    /// The same in every lane: registers set before the program runs (user
    /// data, group ids), scalar loads (constants, tables), immediates.
    Uniform,
    /// Also the lane's place: the program's per-lane inputs (thread ids,
    /// for compute) and the lane's number in its wave.
    Thread,
    /// Also a value carried around a loop: bounded only by the trip count.
    Loop,
    /// Also something a lane loaded from memory.
    Data,
}

/// Per program: how many buffer stores (and atomics) it has, by the worst
/// [`Term`] of their address.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreTerms {
    pub uniform: u32,
    pub thread: u32,
    pub looped: u32,
    pub data: u32,
}

impl StoreTerms {
    pub fn total(&self) -> u32 {
        self.uniform + self.thread + self.looped + self.data
    }

    /// Whether every store's address depends on nothing worse than the
    /// lane's place (and there is at least one store).
    pub fn bounded(&self) -> bool {
        self.total() > 0 && self.looped == 0 && self.data == 0
    }
}

/// Whether `name` (of an instruction of `encoding`) stores to buffer memory
/// or is an atomic on it.
fn is_buffer_store(encoding: Encoding, name: &str) -> bool {
    matches!(encoding, Encoding::Mubuf | Encoding::Mtbuf | Encoding::Flat | Encoding::Global)
        && (name.contains("store") || name.contains("atomic"))
        && !name.contains("scratch")
}

/// Whether an instruction of `encoding` named `name` loads per-lane values
/// from memory (buffers, LDS, images).
fn loads_lanes(encoding: Encoding, name: &str) -> bool {
    match encoding {
        Encoding::Mubuf | Encoding::Mtbuf | Encoding::Flat | Encoding::Global | Encoding::Scratch | Encoding::Mimg => true,
        Encoding::Ds => !name.contains("write") || name.contains("rtn"),
        _ => false,
    }
}

enum Def {
    Input(Reg),
    Phi(usize, usize),
    Inst(usize, usize),
}

struct Classifier<'a> {
    p: &'a Program,
    defs: HashMap<Value, Def>,
    done: HashMap<Value, Term>,
    open: std::collections::HashSet<Value>,
}

impl Classifier<'_> {
    fn term(&mut self, v: Value) -> Term {
        if let Some(&t) = self.done.get(&v) {
            return t;
        }
        if !self.open.insert(v) {
            // Around a loop: what the loop carries.
            return Term::Loop;
        }
        let t = match self.defs.get(&v) {
            None => Term::Data,
            Some(&Def::Input(Reg::V(_))) => Term::Thread,
            Some(&Def::Input(_)) => Term::Uniform,
            Some(&Def::Phi(b, i)) => {
                let args = self.p.blocks[b].phis[i].args.clone();
                args.into_iter().map(|a| self.term(a)).max().unwrap_or(Term::Uniform)
            }
            Some(&Def::Inst(b, i)) => {
                let inst = &self.p.blocks[b].insts[i];
                let name = inst.name();
                let enc = inst.machine.encoding;
                let to_scalar = inst.writes.iter().any(|w| w.1 == v && !matches!(w.0, Reg::V(_)));
                if enc == Encoding::Smem {
                    Term::Uniform
                } else if loads_lanes(enc, name) {
                    Term::Data
                } else if name.starts_with("v_mbcnt") {
                    Term::Thread
                } else {
                    let reads: Vec<Value> = inst
                        .reads
                        .iter()
                        .filter(|r| !matches!(r.0, Reg::S(126) | Reg::S(127)))
                        .map(|r| r.1)
                        .collect();
                    let worst = reads.into_iter().map(|r| self.term(r)).max().unwrap_or(Term::Uniform);
                    if to_scalar && name.starts_with("v_readfirstlane") && worst == Term::Thread {
                        // One lane's value, the same in every lane.
                        Term::Uniform
                    } else {
                        worst
                    }
                }
            }
        };
        self.open.remove(&v);
        // A value inside a loop's cycle is settled once the loop is.
        if self.open.is_empty() || t != Term::Loop {
            self.done.insert(v, t);
        }
        t
    }
}

/// The [`Term`] of each buffer store's address in `p`.
pub fn store_terms(p: &Program) -> StoreTerms {
    let mut defs = HashMap::new();
    for &(reg, v) in &p.inputs {
        defs.insert(v, Def::Input(reg));
    }
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, phi) in block.phis.iter().enumerate() {
            defs.insert(phi.value, Def::Phi(b, i));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                defs.insert(v, Def::Inst(b, i));
            }
        }
    }
    let mut c = Classifier {
        p,
        defs,
        done: HashMap::new(),
        open: std::collections::HashSet::new(),
    };
    let mut out = StoreTerms::default();
    for block in &p.blocks {
        for inst in &block.insts {
            let name = inst.name();
            if !is_buffer_store(inst.machine.encoding, name) {
                continue;
            }
            // The registers that make the address (not the data stored).
            let mut regs: Vec<Reg> = Vec::new();
            let mut operand = |op: &Operand, low_only: bool| match *op {
                Operand::Vector { reg, dwords } => {
                    let n = if low_only { 1 } else { dwords };
                    regs.extend((0..n).map(|d| Reg::V(reg.wrapping_add(d))));
                }
                Operand::Scalar { code, dwords } if code < 106 => {
                    regs.extend((0..dwords.max(1)).map(|d| Reg::S(code.wrapping_add(d))));
                }
                _ => {}
            };
            match &inst.fields {
                Fields::Buffer { addr, soffset, offen, idxen, .. } => {
                    if *offen || *idxen {
                        operand(addr, false);
                    }
                    operand(soffset, false);
                }
                Fields::Flat { addr, saddr, .. } => {
                    if let Some(a) = addr {
                        // With a scalar base the VGPR is a 32-bit offset;
                        // else a 64-bit address whose high dword follows the
                        // low one's carry.
                        operand(a, saddr.is_some() || true);
                    }
                    if let Some(s) = saddr {
                        operand(s, false);
                    }
                }
                _ => {}
            }
            let worst = regs
                .into_iter()
                .filter_map(|r| inst.read(r))
                .map(|v| c.term(v))
                .max()
                .unwrap_or(Term::Uniform);
            match worst {
                Term::Uniform => out.uniform += 1,
                Term::Thread => out.thread += 1,
                Term::Loop => out.looped += 1,
                Term::Data => out.data += 1,
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(words: &[u32]) -> StoreTerms {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 16).expect("the test program builds");
        store_terms(&p)
    }

    /// `v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
    /// s_endpgm`: a store at the thread id times four.
    #[test]
    fn a_store_at_the_thread_id_is_bounded_by_the_dispatch() {
        let t = terms(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        assert_eq!((t.thread, t.total()), (1, 1), "{t:?}");
        assert!(t.bounded());
    }

    /// `buffer_load_dword v3, v0, s[8:11], 0 offen; buffer_store_dword v1,
    /// v3, s[8:11], 0 offen; s_endpgm`: a store at an address loaded from
    /// memory.
    #[test]
    fn a_store_at_a_loaded_address_is_not() {
        let t = terms(&[0xE030_1000, 0x8002_0300, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        assert_eq!((t.data, t.total()), (1, 1), "{t:?}");
        assert!(!t.bounded());
    }
}
