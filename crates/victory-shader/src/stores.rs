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

use crate::gfx10::{Encoding, Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::Site;
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
    matches!(
        encoding,
        Encoding::Mubuf | Encoding::Mtbuf | Encoding::Flat | Encoding::Global
    ) && (name.contains("store") || name.contains("atomic"))
        && !name.contains("scratch")
}

/// Whether an instruction of `encoding` named `name` loads per-lane values
/// from memory (buffers, LDS, images).
fn loads_lanes(encoding: Encoding, name: &str) -> bool {
    match encoding {
        Encoding::Mubuf
        | Encoding::Mtbuf
        | Encoding::Flat
        | Encoding::Global
        | Encoding::Scratch
        | Encoding::Mimg => true,
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
                args.into_iter()
                    .map(|a| self.term(a))
                    .max()
                    .unwrap_or(Term::Uniform)
            }
            Some(&Def::Inst(b, i)) => {
                let inst = &self.p.blocks[b].insts[i];
                let name = inst.name();
                let enc = inst.machine.encoding;
                let to_scalar = inst
                    .writes
                    .iter()
                    .any(|w| w.1 == v && !matches!(w.0, Reg::V(_)));
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
                    let worst = reads
                        .into_iter()
                        .map(|r| self.term(r))
                        .max()
                        .unwrap_or(Term::Uniform);
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
    let mut c = Classifier {
        p,
        defs: defs(p),
        done: HashMap::new(),
        open: std::collections::HashSet::new(),
    };
    store_terms_with(p, &mut c)
}

/// Where each value of `p` is defined.
fn defs(p: &Program) -> HashMap<Value, Def> {
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
    defs
}

fn store_terms_with(p: &Program, c: &mut Classifier<'_>) -> StoreTerms {
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
                Fields::Buffer {
                    addr,
                    soffset,
                    offen,
                    idxen,
                    ..
                } => {
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

/// A node of an address expression ([`Extents`]): what a buffer store's
/// offset is made of, in terms the host can bound when it records the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    Const(u32),
    /// Any value from `.0` to `.1`: the lane's place in its wave.
    Range(u32, u32),
    /// A register as the program starts: user data (root constants), group
    /// ids, thread ids.
    Input(Reg),
    /// Dword `.1` of what the scalar load at `.0` reads.
    Loaded(Site, u8),
    /// Of two nodes (indices into [`Extents::nodes`]).
    Op(Op, u32, u32),
}

/// Operations on 32-bit integers, as the hardware does them (modulo 2^32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Shl,
    Shr,
    And,
    Or,
    Min,
    Max,
    /// One or the other (a select, a join of paths).
    Either,
}

/// A buffer store (or atomic) and its address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Store {
    pub site: Site,
    /// Byte offset in the buffer past the index part (`offen`, `soffset`
    /// and the instruction's offset): a node, or why the bounds lose it.
    pub offset: Result<u32, &'static str>,
    /// With `idxen`: the element index, times the view's stride.
    pub index: Option<Result<u32, &'static str>>,
    /// Bytes the access covers from its offset.
    pub bytes: u32,
}

/// Every buffer store of a program with its address as an expression.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extents {
    pub nodes: Vec<Node>,
    pub stores: Vec<Store>,
}

/// Bytes a buffer store or atomic named `name` writes from its address
/// (formatted stores: four bytes a channel, the widest format).
fn store_bytes(name: &str) -> u32 {
    let has = |s: &str| name.contains(s);
    if has("format") {
        if has("xyzw") {
            16
        } else if has("xyz") {
            12
        } else if has("xy") {
            8
        } else {
            4
        }
    } else if has("dwordx4") {
        16
    } else if has("dwordx3") {
        12
    } else if has("dwordx2") || (has("atomic") && has("x2")) {
        8
    } else if has("byte") {
        1
    } else if has("short") {
        2
    } else {
        4
    }
}

struct Builder<'a> {
    p: &'a Program,
    defs: HashMap<Value, Def>,
    memo: HashMap<Value, Result<u32, &'static str>>,
    open: std::collections::HashSet<Value>,
    nodes: Vec<Node>,
    index: HashMap<Node, u32>,
}

const NO_OPERAND: &str = "an operand the bounds do not follow";

impl<'a> Builder<'a> {
    fn node(&mut self, n: Node) -> u32 {
        if let Some(&i) = self.index.get(&n) {
            return i;
        }
        let i = self.nodes.len() as u32;
        self.nodes.push(n);
        self.index.insert(n, i);
        i
    }

    fn op(&mut self, op: Op, a: u32, b: u32) -> u32 {
        if op == Op::Either && a == b {
            return a;
        }
        self.node(Node::Op(op, a, b))
    }

    fn konst(&mut self, c: u32) -> u32 {
        self.node(Node::Const(c))
    }

    fn value(&mut self, v: Value) -> Result<u32, &'static str> {
        if let Some(&r) = self.memo.get(&v) {
            return r;
        }
        if !self.open.insert(v) {
            return Err("a value carried around a loop");
        }
        let r = match self.defs.get(&v) {
            None => Err("a value without a definition"),
            Some(&Def::Input(reg)) => Ok(self.node(Node::Input(reg))),
            Some(&Def::Phi(b, i)) => {
                let p = self.p;
                let mut joined: Option<u32> = None;
                let mut r = Ok(0);
                for &a in &p.blocks[b].phis[i].args {
                    match self.value(a) {
                        Ok(n) => joined = Some(joined.map_or(n, |j| self.op(Op::Either, j, n))),
                        Err(e) => {
                            r = Err(e);
                            break;
                        }
                    }
                }
                r.and(joined.ok_or("a phi without arguments"))
            }
            Some(&Def::Inst(b, i)) => self.inst(b, i, v),
        };
        self.open.remove(&v);
        self.memo.insert(v, r);
        r
    }

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<u32, &'static str> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.konst(0)),
            Operand::Scalar { code, .. } => self.value(inst.read(Reg::S(code)).ok_or(NO_OPERAND)?),
            Operand::Vector { reg, .. } => self.value(inst.read(Reg::V(reg)).ok_or(NO_OPERAND)?),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|c| self.konst(c))
                .ok_or(NO_OPERAND),
            Operand::Literal(x) => Ok(self.konst(x)),
            _ => Err(NO_OPERAND),
        }
    }

    /// A constant operand's value.
    fn constant(&mut self, inst: &Inst, op: &Operand) -> Result<u32, &'static str> {
        let n = self.operand(inst, op)?;
        match self.nodes[n as usize] {
            Node::Const(c) => Ok(c),
            _ => Err("a bit field at offsets the code computes"),
        }
    }

    /// `(x >> offset) & (2^width - 1)`.
    fn bits(&mut self, x: u32, offset: u32, width: u32) -> u32 {
        let o = self.konst(offset & 31);
        let shifted = self.op(Op::Shr, x, o);
        let width = width & 0x7F;
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let m = self.konst(mask);
        self.op(Op::And, shifted, m)
    }

    fn inst(&mut self, b: usize, i: usize, v: Value) -> Result<u32, &'static str> {
        let inst: &'a Inst = &self.p.blocks[b].insts[i];
        let name = inst.name();
        let enc = inst.machine.encoding;
        if enc == Encoding::Smem {
            if !name.starts_with("s_load_dword") && !name.starts_with("s_buffer_load_dword") {
                return Err("a scalar memory result other than a load");
            }
            let dword = inst
                .writes
                .iter()
                .position(|w| w.1 == v)
                .ok_or(NO_OPERAND)?;
            return Ok(self.node(Node::Loaded((b, i), dword as u8)));
        }
        match &inst.fields {
            Fields::Sop { src, imm, dst } => {
                let mut src_n =
                    |s: &mut Self, k: usize| s.operand(inst, src.get(k).ok_or(NO_OPERAND)?);
                let sext = |imm: Option<u16>| imm.map(|i| i as i16 as i32 as u32).ok_or(NO_OPERAND);
                let two = |s: &mut Self, op: Op, src_n: &mut dyn FnMut(&mut Self, usize) -> Result<u32, &'static str>| {
                    let a = src_n(s, 0)?;
                    let b = src_n(s, 1)?;
                    Ok(s.op(op, a, b))
                };
                match name {
                    "s_mov_b32" => src_n(self, 0),
                    "s_movk_i32" => Ok(self.konst(sext(*imm)?)),
                    "s_add_u32" | "s_add_i32" => two(self, Op::Add, &mut src_n),
                    "s_sub_u32" | "s_sub_i32" => two(self, Op::Sub, &mut src_n),
                    "s_mul_i32" => two(self, Op::Mul, &mut src_n),
                    "s_lshl_b32" => two(self, Op::Shl, &mut src_n),
                    "s_lshr_b32" => two(self, Op::Shr, &mut src_n),
                    "s_and_b32" => two(self, Op::And, &mut src_n),
                    "s_or_b32" => two(self, Op::Or, &mut src_n),
                    "s_min_u32" => two(self, Op::Min, &mut src_n),
                    "s_max_u32" => two(self, Op::Max, &mut src_n),
                    "s_cselect_b32" => two(self, Op::Either, &mut src_n),
                    "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32"
                    | "s_lshl4_add_u32" => {
                        let shift = u32::from(name.as_bytes()[6] - b'0');
                        let a = src_n(self, 0)?;
                        let k = self.konst(shift);
                        let shifted = self.op(Op::Shl, a, k);
                        let b = src_n(self, 1)?;
                        Ok(self.op(Op::Add, shifted, b))
                    }
                    "s_bfe_u32" => {
                        let field = self.constant(inst, src.get(1).ok_or(NO_OPERAND)?)?;
                        let a = src_n(self, 0)?;
                        Ok(self.bits(a, field & 31, field >> 16))
                    }
                    "s_addk_i32" | "s_mulk_i32" => {
                        let Some(Operand::Scalar { code, .. }) = dst else {
                            return Err(NO_OPERAND);
                        };
                        let old = self.value(inst.read(Reg::S(*code)).ok_or(NO_OPERAND)?)?;
                        let k = self.konst(sext(*imm)?);
                        let op = if name == "s_addk_i32" {
                            Op::Add
                        } else {
                            Op::Mul
                        };
                        Ok(self.op(op, old, k))
                    }
                    _ => Err("a scalar instruction the bounds do not follow"),
                }
            }
            Fields::Valu(f) => {
                if f.neg != 0
                    || f.abs != 0
                    || f.opsel != 0
                    || f.clamp
                    || f.omod != 0
                    || f.sdwa.is_some()
                    || f.dpp.is_some()
                {
                    return Err("a vector instruction with modifiers");
                }
                let src = |s: &mut Self, k: usize| s.operand(inst, f.src.get(k).ok_or(NO_OPERAND)?);
                let two = |s: &mut Self, op: Op, a: usize, b: usize| {
                    let x = src(s, a)?;
                    let y = src(s, b)?;
                    Ok(s.op(op, x, y))
                };
                let three = |s: &mut Self, inner: Op, outer: Op| {
                    let x = two(s, inner, 0, 1)?;
                    let z = src(s, 2)?;
                    Ok(s.op(outer, x, z))
                };
                let low = |s: &mut Self, k: usize, bits: u32| {
                    let x = src(s, k)?;
                    let m = s.konst((1u32 << bits) - 1);
                    Ok::<u32, &'static str>(s.op(Op::And, x, m))
                };
                match name {
                    "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32" => src(self, 0),
                    "v_add_nc_u32" | "v_add_co_u32" => two(self, Op::Add, 0, 1),
                    "v_sub_nc_u32" | "v_sub_co_u32" => two(self, Op::Sub, 0, 1),
                    "v_subrev_nc_u32" | "v_subrev_co_u32" => two(self, Op::Sub, 1, 0),
                    "v_add3_u32" => three(self, Op::Add, Op::Add),
                    "v_lshlrev_b32" => two(self, Op::Shl, 1, 0),
                    "v_lshrrev_b32" => two(self, Op::Shr, 1, 0),
                    "v_lshl_add_u32" => three(self, Op::Shl, Op::Add),
                    "v_add_lshl_u32" => three(self, Op::Add, Op::Shl),
                    "v_lshl_or_b32" => three(self, Op::Shl, Op::Or),
                    "v_and_or_b32" => three(self, Op::And, Op::Or),
                    "v_or3_b32" => three(self, Op::Or, Op::Or),
                    "v_and_b32" => two(self, Op::And, 0, 1),
                    "v_or_b32" => two(self, Op::Or, 0, 1),
                    "v_mul_lo_u32" => two(self, Op::Mul, 0, 1),
                    "v_min_u32" => two(self, Op::Min, 0, 1),
                    "v_max_u32" => two(self, Op::Max, 0, 1),
                    "v_min3_u32" => three(self, Op::Min, Op::Min),
                    "v_max3_u32" => three(self, Op::Max, Op::Max),
                    "v_cndmask_b32" => two(self, Op::Either, 0, 1),
                    "v_mul_u32_u24" | "v_mad_u32_u24" | "v_mad_u32_u16" => {
                        let bits = if name.ends_with("u16") { 16 } else { 24 };
                        let a = low(self, 0, bits)?;
                        let b = low(self, 1, bits)?;
                        let m = self.op(Op::Mul, a, b);
                        if name.starts_with("v_mad") {
                            let c = src(self, 2)?;
                            Ok(self.op(Op::Add, m, c))
                        } else {
                            Ok(m)
                        }
                    }
                    "v_bfe_u32" => {
                        let offset = self.constant(inst, f.src.get(1).ok_or(NO_OPERAND)?)?;
                        let width = self.constant(inst, f.src.get(2).ok_or(NO_OPERAND)?)?;
                        let a = src(self, 0)?;
                        Ok(self.bits(a, offset, width & 31))
                    }
                    "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                        let most = if name.contains("_lo_") { 31 } else { 32 };
                        let lanes = self.node(Node::Range(0, most));
                        let c = src(self, 1)?;
                        Ok(self.op(Op::Add, lanes, c))
                    }
                    _ => Err("a vector instruction the bounds do not follow"),
                }
            }
            _ if loads_lanes(enc, name) => Err("an address loaded by the lanes"),
            _ => Err("an instruction the bounds do not follow"),
        }
    }
}

/// Every buffer store (and atomic) of `p` with its address as an expression
/// over what the program starts with, the scalar loads and the lane's
/// place.
pub fn extents(p: &Program) -> Extents {
    let mut b = Builder {
        p,
        defs: defs(p),
        memo: HashMap::new(),
        open: std::collections::HashSet::new(),
        nodes: Vec::new(),
        index: HashMap::new(),
    };
    let mut stores = Vec::new();
    for (bi, block) in p.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            if !matches!(inst.machine.encoding, Encoding::Mubuf | Encoding::Mtbuf)
                || !is_buffer_store(inst.machine.encoding, name)
            {
                continue;
            }
            let Fields::Buffer {
                addr,
                soffset,
                offset,
                offen,
                idxen,
                ..
            } = &inst.fields
            else {
                continue;
            };
            let Operand::Vector { reg, .. } = *addr else {
                continue;
            };
            let vgpr = |b: &mut Builder, r: u8| b.value(inst.read(Reg::V(r)).ok_or(NO_OPERAND)?);
            let index = idxen.then(|| vgpr(&mut b, reg));
            let offset = (|| {
                let v = if *offen {
                    vgpr(&mut b, reg + u8::from(*idxen))?
                } else {
                    b.konst(0)
                };
                let s = b.operand(inst, soffset)?;
                let sum = b.op(Op::Add, v, s);
                let k = b.konst(u32::from(*offset));
                Ok(b.op(Op::Add, sum, k))
            })();
            stores.push(Store {
                site: (bi, ii),
                offset,
                index,
                bytes: store_bytes(name),
            });
        }
    }
    Extents {
        nodes: b.nodes,
        stores,
    }
}

/// The values an expression of `nodes` from `root` can take, as an
/// inclusive range that does not wrap; `leaf` gives the range of each
/// [`Node::Input`] and [`Node::Loaded`] (None: unknown). None if unbounded.
pub fn bounds(
    nodes: &[Node],
    root: u32,
    leaf: &mut dyn FnMut(Node) -> Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    let mut memo: Vec<Option<Option<(i128, i128)>>> = vec![None; nodes.len()];
    let r = range(nodes, root, leaf, &mut memo)?;
    let (lo, hi) = normal(r)?;
    Some((lo as u32, hi as u32))
}

const WRAP: i128 = 1 << 32;

/// `r` as values modulo 2^32, if they do not straddle the wrap.
fn normal((lo, hi): (i128, i128)) -> Option<(i128, i128)> {
    if hi - lo >= WRAP {
        return None;
    }
    let k = lo.div_euclid(WRAP) * WRAP;
    let (lo, hi) = (lo - k, hi - k);
    (hi < WRAP).then_some((lo, hi))
}

fn range(
    nodes: &[Node],
    n: u32,
    leaf: &mut dyn FnMut(Node) -> Option<(u32, u32)>,
    memo: &mut Vec<Option<Option<(i128, i128)>>>,
) -> Option<(i128, i128)> {
    if let Some(r) = memo[n as usize] {
        return r;
    }
    let node = nodes[n as usize];
    let r = (|| {
        Some(match node {
            Node::Const(c) => (i128::from(c), i128::from(c)),
            Node::Range(lo, hi) => (i128::from(lo), i128::from(hi)),
            Node::Input(_) | Node::Loaded(..) => {
                let (lo, hi) = leaf(node)?;
                (i128::from(lo), i128::from(hi))
            }
            Node::Op(op, a, b) => {
                let a = range(nodes, a, leaf, memo)?;
                let b = range(nodes, b, leaf, memo)?;
                match op {
                    Op::Add => (a.0 + b.0, a.1 + b.1),
                    Op::Sub => (a.0 - b.1, a.1 - b.0),
                    _ => {
                        let (a, b) = (normal(a)?, normal(b)?);
                        match op {
                            Op::Mul => (a.0 * b.0, a.1 * b.1),
                            Op::Shl if b.1 < 32 => (a.0 << b.0, a.1 << b.1),
                            Op::Shr if b.1 < 32 => (a.0 >> b.1, a.1 >> b.0),
                            Op::And if a.0 == a.1 && b.0 == b.1 => (a.0 & b.0, a.0 & b.0),
                            Op::And => (0, a.1.min(b.1)),
                            Op::Or if a.0 == a.1 && b.0 == b.1 => (a.0 | b.0, a.0 | b.0),
                            Op::Or => {
                                let top = a.1.max(b.1);
                                let mask = if top == 0 {
                                    0
                                } else {
                                    (1i128 << (128 - top.leading_zeros())) - 1
                                };
                                (a.0.max(b.0), mask)
                            }
                            Op::Min => (a.0.min(b.0), a.1.min(b.1)),
                            Op::Max => (a.0.max(b.0), a.1.max(b.1)),
                            Op::Either => (a.0.min(b.0), a.1.max(b.1)),
                            _ => return None,
                        }
                    }
                }
            }
        })
    })();
    // Keep ranges small enough that products stay exact.
    let r = r.filter(|r| r.0.abs() < 1 << 80 && r.1.abs() < 1 << 80);
    memo[n as usize] = Some(r);
    r
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
        let t = terms(&[
            0xE030_1000,
            0x8002_0300,
            0xE070_1000,
            0x8002_0103,
            0xBF81_0000,
        ]);
        assert_eq!((t.data, t.total()), (1, 1), "{t:?}");
        assert!(!t.bounded());
    }

    /// `v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
    /// s_endpgm`: the store's offset is the thread id times four, which
    /// 64 threads bound to 0..=252 (and four bytes past that).
    #[test]
    fn a_store_at_the_thread_id_has_an_extent() {
        let code: Vec<u8> = [0x3406_0082u32, 0xE070_1000, 0x8002_0103, 0xBF81_0000]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 16).expect("the test program builds");
        let e = extents(&p);
        assert_eq!(e.stores.len(), 1);
        let s = &e.stores[0];
        assert_eq!((s.index, s.bytes), (None, 4));
        let root = s.offset.expect("an offset");
        let mut leaf = |n: Node| match n {
            Node::Input(Reg::V(0)) => Some((0, 63)),
            _ => None,
        };
        assert_eq!(bounds(&e.nodes, root, &mut leaf), Some((0, 252)));
        assert_eq!(bounds(&e.nodes, root, &mut |_| None), None);
    }

    #[test]
    fn bounds_follow_the_wrap() {
        // x + 0xFFFF_FFF0 with x in 16..=32: 0..=16.
        let nodes = [
            Node::Range(16, 32),
            Node::Const(0xFFFF_FFF0),
            Node::Op(Op::Add, 0, 1),
            Node::Range(0, 32),
            Node::Op(Op::Add, 3, 1),
        ];
        assert_eq!(bounds(&nodes, 2, &mut |_| None), Some((0, 16)));
        // Straddling the wrap: unbounded.
        assert_eq!(bounds(&nodes, 4, &mut |_| None), None);
    }

    /// `VICTORY_STORE_BINS=<dir> cargo test -p victory-shader -- --ignored
    /// survey --nocapture`: what the store extents cover over the shader
    /// containers in a directory.
    #[test]
    #[ignore = "reads shaders from VICTORY_STORE_BINS"]
    fn survey() {
        use crate::native::{HwStage, RootType, StageIo};
        let Some(dir) = std::env::var_os("VICTORY_STORE_BINS") else {
            return;
        };
        let mut kinds: std::collections::BTreeMap<String, u32> = Default::default();
        let mut why: std::collections::BTreeMap<&str, u32> = Default::default();
        let mut programs = [0u32; 3];
        for entry in std::fs::read_dir(dir).expect("a directory") {
            let path = entry.expect("an entry").path();
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(c) = crate::Container::parse(&bytes) else {
                continue;
            };
            let Some(part) = c.part(crate::FourCC::DXIL) else {
                continue;
            };
            let Ok(natives) = crate::native_programs(part.data) else {
                continue;
            };
            for n in natives {
                let Ok(p) = crate::ir::build(&n.code, n.wave64, n.vgprs) else {
                    continue;
                };
                let e = extents(&p);
                if e.stores.is_empty() {
                    continue;
                }
                programs[0] += 1;
                let res = crate::structure::structure(&p)
                    .ok()
                    .and_then(|st| crate::resource::track(&p, &n.layout, n.stage, Some(&st)).ok());
                let (groups, threads) = match &n.io {
                    StageIo::Cs(cs) => (cs.system_sgprs().0, cs.thread_id_vgprs()),
                    _ => ([None; 3], 0),
                };
                let param = |r: u8| {
                    n.layout.iter().find(|q| {
                        let first = n.stage.user_sgpr(q.user_data);
                        r >= first && u32::from(r) < u32::from(first) + q.sgprs
                    })
                };
                let mut all = true;
                for s in &e.stores {
                    let roots: Vec<_> = std::iter::once(s.offset).chain(s.index).collect();
                    if let Some(Err(w)) = roots.iter().find(|r| r.is_err()) {
                        *why.entry(w).or_default() += 1;
                        all = false;
                        continue;
                    }
                    let mut set = std::collections::BTreeSet::new();
                    let mut stack: Vec<u32> = roots.iter().map(|r| r.unwrap()).collect();
                    let mut seen = std::collections::HashSet::new();
                    while let Some(i) = stack.pop() {
                        if !seen.insert(i) {
                            continue;
                        }
                        match e.nodes[i as usize] {
                            Node::Op(_, a, b) => stack.extend([a, b]),
                            Node::Input(Reg::V(v)) if v < threads => {
                                set.insert("thread id");
                            }
                            Node::Input(Reg::S(r)) if groups.contains(&Some(r)) => {
                                set.insert("group id");
                            }
                            Node::Input(Reg::S(r)) => {
                                set.insert(match param(r).map(|q| q.root) {
                                    Some(RootType::Constants) => "root constant",
                                    Some(_) => "other root parameter",
                                    None => "other scalar input",
                                });
                            }
                            Node::Input(_) => {
                                set.insert("other vector input");
                            }
                            Node::Loaded(site, _) => {
                                let name = p.blocks[site.0].insts[site.1].name();
                                let from = res.as_ref().and_then(|r| {
                                    let i = r.scalar.binary_search_by_key(&site, |x| x.0).ok()?;
                                    n.layout.get(r.scalar[i].1.param).map(|q| q.root)
                                });
                                set.insert(if name.starts_with("s_buffer_load") {
                                    "constant buffer by descriptor"
                                } else {
                                    match from {
                                        Some(RootType::Descriptor) => {
                                            "loaded from a root descriptor"
                                        }
                                        Some(RootType::Table) => "loaded from a table",
                                        Some(_) => "loaded from another root parameter",
                                        None => "loaded from elsewhere",
                                    }
                                });
                            }
                            Node::Const(_) | Node::Range(..) => {}
                        }
                    }
                    let key = format!(
                        "{} {}",
                        if n.stage == HwStage::Cs { "cs" } else { "gfx" },
                        set.into_iter().collect::<Vec<_>>().join(" + ")
                    );
                    *kinds.entry(key).or_default() += 1;
                }
                programs[if all { 1 } else { 2 }] += 1;
            }
        }
        println!(
            "programs with buffer stores {}: all stores followed {}, some not {}",
            programs[0], programs[1], programs[2]
        );
        for (k, n) in &kinds {
            println!("{n:6} {k}");
        }
        for (w, n) in &why {
            println!("{n:6} not followed: {w}");
        }
    }
}
