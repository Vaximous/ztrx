//! Store extents: the part of a buffer view a compute shader's stores can
//! reach, from its code (`victory_shader::stores`) and what the call is
//! recorded with (group size, dispatch size, root constants, constant
//! buffer values). A view of a whole heap (174 MB in the world) is then
//! declared written only where the shader can write it.
//!
//! A program gets extents only if every buffer store is followed to a
//! descriptor table entry and an offset expression; the leaves of those
//! expressions are thread ids, group ids, root constants and dwords of
//! constant buffers (through a table's descriptor or a root view).
//! Table entries the program never stores through are written nowhere.

use victory_shader::gfx10::{Fields, NULL, Operand};
use victory_shader::ir::Reg;
use victory_shader::native::{HwStage, NativeProgram, RootType, StageIo};
use victory_shader::resource::{Role, Source};
use victory_shader::stores::{Node, bounds, extents};

/// Whether table views are declared written only where the shader's
/// stores can reach (`VICTORY_EXTENTS=1`; off by default: the extents are
/// only counted).
pub(crate) fn narrow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_EXTENTS").is_ok_and(|v| v == "1"))
}

/// Shaders whose store extents are not used (their views are declared
/// whole): `VICTORY_EXTENT_DENY`, hashes (or their beginnings) separated by
/// commas; by default the ones a check run caught storing outside them.
/// ba4e20a2 (run fxchk2): a clear whose row stride is a constant the CPU
/// reads as 56 where the card's stores show 32,768.
pub(crate) fn denied(hash: &str) -> bool {
    static DENY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    DENY.get_or_init(|| {
        let list = std::env::var("VICTORY_EXTENT_DENY").unwrap_or_else(|_| "ba4e20a2fd348338".into());
        list.split(',').map(|h| h.trim().to_ascii_lowercase()).filter(|h| !h.is_empty()).collect()
    })
    .iter()
    .any(|d| hash.to_ascii_lowercase().starts_with(d.as_str()))
}

/// What the host reads to bound a leaf of an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Leaf {
    /// The thread id in the group along an axis: 0 to the group size less 1.
    Thread(u8),
    /// The group id along an axis: 0 to the dispatch size less 1.
    Group(u8),
    /// Dword `dword` of the root constants at `slot`.
    Constant { slot: u8, dword: u32 },
    /// The constant buffer a table entry holds (root slot, dword in the
    /// table): the dword at byte `offset`.
    Cb { slot: u8, entry: u32, offset: u32 },
    /// The constant buffer a root view at `slot` points at: the dword at
    /// byte `offset`.
    RootCb { slot: u8, offset: u32 },
}

/// One store through a table entry: the offset expression (a node), the
/// index expression with the bytes an index counts (None: the view's
/// element, for typed stores), and the bytes written.
type Store = (u32, Option<(u32, Option<u32>)>, u32);

/// A compute program's stores, by the table entry they write through.
#[derive(Debug)]
pub(crate) struct StoreExtents {
    nodes: Vec<Node>,
    /// Each expression leaf and how it is bounded.
    pub(crate) leaves: Vec<(Node, Leaf)>,
    /// (root slot, dword in the table) and the stores through it.
    entries: Vec<((u8, u32), Vec<Store>)>,
    /// Threads per group along each axis.
    pub(crate) threads: [u32; 3],
}

/// The table entry (root slot, dword in the table) a descriptor at `site`
/// is loaded from.
fn entry_of(
    res: &victory_shader::resource::Resources,
    n: &NativeProgram,
    site: victory_shader::resource::Site,
) -> Result<(u8, u32), &'static str> {
    let d = res
        .descriptors
        .iter()
        .find(|d| d.site == site && d.role == Role::Resource)
        .ok_or("a store without a traced descriptor")?;
    let Source::Loaded { load, first } = d.source else {
        return Err("a store through a descriptor not loaded from a table");
    };
    let i = res
        .scalar
        .binary_search_by_key(&load, |x| x.0)
        .map_err(|_| "a descriptor load without an address")?;
    let a = res.scalar[i].1;
    if a.dynamic.is_some() {
        return Err("a descriptor picked by index");
    }
    let q = n.layout.get(a.param).ok_or("a descriptor from no root parameter")?;
    if q.root != RootType::Table {
        return Err("a descriptor not from a table");
    }
    Ok((q.slot, a.offset / 4 + u32::from(first)))
}

impl StoreExtents {
    /// The extents of compute program `n`, or why it has none.
    pub(crate) fn of(n: &NativeProgram) -> Result<Self, &'static str> {
        let StageIo::Cs(cs) = &n.io else {
            return Err("not a compute program");
        };
        if n.stage != HwStage::Cs {
            return Err("not a compute program");
        }
        let p = victory_shader::ir::build(&n.code, n.wave64, n.vgprs).map_err(|_| "no program")?;
        let st = victory_shader::structure::structure(&p).map_err(|_| "no structure")?;
        let res = victory_shader::resource::track(&p, &n.layout, n.stage, Some(&st))
            .map_err(|_| "descriptors not traced")?;
        // Stores by address reach memory no view names.
        let by_address = p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            matches!(
                i.machine.encoding,
                victory_shader::gfx10::Encoding::Flat | victory_shader::gfx10::Encoding::Global
            ) && (i.name().contains("store") || i.name().contains("atomic"))
        });
        if by_address {
            return Err("a store by address");
        }
        let e = extents(&p);
        let mut entries: Vec<((u8, u32), Vec<Store>)> = Vec::new();
        for s in &e.stores {
            let offset = s.offset?;
            let index = s.index.transpose()?;
            let key = entry_of(&res, n, s.site)?;
            // An index counts what the translation multiplies it by: a
            // typed store the host view's elements, an untyped one the
            // used-resource entry's stride (`emit::lower`, raw bytes at
            // index times stride), never the view's own stride.
            let typed = p.blocks[s.site.0].insts[s.site.1].name().contains("format");
            let index = match index {
                Some(i) if typed => Some((i, None)),
                Some(i) => {
                    let stride = n
                        .used
                        .iter()
                        .find(|u| u.root == RootType::Table && u.slot == key.0 && u.table_offset_dwords == key.1)
                        .map(|u| u.stride)
                        .filter(|&s| s >= 4 && s.is_multiple_of(4))
                        .ok_or("an indexed untyped store without a stride")?;
                    Some((i, Some(stride)))
                }
                None => None,
            };
            let store = (offset, index, s.bytes);
            match entries.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(store),
                None => entries.push((key, vec![store])),
            }
        }
        let (groups, _) = cs.system_sgprs();
        let vgprs = cs.thread_id_vgprs();
        let param = |r: u8| {
            n.layout.iter().find_map(|q| {
                let first = n.stage.user_sgpr(q.user_data);
                let dword = u32::from(r).checked_sub(first.into())?;
                (dword < q.sgprs).then_some((q, dword))
            })
        };
        let mut leaves = Vec::new();
        for &node in &e.nodes {
            let leaf = match node {
                Node::Input(Reg::V(v)) if v < vgprs => Leaf::Thread(v),
                Node::Input(Reg::S(r)) if groups.contains(&Some(r)) => {
                    Leaf::Group(groups.iter().position(|g| *g == Some(r)).unwrap_or(0) as u8)
                }
                Node::Input(Reg::S(r)) => match param(r) {
                    Some((q, dword)) if q.root == RootType::Constants => Leaf::Constant { slot: q.slot, dword },
                    _ => continue,
                },
                Node::Loaded(site, dword) => {
                    let inst = &p.blocks[site.0].insts[site.1];
                    if inst.name().starts_with("s_buffer_load") {
                        let Some(at) = buffer_load_offset(inst, dword) else {
                            continue;
                        };
                        let Ok((slot, entry)) = entry_of(&res, n, site) else {
                            continue;
                        };
                        Leaf::Cb { slot, entry, offset: at }
                    } else {
                        let Ok(i) = res.scalar.binary_search_by_key(&site, |x| x.0) else {
                            continue;
                        };
                        let a = res.scalar[i].1;
                        match n.layout.get(a.param) {
                            Some(q) if q.root == RootType::Descriptor && a.dynamic.is_none() => Leaf::RootCb {
                                slot: q.slot,
                                offset: a.offset.wrapping_add(u32::from(dword) * 4),
                            },
                            _ => continue,
                        }
                    }
                }
                _ => continue,
            };
            leaves.push((node, leaf));
        }
        Ok(StoreExtents {
            nodes: e.nodes,
            leaves,
            entries,
            threads: cs.threads,
        })
    }

    /// The bytes `[lo, hi)` of the view at table entry `entry` the program
    /// can write, `element` the view's element size (for indexed stores),
    /// `ranges` each leaf's range (as [`Self::leaves`]). `Some((0, 0))`:
    /// nothing. None: anywhere.
    pub(crate) fn extent(
        &self,
        entry: (u8, u32),
        element: Option<u64>,
        ranges: &[Option<(u32, u32)>],
    ) -> Option<(u64, u64)> {
        let Some((_, stores)) = self.entries.iter().find(|(k, _)| *k == entry) else {
            return Some((0, 0));
        };
        let mut leaf = |n: Node| {
            let i = self.leaves.iter().position(|(m, _)| *m == n)?;
            *ranges.get(i)?
        };
        let mut out: Option<(u64, u64)> = None;
        for &(offset, index, bytes) in stores {
            let (lo, hi) = bounds(&self.nodes, offset, &mut leaf)?;
            let (lo, hi) = (u64::from(lo), u64::from(hi) + u64::from(bytes));
            let (lo, hi) = match index {
                Some((index, stride)) => {
                    let (a, b) = bounds(&self.nodes, index, &mut leaf)?;
                    let element = match stride {
                        Some(stride) => u64::from(stride),
                        None => element?,
                    };
                    (u64::from(a) * element + lo, u64::from(b) * element + hi)
                }
                None => (lo, hi),
            };
            out = Some(out.map_or((lo, hi), |(l, h)| (l.min(lo), h.max(hi))));
        }
        out
    }
}

/// `VICTORY_EXTENT_CHECK=1`: some dispatches whose views have extents get
/// their views' bytes hashed by 64 KiB piece on the card before and after
/// the dispatch; a piece that changed and lies wholly outside the extents
/// is a store the extents missed, logged with the shader. Views are still
/// declared whole unless `VICTORY_EXTENTS=1`.
pub(crate) fn check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_EXTENT_CHECK").is_ok_and(|v| v == "1"))
}

/// A view a dispatch writes through, with its extent, for the check.
pub(crate) struct CheckView {
    pub(crate) shadow: std::sync::Arc<crate::shadow::Shadow>,
    pub(crate) address: u64,
    pub(crate) len: u64,
    /// `[lo, hi)` from `address`.
    pub(crate) extent: (u64, u64),
    pub(crate) entry: (u8, u32),
    /// What the extent was made of, for the report: the view, the element
    /// size and the leaves' ranges.
    pub(crate) detail: String,
}

const CHECK_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint base;     // the first byte hashed, in the buffer (16-byte aligned)
    uint size;     // bytes hashed (a multiple of 16)
    uint pieces;
    uint compare;  // 0: store each piece's hash; else flag pieces whose hash is another
};
RWByteAddressBuffer data : register(u0);
// A 16-byte hash per piece, then a dword per piece: 1 if it changed.
RWByteAddressBuffer hashes : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint lo = id.x * 65536u;
    uint hi = min(lo + 65536u, size);
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ data.Load4(base + at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        hashes.Store(pieces * 16u + id.x * 4u, any(h != hashes.Load4(id.x * 16u)) ? 1u : 0u);
    } else {
        hashes.Store4(id.x * 16u, h);
    }
}
"#;

struct Checker {
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
}

fn checker() -> Option<&'static Checker> {
    use windows::Win32::Graphics::Direct3D12::*;
    static CHECKER: std::sync::OnceLock<Option<Checker>> = std::sync::OnceLock::new();
    CHECKER
        .get_or_init(|| {
            let made = (|| -> windows::core::Result<Checker> {
                let device = &crate::host::host().ok_or_else(windows::core::Error::empty)?.device;
                let bytes = crate::host::compile(CHECK_SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
                // SAFETY: the compiled bytes; the root signature is inside.
                unsafe {
                    let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes)?;
                    let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })?;
                    Ok(Checker { root, pso })
                }
            })();
            made.map_err(|e| tracing::warn!(error = %e, "no store extent check")).ok()
        })
        .as_ref()
}

/// A check recorded in a list, until its flags are back.
struct Recorded {
    hash: String,
    /// Title address of piece 0, and where the hashed bytes end.
    start: u64,
    end: u64,
    /// Absolute ranges the extents allow.
    allowed: Vec<(u64, u64)>,
    entries: Vec<(u8, u32)>,
    details: Vec<String>,
    flags: Box<[u32]>,
    /// Set to 1 once the list has run and the flags are read back.
    done: Box<std::sync::atomic::AtomicU32>,
    /// Frame-rate lines seen waiting.
    age: u32,
}

static RECORDED: std::sync::Mutex<Vec<Recorded>> = std::sync::Mutex::new(Vec::new());
/// Checks made, pieces changed inside the extents, pieces changed wholly
/// outside them, checks never back.
static CHECKS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Whether this dispatch of the shader `hash` is checked (its number
/// among them): its first four, then two in 64.
fn sampled(hash: &str) -> Option<u32> {
    static SEEN: std::sync::Mutex<Option<std::collections::HashMap<String, u32>>> = std::sync::Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let n = seen.get_or_insert_with(Default::default).entry(hash.to_owned()).or_insert(0);
    *n += 1;
    // Two in a row: one of each kind of view (`check_before`).
    (*n <= 4 || *n % 64 <= 1).then_some(*n)
}

/// A check about to be recorded around a dispatch.
pub(crate) struct Pending {
    hashes: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    buffer: u64,
    constants: [u32; 4],
    record: Recorded,
}

/// Records, before a dispatch of the shader `hash` through `views`, the
/// hash of one view's bytes (replacing the list's compute root signature
/// and pipeline: the caller binds its own again). None if not checked.
pub(crate) fn check_before(list: &crate::host::HostList, views: &[CheckView], hash: &str) -> Option<Pending> {
    use windows::Win32::Graphics::Direct3D12::*;
    let n = sampled(hash)?;
    // Alternately the largest view the extents say is written and the
    // largest of all (often one written nowhere).
    let written = views.iter().filter(|v| v.extent.1 > v.extent.0).max_by_key(|v| v.len);
    let view = match written {
        Some(w) if n % 2 == 1 => w,
        _ => views.iter().max_by_key(|v| v.len)?,
    };
    let shadow = &view.shadow;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        return None;
    }
    let c = checker()?;
    let host = crate::host::host()?;
    let start = view.address.next_multiple_of(16);
    let end = (view.address + view.len) & !15;
    if end <= start || shadow.offset(end) > u64::from(u32::MAX) {
        return None;
    }
    let size = (end - start) as u32;
    let pieces = size.div_ceil(64 << 10);
    let hashes = host.uav_buffer(u64::from(pieces) * 20)?;
    let allowed = views
        .iter()
        .filter(|v| std::sync::Arc::ptr_eq(&v.shadow, shadow))
        .map(|v| (v.address + v.extent.0, v.address + v.extent.1))
        .collect();
    let constants = [shadow.offset(start) as u32, size, pieces, 0];
    // SAFETY: live buffers.
    let buffer = unsafe { shadow.buffer.GetGPUVirtualAddress() };
    let l = &list.list;
    // SAFETY: an open list; our pipeline, buffers alive with the list.
    unsafe {
        l.ResourceBarrier(&[crate::host::transition(
            &hashes,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, buffer);
        l.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
        l.Dispatch(pieces.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    list.keep(&hashes);
    Some(Pending {
        hashes,
        buffer,
        constants,
        record: Recorded {
            hash: hash.to_owned(),
            start,
            end,
            allowed,
            entries: views.iter().map(|v| v.entry).collect(),
            details: views.iter().map(|v| format!("{:?} at {:#x}+{:#x}: {}", v.entry, v.address, v.len, v.detail)).collect(),
            flags: vec![u32::MAX; pieces as usize].into_boxed_slice(),
            done: Box::new(std::sync::atomic::AtomicU32::new(0)),
            age: 0,
        },
    })
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

/// After the dispatch: hashes again, flags the changed pieces and reads
/// the flags back (replacing the compute root signature and pipeline).
pub(crate) fn check_after(list: &crate::host::HostList, pending: Pending) {
    use windows::Win32::Graphics::Direct3D12::*;
    let (Some(c), Some(host)) = (checker(), crate::host::host()) else {
        return;
    };
    let Pending { hashes, buffer, mut constants, mut record } = pending;
    let pieces = constants[2];
    let Some(back) = host.read_back_buffer(u64::from(pieces) * 4) else {
        return;
    };
    constants[3] = 1;
    let l = &list.list;
    // SAFETY: as in `check_before`.
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, buffer);
        l.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
        l.Dispatch(pieces.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[crate::host::transition(
            &hashes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        l.CopyBufferRegion(&back, 0, &hashes, u64::from(pieces) * 16, u64::from(pieces) * 4);
    }
    let bytes = pieces as usize * 4;
    list.defer_read_back(back, record.flags.as_mut_ptr().cast(), [bytes, 1, bytes]);
    list.defer_write(record.done.as_ptr().cast(), crate::queue::Written::Value(1), 4);
    CHECKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    RECORDED.lock().unwrap_or_else(|e| e.into_inner()).push(record);
}

/// Since the last frame-rate line: constant buffer dwords extents were
/// bounded with, looked at again at submission, and of them those changed.
static RECHECKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// At submission: whether the constant buffer dwords (title address, value
/// when recorded) a call's extents were bounded with still hold those
/// values.
pub(crate) fn recheck(reads: &[(u64, u32)], hash: &str) {
    use std::sync::atomic::Ordering;
    RECHECKED[0].fetch_add(reads.len() as u64, Ordering::Relaxed);
    for &(at, was) in reads {
        let mut now = [0u8; 4];
        if crate::host::read_into(at as *const u8, &mut now) && u32::from_le_bytes(now) == was {
            continue;
        }
        RECHECKED[1].fetch_add(1, Ordering::Relaxed);
        static TOLD: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let mut told = TOLD.lock().unwrap_or_else(|e| e.into_inner());
        if !told.iter().any(|h| h == hash) {
            told.push(hash.to_owned());
            tracing::warn!(
                hash,
                at = format_args!("{at:#x}"),
                was,
                now = u32::from_le_bytes(now),
                "a constant the store extents were bounded with changed before submission"
            );
        }
    }
}

/// For the frame-rate line: the checks that are back.
pub(crate) fn check_line() {
    use std::sync::atomic::Ordering;
    if !check_on() && !narrow() {
        return;
    }
    let mut recorded = RECORDED.lock().unwrap_or_else(|e| e.into_inner());
    recorded.retain_mut(|r| {
        if r.done.load(Ordering::Acquire) == 0 {
            r.age += 1;
            if r.age > 30 {
                CHECKS[3].fetch_add(1, Ordering::Relaxed);
                // A list may still write them: never freed.
                std::mem::forget(std::mem::take(&mut r.flags));
                std::mem::forget(std::mem::replace(&mut r.done, Box::new(std::sync::atomic::AtomicU32::new(0))));
                return false;
            }
            return true;
        }
        for (i, &flag) in r.flags.iter().enumerate() {
            if flag != 1 {
                continue;
            }
            let lo = r.start + i as u64 * (64 << 10);
            let hi = (lo + (64 << 10)).min(r.end);
            if r.allowed.iter().any(|&(a, b)| a < hi && b > lo) {
                CHECKS[1].fetch_add(1, Ordering::Relaxed);
            } else {
                CHECKS[2].fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    hash = %r.hash,
                    entries = ?r.entries,
                    allowed = ?r.allowed.iter().map(|&(a, b)| (a - r.start, b - r.start)).collect::<Vec<_>>(),
                    piece = format_args!("{:#x}..{:#x}", lo - r.start, hi - r.start),
                    start = format_args!("{:#x}", r.start),
                    details = ?r.details,
                    "a store outside the store extents"
                );
            }
        }
        false
    });
    let c = CHECKS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    let r = RECHECKED.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    tracing::info!(
        "store extent checks: {} made, {} changed pieces inside the extents, {} outside, {} never back;          {} constant dwords looked at again at submission, {} changed; {} in memory the card wrote (not used as          bounds unless VICTORY_EXTENT_CARD_CONSTANTS=1)",
        c[0],
        c[1],
        c[2],
        c[3],
        r[0],
        r[1],
        CARD_CONSTANTS.swap(0, Ordering::Relaxed)
    );
}

/// The byte offset in its constant buffer of dword `dword` of what the
/// scalar buffer load `inst` loads: GFX10's immediate offset is in bytes,
/// as is a literal or null `soffset` (None: a register's).
fn buffer_load_offset(inst: &victory_shader::ir::Inst, dword: u8) -> Option<u32> {
    let Fields::Smem { offset, soffset, .. } = &inst.fields else {
        return None;
    };
    let extra = match *soffset {
        Operand::Scalar { code: NULL, .. } => 0,
        Operand::Literal(x) => x,
        _ => return None,
    };
    Some((*offset as u32).wrapping_add(extra).wrapping_add(u32::from(dword) * 4))
}

/// `VICTORY_EXTENT_DUMP=<shader hash>`: at the first binds of that compute
/// shader, the constant buffer dwords its extents read and the 64 dwords
/// around each (from 64 bytes before), as the CPU finds them then.
pub(crate) fn dump(hash: &str, reads: &[(u64, u32)], groups: Option<[u32; 3]>) {
    static WANTED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(6);
    let Some(wanted) = WANTED.get_or_init(|| std::env::var("VICTORY_EXTENT_DUMP").ok()) else {
        return;
    };
    if !hash.contains(wanted.as_str())
        || LEFT.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |n| n.checked_sub(1)).is_err()
    {
        return;
    }
    for &(address, value) in reads {
        let from = address.saturating_sub(64);
        let mut bytes = [0u8; 256];
        let read = crate::host::read_into(from as *const u8, &mut bytes);
        let dwords: Vec<String> = bytes
            .chunks_exact(4)
            .enumerate()
            .map(|(i, d)| format!("{:+#x}={:#x}", from as i64 + 4 * i as i64 - address as i64, u32::from_le_bytes([d[0], d[1], d[2], d[3]])))
            .collect();
        tracing::info!(
            hash,
            address = format_args!("{address:#x}"),
            value,
            ?groups,
            read,
            "a constant a store extent is bounded with, and around it: {}",
            dwords.join(" ")
        );
    }
}

/// The 64 KiB pieces of title memory that copy-backs wrote (what shaders
/// or the card's copies wrote, found differing), while extents are made.
static CARD_PIECES: std::sync::Mutex<Option<std::collections::HashSet<u64>>> = std::sync::Mutex::new(None);

/// Constant dwords not trusted as bounds since the card wrote their piece
/// (see [`card_wrote`]).
static CARD_CONSTANTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a constant buffer dword in memory the card has written is left
/// unbounded (`VICTORY_EXTENT_CARD_CONSTANTS=1` trusts it). What the CPU
/// reads there when a dispatch is bound need not be what the card reads
/// when it runs: run fxchk2's one shader whose stores went outside their
/// extents (ba4e20a2) multiplies the group's row by a dword the CPU read
/// as 56 that has to be 32,768 on the card for its stores to reach where
/// they did, and no title write changed it before submission.
fn distrust_card_constants() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXTENT_CARD_CONSTANTS").is_none_or(|v| v != "1"))
}

/// A copy-back wrote `[at, at + len)` of title memory.
pub(crate) fn card_wrote(at: usize, len: usize) {
    if len == 0 || !(narrow() || check_on()) {
        return;
    }
    let (first, last) = ((at >> 16) as u64, ((at + len - 1) >> 16) as u64);
    let mut pieces = CARD_PIECES.lock().unwrap_or_else(|e| e.into_inner());
    let pieces = pieces.get_or_insert_with(Default::default);
    pieces.extend(first..=last);
}

/// Whether a constant buffer dword at `address` may bound a store: not if
/// a copy-back ever wrote its piece.
/// (Counted also when they are trusted.)
pub(crate) fn constant_trusted(address: u64) -> bool {
    let card = CARD_PIECES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|p| p.contains(&(address >> 16)));
    if card {
        CARD_CONSTANTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    !card || !distrust_card_constants()
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::stores::Op;

    /// The code of the clear the second check run caught (ba4e20a2), from
    /// its first instruction to `s_endpgm`: the index is
    /// `cb[0x14] * s7 + s6 * 32 + v0`, `s_buffer_load_dword s5, s[12:15],
    /// 0x14` at word 24.
    const CLEAR: [u32; 36] = [
        0xD5803B3D, 0x03FF9D44, 0x6D0F7FDD, 0xD58056EE, 0x03FE8D6D, 0xE93D35FC, 0xBFA00003, 0xBE840305,
        0xBE850380, 0x7E020280, 0xF4080202, 0xFA000000, 0xF4080002, 0xFA000020, 0xF4080302, 0xFA000040,
        0x7E040280, 0x7E060280, 0x7E080280, 0x7E0A02C1, 0x7E0C02C1, 0x7E0E02C1, 0x7E1002C1, 0xBF8CC07F,
        0xF4200146, 0xFA000014, 0x8F048506, 0xBF8CC07F, 0x93050705, 0x81050405, 0x4A000005, 0xE01C2000,
        0x80020100, 0xE01C2000, 0x80000500, 0xBF810000,
    ];

    /// The constant buffer dwords the clear's index is bounded with.
    fn constant_offsets(code: &[u32]) -> Vec<u32> {
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = victory_shader::ir::build(&bytes, false, 9).expect("the clear builds");
        let e = extents(&p);
        assert_eq!(e.stores.len(), 2);
        e.nodes
            .iter()
            .filter_map(|n| match *n {
                Node::Loaded(site, dword) => {
                    let inst = &p.blocks[site.0].insts[site.1];
                    inst.name().starts_with("s_buffer_load").then(|| buffer_load_offset(inst, dword))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .expect("an immediate offset")
    }

    /// GFX10's scalar load offsets are bytes: the clear reads the dword at
    /// byte 0x14 (dword 5), not at byte 0x50; and a dword past the first of
    /// a wider load is four bytes on for each.
    #[test]
    fn constant_buffer_offsets_are_bytes() {
        assert_eq!(constant_offsets(&CLEAR), [0x14]);
        // As `s_buffer_load_dwordx4 s[4:7], s[12:15], 0x10`: s5, s6 and s7
        // (the index's terms) are its second to fourth dwords, at bytes
        // 0x14, 0x18 and 0x1c.
        let mut wide = CLEAR;
        (wide[24], wide[25]) = (0xF4280106, 0xFA000010);
        let mut offsets = constant_offsets(&wide);
        offsets.sort_unstable();
        assert_eq!(offsets, [0x14, 0x18, 0x1c]);
    }

    /// A constant in a piece a copy-back wrote does not bound stores.
    #[test]
    fn constants_the_card_wrote_are_not_trusted() {
        if !distrust_card_constants() {
            return;
        }
        let at = 0x7_1234_0000usize;
        CARD_PIECES.lock().unwrap().get_or_insert_with(Default::default).insert((at >> 16) as u64);
        assert!(!constant_trusted(at as u64 + 0x14));
        assert!(constant_trusted(at as u64 + 0x1_0000));
    }

    /// Stores at the thread id times four through the entry at dword 8, and
    /// at an index times the view's element plus 4 through dword 16.
    fn sample() -> StoreExtents {
        StoreExtents {
            nodes: vec![
                Node::Input(Reg::V(0)),
                Node::Const(2),
                Node::Op(Op::Shl, 0, 1),
                Node::Const(4),
            ],
            leaves: vec![(Node::Input(Reg::V(0)), Leaf::Thread(0))],
            entries: vec![
                ((0, 8), vec![(2, None, 4)]),
                ((0, 16), vec![(3, Some((0, None)), 8)]),
                ((0, 32), vec![(3, Some((0, Some(48))), 4)]),
            ],
            threads: [64, 1, 1],
        }
    }

    #[test]
    fn extents_bound_stores_by_the_group_and_leave_other_entries_unwritten() {
        let e = sample();
        let known = [Some((0, 63))];
        assert_eq!(e.extent((0, 8), None, &known), Some((0, 256)));
        // 63 elements of 32 bytes, then 4 + 8 bytes.
        assert_eq!(e.extent((0, 16), Some(32), &known), Some((4, 63 * 32 + 12)));
        assert_eq!(e.extent((0, 16), None, &known), None);
        assert_eq!(e.extent((0, 24), None, &known), Some((0, 0)));
        // An untyped indexed store counts the used-resource stride, not
        // the view's element.
        assert_eq!(e.extent((0, 32), Some(4), &known), Some((4, 63 * 48 + 8)));
        assert_eq!(e.extent((0, 8), None, &[None]), None);
    }

    #[test]
    fn the_check_compiles() {
        let made = crate::host::compile(CHECK_SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1"));
        assert!(made.is_ok_and(|b| !b.is_empty()));
    }
}
