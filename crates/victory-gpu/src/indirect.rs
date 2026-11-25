//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

/// The most groups along each axis a clamped indirect dispatch record runs
/// (the clamp's `min(asked, 65535)` per axis, kept in step by a test): what
/// store extents bound the group ids of such dispatches with.
pub(crate) const MOST_GROUPS: [u32; 3] = [MAX_GROUPS_PER_AXIS; 3];

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: each record of a group is
    // charged to the budget on its own (`VICTORY_CLAMP_GROUP_CHARGE=0`).
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// One record: what it asks for, bounded, and what it costs.
struct Rec {
    bool live;
    uint src_at;
    uint dst_at;
    uint cost;
    uint2 draw_counts;
    uint3 groups;
};
Rec measure(uint i) {
    Rec r;
    r.live = false;
    r.src_at = 0;
    r.dst_at = 0;
    r.cost = 0;
    r.draw_counts = uint2(0, 0);
    r.groups = uint3(0, 0, 0);
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (i >= eff_count) return r;
    r.live = true;
    uint src_at = args_offset + i * stride;
    r.src_at = src_at;
    r.dst_at = i * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        // Ahead of its dispatch, a record over the most a dispatch runs
        // is a slot not filled: empty, and not charged.
        if (ahead && any(asked > uint3(65535u, 65535u, 65535u))) {
            asked = uint3(0, 0, 0);
        }
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    r.cost = cost;
    r.draw_counts = draw_counts;
    r.groups = groups;
    return r;
}
// Adds `cost` to the list's work so far, saturating at the budget: the
// work before it.
uint charge(uint cost) {
    uint before;
    budget.InterlockedOr(0, 0, before);
    [allow_uav_condition]
    while (before < BUDGET) {
        uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
        uint prev;
        budget.InterlockedCompareExchange(0, before, next, prev);
        if (prev == before) break;
        before = prev;
    }
    return before;
}
// Whether a record costing `cost` with `before` of work ahead of it is past
// the budget.
bool past(uint before, uint cost) {
    return cost > 0 && (before >= BUDGET || BUDGET - before < cost);
}
void store(Rec r, bool over) {
    if (kind <= 1) {
        dst_args.Store2(r.dst_at, over ? uint2(0, 0) : r.draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(r.dst_at + off, src_args.Load(r.src_at + off));
        }
    } else {
        dst_args.Store3(r.dst_at, over ? uint3(0, 0, 0) : r.groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(r.dst_at + off, src_args.Load(r.src_at + off));
        }
    }
}
// The group's work, saturating at the budget, and that of the list before
// the group.
groupshared uint group_cost;
groupshared uint group_before;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint at : SV_GroupIndex) {
    // The records of a group are charged to the budget once, by one
    // thread: each charging its own, the 64 of a window waited on one
    // another's exchanges with the card's memory, 68 microseconds a window
    // (run fxg4). Each record's place in the group's sum stands for the
    // work before it, so a record is past the budget as it was charged one
    // by one.
    if (at == 0) {
        group_cost = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    Rec r = measure(id.x);
    bool alone = (has_count & 4u) != 0;
    uint ahead = 0;
    if (r.cost > 0 && !alone) {
        uint seen = group_cost;
        [allow_uav_condition]
        while (true) {
            uint next = (seen >= BUDGET || BUDGET - seen <= r.cost) ? BUDGET : (seen + r.cost);
            uint prev;
            InterlockedCompareExchange(group_cost, seen, next, prev);
            if (prev == seen) break;
            seen = prev;
        }
        ahead = seen;
    }
    GroupMemoryBarrierWithGroupSync();
    if (at == 0) {
        group_before = group_cost > 0 ? charge(group_cost) : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!r.live) return;
    if (alone) {
        store(r, past(r.cost > 0 ? charge(r.cost) : 0, r.cost));
        return;
    }
    uint before = group_before >= BUDGET || BUDGET - group_before <= ahead ? BUDGET : group_before + ahead;
    store(r, past(before, r.cost));
}
// The same as a draw of one point a record ([`clamp_draws`]): the point
// lies outside the view, so nothing reaches the targets. Each record is
// charged on its own (no group memory in a vertex program): only for
// calls of one record.
[RootSignature(RS)]
float4 vertex(uint i : SV_VertexID) : SV_Position {
    Rec r = measure(i);
    if (r.live) {
        store(r, past(r.cost > 0 ? charge(r.cost) : 0, r.cost));
    }
    return float4(2.0, 2.0, 0.5, 1.0);
}
"#;

/// Whether the records clamped by one group of the compute pass are
/// charged to the list's budget once, by one thread, each at its place in
/// the group's sum (on by default; `VICTORY_CLAMP_GROUP_CHARGE=0` charges
/// each on its own). Each charge is a loop of exchanges with the card's
/// memory until no other thread changed the word between: the 64 records of
/// a window took turns, and a window's clamp cost 68 microseconds against
/// under 5 for a record alone, 24 ms of the card's time a frame in the
/// world (run fxg4, 355 windows a frame).
fn group_charge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_GROUP_CHARGE").is_none_or(|v| v != "0"))
}

/// The work an indirect list's records may ask in all ([`SOURCE`]'s
/// `BUDGET`): records past it run empty.
pub(crate) const BUDGET: u32 = 1 << 27;

/// The most groups per axis of any dispatch or mesh record the clamp
/// passes on ([`SOURCE`]: `min(asked, 65535u)`; mesh records also at most
/// 1 in y and z). Others may rely on it as an upper bound of the group ids
/// of clamped indirect dispatches; kept in step with the shader by a test.
pub(crate) const MAX_GROUPS_PER_AXIS: u32 = 65535;

/// The most groups of one clamped indirect dispatch in all (x * y * z;
/// [`SOURCE`]'s `GROUPS`); a record asking for more runs empty.
#[allow(dead_code, reason = "a bound for store extents of indirect dispatches")]
pub(crate) const MAX_DISPATCH_GROUPS: u32 = 1 << 20;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The clamp as a draw of points ([`clamp_draws`]): no targets, no
    /// pixel program. `None` if it did not compile.
    draw: Option<ID3D12PipelineState>,
    /// Many clamps in one pass ([`merged_source`]).
    merged: Option<(ID3D12RootSignature, ID3D12PipelineState)>,
}

/// Whether the clamp before an indirect draw (not a dispatch) is a draw of
/// one point a record, outside the view, whose vertex program clamps the
/// record (`VICTORY_CLAMP_DRAW=1`; off by default). The compute pass before
/// every indirect draw switched the card from drawing to compute and back:
/// about 10 microseconds a clamp on the 4070 Ti against about 5 as a draw
/// (`bare_cost`, batches (k) and (l)). The targets are unbound around the
/// points (the pipeline names none), and the caller binds its own again
/// ([`clamped_as_draw`]).
fn clamp_draws() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_DRAW").is_some_and(|v| v == "1"))
}

/// Whether the clamp of `records` of `kind` is recorded as a draw
/// (of one record only: the vertex program charges each record alone)
/// ([`clamp_draws`]): then the list's graphics root signature, pipeline,
/// topology and targets are ours after it.
pub(crate) fn clamped_as_draw(kind: u32, records: u32) -> bool {
    kind != 2 && records == 1 && clamp_draws() && clamp().is_some_and(|c| c.draw.is_some())
}



// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    let (root, pso) = make_stage(device, SOURCE, s!("main"), s!("cs_5_1"), |root, bytes| {
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
        }
    })?;
    let draw = make_stage(device, SOURCE, s!("vertex"), s!("vs_5_1"), |root, bytes| {
        // SAFETY: as above.
        unsafe {
            device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            })
        }
    });
    if let Err(e) = &draw {
        tracing::warn!(error = %e, "no argument clamp as a draw: compute passes only");
    }
    let merged = make_stage(device, &merged_source(), s!("main"), s!("cs_5_1"), |root, bytes| {
        // SAFETY: as above.
        unsafe {
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
        }
    });
    if let Err(e) = &merged {
        tracing::warn!(error = %e, "no merged argument clamp: one pass a clamp");
    }
    Ok(Clamp { root, pso, draw: draw.ok().map(|(_, pso)| pso), merged: merged.ok() })
}

/// [`SOURCE`]'s compute pass over many clamps at once ([`clamp_merge`]):
/// the record words become per-thread values read from a table of clamps
/// (`jobs` entries of 8 words: kind, stride, records, records' offset,
/// flags, count's offset, destination's offset, first group), whose
/// buffers are one source, one count, one destination and the budget.
/// A group clamps up to 64 records of one clamp, charged to the budget as
/// [`SOURCE`]'s groups are.
fn merged_source() -> String {
    let start = SOURCE.find("ByteAddressBuffer src_args").expect("the clamp's buffers");
    let end = SOURCE.find("// The same as a draw of one point").expect("the clamp's vertex program");
    let body = SOURCE[start..end]
        .replace("r.dst_at = i * stride;", "r.dst_at = dst_offset + i * stride;")
        .replace(
            "void main(uint3 id : SV_DispatchThreadID, uint at : SV_GroupIndex) {",
            "void main(uint3 gid : SV_GroupID, uint at : SV_GroupIndex) {
    uint g = gid.y * 65535u + gid.x;
    if (g >= group_count) {
        return;
    }
    // The clamp whose groups hold `g`: the last one starting at or before it.
    uint lo = 0;
    uint hi = jobs;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) / 2u;
        if (table.Load(mid * 32u + 28u) <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint4 a = table.Load4(lo * 32u);
    uint4 b = table.Load4(lo * 32u + 16u);
    kind = a.x;
    stride = a.y;
    records = a.z;
    args_offset = a.w;
    has_count = b.x;
    count_offset = b.y;
    dst_offset = b.z;
    uint3 id = uint3((g - b.w) * 64u + at, 0u, 0u);",
        );
    format!(
        "#define RS \"RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), SRV(t2)\"
cbuffer Jobs : register(b0) {{
    uint jobs;
    uint group_count;
}};
ByteAddressBuffer table : register(t2);
static uint kind;
static uint stride;
static uint records;
static uint args_offset;
static uint has_count;
static uint count_offset;
static uint dst_offset;
{body}"
    )
}

/// Whether a list's hoisted clamps ([`hoist_clamps`]) that share their
/// buffers run as one pass ([`merged_source`]) instead of one pass each
/// (`VICTORY_CLAMP_MERGE=1`; off by default until a run shows it).
fn clamp_merge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_MERGE").is_some_and(|v| v == "1"))
}

/// Bytes of a merged clamp's table entry ([`merged_source`]).
const JOB: u64 = 32;

/// The table bytes [`record_hoisted`] needs for `clamps` merged, if
/// [`clamp_merge`]: none without.
pub(crate) fn merged_table_bytes(clamps: &[Hoisted]) -> u64 {
    if clamp_merge() && clamps.len() > 1 { clamps.len() as u64 * JOB } else { 0 }
}

/// `clamps` in runs that one merged pass can clamp: the same source,
/// count, destination buffers and budget, offsets within 4 GiB of their
/// buffers, in order within each run. Each run: its clamps' places in
/// `clamps`, and its table entries.
fn merged_runs(clamps: &[Hoisted]) -> Vec<(Vec<usize>, Vec<[u32; 8]>)> {
    let mut runs: Vec<((u64, u64, u64, u64), Vec<usize>, Vec<[u32; 8]>, u32)> = Vec::new();
    for (i, h) in clamps.iter().enumerate() {
        let src = (h.src_gpu + u64::from(h.constants[3])).checked_sub(h.src_base);
        let count = (h.count_gpu + u64::from(h.constants[5])).checked_sub(h.count_base);
        let dst = h.dst_gpu.checked_sub(h.dst_base);
        let fits = |v: Option<u64>| v.and_then(|v| u32::try_from(v).ok());
        let (Some(src), Some(count), Some(dst)) = (fits(src), fits(count), fits(dst)) else {
            runs.push(((u64::MAX, i as u64, 0, 0), vec![i], Vec::new(), 0));
            continue;
        };
        let key = (h.src_base, h.count_base, h.dst_base, h.budget);
        let at = match runs.iter().position(|r| r.0 == key) {
            Some(at) => at,
            None => {
                runs.push((key, Vec::new(), Vec::new(), 0));
                runs.len() - 1
            }
        };
        let run = &mut runs[at];
        let c = h.constants;
        run.2.push([c[0], c[1], c[2], src, c[4], count, dst, run.3]);
        run.3 += h.records.div_ceil(64);
        run.1.push(i);
    }
    runs.into_iter().map(|(_, places, entries, _)| (places, entries)).collect()
}


/// The clamp's `entry` compiled for `target`, its root signature and the
/// pipeline `make` makes of them.
fn make_stage(
    device: &ID3D12Device,
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
    make: impl FnOnce(&ID3D12RootSignature, &[u8]) -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = make(&root, bytes)?;
        Ok((root, pso))
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    own: bool,
) -> Option<(ID3D12Resource, u64)> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget) = (clamp()?, list.budget()?);
    let (dst, at, fresh) = scratch_place(list, bytes)?;
    let wait = clamp_waits(list);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| clamp_note("clamp", spec, wait));
    let scoped = own && clamp_scoped();
    record_clamp_scoped(c, &list.list, (&dst, at), budget, src, count, spec, false, wait, fresh, clamped_as_draw(spec.0, spec.2), scoped).then(|| {
        clamp_waited(list, wait, scoped);
        (dst, at)
    })
}

/// `VICTORY_CLAMP_RING=1` (off by default; with [`enhanced_barriers`]
/// only): each clamp in a list writes its records after the last one's in
/// the list's scratch buffer (`HostList::indirect_ring`), not at its start,
/// so it need not wait for the indirect call before it to have read its
/// records there: the barrier that waited for that is left out, and the
/// clamp may run beside the call before it. When the ring goes round the
/// wait is made, for every place used before.
pub(crate) fn clamp_ring() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_RING").is_some_and(|v| v == "1") && enhanced_barriers())
}

/// Where a clamp of `bytes` writes in the list's scratch: the buffer, the
/// place, and its first barrier ([`Pre`]; `None` without enhanced
/// barriers).
fn scratch_place(list: &HostList, bytes: u64) -> Option<(ID3D12Resource, u64, Option<Pre>)> {
    if clamp_ring() {
        let (dst, at, round) = list.indirect_ring(bytes)?;
        let pre = if list.first_use(dst.as_raw() as usize ^ CLAMP_USE) {
            Pre::First
        } else if round {
            Pre::AfterCalls
        } else {
            crate::list::barrier_count(crate::list::B_CLAMP_RING, 1);
            Pre::None
        };
        return Some((dst, at, Some(pre)));
    }
    let dst = list.indirect_scratch(bytes)?;
    let fresh = enhanced(list, &dst);
    Some((dst, 0, fresh.map(|first| if first { Pre::First } else { Pre::AfterCalls })))
}

/// A clamp was recorded: its wait, if it waited for everything, is the
/// list's wait for every earlier shader write.
fn clamp_waited(list: &HostList, wait: bool, scoped: bool) {
    list.waited();
    if wait && !scoped {
        list.all_waited();
    }
}

/// With [`enhanced_barriers`]: the barrier on a clamp's scratch before it
/// writes there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pre {
    /// The buffer's first use in the list since its reset (from the common
    /// state, after everything).
    First,
    /// After the indirect calls before it read their records.
    AfterCalls,
    /// None: no call read the place since the list's last wait for them.
    None,
}

/// `VICTORY_CLAMP_SCOPED` (on by default, `=0` off): a clamp's wait for earlier
/// shader writes (the title's argument and count passes) is a UAV barrier
/// on the argument and count buffers alone, not on every resource, where
/// those are host copies of their own (not placed over title memory that
/// other buffers alias). Writes to the records through another resource
/// over the same memory are not waited for.
///
/// After the clamp barriers of vitutu1/victory-perf (`VICTORY_CLAMP_SCOPED`,
/// itself from ZigZagBoi1/ScarlettPort).
pub(crate) fn clamp_scoped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_SCOPED").is_none_or(|v| v != "0"))
}

/// Whether the scratch buffer's barriers around the clamp are enhanced
/// ones (`ID3D12GraphicsCommandList7::Barrier`), which say what waits for
/// what: the clamp waits only for the last indirect call's reading of the
/// arguments, and the indirect call only for the clamp, where a change of
/// the buffer's state waited for all the card's earlier work, the draws
/// before included. Where the card and its driver have them (and not with
/// `VICTORY_CLAMP_ENHANCED=0`).
pub(crate) fn enhanced_barriers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_CLAMP_ENHANCED").is_some_and(|v| v == "0") {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a plain query into a struct of the asked size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut options).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        let on = asked.is_ok() && options.EnhancedBarriersSupported.as_bool();
        tracing::info!("indirect arguments' clamp: enhanced barriers {}", if on { "on" } else { "not supported" });
        on
    })
}

/// With [`enhanced_barriers`]: `None` without them; else whether this is
/// the scratch buffer's first use in the list since its reset (its state
/// before is then the common one).
fn enhanced(list: &HostList, dst: &ID3D12Resource) -> Option<bool> {
    enhanced_barriers().then(|| list.first_use(dst.as_raw() as usize ^ CLAMP_USE))
}

/// [`enhanced`]'s key of a scratch buffer's first use in a list.
const CLAMP_USE: usize = 0x5ca7_c4a3;

/// An enhanced barrier on all of buffer `resource`.
///
/// # Safety
/// An open list and a live buffer.
unsafe fn buffer_barrier(
    l: &ID3D12GraphicsCommandList7,
    resource: &ID3D12Resource,
    before: (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS),
    after: (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS),
) {
    let barrier = D3D12_BUFFER_BARRIER {
        SyncBefore: before.0,
        SyncAfter: after.0,
        AccessBefore: before.1,
        AccessAfter: after.1,
        // SAFETY: borrowed for the call; never released through this copy.
        pResource: std::mem::ManuallyDrop::new(Some(unsafe { std::mem::transmute_copy(resource) })),
        Offset: 0,
        Size: u64::MAX,
    };
    let group = D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_BUFFER,
        NumBarriers: 1,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pBufferBarriers: &raw const barrier,
        },
    };
    // SAFETY: the caller's.
    unsafe { l.Barrier(&[group]) };
}

/// Whether the clamp first waits for every earlier shader write: only if
/// shaders recorded since the list's last such wait may have written
/// buffers (`HostList::writes_unwaited`). Each indirect call had that wait
/// (4,942 a frame in the world, run fxgt2, about 17 microseconds of the
/// card's time each with the clamp): between draws that store to no
/// buffer it waited for nothing they wrote, and kept the card from
/// overlapping them. `VICTORY_CLAMP_ALWAYS_WAITS=1` waits every time.
fn clamp_waits(list: &HostList) -> bool {
    static ALWAYS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALWAYS.get_or_init(|| std::env::var_os("VICTORY_CLAMP_ALWAYS_WAITS").is_some_and(|v| v == "1"))
        || list.writes_unwaited()
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
    own: bool,
) -> Option<(ID3D12Resource, u64)> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let (dst, at, fresh) = scratch_place(list, bytes)?;
    let wait = clamp_waits(list);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| clamp_note("clamp-window", spec, wait));
    let scoped = own && clamp_scoped();
    record_clamp_scoped(c, &list.list, (&dst, at), budget, src, None, spec, true, wait, fresh, clamped_as_draw(spec.0, spec.2), scoped).then(|| {
        clamp_waited(list, wait, scoped);
        (dst, at)
    })
}

/// `VICTORY_HOIST_CLAMPS=1` (off by default): the clamp of an indirect
/// call whose records (and count) no work recorded before it in the list
/// may write is recorded ahead of the list, in its prelude
/// (`HostList::prelude`), with the list's other such clamps: one wait for
/// earlier writes, the clamps' dispatches back to back, one wait, then the
/// list. Each clamp in the list cost a switch to compute and two waits
/// (about 13 microseconds of the card's time each, 64 ms a frame over 5k
/// calls). After Belobug's `VICTORY_HOIST_CLAMPS`, written anew.
///
/// What it keeps: the records a hoisted clamp reads are those the inline
/// clamp would read. Ahead of the list are the CPU's steps (all of them
/// run at submission either way), earlier lists and flights (ahead of the
/// prelude in the queue); the list's own card writes before the call are
/// in what it declared written (`HostList::gpu_written`, the ranges
/// `VICTORY_CLAMP_SCOPED`'s window reuse and the copy-back go by), and a
/// call any of them may reach is clamped in the list as before. Each
/// hoisted clamp has its own place in the list's hoist buffer, so the
/// clamped records stay until the call. What differs: the budget is
/// charged by the hoisted clamps first, side by side (the dispatches are
/// not ordered among themselves), then by the list's own in order; it
/// only decides which records run empty past 2^27 of work in one list.
///
/// `VICTORY_HOIST_CLAMPS=2`: only before the list's first call whose
/// shaders may store (`HostList::write_count`), for a run that does not
/// trust the declarations (as `VICTORY_WINDOW_BY_WRITES`, off by default,
/// does). Only calls whose records are host copies of their own (where
/// `VICTORY_CLAMP_SCOPED` waits on them alone) are hoisted.
pub(crate) fn hoist_clamps() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_HOIST_CLAMPS").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// A clamp recorded ahead of its list ([`hoist_clamps`]).
#[derive(Clone)]
pub(crate) struct Hoisted {
    constants: [u32; 6],
    src_gpu: u64,
    count_gpu: u64,
    dst_gpu: u64,
    budget: u64,
    records: u32,
    /// The start of the source's, the count's and the destination's
    /// buffers ([`merged_runs`]).
    src_base: u64,
    count_base: u64,
    dst_base: u64,
}

/// A list's clamps recorded ahead of it ([`hoist_clamps`]) and the buffer
/// their records go to.
#[derive(Default)]
pub(crate) struct Hoist {
    /// The buffer clamps go to now and its bytes; how many are taken.
    buffer: Option<(ID3D12Resource, u64)>,
    used: u64,
    /// The buffers written since the reset (the last is `buffer`'s).
    written: Vec<ID3D12Resource>,
    clamps: Vec<Hoisted>,
    /// Merged clamps' tables ([`clamp_merge`]): upload memory, its
    /// mapping and bytes, kept over resets.
    table: Option<(ID3D12Resource, usize, u64)>,
}

// SAFETY: the mapping is a persistent one of upload memory the list holds.
unsafe impl Send for Hoist {}

impl Hoist {
    /// Room for `bytes` of clamped records: the buffer and the place. A
    /// buffer outgrown is returned too, to be kept until the reset.
    fn take(&mut self, bytes: u64) -> Option<(ID3D12Resource, u64, Option<ID3D12Resource>)> {
        const ALIGN: u64 = 256;
        let bytes = bytes.max(4).next_multiple_of(ALIGN);
        if let Some((buffer, cap)) = &self.buffer
            && self.used + bytes <= *cap
        {
            let at = self.used;
            self.used += bytes;
            if self.written.last() != Some(buffer) {
                self.written.push(buffer.clone());
            }
            return Some((buffer.clone(), at, None));
        }
        let prev = self.buffer.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev.saturating_mul(2).max(bytes).max(256 << 10).next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        let old = self.buffer.replace((buffer.clone(), alloc)).map(|(b, _)| b);
        self.used = bytes;
        self.written.push(buffer.clone());
        Some((buffer, 0, old))
    }

    /// Upload memory for `bytes` of merged clamps' tables (the list's
    /// prelude is not running): its GPU address and mapping.
    pub(crate) fn table(&mut self, bytes: u64) -> Option<(u64, *mut u8)> {
        if !self.table.as_ref().is_some_and(|t| t.2 >= bytes) {
            let size = bytes.next_power_of_two().max(4 << 10);
            let (buffer, data) = host()?.upload_buffer(size)?;
            // (A table outgrown was read by a run that is over.)
            self.table = Some((buffer, data as usize, size));
        }
        let (buffer, data, _) = self.table.as_ref()?;
        // SAFETY: a live buffer.
        Some((unsafe { buffer.GetGPUVirtualAddress() }, *data as *mut u8))
    }

    /// The list was reset: its clamps are gone (the buffer is kept).
    pub(crate) fn clear(&mut self) {
        self.used = 0;
        self.written.clear();
        self.clamps.clear();
    }

    /// For the prelude: the clamps and the buffers they write. Kept: a
    /// list run again clamps again.
    pub(crate) fn to_record(&self) -> (Vec<Hoisted>, Vec<ID3D12Resource>) {
        (self.clamps.clone(), self.written.clone())
    }
}

/// [`hoist_clamps`]: the clamp of `spec`'s records at `src` (bounded by
/// `count`) recorded ahead of `list`; the buffer and place its records
/// will be at for the call. `ahead`: a window ([`bound_ahead`]). The
/// caller has made sure no work of the list before the call may write the
/// records or the count. `None`: clamp it in the list.
pub(crate) fn hoisted(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> Option<(ID3D12Resource, u64)> {
    if hoist_clamps() == 0 || spec.2 == 0 {
        return None;
    }
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    clamp()?;
    let budget = list.budget()?;
    let (buffer, at, old) = list.hoist().take(bytes)?;
    if let Some(old) = old {
        list.keep(&old);
    }
    // SAFETY: a live buffer.
    let dst_gpu = unsafe { buffer.GetGPUVirtualAddress() } + at;
    let (constants, src_gpu, count_gpu) = clamp_inputs(src, count, spec, ahead, dst_gpu, budget)?;
    #[cfg(feature = "diagnostics")]
    list.note_op(|| clamp_note("clamp-hoisted", spec, false));
    // SAFETY: live buffers.
    let (src_base, count_base, dst_base) = unsafe {
        (
            src.0.GetGPUVirtualAddress(),
            count.map_or(src_gpu, |(c, _)| c.GetGPUVirtualAddress()),
            buffer.GetGPUVirtualAddress(),
        )
    };
    list.hoist().clamps.push(Hoisted {
        constants,
        src_gpu,
        count_gpu,
        dst_gpu,
        budget,
        records: spec.2,
        src_base,
        count_base,
        dst_base,
    });
    HOISTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some((buffer, at))
}

/// Clamps hoisted ahead of their lists ([`hoist_clamps`]), since the last
/// look ([`hoist_line`]).
static HOISTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The frame-rate line's part of [`hoist_clamps`]: clamps hoisted a frame.
pub(crate) fn hoist_line(frames: f64) {
    if hoist_clamps() != 0 {
        let hoisted = HOISTED.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames;
        tracing::info!("indirect clamps hoisted ahead of their lists per frame: {hoisted:.1} (VICTORY_HOIST_CLAMPS)");
        if clamp_merge() {
            let [alone, merged] = [0, 1].map(|i| MERGE_STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
            tracing::info!(
                "hoisted clamps per frame: {merged:.1} merged into shared passes, {alone:.1} in passes of their own (VICTORY_CLAMP_MERGE)"
            );
        }
    }
}

/// Records a list's hoisted clamps into its prelude `l` ([`hoist_clamps`]):
/// one wait for every earlier write (the lists and flights ahead), the
/// buffers to unordered access, the clamps, one wait, the buffers to
/// indirect arguments (as the list's calls read them; they return to the
/// common state at the end of the submission, as buffers do).
///
/// # Safety
/// An open list; live buffers (the list holds them until its reset).
/// `table`: upload memory of [`merged_table_bytes`] for merged clamps
/// ([`clamp_merge`]; GPU address and mapping), not read by a run still
/// going.
pub(crate) unsafe fn record_hoisted(
    l: &ID3D12GraphicsCommandList,
    clamps: &[Hoisted],
    buffers: &[ID3D12Resource],
    table: Option<(u64, *mut u8)>,
) {
    if let Some(c) = clamp() {
        // SAFETY: the caller's.
        unsafe { record_hoisted_with(c, l, clamps, buffers, table) };
    }
}

/// Clamps run as passes of their own, and merged ([`clamp_merge`]): since
/// the last look ([`hoist_line`]).
static MERGE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`record_hoisted`] with clamp `c`.
///
/// # Safety
/// As [`record_hoisted`].
unsafe fn record_hoisted_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    clamps: &[Hoisted],
    buffers: &[ID3D12Resource],
    table: Option<(u64, *mut u8)>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    if clamps.is_empty() {
        return;
    }
    // The runs merged and the clamps left to passes of their own.
    let mut merged: Vec<(u64, u64, u64, u64, u64, u32, u32)> = Vec::new();
    let mut alone: Vec<&Hoisted> = Vec::new();
    match (table, c.merged.as_ref()) {
        (Some((table_gpu, table_ptr)), Some(_)) => {
            let mut at = 0u64;
            for (places, entries) in merged_runs(clamps) {
                let h = &clamps[places[0]];
                if entries.len() < 2 {
                    alone.extend(places.iter().map(|&i| &clamps[i]));
                    continue;
                }
                let groups: u32 = places.iter().map(|&i| clamps[i].records.div_ceil(64)).sum();
                // SAFETY: the caller's table holds a entry for each clamp.
                unsafe {
                    std::ptr::copy_nonoverlapping(entries.as_ptr().cast::<u8>(), table_ptr.add(at as usize), entries.len() * JOB as usize);
                }
                merged.push((table_gpu + at, h.src_base, h.count_base, h.dst_base, h.budget, entries.len() as u32, groups));
                at += entries.len() as u64 * JOB;
            }
        }
        _ => alone.extend(clamps),
    }
    MERGE_STATS[0].fetch_add(alone.len() as u64, Relaxed);
    MERGE_STATS[1].fetch_add((clamps.len() - alone.len()) as u64, Relaxed);
    let mut first = vec![uav_barrier()];
    first.extend(buffers.iter().map(|b| transition(b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)));
    let mut last = vec![uav_barrier()];
    last.extend(
        buffers
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)),
    );
    // SAFETY: the caller's; our root signature and pipeline.
    unsafe {
        l.ResourceBarrier(&first);
        if !alone.is_empty() {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
        }
        for h in alone {
            l.SetComputeRoot32BitConstants(0, 6, h.constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, h.src_gpu);
            l.SetComputeRootShaderResourceView(2, h.count_gpu);
            l.SetComputeRootUnorderedAccessView(3, h.dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, h.budget);
            l.Dispatch(h.records.div_ceil(64), 1, 1);
        }
        if let (Some((root, pso)), false) = (c.merged.as_ref(), merged.is_empty()) {
            l.SetComputeRootSignature(root);
            l.SetPipelineState(pso);
            for (table, src, count, dst, budget, jobs, groups) in merged {
                l.SetComputeRoot32BitConstants(0, 2, [jobs, groups].as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src);
                l.SetComputeRootShaderResourceView(2, count);
                l.SetComputeRootUnorderedAccessView(3, dst);
                l.SetComputeRootUnorderedAccessView(4, budget);
                l.SetComputeRootShaderResourceView(5, table);
                l.Dispatch(groups.min(65535), groups.div_ceil(65535), 1);
            }
        }
        l.ResourceBarrier(&last);
    }
}

/// For timing notes: a clamp's own line, so the card's time of its waits
/// and pass is told apart from the indirect call after it.
#[cfg(feature = "diagnostics")]
fn clamp_note(what: &str, (kind, _, records): (u32, u32, u32), wait: bool) -> String {
    let drawn = if clamped_as_draw(kind, records) { " drawn" } else { "" };
    let wait = if wait { "waiting" } else { "alone" };
    format!("{what}{drawn} {wait} kind {kind} records {records}")
}

#[cfg(test)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_scoped(c, l, (dst, 0), budget, src, count, spec, false, true, None, false, false)
}

/// [`record_clamp`]; `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]); `wait`: first a wait for every earlier
/// shader write ([`clamp_waits`]); `fresh`: with [`enhanced_barriers`],
/// whether `dst` is used for the first time in the list; `as_draw`: the
/// clamp is a draw of points ([`clamped_as_draw`]).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
    wait: bool,
    fresh: Option<bool>,
    as_draw: bool,
) -> bool {
    let fresh = fresh.map(|first| if first { Pre::First } else { Pre::AfterCalls });
    record_clamp_scoped(c, l, (dst, 0), budget, src, count, spec, ahead, wait, fresh, as_draw, false)
}

/// The clamp's inputs for `spec`'s records at `src` (bounded by `count`)
/// into `dst_gpu`: its root constants and the source's and count's
/// addresses. `None` if misaligned.
fn clamp_inputs(
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
    dst_gpu: u64,
    budget: u64,
) -> Option<([u32; 6], u64, u64)> {
    let bytes = validate_layout((kind, stride, records), src_offset, count.map(|(_, off)| off))?;
    // SAFETY: a live D3D12 buffer resource.
    let src_base = unsafe { src.GetGPUVirtualAddress() };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return None;
    }
    let (src_gpu, args_offset) = match src_offset.checked_add(bytes).filter(|&end| end <= u64::from(u32::MAX)) {
        Some(_) => (src_base, src_offset as u32),
        None => (src_base.checked_add(src_offset)?, 0),
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return None;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => (base.checked_add(off)?, 1u32, 0u32),
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let flags = has_count | (u32::from(ahead) << 1) | (u32::from(!group_charge()) << 2);
    Some(([kind, stride, records, args_offset, flags, count_offset], src_gpu, count_gpu))
}

/// [`record_clamp_as`]; `scoped`: the wait for earlier shader writes is on
/// the argument and count buffers alone ([`clamp_scoped`]).
#[allow(clippy::too_many_arguments)]
fn record_clamp_scoped(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, dst_at): (&ID3D12Resource, u64),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
    wait: bool,
    fresh: Option<Pre>,
    as_draw: bool,
    scoped: bool,
) -> bool {
    // SAFETY: a live D3D12 buffer resource.
    let dst_gpu = unsafe { dst.GetGPUVirtualAddress() } + dst_at;
    let Some((constants, src_gpu, count_gpu)) =
        clamp_inputs((src, src_offset), count, (kind, stride, records), ahead, dst_gpu, budget)
    else {
        return false;
    };
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    // Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    // The wait for earlier shader writes: on every resource, or (scoped)
    // on the argument and count buffers alone.
    let mut first: Vec<D3D12_RESOURCE_BARRIER> = if scoped {
        let mut on = vec![uav_barrier_on(src)];
        if let Some((buf, _)) = count.filter(|(buf, _)| *buf != src) {
            on.push(uav_barrier_on(buf));
        }
        on
    } else {
        vec![uav_barrier()]
    };
    let waited = first.len();
    first.push(transition(
        dst,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    ));
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip = usize::from(!waits);
    let draw = if as_draw { c.draw.as_ref() } else { None };
    let shading = if draw.is_some() {
        D3D12_BARRIER_SYNC_VERTEX_SHADING
    } else {
        D3D12_BARRIER_SYNC_COMPUTE_SHADING
    };
    // SAFETY: an open list; our root signature and pipelines; the buffers
    // live (as above).
    let run = || unsafe {
        match (records > 0, draw) {
            (false, _) => {}
            (true, Some(pso)) => {
                // No targets: the pipeline names none.
                l.OMSetRenderTargets(0, None, false, None);
                l.SetGraphicsRootSignature(&c.root);
                l.SetPipelineState(pso);
                l.SetGraphicsRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                l.SetGraphicsRootShaderResourceView(1, src_gpu);
                l.SetGraphicsRootShaderResourceView(2, count_gpu);
                l.SetGraphicsRootUnorderedAccessView(3, dst_gpu);
                l.SetGraphicsRootUnorderedAccessView(4, budget);
                // A point list.
                l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(1));
                l.DrawInstanced(records, 1, 0, 0);
            }
            (true, None) => {
                l.SetComputeRootSignature(&c.root);
                l.SetPipelineState(&c.pso);
                l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src_gpu);
                l.SetComputeRootShaderResourceView(2, count_gpu);
                l.SetComputeRootUnorderedAccessView(3, dst_gpu);
                l.SetComputeRootUnorderedAccessView(4, budget);
                l.Dispatch(records.div_ceil(64), 1, 1);
            }
        }
    };
    if let Some(fresh) = fresh
        && let Ok(l7) = l.cast::<ID3D12GraphicsCommandList7>()
    {
        let before = match fresh {
            Pre::First => Some((D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON)),
            Pre::AfterCalls => Some((D3D12_BARRIER_SYNC_EXECUTE_INDIRECT, D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT)),
            Pre::None => None,
        };
        let written = (shading, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
        count_clamp_barriers(waits && wait, scoped);
        // SAFETY: as above.
        unsafe {
            if waits && wait {
                l.ResourceBarrier(&first[..waited]);
            }
            if let Some(before) = before {
                buffer_barrier(&l7, dst, before, written);
            }
            run();
            buffer_barrier(&l7, dst, written, (D3D12_BARRIER_SYNC_EXECUTE_INDIRECT, D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT));
        }
        return true;
    }
    count_clamp_barriers(waits && wait, scoped);
    // SAFETY: as above.
    unsafe {
        // Earlier shader writes (the title's argument and count passes)
        // first, if any may be unwaited for.
        l.ResourceBarrier(&first[if waits && wait { 0 } else { waited }..]);
        run();
        l.ResourceBarrier(&second[skip..]);
    }
    true
}

/// For the barrier census (`crate::list::BARRIERS`): a clamp's wait for
/// earlier writes, if made, on everything or on its records alone, and its
/// pair of barriers on its scratch.
fn count_clamp_barriers(waited: bool, scoped: bool) {
    use crate::list::{B_CLAMP_PAIR, B_CLAMP_SCOPED, B_CLAMP_WAIT, barrier_count};
    if waited {
        barrier_count(if scoped { B_CLAMP_SCOPED } else { B_CLAMP_WAIT }, 1);
    }
    barrier_count(B_CLAMP_PAIR, 1);
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    // With enhanced barriers the buffer stays as the indirect call reads
    // it; the next clamp's barrier starts from there.
    if enhanced_barriers() {
        return;
    }
    done_on(&list.list, scratch);
}

/// Diagnostics (`crate::meshprobe`): after an indirect call that read
/// `len` bytes of `scratch` at `at`, a copy of them and of the title's
/// records (`source`, as they are by then) into a mapped buffer: the
/// clamped records at 0, the title's at `len`. The scratch buffer is left
/// as the call left it.
#[cfg(feature = "diagnostics")]
pub(crate) fn copy_args_out(
    list: &HostList,
    scratch: &ID3D12Resource,
    at: u64,
    (src, src_at): (&ID3D12Resource, u64),
    len: u64,
) -> Option<(ID3D12Resource, *mut u8)> {
    let (buffer, data) = host()?.shared_buffer(2 * len)?;
    let l = &list.list;
    let enhanced = enhanced_barriers().then(|| l.cast::<ID3D12GraphicsCommandList7>().ok()).flatten();
    let indirect = (D3D12_BARRIER_SYNC_EXECUTE_INDIRECT, D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT);
    let copying = (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE);
    // SAFETY: an open list; the scratch buffer in the state the indirect
    // call read it in; live buffers; ranges inside them (the call read
    // them).
    unsafe {
        // The title's records as every shader recorded before wrote them.
        l.ResourceBarrier(&[uav_barrier()]);
        match &enhanced {
            Some(l7) => {
                buffer_barrier(l7, scratch, indirect, copying);
                buffer_barrier(l7, &buffer, (D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_ACCESS_NO_ACCESS), (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST));
            }
            None => l.ResourceBarrier(&[transition(scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]),
        }
        l.CopyBufferRegion(&buffer, 0, scratch, at, len);
        l.CopyBufferRegion(&buffer, len, src, src_at, len);
        match &enhanced {
            Some(l7) => buffer_barrier(l7, scratch, copying, indirect),
            None => l.ResourceBarrier(&[transition(scratch, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]),
        }
    }
    list.keep(&buffer);
    Some((buffer, data))
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// A wait for earlier unordered writes to `resource` alone.
fn uav_barrier_on(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: borrowed for the barrier's recording, never released
            // through this copy (`ManuallyDrop`).
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(unsafe { std::mem::transmute_copy(resource) })),
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
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

#[cfg(test)]
mod most_groups {
    /// [`super::MOST_GROUPS`] is what the clamp lets a dispatch record ask.
    #[test]
    fn the_clamp_caps_dispatch_groups_at_most_groups() {
        let [x, y, z] = super::MOST_GROUPS;
        let cap = format!("min(asked, uint3({x}u, {y}u, {z}u))");
        assert!(super::SOURCE.contains(&cap), "the clamp no longer caps dispatch groups with {cap}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
        /// One harness at a time: WARP tests on parallel test threads
        /// crashed the test process now and then (an access violation in
        /// about one full run in five once a fifth WARP test was added).
        /// Last, so the device and its objects go first.
        _alone: std::sync::MutexGuard<'static, ()>,
    }

    /// Held by the [`WarpHarness`] alive.
    static WARP_ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    impl WarpHarness {
        fn new() -> Option<Self> {
            let alone = WARP_ALONE.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                    _alone: alone,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso, draw: None, merged: None })
        }
    }

    #[test]
    fn clamp_bounds_match_the_shader() {
        assert_eq!(MAX_GROUPS_PER_AXIS, 65535);
        assert!(SOURCE.contains("uint3 g = min(asked, uint3(65535u, 65535u, 65535u));"));
        assert!(SOURCE.contains("g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));"));
        assert_eq!(MAX_DISPATCH_GROUPS, 1 << 20);
        assert_eq!(BUDGET, 1 << 27);
        assert!(SOURCE.contains("static const uint BUDGET = 1u << 27;"));
        assert!(SOURCE.contains("static const uint GROUPS = 1u << 20;"));
    }

    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        card_compare_stores_the_hash_it_made_when_told(crate::host::CARD_COMPARE, false);
    }

    /// The same of the compare with a group per piece
    /// (`crate::host::CARD_COMPARE_WIDE`).
    #[test]
    fn warp_wide_card_compare_stores_the_hash_it_made_when_told() {
        card_compare_stores_the_hash_it_made_when_told(crate::host::CARD_COMPARE_WIDE, true);
    }

    /// The compare with a group per piece flags the same pieces as the
    /// one with a thread per piece, for changes of one byte anywhere in a
    /// piece (its first and last bytes too) and for two 16-byte runs of a
    /// piece swapped, near each other and 4 KiB apart (one thread of the
    /// group reads both of those). Four pieces, the copy starting 48 bytes
    /// into the first (`skew`), so the first and last are cut short.
    #[test]
    fn warp_wide_and_narrow_card_compares_flag_the_same_pieces() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 4;
        const SKEW: u32 = 48;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251 + i / 4093) as u8).collect();
        let shaders = [
            (compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("narrow"), false),
            (compile_cs(&warp.device, crate::host::CARD_COMPARE_WIDE).expect("wide"), true),
        ];
        let run = |warp: &mut WarpHarness, c: &Clamp, wide: bool, mode: u32| -> [u32; 4] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&c.root);
                warp.list.SetPipelineState(&c.pso);
                let constants = [0u32, SKEW, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                let (x, y) = crate::host::card_compare_groups(PIECES, wide);
                warp.list.Dispatch(x, y, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2, 3].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let reset = || {
            // SAFETY: the mapping holds `size` bytes.
            unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        };
        // (what changes, the pieces it lies in). Piece p of the copy
        // starts at p * 64 KiB - SKEW.
        let piece_at = |p: usize| p * PIECE - SKEW as usize;
        type Change = Box<dyn Fn(*mut u8)>;
        let flip = |at: usize| -> Change { Box::new(move |base: *mut u8| unsafe { *base.add(at) ^= 0x10 }) };
        let swap = |a: usize, b: usize| -> Change {
            Box::new(move |base: *mut u8| unsafe { std::ptr::swap_nonoverlapping(base.add(a), base.add(b), 16) })
        };
        let cases: Vec<(Change, [u32; 4])> = vec![
            (flip(0), [1, 0, 0, 0]),
            (flip(piece_at(1) - 1), [1, 0, 0, 0]),
            (flip(piece_at(1)), [0, 1, 0, 0]),
            (flip(piece_at(2) + 12_345), [0, 0, 1, 0]),
            (flip(size - SKEW as usize - 1), [0, 0, 0, 1]),
            (swap(piece_at(1) + 1024, piece_at(1) + 1040), [0, 1, 0, 0]),
            (swap(piece_at(2) + 4096, piece_at(2) + 8192), [0, 0, 1, 0]),
        ];
        for (n, (change, expected)) in cases.iter().enumerate() {
            for (c, wide) in &shaders {
                reset();
                assert_eq!(run(&mut warp, c, *wide, 0), [7; 4]);
                assert_eq!(run(&mut warp, c, *wide, 1), [0; 4], "case {n}, wide {wide}: nothing changed yet");
                change(now_ptr);
                assert_eq!(run(&mut warp, c, *wide, 1), *expected, "case {n}, wide {wide}");
            }
        }
    }

    /// The wide compare's hash of one piece changes for a flip of one byte
    /// at every 16-byte offset of the bytes it covers: a whole piece, a
    /// short first piece (the copy starting 5,000 bytes into it) and a
    /// short last piece (the copy ending 62,536 bytes into it, not a
    /// multiple of 4 KiB); and not for a byte past what it covers.
    #[test]
    fn warp_wide_card_hash_sees_every_byte() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE_WIDE).expect("the wide compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        // Variants hashed per submission, each in a region of its own.
        const BATCH: usize = 512;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer((PIECE * BATCH) as u64);
        let (_hashes, hashes_ptr, hashes_gpu) = warp.shared_buffer((16 * BATCH) as u64);
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let base: Vec<u8> = (0..PIECE).map(|i| (i * 131 + i / 977 + 7) as u8).collect();
        // (skew, size): the piece's bytes are [0, hi) of its region, hi
        // widened to 16 bytes.
        for (skew, size) in [(0u32, PIECE as u32), (5000, PIECE as u32), (0, 62_536)] {
            let hi = (PIECE as u32 - skew).min(size);
            let covered = (hi as usize).div_ceil(16) * 16;
            // Byte flipped per variant: none, one per 16-byte slot it covers
            // (a different byte of the slot each time), then four past it.
            let mut flips: Vec<Option<usize>> = vec![None];
            flips.extend((0..covered / 16).map(|k| Some(k * 16 + k % 16)));
            let outside: Vec<usize> = (covered..PIECE).step_by(4093).take(4).collect();
            flips.extend(outside.iter().map(|&at| Some(at)));
            let mut hashes: Vec<[u32; 4]> = Vec::new();
            for batch in flips.chunks(BATCH) {
                // SAFETY: our open list, live buffers, the compare's root
                // signature; each region and hash slot is inside its mapping.
                unsafe {
                    for (v, flip) in batch.iter().enumerate() {
                        let region = now_ptr.add(v * PIECE);
                        std::ptr::copy_nonoverlapping(base.as_ptr(), region, PIECE);
                        if let Some(at) = *flip {
                            *region.add(at) ^= 0x5A;
                        }
                    }
                    warp.list.SetComputeRootSignature(&compare.root);
                    warp.list.SetPipelineState(&compare.pso);
                    warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                    for v in 0..batch.len() {
                        // Piece 0 of a copy starting `skew` into it; the
                        // hash stored (mode 0) in the variant's slot.
                        let constants = [0u32, skew, size, 1, 0, 0];
                        warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        warp.list.SetComputeRootShaderResourceView(1, now_gpu + (v * PIECE) as u64);
                        warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu + (v * 16) as u64);
                        let (x, y) = crate::host::card_compare_groups(1, true);
                        warp.list.Dispatch(x, y, 1);
                    }
                }
                warp.submit_and_wait();
                for v in 0..batch.len() {
                    // SAFETY: inside the mapping.
                    hashes.push(std::array::from_fn(|w| unsafe { hashes_ptr.cast::<u32>().add(v * 4 + w).read_volatile() }));
                }
            }
            let first = hashes[0];
            let inside = covered / 16;
            for (k, h) in hashes[1..=inside].iter().enumerate() {
                assert_ne!(*h, first, "skew {skew} size {size}: a flip at byte {} left the hash as it was", k * 16 + k % 16);
            }
            for (at, h) in outside.iter().zip(&hashes[inside + 1..]) {
                assert_eq!(*h, first, "skew {skew} size {size}: a flip at byte {at}, past the piece, changed its hash");
            }
        }
    }

    /// Groups of the wide compare for any number of pieces: the piece of
    /// group (x, y), `y * 65535 + x`, takes every piece once, and groups
    /// past the last piece are all at its end (they return at once).
    #[test]
    fn wide_compare_groups_take_every_piece_once() {
        for pieces in [1u32, 2, 65_534, 65_535, 65_536, 131_070, 131_071, 200_000] {
            let (x, y) = crate::host::card_compare_groups(pieces, true);
            assert!(x <= 65_535 && y <= 65_535 && x >= 1 && y >= 1);
            let mut seen = vec![false; pieces as usize];
            for gy in 0..y {
                for gx in 0..x {
                    let n = gy * 65_535 + gx;
                    if n < pieces {
                        assert!(!seen[n as usize], "{pieces}: piece {n} twice");
                        seen[n as usize] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "{pieces}: a piece left out");
        }
    }

    fn card_compare_stores_the_hash_it_made_when_told(source: &str, wide: bool) {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, source).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                let (x, y) = crate::host::card_compare_groups(PIECES, wide);
                warp.list.Dispatch(x, y, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// The push of flagged pieces (`crate::host::CARD_PUSH`) after the
    /// compare, on the software adapter: its first pass zeroes the plan's
    /// counter; then each flagged piece takes a slot of the plan's while
    /// they last, its bytes go there and its dword in `differs` becomes the
    /// slot plus 2; pieces past the last slot keep their 1, others their 0.
    #[test]
    fn warp_card_push_puts_flagged_pieces_in_slots() {
        card_push_puts_flagged_pieces_in_slots(crate::host::CARD_COMPARE, false);
    }

    /// The same after the wide compare (`crate::host::CARD_COMPARE_WIDE`):
    /// the push reads its `differs` as it reads the narrow one's.
    #[test]
    fn warp_card_push_after_the_wide_compare() {
        card_push_puts_flagged_pieces_in_slots(crate::host::CARD_COMPARE_WIDE, true);
    }

    fn card_push_puts_flagged_pieces_in_slots(compare_source: &str, wide: bool) {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, compare_source).expect("the card compare must compile on WARP");
        let push = compile_cs(&warp.device, crate::host::CARD_PUSH).expect("the card push must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        const STRIDE: usize = crate::host::PUSH_STRIDE;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let (_ring, ring_ptr, ring_gpu) = warp.shared_buffer((STRIDE * 8) as u64);
        let (_counters, counters_ptr, counters_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        let uav_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        // Hashes stored, then (counter 1, a stale count in it) zeroed,
        // compared with mode 2, pushed to slots `base..base + cap`.
        let run = |warp: &mut WarpHarness, mode: u32, base: u32, cap: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, each pass's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                counters_ptr.cast::<u32>().add(1).write_volatile(99);
                warp.list.SetComputeRootSignature(&push.root);
                warp.list.SetPipelineState(&push.pso);
                let zero = [0u32, 0, 0, 0, 0, 0, 0, 4];
                warp.list.SetComputeRoot32BitConstants(0, 8, zero.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, ring_gpu);
                warp.list.SetComputeRootUnorderedAccessView(4, counters_gpu);
                warp.list.Dispatch(1, 1, 1);
                warp.list.ResourceBarrier(std::slice::from_ref(&uav_barrier));
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                let (x, y) = crate::host::card_compare_groups(PIECES, wide);
                warp.list.Dispatch(x, y, 1);
                if mode != 0 {
                    warp.list.ResourceBarrier(std::slice::from_ref(&uav_barrier));
                    warp.list.SetComputeRootSignature(&push.root);
                    warp.list.SetPipelineState(&push.pso);
                    let constants = [0u32, 0, size as u32, PIECES, 0, base, cap, 4];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, ring_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, counters_gpu);
                    warp.list.Dispatch(PIECES, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        // SAFETY (each): inside the mappings.
        let set = |at: usize, value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        let slot = |n: u32| unsafe { std::slice::from_raw_parts(ring_ptr.add(n as usize * STRIDE), PIECE) };
        let piece = |p: usize| unsafe { std::slice::from_raw_parts(now_ptr.add(p * PIECE), PIECE) };
        run(&mut warp, 0, 0, 0);
        // Nothing changed: nothing pushed, the counter zeroed.
        assert_eq!(run(&mut warp, 2, 3, 2), [0, 0, 0]);
        // SAFETY: inside the mapping.
        assert_eq!(unsafe { counters_ptr.cast::<u32>().add(1).read_volatile() }, 0);
        // Pieces 1 and 2 changed, one slot: one of them takes it.
        set(PIECE + 77, content[PIECE + 77] ^ 1);
        set(2 * PIECE + 5, content[2 * PIECE + 5] ^ 1);
        let d = run(&mut warp, 2, 3, 1);
        assert_eq!(d[0], 0);
        let (took, left) = if d[1] == 5 { (1, 2) } else { (2, 1) };
        assert_eq!((d[took], d[left]), (5, 1), "{d:?}");
        assert_eq!(slot(3), piece(took));
        // Both changed again, two slots from 6: both pushed.
        set(PIECE + 78, content[PIECE + 78] ^ 1);
        set(2 * PIECE + 6, content[2 * PIECE + 6] ^ 1);
        let d = run(&mut warp, 2, 6, 2);
        assert_eq!(d[0], 0);
        let mut got = [d[1], d[2]];
        got.sort_unstable();
        assert_eq!(got, [8, 9], "{d:?}");
        assert_eq!(slot(d[1] - 2), piece(1));
        assert_eq!(slot(d[2] - 2), piece(2));
    }

    /// A scoped clamp (`VICTORY_CLAMP_SCOPED`) waits for a dispatch that
    /// wrote its arguments through the same buffer just before it, in the
    /// legacy barriers and the enhanced ones.
    #[test]
    fn warp_scoped_clamp_waits_for_the_argument_write() {
        for enhanced in [false, true] {
            let Some(mut warp) = WarpHarness::new() else {
                return;
            };
            if enhanced && warp.list.cast::<ID3D12GraphicsCommandList7>().is_err() {
                continue;
            }
            const WRITER: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer args : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    args.Store3(0, uint3(5, 1, 1));
}
"#;
            let writer = compile_cs(&warp.device, WRITER).expect("the writer must compile on WARP");
            let (src, src_ptr, src_gpu) = warp.shared_buffer(64);
            let (dst, _, _) = warp.shared_buffer(64);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(64);
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(64);
            // SAFETY: mappings of 64 bytes each.
            unsafe {
                std::ptr::write_bytes(src_ptr, 0, 64);
                std::ptr::write_bytes(budget_ptr, 0, 64);
                std::ptr::write_bytes(hits_ptr, 0, 64);
            }
            let signature = dispatch_signature(&warp.device);
            // SAFETY: an open list, live buffers, our pipelines.
            unsafe {
                warp.list.SetComputeRootSignature(&writer.root);
                warp.list.SetPipelineState(&writer.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, src_gpu);
                warp.list.Dispatch(1, 1, 1);
            }
            assert!(record_clamp_scoped(
                &warp.clamp,
                &warp.list,
                (&dst, 0),
                budget_gpu,
                (&src, 0),
                None,
                (2, 12, 1),
                false,
                true,
                enhanced.then_some(Pre::First),
                false,
                true,
            ));
            // SAFETY: `dst` holds the clamped record for indirect reading.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                warp.list.ExecuteIndirect(&signature, 1, &dst, 0, None, 0);
            }
            if !enhanced {
                done_on(&warp.list, &dst);
            }
            warp.submit_and_wait();
            // SAFETY: the run is over; a mapping of 64 bytes.
            let hits = unsafe { hits_ptr.cast::<u32>().read_volatile() };
            assert_eq!(hits, 5, "the scoped clamp must see the write before it (enhanced: {enhanced})");
        }
    }

    #[test]
    fn warp_scatter_copy_writes_only_its_ranges_bytes() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const SIZE: usize = 128 << 10;
        let pass = crate::scatter::make(&warp.device).expect("scatter shader must compile on WARP");
        let (src, src_ptr, _) = warp.shared_buffer(SIZE as u64);
        let (dst, dst_ptr, _) = warp.shared_buffer(SIZE as u64);
        // Ranges that share words with each other, begin and end inside
        // words, one byte long, and longer than a group's stride many times.
        let ranges = [(1, 2), (3, 7), (9, 10), (11, 12), (13, 300), (301, 302), (2000, 2000 + 70_001), (100_000, 100_004)];
        let (entries, copies) = crate::scatter::split(&ranges, SIZE as u64, u64::MAX);
        assert!(copies.is_empty());
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(entries.len() as u64 * 16);
        // SAFETY: mappings of the sizes made.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(src_ptr, SIZE);
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[3] | 1;
            }
            std::ptr::write_bytes(dst_ptr, 0, SIZE);
            std::ptr::copy_nonoverlapping(entries.as_ptr().cast::<u8>(), table_ptr, entries.len() * 16);
            crate::scatter::record(&warp.list, &pass, &src, &dst, table_gpu, entries.len() as u32);
        }
        warp.submit_and_wait();
        // SAFETY: mappings of the sizes made; the card is done.
        let (src, dst) = unsafe { (std::slice::from_raw_parts(src_ptr, SIZE), std::slice::from_raw_parts(dst_ptr, SIZE)) };
        for i in 0..SIZE {
            let inside = ranges.iter().any(|&(a, b)| (a..b).contains(&i));
            let want = if inside { src[i] } else { 0 };
            assert_eq!(dst[i], want, "byte {i} (in a range: {inside})");
        }
    }

    #[test]
    fn warp_merged_clamps_match_their_own_passes() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        assert!(warp.clamp.merged.is_some(), "the merged clamp must compile on WARP");
        // Four clamps of dispatch records from one source: 70 records (two
        // groups), 1, 3 (one a window slot not filled), and 130 bounded by
        // a count of 100. Each run alone, then all merged; the same records.
        const SRC: usize = 4096;
        let (src, src_ptr, _) = warp.shared_buffer(SRC as u64);
        let (count, count_ptr, _) = warp.shared_buffer(64);
        let specs: [(u64, u32, bool, bool); 4] = [(0, 70, false, false), (1024, 1, false, false), (1040, 3, true, false), (1088, 130, false, true)];
        // SAFETY: mappings of the sizes made.
        unsafe {
            let words = src_ptr.cast::<u32>();
            for i in 0..SRC / 12 {
                let r = [(i as u32 * 7) % 70_000, 1 + (i as u32 % 3), 1];
                std::ptr::copy_nonoverlapping(r.as_ptr(), words.add(i * 3), 3);
            }
            count_ptr.cast::<u32>().write(100);
        }
        let mut results: Vec<Vec<[u32; 3]>> = Vec::new();
        for merge in [false, true] {
            let (dst, dst_ptr, dst_gpu) = warp.shared_buffer(8192);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(64);
            // SAFETY: mappings of the sizes made.
            unsafe {
                std::ptr::write_bytes(dst_ptr, 0xee, 8192);
                std::ptr::write_bytes(budget_ptr, 0, 64);
            }
            let mut at = 0u64;
            let hoisted: Vec<Hoisted> = specs
                .iter()
                .map(|&(offset, records, ahead, counted)| {
                    let place = dst_gpu + at;
                    at += (u64::from(records) * 12).next_multiple_of(256);
                    let bound = counted.then_some((&count, 0u64));
                    let (constants, src_gpu, count_gpu) =
                        clamp_inputs((&src, offset), bound, (2, 12, records), ahead, place, budget_gpu).unwrap();
                    // SAFETY: live buffers.
                    let (src_base, count_base) = unsafe {
                        (src.GetGPUVirtualAddress(), if counted { count.GetGPUVirtualAddress() } else { src_gpu })
                    };
                    Hoisted { constants, src_gpu, count_gpu, dst_gpu: place, budget: budget_gpu, records, src_base, count_base, dst_base: dst_gpu }
                })
                .collect();
            let (_table, table_ptr, table_gpu) = warp.shared_buffer(4096);
            let table = merge.then_some((table_gpu, table_ptr));
            if merge {
                // Two runs: the counted clamp reads another count buffer.
                let runs = merged_runs(&hoisted);
                assert_eq!(runs.iter().map(|r| r.0.clone()).collect::<Vec<_>>(), vec![vec![0, 1, 2], vec![3]]);
            }
            // SAFETY: an open list; live buffers.
            unsafe { record_hoisted_with(&warp.clamp, &warp.list, &hoisted, std::slice::from_ref(&dst), table) };
            warp.submit_and_wait();
            let mut at = 0usize;
            let mut out = Vec::new();
            for &(_, records, _, _) in &specs {
                for r in 0..records as usize {
                    // SAFETY: the run is over; inside the mapping.
                    out.push(unsafe { dst_ptr.add(at + r * 12).cast::<[u32; 3]>().read_volatile() });
                }
                at += (records as usize * 12).next_multiple_of(256);
            }
            results.push(out);
        }
        assert_eq!(results[0], results[1]);
        // The counted clamp's records past the count are left as they were.
        assert_eq!(results[0][70 + 1 + 3 + 100], [0xeeee_eeee; 3]);
        assert_ne!(results[0][0], [0xeeee_eeee; 3]);
    }

    #[test]
    fn warp_hoisted_clamps_run_ahead_into_their_own_places() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let (src, src_ptr, _) = warp.shared_buffer(64);
        let (dst, dst_ptr, dst_gpu) = warp.shared_buffer(1024);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(64);
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(64);
        // Three dispatch records: two that run, one past what a dispatch
        // runs (a window's unfilled slot: empty).
        let records: [[u32; 3]; 3] = [[3, 1, 1], [4, 1, 1], [0x1_0000, 1, 1]];
        // SAFETY: mappings of the sizes made.
        unsafe {
            for (i, r) in records.iter().enumerate() {
                std::ptr::copy_nonoverlapping(r.as_ptr(), src_ptr.cast::<u32>().add(i * 4), 3);
            }
            std::ptr::write_bytes(dst_ptr, 0xff, 1024);
            std::ptr::write_bytes(budget_ptr, 0, 64);
            std::ptr::write_bytes(hits_ptr, 0, 64);
        }
        let hoisted: Vec<Hoisted> = (0..3u64)
            .map(|i| {
                let dst_gpu = dst_gpu + 256 * i;
                let (constants, src_gpu, count_gpu) =
                    clamp_inputs((&src, 16 * i), None, (2, 12, 1), i == 2, dst_gpu, budget_gpu).unwrap();
                Hoisted {
                    constants,
                    src_gpu,
                    count_gpu,
                    dst_gpu,
                    budget: budget_gpu,
                    records: 1,
                    src_base: src_gpu,
                    count_base: count_gpu,
                    dst_base: dst_gpu - 256 * i,
                }
            })
            .collect();
        // The prelude: its own submission ahead of the list.
        // SAFETY: an open list; live buffers.
        unsafe { record_hoisted_with(&warp.clamp, &warp.list, &hoisted, std::slice::from_ref(&dst), None) };
        warp.submit_and_wait();
        let clamped: Vec<[u32; 3]> = (0..3)
            // SAFETY: the run is over; inside the mapping.
            .map(|i| unsafe { dst_ptr.add(i * 256).cast::<[u32; 3]>().read_volatile() })
            .collect();
        assert_eq!(clamped, [[3, 1, 1], [4, 1, 1], [0, 0, 0]]);
        let signature = dispatch_signature(&warp.device);
        // SAFETY: an open list; the clamped records for indirect reading.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            for i in 0..3 {
                warp.list.ExecuteIndirect(&signature, 1, &dst, 256 * i, None, 0);
            }
        }
        warp.submit_and_wait();
        // SAFETY: the run is over; a mapping of 64 bytes.
        let hits = unsafe { hits_ptr.cast::<u32>().read_volatile() };
        assert_eq!(hits, 7, "each call reads its own hoisted place");
    }

    fn dispatch_signature(device: &ID3D12Device) -> ID3D12CommandSignature {
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a one-argument dispatch signature.
        unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature) }.expect("a dispatch signature");
        signature.expect("a dispatch signature")
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// The clamp with enhanced barriers on the scratch buffer (where the
    /// software adapter has them), twice in one list, the second from the
    /// state the first left: the records come out as with the old barriers.
    #[test]
    fn warp_clamp_with_enhanced_barriers() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a plain query into a struct of the asked size.
        let asked = unsafe {
            warp.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut options).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        if asked.is_err() || !options.EnhancedBarriersSupported.as_bool() {
            return;
        }
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 16;
        const RECORDS: u32 = 4;
        let total = (STRIDE * RECORDS) as usize;
        let records: [[u32; 4]; 4] = [[7, 1, 1, 0xA0], [70_000, 1, 1, 0xA1], [0, 1, 1, 0xA2], [3, 9, 9, 0xA3]];
        let src: Vec<u8> = records.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
            std::ptr::write_bytes(budget_ptr, 0, 256);
        }
        let l7: ID3D12GraphicsCommandList7 = warp.list.cast().expect("a list with enhanced barriers");
        for fresh in [true, false] {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                false,
                fresh,
                Some(fresh),
                // The second as a draw of points.
                !fresh,
            ));
        }
        // SAFETY: an open list; live buffers.
        unsafe {
            buffer_barrier(
                &l7,
                &dst_buf,
                (D3D12_BARRIER_SYNC_EXECUTE_INDIRECT, D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT),
                (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE),
            );
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; a mapped buffer.
        let out = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec() };
        assert_eq!(out, [7, 1, 1, 0xA0, 65_535, 1, 1, 0xA1, 0, 1, 1, 0xA2, 3, 1, 1, 0xA3]);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        assert!(warp.clamp.draw.is_some(), "the clamp as a draw must compile on WARP");
        // Each also as a draw of points (`clamp_draws`).
        for (ahead, drawn) in [(true, false), (false, false), (true, true), (false, true)] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
                true,
                None,
                drawn,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    /// A window of dispatch records clamped ahead: filled records come out
    /// as they are and are charged; slots of all ones come out empty and
    /// charge nothing.
    #[test]
    fn warp_dispatch_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 16;
        const RECORDS: u32 = 6;
        let filled = [[3u32, 2, 1], [1, 1, 1]];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            for (w, word) in groups.into_iter().chain([0xCD00 + i as u32]).enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0, total);
        }
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DISPATCH, STRIDE, RECORDS),
            true,
            true,
            None,
            false,
        ));
        done_on(&warp.list, &dst_buf);
        // SAFETY: an open list; the scratch buffer is copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(&dst_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
            warp.list.ResourceBarrier(&[transition(&dst_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
            )
        };
        for (i, groups) in filled.into_iter().enumerate() {
            assert_eq!(out[i * 4..i * 4 + 4], [groups[0], groups[1], groups[2], 0xCD00 + i as u32], "filled record {i}");
        }
        for i in filled.len()..RECORDS as usize {
            assert_eq!(out[i * 4..i * 4 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
        }
        assert_eq!(spent, (6 + 1) * 64, "unfilled slots charge nothing");
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// `HostList::indirect_ring` (`VICTORY_CLAMP_RING`): clamps take places
    /// one after another, 256-byte aligned, in one buffer; at the end the
    /// ring goes round to the start and says so; a reset starts it again.
    #[test]
    fn indirect_ring_places_follow_and_go_round() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let (a, at_a, round_a) = list.indirect_ring(100).unwrap();
        let (b, at_b, round_b) = list.indirect_ring(300).unwrap();
        assert_eq!((at_a, round_a, at_b, round_b), (0, false, 256, false));
        assert_eq!(a.as_raw(), b.as_raw());
        // SAFETY: a live buffer.
        let cap = unsafe { a.GetDesc() }.Width;
        let (_, at_c, round_c) = list.indirect_ring(cap - 512).unwrap();
        assert_eq!((at_c, round_c), (0, true));
        let (_, at_d, round_d) = list.indirect_ring(16).unwrap();
        assert_eq!((at_d, round_d), (cap - 512, false));
        list.close();
        list.reset();
        let (_, at_e, round_e) = list.indirect_ring(16).unwrap();
        assert_eq!((at_e, round_e), (0, false));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let (b1, _) = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64), true).unwrap();
        done(&list, &b1);
        let (b2, _) = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128), true).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let (b3, _) = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000), true).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let (b4, _) = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096), true).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }
}
