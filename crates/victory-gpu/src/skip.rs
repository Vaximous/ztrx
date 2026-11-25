//! Dispatches left out when they would change nothing
//! (`VICTORY_DISPATCH_SKIP`).
//!
//! A dispatch is known by its pipeline, its root arguments, its group
//! counts and the buffers it binds (each view's host address and length).
//! While a dispatch of that kind is followed, the card hashes the bytes of
//! every buffer it binds, through every view, read or written, before it
//! runs and after; it keeps the hashes from after the last run, and
//! whether that run changed nothing. A dispatch whose bytes all hash as
//! they did after a run that changed nothing would find what that run
//! found and write what it wrote: the card gives it no groups
//! (`ExecuteIndirect` with the arguments it picks).
//!
//! `VICTORY_DISPATCH_SKIP_ONLY` (shader hash beginnings, by commas) follows
//! only those shaders.
//!
//! `VICTORY_DISPATCH_SKIP=1` leaves them out; `=check` runs every one and
//! counts, and reports, the runs that were to be left out and changed
//! something all the same (an input the hashes do not see).
//!
//! What the hashes cannot see makes a dispatch one that is not followed:
//! textures, title memory read by address, buffers picked from the heap by
//! a computed index, and more bound bytes than `VICTORY_DISPATCH_SKIP_MB`
//! (64 by default). Constant buffers are hashed as the title's memory holds
//! them at submission (a copy in upload memory). A pipeline whose
//! dispatches changed something in every one of its first followed runs is
//! no longer followed.
//!
//! shortcut: hashes, not the bytes: two contents with one hash (128 bits
//! per 4 KiB) are taken for one.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Skip,
    Check,
}

pub(crate) fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_DISPATCH_SKIP").as_deref() {
        Ok("1") => Mode::Skip,
        Ok("check") => Mode::Check,
        _ => Mode::Off,
    })
}

/// `VICTORY_DISPATCH_SKIP_TRY` (bits, for finding what breaks): 1 runs a
/// followed dispatch directly in the check mode, 2 leaves out the barrier
/// before the hashes.
pub(crate) fn trying(bit: u32) -> bool {
    static TRY: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TRY.get_or_init(|| std::env::var("VICTORY_DISPATCH_SKIP_TRY").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) & bit != 0
}

fn limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_DISPATCH_SKIP_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(64)
            << 20
    })
}

/// Bytes a hash covers.
const PIECE: u64 = 4096;
/// The state's header: dwords, see `SOURCE`.
const HEADER: u64 = 64;
/// Where the header holds the dispatch's arguments.
const ARGS: u64 = 48;
const MAX_KEYS: usize = 2048;
const MAX_KEYS_PER_PIPELINE: u32 = 256;
/// Followed runs of a pipeline without one that changed nothing, after
/// which it is not followed.
const GIVE_UP_RUNS: u32 = 24;

/// Bytes a dispatch reads or writes through one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    gpu: u64,
    len: u64,
    /// Read through a shader resource view (its buffer cannot take
    /// unordered access, or it is upload memory).
    srv: bool,
}

/// What a dispatch binds, gathered while it is bound.
#[derive(Default)]
pub(crate) struct Collect {
    spans: Vec<Span>,
    /// The views by address and length (constant buffers by length: their
    /// bytes are hashed).
    key: Vec<u64>,
    bad: Option<&'static str>,
}

thread_local! {
    static COLLECT: std::cell::RefCell<Option<Collect>> = const { std::cell::RefCell::new(None) };
}

/// From here to [`take`], what is bound is gathered.
pub(crate) fn start() {
    COLLECT.with_borrow_mut(|c| *c = Some(Collect::default()));
}

pub(crate) fn take() -> Option<Collect> {
    COLLECT.with_borrow_mut(Option::take)
}

fn with(f: impl FnOnce(&mut Collect)) {
    COLLECT.with_borrow_mut(|c| {
        if let Some(c) = c {
            f(c);
        }
    });
}

fn collecting() -> bool {
    COLLECT.with_borrow(Option::is_some)
}

/// A buffer view the dispatch reads, or writes (`uav`), in a host copy.
pub(crate) fn note_buffer(shadow: &crate::shadow::Shadow, address: u64, len: u64, uav: bool) {
    if !collecting() {
        return;
    }
    // SAFETY: a live buffer.
    let (desc, start) = unsafe { (shadow.buffer.GetDesc(), shadow.buffer.GetGPUVirtualAddress()) };
    let srv = !uav || desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0;
    let gpu = shadow.gpu(address);
    // Root views have no bounds: no further than the host buffer goes.
    let Some(left) = gpu.checked_sub(start).and_then(|at| desc.Width.checked_sub(at)) else {
        note_bad("a view outside its host copy");
        return;
    };
    let len = len.min(left);
    with(|c| {
        c.key.extend([gpu, len | u64::from(srv) << 62]);
        if len > 0 {
            c.spans.push(Span { gpu, len, srv });
        }
    });
}

/// A constant buffer the dispatch reads: hashed as title memory holds it at
/// submission, from a copy in the list's upload memory.
pub(crate) fn note_constant(list: &crate::host::HostList, address: u64, size: u64) {
    if !collecting() {
        return;
    }
    if size == 0 {
        return;
    }
    let Some((gpu, to)) = list.scratch(size) else {
        note_bad("no upload memory for a constant buffer's copy");
        return;
    };
    list.defer_fill(to, address as *const u8, size as usize);
    // By its bytes alone: the title's constants move from frame to frame
    // (a ring), the bytes much less.
    let _ = address;
    with(|c| {
        c.key.push(size | 1 << 63);
        c.spans.push(Span { gpu, len: size, srv: true });
    });
}

/// Something the hashes cannot see is bound.
pub(crate) fn note_bad(why: &'static str) {
    with(|c| {
        c.bad.get_or_insert(why);
    });
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=12, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) {
    uint pass_;    // 0: hash before; 1: decide; 2: hash after; 3: done
    uint base;     // bytes the view begins past the address bound
    uint size;     // bytes hashed from there (a multiple of 4)
    uint pieces;
    uint piece0;   // the view's first piece among the dispatch's
    uint srv;      // read through t0, else u0
    uint skipping; // leave out (else only count)
    uint total;    // the dispatch's pieces
    uint gx;
    uint gy;
    uint gz;
    uint unused;
};
ByteAddressBuffer rdata : register(t0);
RWByteAddressBuffer wdata : register(u0);
// Dwords: 0 followed once, 1 that run changed nothing, 2 changed since it,
// 3 this run changed something, 4 to be left out, 5 left out, 6 runs,
// 7 runs to be left out that changed something, 8 runs that changed
// nothing, 9 to be left out (counted in both modes); 12..14 the
// arguments. From byte 64: a 16-byte hash per piece after the last run,
// then one per piece before this one.
RWByteAddressBuffer state : register(u1);
uint4 load4(uint at) { return srv != 0u ? rdata.Load4(at) : wdata.Load4(at); }
uint load1(uint at) { return srv != 0u ? rdata.Load(at) : wdata.Load(at); }
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (pass_ == 1u) {
        if (id.x != 0u) return;
        uint would = (state.Load(0) != 0u && state.Load(4) != 0u && state.Load(8) == 0u) ? 1u : 0u;
        uint skip = skipping != 0u ? would : 0u;
        state.Store(16, would);
        state.Store(8, 0u);
        state.Store(20, state.Load(20) + skip);
        state.Store(36, state.Load(36) + would);
        state.Store3(48, skip != 0u ? uint3(0u, 0u, 0u) : uint3(gx, gy, gz));
        return;
    }
    if (pass_ == 3u) {
        if (id.x != 0u) return;
        uint would = state.Load(16);
        if (skipping != 0u && would != 0u) return;
        uint ran = state.Load(12);
        state.Store(24, state.Load(24) + 1u);
        if (ran == 0u) state.Store(32, state.Load(32) + 1u);
        if (would != 0u && ran != 0u) state.Store(28, state.Load(28) + 1u);
        state.Store(4, ran == 0u ? 1u : 0u);
        state.Store(0, 1u);
        state.Store(12, 0u);
        return;
    }
    if (id.x >= pieces) return;
    // Left out: nothing changed; the hashes from before stand.
    if (pass_ == 2u && skipping != 0u && state.Load(16) != 0u) return;
    uint lo = id.x * 4096u;
    uint hi = min(lo + 4096u, size);
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) ^ uint4(hi - lo, 0u, 0u, 0u);
    uint at = lo;
    [loop]
    for (; at + 16u <= hi; at += 16u) {
        h = (h ^ load4(base + at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 4u) {
        h = (h ^ uint4(load1(base + at), 0u, 0u, 0u)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    uint after = 64u + (piece0 + id.x) * 16u;
    uint before = 64u + (total + piece0 + id.x) * 16u;
    if (pass_ == 0u) {
        state.Store4(before, h);
        if (any(h != state.Load4(after))) state.InterlockedOr(8, 1u);
    } else {
        if (any(h != state.Load4(before))) state.InterlockedOr(12, 1u);
        state.Store4(after, h);
    }
}
"#;

struct Hasher {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn hasher() -> Option<&'static Hasher> {
    static HASHER: std::sync::OnceLock<Option<Hasher>> = std::sync::OnceLock::new();
    HASHER
        .get_or_init(|| {
            let made = (|| -> windows::core::Result<Hasher> {
                let device = &crate::host::host().ok_or_else(windows::core::Error::empty)?.device;
                let bytes = crate::host::compile(SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
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
                    Ok(Hasher { root, pso })
                }
            })();
            made.map_err(|e| tracing::warn!(error = %e, "no dispatch skip")).ok()
        })
        .as_ref()
}

/// A view's bytes as a pass reads them: the address bound (on 16 bytes),
/// where they begin past it (on 4 bytes: raw loads are of dwords) and how
/// many (whole dwords).
/// shortcut: bytes of a dword the view only begins or ends inside are not
/// all hashed.
fn bound(span: &Span) -> (u64, u32, u32) {
    let at = span.gpu & !15;
    let base = (span.gpu & 15) & !3;
    (at, base as u32, ((span.len + (span.gpu & 3)) & !3) as u32)
}

/// Whether a pass can hash the span: some bytes, all in reach of 32-bit
/// offsets.
fn hashable(span: &Span) -> bool {
    span.len > 0 && span.len < 1 << 31 && bound(span).2 > 0
}

fn pieces(span: &Span) -> u32 {
    u64::from(bound(span).2).div_ceil(PIECE) as u32
}

/// Where each span's pieces begin, and all of them.
fn layout(spans: &[Span]) -> (Vec<u32>, u32) {
    let mut first = Vec::with_capacity(spans.len());
    let mut total = 0;
    for s in spans {
        first.push(total);
        total += pieces(s);
    }
    (first, total)
}

fn state_size(total: u32) -> u64 {
    HEADER + u64::from(total) * 32
}

/// A small upload buffer, never freed, for a view a pass does not read.
fn unread() -> u64 {
    static UNREAD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *UNREAD.get_or_init(|| {
        crate::host::host().and_then(|h| h.upload_buffer(256)).map_or(0, |(buffer, _)| {
            // SAFETY: a live buffer, kept for the process.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            std::mem::forget(buffer);
            gpu
        })
    })
}

/// Records one pass over `state` (in the unordered access state).
/// `span`: the view hashed and its first piece, for passes 0 and 2; then
/// the dispatch's pieces, whether it is left out and its groups.
///
/// # Safety
/// An open list; `state` and the span's buffer live until it has run.
unsafe fn pass(
    l: &ID3D12GraphicsCommandList,
    h: &Hasher,
    state: &ID3D12Resource,
    which: u32,
    span: Option<(&Span, u32)>,
    (total, skipping, groups): (u32, bool, [u32; 3]),
) {
    // SAFETY: as the caller says.
    unsafe {
        let own = state.GetGPUVirtualAddress();
        let (at, base, size, n, first, srv) = match span {
            Some((s, first)) => {
                let (at, base, size) = bound(s);
                (at, base, size, pieces(s), first, s.srv)
            }
            None => (own, 0, 0, 1, 0, false),
        };
        let constants = [
            which,
            base,
            size,
            n,
            first,
            u32::from(srv),
            u32::from(skipping),
            total,
            groups[0],
            groups[1],
            groups[2],
            0,
        ];
        l.SetComputeRootSignature(&h.root);
        l.SetPipelineState(&h.pso);
        l.SetComputeRoot32BitConstants(0, 12, constants.as_ptr().cast(), 0);
        // Unread, the shader resource view shows a buffer always readable
        // (the state is in the unordered access state).
        l.SetComputeRootShaderResourceView(1, if srv { at } else { unread() });
        l.SetComputeRootUnorderedAccessView(2, if srv { own } else { at });
        l.SetComputeRootUnorderedAccessView(3, own);
        l.Dispatch(n.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[crate::extent::uav_barrier()]);
    }
}

/// Records the hashes before a run and the choice of its groups.
///
/// # Safety
/// As [`pass`]; `state` in the common state, left in the indirect argument
/// state.
unsafe fn record_before(
    l: &ID3D12GraphicsCommandList,
    h: &Hasher,
    state: &ID3D12Resource,
    spans: &[Span],
    skipping: bool,
    groups: [u32; 3],
) {
    let (first, total) = layout(spans);
    // SAFETY: as the caller says.
    unsafe {
        if !trying(2) {
            l.ResourceBarrier(&[crate::extent::uav_barrier()]);
        }
        l.ResourceBarrier(&[crate::host::transition(
            state,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        for (s, &f) in spans.iter().zip(&first) {
            pass(l, h, state, 0, Some((s, f)), (total, skipping, groups));
        }
        pass(l, h, state, 1, None, (total, skipping, groups));
        l.ResourceBarrier(&[crate::host::transition(
            state,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
}

/// Records the hashes after a run and what it learnt; `back`: the header
/// copied into it.
///
/// # Safety
/// As [`pass`]; `state` in the indirect argument state, left in the common
/// state; `back` a read-back buffer of `HEADER` bytes.
unsafe fn record_after(
    l: &ID3D12GraphicsCommandList,
    h: &Hasher,
    state: &ID3D12Resource,
    spans: &[Span],
    skipping: bool,
    back: Option<&ID3D12Resource>,
) {
    let (first, total) = layout(spans);
    // SAFETY: as the caller says.
    unsafe {
        l.ResourceBarrier(&[
            crate::extent::uav_barrier(),
            crate::host::transition(
                state,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ]);
        for (s, &f) in spans.iter().zip(&first) {
            pass(l, h, state, 2, Some((s, f)), (total, skipping, [0; 3]));
        }
        pass(l, h, state, 3, None, (total, skipping, [0; 3]));
        match back {
            Some(back) => {
                l.ResourceBarrier(&[crate::host::transition(
                    state,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyBufferRegion(back, 0, state, 0, HEADER);
                l.ResourceBarrier(&[crate::host::transition(
                    state,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            None => l.ResourceBarrier(&[crate::host::transition(
                state,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]),
        }
    }
}

/// A kind of dispatch followed.
struct Track {
    id: u64,
    state: ID3D12Resource,
    runs: u32,
}

#[derive(Default)]
struct Registry {
    keys: HashMap<Vec<u64>, Track>,
    /// By shader hash: keys made, and whether it is followed no longer.
    pipelines: HashMap<String, (u32, bool)>,
    /// Shader hashes told of (why not followed, or that they are).
    told: std::collections::HashSet<(String, &'static str)>,
    next: u64,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

/// A header on its way back.
struct Back {
    id: u64,
    hash: String,
    data: Box<[u32; 16]>,
    done: Box<AtomicU32>,
    /// Frame-rate lines seen waiting (a list never run is let go).
    age: u32,
}

static BACKS: Mutex<Vec<Back>> = Mutex::new(Vec::new());
/// Per key: its shader and its latest header.
static LAST: Mutex<Option<HashMap<u64, (String, [u32; 16])>>> = Mutex::new(None);
/// Since the last frame-rate line: dispatches followed, not followed.
static SEEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether dispatches of the shader `hash` are followed.
pub(crate) fn wanted(hash: &str) -> bool {
    if mode() == Mode::Off {
        return false;
    }
    // `VICTORY_DISPATCH_SKIP_ONLY`: shader hash beginnings, by commas.
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let only = ONLY.get_or_init(|| {
        std::env::var("VICTORY_DISPATCH_SKIP_ONLY")
            .map(|v| v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect())
            .unwrap_or_default()
    });
    if !only.is_empty() && !only.iter().any(|o| hash.starts_with(o.as_str())) {
        return false;
    }
    let r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    !r.as_ref().and_then(|r| r.pipelines.get(hash)).is_some_and(|p| p.1)
}

/// A dispatch about to run followed.
pub(crate) struct Plan {
    state: ID3D12Resource,
    spans: Vec<Span>,
    back: Option<(Box<[u32; 16]>, Box<AtomicU32>, u64, String)>,
}

impl Plan {
    /// The state holding the arguments, and where.
    pub(crate) fn arguments(&self) -> (&ID3D12Resource, u64) {
        (&self.state, ARGS)
    }
}

/// Before a dispatch of the shader `hash` bound with what `c` gathered:
/// records the hashes and the choice of its groups (replacing the list's
/// compute root signature and pipeline: the caller binds its own again and
/// runs it from [`Plan::arguments`]). `key`: the pipeline, root arguments
/// and groups, hashed. None if not followed.
pub(crate) fn before(list: &crate::host::HostList, c: Collect, key: u64, groups: [u32; 3], hash: &str) -> Option<Plan> {
    let mut r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let r = r.get_or_insert_with(Registry::default);
    let tell = |r: &mut Registry, why: &'static str| {
        if r.told.insert((hash.to_owned(), why)) {
            tracing::info!(hash, why, "dispatch skip: a shader not followed");
        }
    };
    if let Some(why) = c.bad {
        tell(r, why);
        SEEN[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let bytes: u64 = c.spans.iter().map(|s| s.len).sum();
    if c.spans.is_empty() {
        tell(r, "binds no buffers");
        SEEN[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if bytes > limit() {
        tell(r, "binds more bytes than VICTORY_DISPATCH_SKIP_MB");
        SEEN[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let h = hasher()?;
    let host = crate::host::host()?;
    let Collect { spans, key: views, .. } = c;
    // Nothing a pass would loop on: every span has whole dwords.
    if !spans.iter().all(hashable) {
        tell(r, "a view of less than a dword");
        SEEN[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let mut k = Vec::with_capacity(1 + views.len());
    k.push(key);
    k.extend(views);
    let (_, total) = layout(&spans);
    let (made, gone) = *r.pipelines.get(hash).unwrap_or(&(0, false));
    if gone {
        return None;
    }
    let track = match r.keys.get_mut(&k) {
        Some(t) => t,
        None => {
            if r.keys.len() >= MAX_KEYS || made >= MAX_KEYS_PER_PIPELINE {
                tell(r, "too many kinds of its dispatches");
                SEEN[1].fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let state = host.uav_buffer(state_size(total))?;
            crate::host::name(&state, || format!("dispatch skip state of {hash}"));
            r.pipelines.insert(hash.to_owned(), (made + 1, false));
            if made == 0 {
                tracing::info!(hash, views = spans.len(), bytes, "dispatch skip: a shader followed");
            }
            r.next += 1;
            let id = r.next;
            r.keys.entry(k).or_insert(Track { id, state, runs: 0 })
        }
    };
    track.runs += 1;
    let back = (track.runs <= 8 || track.runs.is_multiple_of(32)).then(|| {
        (
            Box::new([0u32; 16]),
            Box::new(AtomicU32::new(0)),
            track.id,
            hash.to_owned(),
        )
    });
    let state = track.state.clone();
    SEEN[0].fetch_add(1, Ordering::Relaxed);
    // SAFETY: an open list; the state and the views' buffers live with it.
    unsafe { record_before(&list.list, h, &state, &spans, mode() == Mode::Skip, groups) };
    list.keep(&state);
    Some(Plan { state, spans, back })
}

/// After the dispatch: hashes again and learns (replacing the compute
/// root signature and pipeline).
pub(crate) fn after(list: &crate::host::HostList, plan: Plan) {
    let (Some(h), Some(host)) = (hasher(), crate::host::host()) else {
        return;
    };
    let Plan { state, spans, back } = plan;
    let back = back.and_then(|b| Some((host.read_back_buffer(HEADER)?, b)));
    // SAFETY: as in `before`.
    unsafe {
        record_after(&list.list, h, &state, &spans, mode() == Mode::Skip, back.as_ref().map(|b| &b.0));
    }
    if let Some((buffer, (mut data, done, id, hash))) = back {
        list.defer_read_back(buffer, data.as_mut_ptr().cast(), [HEADER as usize, 1, HEADER as usize]);
        list.defer_write(done.as_ptr().cast(), crate::queue::Written::Value(1), 4);
        BACKS.lock().unwrap_or_else(|e| e.into_inner()).push(Back { id, hash, data, done, age: 0 });
    }
}

/// Sums of the latest headers of a shader's keys: runs, runs that changed
/// nothing, left out, to be left out, wrong.
fn sums<'a>(last: impl Iterator<Item = &'a [u32; 16]>) -> [u64; 5] {
    let mut s = [0u64; 5];
    for d in last {
        for (to, at) in s.iter_mut().zip([6, 8, 5, 9, 7]) {
            *to += u64::from(d[at]);
        }
    }
    s
}

/// For the frame-rate line: what came back, and pipelines given up.
pub(crate) fn line() {
    if mode() == Mode::Off {
        return;
    }
    let mut backs = BACKS.lock().unwrap_or_else(|e| e.into_inner());
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let last = last.get_or_insert_with(HashMap::new);
    backs.retain_mut(|b| {
        if b.done.load(Ordering::Acquire) == 0 {
            b.age += 1;
            return b.age < 600;
        }
        let was = last.get(&b.id).map_or(0, |l| l.1[7]);
        if b.data[7] > was {
            tracing::warn!(
                hash = b.hash,
                wrong = b.data[7],
                runs = b.data[6],
                "dispatch skip: a run to be left out changed something"
            );
        }
        last.insert(b.id, (b.hash.clone(), *b.data));
        false
    });
    drop(backs);
    let mut by: HashMap<&str, Vec<&[u32; 16]>> = HashMap::new();
    for (hash, d) in last.values() {
        by.entry(hash.as_str()).or_default().push(d);
    }
    let mut r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let r = r.get_or_insert_with(Registry::default);
    let mut gone = Vec::new();
    let mut listed = Vec::new();
    for (hash, ds) in &by {
        let s = sums(ds.iter().copied());
        if s[0] >= u64::from(GIVE_UP_RUNS) && s[1] == 0 && s[3] == 0 {
            gone.push((*hash).to_owned());
        }
        if s[3] > 0 {
            listed.push(((*hash).to_owned(), s));
        }
    }
    for hash in &gone {
        if let Some(p) = r.pipelines.get_mut(hash)
            && !p.1
        {
            p.1 = true;
            tracing::info!(hash, "dispatch skip: a shader whose followed runs all changed something, no longer followed");
        }
    }
    let ids: std::collections::HashSet<u64> = {
        let given: std::collections::HashSet<&str> =
            r.pipelines.iter().filter(|p| p.1.1).map(|p| p.0.as_str()).collect();
        last.iter().filter(|l| given.contains(l.1.0.as_str())).map(|l| *l.0).collect()
    };
    r.keys.retain(|_, t| !ids.contains(&t.id));
    let all = sums(last.values().map(|l| &l.1));
    listed.sort_by_key(|l| std::cmp::Reverse(l.1[3]));
    let top: Vec<String> = listed
        .iter()
        .take(6)
        .map(|(h, s)| format!("{}: {} of {} runs to be left out, {} left out, {} wrong", &h[..h.len().min(16)], s[3], s[0] + s[2], s[2], s[4]))
        .collect();
    tracing::info!(
        "dispatch skip ({:?}): {} kinds followed, {} shaders no longer; since the last line {} dispatches followed, {} not; \
         sampled totals {} runs ({} changed nothing), {} to be left out, {} left out, {} wrong; {top:?}",
        mode(),
        r.keys.len(),
        r.pipelines.values().filter(|p| p.1).count(),
        SEEN[0].swap(0, Ordering::Relaxed),
        SEEN[1].swap(0, Ordering::Relaxed),
        all[0],
        all[1],
        all[3],
        all[2],
        all[4],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    #[test]
    fn the_hasher_compiles() {
        let made = crate::host::compile(SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1"));
        assert!(made.is_ok_and(|b| !b.is_empty()));
    }

    #[test]
    fn views_are_hashed_in_whole_dwords_from_16_bytes() {
        let s = Span { gpu: 0x1_0008, len: 4096 * 2 + 2, srv: false };
        assert_eq!(bound(&s), (0x1_0000, 8, 4096 * 2));
        // Not on 4 bytes: from the dword it begins in.
        let u = Span { gpu: 0x1_0006, len: 7, srv: false };
        assert_eq!(bound(&u), (0x1_0000, 4, 8));
        assert!(!hashable(&Span { gpu: 0x1_0000, len: 3, srv: false }));
        assert!(!hashable(&Span { gpu: 0x1_0000, len: 0, srv: false }));
        assert!(hashable(&Span { gpu: 0x1_0001, len: 3, srv: false }));
        let s = Span { gpu: 0x1_0008, len: 4096 * 2 + 8, srv: false };
        assert_eq!(bound(&s), (0x1_0000, 8, 4096 * 2 + 8));
        assert_eq!(pieces(&s), 3);
        let t = Span { gpu: 0x2_0000, len: 16, srv: true };
        assert_eq!(layout(&[s, t]), (vec![0, 3], 4));
        assert_eq!(state_size(4), 64 + 4 * 32);
    }

    /// Runs the passes on the card over a buffer of 3 pieces: a run that
    /// changed nothing, then one of the same bytes is to be left out; bytes
    /// changed before, it is not; a run to be left out that changes
    /// something is counted wrong.
    #[test]
    fn the_card_leaves_out_a_run_after_one_that_changed_nothing() {
        run_steps(4096 * 2 + 64, |gpu, len| vec![Span { gpu, len, srv: false }]);
    }

    /// As above, over views of a few bytes, not on 16 bytes or 4, and one
    /// reaching a piece past a 4 KiB one.
    #[test]
    fn small_and_unaligned_views_are_hashed_and_left_out() {
        run_steps(4096 * 2 + 64, |gpu, _| {
            vec![
                Span { gpu: gpu + 4, len: 4, srv: false },
                Span { gpu: gpu + 0x26, len: 13, srv: false },
                Span { gpu: gpu + 0x40, len: 4097, srv: false },
            ]
        });
    }

    fn run_steps(len: u64, spans: impl Fn(u64, u64) -> Vec<Span>) {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(h) = hasher() else {
            panic!("no hasher");
        };
        let device = &host.device;
        let data = host.uav_buffer(len).unwrap();
        // SAFETY: a live buffer.
        let spans = spans(unsafe { data.GetGPUVirtualAddress() }, len);
        assert!(spans.iter().all(hashable));
        let state = host.uav_buffer(state_size(layout(&spans).1)).unwrap();
        let (upload, at) = host.upload_buffer(len).unwrap();
        let back = host.read_back_buffer(HEADER).unwrap();
        // SAFETY: plain objects; each list waited for before the next.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut value = 0;
            // One list: bytes changed first (`before`), a run that changes
            // them (`during`), as the skip mode or the check; the header.
            let mut run = |before: Option<u8>, during: Option<u8>, skipping: bool| -> [u32; 16] {
                allocator.Reset().unwrap();
                let l: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                let write = |byte: u8| {
                    std::ptr::write_bytes(at, byte, len as usize);
                    l.ResourceBarrier(&[crate::host::transition(
                        &data,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    l.CopyBufferRegion(&data, 0, &upload, 0, len);
                    l.ResourceBarrier(&[crate::host::transition(
                        &data,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                };
                if let Some(b) = before {
                    write(b);
                }
                l.ResourceBarrier(&[crate::host::transition(
                    &data,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                record_before(&l, h, &state, &spans, skipping, [5, 1, 1]);
                l.ResourceBarrier(&[crate::host::transition(
                    &data,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
                if let Some(b) = during {
                    write(b);
                }
                l.ResourceBarrier(&[crate::host::transition(
                    &data,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                record_after(&l, h, &state, &spans, skipping, Some(&back));
                l.ResourceBarrier(&[crate::host::transition(
                    &data,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
                l.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
                value += 1;
                queue.Signal(&fence, value).unwrap();
                while fence.GetCompletedValue() < value {
                    std::thread::yield_now();
                }
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                let header = *(p as *const [u32; 16]);
                back.Unmap(0, None);
                header
            };
            // Dwords: 4 to be left out, 5 left out, 6 runs, 7 wrong, 8
            // runs that changed nothing; 12..14 the groups.
            let d = run(Some(1), None, false);
            assert_eq!((d[4], d[6], d[8], d[12]), (0, 1, 1, 5));
            let d = run(None, None, false);
            assert_eq!((d[4], d[5], d[6], d[7], d[12]), (1, 0, 2, 0, 5));
            let d = run(None, None, true);
            assert_eq!((d[4], d[5], d[6], d[12]), (1, 1, 2, 0));
            let d = run(Some(2), None, true);
            assert_eq!((d[4], d[5], d[6], d[12]), (0, 1, 3, 5));
            let d = run(None, Some(3), false);
            assert_eq!((d[4], d[6], d[7], d[8]), (1, 4, 1, 3));
            let d = run(None, None, false);
            assert_eq!((d[4], d[6], d[7]), (0, 5, 1));
        }
    }
}
