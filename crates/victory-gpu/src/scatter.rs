//! One compute pass in place of many small buffer copies.
//!
//! Bringing the copies of title buffers in the card's memory up to date
//! (`Host::card_before`) copied each range the CPU side changed with a
//! `CopyBufferRegion` of its own: about 63,000 a frame in gameplay (zall17d),
//! each a call on the submitting thread. With `VICTORY_SCATTER_COPY=1` the
//! ranges below [`most`] go to one dispatch per buffer instead, which reads
//! a table of (source offset, destination offset, bytes) from upload memory;
//! bigger ranges keep their copy.
//!
//! A range may begin and end inside a 4-byte word: the shader writes those
//! words' own bytes only (an atomic AND then OR, so a range next to it in
//! the same word, done by another group, keeps its bytes). Source and
//! destination offsets of an entry must be equal modulo 4.

use windows::Win32::Graphics::Direct3D12::*;

/// Whether small card copies go to [`record`] (`VICTORY_SCATTER_COPY=1`;
/// off by default until a run shows it).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCATTER_COPY").is_some_and(|v| v == "1"))
}

/// The most bytes of a range that goes to the pass
/// (`VICTORY_SCATTER_MOST_KB`, 64): bigger ones keep their copy.
pub(crate) fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_SCATTER_MOST_KB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64).max(1) << 10
    })
}

/// Bytes of a table entry: source, destination, bytes, unused.
pub(crate) const ENTRY: u64 = 16;

pub(crate) const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0), SRV(t1)"
cbuffer Table : register(b0) {
    uint count;    // entries
};
ByteAddressBuffer from : register(t0);
RWByteAddressBuffer to : register(u0);
ByteAddressBuffer entries : register(t1);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    uint e = group.y * 65535u + group.x;
    if (e >= count) {
        return;
    }
    uint3 entry = entries.Load3(e * 16u);
    uint src = entry.x;
    uint dst = entry.y;
    uint end = dst + entry.z;
    uint first = dst & ~3u;
    for (uint word = first + thread * 4u; word < end; word += 256u) {
        uint lo = max(dst, word) - word;
        uint hi = min(end, word + 4u) - word;
        uint high = hi >= 4u ? 0xffffffffu : ((1u << (hi * 8u)) - 1u);
        uint mask = high & ~((1u << (lo * 8u)) - 1u);
        uint value = from.Load(word - first + (src & ~3u));
        if (mask == 0xffffffffu) {
            to.Store(word, value);
        } else {
            uint ignored;
            to.InterlockedAnd(word, ~mask, ignored);
            to.InterlockedOr(word, value & mask, ignored);
        }
    }
}
"#;

/// The pass's root signature and pipeline.
pub(crate) struct Scatter {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) pso: ID3D12PipelineState,
}

/// Compiles the pass for `device`.
pub(crate) fn make(device: &ID3D12Device) -> windows::core::Result<Scatter> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the calls; the out
    // pointers are ours; the root signature is inside the shader.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            windows::core::s!("scatter copy"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?;
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Scatter { root, pso })
    }
}

/// Splits `ranges` (offsets, the same in source and destination) into
/// those for the pass, as table entries, and those for copies of their
/// own: ranges of at most `most` bytes, in buffers under 4 GiB.
pub(crate) fn split(ranges: &[(usize, usize)], size: u64, most: u64) -> (Vec<[u32; 4]>, Vec<(usize, usize)>) {
    let mut entries = Vec::new();
    let mut copies = Vec::new();
    for &(a, b) in ranges {
        if b > a && (b - a) as u64 <= most && size <= u64::from(u32::MAX) {
            entries.push([a as u32, a as u32, (b - a) as u32, 0]);
        } else {
            copies.push((a, b));
        }
    }
    (entries, copies)
}

/// Records the pass over `count` entries at `table` (a GPU address in
/// upload memory, 16-byte aligned): from `from` (readable as a shader
/// resource) to `to` (unordered access). The caller puts `to` in the
/// unordered-access state (a buffer in the common state is promoted) and
/// waits for its writes after.
///
/// # Safety
/// An open list; live buffers holding every entry's bytes.
pub(crate) unsafe fn record(
    list: &ID3D12GraphicsCommandList,
    pass: &Scatter,
    from: &ID3D12Resource,
    to: &ID3D12Resource,
    table: u64,
    count: u32,
) {
    if count == 0 {
        return;
    }
    // SAFETY: as the caller promises.
    unsafe {
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRoot32BitConstant(0, count, 0);
        list.SetComputeRootShaderResourceView(1, from.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, to.GetGPUVirtualAddress());
        list.SetComputeRootShaderResourceView(3, table);
        list.Dispatch(count.min(65535), count.div_ceil(65535), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_keeps_big_ranges_as_copies() {
        let (entries, copies) = split(&[(0, 3), (5, 5 + 64 * 1024), (70_000, 70_000 + 64 * 1024 + 1)], 1 << 20, 64 << 10);
        assert_eq!(entries, vec![[0, 0, 3, 0], [5, 5, 65536, 0]]);
        assert_eq!(copies, vec![(70_000, 70_000 + 65537)]);
        let (entries, copies) = split(&[(0, 16)], 5 << 30, 64 << 10);
        assert!(entries.is_empty());
        assert_eq!(copies, vec![(0, 16)]);
    }

    #[test]
    fn source_strides_match_the_group() {
        assert!(SOURCE.contains("[numthreads(64, 1, 1)]"));
        assert!(SOURCE.contains("word += 256u"));
        assert_eq!(ENTRY, 16);
    }
}
