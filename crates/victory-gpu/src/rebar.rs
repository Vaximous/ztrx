//! Probes for a memory model without per-frame copies: title memory that
//! lives in the card's own memory, which the CPU maps through the
//! resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`), and what the kernel
//! calls that the copies' change tracking makes cost when several threads
//! make them at once. Only tests (`--ignored`): nothing here runs in the
//! emulator.
//!
//! `cargo test -p victory-gpu --lib rebar -- --ignored --nocapture --test-threads=1`
//!
//! - `card_gpu_upload_heaps` opens the first hardware adapter and asks
//!   what it has: whether the runtime and driver give upload heaps in the
//!   card's memory, where their CPU mappings land (the title wants its
//!   addresses below 2^40), how fast the CPU writes and reads them and the
//!   card reads them, whether shaders may write them, and whether such a
//!   heap can be put at an address of our choice (a shared handle mapped
//!   as a section into a placeholder): the crux of backing title memory
//!   with it.
//! - `kernel_calls_scale` measures `GetWriteWatch` (with reset),
//!   `VirtualProtect` on 64 KiB views of a reserved section (as `xmem`
//!   maps the page pool) and `VirtualQuery`, from 1 to 16 threads on
//!   ranges of their own.
//!
//! What they found on this PC (RTX 4070 Ti, resizable BAR on, the
//! system's Direct3D runtime with no Agility SDK, 2026-11-25):
//!
//! - GPU upload heaps are there (`GPUUploadHeapSupported`), and so is the
//!   older custom heap in the card's pool with write-combined CPU pages.
//! - A mapping lands where the driver puts it, around 2.5 TiB (0x288...),
//!   above the 2^40 the title's GPU addresses must stay under, one after
//!   another in a range of the driver's. It is private memory to the
//!   memory manager, write-combined (protection 0x404); `VirtualProtect`
//!   on it is refused (error 87), so neither guards nor write watching
//!   can follow it. A shared GPU upload heap is refused, so there is no
//!   handle to map as a section at an address of ours.
//! - The CPU writes it at 22 GB/s, one thread or eight alike (the PC's
//!   write-back memory: 26 GB/s) and reads it at 54 MB/s (write-back:
//!   10.5 GB/s). The card copies out of it at 237 GB/s, as out of its
//!   default memory (221 GB/s), against 27 GB/s out of the PC's memory.
//!   Shaders may write it, and the CPU sees their stores with no call.
//! - nvidia-smi's BAR1 "used" stays at 16,355 of 16,384 MiB with the
//!   emulator stopped and after 320 MB of these heaps: the driver maps the
//!   whole aperture itself; the figure says nothing of who uses it.
//! - `GetWriteWatch` (with reset) scales badly over threads: 1.2x on 2,
//!   1.7-2x on 4, 2.2-2.6x on 8, 2.9-3x on 16. `VirtualProtect` on 64 KiB
//!   views of a reserved section: 1.6x on 2, 3.9x on 8, 4.5x on 16; on
//!   private memory and `VirtualQuery` nearly linear (9.6x and 8.9x on
//!   16).
//! - `WriteProcessMemory` into our own page that is no-access or read-only
//!   fails (error 998): the kernel cannot merge into a protected page for
//!   us.

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::time::Instant;

    use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory as mem;

    const MB: usize = 1 << 20;
    const K64: usize = 64 << 10;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    /// `GetWriteWatch` resets what it reports.
    const WRITE_WATCH_FLAG_RESET: u32 = 1;

    fn hardware() -> Option<(ID3D12Device, String)> {
        // SAFETY: plain DXGI queries and device creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            for i in 0.. {
                let adapter: IDXGIAdapter1 = factory.EnumAdapters1(i).ok()?;
                let desc = adapter.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                    continue;
                }
                let name = String::from_utf16_lossy(&desc.Description);
                return Some((device?, name.trim_end_matches('\0').to_owned()));
            }
            None
        }
    }

    fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        }
    }

    fn committed(
        device: &ID3D12Device,
        heap: D3D12_HEAP_PROPERTIES,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, flags), state, None, &mut buffer)? };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    fn kind(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
        D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }
    }

    fn map(resource: &ID3D12Resource) -> windows::core::Result<*mut u8> {
        let mut data = std::ptr::null_mut();
        // SAFETY: a CPU-visible buffer of ours.
        unsafe { resource.Map(0, None, Some(&mut data))? };
        Ok(data.cast())
    }

    /// A queue of the test's own, with GPU timestamps.
    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        done: u64,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        back_at: *mut u8,
        frequency: u64,
    }

    impl Rig {
        fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
            // SAFETY: plain creation of objects of our own.
            unsafe {
                let queue: ID3D12CommandQueue =
                    device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 },
                    &mut stamps,
                )?;
                let back = committed(device, kind(D3D12_HEAP_TYPE_READBACK), 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                let back_at = map(&back)?;
                let frequency = queue.GetTimestampFrequency()?;
                Ok(Rig {
                    device: device.clone(),
                    queue,
                    allocator,
                    list,
                    fence,
                    done: 0,
                    stamps: stamps.ok_or_else(windows::core::Error::empty)?,
                    back,
                    back_at,
                    frequency,
                })
            }
        }

        /// Records with `record` between two timestamps, runs it, waits,
        /// and gives the card's milliseconds.
        fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<f64> {
            // SAFETY: our own list and queue, one run at a time.
            unsafe {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                self.list.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                record(&self.list);
                self.list.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                self.list.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                self.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
                self.done += 1;
                self.queue.Signal(&self.fence, self.done)?;
                self.fence.SetEventOnCompletion(self.done, HANDLE::default())?;
                self.device.GetDeviceRemovedReason()?;
                let t = self.back_at.cast::<u64>();
                let ticks = t.add(1).read_volatile().saturating_sub(t.read_volatile());
                Ok(ticks as f64 * 1000.0 / self.frequency as f64)
            }
        }

        /// The card's milliseconds to copy `bytes` from `from` to `to`, the
        /// best of five.
        fn copy_ms(&mut self, to: &ID3D12Resource, from: &ID3D12Resource, bytes: u64) -> windows::core::Result<f64> {
            let mut best = f64::MAX;
            for _ in 0..5 {
                // SAFETY: two live buffers of `bytes` bytes or more.
                best = best.min(self.run(|list| unsafe { list.CopyBufferRegion(to, 0, from, 0, bytes) })?);
            }
            Ok(best)
        }

        /// A compute shader stores, through a root unordered-access view at
        /// `gpu`, the sum of the buffer's first two words in its third.
        fn shade(&mut self, gpu: u64) -> windows::core::Result<f64> {
            use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
            use windows::Win32::Graphics::Direct3D::ID3DBlob;
            use windows::core::s;
            const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(1, 1, 1)]\nvoid main() { b.Store(8, b.Load(0) + b.Load(4)); }\n";
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; then the
            // blob's own bytes, which hold the root signature too.
            let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
                D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("rebar"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
                let code = code.ok_or_else(windows::core::Error::empty)?;
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })?;
                (root, pso)
            };
            // SAFETY: an open list of ours; the view names a live buffer.
            self.run(|list| unsafe {
                list.SetComputeRootSignature(&root);
                list.SetPipelineState(&pso);
                list.SetComputeRootUnorderedAccessView(0, gpu);
                list.Dispatch(1, 1, 1);
            })
        }
    }

    fn gbps(bytes: usize, ms: f64) -> f64 {
        bytes as f64 / 1e6 / ms
    }

    /// CPU writes of `len` bytes at `at` from `threads` threads, best of
    /// three, in GB/s.
    fn cpu_write(at: *mut u8, len: usize, threads: usize) -> f64 {
        let source = vec![0x5au8; len];
        let at = at as usize;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = Instant::now();
            std::thread::scope(|s| {
                let part = len / threads;
                for t in 0..threads {
                    let source = &source;
                    s.spawn(move || {
                        // SAFETY: the caller's mapping of `len` bytes; each
                        // thread its own part.
                        unsafe { std::ptr::copy_nonoverlapping(source.as_ptr().add(t * part), (at as *mut u8).add(t * part), part) };
                    });
                }
            });
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        gbps(len, best)
    }

    /// CPU reads of `len` bytes at `at`, in MB/s (one thread).
    fn cpu_read(at: *const u8, len: usize) -> f64 {
        let mut into = vec![0u8; len];
        let started = Instant::now();
        // SAFETY: the caller's mapping of `len` bytes.
        unsafe { std::ptr::copy_nonoverlapping(at, into.as_mut_ptr(), len) };
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&into);
        len as f64 / 1e3 / ms
    }

    fn region(at: *const c_void) -> String {
        let mut info = mem::MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query that fills our structure.
        let n = unsafe { mem::VirtualQuery(at, &mut info, size_of::<mem::MEMORY_BASIC_INFORMATION>()) };
        if n == 0 {
            return "VirtualQuery failed".into();
        }
        let kind = match info.Type {
            mem::MEM_PRIVATE => "private",
            mem::MEM_MAPPED => "mapped (a section's view)",
            mem::MEM_IMAGE => "image",
            _ => "?",
        };
        format!(
            "{kind}, allocation at {:#x}, region {:#x}+{:#x}, protection {:#x}",
            info.AllocationBase as usize, info.BaseAddress as usize, info.RegionSize, info.Protect
        )
    }

    fn bar1() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .map(|vram| {
                let bar = std::process::Command::new("nvidia-smi")
                    .args(["-q", "-d", "MEMORY"])
                    .output()
                    .ok()
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                    .unwrap_or_default();
                let used = bar
                    .split("BAR1 Memory Usage")
                    .nth(1)
                    .and_then(|s| s.lines().find(|l| l.contains("Used")))
                    .map(|l| l.trim().to_owned())
                    .unwrap_or_default();
                format!("VRAM {vram}; BAR1 {used}")
            })
            .unwrap_or_else(|| "no nvidia-smi".into())
    }

    fn hr<T>(r: &windows::core::Result<T>) -> String {
        match r {
            Ok(_) => "ok".into(),
            Err(e) => format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim()),
        }
    }

    /// What this PC's card and runtime make of title memory in the card's
    /// own memory. Prints its findings; fails only if no card answers.
    #[test]
    #[ignore = "opens the card: cargo test -p victory-gpu --lib rebar -- --ignored --nocapture --test-threads=1"]
    fn card_gpu_upload_heaps() {
        let (device, name) = hardware().expect("a hardware adapter");
        println!("adapter: {name}");
        println!("before: {}", bar1());
        // SAFETY: plain queries that fill our structures.
        let (o16, arch, o) = unsafe {
            let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            let r16 = device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut o16).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
            );
            let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
            let _ = device.CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut arch).cast(),
                size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
            );
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            let _ = device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut o).cast(), size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32);
            println!("OPTIONS16 query: {}", hr(&r16));
            (o16, arch, o)
        };
        println!(
            "GPUUploadHeapSupported {}; UMA {}, CacheCoherentUMA {}, IsolatedMMU {}; resource heap tier {}",
            o16.GPUUploadHeapSupported.as_bool(),
            arch.UMA.as_bool(),
            arch.CacheCoherentUMA.as_bool(),
            arch.IsolatedMMU.as_bool(),
            o.ResourceHeapTier.0
        );
        // The older way to CPU-visible card memory: a custom heap in the
        // card's pool with write-combined pages (refused on discrete cards
        // before GPU upload heaps).
        let l1 = committed(
            &device,
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L1,
                ..Default::default()
            },
            K64 as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        println!("custom heap, pool L1, write-combined CPU pages: {}", hr(&l1));
        if !o16.GPUUploadHeapSupported.as_bool() {
            println!("no GPU upload heaps: nothing more to try");
            return;
        }
        let mut rig = Rig::new(&device).expect("a queue");
        const SIZE: usize = 256 * MB;
        let up = committed(
            &device,
            kind(D3D12_HEAP_TYPE_GPU_UPLOAD),
            SIZE as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        println!("a GPU upload buffer of {} MB (unordered access): {}", SIZE / MB, hr(&up));
        let Ok(up) = up else { return };
        // SAFETY: a plain query of our heap's properties.
        let props = unsafe {
            let mut props = D3D12_HEAP_PROPERTIES::default();
            let mut flags = D3D12_HEAP_FLAGS::default();
            let _ = up.GetHeapProperties(Some(&mut props), Some(&mut flags));
            (props, flags)
        };
        println!("  heap type {} CPU pages {} pool {} flags {:#x}", props.0.Type.0, props.0.CPUPageProperty.0, props.0.MemoryPoolPreference.0, props.1.0);
        let at = map(&up).expect("mapped");
        println!("  mapped at {:p}: {} (title memory must lie in [2^32, 2^40))", at, region(at.cast()));
        println!("  after: {}", bar1());
        // A few more, to see where the driver puts its mappings.
        let more: Vec<(ID3D12Resource, *mut u8)> = (0..4)
            .filter_map(|_| {
                let b = committed(&device, kind(D3D12_HEAP_TYPE_GPU_UPLOAD), 16 * MB as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON).ok()?;
                let p = map(&b).ok()?;
                Some((b, p))
            })
            .collect();
        println!("  four more of 16 MB mapped at {:?}", more.iter().map(|m| m.1).collect::<Vec<_>>());
        for threads in [1, 4, 8] {
            println!("  CPU writes, {threads} thread(s): {:.1} GB/s", cpu_write(at, SIZE, threads));
        }
        println!("  CPU reads (4 MB, one thread): {:.0} MB/s", cpu_read(at, 4 * MB));
        // What the card reads it at, against the other kinds.
        let card = committed(&device, kind(D3D12_HEAP_TYPE_DEFAULT), SIZE as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON).expect("card");
        let card2 = committed(&device, kind(D3D12_HEAP_TYPE_DEFAULT), SIZE as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON).expect("card");
        let upload = committed(&device, kind(D3D12_HEAP_TYPE_UPLOAD), SIZE as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ).expect("upload");
        let shared = committed(
            &device,
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            SIZE as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shared");
        if let Ok(p) = map(&upload) {
            println!("  for comparison, an upload buffer (the PC's memory, write-combined): CPU writes 1 thread {:.1} GB/s", cpu_write(p, SIZE, 1));
        }
        if let Ok(p) = map(&shared) {
            println!(
                "  for comparison, a shared buffer (the PC's memory, write-back, today's host copies): CPU writes 1 thread {:.1} GB/s, reads {:.0} MB/s",
                cpu_write(p, SIZE, 1),
                cpu_read(p, 4 * MB)
            );
        }
        for (what, from) in [
            ("GPU upload (card memory, CPU-mapped)", &up),
            ("default (card memory)", &card2),
            ("upload (PC memory, write-combined)", &upload),
            ("shared (PC memory, write-back)", &shared),
        ] {
            match rig.copy_ms(&card, from, SIZE as u64) {
                Ok(ms) => println!("  the card copies {} MB out of {what} in {ms:.2} ms: {:.0} GB/s", SIZE / MB, gbps(SIZE, ms)),
                Err(e) => println!("  the card copying out of {what}: {e}"),
            }
        }
        // Shaders write the GPU upload buffer; the CPU reads the result
        // with no call in between (it is CPU-visible card memory).
        // SAFETY: our mapping, at least 12 bytes.
        unsafe {
            let w = at.cast::<u32>();
            w.write_volatile(40);
            w.add(1).write_volatile(2);
            w.add(2).write_volatile(0);
        }
        // SAFETY: the buffer is alive.
        let gpu = unsafe { up.GetGPUVirtualAddress() };
        let shaded = rig.shade(gpu);
        // SAFETY: as above.
        let sum = unsafe { at.cast::<u32>().add(2).read_volatile() };
        println!("  a shader's store through unordered access: {}; the CPU reads {sum} (40 + 2 wanted)", hr(&shaded));
        // The crux: can such a heap be shown at an address of our choice?
        // A shared heap gives an NT handle; is it a section the memory
        // manager maps?
        let heap: windows::core::Result<ID3D12Heap> = {
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: plain creation.
            unsafe {
                device
                    .CreateHeap(
                        &D3D12_HEAP_DESC {
                            SizeInBytes: 16 * MB as u64,
                            Properties: kind(D3D12_HEAP_TYPE_GPU_UPLOAD),
                            Alignment: 0,
                            Flags: D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
                        },
                        &mut heap,
                    )
                    .and_then(|()| heap.ok_or_else(windows::core::Error::empty))
            }
        };
        println!("  a shared GPU upload heap of 16 MB: {}", hr(&heap));
        if let Ok(heap) = heap {
            // SAFETY: our heap; the handle is ours to close.
            let handle = unsafe { device.CreateSharedHandle(&heap, None, GENERIC_ALL.0, None) };
            println!("  its shared handle: {}", hr(&handle));
            if let Ok(handle) = handle {
                // SAFETY: a mapping attempt of a handle we own; failures
                // change nothing.
                unsafe {
                    let view = mem::MapViewOfFile(handle.0, mem::FILE_MAP_ALL_ACCESS, 0, 0, 0);
                    println!(
                        "  MapViewOfFile on it: {}",
                        if view.Value.is_null() {
                            format!("refused ({})", std::io::Error::last_os_error())
                        } else {
                            format!("mapped at {:p}: {}", view.Value, region(view.Value))
                        }
                    );
                    if !view.Value.is_null() {
                        mem::UnmapViewOfFile(view);
                    }
                    let place = mem::VirtualAlloc2(
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        16 * MB,
                        mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                        mem::PAGE_NOACCESS,
                        std::ptr::null_mut(),
                        0,
                    );
                    let view = mem::MapViewOfFile3(
                        handle.0,
                        std::ptr::null_mut(),
                        place,
                        0,
                        16 * MB,
                        mem::MEM_REPLACE_PLACEHOLDER,
                        mem::PAGE_READWRITE,
                        std::ptr::null_mut(),
                        0,
                    );
                    println!(
                        "  MapViewOfFile3 into a placeholder of ours: {}",
                        if view.Value.is_null() {
                            format!("refused ({})", std::io::Error::last_os_error())
                        } else {
                            format!("mapped at {:p}", view.Value)
                        }
                    );
                    if view.Value.is_null() {
                        mem::VirtualFree(place, 0, mem::MEM_RELEASE);
                    } else {
                        mem::UnmapViewOfFile(view);
                    }
                    let _ = CloseHandle(handle);
                }
            }
        }
        // Can the tracking we do today work on the mapping (a guard by
        // protection, as on views of the page pool)?
        let mut old = 0;
        // SAFETY: our own mapping; only its protection changes, and back.
        let protect = unsafe { mem::VirtualProtect(at.cast(), K64, mem::PAGE_READONLY, &mut old) };
        println!(
            "  VirtualProtect read-only on the mapping: {}",
            if protect == 0 { format!("refused ({})", std::io::Error::last_os_error()) } else { format!("ok (was {old:#x})") }
        );
        if protect != 0 {
            // SAFETY: as above.
            unsafe { mem::VirtualProtect(at.cast(), K64, old, &mut old) };
        }
        drop(more);
        println!("end: {}", bar1());
    }

    /// For a lazy copy-back into private title memory without a moment in
    /// which another thread sees the page open and not yet merged: whether
    /// the kernel writes into our own page that is `PAGE_NOACCESS` or
    /// read-only (`WriteProcessMemory` on our own process) and leaves its
    /// protection as it was.
    #[test]
    #[ignore = "a probe of the kernel: cargo test -p victory-gpu --lib rebar -- --ignored --nocapture"]
    fn self_write_into_protected_page() {
        use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        for (what, protection) in [("no access", mem::PAGE_NOACCESS), ("read-only", mem::PAGE_READONLY)] {
            // SAFETY: a page of our own; its protection changes, and the
            // kernel is asked to write it.
            unsafe {
                let p = mem::VirtualAlloc(std::ptr::null(), K64, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE);
                assert!(!p.is_null());
                let mut old = 0;
                mem::VirtualProtect(p, K64, protection, &mut old);
                let bytes = [7u8; 64];
                let mut written = 0;
                let ok = WriteProcessMemory(GetCurrentProcess(), p, bytes.as_ptr().cast(), bytes.len(), &mut written);
                let error = std::io::Error::last_os_error();
                let mut info = mem::MEMORY_BASIC_INFORMATION::default();
                mem::VirtualQuery(p, &mut info, size_of::<mem::MEMORY_BASIC_INFORMATION>());
                mem::VirtualProtect(p, K64, mem::PAGE_READWRITE, &mut old);
                let landed = p.cast::<u8>().read_volatile() == 7;
                let mut found = [std::ptr::null_mut::<c_void>(); 16];
                let (mut n, mut granularity) = (found.len(), 0);
                mem::GetWriteWatch(WRITE_WATCH_FLAG_RESET, p, K64, found.as_mut_ptr(), &mut n, &mut granularity);
                println!(
                    "WriteProcessMemory into our own {what} page: {} ({written} bytes{}); protection afterwards {:#x}; bytes there {landed}; write watch saw {n} page(s)",
                    if ok != 0 { "ok" } else { "failed" },
                    if ok != 0 { String::new() } else { format!(", {error}") },
                    info.Protect
                );
                mem::VirtualFree(p, 0, mem::MEM_RELEASE);
            }
        }
    }

    /// Per-call microseconds and the speed-up over one thread.
    fn scale(what: &str, threads: &[usize], calls: usize, each: impl Fn(usize, usize) + Sync) {
        let mut one = 0.0;
        for &t in threads {
            let started = Instant::now();
            std::thread::scope(|s| {
                for k in 0..t {
                    let each = &each;
                    s.spawn(move || {
                        for i in 0..calls {
                            each(k, i);
                        }
                    });
                }
            });
            let secs = started.elapsed().as_secs_f64();
            let per_second = (t * calls) as f64 / secs;
            if t == 1 {
                one = per_second;
            }
            println!(
                "{what}: {t:2} threads, {:.1} us per call per thread, {:.0} calls/s in all, {:.2}x one thread",
                secs * 1e6 / calls as f64,
                per_second,
                per_second / one
            );
        }
    }

    /// How the kernel calls of today's change tracking scale over threads
    /// (each thread on memory of its own).
    #[test]
    #[ignore = "a benchmark: cargo test --release -p victory-gpu --lib rebar -- --ignored --nocapture --test-threads=1"]
    fn kernel_calls_scale() {
        const THREADS: [usize; 5] = [1, 2, 4, 8, 16];
        const MOST: usize = 16;
        // Write-watched private memory, 64 MiB a thread, as `xmem` makes
        // the title's committed memory.
        const PER: usize = 64 * MB;
        let watched: Vec<usize> = (0..MOST)
            .map(|_| {
                // SAFETY: a plain allocation of ours, touched whole.
                unsafe {
                    let p = mem::VirtualAlloc(std::ptr::null(), PER, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE);
                    assert!(!p.is_null());
                    std::ptr::write_bytes(p.cast::<u8>(), 1, PER);
                    p as usize
                }
            })
            .collect();
        let ask = |k: usize, i: usize, len: usize| {
            let base = watched[k];
            // A few pages written since the last asking, as a frame does.
            for j in 0..8 {
                // SAFETY: inside our allocation.
                unsafe { ((base + ((i * 7 + j * 131) * 4096) % len) as *mut u8).write_volatile(2) };
            }
            let mut found = vec![std::ptr::null_mut::<c_void>(); len / 4096];
            let mut n = found.len();
            let mut granularity = 0;
            // SAFETY: our watched allocation and room for every page.
            unsafe { mem::GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, len, found.as_mut_ptr(), &mut n, &mut granularity) };
            std::hint::black_box(n);
        };
        scale("GetWriteWatch with reset, 4 MiB ranges", &THREADS, 2000, |k, i| ask(k, i, 4 * MB));
        scale("GetWriteWatch with reset, 64 MiB ranges", &THREADS, 300, |k, i| ask(k, i, PER));
        // 64 KiB views of a reserved section in placeholders, committed, as
        // `xmem` maps the page pool: 512 views (32 MiB) a thread.
        const VIEWS: usize = 512;
        let mut sections = Vec::new();
        let views: Vec<usize> = (0..MOST)
            .map(|_| {
                // SAFETY: a section, a placeholder and views of our own,
                // made as `xmem` makes them.
                unsafe {
                    let section = mem::CreateFileMappingW(
                        windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                        std::ptr::null(),
                        mem::PAGE_READWRITE | mem::SEC_RESERVE,
                        0,
                        (VIEWS * K64) as u32,
                        std::ptr::null(),
                    );
                    assert!(!section.is_null());
                    sections.push(section as usize);
                    let base = mem::VirtualAlloc2(
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        VIEWS * K64,
                        mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                        mem::PAGE_NOACCESS,
                        std::ptr::null_mut(),
                        0,
                    ) as usize;
                    assert_ne!(base, 0);
                    for v in 0..VIEWS {
                        let at = base + v * K64;
                        if v + 1 < VIEWS {
                            mem::VirtualFree(at as *mut c_void, K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
                        }
                        // Pages in a scattered order, as the pool's are.
                        let page = (v * 37) % VIEWS;
                        let view = mem::MapViewOfFile3(
                            section,
                            std::ptr::null_mut(),
                            at as *const c_void,
                            (page * K64) as u64,
                            K64,
                            mem::MEM_REPLACE_PLACEHOLDER,
                            mem::PAGE_READWRITE,
                            std::ptr::null_mut(),
                            0,
                        );
                        assert_eq!(view.Value as usize, at, "view {v}: {}", std::io::Error::last_os_error());
                        assert!(!mem::VirtualAlloc(at as *const c_void, K64, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
                        (at as *mut u8).write_volatile(1);
                    }
                    base
                }
            })
            .collect();
        scale("VirtualProtect on a 64 KiB view, read-only and back (two calls)", &THREADS, 4000, |k, i| {
            let at = views[k] + (i % VIEWS) * K64;
            let mut old = 0;
            // SAFETY: our own views; only their protection changes, and back.
            unsafe {
                mem::VirtualProtect(at as *const c_void, K64, mem::PAGE_READONLY, &mut old);
                mem::VirtualProtect(at as *const c_void, K64, mem::PAGE_READWRITE, &mut old);
            }
        });
        scale("VirtualQuery in the views", &THREADS, 20000, |k, i| {
            let at = views[k] + (i % VIEWS) * K64;
            let mut info = mem::MEMORY_BASIC_INFORMATION::default();
            // SAFETY: a query that fills our structure.
            unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of::<mem::MEMORY_BASIC_INFORMATION>()) };
            std::hint::black_box(info.RegionSize);
        });
        scale("VirtualProtect on a 64 KiB piece of private memory, read-only and back", &THREADS, 4000, |k, i| {
            let at = watched[k] + (i % (PER / K64)) * K64;
            let mut old = 0;
            // SAFETY: our own memory; only its protection changes, and back.
            unsafe {
                mem::VirtualProtect(at as *const c_void, K64, mem::PAGE_READONLY, &mut old);
                mem::VirtualProtect(at as *const c_void, K64, mem::PAGE_READWRITE, &mut old);
            }
        });
        // SAFETY: our own allocations, views and sections.
        unsafe {
            for p in watched {
                mem::VirtualFree(p as *mut c_void, 0, mem::MEM_RELEASE);
            }
            for base in views {
                for v in 0..VIEWS {
                    mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: (base + v * K64) as *mut c_void });
                }
            }
            for s in sections {
                windows_sys::Win32::Foundation::CloseHandle(s as _);
            }
        }
    }
}
