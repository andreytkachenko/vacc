# Unsafe Audit & Path to 100% Safe

Audit of every `unsafe` in the workspace, with a necessity proof for each
category and a wave plan to eliminate all of it where possible.

- Line-level inventory (current state): [`UNSAFE_INVENTORY.txt`](UNSAFE_INVENTORY.txt).
- Hard constraint: **performance must not regress**. Any change to SIMD or
  hot-loop code requires before/after benchmark comparison (see
  "Performance guardrails"). Baselines: memory `perf_baselines.md`, harness
  `examples/bench.rs`.

## Bottom line (honest feasibility)

| Tier | Crates | Target | Why |
|---|---|---|---|
| A — pure Rust | `vacc-core`, `vacc-common`, `vacc-parser`, `vacc-image`*, `vacc-software-decode`*, `vacc-software-encode`, `vacc-golden-tests`, top-level `vacc` (`src/`), `vacc-examples` | **0 unsafe except SIMD intrinsics** | Every occurrence here is removable: raw-pointer pixel views (redesigned as offsets into the owned buffer — done for the frame types in Wave 1), pointer arithmetic (safe indexing, identical codegen), custom allocators, Send/Sync impls. *Exception: x86 SIMD intrinsics stay (see category 7) — on stable Rust there is no portable SIMD (`std::simd` is still unstable, verified E0658 on rustc 1.96), and the kernels are 40–50% faster than scalar (hard perf constraint). User-confirmed 2026-10-13: keep intrinsics, document as necessary residual. |
| B — FFI backends | `vacc-vulkan*`, `vacc-nvdec-decode`, `vacc-nvenc-encode`, `vacc-npp`, `vacc-vaapi-*`, `vacc-libva` | **provably minimal set, each site SAFETY-proved** | Every `extern "C"` call requires an `unsafe` block by language rule. ash 0.38 (our Vulkan binding) exposes *all* entry points as `unsafe fn` (verified: `create_buffer` etc.), and does **not** generate safe wrappers for the Vulkan Video decode/encode commands, so the proc-addr + typed-fn-pointer pattern is the only way to call them. |

"100% safe" therefore means: **Tier A = zero except the SIMD floor; Tier B =
the language floor, individually justified**. Any future wave that claims a
removal must show the compiler no longer needs the block (or replace it with
an external safe wrapper).

### Final state (all waves done)

Workspace total: **2935 → 1215** lines containing `unsafe`.

| Bucket | Lines (approx) | Status |
|---|---:|---|
| Tier A crates at **0 unsafe**: top-level `src/`, `vacc-core`, `vacc-common`, `vacc-parser`, `vacc-golden-tests`, `vacc-examples`, `vacc-software-encode` | 0 | done (Waves 1–3) |
| `vacc-image`: scalar conv/resize/warp fully safe; residual = SIMD kernels + dispatch | ~57 | SIMD floor (user-confirmed keep) |
| `vacc-software-decode`: slice.rs/decoder.rs/golden paths safe; residual = SIMD kernels (~238) + WppShared (14) | ~252 | SIMD floor + proven-necessary concurrency pattern |
| Tier B FFI backends (Vulkan/ash, NVDEC/CUDA/cuvid, NPP, VAAPI, NVENC) | ~900 | language floor: every `extern`/`unsafe fn` call needs a block; consolidated (cached fn tables, `video_fn!`, safe wrapper layers), crate-level SAFETY inventories in each lib.rs, per-block proofs where non-obvious |

## Inventory (initial, pre-waves): 2935 lines

| # | Category | Count | Removable? | Strategy / wave |
|---|---|---:|---|---|
| 1 | `generated-bindgen` (nvenc_api.rs) | 1525 | **yes, ~all** | bindgen output. The `raw_*` bitfield helpers, standalone extern blocks, and `_raw()` fns are **unused anywhere** → deleted (done, Wave 5a). Residual: fn-pointer *type aliases* only (FFI floor). |
| 2 | Vulkan API calls via ash 0.38 | ~450 | **no** (floor) | Every call is `unsafe fn` in ash 0.38. Consolidate behind safe wrapper methods so only the FFI layer holds unsafe, each block SAFETY-proved. Wave 4. |
| 3 | Vulkan video entry points (proc-addr + `transmute`/`unsafe extern fn`) | ~25 | **no** (floor) | Not in ash's safe API. Consolidate into one cached-function-pointer module with a single SAFETY proof per function (ABI = Vulkan spec, pointers come from the loader). Wave 4. |
| 4 | CUDA / cuvid / NPP / libva call sites (`cu*`, `va*`, `npp*`, vtable calls) | ~250 | **no** (floor) | Same language floor. Consolidate per-crate into `ffi.rs`-style modules; SAFETY comment per block (argument invariants, lifetime ownership, context rules). Wave 5. |
| 5 | `mem::zeroed()` for FFI structs | 53 | **yes, most** | ash `vk::native` structs implement `Default`; bindgen structs derive `Default`; where a type lacks it, add a safe fieldwise-zero constructor (raw-pointer fields use the safe `std::ptr::null()`). Waves 4–5. |
| 6 | Pointer arithmetic / raw slices in pure Rust | 118+34 | **yes** | Offset-based `PixelPlane` (Wave 1, done) + safe row accessors; hot loops use index math that compiles to the same address arithmetic. Waves 2–3. |
| 7 | x86 SIMD intrinsics (`_mm*`, `#[target_feature]`) | ~450 (incl. software-decode kernels) | **no on stable** (floor) | ~~Rewrite in portable `std::simd`~~ — **disproven**: `std::simd` is still unstable on rustc 1.96 (E0658 `portable_simd`, verified by compile probe); calling any `#[target_feature]` fn from a non-enabled context also requires an unsafe block (E0133, verified). So raw `core::arch` intrinsics are the *only* vectorization path on stable Rust. Kernels are 40–50% faster than scalar (conv: 6.03 vs 9.94 ms; resize: 21.38 vs 41.94 ms at 1080p) — dropping them violates the perf hard constraint. **Decision (user-confirmed 2026-10-13): keep intrinsics; each kernel's bounds invariants are documented in its module header; this entry is the necessity proof.** |
| 8 | `unsafe impl Send/Sync` | 26 | **yes for pure-Rust types** | Frame types: removed in Wave 1 (raw-pointer fields became integer handles/offsets). Remaining are all on FFI handle owners (CUDA streams/events, cuvid decoders, libva display, Vulkan device) → Tier-B floor, each with a SAFETY note. AlignedBuf (sw decoder) disappears with the allocator in Wave 3. |
| 9 | Dynamic loader (`GetProcAddress`/`get_device_proc_addr`) + `read_unaligned` on raw handles | 9 | **no** (floor) | OS-level symbol resolution has no safe API. Consolidate into the FFI modules (Waves 4–5). |

## What was changed so far (done, verified)

**Wave 1 — frame types + top-level `src/` + `vacc-core` → 0 unsafe.**
The single highest-leverage design fix: `PixelPlane.data: *const u8` →
`PixelPlane.data_offset: usize` (a view into the owned `PixelData::buffer`, an
invariant verified at all 59 construction sites across 10 files). This made:

- `PixelData` cloneable by derive (no pointer rebasing), with safe
  `plane_view()` / `plane_row()` accessors;
- `DecodedFrame` auto-`Send + Sync` (device handles became opaque integer
  values, matching the existing `GpuFrame.ptr: usize` convention);
- the top-level `vacc` crate, `vacc-core`, `vacc-golden-tests::canonical`,
  and the `decode` example **0 unsafe**.

Verification: full workspace `cargo check --all-targets` clean; all transform
unit tests, AVC fuzz goldens (20/20), HEVC end-to-end goldens (7/7), HEVC
slice-segment goldens (5/5), incremental decode (1/1), software-encode
roundtrip (5/5) and rate-control suites pass. The two failing items on this
dev machine (`vulkan_gpu_track_matches_sw_reference`: no Vulkan video-decode
ICD; a parallel-test SIGSEGV in the VAAPI/iHD driver) were confirmed
pre-existing by running the identical tests on the stashed pristine tree.

**Wave 2 — `vacc-image` scalar + warp paths → 0 unsafe.**
- Scalar conv/resize/warp kernels converted to safe slice indexing
  (`&[u8]` rows, no raw pointers); dead `pixels_ptr` removed.
- `warp/avx2.rs` turned out to be a *fake* AVX2 kernel (zero intrinsics —
  scalar code behind raw pointers with an unneeded `#[target_feature]`).
  Converted to a fully safe function; dispatch in `warp.rs` simplified.
- All 46 vacc-image tests pass (incl. all-kernels-agree); warp bench shows
  Scalar/Sse/Avx2 within 0.3% of each other (no regression).
- Remaining unsafe in the crate: SIMD kernel bodies + dispatch call sites
  only — category-7 floor (see above).

**Wave 5a — `vacc-nvenc-encode/src/nvenc_api.rs` pruned.**
13085 → ~6500 lines. Removed: 6 `raw_*` bitfield helpers, 147 standalone
`unsafe extern "C"` blocks (unused — the code loads `NvEncodeAPICreateInstance`
via libloading and uses only the vtable), 372 `_raw()` fns, 767 orphan attrs.
Converted all bindgen bitfield accessors to safe casts (186 getters, 186
setters, 186 pack-constructor blocks; signed fields cast to `as i16`/`as i32`,
bit-identical to the old same-size transmutes). Residual: **43 unsafe — all
fn-pointer type aliases** (`pub type X = Option<unsafe extern "C" fn(...)>`),
i.e. the FFI floor; 0 unsafe blocks, 0 transmutes. `cargo test -p
vacc-nvenc-encode` passes.

## Performance guardrails

1. **Baseline first.** Captured (memory `perf_baselines.md`, harness
   `examples/bench.rs`): sw H.264/H.265 decode fps; vacc-image conv/resize/warp
   micro-benchmarks at 1080p, all kernels.
2. **Gate.** Any change to a kernel ships only if it is within noise of the
   intrinsic baseline on this CPU (AVX2 + SSE4.2) *and* golden tests pass
   byte-exact. Otherwise the kernel keeps its current form and is recorded as
   a justified exception in the residual ledger below.
3. **Pixel-exactness.** `verify-all.py` (hash-based, vs ffmpeg) on any
   machine with a Vulkan video-decode or NVDEC device; golden tests stand in
   on this dev box (no GPU).

## Wave plan

| Wave | Scope | Status |
|---|---|---|
| 0 | This audit + line-level inventory | done |
| 1 | Frame types + top-level `src/` + `vacc-core` → 0 | **done** (2935 → 2872) |
| 2 | `vacc-image`: ptr-arith → safe indexing; fake-AVX2 warp kernel → fully safe; SIMD intrinsics documented as floor | **done** (→ 1366 workspace total) |
| 3 | `vacc-software-decode` (~450 lines: slice.rs 140, SIMD kernels ~253, coding_tree 14, decoder 14), `vacc-software-encode`, examples: AlignedBuf → Vec, MB raw-pointer views → safe accessors; SIMD stays (category 7) | **done** — slice.rs 140→15 (all residual = SIMD-kernel call sites, category 7), decoder.rs 14→0 (AlignedBuf→Vec + slices), sw-encode already 0; coding_tree.rs 14 stays (WppShared — ledger); goldens byte-exact, decode perf A/B: no regression |
| 4 | Vulkan crates: safe wrapper consolidation (video path incl. cached video fn pointers), `zeroed()` → `Default`, residual ledger entries per call site | **done** — video fn-ptr transmutes consolidated into one documented `video_fn!` macro (`vacc-vulkan/src/video_ffi.rs`, 40 call sites; generic helper impossible — E0512); crate-level SAFETY inventories added (vacc-vulkan, vkimage, vulkan-encode); `zeroed()` stays where ash `vk::native` POD types lack `Default` (verified in ash 0.38 source: only Debug/Copy/Clone derived) |
| 5 | NVDEC/NVENC/NPP/VAAPI: FFI consolidation + SAFETY proofs per block; residual ledger | **done** — 5a (nvenc_api.rs pruned to fn-ptr type-alias floor); 5b (VAAPI pixel unpacking → safe slices 28→11; NVDEC/NPP/VAAPI crate-level SAFETY inventories added; all FFI tests pass) |

## Residual unsafe ledger (target end state, to be filled per wave)

Each remaining `unsafe` must appear here with its necessity proof.

| Crate / file | Count | Proof of necessity |
|---|---:|---|
| `vacc-nvenc-encode/src/nvenc_api.rs` | 43 | All are `pub type … = Option<unsafe extern "C" fn(…)>` function-pointer **type aliases** from bindgen. A function-pointer type to C is inherently `unsafe extern "C"` — no safe spelling exists. The standalone extern *blocks* and all bitfield accessor unsafe were deleted (Wave 5a). |
| `vacc-image/src/{conv,resize}/{avx2,sse4}.rs` + dispatch sites in `conv.rs`/`resize.rs` | ~57 | SIMD floor (category 7): intrinsics are the only vectorization path on stable Rust (`std::simd` unstable — E0658 on rustc 1.96; `#[target_feature]` call sites require unsafe — E0133). Kernels 40–50% faster than scalar; bounds invariants documented per module header (load sizes vs `x + N <= width` proofs). |
| `vacc-software-decode/src/{avc,hevc}/*` SIMD kernels | ~253 | Same SIMD floor (category 7), same proof. Wave 3 removed every *non-SIMD* unsafe in the crate: slice.rs raw-pointer MB/pixel views → safe index/slice accessors (140→15; all 15 are `unsafe fn` calls into the SIMD residual/transform kernels), decoder.rs AlignedBuf allocator + margin pointer math → plain `Vec<u8>` + slices (verified: no aligned SIMD loads exist in this crate). |
| `vacc-software-decode/src/hevc/coding_tree.rs` (WppShared) | 14 | WPP (wavefront parallel processing) row tasks concurrently mutate **disjoint CTU regions of shared grids** with cross-row visibility of completed neighbours. Safe Rust has no expression for "overlapping `&mut` views of one allocation that are disjoint in practice": Mutex/RwLock would serialize adjacent rows (defeating WPP), per-cell atomics don't fit the grid element types and add atomic traffic to the hot loop. The 2 `Send`/`Sync` impls + 12 accessor blocks therefore stay; the invariant (WPP diagonal dependency, `RowSync` condvar synchronization) is documented on the struct and each accessor carries a SAFETY note. |
| `vacc-vulkan` (ash calls + video fn-ptrs) | ~275 | ash 0.38 exposes **all** entry points as `unsafe fn` (verified in registry source) and generates no safe wrappers for the Vulkan Video (KHR_video) commands → the proc-addr + typed-fn-pointer pattern is the only path. Wave 4 consolidated the 40 transmute call sites into one documented `video_fn!` macro (`src/video_ffi.rs`, module-level SAFETY proof: loader-provenance, spec-exact ABI, same-size usize↔fn-ptr on x86_64); crate-level SAFETY inventory in `lib.rs`. Residual = ash-call floor + `zeroed()` on ash `vk::native` PODs (no `Default` in ash 0.38). |
| `vacc-vkimage` | ~68 (code) | GPU image pipeline: ash calls inside safe wrapper fns (already consolidated), device-mapped memory (`vkMapMemory`/`vkUnmapMemory` paired, mutex-guarded — basis of the manual `Send for Gpu`), POD byte views of uniform structs, arena down-cast copy. Crate-level SAFETY inventory added in Wave 4 (`lib.rs` header). |
| `vacc-vulkan-encode` | ~70 (code) | Same ash floor; video fn pointers already cached once in `session::VideoFns` via the documented `load_fn` helper; staging map/unmap paired. Crate-level SAFETY inventory added in Wave 4. Note: `probe_capabilities` test SIGSEGVs on this dev machine's ICD — confirmed pre-existing on the pristine tree (no encode-capable device here). |
| `vacc-vulkan-decode` | 2 | `VulkanDecoder` manual `Send` impl moved here in Wave 1 (SAFETY: all Vulkan handles it owns are Send-safe under single-threaded submission discipline); one ash call. |
| `vacc-nvdec-decode` (device 150, decoder 32, av1 24, vp9 22, h265 17, ffi 7, gpu 6) | ~258 | CUDA Driver + cuvid FFI: entry points resolved once via `libloading` into cached `CudaFuncs`/`NvdecFuncs` tables (`OnceLock`); the 76 `unsafe extern "C" fn` field types are the FFI floor, the ~150 call blocks check returned error codes. Handle wrappers (`CtxPtr`, `CuStream`, `CuEvent`, `DeviceBuffer`, `PinnedHostBuffer`) carry documented manual `Send`/`Sync`; `as_slice/as_mut_slice` are the only slice materializers over CUDA-owned pinned memory. `zeroed()` on `CUVID*` C PODs (no `Default`). Crate-level SAFETY inventory added in Wave 5b. |
| `vacc-npp/src/ffi.rs` | ~28 | One `extern "C"` block + fn-pointer type aliases for the resolved NPP symbols (FFI floor); cached `Ffi` table via `libloading`/`OnceLock`; `NppStreamContext`/`Ffi` Send/Sync documented (handles never dereferenced from Rust, process-lifetime). Crate-level SAFETY inventory added in Wave 5b. |
| `vacc-vaapi-decode/src/decoder.rs` | 11 | libva FFI: `vaCreateImage`/`vaGetImage`/`vaSyncSurface`/`vaDestroyImage` call blocks, `zeroed()` on `VAImageFormat`/`VAImage`/VP9 segment C PODs, two derefs of driver-owned parameter-buffer pointers (`pp.seq_fields.value`, `pp.pic_fields.value`), and the `Send for VaapiDecoder` impl (display/surface handles). Wave 5b removed 17: all packed-format pixel unpacking (Y410/XYUV/AYUV/P010-12) converted from raw-pointer arithmetic to safe slice indexing (the libva `Image::as_ref()` already returns `&[u8]`). |
