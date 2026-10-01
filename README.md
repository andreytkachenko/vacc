# vacc

A Rust workspace for video decoding with five interchangeable backends:
**Vulkan Video**, **NVIDIA NVDEC** (cuvid), **VAAPI**, and two CPU software decoders —
**H.264/AVC** (pure-Rust port of the **edge264** core) and **H.265/HEVC** (pure-Rust port of
the **hevc.js** kernels). The root `vacc` crate ties them together into a single
[`VaccDecoder`](src/lib.rs) with automatic backend fallback. Based on the
[Khronos Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples).

Supports **H.264/AVC**, **H.265/HEVC**, **VP9**, and **AV1** decoding — see the
[Decode Support Matrix](#decode-support-matrix) for what each GPU/driver actually decodes
byte-exact (verified against FFmpeg, 300 frames per sample).

An optional **post-decode image pipeline** can scale frames and convert Y'CbCr to packed
RGB in one pass, on the fastest available backend: NVIDIA NPP, Vulkan compute (any GPU),
or a pure-Rust SIMD fallback — see [Image Pipeline](#image-pipeline-scale--rgb-conversion).

## Quick Start

```rust
use vacc::{Backend, DecoderConfig, VaccDecoder};

let data = std::fs::read("video.h264").unwrap();

// Default chain: vulkan -> nvdec -> vaapi -> software. The first backend that
// can initialize AND decode the stream wins; a failing backend falls through.
let mut decoder = VaccDecoder::new_auto(data).unwrap();
println!("using backend: {}", decoder.backend());
for frame in decoder.decode_all(usize::MAX).unwrap() {
    println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
}

// Or pick the fallback order yourself:
let config = DecoderConfig::new([Backend::Nvdec, Backend::Software]);
let decoder = VaccDecoder::new(&std::fs::read("video.h265").unwrap(), &config).unwrap();

// Optional post-decode image pipeline: scale to 1280x720 and emit packed RGB24.
// GPU-accelerated when available (NPP / Vulkan compute); frame.rgb_pixels carries the result.
use vacc::{ImageConfig, Interpolation, RgbChannels, Scale};
let config = DecoderConfig::default().with_image(ImageConfig {
    scale: Some(Scale::new(1280, 720, Interpolation::Bilinear)),
    rgb: Some(RgbChannels::Rgb24),
    ..Default::default()
});
let decoder = VaccDecoder::new(&data, &config).unwrap();
```

### Zero-copy GPU track

On NVIDIA hosts frames can stay on the device end to end: NVDEC writes into owned
CUDA buffers (no host staging) and the image pipeline (RGB conversion + scaling)
runs on-GPU via NPP. Frames then carry a [`GpuFrame`](src/lib.rs#L116) whose
`device_ptr()` is a stable `CUdeviceptr` an inference engine can consume directly
while the handle (or any clone of it) is alive:

```rust
let config = DecoderConfig::default()
    .with_gpu() // forces NVDEC and the zero-copy track
    .with_image(/* scale / rgb as above */);
let decoder = VaccDecoder::new(&data, &config).unwrap();
for frame in decoder.decode_all(usize::MAX).unwrap() {
    if let Some(gpu) = &frame.gpu {
        feed_to_inference(gpu.device_ptr(), gpu.width, gpu.height);
    }
}
```

Only 4:2:0 content is supported on this track; other chroma formats fall back to
the normal host pipeline with a warning. The `unified` example accepts `--gpu`.

Each backend is an optional cargo feature (`vulkan`, `nvdec`, `vaapi`, `sw`, all on by
default), so a distribution build can omit any GPU dependency:

```toml
# Cargo.toml
[dependencies]
vacc = { path = ".", default-features = false, features = ["vaapi", "sw"] }
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        vacc (workspace)                            │
│                                                                    │
│   vacc               unified decoder: VaccDecoder + fallback       │
│                        chain vulkan → nvdec → vaapi → sw, plus     │
│                        optional post-decode image pipeline         │
│                                                                    │
│   vacc-core          shared types, traits, errors                  │
│   vacc-parser        bitstream parsing (H.264/HEVC/VP9/AV1),       │
│                        common DPB + POC + ref-list state           │
│   vacc-common        shared SW-decoder test infra (golden hashes,  │
│                        PRNGs, saturation helpers)                  │
│                                                                    │
│   Backends:                                                        │
│     vacc-vulkan          Vulkan Video core (ash)                   │
│     vacc-vulkan-common   device init / queue management            │
│     vacc-vulkan-decode   Decoder-trait wrapper over vacc-vulkan    │
│     vacc-nvdec-decode    NVIDIA NVDEC via libnvcuvid (cuvid)       │
│     vacc-vaapi-decode    VAAPI stateless decode                    │
│     vacc-software-decode CPU H.264 (edge264 port) +                │
│                          CPU H.265 (hevc.js port), pure Rust       │
│                                                                    │
│   Image pipeline (post-decode scale / Y'CbCr→RGB):                 │
│     vacc-image           pure-Rust SIMD reference pipeline         │
│     vacc-npp             NVIDIA NPP GPU backend                    │
│     vacc-vkimage         Vulkan compute backend (any GPU)          │
│                                                                    │
│   vacc-examples: decode  unified CLI: -b <backend> -i <file>       │
└────────────────────────────────────────────────────────────────────┘
```

## Crate Overview

### `vacc` (root package)
The unified decoder:
- `VaccDecoder` / `new_auto()` — one API over all backends; default fallback chain
  `vulkan → nvdec → vaapi → software`
- `DecoderConfig` — custom fallback order (`new([Backend…])`, `only(…)`, `default_order()`)
  and the optional post-decode image pipeline (`with_image(ImageConfig)`)
- `detect_codec()` — H.264/H.265/VP9/AV1 bitstream detection
- Per-backend features (`vulkan`, `nvdec`, `vaapi`, `sw`); if every configured backend
  fails, the returned error lists each per-backend failure so callers can report *why*
- PTS normalization: frames carry monotonic presentation timestamps (synthetic 1/30 s
  steps when a backend's PTS is invalid or goes backwards under B-frame reordering)
- Optional post-decode image pipeline — see [Image Pipeline](#image-pipeline-scale--rgb-conversion)

### `vacc-core`
Core types and traits shared across all crates:
- `VideoCodec` - H.264, H.265, AV1, VP9 identification
- `VideoFormat` - Chroma subsampling, bit depth, profile info
- `PictureParametersSet` - SPS/PPS/VPS abstraction
- `DecodedFrame` - Output frame representation (`rgb_pixels: Option<RgbFrame>` carries the
  image-pipeline RGB output when conversion was requested)
- `VideoError` / `VideoResult` - Error handling

### `vacc-parser`
Bitstream parsing for each codec:
- **H.264**: SPS, PPS, slice header parsing
- **H.265**: VPS, SPS, PPS, slice header parsing
- **AV1**: Sequence header parsing (incl. film-grain params, signed segmentation
  features, show-existing-frame OBUs)
- NAL unit extraction and start-code detection
- RBSP (Raw Byte Sequence Payload) handling, emulation prevention byte removal
- **Common state used by every backend**: DPB managers (`H264Dpb`, `H265Dpb`, AV1 DPB),
  POC calculators (types 0/1/2), and spec-correct reference-list construction

### `vacc-common`
Code shared by the H.264/H.265 software decoders: golden-hash test infrastructure,
deterministic test PRNGs, saturation helpers.

### `vacc-vulkan` / `vacc-vulkan-common` / `vacc-vulkan-decode`
Vulkan Video implementation using `ash`:
- `vacc-vulkan-common` - shared device initialization, queue management, debug messenger
- `vacc-vulkan` - low-level pipeline: `VideoSession` (`VkVideoSessionKHR`),
  `BitstreamBuffer`, DPB images, codec-specific decoders (H.264 / H.265 / VP9 / AV1)
  and readback
- `vacc-vulkan-decode` - implements the core `Decoder` trait over the pipeline

### `vacc-nvdec-decode`
NVIDIA NVDEC via `libnvcuvid.so` (loaded dynamically with `libloading`):
- Per-codec decoders (H.264 / HEVC / VP9 / AV1) with cuvid parser bypass
  (bitstream parsed by `vacc-parser`, DPB managed in Rust)
- `query_decoder_caps` / `VACC_PROBE_CUVID=1` — driver capability queries;
  unsupported streams fail up front with a clear message

### `vacc-vaapi-decode`
VAAPI stateless decode on any libva driver (verified with Intel iHD):
- Per-codec decoders using the common DPB/POC state from `vacc-parser`
- Early capability rejections (e.g. H.264 4:2:2 on drivers whose AVC
  pipeline is NV12-only) instead of mid-decode driver errors
- Bindings via the `cros-libva` git dependency

### `vacc-software-decode`
The CPU backends — see [Software (CPU) Backends](#software-cpu-backends--b-sw).

### `vacc-examples`
- `decode` — unified CLI: `-b <vulkan|nvdec|vaapi|sw> -i <file> [-n frames] [-o outdir]`;
  prints pts/size/pixel-hash per frame (the hash is what the verification matrix compares)

### `vacc-image`
The pure-Rust image pipeline used as the reference implementation and last-resort fallback:
- Y'CbCr → RGB conversion in Q14 fixed point (BT.601/BT.709 × limited/full range tables,
  monochrome-safe), SSE4.1/AVX2 kernels
- Resize: nearest, box, bilinear, bicubic (Mitchell) with spec-correct tap tables,
  including the integer-ratio downscale degenerate case; AVX2/SSE4.1 H/V passes
- Affine warp (rotation/translation/scale/shear) via inverse mapping with
  nearest / bilinear / bicubic interpolation (box is resize-only)
- `process(img, cfg, Kernel)` — the entry point the GPU backends are validated against

### `vacc-npp`
NVIDIA NPP GPU backend for the image pipeline (loaded dynamically with `libloading`, same
pattern as NVDEC): Y'CbCr → RGB conversion and resize on the CUDA device. Selected
automatically on NVIDIA hosts; unsupported combinations fall back with a warning.

### `vacc-vkimage`
Vulkan compute backend for the image pipeline — works on **any** GPU (discrete or
integrated, no vendor SDK needed):
- WGSL shaders (`yuv2rgb`, bilinear `resize_yuv`, inverse-mapping `warp_rgb` over
  RGBA32) compiled to SPIR-V at build time via naga
- One host-visible arena buffer per device; an 8-slot command-buffer ring + fence per pass
- Same Q14 conversion math as the reference pipeline, validated against it with drift
  tolerances (≤1 LSB conversion, ≤4 LSB resize) in `cargo test -p vacc-vkimage`

## Image Pipeline (scale + RGB conversion)

`DecoderConfig::with_image(ImageConfig)` enables a post-decode pipeline applied to every
emitted frame. `ImageConfig` fields:

- `scale: Option<Scale>` — `Scale::new(width, height, interpolation)`, interpolation is
  `Nearest`, `Box`, `Bilinear` or `Bicubic`
- `rgb: Option<RgbChannels>` — `Rgb24` (3 bytes/px) or `Rgba32` (4 bytes/px, alpha 0xFF)
- `affine: Option<Warp>` — `Warp::new(transform, interpolation)`: 2×3 warp matrix plus
  the inverse-mapping interpolation (`Nearest`/`Bilinear`/`Bicubic`; requires RGB output)
- `spec: ColorSpec` — matrix (`Bt601`/`Bt709`) and range (`Limited`/`Full`);
  `ColorSpec::auto(height)` picks a sensible default (BT.709 limited at ≥720 lines)

Behavior:
- **Scale only**: the frame's YUV `pixel_data` is replaced by the resized YUV — 8-bit
  semi-planar (NV12) sources stay NV12; 10-bit sources are down-cast to 8-bit first and come
  out planar I420 (matching the reference pipeline).
- **RGB only**: original YUV is kept in `pixel_data`; packed RGB lands in
  `DecodedFrame.rgb_pixels`.
- **Scale + RGB**: the frame carries only the resized RGB in `rgb_pixels`.
- **Warp** (with or without scale): the frame is converted to RGBA32 first, then warped;
  the result lands in `rgb_pixels` (alpha dropped when `Rgb24` was requested).
- Backend dispatch per frame: **NPP** on NVIDIA hosts → **Vulkan compute** (any GPU) →
  **pure-Rust SIMD**. A GPU path that can't handle a combination (e.g. box interpolation,
  or an unsupported format) falls through to the next backend with a one-shot warning.
- No-op by default: without `with_image`, frames are emitted exactly as the decode backend
  produces them.

## Software (CPU) Backends (`-b sw`)

`sw` decodes on CPU only and needs **nothing but a CPU — both cores are pure Rust**
(no C/C++ toolchain). The codec is auto-detected: H.264 → edge264 port, H.265 → hevc.js
port. Both keep the *control plane* (bitstream parsing, DPB, POC, ref lists, display-order
reordering) on the shared `vacc-parser` state; only pixel reconstruction is codec-specific.

### H.264 — edge264 port (`src/avc/`)
- Bit-exact pure-Rust reimplementation of the edge264 slice-decode core: CABAC/CAVLC
  entropy decoding, intra/inter prediction, IDCT + dequantization, deblocking filter
- SSE SIMD kernels for motion compensation (luma/chroma), the deblock filter, and
  IDCT/dequant; every output is pinned to golden hashes (`src/avc/golden_data.rs`)
- Supported: 8-bit 4:2:0 (and monochrome) progressive frame coding. Other formats
  (10-bit, 4:2:2/4:4:4, field/MBAFF, separate colour plane) are rejected up front with a
  clear error instead of mis-decoding

### H.265 — hevc.js port (`src/hevc/`)
- Pure-Rust port of the [hevc.js](https://github.com/lid-labs/hevc.js) decoder kernels
  (MIT, see `crates/vacc-software-decode/HEVC_LICENSE`): CABAC engine, coding tree +
  residual coding, intra/inter prediction, transform/dequantization, deblocking (§8.7.2),
  SAO (§8.7.3)
- AVX2 SIMD kernels for inverse transforms (4x4–32x32 IDCT/DST) and MC FIR
  interpolation; WPP parallel pipeline over CTU rows
- Verified byte-exact: Main / Main 10 4:2:0, including CRA open-GOP and multi-slice WPP
  streams; outputs pinned to SHA-256 goldens

Threading: WPP CTU rows run on the rayon global pool (tune with `RAYON_NUM_THREADS`).

Verification: golden-hash unit tests + full-stream pixel oracles
(`cargo test -p vacc-software-decode`), plus byte-exact agreement with the FFmpeg
reference and hash-identical output vs the GPU backends on the committed sample set and
21 real-world streams (see [Decode Support Matrix](#decode-support-matrix)).

## Vulkan Video Pipeline

The decode pipeline follows the Vulkan Video extension workflow:

```
Bitstream ──► Parser ──► SPS/PPS/VPS ──► Session Parameters
                                         │
Bitstream ──► BitstreamBuffer ──────────┤
                                         ▼
                               VideoSession ──► vkCmdDecodeVideoKHR
                                         │               │
                               DPB Images  ───────────────┘
                                         │
                                         ▼
                                   Decoded Frame (YCbCr)
```

### Key Vulkan Objects

| Rust Type | Vulkan Handle | Purpose |
|-----------|--------------|---------|
| `VulkanDevice` | `VkDevice` | Logical device with video queue |
| `VideoSession` | `VkVideoSessionKHR` | Core decode session |
| `VideoSessionParameters` | `VkVideoSessionParametersKHR` | SPS/PPS/VPS storage |
| `BitstreamBuffer` | `VkBuffer` | Compressed video data |
| `DpbImage` | `VkImage` | Decoded Picture Buffer |

## Decode Support Matrix

Verified 2026-09-27 with `verify-all.py`: Big Buck Bunny 640x360 @ 30 fps, **300 frames** per
sample (the six `t*`/`x*` stress samples are 30-frame files — every available frame is verified).
Each decoded frame is compared **byte-exact** against an FFmpeg software-decode reference in the
stream's native pixel format. Environment:

- **NVIDIA GeForce RTX 3060 (GA106)** — Vulkan Video and NVDEC (`cuvid`) columns
- **Intel Meteor Lake-P iGPU** — VAAPI column (iHD driver). Its Vulkan driver exposes no
  video decode queue in this environment, so the Vulkan column runs on GA106.

Legend: ✅ = 300/300 byte-exact | S(30/30) = sample has only n frames, all verified exact |
HW-n/a = the stream's profile/chroma/depth is not supported by that GPU's hardware or driver
(evidence below; not a bug in this codebase).

| Sample (profile · chroma · depth) | Vulkan Video (GA106) | NVDEC (GA106) | VAAPI/iHD (MTL) |
|---|---|---|---|
| `h264_baseline` (Baseline · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `h264_constrained_baseline` | ✅ | ✅ | ✅ |
| `h264_main` (Main · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `h264_high` (High · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `h264_tC` / `tD` / `tN` / `tW` (transform stress, 30f) | S(30/30) | S(30/30) | S(30/30) |
| `h264_xallI` (all-IDR, 30f) | S(30/30) | S(30/30) | S(30/30) |
| `h264_xfd` (frame-dup stress, 30f) | S(30/30) | S(30/30) | S(30/30) |
| `h264_high10` (High 10 · 4:2:0 · 10b) | HW-n/a | HW-n/a | HW-n/a |
| `h264_high422` (High · 4:2:2 · 8b) | HW-n/a | HW-n/a | HW-n/a |
| `h264_high444` (High · 4:4:4 · 8b) | HW-n/a | HW-n/a | HW-n/a |
| `h265_main` (Main · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `h265_cra` (CRA open-GOP, no IDR) | ✅ | ✅ | ✅ |
| `h265_msp` (multi-slice pictures) | ✅ | ✅ | ✅ |
| `h265_main10` (Main 10 · 4:2:0 · 10b) | ✅ | ✅ | ✅ |
| `vp9_profile0` (P0 · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `vp9_profile1_444` (P1 · 4:4:4 · 8b) | HW-n/a | HW-n/a | ✅ |
| `vp9_profile1` (P1 · 4:2:0 · 10b) | ✅ | ✅ | ✅ |
| `vp9_profile2` (P2 · 4:2:0 · 12b) | ✅ | ✅ | ✅ |
| `av1_main` (main · 4:2:0 · 8b) | ✅ | ✅ | ✅ |
| `av1_high` (high · 4:2:0 · 10b) | ✅ | ✅ | ✅ |
| `av1_professional` (professional · 4:2:2 · 10b) | HW-n/a | HW-n/a | HW-n/a |
| `av1_grain` (main · film-grain SPS, grain not applied) | ✅ | ✅ | ✅ |
| `av1_seg` (main · segmentation + show-existing OBU stress) | ✅ | ✅ | ✅ |

**Result: 64/64 decodable cells byte-exact; 0 failures.** 14 cells are HW-n/a.

### Software backend (`sw`) coverage

The CPU backends implement H.264 and H.265 only (no VP9/AV1), so they have no column in
the table above. Verified against the same FFmpeg references: **all eight-bit 4:2:0 H.264
samples** (baseline → `xfd`, via the edge264 port) and **all four H.265 samples**
(main, cra, msp, main10, via the hevc.js port) decode byte-exact, with per-frame hashes
identical to the GPU backends in every cell above. The three H.264 high-profile 10/4:2:2/4:4:4
samples are rejected up front (the edge264 port targets 8-bit 4:2:0 progressive).

### HW-n/a evidence (measured, not assumed)

- **H.264 High 10-bit — no backend**: GA106 `cuvidGetDecoderCaps` reports H.264 as 8-bit 4:2:0
  only (decoder creation fails with error 801); the GA106 Vulkan Video driver rejects the
  spec-legal profile-110 + 10-bit combo (`ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR`); iHD's
  H.264 caps list is 8-bit only and rejects `RTFormat=YUV420_10`.
- **H.264 4:2:2 / 4:4:4 — no backend**: the Vulkan Video spec exposes no 4:2:2/4:4:4 formats for
  H.264 decode (only HEVC has them) and the GA106 driver rejects those profiles; GA106 NVDEC caps
  report 8-bit 4:2:0 only; iHD's AVC VLD pipeline is NV12-only — it *accepts* a config with
  `RTFormat=YUV422` (a lenient caps fallback) but fails at `vaEndPicture`, and offers no 4:2:2
  surface pixfmt for the config. The decoders detect this up front and fail with a clear
  "HW does not support …" error instead of failing mid-decode.
- **VP9 4:4:4 — Vulkan + NVDEC**: the Vulkan Video spec exposes only 4:2:0 formats for VP9 decode
  (the GA106 driver rejects the 4:4:4 profile); GA106 NVDEC caps report VP9 P1 4:4:4 unsupported.
  iHD *does* support it (✅ in the VAAPI column).
- **AV1 Professional (profile 2, 10-bit 4:2:2) — no backend**: iHD's AV1 decode caps list only
  Profile 0/1; GA106 NVDEC caps report AV1 as main/high 4:2:0 only; the GA106 Vulkan Video driver
  rejects profile 2.

### Notes

- `h265_msp` exercises multi-slice pictures (multiple slice segments per frame) end-to-end on all
  three backends — each segment carries its own slice header, and dependent segments inherit the
  first segment's parameters per spec 7.3.8.
- `av1_grain` carries a film-grain SPS + grain block in every frame header (decoded with
  `apply_grain` forced off to stay pixel-identical to FFmpeg); `av1_seg` is a rav1e encode with
  signed segmentation features and ~half the pictures as show-existing-frame OBUs.
- NVDEC 10/12-bit content decodes into P016 surfaces (the only >8-bit output format in the public
  cuvid API); readback scales to 8-bit with round+clamp, matching the other backends.
- Diagnostics: `VACC_PROBE_CUVID=1` dumps the full cuvid decoder-caps table; `VACC_VA_DUMP=1`
  dumps the exact VA-API picture/slice parameter buffers.

## Test Samples

The single master source is `assets/big_buck_bunney.h265` (Big Buck Bunny, 1920x1080,
300 frames). All 26 samples in `assets/samples/` are codec variants of that one video,
produced by the ffmpeg recipes embedded in `verify-all.py` (per-sample encoder options:
profile, chroma format, bit depth, GOP/stress flags) — including two AV1 stress encodes:
`av1_grain.ivf` (aomenc film-grain table) and `av1_seg.ivf` (rav1e segmentation +
show-existing-frame OBUs).

- `python3 verify-all.py` — verifies the committed samples (default; no encoding).
- `python3 verify-all.py --generate` — encodes only **missing** samples from the master.
- `python3 verify-all.py --regen` — re-encodes **all** samples (overwrites). Regenerated
  files are *structurally equivalent* (same profile/chroma/depth, same keyframe layout)
  but **not byte-identical** to the committed set (encoder-version differences). The
  committed samples are the canonical anchors for embedded test data (h264 NAL constants,
  VP9 golden DPB fixture, AV1 expected-value table), so prefer `--generate` unless you
  deliberately refresh the whole set.

Other assets: `assets/bframe_test.h264` (B-frame/MMCO DPB-parity test) and
`assets/born_trailer.h264` (integration tests). The VP9 golden DPB fixture for
`nvdec-decode` is embedded at `crates/vacc-nvdec-decode/tests/data/vp9_dp_golden_20f.ivf`.

## Re-verification

```bash
cargo build --release --examples          # NOTE: --examples, or the binary goes stale
python3 verify-all.py                     # full 300-frame matrix (refs cached in /tmp/verify_all)
python3 verify-all.py --max-frames 30     # quick smoke
python3 verify-all.py --backends vaapi --samples h265_msp
```

`verify-all.py` decodes each sample with the unified `decode` example (per-frame canonical planar
YUV dumps via `-o`), decodes the same frames with FFmpeg (software reference, native pixel
format), and byte-compares every frame. Cells whose hardware/driver genuinely cannot decode the
stream are listed (with evidence) in `HW_UNSUPPORTED` and reported as `HW-n/a`.

Beyond the sample matrix, the decoders are cross-checked on real-world streams: 21/21 streams
(18 H.264/H.265 files × 4 backends including `sw`, 3 VP9/AV1 files × 3 backends; ~654k frames)
produce hash-identical per-frame output across every applicable backend (2026-09-26).

## Unified Decode Example

```bash
# Decode with a chosen backend; prints pts/size/pixel-hash per frame
./target/release/examples/decode -b <vulkan|nvdec|vaapi|sw> -i <file> [-n frames] [-o outdir]
# sw = CPU, pure Rust: H.264 (edge264 port) / H.265 (hevc.js port), codec auto-detected
```

The root package also ships a `unified` example with the full fallback chain and the image
pipeline:

```bash
cargo run -p vacc --example unified -- -i <file> [options]

  -o, --order    <csv>    backend order, e.g. "nvdec,software"
                          (default: vulkan,nvdec,vaapi,software)
  -n, --max      <num>    stop after this many frames (default: all)
  -w, --width    <px>     resize output width (with -H)
  -H, --height   <px>     resize output height (with -w)
  -f, --filter   <name>   resize filter: box | bilinear | bicubic (default: bilinear)
      --rgb24          convert frames to packed RGB24
      --rgb32          convert frames to packed RGBA32
  -O, --out      <file>   write the first decoded frame as PPM (needs --rgb24/--rgb32)
```
Per-frame lines hash whatever the frame carries after the image pipeline ran (`rgb` when
`--rgb24`/`--rgb32` is active, `yuv` otherwise).

## Vulkan Extensions Required

- `VK_KHR_video_decode_queue` - Video decode queue family
- `VK_EXT_video_decode_h264` - H.264 decode support
- `VK_EXT_video_decode_h265` - H.265 decode support
- `VK_EXT_video_decode_av1` - AV1 decode support
- `VK_KHR_sampler_ycbcr_conversion` - YCbCr sampling

## Dependencies

- **ash** - Vulkan bindings for Rust
- **bytemuck** - Safe memory casting
- **bitflags** - Vulkan flag types
- **thiserror** - Error handling
- **log** / **tracing** - Logging
- **rayon** - Software-decoder threading (WPP CTU rows)
- **libloading** - Dynamic `libnvcuvid.so` (NVDEC) and NPP library loading
- **cros-libva** - VAAPI bindings (git dependency)
- **naga** - Build-time WGSL → SPIR-V compilation for the `vacc-vkimage` shaders

## Building

```bash
cargo build --release --examples   # NOTE: --examples is required; plain builds don't rebuild it

# Run the unified decode example (pts, size, pixel hash per frame)
./target/release/examples/decode -b <vaapi|vulkan|nvdec|sw> -i <file.ivf|h264|h265> [-n frames] [-o outdir]
```

Backend requirements: Vulkan Video device (VAAPI also works on the same stack), NVIDIA driver with
`libnvcuvid.so`, and libva + a VAAPI driver (iHD/Mesa) respectively. `sw` needs nothing but a CPU —
both cores are pure Rust.

Each backend of the root `vacc` package is a cargo feature (all on by default), as are the
backends of the `decode` example, so a build can be trimmed to the backends you need:

```bash
cargo build -p vacc --no-default-features --features "vaapi sw"          # root library subset
cargo build --release -p vacc-examples --example decode --no-default-features \
    --features "hevcjs vaapi"                                            # example subset
```

Features: `edge264` (CPU H.264), `hevcjs` (CPU H.265), `nvdec`, `vaapi`, `vulkan`
(example crate); `sw` (both CPU cores) on the root package.

## Reference

This library is based on the [Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples) from Khronos:

- [Vulkan Video Deep Dive](https://www.khronos.org/assets/uploads/apis/Vulkan-Video-Deep-Dive-Apr21.pdf)
- [Vulkan Video Extensions Spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html)
- [NVIDIA Vulkan Video Driver](https://developer.nvidia.com/vulkan-driver)
- Software cores: the H.264 data plane is a bit-exact port of the edge264 decoder core; the
  H.265 data plane is a port of the [hevc.js](https://github.com/lid-labs/hevc.js) kernels (MIT).

## License

MIT OR Apache-2.0

The H.265 software kernels are derived from hevc.js and remain MIT-licensed
(`crates/vacc-software-decode/HEVC_LICENSE`).
