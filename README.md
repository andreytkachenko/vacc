# vacc

A Rust workspace for hardware-accelerated video decoding with three interchangeable backends:
**Vulkan Video**, **NVIDIA NVDEC** (cuvid), and **VAAPI**. Based on the
[Khronos Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples).

Supports **H.264/AVC**, **H.265/HEVC**, **VP9**, and **AV1** decoding — see the
[Decode Support Matrix](#decode-support-matrix) for what each GPU/driver actually decodes
byte-exact (verified against FFmpeg, 300 frames per sample).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        vacc (workspace)                          │
│                                                                  │
│   vacc-core        shared types, traits, errors              │
│   vacc-parser      bitstream parsing (H.264/HEVC/VP9/AV1),   │
│                        common DPB + POC state (one manager       │
│                        across backends)                          │
│                                                                  │
│   Backends:                                                      │
│     vacc-vulkan    Vulkan Video (ash)                        │
│     vacc-nvdec-decode  NVIDIA NVDEC via libnvcuvid (cuvid)        │
│     vacc-vaapi-decode  VAAPI stateless decode                     │
│     libva              Rust bindings for libva                   │
│                                                                  │
│   vacc-examples: decode  unified CLI: -b <backend> -i <file>     │
└──────────────────────────────────────────────────────────────────┘
```

## Crate Overview

### `vacc-core`
Core types and traits shared across all crates:
- `VideoCodec` - H.264, H.265, AV1, VP9 identification
- `VideoFormat` - Chroma subsampling, bit depth, profile info
- `PictureParametersSet` - SPS/PPS/VPS abstraction
- `DecodedFrame` - Output frame representation
- `VideoError` / `VideoResult` - Error handling

### `vacc-parser`
Bitstream parsing for each codec:
- **H.264**: SPS, PPS, slice header parsing
- **H.265**: VPS, SPS, PPS, slice header parsing
- **AV1**: Sequence header parsing
- NAL unit extraction and start-code detection
- RBSP (Raw Byte Sequence Payload) handling
- Emulation prevention byte removal

### `vacc-vulkan`
Vulkan Video implementation using `ash`:
- `VulkanDevice` - Device initialization with video decode support
- `VideoSession` - `VkVideoSessionKHR` management
- `BitstreamBuffer` - `VkBuffer` for compressed video data
- Codec-specific decoders (H.264 / H.265 / VP9 / AV1) and readback

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

### `libva`
Rust bindings for libva (display, config, context, surface, picture).

## Quick Start

```toml
# Cargo.toml
[dependencies]
vacc-core = { path = "crates/vacc-core" }
vacc-parser = { path = "crates/vacc-parser" }
vacc-vulkan = { path = "crates/vacc-vulkan" }
```

```rust
use vacc_core::codec::VideoCodec;
use vacc_parser::h264::H264Parser;
use vacc_vulkan::{VideoDeviceBuilder, VideoPipeline, VideoCodec};

// 1. Initialize Vulkan device
let device = VideoDeviceBuilder::new()
    .with_video_codecs(ash::vk::VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR)
    .with_validation(true)
    .build()?;

// 2. Create pipeline
let mut pipeline = VideoPipeline::new(&device, VideoCodec::DecodeH264)?;
pipeline.init()?;

// 3. Parse bitstream
let mut parser = H264Parser::new();
let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
parser.init(&format)?;

// 4. Feed packets and decode
for packet in bitstream_packets {
    let result = parser.parse(&packet)?;
    match result {
        ParseResult::ParameterSet { sps, pps, .. } => {
            if let Some(sps) = sps {
                pipeline.decoder_mut()
                    .and_then(|d| match d {
                        DecoderWrapper::H264(dec) => {
                            dec.set_sps(sps.downcast_ref::<H264Sps>()?.clone());
                            Some(())
                        }
                        _ => None,
                    });
            }
        }
        ParseResult::Slice { slice_data_offset, slice_data_len, .. } => {
            // Record decode command
            pipeline.record_decode(...)?;
        }
        ParseResult::EndOfStream => break,
        _ => {}
    }
}
```

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

Verified 2026-08-31 with `verify-all.py`: Big Buck Bunny 640x360 @ 30 fps, **300 frames** per
sample (the six `t*`/`x*` stress samples are 30-frame files — every available frame is verified).
Each decoded frame is compared **byte-exact** against an FFmpeg software-decode reference in the
stream's native pixel format. Environment:

- **NVIDIA GeForce RTX 3060 (GA106)** — Vulkan Video and NVDEC (`cuvid`) columns
- **Intel Meteor Lake-P iGPU** — VAAPI column (iHD driver 26.1.2). Its Vulkan driver exposes no
  video decode queue in this environment, so the Vulkan column runs on GA106.

Legend: ✅ = 300/300 byte-exact | S(n/m) = sample has only n frames, all verified exact |
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

**Result: 40/40 decodable cells byte-exact; 0 failures.** 14 cells are HW-n/a.

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
- NVDEC 10/12-bit content decodes into P016 surfaces (the only >8-bit output format in the public
  cuvid API); readback scales to 8-bit with round+clamp, matching the other backends.
- Diagnostics: `VACC_PROBE_CUVID=1` dumps the full cuvid decoder-caps table; `VACC_VA_DUMP=1`
  dumps the exact VA-API picture/slice parameter buffers.

## Test Samples

The single master source is `assets/big_buck_bunney.h265` (Big Buck Bunny, 1920x1080,
300 frames). All 24 samples in `assets/samples/` are codec variants of that one video,
produced by the ffmpeg recipes embedded in `verify-all.py` (per-sample encoder options:
profile, chroma format, bit depth, GOP/stress flags).

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

## Unified Decode Example

```bash
# Decode with a chosen backend; prints pts/size/pixel-hash per frame
./target/release/examples/decode -b <vulkan|nvdec|vaapi> -i <file> [-n frames] [-o outdir]
```

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

## Building

```bash
cargo build --release --examples   # NOTE: --examples is required; plain builds don't rebuild it

# Run the unified decode example (pts, size, pixel hash per frame)
./target/release/examples/decode -b <vaapi|vulkan|nvdec> -i <file.ivf|h264|h265> [-n frames] [-o outdir]
```

Backend requirements: Vulkan Video device (VAAPI also works on the same stack), NVIDIA driver with
`libnvcuvid.so`, and libva + a VAAPI driver (iHD/Mesa) respectively.

## Reference

This library is based on the [Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples) from Khronos:

- [Vulkan Video Deep Dive](https://www.khronos.org/assets/uploads/apis/Vulkan-Video-Deep-Dive-Apr21.pdf)
- [Vulkan Video Extensions Spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html)
- [NVIDIA Vulkan Video Driver](https://developer.nvidia.com/vulkan-driver)

## License

MIT OR Apache-2.0
