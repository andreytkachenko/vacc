# vk-video

A Rust library for Vulkan hardware-accelerated video decoding, based on the [Khronos Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples).

Supports **H.264/AVC**, **H.265/HEVC**, **VP9**, and **AV1** decoding (Vulkan Video, NVIDIA NVDEC, and VAAPI backends — see [Decode Support Matrix](#decode-support-matrix)).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     vk-video (Workspace)                     │
├──────────────────┬──────────────────┬───────────────────────┤
│  vk-video-core   │ vk-video-parser  │   vk-video-vulkan     │
│  (Core types     │  (Bitstream      │   (Vulkan              │
│   & traits)      │   parsing)       │    implementation)    │
├──────────────────┴──────────────────┴───────────────────────┤
│                        Examples                              │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ basic_decode   │  │ full_pipeline │  │ yuv_processing │  │
│  └────────────────┘  └───────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Crate Overview

### `vk-video-core`
Core types and traits shared across all crates:
- `VideoCodec` - H.264, H.265, AV1, VP9 identification
- `VideoFormat` - Chroma subsampling, bit depth, profile info
- `PictureParametersSet` - SPS/PPS/VPS abstraction
- `DecodedFrame` - Output frame representation
- `VideoError` / `VideoResult` - Error handling

### `vk-video-parser`
Bitstream parsing for each codec:
- **H.264**: SPS, PPS, slice header parsing
- **H.265**: VPS, SPS, PPS, slice header parsing
- **AV1**: Sequence header parsing
- NAL unit extraction and start-code detection
- RBSP (Raw Byte Sequence Payload) handling
- Emulation prevention byte removal

### `vk-video-vulkan`
Vulkan implementation using `ash`:
- `VulkanDevice` - Device initialization with video decode support
- `VideoSession` - `VkVideoSessionKHR` management
- `BitstreamBuffer` - `VkBuffer` for compressed video data
- `H264Decoder` / `H265Decoder` / `Av1Decoder` - Codec-specific decoders
- `VideoPipeline` - End-to-end decode pipeline orchestration
- `DecodedFrame` - YCbCr output frames

## Quick Start

```toml
# Cargo.toml
[dependencies]
vk-video-core = { path = "vk-video-core" }
vk-video-parser = { path = "vk-video-parser" }
vk-video-vulkan = { path = "vk-video-vulkan" }
```

```rust
use vk_video_core::codec::VideoCodec;
use vk_video_parser::h264::H264Parser;
use vk_video_vulkan::{VideoDeviceBuilder, VideoPipeline, VideoCodec};

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

Verified 2026-08-31: Big Buck Bunny 640x360 @ 30 fps, **300 frames** per sample; output compared
**byte-exact** against FFmpeg (software-decode reference). Environment: NVIDIA **Tesla V100-SXM2-32GB**
(Volta), driver 580.159.04 — Vulkan Video + NVDEC, and the same driver's VA-API for the VAAPI column.
AV1 is not available on this GPU (no AV1 decode in the device caps) and is therefore not verified here.

Legend: ✅ = 300/300 byte-exact | ⚠️ = decodes but not byte-exact (rounding / depth down-convert) |
❌ = fails (reason noted) | — = not applicable. "driver/hardware" = the same stream fails identically
through FFmpeg's own hwaccel path (not a bug in our code); "our bug"/"our gap" = FF decodes it, we don't.

| Codec / Profile (chroma, depth) | Vulkan Video | NVDEC | VAAPI |
|---|---|---|---|
| H.264 Baseline (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 Main (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 High (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 gop1 / gop100 (I-only, long GOP) | ✅ | ✅ | ✅ |
| H.264 CRA open-GOP (CRA at 100/200, x264 native) | ✅ | ✅ | ✅ |
| H.264 High 10 (4:2:0 10b) | ❌ driver: profile not in device caps (FF Vulkan fails identically) | ❌ hardware: no 10-bit H.264 on Volta NVDEC (FF CUDA fails identically); our decoder is 8-bit-only regardless | ❌ driver: VAProfile not supported |
| H.264 High 4:2:2 (4:2:2 10b) | ❌ driver: no 4:2:2 in device caps | — (decoder is 8-bit only) | ❌ driver: RT format not supported |
| H.264 High 4:4:4 (4:4:4 8b/10b) | ❌ driver: no 4:4:4 in device caps (FF Vulkan fails identically) | ❌ hardware: cuvid has no 4:4:4 H.264; our decoder is 4:2:0-only | ⚠️ our gap: FF VAAPI decodes it, we fail at render-target format negotiation |
| H.265 Main (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.265 Main 10 (4:2:0 10b) | ✅ | ✅ | ✅ |
| H.265 gop100 (long GOP, native CRA at 100/200) | ✅ | ✅ | ✅ |
| H.265 CRA (no IDR at all; converted + native CRA) | ✅ | ✅ | ✅ |
| H.265 Rext 4:2:2 (4:2:2 10b) | ❌ driver: no HEVC 4:2:2 in device caps | ❌ hardware: cuvid 801 (no HEVC 4:2:2) | ❌ driver: VAProfile not supported |
| H.265 Rext 4:4:4 (4:4:4 8b) | ❌ driver: no HEVC 4:4:4 in device caps | ❌ hardware: cuvid 801 (no HEVC 4:4:4) | ❌ driver: resource allocation failed |
| H.265 Rext 4:4:4 Main 10 (4:4:4 10b) | ❌ driver (as 8-bit 4:4:4) | ❌ hardware: cuvid 801 | ❌ driver (as 8-bit 4:4:4) |
| H.265 all-IDR / Rext gop1 (4:2:0 8b) | ✅ | ✅ \*\*\* | ✅ |
| VP9 Profile 0 (4:2:0 8b) | ✅ \*\* | ✅ | ⚠️ near-exact (≈55 dB; ≤45 luma px differ) |
| VP9 Profile 2 (4:2:0 10b) | ❌ hardware: V100 Vulkan Video VP9 10-bit yields wrong pixels even with a spec-correct session | ⚠️ decodes; readback down-converts to 8-bit (≈54 dB vs 8-bit ref) | ⚠️ decodes at native p010le on GPU; readback down-converts to 8-bit |
| VP9 Profile 2 (4:2:0 12b) | ❌ hardware (as 10-bit) | ⚠️ decodes; down-converts to 8-bit (≈54 dB) | ⚠️ decodes at native p012le; readback down-converts to 8-bit |

\*\* VP9 hidden reference-only frames (`show_frame=0`) are decoded but not displayed; the decoder now
consumes a generous AU budget and stops once the requested display-frame count is reached, so all
300 display frames are produced (was previously 268/274 — fixed).

\*\*\* Fixed: every picture in an all-IDR HEVC stream has POC 0, so POC-ordered presentation stalled
after frame 0 and NVDEC recycled the pending surfaces. When POC never advances, presentation now
falls back to decode order (which equals display order there).

CRA samples: `h264_cra` is x264 native open-GOP (IDR at frame 0, CRA at 100/200); `hevc_cra` is
produced by `samples/make_cra.py`, which converts the IDR NAL to a true `CRA_NUT` (inserting
`pic_order_cnt_lsb` + an empty inline RPS and repairing the slice-alignment padding), so the stream
contains no IDR at all.

Notes:
- Volta (V100) vs Ampere (RTX 3060): V100 **lacks** HEVC 4:2:2 and 4:4:4 decode entirely (Ampere had
  4:4:4), and its Vulkan Video VP9 10/12-bit path is broken at the driver level. All other cells match
  the Ampere results.
- NVDEC H.264: interlaced content decodes but field ordering differs from FFmpeg (top-field-first);
  progressive-only parity.
- The VP9 10/12-bit "⚠️" cells decode correctly on the GPU; the non-exactness is the example readback
  down-converting P010/P012 to 8-bit (rounded), so they compare against an 8-bit reference.
- NVDEC features verified separately: B-frame reordering/flush (byte-perfect vs ffmpeg), decoder
  reset/reconfiguration mid-stream, multi-GPU selection, concurrent decoder instances.

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
# Clone with submodules (for Vulkan-Video-Samples reference)
git clone --recursive https://github.com/andrey/vk-video.git
cd vk-video

# Build
cargo build

# Run example (unified decode: pts, size, pixel hash per frame)
cargo run --release -p examples --example decode -- -b <vaapi|vulkan|nvdec> -i <file.ivf|h264|h265>
```

## Reference

This library is based on the [Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples) from Khronos:

- [Vulkan Video Deep Dive](https://www.khronos.org/assets/uploads/apis/Vulkan-Video-Deep-Dive-Apr21.pdf)
- [Vulkan Video Extensions Spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html)
- [NVIDIA Vulkan Video Driver](https://developer.nvidia.com/vulkan-driver)

## License

MIT OR Apache-2.0
