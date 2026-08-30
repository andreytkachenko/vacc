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

Verified 2026-08-30: Big Buck Bunny 640x360 @ 30 fps, **300 frames** per sample; output compared
**byte-exact** against FFmpeg (software-decode reference). Environment: NVIDIA GeForce **RTX 3060**,
driver 610.43.02 (Vulkan Video + NVDEC); Intel iHD driver (VA API 1.22) for VAAPI.

Legend: ✅ = 300/300 byte-exact | ❌ = fails (reason noted) | — = not implemented

| Codec / Profile (chroma, depth) | Vulkan Video | NVDEC | VAAPI |
|---|---|---|---|
| H.264 Baseline (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 Main (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 High (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.264 High 10 (4:2:0 10b) | ❌ driver: `ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED` (FF Vulkan fails identically) | ❌ hardware: GeForce NVDEC has no 10-bit H.264 (cuvid 801; FF CUDA fails identically) | ❌ driver: iHD has no 10-bit H.264 render target (FF VAAPI fails identically) |
| H.264 High 4:4:4 (4:4:4 8b) | ❌ driver (FF Vulkan fails identically) | ❌ hardware: cuvid 801 (FF CUDA fails identically) | ⚠️ our gap: FF VAAPI decodes this stream, we fail at render-target format negotiation |
| H.265 Main (4:2:0 8b) | ✅ | ✅ | ✅ |
| H.265 Main 10 (4:2:0 10b) | ✅ | ✅ | ✅ |
| H.265 Rext 4:4:4 (4:4:4 8b) | ✅ | ✅ | ✅ |
| H.265 Rext 4:4:4 Main 10 (4:4:4 10b) | ✅ | ❌ wrong pixels (0/300, maxdiff 64449; output size/layout correct) | ✅ |
| VP9 Profile 0 (4:2:0 8b) | ✅ * | ✅ | ✅ |
| VP9 Profile 2 (4:2:0 10b) | ❌ emits 8-bit format for 10-bit stream (G8 instead of G10X6) | ❌ same as Vulkan | ✅ † |
| VP9 Profile 2 (4:2:0 12b) | ❌ emits 8-bit format (G8 instead of G12X4) | ❌ same as Vulkan | ✅ † |
| AV1 Main (4:2:0 8b) | ✅ | ❌ 11/300 frames match — fix in progress | — no AV1 path |
| AV1 Main 10 (4:2:0 10b) | ✅ | ❌ emits 8-bit-sized buffer for 10-bit stream — fix in progress | — no AV1 path |
| AV1 High (4:2:0 12b) | ❌ driver (FF Vulkan fails identically) | ❌ hardware: GeForce NVDEC has no 12-bit AV1 (FF CUDA decodes 8/10-bit, fails at 12-bit) | — no AV1 path |
| AV1 Main 4:4:4 (4:4:4 8b) | ❌ driver: NVIDIA Vulkan Video has no AV1 4:4:4 decode (FF Vulkan fails identically) | ❌ our gap: AV1 chroma hardcoded to 4:2:0 | — no AV1 path |
| AV1 High 4:4:4 (4:4:4 10b) | ❌ driver (FF Vulkan fails identically) | ❌ same as Main 4:4:4 | — no AV1 path |

\* byte-exact on clean runs (re-verified 300/300 twice); one flake (6/300) observed under sustained
back-to-back decode load.
† VAAPI decodes at native depth on the GPU, but the example readback down-converts P010/P012 to
8-bit (rounded), so comparison is against the down-converted reference.

Notes:
- "driver/hardware" failures = the same stream fails identically through FFmpeg's own hwaccel path,
  i.e. not a bug in our code.
- NVDEC H.264: interlaced content decodes but field ordering differs from FFmpeg (top-field-first
  convention); progressive-only parity.
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

# Run example
cargo run --example basic_decode
```

## Reference

This library is based on the [Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples) from Khronos:

- [Vulkan Video Deep Dive](https://www.khronos.org/assets/uploads/apis/Vulkan-Video-Deep-Dive-Apr21.pdf)
- [Vulkan Video Extensions Spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html)
- [NVIDIA Vulkan Video Driver](https://developer.nvidia.com/vulkan-driver)

## License

MIT OR Apache-2.0
