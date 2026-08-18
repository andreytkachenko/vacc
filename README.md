# vk-video

A Rust library for Vulkan hardware-accelerated video decoding, based on the [Khronos Vulkan-Video-Samples](https://github.com/KhronosGroup/Vulkan-Video-Samples).

Supports **H.264/AVC**, **H.265/HEVC**, and **AV1** decoding using Vulkan's video decode extensions.

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

## Supported Codecs

| Codec | Parser | Vulkan Decoder | NVDEC Decoder | Profile Support |
|-------|--------|---------------|---------------|-----------------|
| H.264/AVC | ✅ | ✅ | ✅ | Baseline, Main, High |
| H.265/HEVC | ✅ | ✅ | ❌ | Main, Main10 |
| AV1 | ✅ | ✅ | ❌ | Main, High, Professional |
| VP9 | 🔄 | 🔄 | ❌ | Profile 0-3 |

✅ = Implemented | 🔄 = Planned | ❌ = Not yet implemented

### NVDEC Decoder (`nvdec-decode`)

Hardware-accelerated H.264 decoding via NVIDIA CUVID/NVDEC API.

**What works:**
- H.264 Main, High, and Baseline profiles (8-bit, progressive)
- Various resolutions and frame rates (tested: 1920x1080, 640x480, 320x240)
- Multi-segment streams with resolution changes mid-stream
- B-frame reordering (NVDEC parser handles DPB and display order)
- Proper drain/flush at end-of-stream (2136 frames, byte-perfect vs ffmpeg)
- Decoder reset/reconfiguration mid-stream
- Multi-GPU support (select specific CUDA device)
- Concurrent thread safety (decoder instances per thread)

**What doesn't work on V100:**
- 10-bit H.264 (V100 NVDEC hardware limitation)
- High 4:4:4 Intra profile (V100 NVDEC hardware limitation)

**Known limitations:**
- Interlaced content: field ordering differs from ffmpeg (NVDEC uses top-field-first convention)
- H.264 only — HEVC and AV1 NVDEC support not yet implemented
- Requires NVIDIA GPU with NVDEC hardware (tested on Tesla V100, Driver 580.159.04)

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
