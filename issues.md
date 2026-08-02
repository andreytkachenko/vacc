# vk-video Rust Rewrite — Issue Report

> Generated: 2026-05-31  
> Scope: Comparison of Rust rewrite (`crates/`) against C++ reference (`Vulkan-Video-Samples/`)  
> Build status: **64 compilation errors**, 20+ warnings

---

## Table of Contents

1. [Compilation Errors](#1-compilation-errors)
2. [Missing Features vs C++](#2-missing-features-vs-c)
3. [Parser Issues](#3-parser-issues)
4. [Vulkan Layer Issues](#4-vulkan-layer-issues)
5. [Type / Data Structure Issues](#5-type--data-structure-issues)
6. [Memory Safety Concerns](#6-memory-safety-concerns)
7. [Missing Infrastructure](#7-missing-infrastructure)
8. [Design / Architecture Issues](#8-design--architecture-issues)
9. [Example Issues](#9-example-issues)
10. [Code Quality](#10-code-quality)
11. [Priority Summary](#11-priority-summary)

---

## 1. Compilation Errors

**Status: 🔴 BLOCKER** — The project does not compile. 63 errors across `vk-video-vulkan`.

### 1.1 ash API Mismatches (ash 0.38)

Many methods/fields used in the code do not exist in `ash` 0.38. The code appears written for a different ash version or hand-rolled bindings.

| Error | Location | Fix |
|-------|----------|-----|
| `no method named 'cmd_pipeline_barrier_2'` | `h264.rs`, `h265.rs`, `av1.rs` | ash 0.38 may not expose this; check feature flags or use raw dispatch |
| `no associated item named 'VIDEO_DECODE_WRITE'` | `h264.rs`, `h265.rs`, `av1.rs` | Use `AccessFlags2KHR::VIDEO_DECODE_WRITE` or raw value |
| `no method named 'video_session'` on session params create info | `h264.rs:99`, `h265.rs:99`, `av1.rs:99` | Field name may differ in ash 0.38 builder |
| `no method named 'std_s_p_s_count'` | `h264.rs:133` | Builder method name mismatch |
| `no method named 'reference_session_parameters'` | `h264.rs`, `h265.rs`, `av1.rs` | Builder method name mismatch |
| `no method named 'p_std_picture_info'` | `h264.rs`, `h265.rs`, `av1.rs` | Builder method name mismatch |
| `no method named 'get_physical_device_video_format_properties'` | `device.rs` | May need `get_physical_device_video_format_properties2` or raw dispatch |
| `no method named 'reset_fence'` | `pipeline.rs` | ash 0.38 API change |
| `no method named 'is_null'` on Fence/CommandPool | `pipeline.rs` | Use `handle() == 0` or check ash API |
| `cannot find type 'StdVideoH264SequenceParameterSet' in ash::vk` | `h264.rs` | These types are NOT in ash; your `codec_types.rs` defines them but imports are wrong |
| `cannot find value 'caps'` | `device.rs` | Variable scoping issue in capability query |

### 1.2 Pod Derive Failures (Struct Padding)

| Error | Location | Issue |
|-------|----------|-------|
| `derive(Pod) was applied to a type with padding` | `codec_types.rs:289` | `StdVideoDecodeH264PictureInfo` has padding |
| `derive(Pod) was applied to a type with padding` | `codec_types.rs:341` | `StdVideoDecodeH265PictureInfo` has padding |
| Multiple Pod failures | `codec_types.rs` | Several codec structs have non-trivial padding |

**Fix**: Use `#[repr(C, align(N))]` with explicit padding fields, or use `Zeroable` only and manual layout verification. The Vulkan standard codec structures have specific layouts that must match exactly.

### 1.3 Duplicate Fields

| Error | Location | Issue |
|-------|----------|-------|
| `field 'num_extra_slice_header_bits' is already declared` | `codec_types.rs` | `StdVideoH265PictureParameterSet` has duplicate field |
| `field 'num_extra_slice_header_bits' specified more than once` | `codec_types.rs` | Same struct, same field |

### 1.4 Missing Default for Large Arrays

| Error | Location | Issue |
|-------|----------|-------|
| `the trait bound [u32; 1024]: Default is not satisfied` | `codec_types.rs` | `StdVideoH265VideoParameterSet` has `[u32; 1024]` arrays |

**Fix**: Implement `Default` manually or use `#[default]` attribute with bytemuck 1.15+.

---

## 2. Missing Features vs C++

### 2.1 DPB (Decoded Picture Buffer) Management — 🔴 CRITICAL

**C++**: `VulkanVideoFrameBuffer` provides full DPB lifecycle:
- Image pool allocation with multiple image types (DPB, output, linear, filter, film grain)
- Reference picture tracking with slot management
- Decode order vs display order tracking
- Frame synchronization with fences/semaphores/timeline semaphores
- Image resource views (per-layer views for array images)

**Rust**: `DpbImage` is a simple struct with no lifecycle management:
```rust
pub struct DpbImage {
    pub image: ash::vk::Image,
    pub image_view: ash::vk::ImageView,
    pub layout: ash::vk::ImageLayout,
    pub in_use: bool,
    pub poc: i32,
    pub is_reference: bool,
    pub slot_index: u32,
}
```
No reference picture list building, no DPB reordering, no display queue.

### 2.2 Reference Picture List Building — 🔴 CRITICAL

**C++**: Each codec parser builds reference picture lists from slice headers:
- H.264: `ref_pic_list_reordering`, `DecRefPicMarking`
- H.265: `ref_pic_set`, long-term reference pictures
- AV1: `ref_frame_idx`, frame refresh flags

**Rust**: Decode commands use hardcoded empty reference lists. The `VideoDecodeInfoKHR` has no `dpbRefPicture` populated.

### 2.3 Frame Queue / Reordering — 🔴 CRITICAL

**C++**: `VulkanVideoFrameBuffer` maintains:
- Decode order queue (`QueuePictureForDecode`)
- Display order queue (`DequeueDecodedPicture`)
- PTS queue for timestamp management
- `MAX_DELAY = 32` frame delay buffer

**Rust**: No frame queue. `VideoPipeline::decode_frame()` returns immediately with no reordering.

### 2.4 Synchronization Infrastructure — 🟠 HIGH

**C++**: Full synchronization:
- `VulkanFenceSet` — per-frame fences
- `VulkanSemaphoreSet` — per-frame semaphores
- Timeline semaphores for HW load balancing across queues
- `FrameSynchronizationInfo` with decode/filter/display timeline values

**Rust**: Single fence per pipeline, no semaphores, no timeline synchronization.

### 2.5 Post-Processing / YUV Filter — 🟡 MEDIUM

**C++**: `VulkanFilterYuvCompute` provides:
- YUV to RGB conversion via compute shaders
- YCbCr sampler conversion
- Multiple filter types (YCBCRCOPY, YCBCRTOBGR, etc.)

**Rust**: No compute shader pipeline, no YUV conversion.

### 2.6 Picture Parameter Update Queue — 🟠 HIGH

**C++**: `VkParserVideoPictureParameters` queues parameter updates:
- `AddPictureParametersToQueue` — deferred updates
- `FlushPictureParametersQueue` — batch update
- Tracks used VPS/SPS/PPS IDs with bitsets
- Validates parameter set hierarchy before update

**Rust**: Immediate parameter updates with no queuing, no validation.

### 2.7 Bitstream Buffer Pool — 🟡 MEDIUM

**C++**: `VulkanBitstreamBufferImpl` with:
- Proper alignment handling
- Stream markers for random access
- Pool management via `VulkanVideoRefCountedPool`
- Dynamic resizing

**Rust**: `BitstreamBuffer` has basic pool but no stream markers, no dynamic resizing.

### 2.8 Hardware Load Balancing — 🟢 LOW

**C++**: Multi-queue decode with timeline semaphores for load balancing.

**Rust**: Single queue only.

---

## 3. Parser Issues

### 3.1 H.264 Parser (`vk-video-parser/src/h264.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 No DPB state tracking | CRITICAL | No `frame_num`, `pic_order_cnt` tracking across frames. C++ tracks `prevFrameNum`, `prevPicOrderCntLsb`, `prevPicOrderCntMsb`, `PrevRefFrameNum`. |
| 🔴 No reference picture list | CRITICAL | Missing `RefPicList0/1` construction from slice headers. |
| 🔴 No IDR/sequence handling | CRITICAL | No DPB flush on IDR frames. C++ has `flush_decoded_picture_buffer()`. |
| 🟠 Incomplete SPS parsing | HIGH | Missing VUI parameters (frame rate, HRD, timing). C++ parses full VUI including `vui_parameters_present_flag` contents. |
| 🟠 SPS `max_frame_num` stored as u32 | HIGH | Rust `H264Sps::max_frame_num` is u32 but Vulkan `StdVideoH264SequenceParameterSet` doesn't have this field — it's computed from `log2_max_frame_num_minus4`. |
| 🟠 No scaling list parsing | MEDIUM | `skip_scaling_list` is a stub that doesn't properly advance bit position. |
| 🟡 No SEI parsing | LOW | C++ parses SEI for timing, HDR metadata. |
| 🟡 No slice header POC computation | MEDIUM | `delta_pic_order_cnt` not parsed from slice header. |

### 3.2 H.265 Parser (`vk-video-parser/src/h265.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 `update_session_parameters` is no-op | CRITICAL | Sends 0 VPS/SPS/PPS to Vulkan session (h265.rs:146-157). Decoder will fail. |
| 🔴 No slice header parsing | CRITICAL | No `PicTiming`, `SliceHeader` parsing. No POC computation. |
| 🔴 No DPB state tracking | CRITICAL | No reference picture set tracking, no `NumDeltaPocs` handling. |
| 🟠 Incomplete SPS parsing | HIGH | Missing: scaling lists, short/long term ref pic sets, VUI parameters, `sps_temporal_mvp_enabled_flag`, `sps_strong_intra_smoothing_enabled_flag`. |
| 🟠 Incomplete VPS parsing | HIGH | Only 5 fields parsed. Missing: `vps_max_dec_pic_buffering_minus1`, `vps_max_num_reorder_pics`, `vps_max_latency_increase_plus1`, timing info, HRD. |
| 🟡 No tiles/CTU parsing | LOW | Tiles and CTU-related fields skipped. |

### 3.3 AV1 Parser (`vk-video-parser/src/av1.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 No OBU parsing | CRITICAL | AV1 uses OBU (Open Bitstream Units), not NAL units with start codes. The parser reads raw bytes as if the sequence header starts at byte 0 of the packet. |
| 🔴 No frame header parsing | CRITICAL | AV1 frame headers contain critical decode info (frame type, reference frames, tile info, loop filter, quantization). None parsed. |
| 🔴 No OBU extraction | CRITICAL | No code to extract OBUs from the bitstream (AV1 has its own framing with `obu_type` and `obu_size` fields). |
| 🟠 Naive sequence header parsing | HIGH | Reads raw bytes with hardcoded bit positions. Should use proper OBU framing with `has_size_field` parsing. |
| 🟠 Missing sequence header fields | HIGH | Missing: `monochrome`, `color_description_present_flag`, `film_grain_params_present`, timing info, `decoder_model_info`. |
| 🟡 `is_av1()` detection is unreliable | LOW | Checks last byte for `0x9E` — this is not how AV1 detects streams. |

### 3.4 NAL Unit / Start Code Issues

| Issue | Severity | Details |
|-------|----------|---------|
| 🟠 `find_next_start_code` may miss valid start codes | HIGH | The "not preceded by 0x00" check is overly restrictive. A 3-byte start code `0x00 0x00 0x01` at the start of a NAL unit is valid even if preceded by 0x00 from the previous NAL's trailing data. |
| 🟡 No AV1 OBU support | MEDIUM | `nal.rs` only handles H.264/H.265 NAL units. AV1 needs OBU parsing. |

---

## 4. Vulkan Layer Issues

### 4.1 Device Creation (`device.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 Duplicate extensions | HIGH | `VIDEO_DECODE_EXTENSIONS` and `VIDEO_STD_EXTENSIONS` overlap. `VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`, `VK_KHR_video_decode_av1` appear in both arrays. Creating device with duplicate extensions is undefined behavior. |
| 🟠 Hardcoded alignment values | HIGH | `min_bitstream_buffer_offset_alignment` and `min_bitstream_buffer_size_alignment` hardcoded to 256. Should query from `VkVideoDecodeCapabilitiesKHR` per codec. |
| 🟠 `get_codec_capabilities` returns None | HIGH | Stub function that never queries codec-specific capabilities. Alignment values are never populated from hardware. |
| 🟠 No VP9 capability query | MEDIUM | VP9 is listed in `VideoCodec` enum but never queried. |
| 🟡 `VIDEO_DECODE_QUEUE` extension missing | LOW | Should check for `VK_KHR_video_queue` as a prerequisite (though `VK_KHR_video_decode_queue` implies it on most drivers). |
| 🟡 Debug eprintln statements | LOW | `device.rs` has `eprintln!` debug statements scattered throughout. Should use `tracing::debug!`. |

### 4.2 Session Management (`session.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 Defaults to H264 when profile info is None | HIGH | `VideoSession::create()` defaults to H264 High profile when `codec_profile_info` is `None` (line 127). This silently creates wrong session type. |
| 🔴 `VideoSessionParameters::new()` creates null handle | CRITICAL | Constructor creates a null `VideoSessionParametersKHR` handle. There's no `create()` method that actually calls `vkCreateVideoSessionParametersKHR`. The session parameters are only created inside codec-specific decoders. |
| 🟠 No session compatibility checking | MEDIUM | C++ `VulkanVideoSession::IsCompatible()` checks if existing session can be reused. Rust always creates new session. |
| 🟠 No session recreation on format change | MEDIUM | C++ handles format changes by waiting for idle and recreating session. Rust has no equivalent. |

### 4.3 Decode Command Recording (`h264.rs`, `h265.rs`, `av1.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 Wrong aspect mask for image barriers | CRITICAL | Uses `ImageAspectFlags::COLOR` for semi-planar YUV images. Should use `PLANE_0` for the image view barrier. `COLOR` is not valid for multi-plane formats. |
| 🔴 No DPB setup picture | CRITICAL | `VideoDecodeInfoKHR::dpbSetupPicture` is never set. Required for the first frame and after IDR. |
| 🔴 No DPB reference pictures | CRITICAL | `VideoDecodeInfoKHR::dpbRefPicture` list is empty. Decoder won't have reference frames. |
| 🔴 Hardcoded picture info | CRITICAL | `StdVideoDecodeH264PictureInfo` / `StdVideoDecodeH265PictureInfo` filled with zeros. `frame_num`, `pic_order_cnt_lsb`, `coded_pic_size_in_mbs` etc. must come from parsed slice headers. |
| 🔴 No reference picture memory barriers | CRITICAL | Only the output image gets a barrier. Reference DPB images need `VIDEO_DECODE_DPB_KHR` layout transitions. |
| 🟠 Image barrier uses UNDEFINED every frame | HIGH | `old_layout` is always `UNDEFINED`. After first decode, layout should be `VIDEO_DECODE_DST_KHR` or `VIDEO_DECODE_DPB_KHR`. |
| 🟠 Command buffer begin/end inside decode | HIGH | `record_decode_command()` calls `begin_command_buffer` and `end_command_buffer`. This prevents recording multiple operations in one command buffer. C++ begins/ends at higher level. |
| 🟠 No `vkCmdResetCommandBuffer` | MEDIUM | Command buffer is reused without reset. Should call `reset_command_buffer` before reuse. |
| 🟡 Missing `codedOffset` for interlaced | LOW | `codedOffset` is hardcoded to `(0,0)`. Should be adjusted for interlaced/field modes. |

### 4.4 Bitstream Buffer (`buffer.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🟠 No alignment enforcement | HIGH | Buffer size and offset alignment not enforced. C++ aligns to `minBitstreamBufferOffsetAlignment` and `minBitstreamBufferSizeAlignment`. |
| 🟡 No stream marker tracking | LOW | `add_stream_marker()` and `reset_stream_markers()` are no-ops. |

### 4.5 Image Management (`image.rs`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🟠 Only PLANE_0 image view | HIGH | Creates image view only for `PLANE_0`. For decode operations, the full image is used, not individual planes. |
| 🟠 `read_back_yuv` is a no-op | HIGH | Returns empty vectors. No staging image copy, no transfer commands. |
| 🟡 Staging image mapping order wrong | LOW | `map_memory(offset, size, ...)` has offset and size swapped. Should be `map_memory(memory, 0, WHOLE_SIZE, ...)`. |

---

## 5. Type / Data Structure Issues

### 5.1 `codec_types.rs` — Vulkan Standard Codec Structures

These must match the exact layout from `vulkan_video_codecs_standard_codec_info.h`.

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 Pod derive fails on padded structs | CRITICAL | `StdVideoDecodeH264PictureInfo`, `StdVideoDecodeH265PictureInfo`, `StdVideoDecodeAv1PictureInfo` have padding that violates `Pod` requirements. |
| 🔴 Duplicate field in H265 PPS | CRITICAL | `num_extra_slice_header_bits` declared twice in `StdVideoH265PictureParameterSet`. |
| 🔴 Large arrays don't impl Default | CRITICAL | `[u32; 1024]` and `[[u32; 1024]; 32]` in `StdVideoH265VideoParameterSet` don't implement `Default`. |
| 🟠 `StdVideoH264SequenceParameterSet` field types | HIGH | Boolean fields use `u8` but some fields that should be `u32` (like `profile_idc`) are typed as `u32` aliases. Need to verify exact types against Vulkan spec. |
| 🟠 Missing codec types | MEDIUM | No `StdVideoH264ReferenceFrames`, `StdVideoDecodeH264ReferenceFrames`, `StdVideoDecodeH264DPB`, `StdVideoH265DPB`, `StdVideoDecodeH265ReferenceFrames`, `StdVideoDecodeH265DPB`, `StdVideoAV1ReferenceFrame`, `StdVideoDecodeAV1ReferenceFrames`, `StdVideoDecodeAV1DPB`. |
| 🟡 No VP9 codec types | LOW | VP9 standard codec structures not defined. |

### 5.2 Core Types (`vk-video-core`)

| Issue | Severity | Details |
|-------|----------|---------|
| 🟠 `H264Sps::max_frame_num` not in Vulkan struct | HIGH | This field is computed, not stored in `StdVideoH264SequenceParameterSet`. The Rust struct has it but the Vulkan struct doesn't. |
| 🟠 `H264PictureLayout` as flag bits | MEDIUM | `H264PictureLayout` is an enum with bit values but should be `bitflags!` for proper flag operations. |
| 🟡 `VideoCodec` repr values | LOW | Enum values don't match `VkVideoCodecOperationFlagBitsKHR` bit positions (they should be bit flags, not sequential values). |

---

## 6. Memory Safety Concerns

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 Heavy `std::mem::transmute` usage | CRITICAL | All Vulkan function pointer dispatch uses `transmute`. Should use `std::ffi::FnPtr` or `ash::extensions` for safe dispatch. |
| 🔴 Raw pointer casts for codec structs | CRITICAL | `&pic_info as *const StdVideoDecodeH264PictureInfo as *const std::ffi::c_void` — double cast through `c_void` loses type safety. |
| 🟠 `mapped_ptr: Option<*mut u8>` | HIGH | Raw mutable pointers without invariants. Should use `NonNull<u8>` with safety documentation. |
| 🟠 No RAII for Vulkan handles | HIGH | `ash::Device` and `ash::Instance` are cloned but not properly managed. No guarantee of destruction order. |
| 🟠 `VideoSession` Drop may use stale proc addr | HIGH | `Drop` implementation calls `get_device_proc_addr` on a potentially destroyed device. |
| 🟡 `debug_interface: Option<*mut std::ffi::c_void>` | MEDIUM | Raw pointer in `FrameSyncInfo` with no safety contract. |

---

## 7. Missing Infrastructure

C++ components with no Rust equivalent:

| C++ Component | Purpose | Rust Status |
|--------------|---------|-------------|
| `VulkanVideoFrameBuffer` | Frame buffer management, image pools, sync | ❌ Missing |
| `VulkanBitstreamBufferImpl` | Bitstream buffer with alignment | ❌ Partial (`BitstreamBuffer` lacks alignment) |
| `VkParserVideoPictureParameters` | Parameter queue + update | ❌ Missing |
| `VulkanVideoSession` | Session with compatibility check | ❌ Partial (`VideoSession` lacks compat check) |
| `VulkanFilterYuvCompute` | YUV compute filter | ❌ Missing |
| `VulkanVideoDisplayQueue` | Decoded frame output queue | ❌ Missing |
| `VulkanSemaphoreSet` | Per-frame semaphores | ❌ Missing |
| `VulkanFenceSet` | Per-frame fences | ❌ Missing |
| `VulkanQueryPoolSet` | Timing queries | ❌ Missing |
| `VulkanDescriptorSetLayout` | Compute shader descriptors | ❌ Missing |
| `VulkanComputePipeline` | Compute pipeline | ❌ Missing |
| `VulkanSamplerYcbcrConversion` | YCbCr sampler | ❌ Missing |
| `VulkanCommandBufferPool` | Command buffer pool | ❌ Missing |
| `VulkanVideoImagePool` | Image pool with ref counting | ❌ Missing |
| `VulkanBufferPool` | Buffer pool with ref counting | ❌ Missing |
| `VkThreadPool` | Thread pool | ❌ Missing |
| `VulkanDeviceContext` | Device abstraction with dispatch table | ❌ Partial (`VulkanDevice`) |
| `YCbCrConvUtilsCpu` | CPU YCbCr conversion | ❌ Missing |

---

## 8. Design / Architecture Issues

### 8.1 Duplicate Error Types

Both `vk-video-core` and `vk-video-vulkan` define their own `VideoError`/`VideoResult`. Should have a single error hierarchy.

### 8.2 Session and Decoder Separation

In C++, `VkVideoDecoder` owns the `VulkanVideoSession`. In Rust, `VideoSession` and `H264Decoder` are separate objects that must be manually linked via `set_session()`. This is error-prone.

**Recommendation**: Have the decoder own the session, or use a builder pattern that creates both together.

### 8.3 `DecoderWrapper` Enum

The `DecoderWrapper` enum (`H264`, `H265`, `Av1`) adds indirection. A trait-based approach would be more idiomatic:

```rust
pub trait VideoDecoder {
    fn record_decode_command(&self, ...) -> VideoResult<()>;
    fn create_session_parameters(&mut self) -> VideoResult<VideoSessionParameters>;
    fn update_session_parameters(&self, ...) -> VideoResult<()>;
}
```

### 8.4 `VideoPipeline` Scope Creep

`VideoPipeline` tries to do everything (device management, session, decoder, bitstream buffers, output images) but is incomplete. The C++ code separates these concerns:
- `VulkanDeviceContext` — device/queue management
- `VulkanVideoFrameBuffer` — image/frame management
- `VkVideoDecoder` — decode orchestration
- `NvVkDecodeFrameData` — per-frame data

### 8.5 Example Duplicates Crate Code

`vulkan_decode.rs` reimplements device creation, session creation, and decode recording instead of using the `vk-video-vulkan` crate's `VideoPipeline`. This suggests the crate API is not yet usable.

---

## 9. Example Issues

| Issue | Severity | Details |
|-------|----------|---------|
| 🔴 `vulkan_decode.rs` doesn't use `VideoPipeline` | HIGH | Reimplements everything from scratch. The crate's pipeline API is not demonstrated. |
| 🔴 Only decodes first frame | HIGH | No multi-frame decode loop, no DPB management. |
| 🔴 H265 decode uses null picture info | CRITICAL | `p_std_picture_info` is `std::ptr::null()` for H.265. Will produce invalid results. |
| 🟠 No proper cleanup | MEDIUM | Some Vulkan objects may leak on error paths. |
| 🟠 No error recovery | MEDIUM | No handling of decode failures, no retry logic. |
| 🟡 FFmpeg comparison is basic | LOW | Only compares first frame statistics, not pixel-level comparison. |

---

## 10. Code Quality

### 10.1 Warnings

| Warning | Location | Fix |
|---------|----------|-----|
| Unused doc comment on `bitflags!` | `codec.rs:81-82` | Remove doc comment or move inside macro |
| Unnecessary parentheses | `h265.rs` | Remove parens |
| Unused imports | `av1.rs`, `h264.rs` | Remove unused imports |
| Deprecated `enabled_layer_count` | `device.rs` | Vulkan layers are deprecated; remove |
| Unused variable `builder` | `device.rs:223` | Prefix with `_` |
| Unused variable `output` | `pipeline.rs:380` | Prefix with `_` |
| Unused field `nal_header_extension` | `h264.rs` | Remove or use |
| Unused function `read_flag` | `h264.rs` | Remove or use |

### 10.2 Code Duplication

| Duplicated Code | Locations |
|----------------|-----------|
| SPS/PPS conversion functions | `h264.rs` + `vulkan_decode.rs` |
| Memory type finding | `buffer.rs`, `image.rs`, `device.rs`, `vulkan_decode.rs` |
| Start code detection logic | `nal.rs` (good, centralized) |
| Bit reading functions (`read_ue`, `read_se`, `read_u`) | `h264.rs` + `h265.rs` (duplicated) |
| Decode command recording structure | `h264.rs`, `h265.rs`, `av1.rs` (nearly identical) |

---

## 11. Priority Summary

### 🔴 P0 — Must Fix Before Anything Works

1. **Fix compilation errors** (63 errors) — ash API mismatches, Pod derive failures, duplicate fields
2. **Fix H.265 `update_session_parameters` no-op** — Currently sends 0 parameter sets
3. **Fix image barrier aspect mask** — `COLOR` → `PLANE_0` for semi-planar
4. **Add DPB setup picture and reference pictures** to decode info
5. **Fix duplicate extensions** in device creation
6. **Populate picture info from parser** instead of hardcoded zeros

### 🟠 P1 — Needed for Functional Decoder

7. **Implement DPB management** — reference picture tracking, slot management
8. **Implement frame queue/reordering** — decode order vs display order
9. **Complete H.264/H.265 parsers** — slice headers, POC computation, reference lists
10. **Rewrite AV1 parser** — proper OBU parsing, frame headers
11. **Add proper synchronization** — per-frame fences/semaphores
12. **Fix `VideoSessionParameters` creation** — add `create()` method

### 🟡 P2 — Needed for Production Quality

13. **Implement `VulkanVideoFrameBuffer` equivalent** — image pools, sync
14. **Implement `VkParserVideoPictureParameters` equivalent** — parameter queue
15. **Add YUV readback** — staging images, transfer commands
16. **Add post-processing pipeline** — YUV compute filter
17. **Consolidate error types** — single error hierarchy
18. **Refactor architecture** — decoder owns session, trait-based decoders
19. **Remove `transmute` usage** — safe function pointer dispatch

### 🟢 P3 — Nice to Have

20. **VP9 support** — parser + Vulkan decoder
21. **Hardware load balancing** — multi-queue with timeline semaphores
22. **SEI parsing** — HDR metadata, timing info
23. **Interlaced video support** — field-based decoding
24. **AV1 film grain** — hardware film grain support
25. **SVC/MVC support** — scalable/multiview extensions

---

## Appendix: File-by-File Status

| File | Status | Notes |
|------|--------|-------|
| `vk-video-core/src/lib.rs` | ✅ OK | Good module organization |
| `vk-video-core/src/codec.rs` | ✅ OK | Minor warning about doc comment |
| `vk-video-core/src/error.rs` | ⚠️ PARTIAL | Duplicate error type with vulkan crate |
| `vk-video-core/src/format.rs` | ✅ OK | Comprehensive format mapping |
| `vk-video-core/src/frame.rs` | ⚠️ PARTIAL | `FrameSyncInfo` has raw pointer |
| `vk-video-core/src/picture.rs` | ✅ OK | Good SPS/PPS/VPS structs |
| `vk-video-core/src/session.rs` | ⚠️ PARTIAL | Session params are opaque handles |
| `vk-video-parser/src/lib.rs` | ✅ OK | Good trait design |
| `vk-video-parser/src/bitstream.rs` | ⚠️ PARTIAL | Stream markers are no-ops |
| `vk-video-parser/src/h264.rs` | 🔴 INCOMPLETE | No DPB tracking, no ref lists |
| `vk-video-parser/src/h265.rs` | 🔴 INCOMPLETE | update_session_parameters is no-op |
| `vk-video-parser/src/av1.rs` | 🔴 INCOMPLETE | No OBU parsing, no frame headers |
| `vk-video-parser/src/nal.rs` | ✅ OK | Good start code / NAL handling |
| `vk-video-vulkan/src/lib.rs` | ✅ OK | Good module organization |
| `vk-video-vulkan/src/device.rs` | 🔴 BROKEN | Won't compile, duplicate extensions |
| `vk-video-vulkan/src/session.rs` | 🔴 BROKEN | Won't compile, defaults to H264 |
| `vk-video-vulkan/src/codec_types.rs` | 🔴 BROKEN | Pod derive failures, duplicate fields |
| `vk-video-vulkan/src/h264.rs` | 🔴 BROKEN | Won't compile, missing ref pictures |
| `vk-video-vulkan/src/h265.rs` | 🔴 BROKEN | Won't compile, no-op param update |
| `vk-video-vulkan/src/av1.rs` | 🔴 BROKEN | Won't compile, no frame header info |
| `vk-video-vulkan/src/buffer.rs` | ⚠️ PARTIAL | No alignment enforcement |
| `vk-video-vulkan/src/frame.rs` | ⚠️ PARTIAL | Basic frame pool |
| `vk-video-vulkan/src/image.rs` | ⚠️ PARTIAL | read_back is no-op |
| `vk-video-vulkan/src/pipeline.rs` | 🔴 BROKEN | Won't compile, scope creep |
| `examples/vulkan_decode.rs` | 🔴 BROKEN | Duplicates code, H265 null pointer |
| `examples/basic_decode.rs` | ✅ OK | Good API demonstration |
