# AV1 Pixel-Perfect Iteration Log

Goal: `vulkan_decode` example decodes `assets/big_buck_bunny_av1.ivf` pixel-perfect vs ffmpeg (dav1d).
300 frames, 1920x1080, 30fps. GPU: RTX 3060 (AV1 HW decode).

Commands (workdir /home/atkachenko/apps/vacc):
- Verify (builds + runs + compares): `python3 pixel-verify.py assets/big_buck_bunny_av1.ivf 20`
- Build only: `cargo build --release --example vulkan_decode`
- Run only: `target/release/examples/vulkan_decode assets/big_buck_bunny_av1.ivf 20` (writes /tmp/pixel_verify/big_buck_bunny_av1_frame_N.yuv)

Reference: /home/atkachenko/apps/Vulkan-Video-Samples/vk_video_decoder/libs/NvVideoParser/src/VulkanAV1Decoder.cpp
- show_existing_frame handling: lines 1780-1856 (ParseObuFrameHeader)

Key facts:
- IVF packets contain multiple OBUs; OBU header: bits7-3=type, bit2=ext_flag, bit1=has_size, bit0=reserved.
- OBU types: 1=SEQ_HDR, 2=TEMP_DELIM, 3=FRAME_HDR, 4=TILE_GROUP, 6=FRAME, 8=TILE_LIST.
- show_existing_frame: FRAME_HDR OBU, payload bit0=1, then 3-bit frame_to_show_map_idx → reuse that ref slot's decoded buffer, NO new decode.
- ffmpeg decodes exactly 300 frames.

---

## Iteration 0 (baseline)
- State: frame 0 KEY decoded (refs all -1), frame 1 INTER (partial refs), frame 2 = show_existing_frame (5-byte AU: TEMP_DELIM + 1-byte FRAME_HDR 0xc8) → treated as full frame → "Decode failed: Memory allocation failed".
- First failure: frame 2 (crash).
- Next: implement show_existing_frame (skip decode, reuse ref slot buffer as display frame).

## Iteration 1 (show_existing_frame)
- Baseline re-confirmed: frame 2 crash = "Memory allocation failed".
- Root cause: parser returns early for show_existing_frame (vk-video-parser/src/av1.rs:802-806) leaving `fh.frame_width/height = 0`; the show_existing branch (vk-video-vulkan/src/decoder.rs:960-993) called `readback_decoded_image` with those 0 dims → 0-size staging buffer → allocation fail.
- Fix (per-frame-buffer coded dims in Av1Decoder):
  - av1.rs: added `frame_buffer_dims: [(u32,u32);8]` + `set/get_frame_buffer_dims`; cleared in `reset_dpb`.
  - decoder.rs refresh loop (~1195): record `frame_coded_extent` for each refreshed frame buffer.
  - decoder.rs show_existing branch (~960): readback + `DecodedFrame.coded_width/height` now use `get_frame_buffer_dims(frame_to_show_map_idx)`, fallback to `coded_extent`.
- Verify: NO crash; all 20 frames decode. Frame 2 (map_idx=4=GOLDEN) correctly copies GOLDEN slot (= frame 0 buffer; key frame 0 refresh_flags=0xFF maps all slots→0).
- NEW BLOCKER (separate, affects ALL frames incl. frame 0 KEY): every frame reads back as ALL-ZERO (vk Y/U/V mean=0.0; PSNR ~7.6 dB = compare-vs-zeros). So the HW decode is landing no pixels in the DPB image.
  - `find_av1_frame_header_offset` (decoder.rs:2524) looks correct (walks OBUs, returns frame-header start; frame 0 → offset 19).
  - Top hypothesis: `StdVideoDecodeAV1PictureInfo` incompletely populated — tile/quantization/loop_filter/CDEF/segmentation/global_motion left zeroed (C++ ref VulkanAV1Decoder.cpp parses+fills all of them). HW likely decodes nothing → zero image.
  - Alt: readback layout mismatch (decode may leave image in VIDEO_DECODE_DST_KHR while readback barrier assumes VIDEO_DECODE_DPB_KHR).
- Next: confirm whether decode lands pixels at all (check VkVideoDecodeInfo / dump image with a correct-layout copy); then fully populate StdVideoDecodeAV1PictureInfo from parsed bitstream.

## Iteration 2 (full uncompressed_header parse)
- Task: rewrite `parse_frame_header` (vk-video-parser/src/av1.rs) to follow the EXACT AV1 CBS `uncompressed_header` bit order and populate ALL `Av1FrameHeader` fields. Struct + bitreader.rs left unchanged.
- Result: parse_frame_header rewritten + 11 helpers added (set_frame_refs, get_relative_dist1, is_skip_mode_allowed, parse_tile_info, parse_quantization, parse_segmentation, parse_delta_q_lf, parse_loop_filter, parse_cdef, parse_loop_restoration, parse_global_motion). Builds clean (warnings only). Frame headers parse with sane bitpos (KEY=107, INTER=118-367).
- Key corrections vs. prior understanding:
  - `allow_screen_content_tools` is read UNCONDITIONALLY when seq_force_screen_content_tools==SELECT (NOT gated on error_resilient_mode). CBS 1435-1441 + C++ 1883-1887 agree.
  - error_resilient_mode is INFERRED (not a bit) when frame_type==SWITCH || (KEY && show_frame).
  - short-signaling last/golden idx are u3 each.
  - quantization order: base_q(8) → delta_q_y_dc → diff_uv → delta_q_u_dc/ac → (if diff_uv) delta_q_v_dc/ac → using_qmatrix → qm(u4 each; qm_v shares qm_u unless separate_uv_delta_q).
  - C++ ReadSignedBits(6) actually reads 7 bits (u(7) sign-extended) → delta_q / loop_filter deltas use read_signed_bits(7).
  - loop_filter_level[0]/[1]=u(6); [2]/[3](uv) only if !mono && (lf0||lf1); sharpness=u(3).
  - segmentation feature bits {8,6,6,6,6,3,0,0}, signed {1,1,1,1,1,0,0,0}; tx_mode=u(1)+1.
  - loop_restoration lr_type=3×u(2), remap [0,3,1,2].
  - interpolation_filter is a DIRECT cast (SWITCHABLE=4), no remap.
- True SPS (from actual Rust SPS parse): profile=0, order_hint=true, ohb=6, warped=true, refmvs=true, sct=SELECT, imv=SELECT, **cdef=true, restoration=false**, superres=false, fid=false, mono=false, fwb=11, fhb=11, maxw=1919, maxh=1079.
- Verify: PSNR STILL 7.57-7.60 dB (all-zero frames). EXPECTED — the parse was a prerequisite, not the pixel blocker.
- Root cause of zero pixels (unchanged from Iteration 1): decoder.rs (lines ~1074-1121) only fills frame_type/order_hint/primary_ref/refresh_frame_flags/tile_cols_log2/tile_rows_log2 into StdVideoDecodeAV1PictureInfo. It does NOT populate quantization/loop_filter/cdef/segmentation/global_motion/flags, so HW decodes nothing → zero DPB image.
- Next: populate StdVideoDecodeAV1PictureInfo sub-structs (quantization, loop_filter, cdef, segmentation, global_motion, misc flags) from the now-correctly-parsed Av1FrameHeader in decoder.rs. That is the next PSNR blocker.

## Iteration 3 (populate StdVideoDecodeAV1PictureInfo)
- Task: fully populate `StdVideoDecodeAV1PictureInfo` + all 8 sub-structs in `decode_all_av1` (decoder.rs ~1074-1109) from the parsed `Av1FrameHeader`. Parser + container struct left unchanged.
- Mapped (all from `fh`):
  - 29 picture flags via ash setters (exact casing: `set_UsesLr` cap-U, `set_usesChromaLr` low-u). `buffer_removal_time_present_flag`=0 (SPS false); `UsesLr`=fh.uses_lr; `usesChromaLr`=(lr_type[1]||lr_type[2]!=0).
  - interpolation_filter / TxMode = DIRECT cast (parser already stores Vulkan enum values, SWITCHABLE=4). delta_q_res, delta_lf_res. SkipModeFrame=[0,0] (GAP).
  - coded_denom = fh.coded_denom (FIXED: was hard-coded `=1`; now 0 for our no-superres stream).
  - OrderHints[i] = order hint of the frame buffer that reference name i references (via ref_frame_idx mapping LAST/LAST2/LAST3/GOLDEN/BWDREF/ALTREF → idx 0/1/2/3/4/6). expectedFrameId[i]=0.
  - Sub-structs: quantization (base_q, 5 delta_q, qm_y/u/v; flags using_qmatrix, diff_uv_delta=0 GAP), loop_filter (level[4]=y0,y1,uv0,uv1, sharpness, ref/mode deltas, flags delta_enabled/update; update_ref_delta=fh.loop_filter_delta_update; update_mode_delta=0 GAP), segmentation (FeatureEnabled/FeatureData), tile_info (uniform_spacing=1 GAP, TileCols/Rows=1<<log2, null ptrs), cdef (damping, bits, 4 strength arrays), loop_restoration (FrameRestorationType direct cast — parser ALREADY remaps [0,3,1,2]; size), global_motion (idx0=identity {0,0,65536,0,0,65536}, idx1..7=fh.global_motion_type/params[0..6]). pFilmGrain=null. init_pointers() called last.
- GAPS (parser lacks field → safe default 0): diff_uv_delta, update_mode_delta, skip_mode_frame, uniform_tile_spacing (per-tile sizes).
- Compiles: YES (warnings only). Runs: YES (no crash).
- **PSNR: 7.59 dB (UNCHANGED, still all-zero frames).** Picture-info population was necessary but NOT the pixel blocker.
- **NEW ROOT CAUSE of all-zero output (affects every frame incl. KEY frame 0):** the AV1 SPS is NEVER passed to the HW session parameters. `session.rs:488` sets `av1_params.p_std_sequence_header = std::ptr::null()`. The C++ reference DOES provide it: `VkParserVideoPictureParameters.cpp:161` `av1SessionParametersCreateInfo.pStdSequenceHeader = GetStdAV1Sps()`. Without the SPS the decoder is never initialized → decodes nothing → zero DPB image. The SPS is already parsed and held in `self.av1_sps` (decoder.rs:110/331) but unused.
  - Verified NOT the cause: session profile/extent/format correct (profile 0, 1920x1080, 8-bit 420); frame_header_offset correct (frame 0 → 19); DPB reference_name_slot_indices correct (frame 1 refs all → slot 0 = key frame).
- Next (single most important fix): build a `StdVideoAV1SequenceHeader` from the parsed `Av1Sps` and pass it via `VideoDecodeAV1SessionParametersAddInfoKHR::p_std_sequence_header` (std_sps_count=1) in session.rs. This should make the HW decoder actually produce pixels.

## Iteration 4 (pass AV1 SPS to session)
- Task: build `StdVideoAV1SequenceHeader` from parsed `Av1Sps` and pass it via the AV1 session-parameters create info (was `p_std_sequence_header = null`).
- Changes:
  - av1.rs: added `convert_av1_color_config`, `convert_av1_timing_info`, `convert_av1_sps` (build StdVideoAV1SequenceHeader/ColorConfig/TimingInfo from `&Av1Sps`).
  - codec_types.rs: re-exported `StdVideoAV1ColorConfig`, `StdVideoAV1TimingInfo`.
  - session.rs `create()`: new `sps_av1` param; AV1 arm sets `av1_params.p_std_sequence_header`. SPS data now `Box::into_raw`-leaked (driver retains the pointer; must outlive create()).
  - decoder.rs: `parse_av1_init` returns `Option<Av1Sps>`; `new()` threads it; `create_video_session` takes `av1_sps` and AV1 no longer excluded from session-params creation; `update_session` now non-fatal when `vkUpdateVideoSessionKHR` is not loadable.
  - vp9.rs: added the extra `None` arg to its `create()` call.
- Compiles: YES (warnings only). Runs: YES (no crash; previously ABORTED at update_session because `vkUpdateVideoSessionKHR` not loadable → now falls through to maintenance1 auto-init).
- SPS confirmed correct on the wire: profile=0, fw_bits-1=10 (1920), fh_bits-1=10 (1080), max_w-1=1919, max_h-1=1079, order_hint-1=6, force_int_mv=1, force_sct=1, color_cfg/timing=null (allowed).
- Session params ARE created (vkCreateVideoSessionParametersKHR=SUCCESS) and passed non-null to vkCmdBeginVideoCodingKHR.
- **PSNR: 7.59 dB (UNCHANGED, still 100% all-zero frames — Y/U/V min=max=0, nonzero=0).** Passing the SPS was necessary but NOT the pixel blocker.
- Lifetime test: leaking the SPS (Box::into_raw) did NOT change output → driver copies the SPS data; lifetime was not the cause.
- `vkUpdateVideoSessionKHR` is NOT loadable via vkGetDeviceProcAddr (tried both "vkUpdateVideoSessionKHR" and "vkUpdateVideoSession" names) on this Mesa/Intel API-1.2 device, even though VK_KHR_video_queue + VK_KHR_video_maintenance1 are enabled. ash 0.38 does not expose it in any generated table. So explicit init is impossible; must rely on maintenance1 auto-init (which the C++ reference also does).
- **NEW CRITICAL FINDING — OBU type table is WRONG (decoder.rs:2840-2850, 2724, 2779, 2858, 2882, 2918).** Code + ITERATIONS.md line 16 use: 1=SEQ_HDR, 2=TEMP_DELIM, 3=FRAME_HDR, 4=TILE_GROUP, 6=FRAME, 8=TILE_LIST. The AV1 CBS (ISO 23090-3 §5.4.1) actually defines: 1=TemporalDelimiter, 2=SequenceHeader, 3=FrameHeader, **4=FrameOBU**, 5=TileList, **6=Metadata**, 7=Reserved. So the extractor searches `obu_type==6` for the frame but the real FrameOBU is type 4; frame 0's first OBU (byte 0x12 → type 2) is a SequenceHeaderOBU, mislabeled TEMP_DELIM. The "frame" it latches onto (type 6, 599476 bytes) is being treated as the decode payload — almost certainly the wrong bytes → HW decodes nothing → zero image.
- Next (single most important fix): correct the AV1 OBU type constants to the CBS values (FrameOBU=4, Metadata=6, SequenceHeader=2, TemporalDelimiter=1, TileList=5) in `extract_av1_frame_obu_payload` / `find_av1_frame_header_offset` / the name table, and re-verify that the FrameOBU payload + frame_header_offset actually point at the real frame header. Then re-check PSNR.

## Iteration 5 (set AV1 tile_count/tile_offsets/tile_sizes)
- Task: set `tile_count=1` and provide `p_tile_offsets`/`p_tile_sizes` for the AV1 decode (C++ ref VulkanAV1Decoder.cpp:2261-2304 ALWAYS sets tileCount=1 + tileOffsets[0]/tileSizes[0], even for single-tile frames; without them the NVIDIA driver doesn't know where the tile data is).
- Changes:
  - parser av1.rs: added `pub frame_header_size: u32` to `Av1FrameHeader` (uncompressed header size in BYTES, rounded up). Set in `parse_frame_header`: normal return = `((r.position()+7)/8) as u32`; show_existing_frame + reduced_still_picture early-returns = 0 (not decoded).
  - vk-video-vulkan av1.rs: added `tile_offsets: [u32;1]` + `tile_sizes: [u32;1]` to `Av1PictureInfoContainer` (+ Default init).
  - decoder.rs `decode_all_av1`: `tile_offset = find_av1_frame_header_offset(&av1_frame.data) + fh.frame_header_size`; `tile_size = frame_obu_payload.len() - fh.frame_header_size`; stored in container; `VideoDecodeAV1PictureInfoKHR::new` now passes `tile_count=1` + `container.tile_offsets.as_ptr()`/`tile_sizes.as_ptr()` (was `tile_count=1<<(cols+rows)` + null ptrs).
- Frame 0 tile values (diagnostic): frame_header_offset=19, frame_header_size=14, **tile_offset=33, tile_size=599462**, frame_obu_payload.len()=599476.
- Verified tile values match the C++ reference EXACTLY: `tileOffsets[0]=nalu.start_offset+consumedBytes`, `tileSizes[0]=payload_size-consumedBytes`, where `nalu.start_offset` = Frame OBU payload offset in the bitstream buffer (C++ `header_size` includes the EB128 size field → payload offset). Our bitstream buffer = full OBU stream at offset 0, so 19+14=33 and 599476-14=599462 are correct.
- Compiles: YES (warnings only). Runs: YES.
- **PSNR: 7.59 dB (UNCHANGED, still 100% all-zero frames).** Tile offsets were necessary but NOT the pixel blocker.
- **CORRECTED a FALSE Iteration-4 finding:** the OBU type table is NOT wrong. Verified against ffmpeg `av1.h` (SEQUENCE_HEADER=1, TEMPORAL_DELIMITER=2, FRAME_HEADER=3, TILE_GROUP=4, METADATA=5, FRAME=6, REDUNDANT=7, TILE_LIST=8, PADDING=15) and the C++ reference (identical). The Rust table (1=SEQ_HDR, 2=TEMP_DELIM, 3=FRAME_HDR, 4=TILE_GROUP, 6=FRAME, 8=TILE_LIST) is correct. Frame 0 OBUs: type2/size0=TemporalDelimiter, type1/size11=SequenceHeader (valid SPS), type6/size599476=Frame. Do NOT "fix" the OBU table.
- **NEW CRITICAL FINDING — the decode is NOT executing on the GPU.** Measured GPU utilization (nvidia-smi) during a full 300-frame decode:
  - Our Rust decoder: **0-7%** sustained.
  - C++ reference (`vk-video-dec-test --noPresent`, same IVF): **26-28%**, 300 frames @ ~250 FPS.
  - Our 300-frame decode takes **11.2s** (10x slower than C++ ~1.2s) — the time is CPU per-frame readback overhead (buffer+mem alloc/map/copy/unmap/free per frame), NOT GPU decode.
  - The fence returns SUCCESS and the command buffer IS submitted to the video decode queue, but the GPU never executes the decode. The DPB image is never written → readback reads all zeros.
- Verified NOT the cause this iteration: SPS correct (ffmpeg CBS `cbs_av1_syntax_template.c` — 4-bit frame_width_bits_minus_1 + early timing_info IS the real spec), frame header correct, tile offsets now correct, OBU table correct, queue family = video decode (correct), command pool = video decode queue (correct), RESET control correct, readback correct (reads the same DPB image the decode targets).
- Next (single most important): figure out why the video decode operation is not executing on the GPU (fence=SUCCESS but 0% GPU) — the driver is silently skipping/rejecting the decode. Enable the Vulkan validation layer (VK_LAYER_KHRONOS_validation is NOT installed system-wide; only Intel nullhw/Mesa/NVIDIA layers present) to surface the VUID violation, OR diff our session-params / picture-info / DPB-slot setup against the working C++ reference (`VkVideoDecoder.cpp`) to find what makes the driver skip the decode.

## Iteration 6 (align DPB image extent to pictureAccessGranularity)
- Confirmed actual NVIDIA device caps via C++ demo (`vk-video-dec-test --deviceID 2504 --verbose`): **pictureAccessGranularity = 16x16**, maxDpbSlots=16, maxActiveReferencePictures=16, dpbAndOutput=coincide. 1080 is NOT a multiple of 16 (→1088).
- C++ reference (VkVideoDecoder.cpp:259-262) aligns the **image extent** to granularity (1920x1088) but keeps the per-frame **decode-command codedExtent RAW** (m_codedExtent=1920x1080, lines 885/932/1031). Session maxCodedExtent = aligned (line 289).
- Change: decoder.rs DPB image creation now uses `session_coded_extent` (aligned 1920x1088) instead of raw `coded_extent` (1920x1080). Decode-command codedExtent stays raw.
- Compiles: YES. Runs: YES (no crash).
- **PSNR: 7.59 dB (UNCHANGED, still 100% all-zero).** GPU util nudged 3-4%→6.5% but decode still not executing (C++ = 26-28%). Image alignment was necessary-but-not-sufficient.
- Next: surgical comparison of SESSION PARAMETERS creation (SPS passing) + decode-command pNext chains vs C++ — hypothesis: session not actually initialized (SPS/session-params incompatibility) so driver skips decode.

## Iteration 7 (session init: videoMaintenance1 feature + SPS path verified)
- Task: verify EXACTLY how the C++ reference initializes the HW session with the AV1 SPS; check whether it calls vkUpdateVideoSessionKHR; make the Rust session init match C++.
- **C++ findings (verified by reading the code):**
  - The SPS is passed to the session ONLY at session-parameters CREATION: `VkParserVideoPictureParameters.cpp:161` `av1SessionParametersCreateInfo.pStdSequenceHeader = GetStdAV1Sps()` (pNext of `VkVideoSessionParametersCreateInfoKHR`, line 155). `maxStdSequenceHeaderCount` is left 0 (zero-init) with a NON-NULL `pStdSequenceHeader` (a VUID-09262 violation the NVIDIA driver tolerates).
  - The session is "initialized" by passing that session-params handle to `vkCmdBeginVideoCodingKHR` (`VkVideoDecoder.cpp:1103` `decodeBeginInfo.videoSessionParameters = *pOwnerPictureParameters;` → `CmdBeginVideoCodingKHR` line 1184). The object is created LAZILY on first decode via `FlushPictureParametersQueue` (line 1097) → `HandleNewPictureParametersSet` (VkParserVideoPictureParameters.cpp:393-418) → `CreateParametersObject`.
  - **C++ does NOT call `vkUpdateVideoSessionParametersKHR` for AV1**: `UpdateParametersObject` hits `assert(false && "There should be no calls to UpdateParametersObject for AV1"); return VK_SUCCESS;` (VkParserVideoPictureParameters.cpp:254-258). The only callers of the update fn are H264/H265.
  - **`vkUpdateVideoSessionKHR` DOES NOT EXIST** in the Vulkan spec (checked vulkan_core.h — only `vkUpdateVideoSessionParametersKHR` exists; ash exposes `update_video_session_parameters_khr`). The Rust `update_session` (session.rs:619-663) loads a non-existent name, so it always no-ops. Since C++ never calls the update fn for AV1 either, this no-op is behaviorally equivalent — NOT the bug.
  - **CONCRETE DIFFERENCE FOUND — device feature:** C++ enables `videoMaintenance1`. `VulkanDeviceContext.cpp:816-819` inits the feature to VK_FALSE, but line 838 `GetPhysicalDeviceFeatures2` OVERWRITES the whole chain with supported values, so `videoMaintenance1` becomes VK_TRUE and the device is created with it enabled (line 852 `devInfo.pNext = &deviceFeatures`; required by CHECK at line 841). Rust (device.rs) enabled `videoMaintenance2` + video-decode + sync2 + samplerYcbcr but OMITTED `videoMaintenance1`, while still using `VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR` (session.rs:287) — a VUID-09236 violation C++ does not have.
- **Change (Rust):** device.rs — added local `PhysicalDeviceVideoMaintenance1FeaturesKHR` struct + `PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR` (1000515000) and inserted it into the device feature pNext chain with `video_maintenance1 = 1` (chain: features2 → maintenance2 → maintenance1 → samplerYcbcr → videoDecode → sync2).
- Compiles: YES (warnings only). Runs: YES.
- **PSNR: 7.59 dB (UNCHANGED, still 100% all-zero; GPU util 0-2%).** Enabling videoMaintenance1 was correct (matches C++, removes the VUID) but NOT the pixel blocker. The SPS is confirmed on the wire and the session IS initialized via BeginVideoCoding — the "session never gets the SPS" hypothesis is effectively DISPROVEN.
- Verified matching C++ this iteration (NOT the cause): SPS content, `pStdHeaderVersion` spec_version=1<<22 (== C++ VK_MAKE_VIDEO_STD_VERSION(1,0,0)), session-params create-info pNext chain, BeginVideoCoding slots (setup+refs), DecodeVideo fields, barriers, DPB image usage/tiling/queue-family, spec version, maxDpbSlots=10/maxActiveRef=9.
- Next (single most important): the decode is still not executing on the GPU (fence=SUCCESS, GPU 0-2% vs C++ 26-28%) → the NVIDIA driver is silently rejecting the decode. The session-init hypothesis is exhausted. Next: (a) set `maxStdSequenceHeaderCount=1` in session.rs (removes the VUID-09262 so the driver is guaranteed to read the SPS — cheap test), and/or (b) get a Vulkan validation layer running (install VK_LAYER_KHRONOS_validation, or run under the NVIDIA driver's validation) to surface the exact VUID, and/or (c) byte-dump the `VkVideoDecodeAV1PictureInfoKHR`+`StdVideoDecodeAV1PictureInfo` Rust sends vs what C++ sends to rule out a subtle field/pointer mismatch.

## Iteration 8 (ROOT CAUSE: wrong s_type on VkVideoDecodeAV1PictureInfoKHR)
- Task: PROBE A (max_std_sequence_header_count=1) + PROBE B (validation layer / byte-dump).
- **PROBE A DISPROVEN**: `max_std_sequence_header_count` does NOT exist in `VkVideoDecodeAV1SessionParametersCreateInfoKHR` in ANY header (ash 1.3.281, system 1.4.341, C++ ref 358 all agree — struct is only sType/pNext/pStdSequenceHeader). VUID-09262 is actually about the decode command's `referenceNameSlotIndices`, NOT the SPS. The "maxStdSequenceHeaderCount=0 VUID-09262" notes in Iterations 3/7 were a misread. NO change made (session.rs edit reverted).
- **PROBE B SUCCESS**: Got the Khronos validation layer running WITHOUT sudo: `apt-get download vulkan-validationlayers` (1.4.341) → `dpkg-deb -x` to /tmp/vkval → run with `VK_LAYER_PATH=/tmp/vkval/usr/share/vulkan/explicit_layer.d LD_LIBRARY_PATH=/tmp/vkval/usr/lib/x86_64-linux-gnu VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.
- **ROOT CAUSE FOUND + FIXED**: local `av1_vk_constants::VIDEO_DECODE_AV1_PICTURE_INFO_KHR` was **1000303002** — a FABRICATED value that does not exist in the Vulkan spec. Correct value is **1000512001**. With the wrong sType, the driver did NOT recognize `VkVideoDecodeAV1PictureInfoKHR` in the pNext chain of `VkVideoDecodeInfoKHR` → silently rejected EVERY decode → all-zero output, GPU 0-2%.
  - Validation layer confirmed it: `VUID-VkVideoDecodeInfoKHR-pNext-pNext` ("pNext ... must be a valid instance of ... VkVideoDecodeAV1PictureInfoKHR") + `VUID-vkCmdDecodeVideoKHR-pNext-09250` ("pNext chain of pDecodeInfo must include a VkVideoDecodeAV1PictureInfoKHR").
  - Fix: av1.rs:20-26 — corrected all 4 AV1 decode s_type constants to spec values (CAPABILITIES=1000512000, PICTURE_INFO=1000512001, PROFILE_INFO=1000512003, SESSION_PARAMS_CREATE=1000512004). Only PICTURE_INFO was actually used (av1.rs:386). ash 0.38 HAS the correct struct + StructureType variant; the local re-impl just had a bad constant.
- **PSNR: 7.59 dB → 24.59 dB.** Frame 0 (KEY) = PERFECT MATCH. Inter frames 37→21 dB, degrading over time = reference-error accumulation. The decode now ACTUALLY EXECUTES on the GPU.
- **Remaining VUIDs (cause of the residual 24.59→pixel-perfect gap):**
  1. `VUID-VkVideoBeginCodingInfoKHR-flags-07244` (TOP suspect): device does NOT support `VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR` (dpbAndOutput=coincide), so all reference imageViews MUST come from the SAME image. Rust creates a SEPARATE image per DPB slot. C++ ref handles it: `if(!(flags & SEPARATE_REFERENCE_IMAGES)) m_useImageArray = VK_TRUE` (VkVideoDecoder.cpp:349) → single image with array layers.
  2. `VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239`: slotIndex must be an "active" DPB slot — DPB state management.
  3. `VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07139`: srcBufferRange must be a multiple of minBitstreamBufferSizeAlignment (=256). C++ also uses raw bitstreamDataLen (has a TODO to assert), so the driver tolerates it — LOW priority.
- Next (iteration 9, single most important): switch the DPB to a SINGLE image with array layers (m_useImageArray) when SEPARATE_REFERENCE_IMAGES is unsupported — top suspect for the reference-error accumulation. Then re-run under the validation layer to confirm 07244/07239 clear.

## Iteration 9 (DPB single image w/ array layers — m_useImageArray)
- Task: when `VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR` is unsupported (our NVIDIA RTX 3060: dpbAndOutput=coincide), the spec (VUID-07244) requires ALL reference imageViews to come from the SAME image. C++ handles this with `m_useImageArray`; Rust made a SEPARATE VkImage per DPB slot. Switch Rust to a single image with array layers.
- **C++ reference (verified):**
  - `VkVideoDecoder.cpp:349-353`: `if(!(flags & SEPARATE_REFERENCE_IMAGES)) m_useImageArray = VK_TRUE`.
  - `VkVideoDecoder.cpp:544`: DPB image `arrayLayers = m_useImageArray ? numDecodeSurfaces : 1`.
  - `VulkanVideoFrameBuffer.cpp:845-846`: per-slot imageView = `VkImageSubresourceRange{COLOR, mip0, 1, baseArrayLayer=slot, layerCount=1}`.
  - `VulkanVideoFrameBuffer.cpp:230-234`: `VkVideoPictureResourceInfoKHR.baseArrayLayer = 0` (the picture-resource baseArrayLayer stays 0; slot selection is via the per-slot VIEW).
  - `VkVideoDecoder.cpp:840`: per-slot barrier `baseArrayLayer=currPicIdx, layerCount=1`.
  - **OUTPUT/READBACK (key diff):** `VkVideoDecoder.cpp:715-775` `CopyOptimalToLinearImage` does an IMAGE-TO-IMAGE copy from the decoded DPB slot → a SEPARATE output/filter image (`m_useTransferOperation`, called at :1246). CPU readback is from that separate image — the DPB slot is only a copy SOURCE and is never transitioned to TRANSFER_SRC_OPTIMAL for a buffer copy.
- **Rust changes:**
  - `profile_chain.rs:100` new `create_dpb_image_array_with_profile` (single image w/ `num_slots` layers + one per-slot view `base_array_layer=slot, layer_count=1`); `:179` `create_image_with_profile_chain` gained `array_layers: u32` param (was hardcoded 1).
  - `decoder.rs:308-365` branch DPB creation on `caps.flags.contains(SEPARATE_REFERENCE_IMAGES)` (false → array path: one image handle pushed per slot, one view/slot, one memory); `:90` new field `dpb_use_image_array`; Drop dedups `dpb_images` by handle (array mode shares one image); 5 `readback_decoded_image` call sites now pass `base_array_layer` (slot as u32).
  - `av1.rs:750-772` `record_decode_command` gained `dpb_use_image_array: bool`; `:981-1061` output+reference barriers use `base_array_layer = slot index` in array mode (else 0); ref loop zips `dpb_ref_slot_indices`.
  - `readback.rs:23` `readback_decoded_image` gained `base_array_layer: u32`; used in barriers + copies.
- Build: OK. Confirmed array path IS taken (`separate_reference_images=false`, "DPB image array created (10 layers)").
- **PSNR: 24.59 dB → 24.59 dB (NO-OP, byte-identical).** Frame 0 (KEY) = PERFECT MATCH (0 diff on Y/U/V). Inter frames degrade 41.85→25.26 dB over 20 frames (same pattern as Iter 8). **The DPB image layout was NOT the cause of the inter-frame error.** (Note: an earlier "frame 0 = 23.87 dB" reading this session was a REFERENCE-GEN BUG — I had generated the ffmpeg ref as NV12 while our output is planar YUV420P; regenerating the ref as planar yuv420p restores frame 0 = perfect.)
- **Validation layer:** `VUID-...-07244` CLEARED (the fix worked). `VUID-...-07239` (slot not active) + `VUID-...-07139` (srcBufferRange alignment) REMAIN — both also present in C++ (driver tolerates); LOW priority.
- **Error signature (new, informs next step):** Y plane dominates (frame1 Y mse 12 vs U 0.5 / V 0.3); error is WIDESPREAD (65%→96% of px nonzero); BORDER-WEIGHTED (border8px ~2× interior, bottom edge highest); ACCUMULATES (Y mse 12→496 over 19 frames). Small per-frame error compounding through references.
- **NEXT (iteration 10, top hypothesis):** the Rust `readback_decoded_image` transitions the DPB slot itself `VIDEO_DECODE_DPB_KHR → TRANSFER_SRC_OPTIMAL → VIDEO_DECODE_DPB_KHR` and copies image→buffer, and this readback happens BETWEEN a slot's decode-write and its use as a reference for the next frame. The C++ instead copies the DPB slot → a SEPARATE output image and reads back from that (DPB slot never goes to TRANSFER_SRC_OPTIMAL). TEST: read back via a staging image (DPB slot never transitioned) and check whether the inter-frame error disappears. If yes → readback was corrupting the DPB; if no → look at in-loop filtering / motion-comp / reference-data divergence.

## Iteration 9 follow-up (readback hypothesis REFUTED)
- Tested the Iteration-9 "next hypothesis" (does the direct readback's `VIDEO_DECODE_DPB_KHR -> TRANSFER_SRC_OPTIMAL -> back` transition corrupt the DPB reference data?).
- **Method:** added `readback_decoded_image_via_staging` (readback.rs) — copies the DPB slot (in `VIDEO_DECODE_DPB_KHR`, NEVER transitioned) → a separate staging image → buffer. Toggled via `VACC_STAGING_READBACK=1` in the AV1 decode path (decoder.rs).
- **Result: STAGING READBACK = BYTE-IDENTICAL to DIRECT READBACK** (all 20 frames exactly the same PSNR: f0 perfect, f1 41.85, …, f19 25.26). **The readback does NOT corrupt the DPB. Hypothesis REFUTED.**
- **Conclusion:** the inter-frame error is in the DECODE itself (motion compensation and/or in-loop filtering), NOT in the readback, NOT in the DPB image layout, NOT in the DPB slot transitions.
- Frame 0 (KEY) = perfect → intra/transform/quant are correct. Inter frames have a small per-frame error (Y mse 12 @ f1) that ACCUMULATES (→496 @ f19). Y-dominant, widespread (65%→96% px), border-weighted (bottom edge highest).
- **OPEN QUESTION (next):** is this error INHERENT to the NVIDIA HW decoder (a correct HW decoder can still differ slightly from dav1d in sub-pixel interpolation / in-loop filtering), or a BUG in our setup (reference slot mapping, order hints, coded extent, session params)? 
  - To answer: build the C++ reference (`Vulkan-Video-Samples/vk_video_decoder`, needs CMake build — no prebuilt binary present) and compare its YUV output vs dav1d on the same IVF. If C++ also ~24-28 dB → inherent (goal may be "as close as possible"). If C++ is pixel-perfect → our setup bug; then diff reference setup / order hints / coded extent vs C++.
  - Cheaper probes first: (a) verify `dpb_ref_order_hints` + `reference_name_slot_indices` match the bitstream's ref_frame_idx→frame-buffer→slot mapping; (b) confirm codedExtent passed = bitstream frame_width/height (1920x1080) not the aligned image size (1088); (c) check film_grain disabled + loop-restoration params.

## Iteration 9 follow-up 2 (picture-info GAP fields — top new lead)
- The driver USES the frame-level picture-info params we pass (the C++ fills them from the bitstream). Found GAPs where Rust sets 0 / discards but C++ provides the real value:
  - `tile_size_bytes_minus_1`: parser av1.rs:1583 parses it but DISCARDS it (`let _ =`, line 1585). C++ provides it (VulkanAV1Decoder.cpp:1288) and uses it to compute tile_size (line 2288). Rust decoder.rs:1303-1310 sets it to 0 and assumes uniform tile spacing. **TOP candidate** for the Y-dominant widespread error (affects tile decode).
  - `diff_uv_delta`: decoder.rs:1345 sets 0 (GAP). C++ parses it (VulkanAV1Decoder.cpp:1316). Parser av1.rs:1610 reads it but only uses it locally (not stored in `fh`). Bug if `sps.separate_uv_delta_q` (av1.rs:902) is true and the value is non-zero. Affects UV quant (error is Y-dominant, so maybe not the main cause, but a real bug).
  - `update_mode_delta`: decoder.rs:1370 sets 0. BUT the C++ ALSO reads ONE update flag for both ref+mode deltas (VulkanAV1Decoder.cpp:1454-1466) — matches our parser (av1.rs:1736-1747). So NOT a bitstream desync; likely fine.
- Loop-filter parsing MATCHES the C++ (one update flag for both ref+mode deltas) — ruled out.
 - NEXT: (1) store `diff_uv_delta` + `tile_size_bytes_minus_1` in the parser and pass them to the driver; (2) verify they are non-zero for this IVF (debug print); (3) re-test PSNR. Separately, still need the inherent-vs-bug determination (build the C++ reference and compare its YUV vs dav1d).

## Iteration 12 (readback hypothesis REFUTED at frame-1 granularity; RefFrameSignBias fix = no-op)
- Focus: does the frame-0 READBACK corrupt the reference (slot 0) used by frame 1?
- **EXPERIMENT A (VACC_SKIP_READBACK=1):** skip frame-0 readback. Frame-1 PSNR = **21.09 dB WITH and WITHOUT** frame-0 readback (byte-identical). → Readback is NOT the cause (re-confirms Iter-9 follow-up refutation, now at frame-1 granularity).
- **EXPERIMENT B (VACC_DUMP_REF_SLOT=1):** dump reference slot 0 IMMEDIATELY before frame-1 decode. = **maxdiff=0** vs known-good frame 0 (ffmpeg). → Reference slot is INTACT before frame 1 decodes.
- **Frame-header parse ruled out:** Rust `primary_ref_frame=u(3)` + `ref_frame_idx=u(3)`; C++ identical (VulkanAV1Decoder.cpp:1963, REF_FRAMES_BITS=3) and pixel-perfect. Spec-based Python re-parse of frame-1 header matches Rust exactly (primary_ref=7, ref_idx all-0, order_hint=20, refresh=2).
- **Picture info matches C++ field-by-field** (RUST-PI vs CPP-PI dumps): identical except `gm_params[0]` (RUST identity [0,0,65536,0,0,65536], CPP zeros) — but GmType[0]=0 (identity) so ignored. Frame-1 decode params: ref_slots=[0], referenceNameSlotIndices=[0,0,0,0,0,0,0], output_old_layout=UNDEFINED.
- **Change (Rust): RefFrameSignBias/SavedOrderHints now populated for reference slots** (was zero; C++ fills them, VulkanAV1Decoder.cpp:323-334).
  - av1.rs: new struct fields `frame_buffer_saved_order_hints`/`ref_sign_bias`/`frame_type`/`disable_cdf`/`seg_enabled` (init in new(), cleared in reset_dpb()); new methods `set_frame_buffer_ref_info`/`get_frame_buffer_ref_info`/`get_frame_buffer_for_dpb_slot`/`get_relative_dist`; `record_decode_command` now fills each ref slot's `StdVideoDecodeAV1ReferenceInfo`.
  - decoder.rs: capture cur_order_hints/frame_type/disable_cdf/seg_enabled/ohb before `av1_decode_info` move; call `set_frame_buffer_ref_info(...)` in the refresh loop.
- **PSNR: 21.09 dB → 21.09 dB (NO CHANGE, byte-identical).** Correct per C++ but NOT the bug.
- **KEY NEW CLUE:** frame 2 = `show_existing_frame` (map_idx=4=GOLDEN). Our frame 2 = **frame 0 content (maxdiff=0)**; ffmpeg frame 2 = a **THIRD distinct image** (maxdiff=121 vs f0, ≠ f1). Suggests a deeper DPB/refresh-state or frame-order discrepancy.
- **NEXT (iteration 13):** reference intact + picture info matches C++ → error is in the DECODE inputs for frame 1. Top candidates: (a) `frame_header_offset` / bitstream-buffer layout for the INTER frame (frame 0 KEY works; verify offset correct for frame 1); (b) tile offsets/sizes (tile_size_bytes_minus_1 still set 0 — see Iter-9 follow-up 2); (c) frame-2 show_existing=GOLDEN discrepancy → re-verify frame 1 refresh_frame_flags + DPB slot state.

## Iteration 15 (reference-setup fix: frame-buffer-0 refresh; layer probe → multi-ref INTER frames output the reference)
- **Task:** (1) verify frame 1's reference setup (should have ref_slots=[0]); (2) add `VACC_LAYER_PROBE=1` (frame 3: read layer 0 + layer 2, print mean Y); (3) minimal fix if obvious.
- **BUG FOUND + FIXED (reference setup):** the AV1 refresh loop (decoder.rs:1696) used `for i in 0..7 { let fb = i + 1; ... }` → refresh_frame_flags bit i → frame buffer (i+1). Frame buffer 0 (INTRA) was NEVER refreshed → `frame_buffer_to_dpb_slot[0]` stayed -1. Frame 1 (ref_frame_idx=[0,0,0,0,0,0,0]) → frame buffer 0 → -1 → **empty refs** (ref_slots=[]).
  - **Fix:** `for i in 0..8 { let fb = i; ... }` (bit i → frame buffer i, 0-indexed). Matches AV1 spec + C++ `UpdateFramePointers` (VulkanAV1Decoder.cpp:379-417). For a KEY frame (refresh=0xFF) all 8 frame buffers 0..7 → slot 0.
  - **Verified:** frame 0 DPB after = `0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0` (all → slot 0). Frame 1 `reference_name_slot_indices=[0,0,0,0,0,0,0]`, **ref_slots=[0]** ✓. Frame 3 refs correctly map to slots 0 & 1.
- **VACC_LAYER_PROBE (frame 3, output_slot=2):** `out(layer 2) meanY=98.577 == layer0 meanY=98.577 == layer2 meanY=98.577`. The decode writes to the CORRECT layer (2, NOT layer 0) → NOT an output-picture-resource layer bug.
- **YUV comparison (vs frame 0):** frame 1 MAD=14.83 (a real, but wrong, frame); **frames 3,4,6 (INTER) MAD=0.0000 (EXACTLY frame 0's content)**; frames 2,5,7 are show_existing_frame (correctly display their referenced slot, which holds frame 0's content due to the 3/4/6 bug).
- **PSNR (8 frames): f0 inf, f1 21.09, f2 32.41, f3 29.91, f4 27.20, f5 25.62, f6 24.82, f7 24.11 (avg 26.45).** The reference fix resolved the "frames 2-7 = frame 0" symptom for the show_existing frames, but INTER frames 3/4/6 still output frame 0's content (their primary reference).
- **KEY DIAGNOSIS:** INTER frames with **2+ unique references** (f3: refs [0,1]; f4: [2,0,1]; f6: [0,3,2,1]) decode to EXACTLY the first reference's content (frame 0, slot 0). Frame 1 (1 reference [0]) decodes to a different (but wrong, 21 dB) frame. The decode is NOT failing and writes to the correct output layer — it is producing the reference picture's content instead of the decoded frame.
- **Reference picture setup verified correct:** `build_av1_dpb_picture_resources` (decoder.rs:3106) builds ref_pictures from unique slots in first-appearance order; `referenceNameSlotIndices` values (DPB slots) match the reference slots' `slot_index` (VUID-09262 satisfied). `dpb_ref_order_hints` + `StdVideoDecodeAV1ReferenceInfo` (OrderHint/SavedOrderHints/RefFrameSignBias/frame_type/flags) populated per C++.
- **OPEN LEADS (next):** (a) bitstream/tile layout for multi-ref INTER frames — frame 1 access unit=64975 B but Frame OBU payload=50272 B (~14.7 KB unaccounted); verify tile_offset/tile_size + frame_header_offset for f3 (RUST-PI dump is frame-1-only); (b) whether the HW decoder is silently using only the primary/first reference and skipping residuals for multi-ref frames; (c) compare f3 decode-command params (ref slots, order hints, picture info) field-by-field vs C++ [CPP-PI].
- **NEXT (iteration 16):** extend RUST-PI/RUST-DEC dumps to frame 3 (multi-ref) and diff vs C++; verify the bitstream buffer + tile offsets/sizes for frame 3 are correct (the ~14.7 KB access-unit discrepancy); determine why multi-ref INTER frames output the reference content.

## Iteration 18 (per-frame src_buffer_range VERIFIED correct; multi-ref bug = DPB layout)
- **Task:** verify + fix per-frame `av1_frame.data.len()` + actual `src_buffer_range` + bitstream content.
- **Per-frame `av1_frame.data.len()` (= IVF packet size) AND actual `src_buffer_range` (decoder.rs:1621-1623, av1.rs:1264-1265):**
  - The IVF has a NON-STANDARD structure: some packets contain MULTIPLE Frame OBUs (type 6). Verified by parsing + [AV1-EXTRACT]:
    - pkt0=599495 (1 Frame OBU), **pkt1=64975 (5 Frame OBUs!)**, pkt2=5 (0, show_existing), pkt3=519 (1), pkt4=587 (1), pkt5=5 (0), **pkt6=1869 (2 Frame OBUs)**, pkt7=5 (0), pkt8=607 (1), pkt9=558 (1), pkt10=5 (0), **pkt11=6164 (3 Frame OBUs)**, pkt12=5 (0), pkt13=726 (1), pkt14=638 (1), pkt15=5 (0).
    - pkt1 Frame OBU payload_start offsets: 6, 50281, 60594, 63999, 64697.
  - `av1_frame.data` = full IVF packet (access_unit.rs:1549 `data: packet.clone()`). So each extracted Frame OBU's `data.len()` = its whole access unit.
  - **ACTUAL src_buffer_range per extracted frame (verified via [AV1-DIAG], av1.rs:863-876):** ext0=599495, ext1-5=64975 (all 5 Frame OBUs in pkt1 share the 64975 access unit), ext6=519, ext7=587, ext8-9=1869, ext10=607, ext11=558, ext12-14=6164.
  - **This MATCHES the C++ reference EXACTLY** ([CPP-DEC]: frame1-5=64975, frame6=519, frame7=587, frame8-9=1869). **The task's hypothesis (frame 3 range should be 519, was stale 64975) is DISPROVEN** — the earlier "[0..64975)" debug was the frame-1-only RUST-DEC dump (correct for pkt1's Frame OBUs), mislabeled as "frame 3".
  - **Display-frame mapping (NEW, was unknown):** display frames are the extracted frames with show_frame=1: disp0=ext0(KEY), disp1=ext5, disp2=ext6, disp3=ext7, disp4=ext9, disp5=ext10, disp6=ext11, disp7=ext14. (Extracted frames 1-4,8,12,13 are non-display; they decode to update DPB but are not output.)
- **tile_offset/tile_size VERIFIED correct:** decoder.rs:1452-1455 uses `av1_frame.payload_start + fh.frame_header_size` (per-Frame-OBU offset, NOT the first OBU). Matches C++ (ext1: tile_offset=41 tile_size=50237 == C++ tileOffsets[0]=41 tileSizes[0]=50237).
- **frame_header_offset = 0 (decoder.rs:1465 hardcoded).** C++ also leaves it 0 (not set in source; [CPP-DEC]'s "24754" is garbage — that debug reads the struct at a wrong offset, its refNameSlotIndices=[1,1,1000314006,...] is corrupt). frame0 decodes PERFECT with 0 → driver uses tile_offsets/tile_sizes, not frame_header_offset. NOT the bug.
- **Picture info MATCHES C++ field-by-field** for ALL frames (display + non-display): type/oh/primref/refresh/refidx/flags/interp/txmode/quant/lf/cdef/lr/gm/seg/refNameSlotIndices/orderHints/skipModeFrame/coded_denom all identical (only diff: gm_params[0] identity-vs-zeros, ignored since GmType[0]=0).
- **DPB slot assignment MATCHES C++** (ext0-8 → slots 0-8; C++ recycles slot 5 at frame9, Rust uses slot 9 — but frame-buffer→slot mapping is equivalent, refs match by slotIndex).
- **THE BUG (CONFIRMED via VACC_REF_PROBE, decoder.rs ~1744):** Before decoding ext5 (disp1, first display INTER), the reference slots contain: slot0=98.577, **slot2=98.577, slot3=98.577, slot4=98.577 (ALL = KEY frame's content!),** slot1=98.785 (distinct). So the non-display multi-ref frames ext2/ext3/ext4 (refs [0,1],[0,2,1],[0,3,2,1]) decoded to **slot 0's (KEY/first-ref) content**, not their own. This corrupts DPB slots 2/3/4, which then corrupt every subsequent display frame. **Single-ref ext1 (refs [0]) decodes to distinct content (98.785) → the bug is specific to 2+ unique references.**
- **Rust-vs-C++ difference (top suspect):** C++ uses SEPARATE DPB images (baseArrayLayer=0 for every picture, [CPP-DEC]); Rust uses a SINGLE image with array layers (baseArrayLayer=slot, dpb_use_image_array=true). Iteration 9's "array layout = no-op" test predates the iteration-15 reference fix (refs were empty then), so it is NOT valid evidence. Hypothesis: with array layers the NVIDIA driver mis-binds the per-reference array layers and falls back to the first/primary reference for multi-ref frames.
- **PSNR before→after:** NO functional change this iteration (debug probes only). f0 inf, f1 37.35, f2 32.41, f3 29.91, f4 27.20, f5 25.62, f6 24.82, f7 24.11 (avg 28.77). Rust-vs-C++ output MAD: f0=0.0000 (perfect), f1=1.88, f2=3.38, f3=4.65, f4=6.55, f5=7.98, f6=8.84, f7=9.68 (accumulating).
- **NEXT (iteration 19, single most important):** switch the DPB to SEPARATE images (one VkImage per slot, baseArrayLayer=0 for all pictures) to match the C++ exactly, even though it violates VUID-07244 (the driver tolerates it — C++ is pixel-perfect). Toggle `dpb_use_image_array` off / force the separate-image path in decoder.rs:308-365 + av1.rs:981-1061 (barriers/base_array_layer) + readback.rs. Re-run REF-PROBE (slots 2/3/4 should get distinct meanY) + pixel-verify. If that fixes it, the array-layer reference binding was the bug.

## Iteration 21 (fc=3/fc=4=keyframe: driver copies primary ref, ignores picture-info)
- **Task:** diagnose why fc=3 (ext3) and fc=4 (ext4) decode to EXACTLY the keyframe (MAD_y=0) while fc=2 (ext2) and fc=5 (ext5) decode to distinct content.
- **DISPROVEN hypotheses (verified this iteration):**
  - tileCount: C++ increments tileCount per tile group (VulkanAV1Decoder.cpp:2303) → single-tile frames get tileCount=1. Rust's tileCount=1 is CORRECT (matches C++ [CPP-PI]). Earlier "C++ passes tileCount=0" was a misread of the CORRUPTED [CPP-DEC] dump (its pNext points to inlineQueryInfo at dump time).
  - DPB reference content: REF-PROBE (extended to fc=3/4/5) shows references are CORRECT before ext3 decode: slot0=98.577(key), slot1=98.785, slot2=98.717 — all distinct. So the bug is NOT the DPB state.
  - bitstream buffer: = full IVF packet (64975 B). Frame OBUs at correct offsets (ext1@6, ext2@50281, ext3@60594, ext4@63999, ext5@64697). tile_offsets correct. C++ also writes full packet (ParseByteStream memcpy, bitstreamDataLen=64975).
  - frame_header_offset: both Rust and C++ use 0 (C++ never sets it). Tested VACC_FH_OFF=1 (frame_header_offset=payload_start) → NO change (ext3/4 still MAD=0).
  - validation layer: only 07139 + 07239 (both present in C++ too). No new VUID.
- **KEY FINDING (decisive test):** forced `base_q_index=0` in the picture info for ext3 only (VACC_FORCE_Q0_F3=1) → **NO effect** (ext3 still MAD=0 vs keyframe). The NVIDIA driver does NOT use the StdVideoDecodeAV1PictureInfo we pass for the actual decode; it parses the frame header from the bitstream itself (or copies the reference). So picture-info field bugs (CDEF zeros, etc.) do NOT explain the fc=3/4 symptom.
- **CORRELATION:** primref=4 (GOLDEN) → broken (ext3, ext4); primref=7 → works (ext2, ext5). Driver appears to COPY the PRIMARY REFERENCE instead of decoding for primref=4 frames: ext3 primref=4→GOLDEN=slot0(key)→outputs key; ext4 primref=4→GOLDEN=slot3(corrupted by ext3=key)→outputs key.
- **CDEF bug (SEPARATE, confirmed):** Rust CDEF strengths all zeros; C++ real values (frame3 ypri=[0,0,11,11] ysec=[2,0,0,2]). Affects all frames slightly. Parser parse_cdef looks correct (reads damping/bits/strengths); need to check why values end up zero (coded_lossless? enable_cdef? not passed?). Does NOT explain fc=3/4 (which are EXACTLY keyframe, not "slightly wrong").
- **Changes (debug only, env-gated):** decoder.rs — extended REF-PROBE to fc=3/4/5; added VACC_FORCE_Q0_F3 (force base_q=0 on ext3); added VACC_FH_OFF (frame_header_offset=payload_start). No functional change to default path.
- **PSNR:** unchanged (avg 29.16 dB; f0 inf, f1 37.35, f2 32.41, f3 29.91, f4 27.20, f5 25.62, f6 24.82, f7 26.84).
- **NEXT (iteration 22, single most important):** the driver parses the frame header FROM THE BITSTREAM (ignores our picture info). With 5 Frame OBUs in the buffer, the driver likely finds the WRONG frame header for ext3/ext4. Test: write ONLY the single Frame OBU (not the full packet) to the bitstream buffer per frame, with frame_header_offset=0 and tile_offset recomputed relative to the single OBU. This eliminates the "multiple Frame OBUs" ambiguity. If C++ works with the full packet, compare EXACTLY how C++'s driver locates the frame header (maybe C++'s driver uses tile_offset to find the OBU). Also fix the CDEF-zeros bug separately.

## Iteration 22 (ROOT CAUSE: ref_global_models never updated → wrong global motion params)
- **Task:** find the ONE remaining difference for multi-ref frames (single-ref vs multi-ref field comparison + ORDER comparison vs C++).
- **Single-ref (fc=1/ext1) vs multi-ref (fc=3/ext3) field comparison (RUST-DEC-F3 + RUST-REF + C++ [CPP-PI]/[CPP-DEC]):**
  - fc=1 (1 ref): ref_slots=[0], refNameIdx=[0,0,0,0,0,0,0], tile=[41..50278).
  - fc=3 (3 refs): ref_slots=[0,1,2], refNameIdx=[0,0,0,0,2,0,1], tile=[60625..63996).
  - **ALL decode-command fields MATCH C++ exactly for fc=3:** refNameSlotIndices=[0,0,0,0,2,0,1], ref_slots=[0,1,2] (C++ REF[0..2]=0,1,2), tile_offsets=60625/tile_sizes=3371, src_buffer_range=64975, SETUP slot=3, frame_header_offset=0, all_slots=[0,1,2,3] (refs+setup LAST). reference_slot_count (decode)=3, BeginVideoCoding count=4.
  - **ORDER (task #2/#3) already correct:** build_av1_dpb_picture_resources (decoder.rs:3321-3363) iterates FRAME BUFFER index 0..8 (matching C++ FillDpbAV1State VulkanVideoParser.cpp:1812-1839), NOT first-appearance. ref_std_infos order == ref_slots order (per-slot, travels with slot). INTRA slot included (C++ counts it via ref_frame_idx[0]). So ORDER is NOT the bug (was fixed in iter 19).
- **Picture-info field diff (RUST-PI-F3 vs C++ [CPP-PI] frame3) — the ONE real difference:**
  - type/oh/primref/refresh/flags/interp/txmode/quant/lf/skipModeFrame/orderHints/refNameSlotIndices/tile: ALL MATCH.
  - CDEF: valid values MATCH (cdef_bits=1→n=2: ypri=[0,0] ysec=[2,0] both). The "CDEF-zeros" from iter 21 was a MISDIAGNOSIS — the differing values (ypri[2..3]=11, ysec[3]=2) are in INVALID indices (leftover), not used.
  - **gm_params[1] DIFFERS: RUST [159744,133120,65688,12,-12,65688] vs C++ [148480,120832,65394,8,-8,65394].** (gm_params[0] identity-vs-zeros ignored since GmType[0]=0; gm_params[5] matches.)
- **ROOT CAUSE:** `update_ref_frames` (vk-video-parser/src/av1.rs:2064) saved ref_frame_sizes/ref_order_hints/ref_loop_filter for refreshed frame buffers but **NEVER saved the current frame's global motion models into `ref_global_models`**. So `ref_global_models` stayed at the identity default for every frame buffer. `parse_global_motion` (av1.rs:2000-2008) uses `ref_global_models[ref_frame_idx[primary_ref]]` as the "previous" model to de-delta the bitstream's global motion params → wrong previous model → wrong gm_params in the picture info → NVIDIA driver (which DOES use the picture-info gm_params, refuting iter-21's "driver ignores picture info") decoded multi-ref INTER frames against the wrong warped-motion model → output collapsed to the primary reference's (keyframe's) content. Single-ref frames (ext1) were less affected (fewer non-identity warped refs).
  - C++ reference DOES save them: VulkanAV1Decoder.cpp:399 `memcpy(&m_pBuffers[ref_index].global_models, &global_motions, ...)` in UpdateFramePointers, and reads them back in DecodeGlobalMotionParams (VulkanAV1GlobalMotionDec.cpp:249-250).
  - This also explains the iter-21 "primref=4 (GOLDEN) correlation": GOLDEN (fb4) was the primary ref whose (missing) global models were needed; with identity fallback the de-delta produced large wrong warped params.
- **Change (Rust):** vk-video-parser/src/av1.rs `update_ref_frames` — for each refreshed frame buffer i, build `models[j]=(fh.global_motion_type[j], fh.global_motion_params[j])` for j in 0..7 and store `self.ref_global_models[i]=models` (matches C++ VulkanAV1Decoder.cpp:399). Also added a `[RUST-PI-F3]` full picture-info dump (av1.rs, frame_count==3) for future diffs.
- **PSNR before→after (8 frames):**
  - **Frame 1: 37.35 dB → inf (PERFECT MATCH).**
  - Frame 2: 32.41 → 37.15; Frame 3: 29.91 → 32.76; Frame 4: 27.20 → 28.96; Frame 5: 25.62 → 26.76; Frame 6: 24.82 → 26.94; Frame 7: 26.84 → 25.68.
  - **Avg: 29.16 → 29.71 dB. Perfect matches: 1 → 2. VERDICT: FAIL → PASS.**
  - OUT-PROBE: ext3 MAD_y 0.0000 → 7.98, ext4 MAD_y 0.0000 → 3.38 (no longer exactly the keyframe). The multi-ref "copies primary ref" bug is GONE.
- **Remaining:** residual 25-33 dB on frames 3-7 = reference-error accumulation (DPB slots slightly off, compounding). Not the "exactly keyframe" bug.
- **NEXT (iteration 23):** hunt the residual per-frame error that accumulates. Top candidates: (a) verify gm_params now match C++ for ALL frames (dump RUST-PI-F3 for fc=4/5 too + compare [CPP-PI]); (b) the CDEF invalid-index leftover is harmless, but confirm the driver ignores it; (c) check whether the residual is inherent HW-vs-dav1d sub-pixel/filtering diff by comparing our output vs the C++ reference output (both HW) frame-by-frame — if Rust≈C++ but both ≠ dav1d, it's inherent; if Rust≠C++ on the same frame, there's still a setup diff.

## Iteration 28 (reference std_info: RefFrameSignBias/SavedOrderHints/OrderHint — CONFIRMED correct; C++ binary found broken)
- **Task:** dump the FULL `StdVideoDecodeAV1ReferenceInfo` (OrderHint, SavedOrderHints[8], RefFrameSignBias, frame_type, flags) for EVERY reference of EVERY frame (0-7) and compare EXACTLY against C++ (VulkanAV1Decoder.cpp:314-335), esp. RefFrameSignBias.
- **CRITICAL DISCOVERY — the C++ binary is BROKEN (not the pixel-perfect reference):** the C++ source has UNCOMMITTED modifications. `VulkanVideoParser.cpp` had the AV1 `FillDpbAV1State(...)` call + the entire reference-slot setup block REMOVED from the `DecodePicture` AV1 branch (only the H264/VP9 FillDpb calls remain). Result: running `vk-video-dec-test --deviceID 2504 ... --maxFrameCount 8` prints `[CPP-DEC] frame1: refSlotCount=0` (frame1 should have 1 ref) and **SEGFAULTS (exit 139) after frame1**. So the current C++ binary CANNOT produce ground-truth reference std_info for frames 2-7, and is NOT pixel-perfect. The "C++ is pixel-perfect" premise is from an EARLIER, unmodified build.
  - The reference std_info COMPUTATION itself (VulkanAV1Decoder.cpp:314-335, in `BeginPicture`) is ORIGINAL/untouched — the only C++ diff there is a 32-line `[CPP-REFINFO]` printf dump (+ `#include <cstdio>`). So the C++ LOGIC is still valid for comparison; only the runtime binary is broken.
- **C++ reference std_info logic (read from original code, VulkanAV1Decoder.cpp):**
  - Per ref buffer i (dpbSlotInfos[i]): `OrderHint = m_pBuffers[i].order_hint`; `SavedOrderHints[av1name]` (av1name 1..7) = `m_pBuffers[i].SavedOrderHints[av1name]` (index 0 never set→0); `RefFrameSignBias` bit av1name (1..7) set where `m_pBuffers[i].RefFrameSignBias[av1name] <= 0`; `frame_type`/`dcdf`/`seg` from `m_pBuffers[i]`.
  - Setup slot: `OrderHint = cur.OrderHint`; `SavedOrderHints = cur.OrderHints` (all 8); `RefFrameSignBias` bit av1name (0..7) set where `m_pBuffers[0].RefFrameSignBias[av1name] <= 0` (uses INTRA buffer 0, incl. bit 0); flags from cur.
  - `m_pBuffers[i].RefFrameSignBias[refName]` (refName 1..7) = `GetRelativeDist(pStd->OrderHint, pStd->OrderHints[refName])` stored in `UpdateFramePointers` (line 425); `GetRelativeDist(a,b) = (diff & (m-1)) - (diff & m)`, `m = 1<<(bits-1)`. **KEY: C++ stores the RAW signed distance (init 0) and computes the bitmask at READ time as `(dist <= 0)`.**
- **RUST-REFINFO dump added** (av1.rs `record_decode_command`): prints per-frame, per-ref + SETUP: OrderHint, RefFrameSignBias(hex), frame_type, dcdf, seg, SavedOH[8].
- **Comparison (Python simulation of the exact C++ state machine vs Rust RUST-REFINFO, frames 0-7):**
  - **ALL reference + setup values MATCH C++ EXACTLY for frames 1-7 (the error frames).** OrderHint, SavedOrderHints, RefFrameSignBias, frame_type, flags all identical.
  - **ONLY difference: fc0 SETUP `RefFrameSignBias` = C++ `0xff` vs Rust `0x01`.**
- **ROOT CAUSE of that one difference:** Rust stored the RefFrameSignBias **bitmask directly** (init 0) at refresh time; C++ stores the **raw signed distance** (init 0) and computes `(dist <= 0)` at read time. For an UNREFRESHED buffer (fc0's setup slot, before fb0 is refreshed), C++ gives `(0 <= 0) = true` for all 8 bits → `0xff`; Rust gave `0 | 0x01 = 0x01`. After fc0 (KEY, refresh=0xFF) all buffers are refreshed, so frames 1-7 match.
- **Change (Rust, av1.rs):** store raw distances like C++ and compute the bitmask at read time.
  - `frame_buffer_ref_sign_bias: [u8;8]` → `frame_buffer_ref_dist: [[i8;8];8]` (field decl ~639, init ~659, reset_dpb ~795).
  - `set_frame_buffer_ref_info` (~736): store `rel as i8` per ref_name 1..7 (no bitmask).
  - `get_frame_buffer_ref_info` (~758): compute bias = bits 1..7 where `ref_dist[fb][n] <= 0`.
  - setup slot (~999): compute bias = bits 0..7 where `ref_dist[0][n] <= 0` (bit 0 always set since ref_dist[0][0]=0).
  - fc0 SETUP now prints `RefFrameSignBias=ff` (matches C++).
- **PSNR before→after (8 frames): UNCHANGED.** f0 inf, f1 inf, f2 37.15, f3 32.76, f4 28.96, f5 26.76, f6 26.94, f7 25.68. **Avg 29.71 dB. Perfect: 2. VERDICT PASS.**
- **CONCLUSION: the reference std_info (OrderHint, SavedOrderHints, RefFrameSignBias, flags) is NOT the root cause of the frames 2-7 error — it already matched C++ for all error frames.** The fc0 SETUP fix is correct-but-harmless (fc0 is KEY, no refs, decodes perfectly).
- **NEXT (iteration 29, single most important):** the residual accumulating Y-dominant error (frames 2-7) is NOT in the reference std_info. Two blocking sub-tasks:
  1. **RESTORE the C++ binary** (re-add the `FillDpbAV1State` call + reference-slot block in `VulkanVideoParser.cpp` AV1 branch, keep the `[CPP-PI]`/`[CPP-REFINFO]` dumps, rebuild) so Rust-vs-C++ HW output can be compared frame-by-frame. This definitively answers inherent(HW-vs-dav1d)-vs-bug: if Rust≈C++ but both≠dav1d → inherent; if Rust≠C++ → still a Rust setup diff (then bisect the first diverging frame).
  2. If C++ is trusted as pixel-perfect: re-verify the picture-info fields that drive MOTION COMP for ALL frames 2-7 (esp. `interpolation_filter`/sub-pixel, `gm_params` for non-primary refs, `SkipModeFrame`) via RUST-PI-ALL vs [CPP-PI]; the Y-dominant widespread accumulating signature points at motion-comp (sub-pixel filter or warped-motion), not loop-filter (error is not at block boundaries).
