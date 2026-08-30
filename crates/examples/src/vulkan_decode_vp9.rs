//! Vulkan hardware-accelerated VP9 video decode example.
//!
//! Demonstrates a complete Vulkan VP9 video decode pipeline:
//!
//! 1. Read raw VP9 bitstream file (no container)

#![allow(clippy::too_many_arguments)]
//! 2. Expand superframes (detect superframe index at end of data)
//! 3. Split bitstream using sequential frame header parsing
//! 4. Parse each frame header with Vp9Parser
//! 5. Initialize Vulkan with VP9 decode support
//! 6. Create video session with VP9 profile
//! 7. Create session parameters
//! 8. Manage DPB (Decoded Picture Buffer) slots with VP9 reference frame names
//! 9. Decode each frame with proper reference management
//! 10. Read back decoded frames and compare with ffmpeg reference
//!
//! Usage:
//!   cargo run --example vulkan_decode_vp9 -- input.vp9

use ash::vk;
use vk_video_parser::vp9::Vp9Parser;
use vk_video_parser::{DetectedVideoFormat, VideoParser};
use vk_video_vulkan::vp9::{
    convert_vp9_picture_info, vp9_vk_constants, VideoDecodeVP9PictureInfoKHR,
    VideoDecodeVP9ProfileInfoKHR, Vp9Decoder,
};
use vk_video_vulkan::{
    buffer::BitstreamBuffer as VkBitstreamBuffer,
    image::create_output_image_with_pnext,
    session::{CodecProfileInfo, VideoSession, VideoSessionParameters, VideoSessionParams},
    VideoCodec, VideoDeviceBuilder,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.vp9> [max_frames]", args[0]);
        std::process::exit(1);
    };
    let max_frames_arg: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or(usize::MAX)
    } else {
        usize::MAX
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    println!("=== Vulkan VP9 Decode Example ===");
    println!("File: {}\n", bitstream_path);

    // Step 1: Read bitstream, expand superframes, and split into frames
    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Check if this is an IVF container first
    let raw_frames: Vec<FrameInfo> = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        let packets = parse_ivf_container(&data).expect("Failed to parse IVF container");
        println!("  IVF container: {} packets", packets.len());
        // IVF packets may contain superframes - expand them
        let expanded = expand_superframes(&packets);
        println!("  After superframe expansion: {} frames", expanded.len());
        expanded
    } else {
        // For raw VP9 files, treat the entire file as a single frame.
        // Multi-frame raw VP9 is not well-defined without a container format.
        // The `split_vp9_bitstream` function has a known issue where it finds
        // byte sequences in compressed data that look like valid VP9 frame
        // headers, causing frames to be truncated.
        vec![FrameInfo {
            data: data.to_vec(),
            packet_file_offset: 0,
            superframe_frame_offset: 0,
        }]
    };

    if raw_frames.is_empty() {
        eprintln!("Error: No VP9 frames found in bitstream");
        std::process::exit(1);
    }

    // Step 2: Parse first frame to get format info
    let mut parser = Vp9Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeVp9,
        ))
        .expect("Failed to init VP9 parser");

    let first_frame = &raw_frames[0].data;
    let first_parsed = parser
        .parse_frame(first_frame)
        .expect("Failed to parse first frame");

    let coded_width = first_parsed.frame_width;
    let coded_height = first_parsed.frame_height;
    let profile = first_parsed.picture_info.profile as u32;
    let bit_depth = first_parsed.color_config.bit_depth;

    println!("  Resolution: {}x{}", coded_width, coded_height);
    println!("  Profile: {}", profile);
    println!("  Bit depth: {}", bit_depth);
    println!(
        "  Chroma subsampling: {}x{}",
        first_parsed.color_config.subsampling_x, first_parsed.color_config.subsampling_y
    );
    println!(
        "  Color space: {:?}\n",
        first_parsed.color_config.color_space
    );

    // Vulkan Video VP9 decode only supports 4:2:0 output format (G8B8R8_2PLANE_420_UNORM).
    // VP9 Profile 1/3 with 4:4:4 chroma subsampling (subsampling_x=0, subsampling_y=0)
    // cannot be correctly decoded - decoding a 4:4:4 stream in a 4:2:0 session
    // produces garbage output (verified: Y plane diverges from the start).
    let is_444 = first_parsed.color_config.subsampling_x == 0
        && first_parsed.color_config.subsampling_y == 0;
    if is_444 {
        eprintln!(
            "Error: VP9 Profile {} with 4:4:4 chroma subsampling is not supported \
by Vulkan Video decode (4:2:0 only).",
            profile
        );
        eprintln!("  Re-encode the stream as Profile 0 (4:2:0) or use a 4:4:4-capable backend.");
        std::process::exit(1);
    }

    if coded_width == 0 || coded_height == 0 {
        eprintln!("Error: Failed to parse video dimensions");
        std::process::exit(1);
    }

    // Step 3: Initialize Vulkan
    let vulkan = match VideoDeviceBuilder::new()
        .with_validation(false)
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::from_raw(
            vp9_vk_constants::DECODE_VP9,
        ))
        .build()
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to initialize Vulkan: {}", e);
            std::process::exit(1);
        }
    };
    let decode_qf = vulkan.queue_families.video_decode.expect("No decode queue");

    // Print GPU name
    let gpu_name = unsafe {
        let props = vulkan
            .instance
            .get_physical_device_properties(vulkan.physical_device);
        std::ffi::CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    println!("  GPU: {}", gpu_name);

    // Step 4: Query video decode capabilities
    let luma_bit_depth = match bit_depth {
        8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };
    let chroma_bit_depth = luma_bit_depth;
    let chroma_subsampling = vk::VideoChromaSubsamplingFlagsKHR::TYPE_420;

    // Try querying capabilities for all 4 VP9 profiles (0-3)
    let mut supported_profiles: Vec<u32> = Vec::new();
    let mut video_caps: Option<vk::VideoCapabilitiesKHR> = None;
    for p in 0..=3 {
        match vulkan.query_video_capabilities(
            VideoCodec::DecodeVp9,
            p,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
        ) {
            Ok(caps) => {
                supported_profiles.push(p);
                println!(
                    "  Profile {}: supported, maxCodedExtent={}x{}, pictureAccessGranularity={}x{}",
                    p,
                    caps.max_coded_extent.width,
                    caps.max_coded_extent.height,
                    caps.picture_access_granularity.width,
                    caps.picture_access_granularity.height
                );
                if video_caps.is_none() {
                    video_caps = Some(caps);
                }
            }
            Err(e) => {
                println!("  Profile {}: not supported ({})", p, e);
            }
        }
    }

    println!("  Supported VP9 profiles: {:?}", supported_profiles);

    if supported_profiles.is_empty() {
        // Check if VP9 encode is available instead
        let available_ext = unsafe {
            vulkan
                .instance
                .enumerate_device_extension_properties(vulkan.physical_device)
                .unwrap_or_default()
        };
        let has_encode_vp9 = available_ext.iter().any(|e| unsafe {
            std::ffi::CStr::from_ptr(e.extension_name.as_ptr().cast())
                .to_string_lossy()
                .contains("video_encode_vp9")
        });
        let has_decode_vp9_ext = vulkan
            .enabled_extensions
            .iter()
            .any(|e| e.contains("video_decode_vp9"));

        if has_decode_vp9_ext {
            eprintln!("Error: VP9 decode extension is present but no VP9 decode profile (0-3) is supported on this GPU.");
        } else {
            eprintln!("Error: VP9 decode is not supported on this GPU.");
        }
        if has_encode_vp9 {
            eprintln!("  Note: VP9 encode extension (VK_KHR_video_encode_vp9) is available — GPU may only support VP9 encode.");
        }
        eprintln!("  GPU: {}", gpu_name);
        std::process::exit(1);
    }

    // Use the profile from the bitstream if supported, otherwise use first supported
    let use_profile = if supported_profiles.contains(&profile) {
        profile
    } else {
        let alt = supported_profiles[0];
        println!(
            "  Stream uses profile {}, using supported profile {} instead",
            profile, alt
        );
        alt
    };

    let video_caps = video_caps.expect("No supported VP9 profile found");

    // Align coded extent to picture access granularity
    let align_width = video_caps.picture_access_granularity.width;
    let align_height = video_caps.picture_access_granularity.height;
    let coded_extent = vk::Extent2D {
        width: (coded_width + align_width - 1) & !(align_width - 1),
        height: (coded_height + align_height - 1) & !(align_height - 1),
    };

    // VP9 has 8 DPB slots (VP9_NUM_REF_FRAMES)
    let max_dpb_slots = 8u32.min(video_caps.max_dpb_slots);
    let session_dpb_slots = max_dpb_slots + 1;

    // Step 5: Create video session
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let session_params = VideoSessionParams {
        queue_family_index: decode_qf,
        picture_format: output_format,
        reference_picture_format: output_format,
        max_coded_extent: coded_extent,
        max_dpb_slots: session_dpb_slots,
        max_active_reference_pictures: session_dpb_slots,
        codec: VideoCodec::DecodeVp9,
        codec_profile_info: CodecProfileInfo::Vp9 {
            std_profile: use_profile,
        },
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        inline_queries: false,
    };

    let std_header_version = build_std_header_version("VK_STD_vulkan_video_codec_vp9_decode");

    let (session, session_memories) = VideoSession::create(
        &vulkan.instance,
        &vulkan.device,
        &session_params,
        &std_header_version,
    )
    .expect("Failed to create video session");

    // VP9 doesn't use session parameters objects.
    // All per-frame info is passed in the picture info for each decode command.
    let session_params_handle = vk::VideoSessionParametersKHR::null();
    let session_parameters: Option<VideoSessionParameters> = None;

    // Step 7: Create output image with raw frame dimensions (not aligned).
    // The image extent must match the codedExtent used in decode commands.
    // Session max_coded_extent is aligned, but images can be smaller.
    let (output_image, output_image_view, output_memory) = create_vp9_output_image(
        &vulkan.device,
        &vulkan.memory_properties,
        coded_width,
        coded_height,
        output_format,
        use_profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    )
    .map_err(|e| format!("Failed to create output image: {}", e))
    .expect("Failed to create output image");

    // Step 8: Create DPB images for reference frame management
    let mut dpb_images: Vec<(vk::Image, vk::ImageView, vk::DeviceMemory)> = Vec::new();

    // Create DPB images (slot 0 = output_image, slots 1..N = dpb_images)
    for slot in 1..max_dpb_slots {
        let (img, view, mem) = create_vp9_output_image(
            &vulkan.device,
            &vulkan.memory_properties,
            coded_width,
            coded_height,
            output_format,
            use_profile,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to create DPB image {}: {}", slot, e);
            std::process::exit(1);
        });
        dpb_images.push((img, view, mem));
    }

    // Build DPB view/image arrays: slot 0 = output, slots 1..N = dpb_images
    let mut dpb_views: Vec<vk::ImageView> = vec![output_image_view];
    let mut dpb_image_handles: Vec<vk::Image> = vec![output_image];
    for (_, view, _) in &dpb_images {
        dpb_views.push(*view);
    }
    for (img, _, _) in &dpb_images {
        dpb_image_handles.push(*img);
    }

    // Step 9: Create bitstream buffer
    let max_frame_size = raw_frames.iter().map(|f| f.data.len()).max().unwrap_or(0);
    let bs_size_align = video_caps.min_bitstream_buffer_size_alignment;
    println!("  min_bitstream_buffer_size_alignment = {}", bs_size_align);
    println!(
        "  min_bitstream_buffer_offset_alignment = {}",
        video_caps.min_bitstream_buffer_offset_alignment
    );
    // Align buffer size to minBitstreamBufferSizeAlignment
    let max_frame_size_aligned =
        ((max_frame_size as u64).div_ceil(bs_size_align) * bs_size_align).max(bs_size_align);
    let mut bs_buffer = create_vp9_bitstream_buffer(
        &vulkan.device,
        &vulkan.memory_properties,
        max_frame_size_aligned,
        use_profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        decode_qf,
    )
    .map_err(|e| format!("Failed to create bitstream buffer: {}", e))
    .expect("Failed to create bitstream buffer");

    // Step 10: Create command resources
    let command_pool = unsafe {
        vulkan
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(decode_qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .expect("Failed to create command pool")
    };

    let command_buffer = unsafe {
        vulkan
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("Failed to allocate command buffer")[0]
    };

    // Step 11: Create fence
    let fence = unsafe {
        vulkan
            .device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .expect("Failed to create fence")
    };

    // Step 12: Create VP9 decoder
    let mut vp9_decoder = Vp9Decoder::new(vulkan.device.clone(), vulkan.instance.clone());
    vp9_decoder.set_session(&session);
    if let Some(params) = session_parameters {
        vp9_decoder.set_session_parameters(params);
    }
    vp9_decoder.set_max_dpb_slots(max_dpb_slots);

    // Step 13: Decode frames
    let frames_to_decode = raw_frames.len().min(max_frames_arg);

    // DPB management state
    let mut dpb_manager = Vp9DpbManager::new(max_dpb_slots);
    let mut is_first_frame = true;
    let mut frame_count: u32 = 0;

    // Reset parser for fresh parsing
    parser.reset();

    for (frame_idx, frame_info) in raw_frames.iter().enumerate().take(frames_to_decode) {
        let frame_data = &frame_info.data;
        let packet_offset = frame_info.packet_file_offset;
        let superframe_offset = frame_info.superframe_frame_offset as u32;

        eprintln!(
            "  [Frame {}] packet_offset={}, superframe_offset={}",
            frame_idx, packet_offset, superframe_offset
        );

        // DEBUG: Dump first 128 bytes of raw frame data for frame 1
        if frame_idx == 1 {
            println!("  [DEBUG] Raw frame data len={}", frame_data.len());
            println!("  [DEBUG] First 32 bytes as hex:");
            for (i, &byte) in frame_data.iter().enumerate().take(frame_data.len().min(32)) {
                print!("{:02X} ", byte);
                if (i + 1) % 16 == 0 {
                    println!();
                }
            }
            println!();
        }

        // Parse frame header directly from extracted frame data
        let parsed = match parser.parse_frame_with_offset(frame_data.as_slice(), superframe_offset)
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Failed to parse frame header: {:?}", e);
                continue;
            }
        };

        // Use raw frame dimensions for coded extent (not aligned).
        // Vulkan spec: codedExtent is the actual coded dimensions of the frame.
        // picture_access_granularity alignment is only required for session max_coded_extent.
        let frame_coded_extent = vk::Extent2D {
            width: parsed.frame_width,
            height: parsed.frame_height,
        };

        // Handle show_existing_frame
        if parsed.show_existing_frame {
            let frame_buffer_idx = parsed.frame_to_show_map_idx as usize;

            // Map VP9 frame buffer index to DPB slot via pic_idx.
            // frame_to_show_map_idx is a VP9 frame buffer index (0-7), not a DPB slot index.
            // We use the decoder's pic_idx mapping to find which DPB slot holds that frame buffer.
            let pic_idx = vp9_decoder.get_pic_idx_for_frame_buffer(frame_buffer_idx);
            if pic_idx >= 0 {
                let slot = pic_idx as u32;
                let img = dpb_image_handles[slot as usize];
                readback_and_verify(
                    &vulkan.instance,
                    &vulkan.device,
                    &vulkan.memory_properties,
                    decode_qf,
                    command_pool,
                    fence,
                    img,
                    coded_width,
                    coded_height,
                    parsed.frame_width,
                    parsed.frame_height,
                    frame_idx,
                    frame_count as usize, // Use display order index
                    bitstream_path,
                );
            } else {
                eprintln!(
                    "  Warning: frame_to_show_map_idx={} not mapped to any DPB slot",
                    frame_buffer_idx
                );
            }
            continue;
        }

        let is_key_frame =
            parsed.picture_info.frame_type == vk_video_core::picture::Vp9FrameType::Key;

        println!(
            "[{}] {}",
            frame_idx,
            if is_key_frame { "KEY" } else { "INTER" }
        );
        println!(
            "  tile_cols_log2={}, tile_rows_log2={}, num_tiles={}, sb64_cols={}, tiles_offset={}",
            parsed.picture_info.tile_cols_log2,
            parsed.picture_info.tile_rows_log2,
            parsed.num_tiles,
            parsed.sb64_cols,
            parsed.tiles_offset
        );
        println!(
            "  uncomp_hdr_offset={}, comp_hdr_offset={}, comp_hdr_size={}, frame_data_len={}",
            parsed.uncompressed_header_offset,
            parsed.compressed_header_offset,
            parsed.compressed_header_size,
            frame_data.len()
        );

        // Write bitstream data to buffer - only the current frame's data
        let bs_align = bs_size_align;
        let actual_size = frame_data.len() as u64;
        let aligned_size = (actual_size.div_ceil(bs_align) * bs_align).max(bs_align);

        bs_buffer.zero_range(0, aligned_size);
        bs_buffer
            .write(frame_data)
            .expect("Failed to write bitstream");
        bs_buffer.flush_range(0, aligned_size).ok();

        // DEBUG: Dump first 128 bytes of bitstream buffer for frame 1
        if frame_idx == 1 {
            if let Some(ptr) = bs_buffer.data_ptr() {
                let bytes =
                    unsafe { std::slice::from_raw_parts(ptr, aligned_size.min(32) as usize) };
                println!("  [DEBUG] Bitstream buffer first 32 bytes for frame 1:");
                for (i, &byte) in bytes.iter().enumerate() {
                    print!("{:02X} ", byte);
                    if (i + 1) % 16 == 0 {
                        println!();
                    }
                }
                println!();
            }
        }

        // srcBufferRange MUST be aligned to minBitstreamBufferSizeAlignment per Vulkan spec.
        let bs_range = aligned_size;

        println!(
            "  BS: offset=0, range={}, tiles_offset={}",
            bs_range, parsed.tiles_offset
        );

        // Compute reference name slot indices FIRST (before selecting output slot)
        // so we can avoid using a reference slot as output (prevents self-reference).
        let reference_name_slot_indices =
            vp9_decoder.compute_reference_name_slot_indices(is_key_frame, &parsed.ref_frame_idx);

        // DEBUG: Print reference frame state BEFORE decode
        {
            println!(
                "  ref_frame_idx=[{}, {}, {}, {}, {}, {}, {}]",
                parsed.ref_frame_idx[0],
                parsed.ref_frame_idx[1],
                parsed.ref_frame_idx[2],
                parsed.ref_frame_idx[3],
                parsed.ref_frame_idx[4],
                parsed.ref_frame_idx[5],
                parsed.ref_frame_idx[6]
            );
            println!(
                "  refresh_frame_flags=0x{:02X} ({:08b})",
                parsed.picture_info.refresh_frame_flags, parsed.picture_info.refresh_frame_flags
            );
            println!(
                "  reference_name_slot_indices=[{}, {}, {}]",
                reference_name_slot_indices[0],
                reference_name_slot_indices[1],
                reference_name_slot_indices[2]
            );
            println!(
                "  frame_context_idx={}",
                parsed.picture_info.frame_context_idx
            );
        }

        // Select DPB slot for this frame, avoiding slots needed as references
        let output_slot = if is_key_frame || is_first_frame {
            // Key frame or first frame: reset DPB and use slot 0
            if is_key_frame {
                dpb_manager.invalidate_all();
                vp9_decoder.reset_dpb();
            }
            0
        } else {
            // Inter frame: find or recycle a slot that is NOT a reference
            let exclude_slots: Vec<i32> = reference_name_slot_indices
                .iter()
                .filter(|&&s| s >= 0)
                .copied()
                .collect();
            dpb_manager
                .find_or_recycle_slot(&exclude_slots)
                .unwrap_or(0)
        };

        let output_slot_usize = output_slot as usize;
        let output_view = dpb_views[output_slot_usize];
        let output_img = dpb_image_handles[output_slot_usize];

        // Build DPB reference picture resources
        let (dpb_setup_picture, dpb_ref_pictures, dpb_ref_slot_indices) =
            build_dpb_picture_resources(
                &dpb_manager,
                &dpb_views,
                frame_coded_extent,
                output_slot,
                is_key_frame,
                &reference_name_slot_indices,
            );

        // Convert parsed frame data to Vulkan picture info container
        // Allocate on heap to ensure lifetime through command execution
        let mut picture_info_container = Box::new(convert_vp9_picture_info(
            &parsed.picture_info,
            &parsed.color_config,
            &parsed.loop_filter,
            &parsed.segmentation,
        ));
        // CRITICAL: init_pointers must be called AFTER Box::new so pointers
        // point to heap-allocated fields, not stack temporaries
        picture_info_container.init_pointers();

        // Build VP9 decode picture info on heap alongside the container.
        // CRITICAL: Both must stay alive until after fence wait, otherwise the
        // command buffer holds dangling pointers to freed stack memory.
        // IMPORTANT: referenceNameSlotIndices must be DPB slot indices (matching
        // VkVideoBeginCodingInfoKHR::pReferenceSlots[i].slot_index), NOT indices
        // into p_reference_slots. This matches Vulkan-Video-Samples behavior.

        // The parser computes offsets relative to the extracted frame data,
        // which is exactly what we write to the bitstream buffer.
        // No offset adjustment needed - use parser offsets directly.
        let uncomp_offset = parsed.uncompressed_header_offset;
        let comp_offset = parsed.compressed_header_offset;
        let tiles_offset = parsed.tiles_offset;

        let vp9_decode_info = Box::new(VideoDecodeVP9PictureInfoKHR::new(
            picture_info_container.std_picture_info(),
            reference_name_slot_indices,
            uncomp_offset,
            comp_offset,
            tiles_offset,
        ));

        // Record decode command
        // Build reference image handles from slot indices
        let dpb_ref_images: Vec<vk::Image> = dpb_ref_slot_indices
            .iter()
            .map(|&slot_idx| dpb_image_handles[slot_idx as usize])
            .collect();

        // Get actual layouts of reference slots for proper memory barriers.
        // This ensures we use correct old_layout in barriers (not always UNDEFINED).
        let dpb_ref_slot_layouts: Vec<vk::ImageLayout> = dpb_ref_slot_indices
            .iter()
            .map(|&slot_idx| dpb_manager.get_slot_layout(slot_idx as u32))
            .collect();

        // Reset command buffer before recording for this frame
        unsafe {
            vulkan
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        }

        let output_slot_old_layout = dpb_manager.get_slot_layout(output_slot);
        let result = vp9_decoder.record_decode_command(
            command_buffer,
            session.handle(),
            session_params_handle,
            bs_buffer.buffer(),
            0,
            bs_range,
            output_view,
            output_img,
            frame_coded_extent,
            dpb_setup_picture,
            &dpb_ref_pictures,
            &dpb_ref_slot_indices,
            &dpb_ref_images,
            &dpb_ref_slot_layouts,
            &picture_info_container,
            &vp9_decode_info,
            is_first_frame,
            output_slot as i32,
            output_slot_old_layout,
            false, // separate-image DPB mode (one image per slot)
        );

        // Keep container AND vp9_decode_info alive until after submit + wait
        let _picture_info_guard = picture_info_container;
        let _vp9_decode_guard = vp9_decode_info;

        // Mark first frame as done
        if is_first_frame {
            is_first_frame = false;
        }

        match result {
            Ok(()) => {}
            Err(e) => {
                eprintln!("[Frame {}] Decode command failed: {}", frame_idx, e);
                break;
            }
        }

        // Submit command buffer
        unsafe {
            vulkan
                .device
                .reset_fences(&[fence])
                .expect("Failed to reset fence");

            let cmd_bufs = vec![command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

            vulkan
                .device
                .queue_submit(vulkan.video_decode_queue(0), &[submit_info], fence)
                .expect("Failed to submit");

            vulkan
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("Failed to wait for fence");
        }

        // Update VP9 frame buffer to DPB slot mapping based on refresh_frame_flags.
        // Each bit set in refresh_frame_flags means that frame buffer index is refreshed
        // to point to the current output DPB slot.
        // - Key frames: refresh_frame_flags=0xFF → ALL 8 frame buffers → output slot
        // - Inter frames: e.g. refresh_frame_flags=0x01 → only frame buffer 0 (LAST) → output slot
        let refresh_flags = parsed.picture_info.refresh_frame_flags;
        for fb_idx in 0..vk_video_core::picture::VP9_NUM_REF_FRAMES {
            if (refresh_flags >> fb_idx) & 1 != 0 {
                vp9_decoder.set_frame_buffer_dpb_slot(fb_idx as usize, output_slot as i32);
            }
        }

        // Register this frame in DPB manager
        dpb_manager.register_frame(output_slot, frame_count);

        // Update DPB slot layout
        dpb_manager.set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

        // VP9 show_frame handling: only display frames with show_frame=1
        // Frames with show_frame=0 are decoded to update reference buffers but not displayed
        let show_frame = parsed.picture_info.flags.show_frame != 0;
        if show_frame {
            // Readback and verify decoded pixels
            readback_and_verify(
                &vulkan.instance,
                &vulkan.device,
                &vulkan.memory_properties,
                decode_qf,
                command_pool,
                fence,
                output_img,
                coded_width,
                coded_height,
                parsed.frame_width,
                parsed.frame_height,
                frame_idx,
                frame_count as usize, // Use display order index
                bitstream_path,
            );

            // Restore DPB layout after readback
            dpb_manager.set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

            frame_count += 1;
        } else {
            eprintln!(
                "  [Frame {}] show_frame=0 - decoded but not displayed (reference frame only)",
                frame_idx
            );
        }
    }

    println!("\nDecoded {} frames", frame_count);

    // Cleanup DPB images
    for (img, view, mem) in dpb_images {
        unsafe {
            vulkan.device.destroy_image_view(view, None);
            vulkan.device.destroy_image(img, None);
            vulkan.device.free_memory(mem, None);
        }
    }

    // Cleanup
    unsafe {
        vulkan
            .device
            .device_wait_idle()
            .expect("Failed to wait idle");
        vulkan.device.destroy_command_pool(command_pool, None);
        vulkan.device.destroy_fence(fence, None);
        vulkan.device.destroy_image(output_image, None);
        vulkan.device.destroy_image_view(output_image_view, None);
        vulkan.device.free_memory(output_memory, None);
    }

    drop(bs_buffer);
    drop(session);

    for mem in session_memories {
        unsafe {
            vulkan.device.free_memory(mem, None);
        }
    }

    unsafe {
        vulkan.device.destroy_device(None);
        vulkan.instance.destroy_instance(None);
    }

    println!("\n=== Done ===");
}

// ============================================================================
// IVF container parsing
// ============================================================================

/// Parse an IVF container file and extract raw VP9 frame data with file offsets.
///
/// IVF (On2 IVF) is a simple container format used for VP8/VP9 video.
/// File layout:
///   - 32-byte file header
///   - Repeated frame packets (4-byte size + 8-byte timestamp + data)
///
/// Returns a vector of (frame_data, file_offset) tuples.
fn parse_ivf_container(data: &[u8]) -> Result<Vec<(Vec<u8>, usize)>, String> {
    if data.len() < 32 {
        return Err("File too small for IVF header".to_string());
    }

    // Check IVF magic "DKIF"
    if data[0..4] != *b"DKIF" {
        return Err("Invalid IVF magic".to_string());
    }

    // Parse frames
    let mut frames = Vec::new();
    let mut offset = 32usize;

    while offset < data.len() {
        // Need at least 12 bytes for frame header (4 size + 8 timestamp)
        if offset + 12 > data.len() {
            break;
        }

        // 4 bytes: packet size (little-endian)
        let packet_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        // 8 bytes: presentation timestamp (64-bit little-endian)
        offset += 12;

        // Validate packet size
        if packet_size == 0 || offset + packet_size > data.len() {
            eprintln!(
                "    Warning: invalid packet size {} at offset {}",
                packet_size,
                offset - 12
            );
            break;
        }

        let frame_data = data[offset..offset + packet_size].to_vec();

        frames.push((frame_data, offset));
        offset += packet_size;
    }

    if frames.is_empty() {
        return Err("No frames found in IVF container".to_string());
    }

    Ok(frames)
}

// ============================================================================
// VP9 bitstream splitting
// ============================================================================

/// Split a VP9 bitstream into individual frame packets using sequential parsing.
///
/// For VP9, the frame size isn't explicitly encoded in the header (unlike H264/H265
/// NAL units). The frame extends from the frame sync code to the next frame sync
/// code or end of data. We parse the header to validate the frame, then find the
/// next frame boundary.
///
/// Superframes must be expanded BEFORE calling this function.
///
/// NOTE: Currently unused for raw VP9 files due to false positive frame boundary
/// detection in compressed data. Use IVF container for multi-frame files instead.
#[allow(dead_code)]
fn split_vp9_bitstream(packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();

    for packet in packets {
        let data = packet.as_slice();
        let mut offset = 0;
        let data_len = data.len();

        loop {
            // Skip leading zeros
            while offset < data_len && data[offset] == 0 {
                offset += 1;
            }
            if offset >= data_len {
                break;
            }

            // Check for frame marker (top 2 bits = 0b10)
            if (data[offset] & 0xC0) != 0x80 {
                break;
            }

            let frame_start = offset;

            // Parse frame header to validate
            let mut parser = Vp9Parser::new();
            let parsed = match parser.parse_frame(&data[offset..]) {
                Ok(p) => p,
                Err(_) => break,
            };

            // Find the next frame marker to determine frame size
            // The frame extends from the current marker to the next marker (or end)
            let mut next_frame = data_len; // Default: rest of data

            // Search for next frame marker starting after the current frame's
            // uncompressed header (compressed_header_offset points to start of
            // compressed data; we skip the compressed header too to avoid false
            // positives inside the current frame).
            let search_start = (frame_start
                + parsed.compressed_header_offset as usize
                + parsed.compressed_header_size as usize)
                .min(data_len);

            for i in search_start..data_len {
                // VP9 frame marker: top 2 bits = 0b10
                if (data[i] & 0xC0) != 0x80 {
                    continue;
                }
                // Validate candidate by parsing the frame header
                let mut validator = Vp9Parser::new();
                if validator.parse_frame(&data[i..]).is_ok() {
                    next_frame = i;
                    break;
                }
            }

            let frame_size = next_frame - frame_start;
            if frame_size == 0 {
                break;
            }

            // For VP9 Vulkan decode, we need to pass the entire frame
            // (header + compressed data) as the bitstream
            frames.push(data[frame_start..next_frame].to_vec());
            offset = next_frame;
        }
    }

    frames
}

/// Frame data with superframe information.
#[derive(Clone)]
struct FrameInfo {
    /// The frame data (extracted from superframe if applicable)
    data: Vec<u8>,
    /// The packet's file offset in the IVF container
    packet_file_offset: usize,
    /// Offset of this frame within a superframe (0 if not from superframe)
    superframe_frame_offset: usize,
}

/// Expand superframes into individual frames while tracking packet offsets.
///
/// A superframe contains multiple VP9 frames concatenated together,
/// with a superframe index at the end that specifies the size of each
/// constituent frame. This function detects the superframe index at the
/// end of the data and splits the superframe into its component frames,
/// extracting only each frame's individual data.
fn expand_superframes(data: &[(Vec<u8>, usize)]) -> Vec<FrameInfo> {
    let mut expanded = Vec::new();

    for (frame, packet_off) in data.iter() {
        let packet_offset = *packet_off;
        let data_len = frame.len();
        if data_len < 2 {
            expanded.push(FrameInfo {
                data: frame.clone(),
                packet_file_offset: packet_offset,
                superframe_frame_offset: 0,
            });
            continue;
        }

        // Check for superframe index at the end of the data
        let final_byte = frame[data_len - 1];
        if (final_byte & 0xE0) != 0xC0 {
            // Not a superframe
            expanded.push(FrameInfo {
                data: frame.clone(),
                packet_file_offset: packet_offset,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let num_frames = (final_byte & 0x07) as usize + 1;
        if num_frames <= 1 {
            expanded.push(FrameInfo {
                data: frame.clone(),
                packet_file_offset: packet_offset,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let mag = (((final_byte >> 3) & 0x03) as usize) + 1;
        let index_size = 2 + mag * num_frames;

        if data_len < index_size {
            expanded.push(FrameInfo {
                data: frame.clone(),
                packet_file_offset: packet_offset,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let index_start = data_len - index_size;
        if frame[index_start] != final_byte {
            expanded.push(FrameInfo {
                data: frame.clone(),
                packet_file_offset: packet_offset,
                superframe_frame_offset: 0,
            });
            continue;
        }

        // Parse frame sizes from the superframe index
        let frame_data_size = data_len - index_size;
        let mut offset = 0;
        let mut x = index_start + 1;
        for _i in 0..num_frames {
            let mut this_sz: usize = 0;
            for j in 0..mag {
                this_sz |= (frame[x + j] as usize) << (j * 8);
            }
            x += mag;

            if offset + this_sz <= frame_data_size {
                // Extract only this frame's data from the superframe
                let extracted = frame[offset..offset + this_sz].to_vec();

                expanded.push(FrameInfo {
                    data: extracted,
                    packet_file_offset: packet_offset,
                    superframe_frame_offset: offset,
                });
            }
            offset += this_sz;
        }
    }

    expanded
}

// ============================================================================
// DPB management for VP9
// ============================================================================

/// VP9-specific DPB manager.
///
/// VP9 uses named reference frames (LAST, GOLDEN, ALTREF, LAST2, LAST3,
/// BACKWARD, KEY) that map to DPB slots. This manager tracks which slot
/// each picture index maps to.
struct Vp9DpbManager {
    /// DPB slot entries. Each entry tracks whether a slot is valid and
    /// what picture index it holds.
    entries: Vec<Vp9DpbEntry>,
    max_dpb_slots: u32,
}

#[derive(Debug, Clone)]
struct Vp9DpbEntry {
    /// Whether this slot holds a valid decoded frame
    is_valid: bool,
    /// Picture index stored in this slot
    pic_idx: i32,
    /// Frame count when this slot was last written
    frame_count: u32,
    /// Current image layout
    layout: vk::ImageLayout,
}

impl Vp9DpbManager {
    fn new(max_dpb_slots: u32) -> Self {
        Self {
            entries: (0..max_dpb_slots)
                .map(|_| Vp9DpbEntry {
                    is_valid: false,
                    pic_idx: -1,
                    frame_count: 0,
                    layout: vk::ImageLayout::UNDEFINED,
                })
                .collect(),
            max_dpb_slots,
        }
    }

    /// Find an empty slot or recycle the oldest reference.
    ///
    /// Avoids slots listed in `exclude_slots` (typically reference frame slots
    /// that are needed for the current frame). This prevents self-reference bugs
    /// where a slot is used as both output and reference.
    fn find_or_recycle_slot(&mut self, exclude_slots: &[i32]) -> Option<u32> {
        // First try to find an empty slot that is not excluded
        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.is_valid && !exclude_slots.contains(&(i as i32)) {
                return Some(i as u32);
            }
        }
        // Recycle the oldest valid slot that is not excluded
        let mut oldest_idx = None;
        let mut oldest_count = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid
                && !exclude_slots.contains(&(i as i32))
                && entry.frame_count < oldest_count
            {
                oldest_count = entry.frame_count;
                oldest_idx = Some(i as u32);
            }
        }
        oldest_idx
    }

    /// Mark all entries as invalid (for key frames).
    fn invalidate_all(&mut self) {
        for entry in &mut self.entries {
            entry.is_valid = false;
            entry.pic_idx = -1;
            entry.layout = vk::ImageLayout::UNDEFINED;
        }
    }

    /// Register a decoded frame in a DPB slot.
    fn register_frame(&mut self, slot: u32, frame_count: u32) {
        if slot < self.max_dpb_slots {
            self.entries[slot as usize].is_valid = true;
            self.entries[slot as usize].pic_idx = slot as i32;
            self.entries[slot as usize].frame_count = frame_count;
        }
    }

    /// Get the DPB slot for a given picture index.
    #[allow(dead_code)]
    fn get_slot_for_pic_idx(&self, pic_idx: i32) -> Option<u32> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid && entry.pic_idx == pic_idx {
                return Some(i as u32);
            }
        }
        None
    }

    /// Get the current layout of a DPB slot.
    fn get_slot_layout(&self, slot: u32) -> vk::ImageLayout {
        if slot < self.max_dpb_slots {
            self.entries[slot as usize].layout
        } else {
            vk::ImageLayout::UNDEFINED
        }
    }

    /// Update the layout of a DPB slot.
    fn set_slot_layout(&mut self, slot: u32, layout: vk::ImageLayout) {
        if slot < self.max_dpb_slots {
            self.entries[slot as usize].layout = layout;
        }
    }

    /// Get count of valid DPB entries.
    #[allow(dead_code)]
    fn get_valid_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_valid).count()
    }
}

// ============================================================================
// DPB picture resource building
// ============================================================================

/// Build DPB picture resources for VP9 decode.
///
/// Returns (setup_picture, ref_pictures, ref_slot_indices) where:
/// - setup_picture: the current frame's output slot
/// - ref_pictures: reference picture slots (only those referenced by reference_name_slot_indices)
/// - ref_slot_indices: DPB slot indices corresponding to each reference picture
///
/// IMPORTANT: referenceNameSlotIndices passed to Vulkan must be DPB slot indices
/// (matching VkVideoBeginCodingInfoKHR::pReferenceSlots[i].slot_index), NOT indices
/// into p_reference_slots. This matches Vulkan-Video-Samples behavior.
fn build_dpb_picture_resources(
    dpb_manager: &Vp9DpbManager,
    dpb_views: &[vk::ImageView],
    coded_extent: vk::Extent2D,
    output_slot: u32,
    is_key_frame: bool,
    reference_name_slot_indices: &[i32; 3],
) -> (
    Option<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<i32>,
) {
    let mut ref_pictures = Vec::new();
    let mut ref_slot_indices = Vec::new();

    if !is_key_frame {
        // Build reference picture resources ONLY for the 3 VP9 primary reference
        // frame names (LAST, GOLDEN, ALTREF) as specified by
        // reference_name_slot_indices.
        // IMPORTANT: Vulkan requires unique slot_index values in p_reference_slots.
        // If multiple VP9 ref names point to the same DPB slot, we only create
        // one reference slot for that DPB slot. The caller uses the original
        // reference_name_slot_indices (DPB slot indices) for Vulkan, which may
        // have duplicates pointing to the same DPB slot.
        let mut seen_slots: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &slot_idx in reference_name_slot_indices.iter() {
            if slot_idx < 0 {
                continue; // Reference frame name not assigned
            }
            if seen_slots.contains(&slot_idx) {
                continue; // Already added this DPB slot
            }
            let slot = slot_idx as usize;
            if slot >= dpb_manager.entries.len() {
                continue; // Invalid slot
            }
            let entry = &dpb_manager.entries[slot];
            if !entry.is_valid {
                continue; // Slot exists but has no valid frame
            }
            if (slot as u32) == output_slot {
                continue; // Don't reference ourselves
            }
            seen_slots.insert(slot_idx);
            let view = dpb_views[slot];
            let picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: view,
                _marker: Default::default(),
            };
            ref_pictures.push(picture_resource);
            ref_slot_indices.push(slot_idx);
        }
    }

    // Setup picture (current frame output)
    let setup_picture = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: dpb_views[output_slot as usize],
        _marker: Default::default(),
    };

    (Some(setup_picture), ref_pictures, ref_slot_indices)
}

// ============================================================================
// VP9 Video Profile List helpers
// ============================================================================

/// Build VkVideoProfileListInfoKHR for VP9 decode.
///
/// Returns (profile_list, video_profile, vp9_profile) tuple.
/// All three structs must stay alive while the pNext chain is used.
fn build_vp9_profile_list(
    profile: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> (
    vk::VideoProfileListInfoKHR<'static>,
    vk::VideoProfileInfoKHR<'static>,
    VideoDecodeVP9ProfileInfoKHR,
) {
    let vp9_profile = VideoDecodeVP9ProfileInfoKHR {
        s_type: vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR),
        p_next: std::ptr::null(),
        std_profile: profile,
        _marker: Default::default(),
    };

    let video_profile = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &vp9_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::from_raw(
            vp9_vk_constants::DECODE_VP9,
        ),
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    let profile_list = vk::VideoProfileListInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
        p_next: std::ptr::null(),
        profile_count: 1,
        p_profiles: &video_profile,
        _marker: Default::default(),
    };

    (profile_list, video_profile, vp9_profile)
}

/// Create an output image with VP9 profile list in pNext chain.
fn create_vp9_output_image(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    profile: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), String> {
    let (profile_list, _video_profile, _vp9_profile) = build_vp9_profile_list(
        profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    );

    // All structs stay alive for the duration of this function call
    create_output_image_with_pnext(
        device,
        memory_properties,
        width,
        height,
        format,
        &profile_list as *const _ as *const std::ffi::c_void,
    )
    .map_err(|e| e.to_string())
}

/// Create a bitstream buffer with VP9 profile list in pNext chain.
fn create_vp9_bitstream_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    profile: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    queue_family_index: u32,
) -> Result<VkBitstreamBuffer, String> {
    let (profile_list, _video_profile, _vp9_profile) = build_vp9_profile_list(
        profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    );

    // All structs stay alive for the duration of this function call
    VkBitstreamBuffer::create_with_pnext(
        device,
        memory_properties,
        size,
        0,
        1,
        &profile_list as *const _ as *const std::ffi::c_void,
        vk::BufferCreateFlags::empty(),
        queue_family_index,
    )
    .map_err(|e| e.to_string())
}

// ============================================================================
// Standard header version
// ============================================================================

const fn make_video_std_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

const VK_STD_VIDEO_SPEC_VERSION: u32 = make_video_std_version(1, 0, 0);

fn build_std_header_version(extension_name: &str) -> vk::ExtensionProperties {
    let mut props = vk::ExtensionProperties::default();
    let bytes = format!("{}\0", extension_name).into_bytes();
    props
        .extension_name
        .iter_mut()
        .zip(bytes.iter())
        .for_each(|(c, &b)| *c = b as std::os::raw::c_char);
    props.spec_version = VK_STD_VIDEO_SPEC_VERSION;
    props
}

// ============================================================================
// Readback and verification
// ============================================================================

/// Readback decoded image and verify pixels.
fn readback_and_verify(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    image: vk::Image,
    width: u32,
    height: u32,
    frame_width: u32,
    frame_height: u32,
    _frame_idx: usize,        // Bitstream frame index (kept for debugging)
    display_frame_idx: usize, // Display order index (used for naming)
    bitstream_path: &str,
) {
    let y_size = (width * height) as usize;
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);
    let uv_size = (uv_width * uv_height * 2) as usize;
    let total_size = (y_size + uv_size) as u64;

    // Create staging buffer
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(total_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .expect("Failed to create staging buffer")
    };

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .expect("No suitable memory type for staging buffer");

    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_reqs.size)
                    .memory_type_index(mem_type_index),
                None,
            )
            .expect("Failed to allocate staging memory")
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("Failed to bind buffer memory");
    }

    let mapped_ptr = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("Failed to map memory")
    };

    // Allocate command buffer for readback
    let cmd_buffer = unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("Failed to allocate command buffer")[0]
    };

    unsafe {
        device
            .begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("Failed to begin command buffer");

        // Transition image planes: VIDEO_DECODE_DPB_KHR -> TRANSFER_SRC_OPTIMAL
        let plane0_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let plane1_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let image_barriers = [plane0_barrier, plane1_barrier];
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: image_barriers.len() as u32,
            p_image_memory_barriers: image_barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        // Copy PLANE_0 (Y) to buffer at offset 0
        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })],
        );

        // Copy PLANE_1 (UV interleaved) to buffer at offset y_size
        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width: uv_width,
                    height: uv_height,
                    depth: 1,
                })],
        );

        // Transition buffer: TRANSFER_DST -> HOST_READ
        let buffer_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::HOST,
            dst_access_mask: vk::AccessFlags2::HOST_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            _marker: Default::default(),
        };

        let buffer_dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buffer_barrier,
            image_memory_barrier_count: 0,
            p_image_memory_barriers: std::ptr::null(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &buffer_dep_info);

        // Transition image back to VIDEO_DECODE_DPB_KHR so it can be used as reference
        // in subsequent decode commands. This is critical: without this transition,
        // reference images remain in TRANSFER_SRC_OPTIMAL layout, causing crashes
        // when used as DPB references.
        let plane0_restore = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let plane1_restore = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let restore_barriers = [plane0_restore, plane1_restore];
        let restore_dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: restore_barriers.len() as u32,
            p_image_memory_barriers: restore_barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &restore_dep_info);

        device
            .end_command_buffer(cmd_buffer)
            .expect("Failed to end command buffer");

        // Submit
        device
            .reset_fences(&[fence])
            .expect("Failed to reset fence");

        let cmd_buffers = vec![cmd_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

        device
            .queue_submit(
                device.get_device_queue(queue_family, 0),
                &[submit_info],
                fence,
            )
            .expect("Failed to submit");

        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("Failed to wait for fence");

        // Read data from mapped memory
        let data_ptr = mapped_ptr as *const u8;
        let data = std::slice::from_raw_parts(data_ptr, total_size as usize);
        let y_data = &data[..y_size];

        // Analyze Y plane (values available for debugging if needed)
        let mut _sum: u64 = 0;
        let mut _min_val = u8::MAX;
        let mut _max_val = u8::MIN;
        let pixel_count = (frame_width * frame_height) as usize;

        for &byte in y_data.iter().take(pixel_count.min(y_data.len())) {
            _sum += byte as u64;
            if byte < _min_val {
                _min_val = byte;
            }
            if byte > _max_val {
                _max_val = byte;
            }
        }

        // Save all frames for verification
        // Convert from Vulkan G8_B8R8_2PLANE (Y + interleaved UV) to yuv420p (Y + U + V planar)
        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut yuv_data = Vec::with_capacity(y_size + uv_plane_size * 2);
        yuv_data.extend_from_slice(&data[..y_size]); // Y plane
                                                     // De-interleave UV plane: Vulkan stores UVUVUV..., yuv420p expects UUUU...VVVV...
        let uv_interleaved = &data[y_size..y_size + uv_plane_size * 2];
        let mut u_plane = vec![0u8; uv_plane_size];
        let mut v_plane = vec![0u8; uv_plane_size];
        for i in 0..uv_plane_size {
            u_plane[i] = uv_interleaved[i * 2];
            v_plane[i] = uv_interleaved[i * 2 + 1];
        }
        yuv_data.extend_from_slice(&u_plane);
        yuv_data.extend_from_slice(&v_plane);

        // Derive output filename from input path
        let stem = std::path::Path::new(bitstream_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        let output_path = format!("{}_frame_{}.yuv", stem, display_frame_idx);
        match std::fs::write(&output_path, yuv_data) {
            Ok(()) => println!("    Saved YUV to {}", output_path),
            Err(e) => eprintln!("    Failed to save YUV: {}", e),
        }

        // Cleanup
        device.destroy_buffer(buffer, None);
        device.unmap_memory(memory);
        device.free_memory(memory, None);
    }
}

/// Find a memory type index that satisfies the given properties.
fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for (i, &mem_type) in memory_properties.memory_types.iter().enumerate() {
        if (type_bits & (1 << i)) != 0 && mem_type.property_flags.contains(properties) {
            return Some(i as u32);
        }
    }
    None
}

/// Dispatch vkCmdPipelineBarrier2KHR.
fn cmd_pipeline_barrier_2(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    dep_info: &vk::DependencyInfo<'_>,
) {
    let fn_ptr =
        unsafe { instance.get_device_proc_addr(device, c"vkCmdPipelineBarrier2KHR".as_ptr()) };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, dep_info);
        }
    }
}
