//! Vulkan hardware-accelerated VP9 video decode example.
//!
//! Demonstrates a complete Vulkan VP9 video decode pipeline:
//!
//! 1. Read raw VP9 bitstream file (no container)
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
use vk_video_parser::{
    DetectedVideoFormat, VideoParser,
};
use vk_video_parser::vp9::Vp9Parser;
use vk_video_vulkan::{
    buffer::BitstreamBuffer as VkBitstreamBuffer,
    image::create_output_image_with_pnext,
    session::{CodecProfileInfo, VideoSession, VideoSessionParameters, VideoSessionParams},
    VideoCodec, VideoDeviceBuilder,
};
use vk_video_vulkan::vp9::{
    convert_vp9_picture_info, vp9_vk_constants, Vp9Decoder, Vp9PictureInfoContainer,
    StdVideoDecodeVP9PictureInfo, StdVideoVP9ColorConfig, StdVideoVP9LoopFilter,
    StdVideoVP9Segmentation, VideoDecodeVP9PictureInfoKHR, VideoDecodeVP9ProfileInfoKHR,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.vp9>", args[0]);
        std::process::exit(1);
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    println!("=== Vulkan VP9 Decode Example ===");
    println!("File: {}\n", bitstream_path);

    // Step 1: Read bitstream, expand superframes, and split into frames
    println!("--- Step 1: Read and split VP9 bitstream ---");
    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Check if this is an IVF container first
    let raw_frames = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        println!("  Detected IVF container format");
        parse_ivf_container(&data).expect("Failed to parse IVF container")
    } else {
        // Expand superframes BEFORE splitting — superframe index at end of data
        // must be detected and removed before frame splitting
        let expanded_packets = expand_superframes(&[data.to_vec()]);
        println!("  Expanded superframes: 1 packet -> {} packets", expanded_packets.len());

        // Split bitstream into individual frames using sequential parsing
        split_vp9_bitstream(&expanded_packets)
    };
    println!("  Found {} raw frame packets\n", raw_frames.len());

    if raw_frames.is_empty() {
        eprintln!("Error: No VP9 frames found in bitstream");
        std::process::exit(1);
    }

    // Step 2: Parse first frame to get format info
    println!("--- Step 2: Parse first frame for format info ---");
    let mut parser = Vp9Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeVp9,
        ))
        .expect("Failed to init VP9 parser");

    let first_frame = &raw_frames[0];
    let first_parsed = parser.parse_frame(first_frame).expect("Failed to parse first frame");

    let coded_width = first_parsed.frame_width;
    let coded_height = first_parsed.frame_height;
    let profile = first_parsed.picture_info.profile as u32;
    let bit_depth = first_parsed.color_config.bit_depth;

    println!("  Resolution: {}x{}", coded_width, coded_height);
    println!("  Profile: {}", profile);
    println!("  Bit depth: {}\n", bit_depth);

    if coded_width == 0 || coded_height == 0 {
        eprintln!("Error: Failed to parse video dimensions");
        std::process::exit(1);
    }

    // Step 3: Initialize Vulkan
    println!("--- Step 3: Vulkan initialization ---");
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
        let props = vulkan.instance.get_physical_device_properties(vulkan.physical_device);
        std::ffi::CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    println!("  GPU: {}", gpu_name);
    println!("  Video decode queue family = {}", decode_qf);
    println!("  Extensions: {}\n", vulkan.enabled_extensions.join(", "));

    // Step 4: Query video decode capabilities
    println!("--- Step 4: Query video capabilities ---");
    let luma_bit_depth = match bit_depth {
        8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };
    let chroma_bit_depth = luma_bit_depth;
    let chroma_subsampling = vk::VideoChromaSubsamplingFlagsKHR::TYPE_420;

    // Try querying capabilities for all 4 VP9 profiles (0-3)
    println!("  Testing VP9 decode profile support:");
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
                println!("    Profile {}: SUPPORTED", p);
                supported_profiles.push(p);
                if video_caps.is_none() {
                    video_caps = Some(caps);
                }
            }
            Err(e) => {
                println!("    Profile {}: NOT SUPPORTED ({})", p, e);
            }
        }
    }

    if supported_profiles.is_empty() {
        // Check if VP9 encode is available instead
        println!();
        let available_ext = unsafe {
            vulkan.instance
                .enumerate_device_extension_properties(vulkan.physical_device)
                .unwrap_or_default()
        };
        let has_encode_vp9 = available_ext.iter().any(|e| {
            unsafe {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr().cast())
                    .to_string_lossy()
                    .contains("video_encode_vp9")
            }
        });
        let has_decode_vp9_ext = vulkan.enabled_extensions.iter().any(|e| e.contains("video_decode_vp9"));

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
        println!("  Stream uses profile {}, using supported profile {} instead", profile, alt);
        alt
    };

    let video_caps = video_caps.expect("No supported VP9 profile found");

    println!(
        "  minBitstreamBufferSizeAlignment: {}",
        video_caps.min_bitstream_buffer_size_alignment
    );
    println!("  maxDPBSlots: {}", video_caps.max_dpb_slots);
    println!(
        "  pictureAccessGranularity: {}x{}\n",
        video_caps.picture_access_granularity.width,
        video_caps.picture_access_granularity.height
    );

    // Align coded extent to picture access granularity
    let align_width = video_caps.picture_access_granularity.width;
    let align_height = video_caps.picture_access_granularity.height;
    let coded_extent = vk::Extent2D {
        width: (coded_width + align_width - 1) & !(align_width - 1),
        height: (coded_height + align_height - 1) & !(align_height - 1),
    };
    println!(
        "  Coded extent aligned: {}x{} -> {}x{}\n",
        coded_width,
        coded_height,
        coded_extent.width,
        coded_extent.height
    );

    // VP9 has 8 DPB slots (VP9_NUM_REF_FRAMES)
    let max_dpb_slots = 8u32.min(video_caps.max_dpb_slots);
    let session_dpb_slots = max_dpb_slots + 1;

    // Step 5: Create video session
    println!("--- Step 5: Create video session ---");
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
    };

    let std_header_version = build_std_header_version("VK_STD_vulkan_video_codec_vp9_decode");

    let (session, session_memories) = VideoSession::create(
        &vulkan.instance,
        &vulkan.device,
        &session_params,
        &std_header_version,
    )
    .expect("Failed to create video session");

    println!("  Video session created\n");

    // VP9 doesn't use session parameters objects.
    // All per-frame info is passed in the picture info for each decode command.
    // See: Vulkan-Video-Samples C++ decoder uses VK_NULL_HANDLE for VP9.
    println!("--- Step 6: Session parameters (VP9 uses NULL) ---");
    let session_params_handle = vk::VideoSessionParametersKHR::null();
    let session_parameters: Option<VideoSessionParameters> = None;
    println!("  VP9: using VK_NULL_HANDLE for session parameters\n");

    // Step 7: Create output image
    println!("--- Step 7: Create output image ---");
    let (output_image, output_image_view, output_memory) = create_vp9_output_image(
        &vulkan.device,
        &vulkan.memory_properties,
        coded_extent.width,
        coded_extent.height,
        output_format,
        use_profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    )
    .map_err(|e| format!("Failed to create output image: {}", e))
    .expect("Failed to create output image");
    // [CHECK 5] Image view creation verification
    println!(
        "  Output image: {}x{} format={:?}\n",
        coded_extent.width, coded_extent.height, output_format
    );
    println!(
        "  [ImageView CHECK] format={:?} uses COLOR aspect (no SamplerYcbcrConversion)",
        output_format
    );
    println!(
        "  [ImageView CHECK] For 2-plane formats, COLOR aspect covers all planes per Vulkan spec\n"
    );

    // Step 8: Create DPB images for reference frame management
    println!("--- Step 8: Create DPB images ---");
    let mut dpb_images: Vec<(vk::Image, vk::ImageView, vk::DeviceMemory)> = Vec::new();

    // Create DPB images (slot 0 = output_image, slots 1..N = dpb_images)
    for slot in 1..max_dpb_slots {
        let (img, view, mem) = create_vp9_output_image(
            &vulkan.device,
            &vulkan.memory_properties,
            coded_extent.width,
            coded_extent.height,
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
    println!("  DPB slots: {}\n", max_dpb_slots);

    // Step 9: Create bitstream buffer
    println!("--- Step 9: Create bitstream buffer ---");
    let max_frame_size = raw_frames.iter().map(|f| f.len()).max().unwrap_or(0);
    let bs_size_align = video_caps.min_bitstream_buffer_size_alignment;
    // Align buffer size to minBitstreamBufferSizeAlignment
    let max_frame_size_aligned = ((max_frame_size as u64 + bs_size_align as u64 - 1) / bs_size_align as u64 * bs_size_align as u64).max(bs_size_align as u64);
    println!("  max_frame_size={} bs_size_align={} aligned_size={}", max_frame_size, bs_size_align, max_frame_size_aligned);
    let mut bs_buffer = create_vp9_bitstream_buffer(
        &vulkan.device,
        &vulkan.memory_properties,
        max_frame_size_aligned,
        use_profile,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    )
    .map_err(|e| format!("Failed to create bitstream buffer: {}", e))
    .expect("Failed to create bitstream buffer");
    println!("  Bitstream buffer: {} bytes\n", max_frame_size_aligned);

    // Step 10: Create command resources
    println!("--- Step 10: Create command resources ---");
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
    println!("  Command buffer allocated\n");

    // Step 11: Create fence
    println!("--- Step 11: Create fence ---");
    let fence = unsafe {
        vulkan
            .device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .expect("Failed to create fence")
    };
    println!("  Fence created\n");

    // Step 12: Create VP9 decoder
    println!("--- Step 12: Create VP9 decoder ---");
    let mut vp9_decoder = Vp9Decoder::new(vulkan.device.clone(), vulkan.instance.clone());
    vp9_decoder.set_session(&session);
    if let Some(params) = session_parameters {
        vp9_decoder.set_session_parameters(params);
    }
    vp9_decoder.set_max_dpb_slots(max_dpb_slots);
    println!("  VP9 decoder created\n");

    // Step 13: Decode frames
    println!("--- Step 13: Decode frames ---");
    let max_frames_to_decode = 20;

    let frames_to_decode = raw_frames.len().min(max_frames_to_decode);
    println!("  Will decode {} frames\n", frames_to_decode);

    // DPB management state
    let mut dpb_manager = Vp9DpbManager::new(max_dpb_slots);
    let mut is_first_frame = true;
    let mut frame_count: u32 = 0;

    // Reset parser for fresh parsing
    parser.reset();

    for (frame_idx, frame_data) in raw_frames.iter().enumerate().take(frames_to_decode) {
        println!("\n[Frame {}]", frame_idx);

        // Parse frame header
        let parsed = match parser.parse_frame(frame_data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Failed to parse frame header: {:?}", e);
                continue;
            }
        };

        // Compute per-frame coded extent aligned to picture access granularity
        let frame_coded_extent = vk::Extent2D {
            width: (parsed.frame_width + align_width - 1) & !(align_width - 1),
            height: (parsed.frame_height + align_height - 1) & !(align_height - 1),
        };

        // Handle show_existing_frame
        if parsed.show_existing_frame {
            let show_idx = parsed.frame_to_show_map_idx as usize;
            println!(
                "  show_existing_frame: displaying frame_to_show_map_idx={}",
                show_idx
            );

            // Find the DPB slot for this frame
            let slot = dpb_manager.get_slot_for_pic_idx(show_idx as i32);
        if let Some(slot) = slot {
            let img = dpb_image_handles[slot as usize];
                println!("  Reading back existing frame from slot {}", slot);
                readback_and_verify(
                    &vulkan.instance,
                    &vulkan.device,
                    &vulkan.memory_properties,
                    decode_qf,
                    command_pool,
                    fence,
                    img,
                    coded_extent.width,
                    coded_extent.height,
                    parsed.frame_width,
                    parsed.frame_height,
                    frame_idx,
                    bitstream_path,
                );
            } else {
                eprintln!("  Warning: frame_to_show_map_idx={} not found in DPB", show_idx);
            }
            continue;
        }

        let is_key_frame = parsed.picture_info.frame_type
            == vk_video_core::picture::Vp9FrameType::Key;

        println!(
            "  Type: {} | Size: {}x{} | Refresh flags: 0x{:02x}",
            if is_key_frame { "KEY" } else { "INTER" },
            parsed.frame_width,
            parsed.frame_height,
            parsed.picture_info.refresh_frame_flags,
        );

        // Write bitstream data to buffer
        // Zero the aligned range first to avoid feeding garbage to the decoder
        // Align to minBitstreamBufferSizeAlignment (e.g., 256 bytes) per Vulkan spec
        let bs_align = bs_size_align as u64;
        let flush_size = ((frame_data.len() as u64 + bs_align - 1) / bs_align * bs_align).max(bs_align);
        bs_buffer.zero_range(0, flush_size);
        bs_buffer.write(frame_data).expect("Failed to write bitstream");
        bs_buffer.flush_range(0, flush_size).ok();

        // Select DPB slot for this frame
        let output_slot;
        if is_key_frame || is_first_frame {
            // Key frame or first frame: reset DPB and use slot 0
            if is_key_frame {
                dpb_manager.invalidate_all();
                vp9_decoder.reset_dpb();
            }
            output_slot = 0;
        } else {
            // Inter frame: find or recycle a slot
            output_slot = dpb_manager.find_or_recycle_slot().unwrap_or(0);
        }

        let output_slot_usize = output_slot as usize;
        let output_view = dpb_views[output_slot_usize];
        let output_img = dpb_image_handles[output_slot_usize];

        // Compute reference name slot indices
        let reference_name_slot_indices =
            vp9_decoder.compute_reference_name_slot_indices(is_key_frame);

        println!(
            "  Output slot: {} | DPB Ref slots: [{}, {}, {}]",
            output_slot,
            reference_name_slot_indices[0],
            reference_name_slot_indices[1],
            reference_name_slot_indices[2],
        );

        // Build DPB reference picture resources
        let (dpb_setup_picture, dpb_ref_pictures, dpb_ref_slot_indices, vulkan_ref_name_slot_indices) = build_dpb_picture_resources(
            &dpb_manager,
            &dpb_views,
            frame_coded_extent,
            output_slot,
            is_key_frame,
            &reference_name_slot_indices,
        );

        // DEBUG: print values being passed to Vulkan
        let actual_bs_size = frame_data.len() as u64;
        // Align srcBufferRange to minBitstreamBufferSizeAlignment per Vulkan spec
        let bs_range_aligned = ((actual_bs_size + bs_align - 1) / bs_align * bs_align).max(bs_align);
        println!(
            "  Bitstream: offset=0 range={} (frame_data.len={} aligned={})",
            bs_range_aligned,
            frame_data.len(),
            bs_range_aligned,
        );
        // [CHECK 2] Bitstream data integrity - hex dump of first 32 bytes
        let first_bytes: Vec<String> = frame_data.iter().take(32).map(|b| format!("{:02x}", b)).collect();
        println!("  Bitstream first 32 bytes: {}", first_bytes.join(" "));
        // CRC32-like checksum of entire frame for integrity check
        let checksum: u32 = frame_data.iter().fold(0u32, |acc, &b| {
            acc.wrapping_add(b as u32).wrapping_mul(31)
        });
        println!("  Bitstream checksum (simple): 0x{:08x} (len={})", checksum, frame_data.len());
        // [CHECK 1] compressed_header_size - critical for tiles_offset calculation
        println!(
            "  compressed_header_size: {} (0x{:04x})",
            parsed.compressed_header_size,
            parsed.compressed_header_size,
        );
        println!(
            "  Header offsets: uncompressed={} compressed={} tiles={}",
            parsed.uncompressed_header_offset,
            parsed.compressed_header_offset,
            parsed.tiles_offset,
        );
        // Verify: tiles_offset should equal compressed_header_offset + compressed_header_size
        let expected_tiles = parsed.compressed_header_offset + parsed.compressed_header_size;
        if parsed.tiles_offset != expected_tiles {
            println!(
                "  *** WARNING: tiles_offset mismatch! expected={} actual={}",
                expected_tiles, parsed.tiles_offset
            );
        }
        // Verify tiles_offset is within frame data bounds
        if parsed.tiles_offset as usize > frame_data.len() {
            println!(
                "  *** WARNING: tiles_offset ({}) exceeds frame_data.len() ({})",
                parsed.tiles_offset, frame_data.len()
            );
        } else {
            println!(
                "  tiles_offset ({}) is within bounds (frame_data.len()={})",
                parsed.tiles_offset, frame_data.len()
            );
        }
        // [CHECK 1b] Verify compressed_header_size against raw bitstream bytes
        // compressed_header_size is a 16-bit value at the end of the uncompressed header.
        // The compressed_header_offset includes these 16 bits, so the raw bytes are
        // approximately at (compressed_header_offset - 2) to compressed_header_offset.
        let chs_start = if parsed.compressed_header_offset >= 2 {
            parsed.compressed_header_offset as usize - 2
        } else {
            0
        };
        let chs_end = (parsed.compressed_header_offset as usize).min(frame_data.len());
        if chs_end > chs_start {
            let raw_bytes: Vec<String> = frame_data[chs_start..chs_end]
                .iter().map(|b| format!("{:02x}", b)).collect();
            let raw_val = if chs_end - chs_start >= 2 {
                u16::from_le_bytes([frame_data[chs_end - 2], frame_data[chs_end - 1]]) as u32
            } else {
                0
            };
            println!(
                "  [CHS VERIFY] raw bytes at [{}..{}]: {} (LE u16={})",
                chs_start, chs_end, raw_bytes.join(" "), raw_val
            );
            if raw_val != parsed.compressed_header_size {
                println!(
                    "  *** WARNING: raw LE u16 ({}) != parsed compressed_header_size ({}) - bit alignment may differ",
                    raw_val, parsed.compressed_header_size
                );
            }
        }
        println!(
            "  Coded extent: {}x{}",
            frame_coded_extent.width,
            frame_coded_extent.height,
        );
        println!(
            "  p_std_picture_info: frame_type={} profile={} refresh_flags=0x{:02x}",
            if is_key_frame { "KEY" } else { "INTER" },
            parsed.picture_info.profile as u32,
            parsed.picture_info.refresh_frame_flags,
        );
        // Print additional picture info fields
        println!(
            "  Picture info detail: frame_ctx_idx={} reset_ctx={} sign_bias_mask=0x{:02x} interp_filter={}",
            parsed.picture_info.frame_context_idx,
            parsed.picture_info.flags.reset_frame_context,
            parsed.picture_info.ref_frame_sign_bias_mask,
            match parsed.picture_info.interpolation_filter {
                vk_video_core::picture::Vp9InterpolationFilter::EightTap => "EIGHTTAP",
                vk_video_core::picture::Vp9InterpolationFilter::EightTapSmooth => "EIGHTTAP_SMOOTH",
                vk_video_core::picture::Vp9InterpolationFilter::EightTapSharp => "EIGHTTAP_SHARP",
                vk_video_core::picture::Vp9InterpolationFilter::Bilinear => "BILINEAR",
                vk_video_core::picture::Vp9InterpolationFilter::Switchable => "SWITCHABLE",
            },
        );
        println!(
            "  Picture info quant: base_q={} dq_ydc={} dq_uvdc={} dq_uvac={}",
            parsed.picture_info.base_q_idx,
            parsed.picture_info.delta_q_y_dc,
            parsed.picture_info.delta_q_uv_dc,
            parsed.picture_info.delta_q_uv_ac,
        );
        println!(
            "  Picture info tiles: cols_log2={} rows_log2={}",
            parsed.picture_info.tile_cols_log2,
            parsed.picture_info.tile_rows_log2,
        );
        println!(
            "  Picture info flags: error_resilient={} intra_only={} high_prec_mv={} refresh_ctx={} parallel={} seg={} show={} prev_mv={}",
            parsed.picture_info.flags.error_resilient_mode,
            parsed.picture_info.flags.intra_only,
            parsed.picture_info.flags.allow_high_precision_mv,
            parsed.picture_info.flags.refresh_frame_context,
            parsed.picture_info.flags.frame_parallel_decoding_mode,
            parsed.picture_info.flags.segmentation_enabled,
            parsed.picture_info.flags.show_frame,
            parsed.picture_info.flags.use_prev_frame_mvs,
        );
        // [CHECK 3 & 4] DPB setup + VkVideoPictureResourceInfoKHR verification
        println!(
            "  DPB: setup_slot={} ref_count={}",
            output_slot,
            dpb_ref_pictures.len(),
        );
        for (i, &slot) in dpb_ref_slot_indices.iter().enumerate() {
            println!("    ref[{}] -> slot {}", i, slot);
        }
        // [CHECK 4] For key frame, verify DPB is clean
        if is_key_frame {
            println!("  [DPB CHECK] Key frame: dpb_ref_pictures.len()={} (expected 0)", dpb_ref_pictures.len());
            println!("  [DPB CHECK] Key frame: dpb_ref_slot_indices.len()={} (expected 0)", dpb_ref_slot_indices.len());
            println!("  [DPB CHECK] Key frame: dpb_setup_picture.is_some()={} (expected true)", dpb_setup_picture.is_some());
            if dpb_ref_pictures.is_empty() && dpb_ref_slot_indices.is_empty() && dpb_setup_picture.is_some() {
                println!("  [DPB CHECK] Key frame DPB setup: OK");
            } else {
                println!("  [DPB CHECK] Key frame DPB setup: UNEXPECTED!");
            }
        }
        // [CHECK 3] Print VkVideoPictureResourceInfoKHR fields for setup picture
        if let Some(ref setup) = dpb_setup_picture {
            println!(
                "  [PictureResource] setup: coded_extent={}x{} coded_offset=({},{}) base_array_layer={} image_view_binding={:?}",
                setup.coded_extent.width,
                setup.coded_extent.height,
                setup.coded_offset.x,
                setup.coded_offset.y,
                setup.base_array_layer,
                setup.image_view_binding,
            );
        }
        for (i, ref pr) in dpb_ref_pictures.iter().enumerate() {
            println!(
                "  [PictureResource] ref[{}]: coded_extent={}x{} coded_offset=({},{}) base_array_layer={} image_view_binding={:?}",
                i,
                pr.coded_extent.width,
                pr.coded_extent.height,
                pr.coded_offset.x,
                pr.coded_offset.y,
                pr.base_array_layer,
                pr.image_view_binding,
            );
        }

        // Convert parsed frame data to Vulkan picture info container
        // Allocate on heap to ensure lifetime through command execution
        let picture_info_container = Box::new({
            let mut c = convert_vp9_picture_info(
                &parsed.picture_info,
                &parsed.color_config,
                &parsed.loop_filter,
                &parsed.segmentation,
            );
            c.init_pointers();
            c
        });

        // DEBUG: dump struct layouts for frame 0
        if frame_idx == 0 {
            dump_vp9_struct_debug(
                &picture_info_container,
                &reference_name_slot_indices,
                parsed.uncompressed_header_offset,
                parsed.compressed_header_offset,
                parsed.tiles_offset,
            );
        }

        // Build VP9 decode picture info on heap alongside the container.
        // CRITICAL: Both must stay alive until after fence wait, otherwise the
        // command buffer holds dangling pointers to freed stack memory.
        let vp9_decode_info = Box::new(VideoDecodeVP9PictureInfoKHR::new(
            picture_info_container.std_picture_info(),
            vulkan_ref_name_slot_indices,
            parsed.uncompressed_header_offset,
            parsed.compressed_header_offset,
            parsed.tiles_offset,
        ));

        // Record decode command
        let result = vp9_decoder.record_decode_command(
            command_buffer,
            session.handle(),
            session_params_handle,
            bs_buffer.buffer(),
            0,
            bs_range_aligned,
            output_view,
            output_img,
            frame_coded_extent,
            dpb_setup_picture,
            &dpb_ref_pictures,
            &dpb_ref_slot_indices,
            &picture_info_container,
            &vp9_decode_info,
            is_first_frame,
            output_slot as i32,
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
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_bufs);

            vulkan
                .device
                .queue_submit(
                    vulkan.video_decode_queue(0),
                    &[submit_info],
                    fence,
                )
                .expect("Failed to submit");

            vulkan
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("Failed to wait for fence");
        }

        // Update frame pointers based on refresh_frame_flags
        let current_pic_idx = output_slot as i32;
        vp9_decoder.update_frame_pointers(
            parsed.picture_info.refresh_frame_flags,
            current_pic_idx,
        );

        // Register this frame in DPB manager
        dpb_manager.register_frame(output_slot as u32, frame_count);

        // Update DPB slot layout
        dpb_manager.set_slot_layout(
            output_slot,
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        );

        // Readback and verify decoded pixels
        println!("  Reading back decoded pixels...");
        readback_and_verify(
            &vulkan.instance,
            &vulkan.device,
            &vulkan.memory_properties,
            decode_qf,
            command_pool,
            fence,
            output_img,
            coded_extent.width,
            coded_extent.height,
            parsed.frame_width,
            parsed.frame_height,
            frame_idx,
            bitstream_path,
        );

        // Restore DPB layout after readback
        dpb_manager.set_slot_layout(
            output_slot,
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        );

        println!(
            "  DPB state: {} valid frames",
            dpb_manager.get_valid_count()
        );

        frame_count += 1;
    }

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", frame_count);
    println!("DPB valid frames: {}", dpb_manager.get_valid_count());

    // Cleanup DPB images
    for (img, view, mem) in dpb_images {
        unsafe {
            vulkan.device.destroy_image_view(view, None);
            vulkan.device.destroy_image(img, None);
            vulkan.device.free_memory(mem, None);
        }
    }

    // Cleanup
    println!("\n--- Cleanup ---");
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
// Debug dump for VP9 struct layouts
// ============================================================================

/// Dump struct sizes, field offsets, raw bytes, and key field values for VP9 Vulkan structs.
/// Called once for frame 0 to diagnose struct layout issues and compare with C++ reference.
fn dump_vp9_struct_debug(
    picture_info_container: &Vp9PictureInfoContainer,
    reference_name_slot_indices: &[i32; 3],
    uncompressed_header_offset: u32,
    compressed_header_offset: u32,
    tiles_offset: u32,
) {
    use std::fmt::Write;

    let std_info = &picture_info_container.std_picture_info;

    // =========================================================================
    // Print key field values for comparison with C++ reference
    // =========================================================================
    eprintln!("\n========== VP9 Picture Info Key Fields (Rust) ==========");
    eprintln!("  profile: {} (StdVideoVP9Profile={:?})",
        std_info.profile as u32, std_info.profile);
    eprintln!("  frame_type: {} ({})",
        std_info.frame_type as u32,
        if std_info.frame_type as u32 == 0 { "KEY" } else { "INTER" });
    eprintln!("  refresh_frame_flags: 0x{:02x} ({})",
        std_info.refresh_frame_flags, std_info.refresh_frame_flags);
    eprintln!("  base_q_idx: {} (0x{:02x})", std_info.base_q_idx, std_info.base_q_idx);
    eprintln!("  delta_q_y_dc: {}", std_info.delta_q_y_dc);
    eprintln!("  delta_q_uv_dc: {}", std_info.delta_q_uv_dc);
    eprintln!("  delta_q_uv_ac: {}", std_info.delta_q_uv_ac);
    eprintln!("  tile_cols_log2: {}", std_info.tile_cols_log2);
    eprintln!("  tile_rows_log2: {}", std_info.tile_rows_log2);
    eprintln!("  interpolation_filter: {} ({})",
        std_info.interpolation_filter as u32,
        match std_info.interpolation_filter {
            vk_video_vulkan::vp9::StdVideoVP9InterpolationFilter::EightTap => "EIGHTTAP",
            vk_video_vulkan::vp9::StdVideoVP9InterpolationFilter::EightTapSmooth => "EIGHTTAP_SMOOTH",
            vk_video_vulkan::vp9::StdVideoVP9InterpolationFilter::EightTapSharp => "EIGHTTAP_SHARP",
            vk_video_vulkan::vp9::StdVideoVP9InterpolationFilter::Bilinear => "BILINEAR",
            vk_video_vulkan::vp9::StdVideoVP9InterpolationFilter::Switchable => "SWITCHABLE",
        });
    eprintln!("  frame_context_idx: {}", std_info.frame_context_idx);
    eprintln!("  reset_frame_context: {}", std_info.reset_frame_context);
    eprintln!("  ref_frame_sign_bias_mask: 0x{:02x}", std_info.ref_frame_sign_bias_mask);
    eprintln!("  flags: 0x{:08x}", std_info.flags.bits);
    eprintln!("  ref_name_slot_indices: [{}, {}, {}]",
        reference_name_slot_indices[0],
        reference_name_slot_indices[1],
        reference_name_slot_indices[2]);
    eprintln!("  uncompressed_header_offset: {}", uncompressed_header_offset);
    eprintln!("  compressed_header_offset: {}", compressed_header_offset);
    eprintln!("  tiles_offset: {}", tiles_offset);
    eprintln!("  color_config: bit_depth={} subsampling_x={} subsampling_y={} color_space={}",
        picture_info_container.color_config.bit_depth,
        picture_info_container.color_config.subsampling_x,
        picture_info_container.color_config.subsampling_y,
        picture_info_container.color_config.color_space as u32);
    eprintln!("  loop_filter: level={} sharpness={} update_ref_delta={}",
        picture_info_container.loop_filter.loop_filter_level,
        picture_info_container.loop_filter.loop_filter_sharpness,
        picture_info_container.loop_filter.update_ref_delta);
    eprintln!("  loop_filter_ref_deltas: [{}, {}, {}, {}]",
        picture_info_container.loop_filter.loop_filter_ref_deltas[0],
        picture_info_container.loop_filter.loop_filter_ref_deltas[1],
        picture_info_container.loop_filter.loop_filter_ref_deltas[2],
        picture_info_container.loop_filter.loop_filter_ref_deltas[3]);
    eprintln!("  loop_filter_mode_deltas: [{}, {}]",
        picture_info_container.loop_filter.loop_filter_mode_deltas[0],
        picture_info_container.loop_filter.loop_filter_mode_deltas[1]);
    eprintln!("==========================================\n");

    // =========================================================================
    // Dump raw binary of StdVideoDecodeVP9PictureInfo to file
    // =========================================================================
    let std_info_bytes = unsafe {
        std::slice::from_raw_parts(
            std_info as *const _ as *const u8,
            std::mem::size_of::<StdVideoDecodeVP9PictureInfo>(),
        )
    };
    if let Err(e) = std::fs::write("rust_std_picture_info.bin", std_info_bytes) {
        eprintln!("Failed to write rust_std_picture_info.bin: {}", e);
    } else {
        eprintln!("Wrote rust_std_picture_info.bin ({} bytes)", std_info_bytes.len());
    }

    // =========================================================================
    // Create and dump VideoDecodeVP9PictureInfoKHR
    // =========================================================================
    let vp9_khr = VideoDecodeVP9PictureInfoKHR::new(
        std_info,
        *reference_name_slot_indices,
        uncompressed_header_offset,
        compressed_header_offset,
        tiles_offset,
    );

    let khr_bytes = unsafe {
        std::slice::from_raw_parts(
            &vp9_khr as *const _ as *const u8,
            std::mem::size_of::<VideoDecodeVP9PictureInfoKHR>(),
        )
    };
    if let Err(e) = std::fs::write("rust_vp9_picture_info_khr.bin", khr_bytes) {
        eprintln!("Failed to write rust_vp9_picture_info_khr.bin: {}", e);
    } else {
        eprintln!("Wrote rust_vp9_picture_info_khr.bin ({} bytes)", khr_bytes.len());
    }

    // =========================================================================
    // Full struct layout dump
    // =========================================================================
    let mut dump = String::new();

    // StdVideoDecodeVP9PictureInfo
    let expected_size = 56usize;
    let actual_size = std::mem::size_of::<StdVideoDecodeVP9PictureInfo>();
    writeln!(dump, "=== StdVideoDecodeVP9PictureInfo ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of flags: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, flags)).unwrap();
    writeln!(dump, "  offset_of profile: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, profile)).unwrap();
    writeln!(dump, "  offset_of frame_type: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, frame_type)).unwrap();
    writeln!(dump, "  offset_of frame_context_idx: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, frame_context_idx)).unwrap();
    writeln!(dump, "  offset_of reset_frame_context: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, reset_frame_context)).unwrap();
    writeln!(dump, "  offset_of refresh_frame_flags: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, refresh_frame_flags)).unwrap();
    writeln!(dump, "  offset_of ref_frame_sign_bias_mask: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, ref_frame_sign_bias_mask)).unwrap();
    writeln!(dump, "  offset_of interpolation_filter: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, interpolation_filter)).unwrap();
    writeln!(dump, "  offset_of base_q_idx: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, base_q_idx)).unwrap();
    writeln!(dump, "  offset_of delta_q_y_dc: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, delta_q_y_dc)).unwrap();
    writeln!(dump, "  offset_of delta_q_uv_dc: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, delta_q_uv_dc)).unwrap();
    writeln!(dump, "  offset_of delta_q_uv_ac: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, delta_q_uv_ac)).unwrap();
    writeln!(dump, "  offset_of tile_cols_log2: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, tile_cols_log2)).unwrap();
    writeln!(dump, "  offset_of tile_rows_log2: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, tile_rows_log2)).unwrap();
    writeln!(dump, "  offset_of reserved1: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, reserved1)).unwrap();
    writeln!(dump, "  offset_of p_color_config: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, p_color_config)).unwrap();
    writeln!(dump, "  offset_of p_loop_filter: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, p_loop_filter)).unwrap();
    writeln!(dump, "  offset_of p_segmentation: {}", std::mem::offset_of!(StdVideoDecodeVP9PictureInfo, p_segmentation)).unwrap();

    // StdVideoVP9ColorConfig
    let expected_size = 12usize;
    let actual_size = std::mem::size_of::<StdVideoVP9ColorConfig>();
    writeln!(dump, "\n=== StdVideoVP9ColorConfig ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of flags: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, flags)).unwrap();
    writeln!(dump, "  offset_of bit_depth: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, bit_depth)).unwrap();
    writeln!(dump, "  offset_of subsampling_x: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, subsampling_x)).unwrap();
    writeln!(dump, "  offset_of subsampling_y: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, subsampling_y)).unwrap();
    writeln!(dump, "  offset_of reserved1: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, reserved1)).unwrap();
    writeln!(dump, "  offset_of color_space: {}", std::mem::offset_of!(StdVideoVP9ColorConfig, color_space)).unwrap();

    // StdVideoVP9LoopFilter - C compiler gives 16 bytes (with natural alignment)
    let expected_size = 16usize;
    let actual_size = std::mem::size_of::<StdVideoVP9LoopFilter>();
    writeln!(dump, "\n=== StdVideoVP9LoopFilter ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of flags: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, flags)).unwrap();
    writeln!(dump, "  offset_of loop_filter_level: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, loop_filter_level)).unwrap();
    writeln!(dump, "  offset_of loop_filter_sharpness: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, loop_filter_sharpness)).unwrap();
    writeln!(dump, "  offset_of update_ref_delta: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, update_ref_delta)).unwrap();
    writeln!(dump, "  offset_of loop_filter_ref_deltas: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, loop_filter_ref_deltas)).unwrap();
    writeln!(dump, "  offset_of update_mode_delta: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, update_mode_delta)).unwrap();
    writeln!(dump, "  offset_of loop_filter_mode_deltas: {}", std::mem::offset_of!(StdVideoVP9LoopFilter, loop_filter_mode_deltas)).unwrap();

    // StdVideoVP9Segmentation - C compiler gives 88 bytes (with natural alignment)
    let expected_size = 88usize;
    let actual_size = std::mem::size_of::<StdVideoVP9Segmentation>();
    writeln!(dump, "\n=== StdVideoVP9Segmentation ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of flags: {}", std::mem::offset_of!(StdVideoVP9Segmentation, flags)).unwrap();
    writeln!(dump, "  offset_of segmentation_tree_probs: {}", std::mem::offset_of!(StdVideoVP9Segmentation, segmentation_tree_probs)).unwrap();
    writeln!(dump, "  offset_of segmentation_pred_prob: {}", std::mem::offset_of!(StdVideoVP9Segmentation, segmentation_pred_prob)).unwrap();
    writeln!(dump, "  offset_of feature_enabled: {}", std::mem::offset_of!(StdVideoVP9Segmentation, feature_enabled)).unwrap();
    writeln!(dump, "  offset_of feature_data: {}", std::mem::offset_of!(StdVideoVP9Segmentation, feature_data)).unwrap();

    // Vp9PictureInfoContainer - 56+12+16+88=172, but with alignment padding = 176
    let expected_size = 176usize;
    let actual_size = std::mem::size_of::<Vp9PictureInfoContainer>();
    writeln!(dump, "\n=== Vp9PictureInfoContainer ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of std_picture_info: {}", std::mem::offset_of!(Vp9PictureInfoContainer, std_picture_info)).unwrap();
    writeln!(dump, "  offset_of color_config: {}", std::mem::offset_of!(Vp9PictureInfoContainer, color_config)).unwrap();
    writeln!(dump, "  offset_of loop_filter: {}", std::mem::offset_of!(Vp9PictureInfoContainer, loop_filter)).unwrap();
    writeln!(dump, "  offset_of segmentation: {}", std::mem::offset_of!(Vp9PictureInfoContainer, segmentation)).unwrap();

    // VideoDecodeVP9PictureInfoKHR - C compiler gives 48 bytes (with pointer alignment)
    let expected_size = 48usize;
    let actual_size = std::mem::size_of::<VideoDecodeVP9PictureInfoKHR>();
    writeln!(dump, "\n=== VideoDecodeVP9PictureInfoKHR ===").unwrap();
    writeln!(dump, "  size: expected={} actual={} {}",
        expected_size, actual_size,
        if expected_size == actual_size { "OK" } else { "MISMATCH!" }).unwrap();
    writeln!(dump, "  offset_of s_type: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, s_type)).unwrap();
    writeln!(dump, "  offset_of p_next: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, p_next)).unwrap();
    writeln!(dump, "  offset_of p_std_picture_info: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, p_std_picture_info)).unwrap();
    writeln!(dump, "  offset_of reference_name_slot_indices: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, reference_name_slot_indices)).unwrap();
    writeln!(dump, "  offset_of uncompressed_header_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, uncompressed_header_offset)).unwrap();
    writeln!(dump, "  offset_of compressed_header_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, compressed_header_offset)).unwrap();
    writeln!(dump, "  offset_of tiles_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, tiles_offset)).unwrap();

    // VkVideoDecodeInfoKHR
    writeln!(dump, "\n=== VkVideoDecodeInfoKHR ===").unwrap();
    writeln!(dump, "  size: {}", std::mem::size_of::<vk::VideoDecodeInfoKHR>()).unwrap();
    writeln!(dump, "  offset_of s_type: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, s_type)).unwrap();
    writeln!(dump, "  offset_of p_next: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_next)).unwrap();
    writeln!(dump, "  offset_of flags: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, flags)).unwrap();
    writeln!(dump, "  offset_of src_buffer: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer)).unwrap();
    writeln!(dump, "  offset_of src_buffer_offset: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer_offset)).unwrap();
    writeln!(dump, "  offset_of src_buffer_range: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer_range)).unwrap();
    writeln!(dump, "  offset_of dst_picture_resource: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, dst_picture_resource)).unwrap();
    writeln!(dump, "  offset_of p_setup_reference_slot: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_setup_reference_slot)).unwrap();
    writeln!(dump, "  offset_of reference_slot_count: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, reference_slot_count)).unwrap();
    writeln!(dump, "  offset_of p_reference_slots: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_reference_slots)).unwrap();

    // Raw hex dump of Vp9PictureInfoContainer
    let container_bytes = unsafe {
        std::slice::from_raw_parts(
            picture_info_container as *const _ as *const u8,
            std::mem::size_of::<Vp9PictureInfoContainer>(),
        )
    };

    writeln!(dump, "\n=== Vp9PictureInfoContainer raw hex dump ===").unwrap();
    for (i, chunk) in container_bytes.chunks(16).enumerate() {
        let offset_str = format!("{:04x}", i * 16);
        let hex_str: String = chunk.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
        let ascii_str: String = chunk.iter().map(|b| if *b >= 32 && *b < 127 { *b as char } else { '.' }).collect();
        writeln!(dump, "  {}  {:<48}  |{}|", offset_str, hex_str, ascii_str).unwrap();
    }

    // Print to stderr
    eprintln!("{}", dump);

    // Write to file
    if let Err(e) = std::fs::write("vp9_struct_dump.txt", &dump) {
        eprintln!("Failed to write vp9_struct_dump.txt: {}", e);
    } else {
        eprintln!("Wrote vp9_struct_dump.txt");
    }
}

// ============================================================================
// IVF container parsing
// ============================================================================

/// Parse an IVF container file and extract raw VP9 frame data.
///
/// IVF (On2 IVF) is a simple container format used for VP8/VP9 video.
/// File layout:
///   - 32-byte file header
///   - Repeated frame packets (4-byte size + 8-byte timestamp + data)
///
/// Returns a vector of raw VP9 frame data bytes.
fn parse_ivf_container(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    if data.len() < 32 {
        return Err("File too small for IVF header".to_string());
    }

    // Check IVF magic "DKIF"
    if data[0..4] != *b"DKIF" {
        return Err("Invalid IVF magic".to_string());
    }

    // Parse 32-byte IVF header
    let version = u16::from_le_bytes([data[4], data[5]]);
    let header_stride = u16::from_le_bytes([data[6], data[7]]);
    let fourcc = String::from_utf8_lossy(&data[8..12]).to_string();
    let width = u16::from_le_bytes([data[12], data[13]]);
    let height = u16::from_le_bytes([data[14], data[15]]);
    let rate_num = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let rate_den = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    let frame_count = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let time_base = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);

    println!("    IVF: version={} stride={} codec={} {}x{} rate={}/{} frames={} time_base={}",
        version, header_stride, fourcc, width, height, rate_num, rate_den, frame_count, time_base);

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
        let timestamp = u64::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);

        offset += 12;

        // Validate packet size
        if packet_size == 0 || offset + packet_size > data.len() {
            eprintln!("    Warning: invalid packet size {} at offset {}", packet_size, offset - 12);
            break;
        }

        let frame_data = data[offset..offset + packet_size].to_vec();
        println!("    Frame {}: size={} timestamp={}", frames.len(), packet_size, timestamp);
        frames.push(frame_data);
        offset += packet_size;
    }

    if frames.is_empty() {
        return Err("No frames found in IVF container".to_string());
    }

    println!("    Parsed {} frames from IVF container", frames.len());
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

/// Expand superframes into individual frames.
///
/// A superframe contains multiple VP9 frames concatenated together,
/// with a superframe index at the end that specifies the size of each
/// constituent frame. This function detects the superframe index at the
/// end of the data and splits the superframe into its component frames.
fn expand_superframes(data: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut expanded = Vec::new();

    for frame in data {
        let data_len = frame.len();
        if data_len < 2 {
            expanded.push(frame.clone());
            continue;
        }

        // Check for superframe index at the end of the data
        let final_byte = frame[data_len - 1];
        if (final_byte & 0xE0) != 0xC0 {
            // Not a superframe
            expanded.push(frame.clone());
            continue;
        }

        let num_frames = (final_byte & 0x07) as usize + 1;
        if num_frames <= 1 {
            expanded.push(frame.clone());
            continue;
        }

        let mag = (((final_byte >> 3) & 0x03) as usize) + 1;
        let index_size = 2 + mag * num_frames;

        if data_len < index_size {
            expanded.push(frame.clone());
            continue;
        }

        let index_start = data_len - index_size;
        if frame[index_start] != final_byte {
            expanded.push(frame.clone());
            continue;
        }

        // Parse frame sizes from the superframe index
        let frame_data_size = data_len - index_size;
        let mut offset = 0;
        let mut x = index_start + 1;
        for _ in 0..num_frames {
            let mut this_sz: usize = 0;
            for j in 0..mag {
                this_sz |= (frame[x + j] as usize) << (j * 8);
            }
            x += mag;

            if offset + this_sz <= frame_data_size {
                expanded
                    .push(frame[offset..offset + this_sz].to_vec());
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
    fn find_or_recycle_slot(&mut self) -> Option<u32> {
        // First try to find an empty slot
        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.is_valid {
                return Some(i as u32);
            }
        }
        // Recycle the oldest valid slot
        let mut oldest_idx = None;
        let mut oldest_count = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid && entry.frame_count < oldest_count {
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
    fn get_valid_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_valid).count()
    }
}

// ============================================================================
// DPB picture resource building
// ============================================================================

/// Build DPB picture resources for VP9 decode.
///
/// Returns (setup_picture, ref_pictures, ref_slot_indices, vulkan_ref_name_slot_indices) where:
/// - setup_picture: the current frame's output slot
/// - ref_pictures: reference picture slots (only those referenced by reference_name_slot_indices)
/// - ref_slot_indices: DPB slot indices corresponding to each reference picture
/// - vulkan_ref_name_slot_indices: indices into p_reference_slots for Vulkan (not DPB slot indices)
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
    [i32; 3],
) {
    let mut ref_pictures = Vec::new();
    let mut ref_slot_indices = Vec::new();
    // Vulkan expects indices into p_reference_slots, not DPB slot indices
    let mut vulkan_ref_name_slot_indices: [i32; 3] = [-1, -1, -1];

    if !is_key_frame {
        // Build reference picture resources ONLY for the 3 VP9 primary reference
        // frame names (LAST, GOLDEN, ALTREF) as specified by
        // reference_name_slot_indices. Track mapping from VP9 ref name index
        // to p_reference_slots index.
        let mut p_ref_slot_idx: i32 = 0;
        for (ref_name_idx, &slot_idx) in reference_name_slot_indices.iter().enumerate() {
            if slot_idx < 0 {
                continue; // Reference frame name not assigned
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
            // Map VP9 ref name index -> p_reference_slots index
            vulkan_ref_name_slot_indices[ref_name_idx] = p_ref_slot_idx;
            p_ref_slot_idx += 1;
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

    (Some(setup_picture), ref_pictures, ref_slot_indices, vulkan_ref_name_slot_indices)
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
        s_type: vk::StructureType::from_raw(
            vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR,
        ),
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
    frame_idx: usize,
    bitstream_path: &str,
) {
    let y_size = (width * height) as usize;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
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

        device
            .end_command_buffer(cmd_buffer)
            .expect("Failed to end command buffer");

        // Submit
        device
            .reset_fences(&[fence])
            .expect("Failed to reset fence");

        let cmd_buffers = vec![cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers);

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

        // Analyze Y plane
        let mut sum: u64 = 0;
        let mut min_val = u8::MAX;
        let mut max_val = u8::MIN;
        let pixel_count = (frame_width * frame_height) as usize;

        for i in 0..pixel_count.min(y_data.len()) {
            let val = y_data[i] as u64;
            sum += val;
            if y_data[i] < min_val {
                min_val = y_data[i];
            }
            if y_data[i] > max_val {
                max_val = y_data[i];
            }
        }

        let avg = if pixel_count > 0 { sum as f64 / pixel_count as f64 } else { 0.0 };

        println!(
            "    Y plane: avg={:.1} min={} max={} ({}x{})",
            avg, min_val, max_val, frame_width, frame_height
        );

        // Save full YUV data for frame 0
        if frame_idx == 0 {
            // Derive output filename from input path
            let stem = std::path::Path::new(bitstream_path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            let output_path = format!("{}_frame_0.yuv", stem);
            match std::fs::write(&output_path, data) {
                Ok(()) => println!("    Saved YUV to {}", output_path),
                Err(e) => eprintln!("    Failed to save YUV: {}", e),
            }
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
        if (type_bits & (1 << i)) != 0
            && mem_type.property_flags.contains(properties)
        {
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
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(
            device,
            b"vkCmdPipelineBarrier2KHR\0".as_ptr().cast(),
        )
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, dep_info);
        }
    }
}
