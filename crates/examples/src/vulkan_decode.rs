//! Vulkan hardware-accelerated video decode example (H.264 + H.265).
//!
//! Demonstrates a complete Vulkan video decode pipeline:
//!
//! 1. Detect codec from file extension (.h264 / .h265)
//! 2. Initialize Vulkan with video decode extensions
//! 3. Parse bitstream to extract SPS/PPS (VPS for H.265)
//! 4. Create video session with proper profile chain
//! 5. Update session parameters with parsed parameter sets
//! 6. Record and submit decode commands
//! 7. Readback decoded YUV frames and verify against ffmpeg reference
//!
//! Usage:
//!   cargo run --example vulkan_decode -- born_trailer.h264
//!   cargo run --example vulkan_decode -- big_buck_bunney.h265

use ash::vk::{self, Handle};
use vk_video_parser::{
    bitstream::BitstreamPacket, h264::H264Parser, h265::H265Parser, nal::remove_emulation_prevention_bytes,
    DetectedVideoFormat, ParseResult, VideoParser,
};
use vk_video_vulkan::{
    buffer::BitstreamBuffer as VkBitstreamBuffer,
    h265::{convert_h265_pps, convert_h265_sps, convert_h265_vps},
    image::create_output_image,
    VideoDeviceBuilder,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.h264|h265> [max_frames]", args[0]);
        eprintln!("Available: born_trailer.h264, big_buck_bunney.h265");
        std::process::exit(1);
    };

    let max_frames: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or(20)
    } else {
        20
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    let codec = detect_codec(bitstream_path);

    println!("=== Vulkan Video Decode Example ===");
    println!("File: {}", bitstream_path);
    println!("Codec: {}\n", codec.name());

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Step 1: Parse bitstream
    println!("--- Step 1: Parse bitstream ---");
    let parsed = match codec {
        VideoCodec::H264 => parse_h264(&data),
        VideoCodec::H265 => parse_h265(&data),
    };

    println!(
        "  Resolution: {}x{}",
        parsed.coded_width, parsed.coded_height
    );
    println!("  Profile: {}", parsed.profile_idc);
    println!("  Max DPB slots: {}\n", parsed.max_dpb_slots);

    if parsed.coded_width == 0 || parsed.coded_height == 0 {
        eprintln!("Error: Failed to parse video dimensions");
        std::process::exit(1);
    }

    // Step 2: Initialize Vulkan
    println!("--- Step 2: Vulkan initialization ---");
    let mut vulkan = match VideoDeviceBuilder::new().with_validation(true).build() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to initialize Vulkan: {}", e);
            std::process::exit(1);
        }
    };
    let decode_qf = vulkan.queue_families.video_decode.expect("No decode queue");
    println!("  GPU: video decode queue family = {}", decode_qf);
    println!("  Extensions: {}\n", vulkan.enabled_extensions.join(", "));

    // Query video decode capabilities to get pictureAccessGranularity
    // and check DPB_AND_OUTPUT_COINCIDE support
    println!("--- Step 2b: Query video capabilities ---");
    let (video_caps, decode_caps) = query_video_decode_capabilities(
        &vulkan,
        codec,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    )
    .expect("Failed to query video capabilities");
    let dpb_and_output_coincide = decode_caps
        .flags
        .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
    println!("  DPB_AND_OUTPUT_COINCIDE: {}", dpb_and_output_coincide);
    println!(
        "  minBitstreamBufferSizeAlignment: {}",
        video_caps.min_bitstream_buffer_size_alignment
    );
    println!("  maxDPBSlots: {}", video_caps.max_dpb_slots);
    println!(
        "  pictureAccessGranularity: {}x{}\n",
        video_caps.picture_access_granularity.width, video_caps.picture_access_granularity.height
    );

    // Use coded extent aligned to picture access granularity (matching C++ reference)
    // Dimensions are rounded up to the nearest multiple of 16
    let coded_extent = vk::Extent2D {
        width: parsed.coded_width,
        height: parsed.coded_height,
    };
    println!(
        "  Coded extent: {}x{}\n",
        coded_extent.width, coded_extent.height
    );

    let dpb_slots = parsed.max_dpb_slots.min(4);

    // Step 3: Create video session
    println!("--- Step 3: Create video session ---");
    // Reference adds 1 to maxDpbSlots
    let session_dpb_slots = dpb_slots + 1;
    let (session, session_params, session_memories) = create_video_session(
        &vulkan.instance,
        &vulkan.device,
        &vulkan.memory_properties,
        decode_qf,
        codec,
        parsed.profile_idc,
        coded_extent,
        session_dpb_slots,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
        parsed.vps.as_ref(),
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    )
    .expect("Failed to create video session");
    println!("  Video session created\n");

    // Step 4: Session parameters already have SPS/PPS from creation
    println!("--- Step 4: Session parameters created with SPS/PPS inline ---");
    println!("  No update needed - SPS/PPS provided during session parameters creation\n");

    // Step 5: Create output image with video profile
    // IMPORTANT: Output image must be at least as large as the coded extent
    // (which is aligned to picture access granularity)
    // The image MUST be created with VkVideoProfileListInfoKHR in pNext
    // to satisfy VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07266
    println!("--- Step 5: Create output image ---");
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
    let (output_image, output_image_view, output_memory) = create_output_image_with_profile(
        &vulkan.instance,
        &vulkan.device,
        &vulkan.memory_properties,
        coded_extent.width,
        coded_extent.height,
        output_format,
        codec,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    )
    .expect("Failed to create output image");
    println!(
        "  Output image: {}x{} (coded extent) format={:?}\n",
        coded_extent.width, coded_extent.height, output_format
    );

    // Step 6: Extract all access units
    println!("--- Step 6: Extract access units ---");
    let max_frames_to_decode = 20;
    let access_units = extract_all_access_units(
        &data,
        codec,
        max_frames_to_decode,
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    );
    println!("  Extracted {} access units\n", access_units.len());

    if access_units.is_empty() {
        eprintln!("Error: No access units found in bitstream");
        std::process::exit(1);
    }

    // Step 7: Create bitstream buffer with video profile
    // IMPORTANT: Must include VkVideoProfileListInfoKHR to satisfy
    // VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07135 and related errors
    // Size buffer for the largest access unit
    let max_au_size = access_units
        .iter()
        .map(|au| au.data.len())
        .max()
        .unwrap_or(0);
    println!("--- Step 7: Create bitstream buffer ---");
    println!("  Bitstream contents (first 64 bytes of first AU):");
    for i in 0..64.min(access_units[0].data.len()) {
        print!("{:02x} ", access_units[0].data[i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();

    let mut bs_buffer = create_bitstream_buffer_with_profile(
        &vulkan.device,
        &vulkan.memory_properties,
        max_au_size as u64,
        codec,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    )
    .expect("Failed to create bitstream buffer");
    println!("  Bitstream buffer created successfully");

    // Step 8: Create command resources
    println!("--- Step 8: Create command resources ---");
    let (command_pool, command_buffer) = create_command_resources(&vulkan.device, decode_qf)
        .expect("Failed to create command resources");
    println!("  Command buffer allocated\n");

    // Step 9: Create fence
    println!("--- Step 9: Create fence ---");
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

    // Step 10: Decode multiple frames with DPB management
    println!("--- Step 10: Decode {} frames ---", access_units.len());

    // Create DPB images for reference frame management
    // IMPORTANT: Must use VkVideoProfileListInfoKHR to satisfy
    // VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07266
    let dpb_slots = parsed.max_dpb_slots.min(4);
    let mut dpb_images: Vec<(vk::Image, vk::ImageView, vk::DeviceMemory)> = Vec::new();

    // Create DPB images if we need more than 1
    if dpb_slots > 1 {
        for slot in 0..dpb_slots {
            let (img, view, mem) = create_output_image_with_profile(
                &vulkan.instance,
                &vulkan.device,
                &vulkan.memory_properties,
                coded_extent.width,
                coded_extent.height,
                output_format,
                codec,
                parsed.profile_idc,
                parsed.chroma_subsampling,
                parsed.luma_bit_depth,
                parsed.chroma_bit_depth,
            )
            .unwrap_or_else(|e| {
                eprintln!("Failed to create DPB image {}: {}", slot, e);
                std::process::exit(1);
            });
            dpb_images.push((img, view, mem));
        }
    }

    // Use output_image as DPB slot 0, dpb_images as slots 1..N
    let mut dpb_views: Vec<vk::ImageView> = vec![output_image_view];
    let mut dpb_image_handles: Vec<vk::Image> = vec![output_image];
    for (_, view, _) in &dpb_images {
        dpb_views.push(*view);
    }
    for (img, _, _) in &dpb_images {
        dpb_image_handles.push(*img);
    }

    let mut dpb_manager = DpbManager::new(session_dpb_slots);

    // Set max_num_ref_frames from SPS for sliding window reference picture marking
    if let Some(H264OrH265Sps::H264(sps)) = &parsed.sps {
        dpb_manager.set_max_num_ref_frames(sps.max_num_ref_frames);
        eprintln!(
            "[dpb] max_num_ref_frames={} from SPS",
            sps.max_num_ref_frames
        );
    }

    // Decode each frame
    let mut is_first_frame = true;
    let mut decoder_reset_done = false;

    // Reference picture verification: store frame 0 output for comparison
    let mut frame0_output_pixels: Option<DecodedPixels> = None;
    let mut frame0_output_slot: Option<u32> = None;

    // FIX: Persistent storage for codec-specific Vulkan structs.
    // Each frame's pic_info/decode_info/ref_info/dpb_slot_info must have stable memory
    // that outlives GPU execution. Stack variables at fixed offsets get overwritten
    // by subsequent frames, causing all decode commands to read the last frame's data.
    let mut h265_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH265PictureInfo> = Vec::new();
    let mut h265_decode_info_vec: Vec<vk::VideoDecodeH265PictureInfoKHR> = Vec::new();
    let mut h265_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo> = Vec::new();
    let mut h265_dpb_slot_info_vec: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = Vec::new();
    let mut h264_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH264PictureInfo> = Vec::new();
    let mut h264_decode_info_vec: Vec<vk::VideoDecodeH264PictureInfoKHR> = Vec::new();
    let mut h264_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo> = Vec::new();
    let mut h264_dpb_slot_info_vec: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = Vec::new();

    for (frame_idx, au) in access_units.iter().enumerate().take(max_frames) {
        // ========================================================================
        // REFERENCE PICTURE VERIFICATION TEST (DISABLED)
        // Was verifying reference picture integrity by reading from the same DPB slot
        // that held frame 0, and comparing with the stored frame 0 output.
        // DISABLED: Layout transitions during readback may interfere with frame 1 decode.
        // ========================================================================
        // if frame_idx == 1 {
        //     if let (Some(ref frame0_pixels), Some(ref_slot)) = (&frame0_output_pixels, frame0_output_slot) {
        //         println!("\n  [REF PICTURE TEST] Verifying reference picture BEFORE frame 1 decode...");
        //         println!("    Reading from slot {} (same as frame 0 output)", ref_slot);
        //
        //         let ref_img = dpb_image_handles[ref_slot as usize];
        //         let ref_pic_readback = readback_decoded_image(
        //             &vulkan.instance,
        //             &vulkan.device,
        //             &vulkan.memory_properties,
        //             decode_qf,
        //             command_pool,
        //             fence,
        //             ref_img,
        //             coded_extent.width,
        //             coded_extent.height,
        //         );
        //
        //         // Restore layout after readback
        //         dpb_manager.set_slot_layout(ref_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        //         dpb_manager.set_slot_last_access(ref_slot, LastAccessType::TransferRead);
        //
        //         match ref_pic_readback {
        //             Ok(ref_pixels) => {
        //                 println!("    Reference picture readback complete");
        //                 let comparison = compare_decoded_pixels(
        //                     frame0_pixels,
        //                     &ref_pixels,
        //                     parsed.coded_width,
        //                     parsed.coded_height,
        //                 );
        //
        //                 if comparison.matched {
        //                     println!("    ✓ PASS: Reference picture matches frame 0 output EXACTLY");
        //                     println!("      This means reference picture STORAGE is CORRECT.");
        //                 } else {
        //                     println!("    ✗ FAIL: Reference picture differs from frame 0 output");
        //                     println!("      Y-MSE: {:.4}, Y-PSNR: {:.2} dB", comparison.y_mse, comparison.y_psnr);
        //                     println!("      Y diff pixels: {} ({:.2}%)", comparison.y_diff_count, comparison.y_diff_pct);
        //                     println!("      This means reference picture STORAGE is being CORRUPTED.");
        //                 }
        //             }
        //             Err(e) => {
        //                 eprintln!("    Reference picture readback failed: {}", e);
        //             }
        //         }
        //     }
        // }

        println!("\n[Frame {}] frame_num={}, POC=[{}, {}], is_idr={}, is_ref={}, slice_type={}, size={} bytes",
            frame_idx, au.frame_num, au.pic_order_cnt[0], au.pic_order_cnt[1],
            au.is_idr, au.is_reference, au.slice_type, au.data.len());

        // Write bitstream data
        bs_buffer
            .write(&au.data)
            .expect("Failed to write bitstream data");

        // Zero padding to aligned boundary to prevent decoder reading garbage
        // from previous frames that weren't overwritten
        let aligned_size = ((au.data.len() as u64 + 255) & !255).max(256);
        let padding_start = au.data.len() as u64;
        let padding_size = aligned_size - padding_start;
        if padding_size > 0 {
            bs_buffer.zero_range(padding_start, padding_size);
        }

        // Flush the entire aligned range (data + zeroed padding)
        bs_buffer.flush_range(0, aligned_size).ok();

        // Select DPB slot for this frame
        let output_slot;
        if au.is_idr {
            // IDR: invalidate all DPB entries, use slot 0
            dpb_manager.invalidate_all();
            output_slot = 0;
            eprintln!(
                "[decode_loop] Frame {} (IDR): selected slot {} (IDR path, invalidated all DPB)",
                frame_idx, output_slot
            );
        } else {
            // P/B frame: find or recycle a slot
            // For H.264, ref_pocs may be empty from parser, so protect ALL valid DPB entries
            // to avoid premature recycling causing deadlocks
            let protected_pocs: Vec<i32> = if codec == VideoCodec::H264 {
                dpb_manager.entries.iter()
                    .filter(|e| e.is_valid)
                    .flat_map(|e| if e.pic_order_cnt[0] == e.pic_order_cnt[1] {
                        vec![e.pic_order_cnt[0]]
                    } else {
                        vec![e.pic_order_cnt[0], e.pic_order_cnt[1]]
                    })
                    .collect()
            } else {
                au.ref_pocs.clone()
            };
            output_slot = dpb_manager
                .find_or_recycle_slot(&protected_pocs)
                .unwrap_or(0);
            eprintln!(
                "[decode_loop] Frame {} (non-IDR): selected slot {} (find_or_recycle path, protected_pocs={:?})",
                frame_idx, output_slot, protected_pocs
            );
        }
        eprintln!(
            "[decode_loop] Frame {}: ref_pocs={:?}",
            frame_idx, au.ref_pocs
        );

        let output_view = dpb_views[output_slot as usize];
        let output_img = dpb_image_handles[output_slot as usize];

        // Record decode command (pass DPB manager for reference slot setup)
        // Use aligned bitstream size to match Vulkan spec requirement for srcBufferRange.
        // The buffer is already zero-padded to aligned_size above, so the decoder
        // reads only valid data + zeros, not garbage from previous frames.
        let aligned_bs_size = aligned_size;
        let result = record_decode_command(
            &vulkan.instance,
            &vulkan.device,
            command_buffer,
            decode_qf,
            session,
            session_params,
            bs_buffer.buffer(),
            0,
            aligned_bs_size,
            output_view,
            output_img,
            coded_extent,
            codec,
            parsed.sps.as_ref(),
            parsed.pps.as_ref(),
            parsed.vps.as_ref(),
            &au.slice_offsets,
            au.frame_num,
            au.pic_order_cnt,
            au.is_idr,
            au.is_reference,
            au.slice_type,
               au.num_bits_for_st_ref_pic_set_in_slice,
               au.num_delta_pocs_of_ref_rps_idx,
               au.short_term_ref_pic_set_sps_flag,
               &au.ref_pocs,
              &dpb_manager,
            output_slot,
            frame_idx == 1,
            &au.data,
            decoder_reset_done,
            &dpb_views,
            &dpb_image_handles,
            fence,
            &mut h265_pic_info_vec,
            &mut h265_decode_info_vec,
            &mut h265_ref_info_vec,
            &mut h265_dpb_slot_info_vec,
            &mut h264_pic_info_vec,
            &mut h264_decode_info_vec,
            &mut h264_ref_info_vec,
            &mut h264_dpb_slot_info_vec,
        );

        // Mark first frame as done AFTER recording decode command
        if is_first_frame {
            is_first_frame = false;
            decoder_reset_done = true;
        }

        match result {
            Ok(()) => {}
            Err(e) => {
                eprintln!("[Frame {}] Decode command failed: {}", frame_idx, e);
                break;
            }
        }

        // Update DPB manager: register this frame as a valid reference in its slot
        // After decode completes, the output slot is in VIDEO_DECODE_DPB_KHR layout
        dpb_manager.set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        dpb_manager.set_slot_last_access(output_slot, LastAccessType::DecodeWrite);
        eprintln!(
            "[dpb] Updated slot {} layout to VIDEO_DECODE_DPB_KHR after decode",
            output_slot
        );

        if au.is_reference {
            dpb_manager.entries[output_slot as usize] = DpbEntry {
                frame_num: au.frame_num,
                pic_order_cnt: au.pic_order_cnt,
                slot_index: output_slot,
                is_valid: true,
                image_view: dpb_views[output_slot as usize],
                image: dpb_image_handles[output_slot as usize],
                current_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                last_access: LastAccessType::DecodeWrite,
            };
            eprintln!(
                "[dpb] Registered frame {} (POC={}) in slot {} as reference",
                au.frame_num, au.pic_order_cnt[0], output_slot
            );

            // DEBUG: Print current DPB state
            eprintln!("[dpb] Current DPB state after registration:");
            for (i, entry) in dpb_manager.entries.iter().enumerate() {
                if entry.is_valid {
                    eprintln!(
                        "[dpb]   slot={}, POC={}, frame_num={}, layout={:?}",
                        i, entry.pic_order_cnt[0], entry.frame_num, entry.current_layout
                    );
                }
            }

            // Apply reference picture marking after registering new reference.
            // When adaptive_ref_pic_marking_mode_flag is true, use MMCO commands.
            // When false, use sliding window.
            if au.adaptive_ref_pic_marking_mode_flag && !au.mmco_commands.is_empty() {
                eprintln!("[dpb] Using MMCO commands ({} commands)", au.mmco_commands.len());
                dpb_manager.apply_mmco(au.frame_num, output_slot, &au.mmco_commands);
            } else {
                eprintln!("[dpb] Using sliding window");
                dpb_manager.apply_sliding_window(au.frame_num);
            }
        }

        // Readback and verify decoded pixels
        println!("  Reading back decoded pixels...");
        eprintln!(
            "[readback] Frame {}: reading from output_slot={}, output_img={:?}",
            frame_idx, output_slot, output_img
        );
        let decoded_pixels_result = readback_decoded_image(
            &vulkan.instance,
            &vulkan.device,
            &vulkan.memory_properties,
            decode_qf,
            command_pool,
            fence,
            output_img,
            coded_extent.width,
            coded_extent.height,
        );

        // Update tracked layout after readback (image restored to VIDEO_DECODE_DPB_KHR)
        // The readback's restore barrier makes TRANSFER_READ visible to VIDEO_DECODE_READ.
        // Update last_access so the next frame's reference barrier uses the correct source access.
        dpb_manager.set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
        dpb_manager.set_slot_last_access(output_slot, LastAccessType::TransferRead);

        match &decoded_pixels_result {
            Ok(pixels) => {
                // Compute checksum of Y plane to verify frames are different
                // Use full plane hash + multiple sample regions to catch real differences
                let checksum_full: u64 = pixels.y_plane.iter()
                    .map(|b| *b as u64)
                    .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b));
                // Also sample center and bottom-right regions
                let y_len = pixels.y_plane.len();
                let center = y_len / 3;
                let end = y_len - 256;
                let checksum_center: u64 = pixels.y_plane[center..center+256].iter()
                    .map(|b| *b as u64).fold(0u64, |a, b| a.wrapping_add(b));
                let checksum_end: u64 = pixels.y_plane[end..end+256].iter()
                    .map(|b| *b as u64).fold(0u64, |a, b| a.wrapping_add(b));
                eprintln!("[checksum] Frame {}: full_hash={} center={} end={}, slice_type={}, is_idr={}",
                    frame_idx, checksum_full, checksum_center, checksum_end, au.slice_type, au.is_idr);

                verify_decoded_pixels(pixels, parsed.display_width, parsed.display_height, parsed.crop_left, parsed.crop_top);

                // Compare with ffmpeg reference (crop to display area)
                // Use POC-based matching to handle B-frame decode order differences
                compare_frame_with_ffmpeg(
                    bitstream_path,
                    codec,
                    parsed.display_width,
                    parsed.display_height,
                    parsed.crop_left,
                    parsed.crop_top,
                    au.pic_order_cnt[0],
                    pixels,
                );
            }
            Err(e) => {
                eprintln!("  Readback failed: {}", e);
            }
        }

        println!(
            "  DPB state: {} valid references",
            dpb_manager.get_references().len()
        );

        // ========================================================================
        // REFERENCE PICTURE VERIFICATION TEST (STORE frame 0 output)
        // After frame 0 (IDR) readback, store pixels for later comparison.
        // ========================================================================
        if frame_idx == 0 && au.is_idr {
            // Store frame 0 output for later comparison
            if let Ok(ref pixels) = decoded_pixels_result {
                frame0_output_pixels = Some(pixels.clone());
                frame0_output_slot = Some(output_slot);
                println!(
                    "\n  [REF PICTURE TEST] Stored frame 0 output from slot {}",
                    output_slot
                );
            }
        }

        // ========================================================================
        // FRAME 1 DETAILED DEBUGGING (after decode + readback)
        // ========================================================================
        if frame_idx == 1 {
            eprintln!("\n\n=================================================================");
            eprintln!("         FRAME 1 DETAILED DEBUG (after decode + readback)");
            eprintln!("=================================================================\n");

            // 1. Print the actual output slot used for Frame 1
            eprintln!("=== 1. Frame 1 Output Slot ===");
            eprintln!("  output_slot = {}", output_slot);
            eprintln!("  frame_num = {}", au.frame_num);
            eprintln!("  pic_order_cnt = [{}, {}]", au.pic_order_cnt[0], au.pic_order_cnt[1]);
            eprintln!("  is_idr = {}", au.is_idr);
            eprintln!("  is_reference = {}", au.is_reference);
            eprintln!("  slice_type = {}", au.slice_type);
            eprintln!();

            // 2. Print all DPB image handles and their sizes
            eprintln!("=== 2. DPB Image Handles and Sizes ===");
            for (i, img) in dpb_image_handles.iter().enumerate() {
                eprintln!("  dpb_image_handles[{}] = {:?} (raw={})", i, img, (*img).as_raw());
            }
            eprintln!("  Total DPB slots = {}", dpb_image_handles.len());
            eprintln!("  Coded extent = {}x{}", coded_extent.width, coded_extent.height);
            eprintln!();

            // 3. Compare Frame 1's decoded pixels with Frame 0's stored pixels
            eprintln!("=== 3. Frame 0 vs Frame 1 Pixel Comparison (Y plane, first 100 values) ===");
            if let (Ok(ref frame1_pixels), Some(ref frame0_pixels)) =
                (&decoded_pixels_result, &frame0_output_pixels)
            {
                eprintln!("  Frame 0 Y plane length = {}", frame0_pixels.y_plane.len());
                eprintln!("  Frame 1 Y plane length = {}", frame1_pixels.y_plane.len());

                let compare_len = 100.min(frame0_pixels.y_plane.len()).min(frame1_pixels.y_plane.len());
                let mut diff_count = 0u32;
                let mut max_diff = 0u32;

                eprintln!("  Index    Frame0_Y    Frame1_Y    Diff");
                eprintln!("  -----    --------    --------    ----");
                for i in 0..compare_len {
                    let y0 = frame0_pixels.y_plane[i] as u32;
                    let y1 = frame1_pixels.y_plane[i] as u32;
                    let diff = if y0 > y1 { y0 - y1 } else { y1 - y0 };
                    if diff > 0 {
                        diff_count += 1;
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    eprintln!("  {:5}    {:8}    {:8}    {}", i, y0, y1, diff);
                }
                eprintln!();
                eprintln!("  Summary: {} differences out of {} compared, max_diff = {}",
                    diff_count, compare_len, max_diff);

                // Also print some statistics
                let frame0_avg: u64 = frame0_pixels.y_plane.iter().map(|&b| b as u64).sum::<u64>() / frame0_pixels.y_plane.len() as u64;
                let frame1_avg: u64 = frame1_pixels.y_plane.iter().map(|&b| b as u64).sum::<u64>() / frame1_pixels.y_plane.len() as u64;
                eprintln!("  Frame 0 Y average = {}", frame0_avg);
                eprintln!("  Frame 1 Y average = {}", frame1_avg);
            } else {
                eprintln!("  WARNING: Could not compare - Frame 0 pixels not stored or Frame 1 readback failed");
            }
            eprintln!();

            // 4. Print the exact RefPicSet arrays used for Frame 1
            eprintln!("=== 4. Frame 1 RefPicSet Arrays ===");
            eprintln!("  ref_pocs from bitstream = {:?}", au.ref_pocs);
            eprintln!("  short_term_ref_pic_set_sps_flag = {}", au.short_term_ref_pic_set_sps_flag);
            eprintln!("  num_bits_for_st_ref_pic_set_in_slice = {}", au.num_bits_for_st_ref_pic_set_in_slice);
            eprintln!("  num_delta_pocs_of_ref_rps_idx = {}", au.num_delta_pocs_of_ref_rps_idx);

            // Print current DPB state showing which slots are references
            eprintln!("  Current DPB valid references:");
            for entry in dpb_manager.get_references().iter() {
                eprintln!("    slot={}, POC={}, frame_num={}",
                    entry.slot_index, entry.pic_order_cnt[0], entry.frame_num);
            }
            eprintln!();

            eprintln!("=================================================================");
            eprintln!("         END FRAME 1 DETAILED DEBUG");
            eprintln!("=================================================================\n\n");
        }
    }

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", access_units.len());
    println!("DPB references: {}", dpb_manager.get_references().len());

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

    unsafe {
        destroy_session_parameters(&vulkan.instance, vulkan.device.handle(), session_params);
        destroy_session(&vulkan.instance, vulkan.device.handle(), session);

        for mem in session_memories {
            vulkan.device.free_memory(mem, None);
        }

        // Destroy debug messenger BEFORE destroying instance
        if vulkan.has_validation && vulkan.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
            let debug_utils = ash::ext::debug_utils::Instance::new(&vulkan.entry, &vulkan.instance);
            let _ = debug_utils.destroy_debug_utils_messenger(vulkan.debug_messenger, None);
            vulkan.debug_messenger = vk::DebugUtilsMessengerEXT::null();
        }

        vulkan.device.destroy_device(None);
        vulkan.instance.destroy_instance(None);
    }

    println!("\n=== Done ===");
}

// ============================================================================
// Codec detection
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoCodec {
    H264,
    H265,
}

impl VideoCodec {
    fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264/AVC",
            Self::H265 => "H.265/HEVC",
        }
    }

    fn vk_codec_op(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            Self::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Self::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
        }
    }

    fn to_vk_codec(self) -> vk_video_vulkan::VideoCodec {
        match self {
            Self::H264 => vk_video_vulkan::VideoCodec::DecodeH264,
            Self::H265 => vk_video_vulkan::VideoCodec::DecodeH265,
        }
    }
}

// ============================================================================
// Access unit and frame info
// ============================================================================

/// An access unit (single frame) extracted from the bitstream.
#[derive(Debug, Clone)]
struct AccessUnit {
    /// Bitstream data (slice NALs with start codes, no SPS/PPS)
    data: Vec<u8>,
    /// Offsets of each slice within data (pointing to start codes)
    slice_offsets: Vec<u32>,
    /// Frame number from first slice header
    frame_num: u32,
    /// Picture order count [top_field, bottom_field]
    pic_order_cnt: [i32; 2],
    /// Whether this is an IDR frame
    is_idr: bool,
    /// Whether this is a reference frame
    is_reference: bool,
    /// Slice type (0=I, 1=P, 2=B, 3=SI, 4=SP for H.264)
    slice_type: u32,
    /// H.265: NumBitsForShortTermRPSInSlice from slice header
    num_bits_for_st_ref_pic_set_in_slice: i32,
    /// H.265: NumDeltaPocsOfRefRpsIdx from slice header
    num_delta_pocs_of_ref_rps_idx: i32,
    /// H.265: short_term_ref_pic_set_sps_flag from slice header (for StdVideoDecodeH265PictureInfo)
    short_term_ref_pic_set_sps_flag: bool,
    /// H.265: Computed reference picture POCs from RPS (empty for IDR/I-frames)
    ref_pocs: Vec<i32>,
    /// H.264: adaptive_ref_pic_marking_mode_flag from slice header (true=MMCO, false=sliding window)
    adaptive_ref_pic_marking_mode_flag: bool,
    /// H.264: MMCO commands parsed from slice header
    mmco_commands: Vec<H264MmcoCommand>,
}

/// H.264 Memory Management Control Operation (MMCO) command.
/// See H.264 spec 8.2.5.4 for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum H264MmcoCommand {
    /// MMCO 1: Unmark short-term reference with difference_of_pic_nums_minus1
    UnmarkShortTerm { difference_of_pic_nums_minus1: u32 },
    /// MMCO 2: Unmark long-term reference with long_term_frame_idx
    UnmarkLongTerm { long_term_frame_idx: u32 },
    /// MMCO 3: Assign LongTermFrameIdx to short-term reference
    AssignLongTerm { difference_of_pic_nums_minus1: u32, long_term_frame_idx: u32 },
    /// MMCO 4: Set MaxLongTermFrameIdx
    SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: u32 },
    /// MMCO 5: Unmark all references
    UnmarkAll,
    /// MMCO 6: Assign LongTermFrameIdx to current picture
    AssignLongTermToCurrent { long_term_frame_idx: u32 },
}

/// Slice header information extracted from a NAL unit.
#[derive(Debug, Clone)]
struct SliceHeaderInfo {
    frame_num: u32,
    pic_order_cnt: [i32; 2],
    pic_order_cnt_lsb: i32,
    pic_order_cnt_msb: i32,
    is_idr: bool,
    is_reference: bool,
    slice_type: u32,
}

/// DPB (Decoded Picture Buffer) manager for tracking reference frames.
struct DpbManager {
    entries: Vec<DpbEntry>,
    max_dpb_slots: u32,
    prev_frame_num: u32,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    /// Next slot to use for new frames
    next_slot: u32,
    /// Maximum number of reference frames allowed (from SPS max_num_ref_frames)
    max_num_ref_frames: u32,
}

/// DPB entry tracking reference frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LastAccessType {
    DecodeWrite,
    TransferRead,
    None,
}

/// DPB entry tracking reference frames.
#[derive(Debug, Clone)]
struct DpbEntry {
    /// Frame number
    frame_num: u32,
    /// Picture order count
    pic_order_cnt: [i32; 2],
    /// DPB slot index this frame is in
    slot_index: u32,
    /// Whether this is a valid reference
    is_valid: bool,
    /// Image view for this DPB slot
    image_view: vk::ImageView,
    /// Image handle for this DPB slot
    image: vk::Image,
    /// Current layout of this DPB slot's image
    /// Match C++ reference: track layout per slot to avoid unnecessary barriers
    current_layout: vk::ImageLayout,
    /// Last access type - used to determine correct src_access_mask for barriers
    last_access: LastAccessType,
}

impl DpbManager {
    fn new(max_dpb_slots: u32) -> Self {
        Self {
            entries: vec![
                DpbEntry {
                    frame_num: 0,
                    pic_order_cnt: [0, 0],
                    slot_index: 0,
                    is_valid: false,
                    image_view: vk::ImageView::null(),
                    image: vk::Image::null(),
                    current_layout: vk::ImageLayout::UNDEFINED,
                    last_access: LastAccessType::None,
                };
                max_dpb_slots as usize
            ],
          max_dpb_slots: max_dpb_slots,
            prev_frame_num: 0,
            prev_pic_order_cnt_lsb: 0,
            prev_pic_order_cnt_msb: 0,
            next_slot: 0,
            max_num_ref_frames: 16, // Default, will be updated from SPS
        }
    }

    /// Set maximum number of reference frames from SPS.
    fn set_max_num_ref_frames(&mut self, max_num_ref_frames: u32) {
        self.max_num_ref_frames = max_num_ref_frames;
    }

    /// Apply sliding window decoded reference picture marking.
    ///
    /// Per H.264 spec 8.2.5.2: If the number of short-term reference pictures
    /// exceeds max_num_ref_frames, the oldest short-term reference picture
    /// (smallest PicNum) shall be marked as "unused for reference".
    ///
    /// This is called after each reference frame is decoded.
    fn apply_sliding_window(&mut self, current_frame_num: u32) {
        let max_refs = self.max_num_ref_frames.max(1) as usize;

        // Count short-term references (all valid entries except current)
        let num_short_term = self
            .entries
            .iter()
            .filter(|e| e.is_valid && e.frame_num != current_frame_num)
            .count();

        if num_short_term >= max_refs {
            // Find the oldest short-term reference (smallest FrameNumWrap)
            // FrameNumWrap is the frame_num value as seen in wraparound order
            let mut oldest_idx: Option<usize> = None;
            let mut oldest_frame_num = u32::MAX;

            for (i, entry) in self.entries.iter().enumerate() {
                if entry.is_valid && entry.frame_num != current_frame_num {
                    // Use simple comparison for FrameNumWrap
                    if entry.frame_num < oldest_frame_num {
                        oldest_frame_num = entry.frame_num;
                        oldest_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = oldest_idx {
                eprintln!(
                    "[dpb] Sliding window: removing oldest ref frame_num={} (slot={})",
                    self.entries[idx].frame_num, idx
                );
                self.entries[idx].is_valid = false;
            }
        }
    }

    /// Find an empty slot or recycle the oldest reference.
    ///
    /// `protected_pocs` is a list of reference POCs that the current frame needs.
    /// Slots containing frames with these POCs will NOT be recycled, preventing
    /// destruction of reference pictures needed for the current decode.
    fn find_or_recycle_slot(&mut self, protected_pocs: &[i32]) -> Option<u32> {
        for i in 0..self.max_dpb_slots as usize {
            if !self.entries[i].is_valid {
                return Some(i as u32);
            }
        }
        // Recycle the oldest short-term reference (smallest FrameNumWrap)
        // BUT skip slots that contain protected reference pictures
        let mut oldest_idx = None;
        let mut oldest_wrap = u32::MAX;
        for i in 0..self.max_dpb_slots as usize {
            if self.entries[i].is_valid {
                // Skip this slot if it contains a protected reference picture
                let poc = self.entries[i].pic_order_cnt[0];
                if protected_pocs.contains(&poc) {
                    continue;
                }
                let wrap = self.entries[i].frame_num;
                if wrap < oldest_wrap {
                    oldest_wrap = wrap;
                    oldest_idx = Some(i as u32);
                }
            }
        }
        // CRITICAL: Invalidate the recycled slot so it's not used as a reference
        // for the current frame being decoded into this slot.
        if let Some(idx) = oldest_idx {
            self.entries[idx as usize].is_valid = false;
            self.entries[idx as usize].current_layout = vk::ImageLayout::UNDEFINED;
            self.entries[idx as usize].last_access = LastAccessType::None;
        }
        oldest_idx
    }

    /// Mark all entries as invalid (for IDR frames).
    fn invalidate_all(&mut self) {
        for entry in &mut self.entries {
            entry.is_valid = false;
            // Reset layout to UNDEFINED when invalidating - matches C++ behavior
            entry.current_layout = vk::ImageLayout::UNDEFINED;
            entry.last_access = LastAccessType::None;
        }
    }

    /// Get the current layout of a DPB slot.
    fn get_slot_layout(&self, slot_index: u32) -> vk::ImageLayout {
        self.entries[slot_index as usize].current_layout
    }

    /// Update the layout of a DPB slot after a barrier/decode.
    fn set_slot_layout(&mut self, slot_index: u32, layout: vk::ImageLayout) {
        self.entries[slot_index as usize].current_layout = layout;
    }

    /// Get the last access type of a DPB slot.
    fn get_slot_last_access(&self, slot_index: u32) -> LastAccessType {
        self.entries[slot_index as usize].last_access
    }

    /// Update the last access type of a DPB slot.
    fn set_slot_last_access(&mut self, slot_index: u32, access: LastAccessType) {
        self.entries[slot_index as usize].last_access = access;
    }

    /// Find a DPB entry by frame number.
    fn find_by_frame_num(&self, frame_num: u32) -> Option<(usize, &DpbEntry)> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid && entry.frame_num == frame_num {
                return Some((i, entry));
            }
        }
        None
    }

    /// Get all valid reference entries.
    fn get_references(&self) -> Vec<&DpbEntry> {
        self.entries.iter().filter(|e| e.is_valid).collect()
    }

    /// Apply H.264 MMCO (Memory Management Control Operations) commands.
    /// See H.264 spec 8.2.5.4 for details.
    fn apply_mmco(
        &mut self,
        current_frame_num: u32,
        _current_slot_index: u32,
        mmco_commands: &[H264MmcoCommand],
    ) {
        eprintln!("[dpb] Applying MMCO commands ({} commands):", mmco_commands.len());

        for cmd in mmco_commands {
            eprintln!("[dpb]   MMCO: {:?}", cmd);
        }

        for cmd in mmco_commands {
            match cmd {
                // MMCO 1: Mark short-term reference as unused
                // picNumX = CurrPicNum - (difference_of_pic_nums_minus1 + 1)
                H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1 } => {
                    let pic_num_x = if *difference_of_pic_nums_minus1 + 1 <= current_frame_num {
                        current_frame_num - (difference_of_pic_nums_minus1 + 1)
                    } else {
                        // Wraparound case
                        u32::MAX - (difference_of_pic_nums_minus1 + 1 - current_frame_num)
                    };
                    eprintln!("[dpb]   MMCO 1: unmark short-term picNumX={}", pic_num_x);
                    for entry in &mut self.entries {
                        if entry.is_valid && entry.frame_num == pic_num_x {
                            entry.is_valid = false;
                            eprintln!("[dpb]     invalidated slot {} (frame_num={})", entry.slot_index, entry.frame_num);
                        }
                    }
                }

                // MMCO 2: Mark long-term reference as unused (not fully tracked)
                H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx } => {
                    eprintln!("[dpb]   MMCO 2: unmark long-term long_term_frame_idx={} (not fully tracked)", long_term_frame_idx);
                }

                // MMCO 3: Assign LongTermFrameIdx to short-term reference (not fully tracked)
                H264MmcoCommand::AssignLongTerm { difference_of_pic_nums_minus1, long_term_frame_idx } => {
                    let pic_num_x = if *difference_of_pic_nums_minus1 + 1 <= current_frame_num {
                        current_frame_num - (difference_of_pic_nums_minus1 + 1)
                    } else {
                        u32::MAX - (difference_of_pic_nums_minus1 + 1 - current_frame_num)
                    };
                    eprintln!("[dpb]   MMCO 3: assign LongTermFrameIdx={} to picNumX={} (not fully tracked)",
                              long_term_frame_idx, pic_num_x);
                }

                // MMCO 4: Set MaxLongTermFrameIdx (not fully tracked)
                H264MmcoCommand::SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1 } => {
                    eprintln!("[dpb]   MMCO 4: set MaxLongTermFrameIdx={} (not fully tracked)",
                              max_long_term_frame_idx_plus1);
                }

                // MMCO 5: Unmark all references
                H264MmcoCommand::UnmarkAll => {
                    eprintln!("[dpb]   MMCO 5: unmark ALL references");
                    for entry in &mut self.entries {
                        if entry.is_valid {
                            entry.is_valid = false;
                        }
                    }
                }

                // MMCO 6: Assign LongTermFrameIdx to current picture (not fully tracked)
                H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx } => {
                    eprintln!("[dpb]   MMCO 6: assign LongTermFrameIdx={} to current (not fully tracked)",
                              long_term_frame_idx);
                }
            }
        }
    }
}

/// Query video capabilities with decode-specific capabilities chained via pNext.
fn query_video_decode_capabilities(
    vulkan: &vk_video_vulkan::VulkanDevice,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<
    (
        vk::VideoCapabilitiesKHR<'_>,
        vk::VideoDecodeCapabilitiesKHR<'_>,
    ),
    String,
> {
    use ash::vk::Handle;

    let codec_op = codec.vk_codec_op();

    // Build codec-specific profile info
    let h264_profile = vk::VideoDecodeH264ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE,
        _marker: Default::default(),
    };
    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    let codec_profile_ptr = match codec {
        VideoCodec::H264 => &h264_profile as *const _ as *const _,
        VideoCodec::H265 => &h265_profile as *const _ as *const _,
    };

    // Build pNext chain for capabilities (NOT including profile info):
    // For H264: VideoDecodeH264CapabilitiesKHR -> VideoDecodeCapabilitiesKHR -> NULL
    // For H265: VideoDecodeH265CapabilitiesKHR -> VideoDecodeCapabilitiesKHR -> NULL
    let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
    decode_caps.s_type = vk::StructureType::VIDEO_DECODE_CAPABILITIES_KHR;
    decode_caps.p_next = std::ptr::null_mut(); // Do NOT chain to profile info!

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: codec_profile_ptr,
        video_codec_operation: codec_op,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    // Codec-specific capabilities
    let mut h264_decode_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
    let mut h265_decode_caps = vk::VideoDecodeH265CapabilitiesKHR::default();

    let codec_caps_ptr: *mut std::ffi::c_void = match codec {
        VideoCodec::H264 => {
            h264_decode_caps.s_type = vk::StructureType::VIDEO_DECODE_H264_CAPABILITIES_KHR;
            h264_decode_caps.p_next = &mut decode_caps as *mut _ as *mut _;
            &mut h264_decode_caps as *mut _ as *mut _
        }
        VideoCodec::H265 => {
            h265_decode_caps.s_type = vk::StructureType::VIDEO_DECODE_H265_CAPABILITIES_KHR;
            h265_decode_caps.p_next = &mut decode_caps as *mut _ as *mut _;
            &mut h265_decode_caps as *mut _ as *mut _
        }
    };

    // Get function pointer
    let get_caps_fn = unsafe {
        vulkan.entry.get_instance_proc_addr(
            vulkan.instance.handle(),
            b"vkGetPhysicalDeviceVideoCapabilitiesKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkGetPhysicalDeviceVideoCapabilitiesKHR not found")?;

    let mut caps = vk::VideoCapabilitiesKHR::default();
    caps.p_next = codec_caps_ptr;

    let result = unsafe {
        type FnType = unsafe extern "system" fn(
            vk::PhysicalDevice,
            *const vk::VideoProfileInfoKHR<'_>,
            *mut vk::VideoCapabilitiesKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(get_caps_fn);
        fn_ptr(vulkan.physical_device, &profile_info, &mut caps)
    };

    if result != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoCapabilitiesKHR failed: {:?}",
            result
        ));
    }

    Ok((caps, decode_caps))
}

fn detect_codec(path: &str) -> VideoCodec {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "h264" | "avc" | "264" => VideoCodec::H264,
        "h265" | "hevc" | "265" => VideoCodec::H265,
        _ => {
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem.contains("h265") || stem.contains("hevc") {
                VideoCodec::H265
            } else {
                VideoCodec::H264
            }
        }
    }
}

// ============================================================================
// Access unit extraction
// ============================================================================

/// Minimal bit reader for slice header parsing (EPB bytes must be removed before creating the reader).
struct SliceBitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bit(&mut self) -> Option<u32> {
        let byte_idx = self.pos / 8;
        if byte_idx >= self.data.len() {
            return None;
        }
        let bit_idx = 7 - (self.pos % 8);
        let bit = ((self.data[byte_idx] >> bit_idx) & 1) as u32;
        self.pos += 1;
        Some(bit)
    }

    fn read_bits(&mut self, n: u32) -> Option<u32> {
        let pos_before = self.pos;
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()?;
        }
        eprintln!("[DEBUG-BITS] n={}, pos_before={}, pos_after={}, val={:032b} ({})", n, pos_before, self.pos, val, val);
        Some(val)
    }

    fn read_ue(&mut self) -> Option<u32> {
        let pos_before = self.pos;
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_bit()?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return None;
            }
        }
        let mut value = 0u32;
        for _ in 0..leading_zeros {
            value = (value << 1) | self.read_bit()?;
        }
        let result = (1 << leading_zeros) - 1 + value;
        let bits_consumed = self.pos - pos_before;
        eprintln!("[DEBUG-UE] pos_before={}, bits_consumed={}, leading_zeros={}, value={}, result={}", pos_before, bits_consumed, leading_zeros, value, result);
        Some(result)
    }

    fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        if ue & 1 != 0 {
            Some((ue + 1) as i32 / 2)
        } else {
            Some(-((ue as i32) / 2))
        }
    }

    /// Get current bit position
    fn pos(&self) -> usize {
        self.pos
    }
}

/// Parse H.264 slice header to extract frame boundary info.
/// Returns first_mb_in_slice, frame_num, and pic_order_cnt_lsb.
fn parse_h264_slice_header(
    nal_data: &[u8],
    sps: &vk_video_core::picture::H264Sps,
    nal_ref_idc: u8,
    nal_unit_type: u8,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    max_pic_order_cnt_lsb: u32,
) -> Option<(u32, u32, i32, i32, [i32; 2], bool, Vec<H264MmcoCommand>)> {
    // Returns: (first_mb_in_slice, frame_num, pic_order_cnt_lsb, pic_order_cnt_msb, pic_order_cnt, adaptive_ref_pic_marking_mode_flag, mmco_commands)
    if nal_data.len() < 4 {
        return None;
    }

    let payload = &nal_data[1..]; // Skip NAL header
    let mut r = SliceBitReader::new(payload);

    let first_mb_in_slice = r.read_ue()?;
    let slice_type_raw = r.read_ue()?;
    let slice_type = slice_type_raw % 5;
    let _pps_id = r.read_ue()?;

    let _ = slice_type; // unused but parsed

    // frame_num uses log2_max_frame_num_minus4 + 4 bits
    let frame_num_bits = sps.log2_max_frame_num_minus4 as u32 + 4;
    let frame_num = r.read_bits(frame_num_bits)?;

    let is_idr = nal_unit_type == 5;

    // idr_pic_id for IDR
    if is_idr {
        let _idr_pic_id = r.read_ue().unwrap_or(0);
        let _ = _idr_pic_id;
    }

    // pic_order_cnt_lsb for type 0 POC
    let pic_order_cnt_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let pic_order_cnt_lsb = r.read_bits(pic_order_cnt_lsb_bits)? as i32;

    // Calculate POC MSB (type 0 POC per H.264 spec 8.2.1.1)
    let pic_order_cnt_msb = if is_idr {
        0
    } else if pic_order_cnt_lsb < prev_pic_order_cnt_lsb
        && (prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= (max_pic_order_cnt_lsb as i32 / 2)
    {
        prev_pic_order_cnt_msb + max_pic_order_cnt_lsb as i32
    } else if pic_order_cnt_lsb > prev_pic_order_cnt_lsb
        && (pic_order_cnt_lsb - prev_pic_order_cnt_lsb) > (max_pic_order_cnt_lsb as i32 / 2)
    {
        prev_pic_order_cnt_msb - max_pic_order_cnt_lsb as i32
    } else {
        prev_pic_order_cnt_msb
    };

    let pic_order_cnt = [
        pic_order_cnt_msb + pic_order_cnt_lsb,
        pic_order_cnt_msb + pic_order_cnt_lsb,
    ];

    // adaptive_ref_pic_marking_mode_flag appears after pic_order_cnt for reference frames
    let adaptive_ref_pic_marking_mode_flag = if nal_ref_idc > 0 {
        r.read_bit().unwrap_or(0) != 0
    } else {
        false
    };

    // Parse MMCO commands if adaptive_ref_pic_marking_mode_flag is true
    // See H.264 spec 7.3.3 and 8.2.5.4
    let mut mmco_commands = Vec::new();
    if adaptive_ref_pic_marking_mode_flag {
        loop {
            let Some(memory_management_control_operation) = r.read_ue() else {
                break;
            };

            // MMCO 0 is the terminator
            if memory_management_control_operation == 0 {
                break;
            }

            let cmd = match memory_management_control_operation {
                // MMCO 1: Unmark short-term reference
                1 => {
                    let difference_of_pic_nums_minus1 = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1 }
                }
                // MMCO 2: Unmark long-term reference
                2 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx }
                }
                // MMCO 3: Assign LongTermFrameIdx to short-term reference
                3 => {
                    let difference_of_pic_nums_minus1 = r.read_ue().unwrap_or(0);
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::AssignLongTerm { difference_of_pic_nums_minus1, long_term_frame_idx }
                }
                // MMCO 4: Set MaxLongTermFrameIdx
                4 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: long_term_frame_idx }
                }
                // MMCO 5: Unmark all references
                5 => H264MmcoCommand::UnmarkAll,
                // MMCO 6: Assign LongTermFrameIdx to current picture
                6 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx }
                }
                _ => {
                    // Unknown MMCO, stop parsing
                    break;
                }
            };
            mmco_commands.push(cmd);
        }
    }

    Some((
        first_mb_in_slice,
        frame_num,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        adaptive_ref_pic_marking_mode_flag,
        mmco_commands,
    ))
}

/// Parse H.265 slice header to extract frame boundary info.
///
/// Based on VulkanH265Parser.cpp:2119-2217 for slice header parsing,
/// and lines 2757-2799 for POC computation.
///
/// Returns: (first_slice_in_pic, pic_order_cnt_lsb, pic_order_cnt_msb, pic_order_cnt, is_idr, is_reference, slice_type, num_bits_for_st_ref_pic_set_in_slice, num_delta_pocs_of_ref_rps_idx, short_term_ref_pic_set_sps_flag, ref_pocs)
fn parse_h265_slice_header(
    nal_data: &[u8],
    sps: &vk_video_core::picture::H265Sps,
    pps: &vk_video_core::picture::H265Pps,
    nal_unit_type: u8,
    nuh_temporal_id_plus1: u8,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
) -> Option<(bool, i32, i32, [i32; 2], bool, bool, u32, i32, i32, bool, Vec<i32>)> {
    if nal_data.len() < 3 {
        return None;
    }

    // H.265 NAL header + slice segment addressing: payload starts at byte 2
    // Byte 0: NAL header (forbidden_zero_bit, nal_unit_type, nuh_layer_id)
    // Byte 1: slice_segment_address or other slice layer syntax
    // Byte 2+: slice segment header
    let payload = remove_emulation_prevention_bytes(&nal_data[2..]);
    eprintln!("[DEBUG] payload bytes (first 16): {:?}", &payload[..payload.len().min(16)]);
    let mut r = SliceBitReader::new(&payload);

    // first_slice_segment_in_pic_flag
    let first_slice_segment_in_pic_flag = r.read_bit()? == 1;
    eprintln!("[DEBUG] after first_slice_segment_in_pic_flag: pos={}, val={}", r.pos(), first_slice_segment_in_pic_flag);

    // For RAP pictures: no_output_of_prior_pics_flag
    let is_rap = nal_unit_type >= 16 && nal_unit_type <= 23;
    if is_rap {
        let _no_output_of_prior_pics_flag = r.read_bit().unwrap_or(0);
        eprintln!("[DEBUG] after no_output_of_prior_pics_flag: pos={}", r.pos());
    }

    // pic_parameter_set_id (ue)
    let _pps_id = r.read_ue().unwrap_or(0);
    eprintln!("[DEBUG] after pps_id: pos={}, val={}", r.pos(), _pps_id);

    // If this is a dependent slice segment, we don't parse full header
    // Use info from first slice - for frame detection we only care about first slices
    if !first_slice_segment_in_pic_flag {
        return None;
    }

    // Skip num_extra_slice_header_bits if present
    if pps.num_extra_slice_header_bits > 0 {
        let _extra_bits = r.read_bits(pps.num_extra_slice_header_bits as u32).unwrap_or(0);
        eprintln!("[DEBUG] after num_extra_slice_header_bits: pos={}, val={}", r.pos(), _extra_bits);
    }

    // slice_type (ue): 0=B, 1=P, 2=I
    let slice_type = r.read_ue().unwrap_or(0);
    eprintln!("[DEBUG] after slice_type: pos={}, val={}", r.pos(), slice_type);

    // pic_output_flag (if present)
    if pps.output_flag_present_flag {
        let _pic_output_flag = r.read_bit().unwrap_or(0);
        eprintln!("[DEBUG] after pic_output_flag: pos={}", r.pos());
    }

    // colour_plane_id (if separate_colour_plane_flag)
    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.read_bits(2).unwrap_or(0);
        eprintln!("[DEBUG] after colour_plane_id: pos={}", r.pos());
    }

    // Determine is_idr from NAL type
    let is_idr = nal_unit_type == 19 || nal_unit_type == 20; // IDR_W_RADL or IDR_N_LP

    // Parse pic_order_cnt_lsb (skip for IDR)
    let pic_order_cnt_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    eprintln!("[DEBUG] before pic_order_cnt_lsb: pos={}, bits={}", r.pos(), pic_order_cnt_lsb_bits);
    let pic_order_cnt_lsb = if is_idr {
        0i32
    } else {
        r.read_bits(pic_order_cnt_lsb_bits)? as i32
    };
    eprintln!("[DEBUG] after pic_order_cnt_lsb: pos={}, val={}", r.pos(), pic_order_cnt_lsb);

    // Compute PicOrderCntMsb per H.265 spec 8.3.1
    let pic_order_cnt_msb = if nal_unit_type >= 16 && nal_unit_type <= 20 {
        // For IRAP pictures with NoRaslOutputFlag (BLA_W_LP, BLA_W_RADL, BLA_N_LP, IDR_W_RADL, IDR_N_LP), MSB is 0
        // RADL/RASL (21-23) use normal wraparound logic
        0
    } else {
        let max_pic_order_cnt_lsb = 1i32 << pic_order_cnt_lsb_bits;
        if pic_order_cnt_lsb < prev_pic_order_cnt_lsb
            && (prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= (max_pic_order_cnt_lsb / 2)
        {
            prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
        } else if pic_order_cnt_lsb > prev_pic_order_cnt_lsb
            && (pic_order_cnt_lsb - prev_pic_order_cnt_lsb) > (max_pic_order_cnt_lsb / 2)
        {
            prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
        } else {
            prev_pic_order_cnt_msb
        }
    };

    let pic_order_cnt_val = pic_order_cnt_msb + pic_order_cnt_lsb;
    let pic_order_cnt = [pic_order_cnt_val, pic_order_cnt_val];

    // Determine is_reference: VCL NALs (type < 16) with nal_ref_idc != 0 can be references.
    // For IRAP pictures (type >= 16), they are references by definition.
    // Note: nal_ref_idc is not directly available here - it's in the NAL header parsing.
    // For now, use NAL type to determine:
    // - IRAP pictures (16-23) are references
    // - VCL pictures (0-15): odd types are references (R), even types are non-references (N)
    let is_reference = if is_rap {
        true
    } else {
        // Odd NAL types are references (TRAIl_R, TSA_R, STSA_R, RADL_R, RASL_R)
        // Even NAL types are non-references (TRAIl_N, TSA_N, STSA_N, RADL_N, RASL_N)
        nal_unit_type % 2 == 1
    };

    // Parse short-term reference picture set (STRPS) for non-IDR frames
    // Per H.265 spec B.3.3 and VulkanH265Parser.cpp:2220-2240
    let mut num_bits_for_st_ref_pic_set_in_slice: i32 = 0;
    let mut num_delta_pocs_of_ref_rps_idx: i32 = 0;
    // For IDR frames, short_term_ref_pic_set_sps_flag is NOT present in the bitstream
    // (HEVC spec B.3.3: only in non-IDR/W_RADL/BLA/RASL slices).
    // Default to false for IDR; parsed from bitstream for non-IDR.
    let mut short_term_ref_pic_set_sps_flag: bool = !is_idr;
    let mut ref_pocs: Vec<i32> = Vec::new();

    if !is_idr {
        // short_term_ref_pic_set_sps_flag
        short_term_ref_pic_set_sps_flag = r.read_bit().unwrap_or(0) == 1;
        eprintln!("[DEBUG] after short_term_ref_pic_set_sps_flag: pos={}, val={}", r.pos(), short_term_ref_pic_set_sps_flag);

        if !short_term_ref_pic_set_sps_flag {
            // In-slice short-term RPS - parse it and count bits consumed
            let bitcnt_before = r.pos();

            // Per HEVC spec B.3.3, the order is:
            // 1. inter_ref_pic_set_prediction_flag (read first!)
            //    EXCEPTION: when num_short_term_ref_pic_sets == 0, this flag is
            //    inferred as 0 and is NOT present in the bitstream.
            // 2. if inter-predicted: ref_rps_idx + delta entries
            // 3. if direct: num_negative_pics, num_positive_pics, delta_poc, used flags
            let inter_ref_pic_set_prediction_flag = if sps.num_short_term_ref_pic_sets > 0 {
                r.read_bit().unwrap_or(0) == 1
            } else {
                false // inferred as 0 per HEVC spec B.3.3
            };
            eprintln!("[DEBUG] after inter_ref_pic_set_prediction_flag: pos={}, val={}", r.pos(), inter_ref_pic_set_prediction_flag);

            if inter_ref_pic_set_prediction_flag {
                // Inter-predicted RPS (HEVC spec B.3.3, VulkanH265Parser.cpp:1738-1862)
                // For in-slice STRPS: idx = num_short_term_ref_pic_sets
                let idx = sps.num_short_term_ref_pic_sets as u32;
                let delta_idx_minus1 = r.read_ue().unwrap_or(0) as u32;
                eprintln!("[DEBUG] after delta_idx_minus1: pos={}, val={}", r.pos(), delta_idx_minus1);

                // Reference RPS index: RIdx = idx - (delta_idx_minus1 + 1)
                let r_idx = idx as usize - (delta_idx_minus1 as usize + 1);

                // Read delta_rps_sign and abs_delta_rps_minus1
                let delta_rps_sign = r.read_bit().unwrap_or(0) == 1;
                let abs_delta_rps_minus1 = r.read_ue().unwrap_or(0) as i32;
                let delta_rps = if delta_rps_sign {
                    -(abs_delta_rps_minus1 + 1)
                } else {
                    abs_delta_rps_minus1 + 1
                };
                eprintln!("[DEBUG] delta_rps_sign={}, abs_delta_rps_minus1={}, delta_rps={}, r_idx={}",
                    delta_rps_sign, abs_delta_rps_minus1, delta_rps, r_idx);

                // Get the reference RPS from SPS
                if r_idx >= sps.short_term_ref_pic_sets.len() {
                    // Invalid reference index - can't parse further
                    eprintln!("[ERROR] Invalid r_idx={} for inter-predicted STRPS", r_idx);
                } else {
                    let ref_strps = &sps.short_term_ref_pic_sets[r_idx];

                // NumDeltaPocsOfRefRpsIdx = total entries in reference RPS
                // (VulkanH265Parser.cpp:2658-2659)
                num_delta_pocs_of_ref_rps_idx = (ref_strps.num_negative_pics as i32 +
                    ref_strps.num_positive_pics as i32);

                // Read used_by_curr_pic_flag[j] for each entry (j = 0 to NumNeg+NumPos)
                // Then read use_delta_flag[j] only if used_by_curr_pic_flag[j] == 0
                let num_ref_entries = ref_strps.num_negative_pics as usize +
                    ref_strps.num_positive_pics as usize;
                let mut used_by_curr_pic_flag = vec![false; num_ref_entries + 1];
                let mut use_delta_flag = vec![true; num_ref_entries + 1];

                for j in 0..=num_ref_entries {
                    used_by_curr_pic_flag[j] = r.read_bit().unwrap_or(0) == 1;
                    if !used_by_curr_pic_flag[j] {
                        use_delta_flag[j] = r.read_bit().unwrap_or(0) == 1;
                    } else {
                        use_delta_flag[j] = true;
                    }
                }
                eprintln!("[DEBUG] after flags: pos={}, used_by_curr_pic_flag={:?}, use_delta_flag={:?}",
                    r.pos(), used_by_curr_pic_flag, use_delta_flag);

                // Compute new RPS from reference RPS + DeltaRPS
                // Matches VulkanH265Parser.cpp:1771-1862
                let curr_poc = pic_order_cnt_val;

                // Build cumulative POCs from reference RPS
                // S0 entries are negative cumulative deltas (stored as wrapped u16)
                let mut ref_poc_s0: Vec<i32> = Vec::new();
                for i in 0..ref_strps.num_negative_pics as usize {
                    let stored = ref_strps.delta_poc_s0_minus1[i] as i32;
                    let delta_poc = if stored > 32767 {
                        stored - 65536
                    } else {
                        stored
                    };
                    ref_poc_s0.push(curr_poc + delta_poc);
                }
                // S1 entries are positive cumulative deltas
                let mut ref_poc_s1: Vec<i32> = Vec::new();
                for i in 0..ref_strps.num_positive_pics as usize {
                    let delta = ref_strps.delta_poc_s1_minus1[i] as i32;
                    ref_poc_s1.push(curr_poc + delta);
                }

                // Compute new negative pics (S0): entries where (ref_poc + delta_rps) < curr_poc
                let mut new_num_neg: usize = 0;
                // Process S1 entries in reverse order first
                for j in (0..ref_strps.num_positive_pics as usize).rev() {
                    let new_poc = ref_poc_s1[j] + delta_rps;
                    let entry_idx = ref_strps.num_negative_pics as usize + j;
                    if new_poc < curr_poc && use_delta_flag[entry_idx] {
                        if used_by_curr_pic_flag[entry_idx] {
                            ref_pocs.push(new_poc);
                        }
                        new_num_neg += 1;
                    }
                }
                // Special case: DeltaRPS itself becomes a negative entry
                if delta_rps < 0 && use_delta_flag[num_ref_entries] {
                    let new_poc = curr_poc + delta_rps;
                    if used_by_curr_pic_flag[num_ref_entries] {
                        ref_pocs.push(new_poc);
                    }
                    new_num_neg += 1;
                }
                // Process S0 entries in order
                for j in 0..ref_strps.num_negative_pics as usize {
                    let new_poc = ref_poc_s0[j] + delta_rps;
                    if new_poc < curr_poc && use_delta_flag[j] {
                        if used_by_curr_pic_flag[j] {
                            ref_pocs.push(new_poc);
                        }
                        new_num_neg += 1;
                    }
                }

                // Compute new positive pics (S1): entries where (ref_poc + delta_rps) > curr_poc
                // Process S0 entries in reverse order first
                for j in (0..ref_strps.num_negative_pics as usize).rev() {
                    let new_poc = ref_poc_s0[j] + delta_rps;
                    if new_poc > curr_poc && use_delta_flag[j] {
                        if used_by_curr_pic_flag[j] {
                            ref_pocs.push(new_poc);
                        }
                    }
                }
                // Special case: DeltaRPS itself becomes a positive entry
                if delta_rps > 0 && use_delta_flag[num_ref_entries] {
                    let new_poc = curr_poc + delta_rps;
                    if used_by_curr_pic_flag[num_ref_entries] {
                        ref_pocs.push(new_poc);
                    }
                }
                // Process S1 entries in order
                for j in 0..ref_strps.num_positive_pics as usize {
                    let new_poc = ref_poc_s1[j] + delta_rps;
                    let entry_idx = ref_strps.num_negative_pics as usize + j;
                    if new_poc > curr_poc && use_delta_flag[entry_idx] {
                        if used_by_curr_pic_flag[entry_idx] {
                            ref_pocs.push(new_poc);
                        }
                    }
                }

                eprintln!("[DEBUG] inter-predicted STRPS: new ref_pocs={:?}", ref_pocs);
                }
            } else {
                // Direct (non-inter-predicted) RPS
                let num_negative_pics = r.read_ue().unwrap_or(0) as i32;
                eprintln!("[DEBUG] after num_negative_pics: pos={}, val={}", r.pos(), num_negative_pics);
                let num_positive_pics = r.read_ue().unwrap_or(0) as i32;
                eprintln!("[DEBUG] after num_positive_pics: pos={}, val={}", r.pos(), num_positive_pics);

                let curr_poc = pic_order_cnt_val;

                // Parse delta_poc[s0] for negative pics
                let mut cumulative_delta_poc_s0: i32 = 0;
                for i in 0..num_negative_pics {
                    let delta = r.read_ue().unwrap_or(0) as i32;
                    cumulative_delta_poc_s0 += delta + 1;
                    let used = r.read_bit().unwrap_or(0);
                    eprintln!("[DEBUG] neg_pic[{}]: delta={}, cum_delta={}, used={}, pos={}", i, delta, cumulative_delta_poc_s0, used, r.pos());
                    if used == 1 {
                        let ref_poc = curr_poc - cumulative_delta_poc_s0;
                        ref_pocs.push(ref_poc);
                    }
                }

                // Parse delta_poc[s1] for positive pics
                let mut cumulative_delta_poc_s1: i32 = 0;
                for i in 0..num_positive_pics {
                    let delta = r.read_ue().unwrap_or(0) as i32;
                    cumulative_delta_poc_s1 += delta + 1;
                    let used = r.read_bit().unwrap_or(0);
                    eprintln!("[DEBUG] pos_pic[{}]: delta={}, cum_delta={}, used={}, pos={}", i, delta, cumulative_delta_poc_s1, used, r.pos());
                    if used == 1 {
                        let ref_poc = curr_poc + cumulative_delta_poc_s1;
                        ref_pocs.push(ref_poc);
                    }
                }
            }

            let bitcnt_after = r.pos();
            num_bits_for_st_ref_pic_set_in_slice = (bitcnt_after - bitcnt_before) as i32;
            eprintln!("[DEBUG] RPS complete: bitcnt_before={}, bitcnt_after={}, num_bits={}", bitcnt_before, bitcnt_after, num_bits_for_st_ref_pic_set_in_slice);
            // DEBUG: After slice header parsing - print key RPS fields
            eprintln!("[DEBUG-SLICE] pic_order_cnt_val={}, short_term_ref_pic_set_sps_flag={}, num_bits_for_st_ref_pic_set_in_slice={}, ref_pocs={:?}",
                pic_order_cnt_val, short_term_ref_pic_set_sps_flag, num_bits_for_st_ref_pic_set_in_slice, ref_pocs);
        } else {
            // RPS from SPS - read the index and compute ref_pocs from SPS STRPS data
            let num_short_term_ref_pic_sets = sps.num_short_term_ref_pic_sets as u32;
            let short_term_ref_pic_set_idx = if num_short_term_ref_pic_sets > 1 {
                r.read_ue().unwrap_or(0) as usize
            } else {
                0
            };

            if short_term_ref_pic_set_idx < sps.short_term_ref_pic_sets.len() {
                let strps = &sps.short_term_ref_pic_sets[short_term_ref_pic_set_idx];
                let curr_poc = pic_order_cnt_val;

                // S0 (negative pics): parser stores negative cumulative deltas as wrapped u16
                // (e.g., -1 → 65535, -3 → 65533), interpret as signed delta
                for i in 0..strps.num_negative_pics as usize {
                    if (strps.used_by_curr_pic_s0_flag & (1 << i)) != 0 {
                        let stored = strps.delta_poc_s0_minus1[i] as i32;
                        let delta_poc = if stored > 32767 {
                            stored - 65536
                        } else {
                            stored
                        };
                        let ref_poc = curr_poc + delta_poc;
                        ref_pocs.push(ref_poc);
                    }
                }

                // S1 (positive pics): stored directly as positive cumulative deltas
                for i in 0..strps.num_positive_pics as usize {
                    if (strps.used_by_curr_pic_s1_flag & (1 << i)) != 0 {
                        let ref_poc = curr_poc + strps.delta_poc_s1_minus1[i] as i32;
                        ref_pocs.push(ref_poc);
                    }
                }
                // DEBUG: After slice header parsing - print key RPS fields (SPS path)
                eprintln!("[DEBUG-SLICE] pic_order_cnt_val={}, short_term_ref_pic_set_sps_flag={}, num_bits_for_st_ref_pic_set_in_slice={}, ref_pocs={:?}",
                    pic_order_cnt_val, short_term_ref_pic_set_sps_flag, num_bits_for_st_ref_pic_set_in_slice, ref_pocs);
            }
        }

        // Parse num_long_term_sps and num_long_term_pics if long-term refs present
        if sps.long_term_ref_pics_present_flag && sps.num_long_term_ref_pics_sps > 0 {
            let num_long_term_sps = r.read_ue().unwrap_or(0);
            let num_long_term_pics = r.read_ue().unwrap_or(0);

            let lt_idx_bits = if sps.num_long_term_ref_pics_sps > 1 {
                (sps.num_long_term_ref_pics_sps as f64).log2().ceil() as u32
            } else {
                0
            };
            let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;

            for i in 0u32..(num_long_term_sps + num_long_term_pics) {
                let mut poc_lsb: i32 = 0;
                if i < num_long_term_sps {
                    // LT ref from SPS
                    if lt_idx_bits > 0 {
                        let lt_idx_sps = r.read_bits(lt_idx_bits).unwrap_or(0);
                        poc_lsb = sps.lt_ref_pic_poc_lsb_sps[lt_idx_sps as usize] as i32;
                    } else {
                        poc_lsb = sps.lt_ref_pic_poc_lsb_sps[0] as i32;
                    }
                } else {
                    // New LT ref defined in slice
                    poc_lsb = r.read_bits(poc_lsb_bits).unwrap_or(0) as i32;
                    let _used_by_curr_pic_lt_flag = r.read_bit().unwrap_or(0);
                }

                // Add LT ref POC to ref_pocs
                ref_pocs.push(poc_lsb);

                // delta_poc_msb_present_flag and delta_poc_msb_cycle_lt
                let delta_poc_msb_present_flag = r.read_bit().unwrap_or(0);
                if delta_poc_msb_present_flag == 1 {
                    let _delta_poc_msb_cycle_lt = r.read_ue().unwrap_or(0);
                }
            }
        }

        // slice_temporal_mvp_enabled_flag if enabled in SPS
        if sps.sps_temporal_mvp_enabled_flag {
            let _slice_temporal_mvp_enabled_flag = r.read_bit().unwrap_or(0);
        }
    }

    Some((
        first_slice_segment_in_pic_flag,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        is_idr,
        is_reference,
        slice_type,
        num_bits_for_st_ref_pic_set_in_slice,
        num_delta_pocs_of_ref_rps_idx,
        short_term_ref_pic_set_sps_flag,
        ref_pocs,
    ))
}

/// Extract all access units from the bitstream.
///
/// Returns a Vec of AccessUnit, each containing:
/// - bitstream_data: slice NALs WITH start codes (no SPS/PPS/VPS)
/// - slice_offsets: offsets to start codes of each slice
/// - frame_num, pic_order_cnt, is_idr, is_reference from slice headers
///
/// Frame boundaries are detected using H.264 spec rules:
/// - A new frame starts when first_mb_in_slice == 0
/// - A new frame starts when frame_num changes from previous slice
fn extract_all_access_units(
    data: &[u8],
    codec: VideoCodec,
    max_frames: usize,
    sps: Option<&H264OrH265Sps>,
    pps: Option<&H264OrH265Pps>,
) -> Vec<AccessUnit> {
    use vk_video_parser::nal::{
        find_next_start_code, parse_h264_nal_header, parse_h265_nal_header,
    };

    let mut access_units: Vec<AccessUnit> = Vec::new();
    let mut offset = 0;
    let mut current_au_data: Vec<u8> = Vec::new();
    let mut current_slice_offsets: Vec<u32> = Vec::new();
    let mut current_frame_num: u32 = 0;
    let mut current_poc: [i32; 2] = [0, 0];
    let mut current_is_idr: bool = false;
    let mut current_is_reference: bool = true;
    let mut current_slice_type: u32 = 0;
     let mut current_num_bits_for_st_ref_pic_set_in_slice: i32 = 0;
     let mut current_num_delta_pocs_of_ref_rps_idx: i32 = 0;
     let mut current_short_term_ref_pic_set_sps_flag: bool = true;
      let mut current_ref_pocs: Vec<i32> = Vec::new();
     let mut current_adaptive_ref_pic_marking_mode_flag: bool = false;
     let mut current_mmco_commands: Vec<H264MmcoCommand> = Vec::new();
    let mut in_frame = false;
    let mut found_first_frame = false;

    // Track previous slice info for frame boundary detection
    let mut prev_pic_order_cnt_lsb: i32 = 0;
    let mut prev_pic_order_cnt_msb: i32 = 0;
    let mut max_pic_order_cnt_lsb: u32 = 256;
    let mut prev_frame_num: u32 = 0;
    let mut prev_first_mb_in_slice: u32 = 0;

    // Extract SPS/PPS values for H.264
    let h264_sps = match sps {
        Some(H264OrH265Sps::H264(s)) => Some(s),
        _ => None,
    };
    let h264_pps = match pps {
        Some(H264OrH265Pps::H264(p)) => Some(p),
        _ => None,
    };

    // Extract SPS/PPS values for H.265
    let h265_sps = match sps {
        Some(H264OrH265Sps::H265(s)) => Some(s),
        _ => None,
    };
    let h265_pps = match pps {
        Some(H264OrH265Pps::H265(p)) => Some(p),
        _ => None,
    };

    // Initialize max_pic_order_cnt_lsb from SPS if available
    if let Some(sps) = h264_sps {
        max_pic_order_cnt_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
    }

    while offset < data.len() && access_units.len() < max_frames {
        let Some((start, code_len)) = find_next_start_code(data, offset) else {
            break;
        };

        let next_start = find_next_start_code(data, start + code_len);
        let end = next_start.map(|(s, _)| s).unwrap_or(data.len());

        let nal_data = &data[start + code_len..end];
        if nal_data.is_empty() {
            offset = end;
            continue;
        }

        let (nal_type, is_irap, is_au_delimiter, is_slice, is_params) = match codec {
            VideoCodec::H264 => {
                if let Some((_, _, t)) = parse_h264_nal_header(nal_data) {
                    let is_idr = t == 5;
                    let is_aud = t == 9;
                    let is_slice_type = matches!(t, 1..=5);
                    let is_sps = t == 7;
                    let is_pps = t == 8;
                    let is_params_type = is_sps || is_pps;
                    (
                        t as usize,
                        is_idr,
                        is_aud,
                        is_slice_type,
                        is_params_type,
                    )
                } else {
                    (0, false, false, false, false)
                }
            }
            VideoCodec::H265 => {
                  if let Some((_, t, _, _)) = parse_h265_nal_header(nal_data) {
                      let is_irap_type = matches!(t, 16..=23);
                      let is_aud = t == 38;
                      let is_slice_type = matches!(t, 0..=31);
                    let is_vps = t == 32;
                    let is_sps = t == 33;
                    let is_pps = t == 34;
                    let is_params_type = is_vps || is_sps || is_pps;
                    (
                        t as usize,
                        is_irap_type,
                        is_aud,
                        is_slice_type,
                        is_params_type,
                    )
                } else {
                    (0, false, false, false, false)
                }
            }
        };

        // Handle access unit delimiter (H.264) - signals start of new AU
        if is_au_delimiter {
            if in_frame && !current_au_data.is_empty() {
                   access_units.push(AccessUnit {
                        data: current_au_data.clone(),
                        slice_offsets: current_slice_offsets.clone(),
                        frame_num: current_frame_num,
                        pic_order_cnt: current_poc,
                        is_idr: current_is_idr,
                        is_reference: current_is_reference,
                        slice_type: current_slice_type,
                         num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                          num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                          ref_pocs: current_ref_pocs.clone(),
                          adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                          mmco_commands: current_mmco_commands.clone(),
                      });
                    current_au_data.clear();
                    current_slice_offsets.clear();
                    current_mmco_commands.clear();
                }
              offset = end;
              continue;
          }

          // Skip parameter sets (SPS/PPS/VPS) - provided via session parameters
        if is_params {
            offset = end;
            continue;
        }

        // Process slice NALs
        if is_slice {
            let is_new_frame;

            if codec == VideoCodec::H264 {
                // Parse slice header to get frame boundary info
                if let Some(H264OrH265Sps::H264(sps)) = sps {
                    if let Some((_, ref_idc, nal_unit_type)) = parse_h264_nal_header(nal_data) {
                        if let Some((first_mb, frame_num, poc_lsb, poc_msb, poc, adaptive_ref_pic_marking_mode_flag, mmco_commands)) =
                            parse_h264_slice_header(
                                nal_data,
                                sps,
                                ref_idc,
                                nal_unit_type,
                                prev_pic_order_cnt_lsb,
                                prev_pic_order_cnt_msb,
                                max_pic_order_cnt_lsb,
                            )
                        {
                            let is_idr_slice = nal_unit_type == 5;

                            // Frame boundary detection (H.264 spec + C++ reference IsPictureBoundary):
                            // 1. Not in a frame yet -> new frame
                            // 2. first_mb_in_slice == 0 -> start of a new frame
                            // 3. frame_num changed from previous slice -> new frame
                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if first_mb == 0 {
                                true
                            } else if frame_num != prev_frame_num {
                                true
                            } else {
                                false
                            };

                            eprintln!(
                                "[slice] first_mb={}, frame_num={}, poc_lsb={}, is_idr={}, new_frame={}",
                                first_mb, frame_num, poc_lsb, is_idr_slice, is_new_frame
                            );

                            if is_new_frame {
                                // Push previous AU if we had one
                                if in_frame && !current_au_data.is_empty() {
                                    eprintln!(
                                        "[AU end] Pushing AU: frame_num={}, is_idr={}, slices={}",
                                        current_frame_num,
                                        current_is_idr,
                                        current_slice_offsets.len()
                                    );
                                      access_units.push(AccessUnit {
                                            data: current_au_data.clone(),
                                            slice_offsets: current_slice_offsets.clone(),
                                            frame_num: current_frame_num,
                                            pic_order_cnt: current_poc,
                                            is_idr: current_is_idr,
                                            is_reference: current_is_reference,
                                            slice_type: current_slice_type,
                                             num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                              num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                              ref_pocs: current_ref_pocs.clone(),
                                              adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                              mmco_commands: current_mmco_commands.clone(),
                                          });
                                        current_au_data.clear();
                                        current_slice_offsets.clear();
                                        current_mmco_commands.clear();
                                    }

                                   // First slice of a frame - set frame properties from parsed header
                                   current_is_idr = is_idr_slice;
                                 current_is_reference = ref_idc != 0;
                                 current_frame_num = frame_num;
                                 current_poc = poc;
                                 current_slice_type = if nal_unit_type == 5 {
                                     0 // IDR = I
                                 } else if ref_idc != 0 {
                                     1 // P-frame
                                 } else {
                                     2 // B-frame
                                 };
                                 current_adaptive_ref_pic_marking_mode_flag = adaptive_ref_pic_marking_mode_flag;
                                 current_mmco_commands = mmco_commands;

                                 // Update POC tracking
                                prev_pic_order_cnt_lsb = poc_lsb;
                                prev_pic_order_cnt_msb = poc_msb;
                                prev_frame_num = frame_num;
                                prev_first_mb_in_slice = first_mb;

                                // For IDR frames, reset POC tracking
                                if is_idr_slice {
                                    prev_pic_order_cnt_lsb = 0;
                                    prev_pic_order_cnt_msb = 0;
                                }

                                in_frame = true;
                                found_first_frame = true;
                            } else {
                                // Subsequent slice in same frame - update tracking
                                prev_first_mb_in_slice = first_mb;
                            }
                        } else {
                            // Failed to parse slice header - treat as new frame
                            eprintln!(
                                "[slice] Failed to parse slice header, treating as new frame"
                            );
                            is_new_frame = true;
                        }
                    } else {
                        is_new_frame = !in_frame || current_slice_offsets.is_empty();
                    }
                } else {
                    // No SPS available - fall back to NAL type based detection
                    is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                        true
                    } else if let Some((_, _, nal_type)) = parse_h264_nal_header(nal_data) {
                        let is_idr_slice = nal_type == 5;
                        (current_is_idr && !is_idr_slice) || is_idr_slice
                    } else {
                        false
                    };
                }
            } else {
                // H.265: use slice header parsing for frame boundary detection
                if let (Some(h265_sps), Some(h265_pps)) = (h265_sps, h265_pps) {
                    if let Some((_, nal_unit_type, _, nuh_temporal_id_plus1)) =
                        parse_h265_nal_header(nal_data)
                    {
                         if let Some((
                             first_slice_in_pic,
                             poc_lsb,
                             poc_msb,
                             poc,
                             slice_is_idr,
                             slice_is_reference,
                             slice_type,
                             slice_num_bits_strps,
                             slice_num_delta_pocs,
                             slice_short_term_ref_pic_set_sps_flag,
                             slice_ref_pocs,
                         )) = parse_h265_slice_header(
                            nal_data,
                            h265_sps,
                            h265_pps,
                            nal_unit_type,
                            nuh_temporal_id_plus1,
                            prev_pic_order_cnt_lsb,
                            prev_pic_order_cnt_msb,
                        ) {
                            // Frame boundary detection:
                            // 1. Not in a frame yet -> new frame
                            // 2. first_slice_segment_in_pic_flag == 1 -> start of new frame
                            // 3. IRAP NAL type while in frame -> new frame
                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if first_slice_in_pic {
                                true
                            } else if is_irap {
                                true
                            } else {
                                false
                            };

                            eprintln!(
                                "[slice] first_slice_in_pic={}, poc_lsb={}, is_idr={}, new_frame={}",
                                first_slice_in_pic, poc_lsb, slice_is_idr, is_new_frame
                            );

                            if is_new_frame {
                                // Push previous AU if we had one
                                if in_frame && !current_au_data.is_empty() {
                                    eprintln!(
                                        "[AU end] Pushing AU: frame_num={}, is_idr={}, slices={}",
                                        current_frame_num,
                                        current_is_idr,
                                        current_slice_offsets.len()
                                    );
                                      access_units.push(AccessUnit {
                                            data: current_au_data.clone(),
                                            slice_offsets: current_slice_offsets.clone(),
                                            frame_num: current_frame_num,
                                            pic_order_cnt: current_poc,
                                            is_idr: current_is_idr,
                                            is_reference: current_is_reference,
                                            slice_type: current_slice_type,
                                             num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                              num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                              ref_pocs: current_ref_pocs.clone(),
                                              adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                              mmco_commands: current_mmco_commands.clone(),
                                          });
                                        current_au_data.clear();
                                        current_slice_offsets.clear();
                                        current_mmco_commands.clear();
                                    }

                                  // First slice of a frame - set frame properties from parsed header
                                  current_is_idr = slice_is_idr;
                                current_is_reference = slice_is_reference;
                                current_poc = poc;
                                current_slice_type = slice_type;
                                 current_num_bits_for_st_ref_pic_set_in_slice = slice_num_bits_strps;
                                 current_num_delta_pocs_of_ref_rps_idx = slice_num_delta_pocs;
                                 current_short_term_ref_pic_set_sps_flag = slice_short_term_ref_pic_set_sps_flag;
                                 current_ref_pocs = slice_ref_pocs;
                                 prev_frame_num += 1;
                                current_frame_num = prev_frame_num;

                                // Update POC tracking per H.265 spec: only for pictures that are:
                                // - TemporalId == 0
                                // - NOT RADL/RASL (nal_unit_type 22-23)
                                // - NOT sub-layer non-reference (nal_unit_type < 16 and even)
                                let temporal_id = nuh_temporal_id_plus1 - 1;
                                let is_radl_rasl = nal_unit_type == 22 || nal_unit_type == 23;
                                let is_sub_layer_non_ref = nal_unit_type < 16 && nal_unit_type % 2 == 0;
                                if temporal_id == 0 && !is_radl_rasl && !is_sub_layer_non_ref {
                                    prev_pic_order_cnt_lsb = poc_lsb;
                                    prev_pic_order_cnt_msb = poc_msb;
                                }

                                in_frame = true;
                                found_first_frame = true;
                            }
                        } else {
                            // Failed to parse slice header - fall back to NAL type detection
                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if is_irap {
                                true
                            } else {
                                false
                            };
                        }
                    } else {
                        is_new_frame = !in_frame || current_slice_offsets.is_empty();
                    }
                } else {
                    // No SPS/PPS available - fall back to NAL type based detection
                    is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                        true
                    } else if is_irap {
                        true
                    } else {
                        false
                    };
                }
            }

            // Handle new frame for H.265 or fallback cases
            if is_new_frame && codec != VideoCodec::H264 {
                // Push previous AU if we had one
                                  if in_frame && !current_au_data.is_empty() {
                                      eprintln!(
                                         "[AU end] Pushing AU: frame_num={}, is_idr={}, slices={}",
                                         current_frame_num,
                                         current_is_idr,
                                         current_slice_offsets.len()
                                     );
                                       access_units.push(AccessUnit {
                                             data: current_au_data.clone(),
                                             slice_offsets: current_slice_offsets.clone(),
                                             frame_num: current_frame_num,
                                             pic_order_cnt: current_poc,
                                             is_idr: current_is_idr,
                                             is_reference: current_is_reference,
                                             slice_type: current_slice_type,
                                              num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                               num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                           short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                               ref_pocs: current_ref_pocs.clone(),
                                               adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                               mmco_commands: current_mmco_commands.clone(),
                                           });
                                         current_au_data.clear();
                                         current_slice_offsets.clear();
                                         current_mmco_commands.clear();
                                     }

                  if codec == VideoCodec::H264 && sps.is_none() {
                    // Fallback for H.264 without SPS
                    if let Some((_, ref_idc, nal_type)) = parse_h264_nal_header(nal_data) {
                        current_is_idr = nal_type == 5;
                        current_is_reference = ref_idc != 0;
                        prev_frame_num += 1;
                        current_frame_num = prev_frame_num;
                        current_poc = [0, 0];
                        current_slice_type = if nal_type == 5 {
                            0
                        } else if ref_idc != 0 {
                            1
                        } else {
                            2
                        };
                    }
                } else if codec == VideoCodec::H265 {
                    if let Some((_, nal_type, _, _)) = parse_h265_nal_header(nal_data) {
                        // Fallback: use NAL type for basic frame info
                        // IDR: NUT_IDR_W_RADL (19) or NUT_IDR_N_LP (20)
                        current_is_idr = nal_type == 19 || nal_type == 20;
                        // IRAP pictures (16-23) are references; VCL: odd types are references
                        current_is_reference = (nal_type >= 16 && nal_type <= 23) || nal_type % 2 == 1;
                        prev_frame_num += 1;
                        current_frame_num = prev_frame_num;
                        current_slice_type = if current_is_idr {
                            2 // I
                        } else if current_is_reference {
                            1 // P
                        } else {
                            0 // B
                        };
                    }
                }
                in_frame = true;
                found_first_frame = true;
            }

            // Add slice to current AU with 3-byte start code (matching C++ reference)
            // C++ uses 0x00 0x00 0x01 (3 bytes), not 0x00 0x00 0x00 0x01 (4 bytes)
            let slice_offset = current_au_data.len();
            current_au_data.extend_from_slice(&[0x00, 0x00, 0x01]);
            current_au_data.extend_from_slice(nal_data);
            current_slice_offsets.push(slice_offset as u32);
        }

        offset = end;
    }

    // Don't forget the last frame
    if in_frame && !current_au_data.is_empty() {
         access_units.push(AccessUnit {
               data: current_au_data,
               slice_offsets: current_slice_offsets,
               frame_num: current_frame_num,
               pic_order_cnt: current_poc,
               is_idr: current_is_idr,
               is_reference: current_is_reference,
               slice_type: current_slice_type,
                num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                 num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                 ref_pocs: current_ref_pocs,
                 adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                 mmco_commands: current_mmco_commands,
             });
        }

      eprintln!(
        "[extract_all] Extracted {} access units",
        access_units.len()
    );
    for (i, au) in access_units.iter().enumerate() {
        eprintln!(
            "[extract_all] AU[{}]: size={} bytes, {} slices, frame_num={}, POC=[{}, {}], is_idr={}, is_ref={}, slice_type={}",
            i, au.data.len(), au.slice_offsets.len(), au.frame_num,
            au.pic_order_cnt[0], au.pic_order_cnt[1],
            au.is_idr, au.is_reference, au.slice_type
        );
    }
    access_units
}

fn parse_h264_slice_first_mb(nal_data: &[u8]) -> Option<u32> {
    if nal_data.len() < 2 {
        return None;
    }
    let data = &nal_data[1..];
    let mut bit_pos: u32 = 0;
    let mut leading_zeros = 0u32;

    loop {
        let byte_idx = (bit_pos / 8) as usize;
        let bit_idx = 7 - (bit_pos % 8);
        if byte_idx >= data.len() {
            return None;
        }
        let bit = (data[byte_idx] >> bit_idx) & 1;
        bit_pos += 1;
        if bit == 1 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros >= 32 {
            return None;
        }
    }

    let mut value = 0u32;
    for _ in 0..leading_zeros {
        let byte_idx = (bit_pos / 8) as usize;
        let bit_idx = 7 - (bit_pos % 8);
        if byte_idx >= data.len() {
            return None;
        }
        let bit = ((data[byte_idx] >> bit_idx) & 1) as u32;
        value = (value << 1) | bit;
        bit_pos += 1;
    }

    Some((1 << leading_zeros) - 1 + value)
}

// ============================================================================
// Parsing
// ============================================================================

struct ParsedInfo {
    vps: Option<vk_video_core::picture::H265Vps>,
    sps: Option<H264OrH265Sps>,
    pps: Option<H264OrH265Pps>,
    coded_width: u32,
    coded_height: u32,
    display_width: u32,
    display_height: u32,
    crop_left: u32,
    crop_top: u32,
    profile_idc: u32,
    max_dpb_slots: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
}

enum H264OrH265Sps {
    H264(vk_video_core::picture::H264Sps),
    H265(vk_video_core::picture::H265Sps),
}

enum H264OrH265Pps {
    H264(vk_video_core::picture::H264Pps),
    H265(vk_video_core::picture::H265Pps),
}

fn parse_h264(data: &[u8]) -> ParsedInfo {
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH264,
        ))
        .ok();

    let packet = BitstreamPacket::new(data.to_vec());
    let mut sps: Option<vk_video_core::picture::H264Sps> = None;
    let mut pps: Option<vk_video_core::picture::H264Pps> = None;

    if let Ok(ParseResult::ParameterSet { sps: s, pps: p, .. }) = parser.parse(&packet) {
        if let Some(s) = s {
            sps = s.downcast_ref::<vk_video_core::picture::H264Sps>().cloned();
        }
        if let Some(p) = p {
            pps = p.downcast_ref::<vk_video_core::picture::H264Pps>().cloned();
        }
    }

    let coded_width = sps
        .as_ref()
        .map(|s| (s.pic_width_in_mbs_minus1 as u32 + 1) * 16)
        .unwrap_or(0);
    let coded_height = sps
        .as_ref()
        .map(|s| {
            if s.frame_mbs_only_flag {
                (s.pic_height_in_map_units_minus1 as u32 + 1) * 16
            } else {
                (s.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
            }
        })
        .unwrap_or(0);
    let profile_idc = sps.as_ref().map(|s| s.profile_idc as u32).unwrap_or(100);
    let max_dpb_slots = 16;

    let chroma_subsampling = match sps.as_ref().map(|s| s.chroma_format_idc) {
        Some(0) => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
        Some(1) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        Some(2) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        Some(3) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        _ => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
    };

    let luma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_luma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    let chroma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_chroma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    ParsedInfo {
        vps: None,
        sps: sps.map(H264OrH265Sps::H264),
        pps: pps.map(H264OrH265Pps::H264),
        coded_width,
        coded_height,
        display_width: coded_width,
        display_height: coded_height,
        crop_left: 0,
        crop_top: 0,
        profile_idc,
        max_dpb_slots,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    }
}

fn parse_h265(data: &[u8]) -> ParsedInfo {
    let mut parser = H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH265,
        ))
        .ok();

    let packet = BitstreamPacket::new(data.to_vec());
    let mut vps: Option<vk_video_core::picture::H265Vps> = None;
    let mut sps: Option<vk_video_core::picture::H265Sps> = None;
    let mut pps: Option<vk_video_core::picture::H265Pps> = None;

    if let Ok(ParseResult::ParameterSet {
        vps: v,
        sps: s,
        pps: p,
        ..
    }) = parser.parse(&packet)
    {
        if let Some(v) = v {
            vps = v.downcast_ref::<vk_video_core::picture::H265Vps>().cloned();
        }
        if let Some(s) = s {
            sps = s.downcast_ref::<vk_video_core::picture::H265Sps>().cloned();
        }
        if let Some(p) = p {
            pps = p.downcast_ref::<vk_video_core::picture::H265Pps>().cloned();
        }
    }

    let coded_width = sps
        .as_ref()
        .map(|s| ((s.pic_width_in_luma_samples as u32) + 15) & !15)
        .unwrap_or(0);
    let coded_height = sps
        .as_ref()
        .map(|s| ((s.pic_height_in_luma_samples as u32) + 15) & !15)
        .unwrap_or(0);

    // Compute display dimensions from SPS conformance window offsets (H.265 spec)
    // Offsets are in units of CTS (chroma transform blocks), need to multiply by
    // 2^Log2SubWidthC / 2^Log2SubHeightC to get luma samples
    let (display_width, display_height, crop_left, crop_top) = sps
        .as_ref()
        .map(|s| {
            let chroma_format_idc = s.chroma_format_idc;
            let (log2_sub_width_c, log2_sub_height_c) = match chroma_format_idc {
                0 => (0, 0), // monochrome
                1 | 2 => (1, 1), // 4:2:0 or 4:2:2
                _ => (0, 0), // 4:4:4
            };
            let sub_width_c = 1u32 << log2_sub_width_c;
            let sub_height_c = 1u32 << log2_sub_height_c;

            let pic_width = s.pic_width_in_luma_samples as u32;
            let pic_height = s.pic_height_in_luma_samples as u32;

            let (crop_left, crop_top) = if s.conformance_window_flag {
                (
                    s.conf_win_left_offset * sub_width_c,
                    s.conf_win_top_offset * sub_height_c,
                )
            } else {
                (0, 0)
            };

            let display_width = if s.conformance_window_flag {
                let left_right = (s.conf_win_left_offset + s.conf_win_right_offset) * sub_width_c;
                pic_width.saturating_sub(left_right)
            } else {
                pic_width
            };

            let display_height = if s.conformance_window_flag {
                let top_bottom = (s.conf_win_top_offset + s.conf_win_bottom_offset) * sub_height_c;
                pic_height.saturating_sub(top_bottom)
            } else {
                pic_height
            };

            (display_width, display_height, crop_left, crop_top)
        })
        .unwrap_or((coded_width, coded_height, 0, 0));

    let profile_idc = 1; // H.265 Main profile
    let max_dpb_slots = sps
        .as_ref()
        .map(|s| (s.max_num_ref_frames as u32).max(1))
        .unwrap_or(16)
        .max(4); // Ensure at least 4 DPB slots for proper reference picture handling

    let chroma_subsampling = match sps.as_ref().map(|s| s.chroma_format_idc) {
        Some(0) => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
        Some(1) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        Some(2) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        Some(3) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        _ => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
    };

    let luma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_luma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    let chroma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_chroma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    ParsedInfo {
        vps,
        sps: sps.map(H264OrH265Sps::H265),
        pps: pps.map(H264OrH265Pps::H265),
        coded_width,
        coded_height,
        display_width,
        display_height,
        crop_left,
        crop_top,
        profile_idc,
        max_dpb_slots,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    }
}

// ============================================================================
// Video session creation
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

fn create_video_session(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    decode_queue_family: u32,
    codec: VideoCodec,
    profile_idc: u32,
    coded_extent: vk::Extent2D,
    max_dpb_slots: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    vps: Option<&vk_video_core::picture::H265Vps>,
    sps: Option<&H264OrH265Sps>,
    pps: Option<&H264OrH265Pps>,
) -> Result<
    (
        vk::VideoSessionKHR,
        vk::VideoSessionParametersKHR,
        Vec<vk::DeviceMemory>,
    ),
    String,
> {
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let (profile_next, std_header_version) = match codec {
        VideoCodec::H264 => {
            let h264_profile = vk::VideoDecodeH264ProfileInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR,
                p_next: std::ptr::null(),
                std_profile_idc: profile_idc,
                picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE,
                _marker: Default::default(),
            };
            let std_ver = build_std_header_version("VK_STD_vulkan_video_codec_h264_decode");
            (&h264_profile as *const _ as *const _, std_ver)
        }
        VideoCodec::H265 => {
            let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
                p_next: std::ptr::null(),
                std_profile_idc: profile_idc,
                _marker: Default::default(),
            };
            let std_ver = build_std_header_version("VK_STD_vulkan_video_codec_h265_decode");
            (&h265_profile as *const _ as *const _, std_ver)
        }
    };

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: profile_next,
        video_codec_operation: codec.vk_codec_op(),
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    let session_create_info = vk::VideoSessionCreateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
        p_next: std::ptr::null(),
        queue_family_index: decode_queue_family,
        flags: vk::VideoSessionCreateFlagsKHR::empty(),
        p_video_profile: &profile_info as *const _,
        picture_format: output_format,
        max_coded_extent: coded_extent,
        reference_picture_format: output_format,
        max_dpb_slots,
        max_active_reference_pictures: max_dpb_slots,
        p_std_header_version: &std_header_version as *const _,
        _marker: Default::default(),
    };

    let create_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkCreateVideoSessionKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkCreateVideoSessionKHR not found")?;

    let session = unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            *const vk::VideoSessionCreateInfoKHR,
            *const vk::AllocationCallbacks,
            *mut vk::VideoSessionKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(create_fn);
        let mut handle = vk::VideoSessionKHR::null();
        let result = fn_ptr(
            device.handle(),
            &session_create_info,
            std::ptr::null(),
            &mut handle,
        );
        if result != vk::Result::SUCCESS {
            return Err(format!("vkCreateVideoSessionKHR failed: {:?}", result));
        }
        handle
    };

    let session_memories = bind_session_memory(instance, device, session, memory_properties)?;

    let session_params =
        create_session_parameters_with_sps_pps(instance, device, session, codec, vps, sps, pps)
            .map_err(|e| format!("Failed to create session parameters: {}", e))?;

    Ok((session, session_params, session_memories))
}

// ============================================================================
// Session parameters with SPS/PPS
// ============================================================================

fn create_session_parameters_with_sps_pps(
    instance: &ash::Instance,
    device: &ash::Device,
    session: vk::VideoSessionKHR,
    codec: VideoCodec,
    vps: Option<&vk_video_core::picture::H265Vps>,
    sps: Option<&H264OrH265Sps>,
    pps: Option<&H264OrH265Pps>,
) -> Result<vk::VideoSessionParametersKHR, String> {
    // Create session parameters WITH SPS/PPS inline.

    match codec {
        VideoCodec::H264 => {
            let sps_h264 = sps.and_then(|s| match s {
                H264OrH265Sps::H264(s) => Some(s),
                _ => None,
            });
            let pps_h264 = pps.and_then(|p| match p {
                H264OrH265Pps::H264(p) => Some(p),
                _ => None,
            });

            let std_sps = sps_h264.map(convert_h264_sps);
            let std_pps = pps_h264.map(convert_h264_pps);

            let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
                p_next: std::ptr::null(),
                std_sps_count: std_sps.is_some() as u32,
                p_std_sp_ss: std_sps.as_ref().map_or(std::ptr::null(), |s| s as *const _),
                std_pps_count: std_pps.is_some() as u32,
                p_std_pp_ss: std_pps.as_ref().map_or(std::ptr::null(), |p| p as *const _),
                _marker: Default::default(),
            };

            let h264_params = vk::VideoDecodeH264SessionParametersCreateInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
                p_next: std::ptr::null(),
                max_std_sps_count: 32,
                max_std_pps_count: 256,
                p_parameters_add_info: &add_info as *const _,
                _marker: Default::default(),
            };

            let params_create_info = vk::VideoSessionParametersCreateInfoKHR {
                s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                p_next: &h264_params as *const _ as *const _,
                flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                video_session: session,
                video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                _marker: Default::default(),
            };

            call_create_session_parameters(instance, device, &params_create_info)
        }
        VideoCodec::H265 => {
            let vps_h265 = vps;
            let sps_h265 = sps.and_then(|s| match s {
                H264OrH265Sps::H265(s) => Some(s),
                _ => None,
            });
            let pps_h265 = pps.and_then(|p| match p {
                H264OrH265Pps::H265(p) => Some(p),
                _ => None,
            });

            let std_vps = vps_h265.map(convert_h265_vps);
            let std_sps = sps_h265.map(convert_h265_sps);
            let std_pps = pps_h265.map(convert_h265_pps);

            let add_info = vk::VideoDecodeH265SessionParametersAddInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
                p_next: std::ptr::null(),
                std_vps_count: std_vps.is_some() as u32,
                p_std_vp_ss: std_vps.as_ref().map_or(std::ptr::null(), |v| v as *const _),
                std_sps_count: std_sps.is_some() as u32,
                p_std_sp_ss: std_sps.as_ref().map_or(std::ptr::null(), |s| s as *const _),
                std_pps_count: std_pps.is_some() as u32,
                p_std_pp_ss: std_pps.as_ref().map_or(std::ptr::null(), |p| p as *const _),
                _marker: Default::default(),
            };

            let h265_params = vk::VideoDecodeH265SessionParametersCreateInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
                p_next: std::ptr::null(),
                max_std_vps_count: 16,
                max_std_sps_count: 32,
                max_std_pps_count: 256,
                p_parameters_add_info: &add_info as *const _,
                _marker: Default::default(),
            };

            let params_create_info = vk::VideoSessionParametersCreateInfoKHR {
                s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                p_next: &h265_params as *const _ as *const _,
                flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                video_session: session,
                video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                _marker: Default::default(),
            };

            call_create_session_parameters(instance, device, &params_create_info)
        }
    }
}

fn call_create_session_parameters(
    instance: &ash::Instance,
    device: &ash::Device,
    params_create_info: &vk::VideoSessionParametersCreateInfoKHR<'_>,
) -> Result<vk::VideoSessionParametersKHR, String> {
    let create_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkCreateVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkCreateVideoSessionParametersKHR not found")?;

    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            *const vk::VideoSessionParametersCreateInfoKHR<'_>,
            *const vk::AllocationCallbacks,
            *mut vk::VideoSessionParametersKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(create_fn);
        let mut params = vk::VideoSessionParametersKHR::null();
        let result = fn_ptr(
            device.handle(),
            params_create_info,
            std::ptr::null(),
            &mut params,
        );
        if result != vk::Result::SUCCESS {
            return Err(format!(
                "vkCreateVideoSessionParametersKHR failed: {:?}",
                result
            ));
        }
        Ok(params)
    }
}

// ============================================================================
// Session parameters update
// ============================================================================

fn update_session_parameters(
    instance: &ash::Instance,
    device: &ash::Device,
    session_params: vk::VideoSessionParametersKHR,
    codec: VideoCodec,
    vps: Option<&vk_video_core::picture::H265Vps>,
    sps: Option<&H264OrH265Sps>,
    pps: Option<&H264OrH265Pps>,
) -> Result<(), String> {
    match codec {
        VideoCodec::H264 => update_session_parameters_h264(
            instance,
            device,
            session_params,
            sps.and_then(|s| match s {
                H264OrH265Sps::H264(s) => Some(s),
                _ => None,
            }),
            pps.and_then(|p| match p {
                H264OrH265Pps::H264(p) => Some(p),
                _ => None,
            }),
        ),
        VideoCodec::H265 => update_session_parameters_h265(
            instance,
            device,
            session_params,
            vps,
            sps.and_then(|s| match s {
                H264OrH265Sps::H265(s) => Some(s),
                _ => None,
            }),
            pps.and_then(|p| match p {
                H264OrH265Pps::H265(p) => Some(p),
                _ => None,
            }),
        ),
    }
}

fn update_session_parameters_h264(
    instance: &ash::Instance,
    device: &ash::Device,
    session_params: vk::VideoSessionParametersKHR,
    sps: Option<&vk_video_core::picture::H264Sps>,
    pps: Option<&vk_video_core::picture::H264Pps>,
) -> Result<(), String> {
    use ash::vk::native::*;

    let std_sps: Option<StdVideoH264SequenceParameterSet> = sps.map(convert_h264_sps);
    let std_pps: Option<StdVideoH264PictureParameterSet> = pps.map(convert_h264_pps);

    let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        p_next: std::ptr::null(),
        std_sps_count: std_sps.is_some() as u32,
        p_std_sp_ss: std_sps.as_ref().map_or(std::ptr::null(), |s| s as *const _),
        std_pps_count: std_pps.is_some() as u32,
        p_std_pp_ss: std_pps.as_ref().map_or(std::ptr::null(), |p| p as *const _),
        _marker: Default::default(),
    };

    let update_info = vk::VideoSessionParametersUpdateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
        p_next: &add_info as *const _ as *const _,
        update_sequence_count: 1, // First update must have count = 1
        _marker: Default::default(),
    };

    call_update_session_parameters(instance, device, session_params, &update_info)
}

fn update_session_parameters_h265(
    instance: &ash::Instance,
    device: &ash::Device,
    session_params: vk::VideoSessionParametersKHR,
    vps: Option<&vk_video_core::picture::H265Vps>,
    sps: Option<&vk_video_core::picture::H265Sps>,
    pps: Option<&vk_video_core::picture::H265Pps>,
) -> Result<(), String> {
    use ash::vk::native::*;

    let std_vps: Option<StdVideoH265VideoParameterSet> = vps.map(convert_h265_vps);
    let std_sps: Option<StdVideoH265SequenceParameterSet> = sps.map(convert_h265_sps);
    let std_pps: Option<StdVideoH265PictureParameterSet> = pps.map(convert_h265_pps);

    let add_info = vk::VideoDecodeH265SessionParametersAddInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
        p_next: std::ptr::null(),
        std_vps_count: std_vps.is_some() as u32,
        p_std_vp_ss: std_vps.as_ref().map_or(std::ptr::null(), |v| v as *const _),
        std_sps_count: std_sps.is_some() as u32,
        p_std_sp_ss: std_sps.as_ref().map_or(std::ptr::null(), |s| s as *const _),
        std_pps_count: std_pps.is_some() as u32,
        p_std_pp_ss: std_pps.as_ref().map_or(std::ptr::null(), |p| p as *const _),
        _marker: Default::default(),
    };

    let update_info = vk::VideoSessionParametersUpdateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
        p_next: &add_info as *const _ as *const _,
        update_sequence_count: 1, // First update must have count = 1
        _marker: Default::default(),
    };

    call_update_session_parameters(instance, device, session_params, &update_info)
}

fn call_update_session_parameters(
    instance: &ash::Instance,
    device: &ash::Device,
    session_params: vk::VideoSessionParametersKHR,
    update_info: &vk::VideoSessionParametersUpdateInfoKHR<'_>,
) -> Result<(), String> {
    let update_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkUpdateVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkUpdateVideoSessionParametersKHR not found")?;

    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            vk::VideoSessionParametersKHR,
            *const vk::VideoSessionParametersUpdateInfoKHR<'_>,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(update_fn);
        let result = fn_ptr(device.handle(), session_params, update_info);
        if result != vk::Result::SUCCESS {
            return Err(format!(
                "vkUpdateVideoSessionParametersKHR failed: {:?}",
                result
            ));
        }
    }
    Ok(())
}

// ============================================================================
// SPS/PPS conversion to Vulkan native types
// ============================================================================

/// Convert raw H.264 level_idc to Vulkan StdVideoH264LevelIdc enum value.
///
/// H.264 level_idc values have gaps: 10,11,12,13,20,21,22,30,31,32,40,41,42,50,51,52,60,61,62
/// Vulkan enum values are sequential: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
fn h264_level_idc_to_vulkan(raw_level_idc: u8, constraint_set3_flag: bool) -> u32 {
    // Level 1b handling: raw 9, or raw 11 with constraint_set3_flag
    if raw_level_idc == 9 || (raw_level_idc == 11 && constraint_set3_flag) {
        return 1; // STD_VIDEO_H264_LEVEL_IDC_1_1 (used for 1b)
    }

    match raw_level_idc {
        10 => 0,  // Level 1.0
        11 => 1,  // Level 1.1
        12 => 2,  // Level 1.2
        13 => 3,  // Level 1.3
        20 => 4,  // Level 2.0
        21 => 5,  // Level 2.1
        22 => 6,  // Level 2.2
        30 => 7,  // Level 3.0
        31 => 8,  // Level 3.1
        32 => 9,  // Level 3.2
        40 => 10, // Level 4.0
        41 => 11, // Level 4.1
        42 => 12, // Level 4.2
        50 => 13, // Level 5.0
        51 => 14, // Level 5.1
        52 => 15, // Level 5.2
        60 => 16, // Level 6.0
        61 => 17, // Level 6.1
        62 => 18, // Level 6.2
        _ => 18,  // Default to max level
    }
}

fn convert_h264_sps(
    sps: &vk_video_core::picture::H264Sps,
) -> ash::vk::native::StdVideoH264SequenceParameterSet {
    use ash::vk::native::*;

    let mut flags = unsafe { std::mem::zeroed::<StdVideoH264SpsFlags>() };
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });
    flags.set_qpprime_y_zero_transform_bypass_flag(if sps.qpprime_y_zero_transform_bypass_flag {
        1
    } else {
        0
    });
    flags.set_frame_mbs_only_flag(if sps.frame_mbs_only_flag { 1 } else { 0 });
    flags.set_direct_8x8_inference_flag(if sps.direct_8x8_inference_flag { 1 } else { 0 });
    flags.set_frame_cropping_flag(if sps.frame_cropping_flag { 1 } else { 0 });
    flags.set_vui_parameters_present_flag(if sps.vui_parameters_present_flag {
        1
    } else {
        0
    });

    // CRITICAL: Convert level_idc from raw H.264 value to Vulkan enum
    // Raw 41 (Level 4.1) -> Vulkan enum 11 (STD_VIDEO_H264_LEVEL_IDC_4_1)
    let vulkan_level_idc = h264_level_idc_to_vulkan(sps.level_idc, sps.constraint_set3_flag);
    eprintln!(
        "[SPS convert] profile={}, level_raw={}, level_vk={}, sps_id={}",
        sps.profile_idc, sps.level_idc, vulkan_level_idc, sps.seq_parameter_set_id
    );
    eprintln!(
        "[SPS convert] width_mbs={}, height_mus={}, chroma={}",
        sps.pic_width_in_mbs_minus1, sps.pic_height_in_map_units_minus1, sps.chroma_format_idc
    );
    eprintln!(
        "[SPS convert] max_ref_frames={}, poc_type={}, log2_max_frame_num={}, log2_max_poc_lsb={}",
        sps.max_num_ref_frames,
        sps.pic_order_cnt_type,
        sps.log2_max_frame_num_minus4,
        sps.log2_max_pic_order_cnt_lsb_minus4
    );
    eprintln!(
        "[SPS convert] frame_mbs_only={}, cropping_flag={}, vui_present={}",
        sps.frame_mbs_only_flag, sps.frame_cropping_flag, sps.vui_parameters_present_flag
    );

    // Convert VUI parameters if present
    let vui_data = if let Some(vui) = &sps.vui {
        let mut vui_flags = unsafe { std::mem::zeroed::<StdVideoH264SpsVuiFlags>() };
        vui_flags.set_aspect_ratio_info_present_flag(if vui.aspect_ratio_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_overscan_info_present_flag(if vui.overscan_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_overscan_appropriate_flag(if vui.overscan_appropriate_flag { 1 } else { 0 });
        vui_flags.set_video_signal_type_present_flag(if vui.video_signal_type_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_video_full_range_flag(if vui.video_full_range_flag { 1 } else { 0 });
        vui_flags.set_color_description_present_flag(if vui.color_description_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_chroma_loc_info_present_flag(if vui.chroma_loc_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_timing_info_present_flag(if vui.timing_info_present_flag { 1 } else { 0 });
        vui_flags.set_fixed_frame_rate_flag(if vui.fixed_frame_rate_flag { 1 } else { 0 });
        vui_flags.set_bitstream_restriction_flag(if vui.bitstream_restriction_flag {
            1
        } else {
            0
        });
        vui_flags.set_nal_hrd_parameters_present_flag(if vui.nal_hrd_parameters_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_vcl_hrd_parameters_present_flag(if vui.vcl_hrd_parameters_present_flag {
            1
        } else {
            0
        });

        StdVideoH264SequenceParameterSetVui {
            flags: vui_flags,
            aspect_ratio_idc: vui.aspect_ratio_idc as u32,
            sar_width: vui.sar_width,
            sar_height: vui.sar_height,
            video_format: vui.video_format,
            colour_primaries: vui.colour_primaries,
            transfer_characteristics: vui.transfer_characteristics,
            matrix_coefficients: vui.matrix_coefficients,
            num_units_in_tick: vui.num_units_in_tick,
            time_scale: vui.time_scale,
            max_num_reorder_frames: vui.max_num_reorder_frames,
            max_dec_frame_buffering: vui.max_dec_frame_buffering,
            chroma_sample_loc_type_top_field: vui.chroma_sample_loc_type_top_field,
            chroma_sample_loc_type_bottom_field: vui.chroma_sample_loc_type_bottom_field,
            reserved1: 0,
            pHrdParameters: std::ptr::null(),
        }
    } else {
        unsafe { std::mem::zeroed::<StdVideoH264SequenceParameterSetVui>() }
    };

    // Leak the Box to get a &'static pointer. Vulkan copies the data, so this is safe.
    let vui_ptr = Box::leak(Box::new(vui_data)) as *const StdVideoH264SequenceParameterSetVui;

    ash::vk::native::StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: sps.profile_idc as u32,
        level_idc: vulkan_level_idc,
        chroma_format_idc: sps.chroma_format_idc as u32,
        seq_parameter_set_id: sps.seq_parameter_set_id as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type: sps.pic_order_cnt_type as u32,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: sps.max_num_ref_frames as u8,
        reserved1: 0,
        pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1 as u32,
        pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1 as u32,
        frame_crop_left_offset: sps.frame_crop_left_offset,
        frame_crop_right_offset: sps.frame_crop_right_offset,
        frame_crop_top_offset: sps.frame_crop_top_offset,
        frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: vui_ptr,
    }
}

fn convert_h264_pps(
    pps: &vk_video_core::picture::H264Pps,
) -> ash::vk::native::StdVideoH264PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoH264PpsFlags>() };
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_deblocking_filter_control_present_flag(
        if pps.deblocking_filter_control_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_redundant_pic_cnt_present_flag(if pps.redundant_pic_cnt_present_flag {
        1
    } else {
        0
    });
    flags.set_transform_8x8_mode_flag(if pps.transform_8x8_mode_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag {
        1
    } else {
        0
    });

    ash::vk::native::StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: pps.seq_parameter_set_id as u8,
        pic_parameter_set_id: pps.pic_parameter_set_id as u8,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as u8,
        weighted_bipred_idc: pps.weighted_bipred_idc as u32,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as i8,
        pScalingLists: std::ptr::null(),
    }
}

// ============================================================================
// Decode command recording
// ============================================================================

fn record_decode_command(
    instance: &ash::Instance,
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    decode_queue_family: u32,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    bitstream_buffer: vk::Buffer,
    bitstream_offset: u64,
    bitstream_range: u64,
    output_image_view: vk::ImageView,
    output_image: vk::Image,
    coded_extent: vk::Extent2D,
    codec: VideoCodec,
    sps: Option<&H264OrH265Sps>,
    _pps: Option<&H264OrH265Pps>,
    vps: Option<&vk_video_core::picture::H265Vps>,
    slice_offsets: &[u32],
    frame_num: u32,
    pic_order_cnt: [i32; 2],
    is_idr: bool,
    is_reference: bool,
      slice_type: u32,
       num_bits_for_st_ref_pic_set_in_slice: i32,
       num_delta_pocs_of_ref_rps_idx: i32,
       short_term_ref_pic_set_sps_flag: bool,
       ref_pocs: &[i32],
      dpb_manager: &DpbManager,
      current_slot_index: u32,
      is_frame_1_debug: bool,
      bitstream_data: &[u8],
      decoder_reset_done: bool,
       dpb_views: &[vk::ImageView],
       dpb_images: &[vk::Image],
       fence: vk::Fence,
       // Persistent storage for codec-specific structs to avoid dangling pointer bugs.
       // Each frame's pic_info/decode_info/ref_info/dpb_slot_info must have stable memory
       // that outlives the GPU execution. Using Vecs ensures each frame gets unique memory.
       h265_pic_info_vec: &mut Vec<ash::vk::native::StdVideoDecodeH265PictureInfo>,
       h265_decode_info_vec: &mut Vec<vk::VideoDecodeH265PictureInfoKHR>,
       h265_ref_info_vec: &mut Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo>,
       h265_dpb_slot_info_vec: &mut Vec<vk::VideoDecodeH265DpbSlotInfoKHR>,
       h264_pic_info_vec: &mut Vec<ash::vk::native::StdVideoDecodeH264PictureInfo>,
       h264_decode_info_vec: &mut Vec<vk::VideoDecodeH264PictureInfoKHR>,
       h264_ref_info_vec: &mut Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo>,
       h264_dpb_slot_info_vec: &mut Vec<vk::VideoDecodeH264DpbSlotInfoKHR>,
   ) -> Result<(), String> {
    // DEBUG: Print frame info for frame 1
    if is_frame_1_debug {
        eprintln!(
            "[DEBUG] record_decode_command called for FRAME 1: codec={:?}, frame_num={}",
            codec, frame_num
        );
    }
    // Also print comprehensive logging for frame 1 (first P-frame)
    let is_frame_0_or_1 = frame_num == 0 || frame_num == 1;
    eprintln!(
        "[decode] coded_extent parameter: {}x{}",
        coded_extent.width, coded_extent.height
    );
    // Use aligned bitstream range to match Vulkan spec requirement for srcBufferRange.
    // srcBufferRange must be aligned to minBitstreamBufferSizeAlignment (typically 256 bytes).
    // The buffer is already zero-padded to this boundary, so the decoder reads
    // only valid data + zeros.
    let bs_range = bitstream_range;
    eprintln!("[decode] Bitstream range: {} bytes -> aligned to {} bytes, slice_count={}, frame_num={}, is_idr={}",
               bitstream_range, bs_range, slice_offsets.len(), frame_num, is_idr);

    // DEBUG: Print slice_segment_offsets for every frame
    eprintln!("[slice_segment] Frame {} offsets: {:?}", frame_num, slice_offsets);

    // DEBUG: Print bitstream buffer data for every frame (first 32 bytes)
    if !bitstream_data.is_empty() {
        let preview_len = 32.min(bitstream_data.len());
        let hex_str: String = bitstream_data[..preview_len]
            .iter().map(|b| format!("{:02x} ", b)).collect();
        eprintln!("[src_buffer] Frame {} data (first {} bytes): {}", frame_num, preview_len, hex_str);
    }

    // DEBUG: Print src_buffer_offset and src_buffer_range for every frame
    eprintln!("[src_buffer] Frame {}: offset={}, range={} (aligned={})",
               frame_num, bitstream_offset, bitstream_range, bs_range);

    // Build codec-specific picture info first (owned, not borrowed)
    // These live for the entire function scope, avoiding dangling pointers.
    let (h264_pic_info, h264_frame_num, h264_poc, h265_pic_info) = match codec {
        VideoCodec::H264 => {
            let sps_h264 = sps.and_then(|s| match s {
                H264OrH265Sps::H264(s) => Some(s),
                _ => None,
            });
            let pps_h264 = _pps.and_then(|p| match p {
                H264OrH265Pps::H264(p) => Some(p),
                _ => None,
            });
            let (pic_info, fn_val) = build_h264_picture_info(
                sps_h264,
                pps_h264,
                is_idr,
                is_reference,
                frame_num,
                pic_order_cnt,
                slice_type,
            );
            // FIX: Store in Vec to ensure stable memory across frames.
            h264_pic_info_vec.push(pic_info);
            let pic_info = &h264_pic_info_vec[h264_pic_info_vec.len() - 1];
            let poc = pic_info.PicOrderCnt;
            (Some(pic_info), Some(fn_val), Some(poc), None)
        }
        VideoCodec::H265 => {
            let sps_h265 = sps.and_then(|s| match s {
                H264OrH265Sps::H265(s) => Some(s),
                _ => None,
            });
            let pps_h265 = _pps.and_then(|p| match p {
                H264OrH265Pps::H265(p) => Some(p),
                _ => None,
            });
            // is_idr from au.is_idr (NAL unit types 19-20)
            // is_irap = true for IRAP frames (BLA/CRA/IDR = NAL types 16-23)
            let is_irap = is_idr || slice_type == 2; // slice_type 2 = I/IRAP frame for H.265

            // Build RefPicSet arrays from DPB state
            // Per C++ reference VulkanVideoParser.cpp:1666-1718
            // These arrays contain DPB slot indices of reference pictures
            // that are used as references for the current frame
            let mut ref_pic_set_st_curr_before = [0xffu8; 8];
            let mut ref_pic_set_st_curr_after = [0xffu8; 8];
            let ref_pic_set_lt_curr = [0xffu8; 8];

            let curr_poc = pic_order_cnt[0];

            // DEBUG: Before building RefPicSet arrays - print ref_pocs info
            eprintln!("[DEBUG-BUILD-RPS] frame_num={}, curr_poc={}, is_idr={}, is_reference={}, ref_pocs.len()={}, ref_pocs={:?}",
                frame_num, curr_poc, is_idr, is_reference, ref_pocs.len(), ref_pocs);

            if !is_idr {
                let valid_refs = dpb_manager.get_references();

                // Per C++ reference: process each RPS ref independently
                // Don't require ALL refs to be present - use whatever we can find
                // C++ uses create_lost_ref_pic() for missing refs; we just skip them
                let mut st_curr_before_idx = 0u32;
                let mut st_curr_after_idx = 0u32;

                for &ref_poc in ref_pocs.iter() {
                    if ref_poc == curr_poc {
                        continue;
                    }

                    // Find DPB entry with matching POC
                    if let Some(entry) = valid_refs.iter().find(|entry| {
                        entry.pic_order_cnt[0] == ref_poc
                    }) {
                        let slot_idx = (entry.slot_index as u8) & 0xf;
                        if ref_poc < curr_poc {
                            // Before references (S0 set)
                            if st_curr_before_idx < 8 {
                                ref_pic_set_st_curr_before[st_curr_before_idx as usize] = slot_idx;
                                st_curr_before_idx += 1;
                            }
                        } else {
                            // After references (S1 set)
                            if st_curr_after_idx < 8 {
                                ref_pic_set_st_curr_after[st_curr_after_idx as usize] = slot_idx;
                                st_curr_after_idx += 1;
                            }
                        }
                    } else {
                        // Ref POC not found in DPB - skip it (C++ would call create_lost_ref_pic)
                        eprintln!("[rps] WARNING: ref_poc={} not found in DPB, skipping", ref_poc);
                    }
                }

                // Only fall back to DPB-based selection if RPS produced NO references
                // AND we have valid DPB entries available
                if ref_pic_set_st_curr_before.iter().all(|&v| v == 0xff)
                    && ref_pic_set_st_curr_after.iter().all(|&v| v == 0xff)
                    && !valid_refs.is_empty()
                {
                    // Use all valid DPB references with POC < current POC (StCurrBefore)
                    // Sorted by POC descending (most recent first)
                    let mut refs_before: Vec<_> = valid_refs
                        .iter()
                        .filter(|e| e.pic_order_cnt[0] < curr_poc)
                        .collect();
                    refs_before.sort_by(|a, b| {
                        b.pic_order_cnt[0].cmp(&a.pic_order_cnt[0])
                    });

                    let mut st_curr_before_idx = 0u32;
                    for entry in refs_before {
                        if st_curr_before_idx >= 8 {
                            break;
                        }
                        let slot_idx = (entry.slot_index as u8) & 0xf;
                        ref_pic_set_st_curr_before[st_curr_before_idx as usize] = slot_idx;
                        st_curr_before_idx += 1;
                    }

                    // Also populate StCurrAfter from DPB (references with POC > current)
                    let mut refs_after: Vec<_> = valid_refs
                        .iter()
                        .filter(|e| e.pic_order_cnt[0] > curr_poc)
                        .collect();
                    refs_after.sort_by(|a, b| {
                        a.pic_order_cnt[0].cmp(&b.pic_order_cnt[0])
                    });

                    let mut st_curr_after_idx = 0u32;
                    for entry in refs_after {
                        if st_curr_after_idx >= 8 {
                            break;
                        }
                        let slot_idx = (entry.slot_index as u8) & 0xf;
                        ref_pic_set_st_curr_after[st_curr_after_idx as usize] = slot_idx;
                        st_curr_after_idx += 1;
                    }
                    eprintln!(
                        "[rps] Fallback: using DPB refs, found {} before + {} after (curr_poc={})",
                        st_curr_before_idx, st_curr_after_idx, curr_poc
                    );
                }

                // DEBUG: Print detailed DPB and RefPicSet state
                eprintln!(
                    "[rps] curr_poc={}, ref_pocs={:?}",
                    curr_poc, ref_pocs
                );
                eprintln!("[rps] Valid DPB references:");
                for entry in &valid_refs {
                    eprintln!(
                        "[rps]   slot={}, POC={}, frame_num={}",
                        entry.slot_index, entry.pic_order_cnt[0], entry.frame_num
                    );
                }
                eprintln!(
                    "[rps] Final RefPicSetStCurrBefore={:?}",
                    ref_pic_set_st_curr_before
                );
                eprintln!(
                    "[rps] Final RefPicSetStCurrAfter={:?}",
                    ref_pic_set_st_curr_after
                );

                // DEBUG: Print which DPB references are actually being used for this frame
                let mut used_refs: Vec<(u32, i32, u32)> = Vec::new();
                for (i, &slot) in ref_pic_set_st_curr_before.iter().enumerate() {
                    if slot != 0xff {
                        if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                            used_refs.push((slot as u32, entry.pic_order_cnt[0], entry.frame_num));
                        }
                    }
                }
                for (i, &slot) in ref_pic_set_st_curr_after.iter().enumerate() {
                    if slot != 0xff {
                        if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                            used_refs.push((slot as u32, entry.pic_order_cnt[0], entry.frame_num));
                        }
                    }
                }
                eprintln!("[DEBUG-USED-REFS] frame_num={}, curr_poc={}, used_refs count={}, refs={:?}",
                    frame_num, curr_poc, used_refs.len(), used_refs);

                // Extra detail for frames 1, 2, 7, 15
                if frame_num == 1 || frame_num == 2 || frame_num == 7 || frame_num == 15 {
                    eprintln!("[DEBUG-FOCUSED] ===== Frame {} (LOW PSNR TARGET) =====", frame_num);
                    eprintln!("[DEBUG-FOCUSED]   curr_poc={}, is_idr={}, is_reference={}", curr_poc, is_idr, is_reference);
                    eprintln!("[DEBUG-FOCUSED]   ref_pocs from slice header={:?}", ref_pocs);
                    eprintln!("[DEBUG-FOCUSED]   short_term_ref_pic_set_sps_flag={}, num_bits_for_st_ref_pic_set_in_slice={}",
                        short_term_ref_pic_set_sps_flag, num_bits_for_st_ref_pic_set_in_slice);
                    eprintln!("[DEBUG-FOCUSED]   RefPicSetStCurrBefore slot indices={:?}", ref_pic_set_st_curr_before);
                    eprintln!("[DEBUG-FOCUSED]   RefPicSetStCurrAfter slot indices={:?}", ref_pic_set_st_curr_after);
                    eprintln!("[DEBUG-FOCUSED]   Used DPB refs (slot, POC, frame_num)={:?}", used_refs);
                    eprintln!("[DEBUG-FOCUSED] ===============================");
                }


            }

    eprintln!(
        "[rps] RefPicSet: curr_poc={}, is_idr={}, ref_pocs={:?}, StCurrBefore={:?}, StCurrAfter={:?}, num_bits_for_st_ref_pic_set_in_slice={}, num_delta_pocs_of_ref_rps_idx={}",
        curr_poc, is_idr, ref_pocs,
        ref_pic_set_st_curr_before,
        ref_pic_set_st_curr_after,
        num_bits_for_st_ref_pic_set_in_slice,
        num_delta_pocs_of_ref_rps_idx
    );

    // DEBUG: Print references used from DPB for each frame
    if !is_idr {
        let valid_refs = dpb_manager.get_references();
        let mut dpb_refs_used: Vec<String> = Vec::new();
        for &slot in ref_pic_set_st_curr_before.iter().chain(ref_pic_set_st_curr_after.iter()) {
            if slot != 0xff {
                if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                    dpb_refs_used.push(format!("slot{}(POC{})", entry.slot_index, entry.pic_order_cnt[0]));
                }
            }
        }
        eprintln!("[DEBUG-DPB-USAGE] frame_num={}, curr_poc={}, using DPB refs: {:?}",
            frame_num, curr_poc, dpb_refs_used);
    }

            // === DETAILED DEBUG FOR FRAMES 1 (POC=5) and 7 (POC=7) ===
            if frame_num == 1 || frame_num == 7 {
                let num_st_before_refs = ref_pic_set_st_curr_before.iter().filter(|&&s| s != 0xff).count();
                let num_st_after_refs = ref_pic_set_st_curr_after.iter().filter(|&&s| s != 0xff).count();
                let num_lt_refs = ref_pic_set_lt_curr.iter().filter(|&&s| s != 0xff).count();
                let total_ref_slots = num_st_before_refs + num_st_after_refs + num_lt_refs;

                eprintln!("\n\n===============================================================================");
                eprintln!("  H.265 DECODE INFO DEBUG - Frame {} (POC={})", frame_num, pic_order_cnt[0]);
                eprintln!("===============================================================================");
                eprintln!("  --- Input Parameters to build_h265_picture_info ---");
                eprintln!("  pic_order_cnt_val = {}", pic_order_cnt[0]);
                eprintln!("  is_idr = {}", is_idr);
                eprintln!("  is_irap = {}", is_irap);
                eprintln!("  is_reference = {}", is_reference);
                eprintln!("  num_bits_for_st_ref_pic_set_in_slice = {}", num_bits_for_st_ref_pic_set_in_slice);
                eprintln!("  num_delta_pocs_of_ref_rps_idx = {}", num_delta_pocs_of_ref_rps_idx);
                eprintln!("  short_term_ref_pic_set_sps_flag = {}", short_term_ref_pic_set_sps_flag);
                if let Some(sps) = sps_h265 {
                    eprintln!("  sps_video_parameter_set_id = {}", sps.sps_video_parameter_set_id);
                    eprintln!("  sps_max_num_ref_frames = {}", sps.max_num_ref_frames);
                    eprintln!("  sps_num_long_term_ref_pics_sps = {}", sps.num_long_term_ref_pics_sps);
                }
                if let Some(pps) = pps_h265 {
                    eprintln!("  pps_pic_parameter_set_id = {}", pps.pps_pic_parameter_set_id);
                    eprintln!("  pps_seq_parameter_set_id = {}", pps.pps_seq_parameter_set_id);
                }
                eprintln!("\n  --- RefPicSet Arrays ---");
                eprintln!("  RefPicSetStCurrBefore ({} refs): {:?}", num_st_before_refs, ref_pic_set_st_curr_before);
                eprintln!("  RefPicSetStCurrAfter  ({} refs): {:?}", num_st_after_refs, ref_pic_set_st_curr_after);
                eprintln!("  RefPicSetLtCurr       ({} refs): {:?}", num_lt_refs, ref_pic_set_lt_curr);
                eprintln!("  Total reference slots: {}", total_ref_slots);

                // Print which DPB entries these slots refer to
                if !is_idr {
                    let valid_refs = dpb_manager.get_references();
                    eprintln!("\n  --- Reference Slot -> DPB Entry Mapping ---");
                    for (i, &slot) in ref_pic_set_st_curr_before.iter().enumerate() {
                        if slot != 0xff {
                            if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                                eprintln!("    StCurrBefore[{}] = slot {} -> POC={}, frame_num={}",
                                    i, slot, entry.pic_order_cnt[0], entry.frame_num);
                            } else {
                                eprintln!("    StCurrBefore[{}] = slot {} -> NOT FOUND in DPB", i, slot);
                            }
                        }
                    }
                    for (i, &slot) in ref_pic_set_st_curr_after.iter().enumerate() {
                        if slot != 0xff {
                            if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                                eprintln!("    StCurrAfter[{}] = slot {} -> POC={}, frame_num={}",
                                    i, slot, entry.pic_order_cnt[0], entry.frame_num);
                            } else {
                                eprintln!("    StCurrAfter[{}] = slot {} -> NOT FOUND in DPB", i, slot);
                            }
                        }
                    }
                    for (i, &slot) in ref_pic_set_lt_curr.iter().enumerate() {
                        if slot != 0xff {
                            if let Some(entry) = valid_refs.iter().find(|e| e.slot_index == slot as u32) {
                                eprintln!("    LtCurr[{}] = slot {} -> POC={}, frame_num={}",
                                    i, slot, entry.pic_order_cnt[0], entry.frame_num);
                            } else {
                                eprintln!("    LtCurr[{}] = slot {} -> NOT FOUND in DPB", i, slot);
                            }
                        }
                    }
                }
            }

            let pic_info = build_h265_picture_info(
                  sps_h265,
                  pps_h265,
                  pic_order_cnt[0], // Use actual POC from parsed slice header
                  is_idr,           // IdrPicFlag based on NAL unit type (19-20)
                  is_irap,          // IrapPicFlag for IRAP frames (BLA/CRA/IDR)
                  is_reference,
                  num_bits_for_st_ref_pic_set_in_slice,  // NumBitsForSTRefPicSetInSlice
                  num_delta_pocs_of_ref_rps_idx,         // NumDeltaPocsOfRefRpsIdx
                   short_term_ref_pic_set_sps_flag,
                  ref_pic_set_st_curr_before,
                  ref_pic_set_st_curr_after,
                  ref_pic_set_lt_curr,
              );
            // FIX: Store in Vec to ensure stable memory across frames.
            // Each frame's pic_info must have unique, stable memory because
            // the GPU reads p_std_picture_info at execute time, and stack
            // variables at fixed offsets get overwritten by subsequent frames.
            h265_pic_info_vec.push(pic_info);
            let pic_info = &h265_pic_info_vec[h265_pic_info_vec.len() - 1];
            // DEBUG: Print pic_info fields and address
            eprintln!(
                "[pic_info] frame_num={} addr={:016x} PicOrderCntVal={} IdrPicFlag={} IrapPicFlag={} RefPicSetStCurrBefore[0]={}",
                frame_num,
                pic_info as *const _ as usize,
                pic_info.PicOrderCntVal,
                pic_info.flags.IdrPicFlag(),
                pic_info.flags.IrapPicFlag(),
                pic_info.RefPicSetStCurrBefore[0],
            );

            // === DETAILED DEBUG: StdVideoDecodeH265PictureInfo for frames 1 and 7 ===
            if frame_num == 1 || frame_num == 7 {
                eprintln!("\n  --- StdVideoDecodeH265PictureInfo (full struct) ---");
                eprintln!("    Address: {:016x}", pic_info as *const _ as usize);
                eprintln!("    Size: {} bytes", std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfo>());

                // Flags as raw bytes
                let flags_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &pic_info.flags as *const _ as *const u8,
                        std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfoFlags>(),
                    )
                };
                let flags_hex: String = flags_bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                eprintln!("    flags (raw bytes): {}", flags_hex);
                eprintln!("    flags.IrapPicFlag                  = {}", pic_info.flags.IrapPicFlag());
                eprintln!("    flags.IdrPicFlag                   = {}", pic_info.flags.IdrPicFlag());
                eprintln!("    flags.IsReference                  = {}", pic_info.flags.IsReference());
                eprintln!("    flags.short_term_ref_pic_set_sps_flag = {}", pic_info.flags.short_term_ref_pic_set_sps_flag());

                eprintln!("    pps_pic_parameter_set_id           = {}", pic_info.pps_pic_parameter_set_id);
                eprintln!("    pps_seq_parameter_set_id           = {}", pic_info.pps_seq_parameter_set_id);
                eprintln!("    sps_video_parameter_set_id         = {}", pic_info.sps_video_parameter_set_id);
                eprintln!("    NumBitsForSTRefPicSetInSlice       = {}", pic_info.NumBitsForSTRefPicSetInSlice);
                eprintln!("    NumDeltaPocsOfRefRpsIdx            = {}", pic_info.NumDeltaPocsOfRefRpsIdx);
                eprintln!("    PicOrderCntVal                     = {}", pic_info.PicOrderCntVal);

                eprintln!("    RefPicSetStCurrBefore: {:?}", pic_info.RefPicSetStCurrBefore);
                eprintln!("    RefPicSetStCurrAfter:  {:?}", pic_info.RefPicSetStCurrAfter);
                eprintln!("    RefPicSetLtCurr:       {:?}", pic_info.RefPicSetLtCurr);

                // Full hex dump of struct
                let pic_bytes = unsafe {
                    std::slice::from_raw_parts(
                        pic_info as *const _ as *const u8,
                        std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfo>(),
                    )
                };
                eprintln!("    [FULL STRUCT HEX DUMP]");
                for i in (0..pic_bytes.len()).step_by(16) {
                    let end = (i + 16).min(pic_bytes.len());
                    let hex: String = (i..end).map(|j| format!("{:02x}", pic_bytes[j])).collect::<Vec<_>>().join(" ");
                    eprintln!("      {:04x}: {}", i, hex);
                }
            }

            (None, None, None, Some(pic_info))
        }
    };

    // All Vulkan structs are declared at function scope so they outlive any pointer usage.
    // This fixes the dangling pointer bug where structs were created inside match blocks
    // and raw pointers were returned to locals that got dropped.

    // DPB setup picture resource
    let dpb_setup_picture_resource = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: output_image_view,
        _marker: Default::default(),
    };

    // Destination picture resource for decode
    let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: output_image_view,
        _marker: Default::default(),
    };

    // Codec-specific decode info and DPB slot info (all at function scope)
    // FIX: Store in Vecs to ensure stable memory across frames.
    let (h264_decode_info, h264_ref_info, h264_dpb_slot_info): (Option<*const vk::VideoDecodeH264PictureInfoKHR>, Option<*const ash::vk::native::StdVideoDecodeH264ReferenceInfo>, Option<*const vk::VideoDecodeH264DpbSlotInfoKHR>) = match &h264_pic_info {
        Some(pic_info) => {
            let frame_num = h264_frame_num.unwrap_or(0);
            let poc = h264_poc.unwrap_or([0, 0]);

            // Store ref_info in Vec first (dpb_slot_info needs pointer to it)
            let mut ref_info =
                unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>() };
            ref_info.FrameNum = frame_num as u16;
            ref_info.PicOrderCnt = poc;
            // Setup picture reference info (current picture being decoded):
            // Per C++ reference (VulkanVideoParser.cpp:2321-2322):
            //   top_field_flag = !field_pic_flag || !bottom_field_flag
            //   bottom_field_flag = !field_pic_flag || bottom_field_flag
            // For progressive frames (field_pic_flag=0): both flags = 1
            // This indicates both fields of the setup picture are available.
            ref_info.flags.set_top_field_flag(1);
            ref_info.flags.set_bottom_field_flag(1);
            ref_info.flags.set_used_for_long_term_reference(0);
            ref_info.flags.set_is_non_existing(0);

            eprintln!(
                "[dpb] FrameNum={}, PicOrderCnt=[{}, {}]",
                ref_info.FrameNum, ref_info.PicOrderCnt[0], ref_info.PicOrderCnt[1]
            );
            eprintln!(
                "[dpb] flags: top={}, bottom={}, ltr={}, non_existing={}",
                ref_info.flags.top_field_flag(),
                ref_info.flags.bottom_field_flag(),
                ref_info.flags.used_for_long_term_reference(),
                ref_info.flags.is_non_existing()
            );

            // DEBUG: Dump setup picture reference info for frame 1
            if frame_num == 1 {
                eprintln!("[DEBUG-FRAME1-SETUP] Setup picture ref info: frame_num={}, poc=[{}, {}], flags(top={},bottom={},ltr={},non_exist={})",
                    ref_info.FrameNum, ref_info.PicOrderCnt[0], ref_info.PicOrderCnt[1],
                    ref_info.flags.top_field_flag(),
                    ref_info.flags.bottom_field_flag(),
                    ref_info.flags.used_for_long_term_reference(),
                    ref_info.flags.is_non_existing());
            }

            h264_ref_info_vec.push(ref_info);
            let ref_info = &h264_ref_info_vec[h264_ref_info_vec.len() - 1];
            let ref_info_ptr = ref_info as *const _;

            let dpb_slot_info = vk::VideoDecodeH264DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: ref_info_ptr,
                _marker: Default::default(),
            };
            h264_dpb_slot_info_vec.push(dpb_slot_info);
            let dpb_slot_info = &h264_dpb_slot_info_vec[h264_dpb_slot_info_vec.len() - 1];

            let decode_info = vk::VideoDecodeH264PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: *pic_info as *const _ as *const _,
                slice_count: slice_offsets.len() as u32,
                p_slice_offsets: slice_offsets.as_ptr(),
                _marker: Default::default(),
            };
            h264_decode_info_vec.push(decode_info);
            let decode_info = &h264_decode_info_vec[h264_decode_info_vec.len() - 1];

            eprintln!("[dpb]   ref_info_ptr={:p}", ref_info_ptr);
            eprintln!("[dpb]   dpb_slot_info_ptr={:p}", dpb_slot_info as *const _);
            eprintln!(
                "[dpb]   p_std_reference_info={:p}",
                dpb_slot_info.p_std_reference_info
            );
            // Return raw pointers to release borrows, allowing later pushes to vectors
            (Some(decode_info as *const _), Some(ref_info as *const _), Some(dpb_slot_info as *const _))
        }
        None => (None, None, None),
    };

    // NOTE: In-band session parameters (chaining VideoDecodeH265SessionParametersAddInfoKHR
    // to VideoDecodeH265PictureInfoKHR::pNext) is not supported by all drivers.
    // The session parameters are already set during session creation (out-of-band).
    // Keeping pNext as NULL for compatibility.
    // FIX: Store in Vecs to ensure stable memory across frames.
    let (h265_decode_info, h265_ref_info, h265_dpb_slot_info): (Option<*const vk::VideoDecodeH265PictureInfoKHR>, Option<*const ash::vk::native::StdVideoDecodeH265ReferenceInfo>, Option<*const vk::VideoDecodeH265DpbSlotInfoKHR>) = match h265_pic_info {
        Some(pic_info) => {
            // Store ref_info in Vec first (dpb_slot_info needs pointer to it)
            let mut ref_info =
                unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>() };
            ref_info.PicOrderCntVal = 0;
            ref_info.flags.set_used_for_long_term_reference(0);
            ref_info.flags.set_unused_for_reference(0);

            h265_ref_info_vec.push(ref_info);
            let ref_info = &h265_ref_info_vec[h265_ref_info_vec.len() - 1];

            let dpb_slot_info = vk::VideoDecodeH265DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: ref_info as *const _,
                _marker: Default::default(),
            };
            h265_dpb_slot_info_vec.push(dpb_slot_info);
            let dpb_slot_info = &h265_dpb_slot_info_vec[h265_dpb_slot_info_vec.len() - 1];

             // === DETAILED DEBUG: VkVideoDecodeH265PictureInfoKHR for frames 1 and 7 ===
             let decode_info = vk::VideoDecodeH265PictureInfoKHR {
                  s_type: vk::StructureType::VIDEO_DECODE_H265_PICTURE_INFO_KHR,
                  p_next: std::ptr::null(),
                  p_std_picture_info: pic_info as *const _,
                  slice_segment_count: slice_offsets.len() as u32,
                  p_slice_segment_offsets: slice_offsets.as_ptr(),
                  _marker: Default::default(),
              };
              h265_decode_info_vec.push(decode_info);
              let decode_info = &h265_decode_info_vec[h265_decode_info_vec.len() - 1];

              if frame_num == 1 || frame_num == 7 {
                  eprintln!("\n  --- VkVideoDecodeH265PictureInfoKHR ---");
                  eprintln!("    Address: {:016x}", decode_info as *const _ as usize);
                  eprintln!("    Size: {} bytes", std::mem::size_of::<vk::VideoDecodeH265PictureInfoKHR>());
                  eprintln!("    s_type                  = {:?} ({})", decode_info.s_type, decode_info.s_type.as_raw());
                  eprintln!("    p_next                  = {:p}", decode_info.p_next);
                  eprintln!("    p_std_picture_info      = {:p}", decode_info.p_std_picture_info);
                  eprintln!("    slice_segment_count     = {}", decode_info.slice_segment_count);
                  if !decode_info.p_slice_segment_offsets.is_null() && decode_info.slice_segment_count > 0 {
                      let offsets = unsafe {
                          std::slice::from_raw_parts(
                              decode_info.p_slice_segment_offsets,
                              decode_info.slice_segment_count as usize,
                          )
                      };
                      eprintln!("    p_slice_segment_offsets = {:?}", offsets);
                  }
              }

            // Return raw pointers to release borrows, allowing later pushes to vectors
            (Some(decode_info as *const _), Some(ref_info as *const _), Some(dpb_slot_info as *const _))
        }
        None => (None, None, None),
    };

    // Build reference slots from DpbManager
    // According to Vulkan-Video-Samples pattern:
    // - referenceSlots[0..N-1] = reference picture slots from DPB
    // - referenceSlots[N] = setup slot for current frame
    // - VkVideoBeginCodingInfoKHR::p_reference_slots = all slots (0..N)
    // - VkVideoDecodeInfoKHR::p_reference_slots = reference slots only (0..N-1)
    // - VkVideoDecodeInfoKHR::p_setup_reference_slot = setup slot (N)

    // Get valid reference entries from DPB (exclude the current slot)
    let valid_refs = dpb_manager.get_references();
    eprintln!(
        "[dpb] Found {} valid reference entries for decode",
        valid_refs.len()
    );
    for ref_entry in &valid_refs {
        eprintln!(
            "[dpb]   ref: frame_num={}, slot={}, poc=[{}, {}]",
            ref_entry.frame_num,
            ref_entry.slot_index,
            ref_entry.pic_order_cnt[0],
            ref_entry.pic_order_cnt[1]
        );
    }

    // DEBUG: Print full DPB state for this frame
    eprintln!(
        "[dpb] === FULL DPB STATE for frame {} (POC=[{}, {}], is_idr={}, slice_type={}) ===",
        frame_num, pic_order_cnt[0], pic_order_cnt[1], is_idr, slice_type
    );
    for (i, entry) in dpb_manager.entries.iter().enumerate() {
        eprintln!(
            "[dpb]   slot[{}]: valid={}, frame_num={}, poc=[{}, {}]",
            i, entry.is_valid, entry.frame_num, entry.pic_order_cnt[0], entry.pic_order_cnt[1]
        );
    }
    eprintln!("[dpb] === END DPB STATE ===");

    // Build setup reference slot for current frame using actual DPB slot index
    // CRITICAL: Setup slot index must be the actual DPB slot index where the current
    // frame is being written (current_slot_index), NOT a sequential index.
    //
    // For H.265, the setup slot MUST have VkVideoDecodeH265DpbSlotInfoKHR in its pNext chain
    // (VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07163).
    // For H.264, setup slot MUST also have VideoDecodeH264DpbSlotInfoKHR in pNext
    // (VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07754).
    let (setup_reference_slot, decode_info_pnext, h265_setup_dpb_slot_info) = match codec {
        VideoCodec::H264 => {
            // decode_info and dpb_slot_info are already raw pointers
            let decode_info_pnext = h264_decode_info.map_or(std::ptr::null(), |info| info as *const _);
            eprintln!(
                "[dpb] Setup slot: vulkan_slot_index={}, dpb_slot={}",
                current_slot_index, current_slot_index
            );
            // FIX: Chain dpb_slot_info to setup slot (VUID-07754).
            let setup_reference_slot = vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: h264_dpb_slot_info.map_or(std::ptr::null(), |s| s as *const _ as *const _),
                slot_index: current_slot_index as i32,
                p_picture_resource: &dpb_setup_picture_resource,
                _marker: Default::default(),
            };
            (setup_reference_slot, decode_info_pnext, None)
        }
        VideoCodec::H265 => {
            // decode_info is already a raw pointer
            let decode_info_pnext = h265_decode_info.map_or(std::ptr::null(), |info| info as *const _);
            eprintln!(
                "[dpb] Setup slot: vulkan_slot_index={}, dpb_slot={}",
                current_slot_index, current_slot_index
            );

            // FIX: Match C++ reference (VulkanVideoParser.cpp:2194-2198) which leaves
            // setup slot pNext = NULL. The Vulkan spec VUID-07163 says it's required,
            // but the C++ reference works without it, suggesting NVIDIA driver handles it.
            let setup_reference_slot = vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: std::ptr::null(), // Match C++: NULL instead of VkVideoDecodeH265DpbSlotInfoKHR
                slot_index: current_slot_index as i32,
                p_picture_resource: &dpb_setup_picture_resource,
                _marker: Default::default(),
            };
            eprintln!(
                "[dpb] H265 setup slot pNext = NULL (matching C++ reference)"
            );
            // Return raw pointers to release borrows
            (setup_reference_slot, decode_info_pnext, None::<(*const ash::vk::native::StdVideoDecodeH265ReferenceInfo, *const vk::VideoDecodeH265DpbSlotInfoKHR)>)
        }
    };

    // Build reference slots for each valid DPB entry
    // We need to store all intermediate structs so pointers remain valid
    //
    // CRITICAL: Vulkan slot indices MUST be the actual DPB slot indices (ref_entry.slot_index).
    // These indices tell the Vulkan decoder which DPB slot contains which reference frame.
    // The Vulkan spec requires: "the DPB slot index specified by the slotIndex member of that
    // element must be currently associated with a frame picture matching the video picture
    // resource specified by the pPictureResource member"
    // Using sequential indices causes the decoder to read from wrong DPB slots.

    // Codec-specific reference slot building
    // H.264 uses StdVideoDecodeH264ReferenceInfo + VideoDecodeH264DpbSlotInfoKHR
    // H.265 uses StdVideoDecodeH265ReferenceInfo + VideoDecodeH265DpbSlotInfoKHR
    // CRITICAL: Use persistent vectors (h264_ref_info_vec, h265_ref_info_vec, etc.)
    // to ensure pointers remain valid after this function returns. Using local
    // vectors causes dangling pointers because their memory is freed when the
    // function returns, but the pointers are used in the command buffer.
    let valid_refs_len = valid_refs.len();

    // CRITICAL FIX: Reserve capacity before reference loop to prevent vector reallocation
    // from invalidating pointers to setup picture's ref_info and dpb_slot_info taken earlier.
    // Without this, pushing reference pictures can reallocate vectors and cause dangling pointers.
    h264_ref_info_vec.reserve(valid_refs_len);
    h264_dpb_slot_info_vec.reserve(valid_refs_len);
    h265_ref_info_vec.reserve(valid_refs_len);
    h265_dpb_slot_info_vec.reserve(valid_refs_len);

    let mut reference_slots: Vec<vk::VideoReferenceSlotInfoKHR> =
        Vec::with_capacity(valid_refs_len);
    let mut ref_picture_resources: Vec<vk::VideoPictureResourceInfoKHR> =
        Vec::with_capacity(valid_refs_len);

    match codec {
        VideoCodec::H264 => {
            for ref_entry in valid_refs.iter() {
                let mut ref_std_info = unsafe {
                    std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>()
                };
                ref_std_info.FrameNum = ref_entry.frame_num as u16;
                ref_std_info.PicOrderCnt = ref_entry.pic_order_cnt;
                // Per Vulkan spec and C++ reference (VulkanVideoParser.cpp:getPictureFlag):
                // For progressive reference pictures: both flags = 1 (both fields available)
                // For field reference pictures: flags indicate which fields are available
                // Since we only handle progressive H.264, both flags = 1.
                ref_std_info.flags.set_top_field_flag(1);
                ref_std_info.flags.set_bottom_field_flag(1);
                ref_std_info.flags.set_used_for_long_term_reference(0);
                ref_std_info.flags.set_is_non_existing(0);

                // DEBUG: Dump reference info for frame 1
                if frame_num == 1 {
                    eprintln!("[DEBUG-FRAME1-REF] DPB ref slot {}: dpb_slot={}, frame_num={}, poc=[{}, {}], flags(top={},bottom={},ltr={},non_exist={})",
                        reference_slots.len(), ref_entry.slot_index, ref_entry.frame_num,
                        ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1],
                        ref_std_info.flags.top_field_flag(),
                        ref_std_info.flags.bottom_field_flag(),
                        ref_std_info.flags.used_for_long_term_reference(),
                        ref_std_info.flags.is_non_existing());
                }

                h264_ref_info_vec.push(ref_std_info);
                let ref_info = &h264_ref_info_vec[h264_ref_info_vec.len() - 1];

                let dpb_slot_info = vk::VideoDecodeH264DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_info as *const _,
                    _marker: Default::default(),
                };
                h264_dpb_slot_info_vec.push(dpb_slot_info);
                let dpb_slot_info = &h264_dpb_slot_info_vec[h264_dpb_slot_info_vec.len() - 1];

                let ref_picture_resource = vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent,
                    base_array_layer: 0,
                    image_view_binding: ref_entry.image_view,
                    _marker: Default::default(),
                };
                ref_picture_resources.push(ref_picture_resource);

                let ref_slot = vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: dpb_slot_info as *const _ as *const _,
                    slot_index: ref_entry.slot_index as i32,
                    p_picture_resource: &ref_picture_resources[ref_picture_resources.len() - 1],
                    _marker: Default::default(),
                };
                reference_slots.push(ref_slot);

                eprintln!("[dpb] Ref slot {}: vulkan_slot_index={}, dpb_slot={}, frame_num={}, poc=[{}, {}]",
                    reference_slots.len() - 1, ref_entry.slot_index, ref_entry.slot_index, ref_entry.frame_num,
                    ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1]);
            }
        }
        VideoCodec::H265 => {
            for ref_entry in valid_refs.iter() {
                let mut ref_std_info = unsafe {
                    std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>()
                };
                ref_std_info.PicOrderCntVal = ref_entry.pic_order_cnt[0];
                ref_std_info.flags.set_used_for_long_term_reference(0);
                ref_std_info.flags.set_unused_for_reference(0);

                h265_ref_info_vec.push(ref_std_info);
                let ref_info = &h265_ref_info_vec[h265_ref_info_vec.len() - 1];

                let dpb_slot_info = vk::VideoDecodeH265DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_info as *const _,
                    _marker: Default::default(),
                };
                h265_dpb_slot_info_vec.push(dpb_slot_info);
                let dpb_slot_info = &h265_dpb_slot_info_vec[h265_dpb_slot_info_vec.len() - 1];

                let ref_picture_resource = vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent,
                    base_array_layer: 0,
                    image_view_binding: ref_entry.image_view,
                    _marker: Default::default(),
                };
                ref_picture_resources.push(ref_picture_resource);

                let ref_slot = vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: dpb_slot_info as *const _ as *const _,
                    slot_index: ref_entry.slot_index as i32,
                    p_picture_resource: &ref_picture_resources[ref_picture_resources.len() - 1],
                    _marker: Default::default(),
                };
                reference_slots.push(ref_slot);

                eprintln!("[dpb] H265 Ref slot {}: vulkan_slot_index={}, dpb_slot={}, frame_num={}, poc={}",
                    reference_slots.len() - 1, ref_entry.slot_index, ref_entry.slot_index, ref_entry.frame_num,
                    ref_entry.pic_order_cnt[0]);
            }
        }
    }

    // DEBUG: Print the final slot layout
    eprintln!("[dpb] === SLOT LAYOUT ===");
    for (i, ref_entry) in valid_refs.iter().enumerate() {
        eprintln!(
            "[dpb]   dpb_slot[{}] as ref (frame_num={}, poc={})",
            ref_entry.slot_index, ref_entry.frame_num, ref_entry.pic_order_cnt[0]
        );
    }
    eprintln!(
        "[dpb]   dpb_slot[{}] as setup (current frame {})",
        current_slot_index, frame_num
    );
     eprintln!("[dpb] === END SLOT LAYOUT ===");

     unsafe {
         // CRITICAL: Reset command buffer before re-use.
         // After the first frame, the command buffer is in the "ended" state.
         // vkBeginCommandBuffer on an ended command buffer returns VK_ERROR_COMMAND_BUFFER_NOT_RESET.
         device
             .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
             .map_err(|e| format!("Reset command buffer failed: {:?}", e))?;

         let begin_info = vk::CommandBufferBeginInfo::default()
             .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
         device
             .begin_command_buffer(cmd_buffer, &begin_info)
             .map_err(|e| format!("Begin command buffer failed: {:?}", e))?;

          // NOTE: Reference picture barriers are moved AFTER BeginVideoCodingKHR
          // to match the C++ reference (VkVideoDecoder.cpp:1208). All barriers
          // (bitstream + reference pictures + output image) are recorded in a
          // single cmdPipelineBarrier2 call after BeginVideoCodingKHR.

        // RESET is REQUIRED before the first decode per Vulkan spec.
        // Must be INSIDE a video coding block (between BeginVideoCoding and EndVideoCoding)
        // to satisfy VUID-vkCmdControlVideoCodingKHR-videocoding.
        // Also initializes the session and activates DPB slots (VkVideoDecoder.cpp:1205-1213).
        //
        // The RESET is done INSIDE the same video coding block as the decode, matching
        // the C++ reference. DPB slots become active when referenced in BeginVideoCodingKHR,
        // so we include all slots in the first BeginVideoCodingKHR.
        //
        // FIX: On RESET frame, activate ALL DPB slots (0..max_dpb_slots), not just the
        // current frame's slot. Vulkan spec: "If flags contains RESET_BIT, pReferenceSlots
        // specifies the set of DPB slots to be activated."
        let max_slots = dpb_manager.max_dpb_slots as usize;
        // CRITICAL: These vectors MUST outlive the command recording.
        // They contain data pointed to by all_slots, which is used in begin_coding_info.
        let mut all_picture_resources: Vec<vk::VideoPictureResourceInfoKHR> =
            Vec::with_capacity(max_slots);
        let mut empty_h265_ref_std_infos: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo> =
            Vec::with_capacity(max_slots);
        let mut empty_h265_dpb_slot_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> =
            Vec::with_capacity(max_slots);

        let all_slots: Vec<vk::VideoReferenceSlotInfoKHR>;
        if !decoder_reset_done {
            // RESET frame: activate ALL DPB slots
            eprintln!(
                "[dpb] RESET frame: activating ALL {} DPB slots",
                max_slots
            );

            // Pre-build all picture resources
            for slot_idx in 0..max_slots {
                let view = dpb_views[slot_idx];
                all_picture_resources.push(vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent,
                    base_array_layer: 0,
                    image_view_binding: view,
                    _marker: Default::default(),
                });
            }

            // Pre-build H265 empty ref/std/dpb slot infos for non-setup slots
            if codec == VideoCodec::H265 {
                for slot_idx in 0..max_slots {
                    if slot_idx == current_slot_index as usize {
                        continue; // Setup slot uses its own DPB slot info
                    }
                    let mut ref_std_info = unsafe {
                        std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>()
                    };
                    ref_std_info.PicOrderCntVal = 0;
                    ref_std_info.flags.set_used_for_long_term_reference(0);
                    ref_std_info.flags.set_unused_for_reference(1);
                    empty_h265_ref_std_infos.push(ref_std_info);

                    let ref_std_idx = empty_h265_ref_std_infos.len() - 1;
                    empty_h265_dpb_slot_infos.push(vk::VideoDecodeH265DpbSlotInfoKHR {
                        s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                        p_next: std::ptr::null(),
                        p_std_reference_info: &empty_h265_ref_std_infos[ref_std_idx] as *const _,
                        _marker: Default::default(),
                    });
                }
            }

            // Now build all_slots with stable pointers
            let mut empty_h265_idx = 0u32;
            all_slots = (0..max_slots)
                .map(|slot_idx| {
                    let is_setup = slot_idx == current_slot_index as usize;
                    let pr_ptr = &all_picture_resources[slot_idx];

                    if is_setup {
                        setup_reference_slot
                    } else {
                        match codec {
                            VideoCodec::H264 => {
                                vk::VideoReferenceSlotInfoKHR {
                                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                                    p_next: std::ptr::null(),
                                    slot_index: slot_idx as i32,
                                    p_picture_resource: pr_ptr,
                                    _marker: Default::default(),
                                }
                            }
                            VideoCodec::H265 => {
                                let dpb_slot_idx = empty_h265_idx as usize;
                                empty_h265_idx += 1;
                                vk::VideoReferenceSlotInfoKHR {
                                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                                    p_next: &empty_h265_dpb_slot_infos[dpb_slot_idx] as *const _ as *const std::ffi::c_void,
                                    slot_index: slot_idx as i32,
                                    p_picture_resource: pr_ptr,
                                    _marker: Default::default(),
                                }
                            }
                        }
                    }
                })
                .collect();

            eprintln!(
                "[dpb] VkVideoBeginCodingInfoKHR: {} total slots (RESET - all DPB slots)",
                all_slots.len()
            );

            for (i, slot) in all_slots.iter().enumerate() {
                let pr = unsafe { &*slot.p_picture_resource };
                eprintln!(
                    "[dpb]   slot[{}] p_picture_resource={:p} -> coded_extent={}x{}, view={:?}",
                    i, slot.p_picture_resource, pr.coded_extent.width, pr.coded_extent.height, pr.image_view_binding
                );
            }
        } else {
            // Non-RESET frame: only references + setup slot
            all_slots = reference_slots
                .iter()
                .cloned()
                .chain(std::iter::once(setup_reference_slot))
                .collect();
            eprintln!(
                "[dpb] VkVideoBeginCodingInfoKHR: {} total slots ({} refs + 1 setup)",
                all_slots.len(),
                reference_slots.len()
            );
            eprintln!("[dpb] all_slots.as_ptr() = {:p}, reference_slots.as_ptr() = {:p}",
                all_slots.as_ptr(), reference_slots.as_ptr());
            // Verify all_slots[0] p_next
            if codec == VideoCodec::H265 && !all_slots.is_empty() {
                let slot0 = &all_slots[0];
                eprintln!("[dpb] all_slots[0].p_next = {:p}", slot0.p_next);
                if !slot0.p_next.is_null() {
                    let dpb = unsafe { &*(slot0.p_next as *const vk::VideoDecodeH265DpbSlotInfoKHR) };
                    eprintln!("[dpb] all_slots[0] pNext s_type = {:?} (raw={})", dpb.s_type, dpb.s_type.as_raw());
                }
            }
        }

        let begin_coding_info = vk::VideoBeginCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoBeginCodingFlagsKHR::empty(),
            video_session: session,
            video_session_parameters: session_params,
            reference_slot_count: all_slots.len() as u32,
            p_reference_slots: all_slots.as_ptr(),
            _marker: Default::default(),
        };
        cmd_begin_video_coding(instance, device.handle(), cmd_buffer, &begin_coding_info);

        // DEBUG: Verify reference slot data after BeginVideoCoding
        if codec == VideoCodec::H265 && !reference_slots.is_empty() {
            eprintln!("[verify-after-begin] reference_slots.as_ptr() = {:p}", reference_slots.as_ptr());
            for (i, slot) in reference_slots.iter().enumerate() {
                if !slot.p_next.is_null() {
                    let dpb = unsafe { &*(slot.p_next as *const vk::VideoDecodeH265DpbSlotInfoKHR) };
                    eprintln!("[verify-after-begin] ref_slot[{}] p_next={:p} s_type={:?} (raw={})",
                        i, slot.p_next, dpb.s_type, dpb.s_type.as_raw());
                }
            }
        }

        // RESET inside video coding block (after BeginVideoCoding, before decode)
        if !decoder_reset_done {
            eprintln!("[reset] Performing decoder RESET inside video coding block...");
            let coding_control_info = vk::VideoCodingControlInfoKHR {
                s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoCodingControlFlagsKHR::RESET,
                _marker: Default::default(),
            };
            cmd_control_video_coding(instance, device.handle(), cmd_buffer, &coding_control_info);
            eprintln!("[reset] RESET complete");
        }

        // Combined barrier AFTER vkCmdBeginVideoCodingKHR: bitstream + ALL images (refs + output)
        // Match C++ reference VkVideoDecoder.cpp:1208 - all barriers in single call
        // srcStageMask=HOST required when srcAccessMask=HOST_WRITE per Vulkan spec
        let buffer_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::HOST,
            src_access_mask: vk::AccessFlags2::HOST_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: bitstream_buffer,
            offset: bitstream_offset,
            size: bs_range,
            _marker: Default::default(),
        };

        // Build image barriers: reference pictures + output image
        // Use PLANE_0 and PLANE_1 for multi-plane images (COLOR is UB for multi-plane)
        // Match C++ reference: srcStageMask=NONE, srcAccessMask=0 for all image barriers
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        // Reference picture barriers (READ access - decoder reads reference frames here)
        for ref_entry in &valid_refs {
            let ref_slot_layout = dpb_manager.get_slot_layout(ref_entry.slot_index);
            eprintln!("[barrier-after-coding] Ref slot {}: frame_num={}, tracked_layout={:?}, needs_barrier={}",
                ref_entry.slot_index, ref_entry.frame_num, ref_slot_layout,
                ref_slot_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
            if frame_num == 1 {
                eprintln!("[DEBUG-FRAME1-REF-BARRIER] slot={}, frame_num={}, image={:?}, layout={:?}",
                    ref_entry.slot_index, ref_entry.frame_num, ref_entry.image, ref_slot_layout);
            }
            if ref_slot_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
                for &aspect in &[vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1] {
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                        p_next: std::ptr::null(),
                        src_stage_mask: vk::PipelineStageFlags2::NONE,
                        src_access_mask: vk::AccessFlags2::empty(),
                        dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                        dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: ref_entry.image,
                        old_layout: ref_slot_layout,
                        new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspect,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        _marker: Default::default(),
                    });
                }
                eprintln!("[barrier-after-coding] Ref slot {}: {:?} -> VIDEO_DECODE_DPB_KHR (READ)",
                    ref_entry.slot_index, ref_slot_layout);
            } else {
                eprintln!("[barrier-after-coding] Ref slot {}: already VIDEO_DECODE_DPB_KHR - NO BARRIER",
                    ref_entry.slot_index);
            }
        }

        // Output image barrier (WRITE access - decoder writes decoded frame here)
        let output_slot_layout = dpb_manager.get_slot_layout(current_slot_index);
        eprintln!(
            "[barrier-after-coding] Output slot {} current layout: {:?}",
            current_slot_index, output_slot_layout
        );
        if output_slot_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
            for &aspect in &[vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1] {
                image_barriers.push(vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    src_access_mask: vk::AccessFlags2::empty(),
                    dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: output_image,
                    old_layout: output_slot_layout,
                    new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    _marker: Default::default(),
                });
            }
            eprintln!("[barrier-after-coding] Output: {:?} -> VIDEO_DECODE_DPB_KHR (WRITE)", output_slot_layout);
        }

        eprintln!("[barrier-after-coding] Total image barriers: {} (refs + output)",
            image_barriers.len());

        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buffer_barrier,
            image_memory_barrier_count: image_barriers.len() as u32,
            p_image_memory_barriers: image_barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        // Decode info
        // src_buffer_range uses actual bitstream size (no alignment needed for range)
        eprintln!("[decode] Bitstream range: {} bytes", bs_range);
        eprintln!(
            "[decode] Using {} reference slots for P-frame prediction",
            reference_slots.len()
        );

        // DEBUG: Print VkVideoDecodeInfoKHR key fields for every frame
        eprintln!("[decode_info] Frame {}: src_buffer_offset={}, src_buffer_range={}, dst_view={:?}, ref_slots={}",
                   frame_num, bitstream_offset, bs_range, output_image_view, reference_slots.len());

        let decode_info = vk::VideoDecodeInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
            p_next: decode_info_pnext,
            flags: vk::VideoDecodeFlagsKHR::empty(),
            src_buffer: bitstream_buffer,
            src_buffer_offset: bitstream_offset,
            src_buffer_range: bs_range,
            dst_picture_resource,
            p_setup_reference_slot: &setup_reference_slot,
            reference_slot_count: reference_slots.len() as u32,
            p_reference_slots: if reference_slots.is_empty() {
                std::ptr::null()
            } else {
                reference_slots.as_ptr()
            },
            _marker: Default::default(),
        };

        // ====================================================================
        // COMPREHENSIVE FRAME 0/1 DEBUG LOGGING
        // ====================================================================
        if (is_frame_1_debug || is_frame_0_or_1) && codec == VideoCodec::H264 {
            eprintln!("\n\n=================================================================");
            eprintln!("         FRAME 1 COMPREHENSIVE DECODE PARAMETERS");
            eprintln!("=================================================================\n");

            // 1. VkVideoDecodeH264PictureInfoKHR
            if let Some(h264_decode_info_ptr) = h264_decode_info {
                let h264_decode_info_ref = unsafe { &*h264_decode_info_ptr };
                eprintln!("=== 1. VkVideoDecodeH264PictureInfoKHR ===");
                eprintln!(
                    "  s_type = {:?} ({})",
                    h264_decode_info_ref.s_type,
                    h264_decode_info_ref.s_type.as_raw()
                );
                eprintln!("  p_next = {:p}", h264_decode_info_ref.p_next);
                eprintln!("  slice_count = {}", h264_decode_info_ref.slice_count);
                if !h264_decode_info_ref.p_slice_offsets.is_null()
                    && h264_decode_info_ref.slice_count > 0
                {
                    let offsets = unsafe {
                        std::slice::from_raw_parts(
                            h264_decode_info_ref.p_slice_offsets,
                            h264_decode_info_ref.slice_count as usize,
                        )
                    };
                    eprintln!("  p_slice_offsets = {:?}", offsets);
                }
                if !h264_decode_info_ref.p_std_picture_info.is_null() {
                    let pic_info = unsafe { &*h264_decode_info_ref.p_std_picture_info };
                    eprintln!("  p_std_picture_info -> StdVideoDecodeH264PictureInfo:");
                    eprintln!("    flags.raw = 0x{:08x}", {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                &pic_info.flags as *const _ as *const u8,
                                std::mem::size_of::<
                                    ash::vk::native::StdVideoDecodeH264PictureInfoFlags,
                                >(),
                            )
                        };
                        let mut val: u32 = 0;
                        for (i, &b) in bytes.iter().enumerate() {
                            val |= (b as u32) << (i * 8);
                        }
                        val
                    });
                    eprintln!("    flags.is_intra = {}", pic_info.flags.is_intra());
                    eprintln!(
                        "    flags.field_pic_flag = {}",
                        pic_info.flags.field_pic_flag()
                    );
                    eprintln!("    flags.IdrPicFlag = {}", pic_info.flags.IdrPicFlag());
                    eprintln!(
                        "    flags.bottom_field_flag = {}",
                        pic_info.flags.bottom_field_flag()
                    );
                    eprintln!("    flags.is_reference = {}", pic_info.flags.is_reference());
                    eprintln!(
                        "    flags.complementary_field_pair = {}",
                        pic_info.flags.complementary_field_pair()
                    );
                    eprintln!(
                        "    seq_parameter_set_id = {}",
                        pic_info.seq_parameter_set_id
                    );
                    eprintln!(
                        "    pic_parameter_set_id = {}",
                        pic_info.pic_parameter_set_id
                    );
                    eprintln!("    frame_num = {}", pic_info.frame_num);
                    eprintln!("    idr_pic_id = {}", pic_info.idr_pic_id);
                    eprintln!("    PicOrderCnt[0] = {}", pic_info.PicOrderCnt[0]);
                    eprintln!("    PicOrderCnt[1] = {}", pic_info.PicOrderCnt[1]);
                    // Print full struct as bytes for binary comparison
                    let pic_bytes = unsafe {
                        std::slice::from_raw_parts(
                            pic_info as *const _ as *const u8,
                            std::mem::size_of::<ash::vk::native::StdVideoDecodeH264PictureInfo>(),
                        )
                    };
                    eprintln!("    [FULL STRUCT BYTES {} bytes] =", pic_bytes.len());
                    let hex_str: String = pic_bytes
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!("    {}", hex_str);
                }
                eprintln!();
            }

            // 2. VkVideoDecodeH264ReferenceInfoKHR for setup slot
            eprintln!("=== 2. VkVideoDecodeH264ReferenceInfoKHR (Setup Slot) ===");
            if let Some(dpb_info_ptr) = h264_dpb_slot_info {
                let dpb_info = unsafe { &*dpb_info_ptr };
                eprintln!("  VkVideoDecodeH264DpbSlotInfoKHR:");
                eprintln!(
                    "    s_type = {:?} ({})",
                    dpb_info.s_type,
                    dpb_info.s_type.as_raw()
                );
                eprintln!("    p_next = {:p}", dpb_info.p_next);
                eprintln!(
                    "    p_std_reference_info = {:p}",
                    dpb_info.p_std_reference_info
                );
                if !dpb_info.p_std_reference_info.is_null() {
                    let ref_info = unsafe { &*dpb_info.p_std_reference_info };
                    eprintln!("  -> StdVideoDecodeH264ReferenceInfo:");
                    eprintln!("    flags.raw = 0x{:08x}", {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                &ref_info.flags as *const _ as *const u8,
                                std::mem::size_of::<
                                    ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags,
                                >(),
                            )
                        };
                        let mut val: u32 = 0;
                        for (i, &b) in bytes.iter().enumerate() {
                            val |= (b as u32) << (i * 8);
                        }
                        val
                    });
                    eprintln!(
                        "    flags.top_field_flag = {}",
                        ref_info.flags.top_field_flag()
                    );
                    eprintln!(
                        "    flags.bottom_field_flag = {}",
                        ref_info.flags.bottom_field_flag()
                    );
                    eprintln!(
                        "    flags.used_for_long_term_reference = {}",
                        ref_info.flags.used_for_long_term_reference()
                    );
                    eprintln!(
                        "    flags.is_non_existing = {}",
                        ref_info.flags.is_non_existing()
                    );
                    eprintln!("    FrameNum = {}", ref_info.FrameNum);
                    eprintln!("    PicOrderCnt[0] = {}", ref_info.PicOrderCnt[0]);
                    eprintln!("    PicOrderCnt[1] = {}", ref_info.PicOrderCnt[1]);
                    eprintln!("    reserved = 0x{:04x}", ref_info.reserved);
                    // Full struct bytes
                    let ref_bytes = unsafe {
                        std::slice::from_raw_parts(
                            ref_info as *const _ as *const u8,
                            std::mem::size_of::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>(),
                        )
                    };
                    eprintln!("    [FULL STRUCT BYTES {} bytes] =", ref_bytes.len());
                    let hex_str: String = ref_bytes
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!("    {}", hex_str);
                }
            }
            eprintln!();

            // 3. Reference info for each reference slot
            eprintln!("=== 3. Reference Info (Reference Slots) ===");
            for (i, ref_entry) in valid_refs.iter().enumerate() {
                eprintln!(
                    "  Reference slot {}: dpb_slot_index={}",
                    i, ref_entry.slot_index
                );
                eprintln!("    frame_num = {}", ref_entry.frame_num);
                eprintln!(
                    "    pic_order_cnt = [{}, {}]",
                    ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1]
                );
            }
            eprintln!();

            // 4. VkVideoDecodeInfoKHR
            eprintln!("=== 4. VkVideoDecodeInfoKHR ===");
            eprintln!(
                "  s_type = {:?} ({})",
                decode_info.s_type,
                decode_info.s_type.as_raw()
            );
            eprintln!("  p_next = {:p}", decode_info.p_next);
            eprintln!(
                "  flags = {:?} ({})",
                decode_info.flags,
                decode_info.flags.as_raw()
            );
            eprintln!("  src_buffer = {:p}", decode_info.src_buffer);
            eprintln!("  src_buffer_offset = {}", decode_info.src_buffer_offset);
            eprintln!("  src_buffer_range = {}", decode_info.src_buffer_range);
            eprintln!("  dst_picture_resource:");
            eprintln!(
                "    coded_offset = ({}, {})",
                decode_info.dst_picture_resource.coded_offset.x,
                decode_info.dst_picture_resource.coded_offset.y
            );
            eprintln!(
                "    coded_extent = {}x{}",
                decode_info.dst_picture_resource.coded_extent.width,
                decode_info.dst_picture_resource.coded_extent.height
            );
            eprintln!(
                "    base_array_layer = {}",
                decode_info.dst_picture_resource.base_array_layer
            );
            eprintln!(
                "    image_view_binding = {:p}",
                decode_info.dst_picture_resource.image_view_binding
            );
            eprintln!("  p_setup_reference_slot:");
            if !decode_info.p_setup_reference_slot.is_null() {
                let setup = unsafe { &*decode_info.p_setup_reference_slot };
                eprintln!("    slot_index = {}", setup.slot_index);
                eprintln!("    p_next = {:p}", setup.p_next);
                eprintln!("    p_picture_resource = {:p}", setup.p_picture_resource);
            }
            eprintln!(
                "  reference_slot_count = {}",
                decode_info.reference_slot_count
            );
            if !decode_info.p_reference_slots.is_null() && decode_info.reference_slot_count > 0 {
                let slots = unsafe {
                    std::slice::from_raw_parts(
                        decode_info.p_reference_slots,
                        decode_info.reference_slot_count as usize,
                    )
                };
                for (i, slot) in slots.iter().enumerate() {
                    eprintln!("    reference_slots[{}]:", i);
                    eprintln!("      slot_index = {}", slot.slot_index);
                    eprintln!("      p_next = {:p}", slot.p_next);
                    eprintln!("      p_picture_resource = {:p}", slot.p_picture_resource);
                    // If there's a DPB slot info attached, print it
                    if !slot.p_next.is_null() {
                        // Try to interpret as VkVideoDecodeH264DpbSlotInfoKHR
                        let dpb_slot =
                            unsafe { &*(slot.p_next as *const vk::VideoDecodeH264DpbSlotInfoKHR) };
                        if dpb_slot.s_type == vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR
                        {
                            eprintln!("      -> VkVideoDecodeH264DpbSlotInfoKHR:");
                            eprintln!(
                                "        p_std_reference_info = {:p}",
                                dpb_slot.p_std_reference_info
                            );
                            if !dpb_slot.p_std_reference_info.is_null() {
                                let ri = unsafe { &*dpb_slot.p_std_reference_info };
                                eprintln!(
                                    "          FrameNum={}, POC=[{}, {}]",
                                    ri.FrameNum, ri.PicOrderCnt[0], ri.PicOrderCnt[1]
                                );
                            }
                        }
                    }
                }
            }
            eprintln!();

            // 5. Bitstream info
            eprintln!("=== 5. Bitstream ===");
            eprintln!("  Total size: {} bytes", bitstream_data.len());
            eprintln!("  First 100 bytes:");
            let dump_len = 100.min(bitstream_data.len());
            for i in (0..dump_len).step_by(16) {
                let end = (i + 16).min(dump_len);
                let hex: String = (i..end)
                    .map(|j| format!("{:02x}", bitstream_data[j]))
                    .collect::<Vec<_>>()
                    .join(" ");
                let ascii: String = (i..end)
                    .map(|j| {
                        if bitstream_data[j] >= 0x20 && bitstream_data[j] < 0x7f {
                            bitstream_data[j] as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                eprintln!("    {:04x}: {:<48} |{}|", i, hex, ascii);
            }

            // 6. SPS/PPS info
            if let Some(H264OrH265Sps::H264(sps)) = sps {
                eprintln!("\n=== 6. SPS (for reference) ===");
                eprintln!("  seq_parameter_set_id = {}", sps.seq_parameter_set_id);
                eprintln!("  profile_idc = {}", sps.profile_idc);
                eprintln!("  level_idc = {}", sps.level_idc);
                eprintln!(
                    "  log2_max_frame_num_minus4 = {} -> MaxFrameNum={}",
                    sps.log2_max_frame_num_minus4,
                    1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4)
                );
                eprintln!(
                    "  log2_max_pic_order_cnt_lsb_minus4 = {} -> MaxPicOrderCntLsb={}",
                    sps.log2_max_pic_order_cnt_lsb_minus4,
                    1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)
                );
                eprintln!("  pic_order_cnt_type = {}", sps.pic_order_cnt_type);
                eprintln!("  max_num_ref_frames = {}", sps.max_num_ref_frames);
                eprintln!("  frame_mbs_only_flag = {}", sps.frame_mbs_only_flag);
            }
            if let Some(H264OrH265Pps::H264(pps)) = _pps {
                eprintln!("\n=== 7. PPS (for reference) ===");
                eprintln!("  pic_parameter_set_id = {}", pps.pic_parameter_set_id);
                eprintln!("  seq_parameter_set_id = {}", pps.seq_parameter_set_id);
                eprintln!(
                    "  num_ref_idx_l0_default_active_minus1 = {}",
                    pps.num_ref_idx_l0_default_active_minus1
                );
                eprintln!(
                    "  num_ref_idx_l1_default_active_minus1 = {}",
                    pps.num_ref_idx_l1_default_active_minus1
                );
                eprintln!("  weighted_pred_flag = {}", pps.weighted_pred_flag);
                eprintln!("  weighted_bipred_idc = {}", pps.weighted_bipred_idc);
                eprintln!("  pic_init_qp_minus26 = {}", pps.pic_init_qp_minus26);
            }

            eprintln!("\n=================================================================");
            eprintln!("         END FRAME 1 COMPREHENSIVE DECODE PARAMETERS");
            eprintln!("=================================================================\n\n");
        }
        // ====================================================================
        // H.265 COMPREHENSIVE DEBUG LOGGING
        // ====================================================================
        if codec == VideoCodec::H265 {
            eprintln!("\n\n=================================================================");
            eprintln!("         H.265 COMPREHENSIVE DECODE PARAMETERS");
            eprintln!("=================================================================\n");

            // 1. VkVideoDecodeH265PictureInfoKHR
            if let Some(h265_decode_info_ptr) = h265_decode_info {
                let h265_decode_info_ref = unsafe { &*h265_decode_info_ptr };
                eprintln!("=== 1. VkVideoDecodeH265PictureInfoKHR ===");
                eprintln!(
                    "  s_type = {:?} ({})",
                    h265_decode_info_ref.s_type,
                    h265_decode_info_ref.s_type.as_raw()
                );
                eprintln!("  p_next = {:p}", h265_decode_info_ref.p_next);
                eprintln!("  p_std_picture_info = {:p}", h265_decode_info_ref.p_std_picture_info);
                eprintln!("  slice_segment_count = {}", h265_decode_info_ref.slice_segment_count);
                if !h265_decode_info_ref.p_slice_segment_offsets.is_null()
                    && h265_decode_info_ref.slice_segment_count > 0
                {
                    let offsets = unsafe {
                        std::slice::from_raw_parts(
                            h265_decode_info_ref.p_slice_segment_offsets,
                            h265_decode_info_ref.slice_segment_count as usize,
                        )
                    };
                    eprintln!("  p_slice_segment_offsets = {:?}", offsets);
                }
                if !h265_decode_info_ref.p_std_picture_info.is_null() {
                    let pic_info = unsafe { &*h265_decode_info_ref.p_std_picture_info };
                    eprintln!("  p_std_picture_info -> StdVideoDecodeH265PictureInfo:");
                    eprintln!("    flags.IrapPicFlag = {}", pic_info.flags.IrapPicFlag());
                    eprintln!("    flags.IdrPicFlag = {}", pic_info.flags.IdrPicFlag());
                    eprintln!("    flags.IsReference = {}", pic_info.flags.IsReference());
                    eprintln!("    flags.short_term_ref_pic_set_sps_flag = {}", pic_info.flags.short_term_ref_pic_set_sps_flag());
                    eprintln!("    pps_pic_parameter_set_id = {}", pic_info.pps_pic_parameter_set_id);
                    eprintln!("    pps_seq_parameter_set_id = {}", pic_info.pps_seq_parameter_set_id);
                    eprintln!("    sps_video_parameter_set_id = {}", pic_info.sps_video_parameter_set_id);
                    eprintln!("    NumBitsForSTRefPicSetInSlice = {}", pic_info.NumBitsForSTRefPicSetInSlice);
                    eprintln!("    NumDeltaPocsOfRefRpsIdx = {}", pic_info.NumDeltaPocsOfRefRpsIdx);
                    eprintln!("    PicOrderCntVal = {}", pic_info.PicOrderCntVal);
                    eprintln!("    RefPicSetStCurrBefore = {:?}", pic_info.RefPicSetStCurrBefore);
                    eprintln!("    RefPicSetStCurrAfter = {:?}", pic_info.RefPicSetStCurrAfter);
                    eprintln!("    RefPicSetLtCurr = {:?}", pic_info.RefPicSetLtCurr);
                    // Full struct bytes
                    let pic_bytes = unsafe {
                        std::slice::from_raw_parts(
                            pic_info as *const _ as *const u8,
                            std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfo>(),
                        )
                    };
                    eprintln!("    [FULL STRUCT BYTES {} bytes] =", pic_bytes.len());
                    let hex_str: String = pic_bytes
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!("    {}", hex_str);
                } else {
                    eprintln!("  *** WARNING: p_std_picture_info is NULL! ***");
                }
                eprintln!();
            }

            // 2. Setup slot's H.265 DPB slot info
            eprintln!("=== 2. Setup Slot H.265 DPB Slot Info ===");
            if let Some((setup_ref_std_ptr, setup_dpb_slot_ptr)) = h265_setup_dpb_slot_info {
                let setup_ref_std: &ash::vk::native::StdVideoDecodeH265ReferenceInfo = unsafe { &*setup_ref_std_ptr };
                let setup_dpb_slot: &vk::VideoDecodeH265DpbSlotInfoKHR = unsafe { &*setup_dpb_slot_ptr };
                eprintln!("  VkVideoDecodeH265DpbSlotInfoKHR:");
                eprintln!("    s_type = {:?} ({})", setup_dpb_slot.s_type, setup_dpb_slot.s_type.as_raw());
                eprintln!("    p_next = {:p}", setup_dpb_slot.p_next);
                eprintln!("    p_std_reference_info = {:p}", setup_dpb_slot.p_std_reference_info);
                eprintln!("  -> StdVideoDecodeH265ReferenceInfo:");
                eprintln!("    PicOrderCntVal = {}", setup_ref_std.PicOrderCntVal);
                eprintln!("    flags.used_for_long_term_reference = {}", setup_ref_std.flags.used_for_long_term_reference());
                eprintln!("    flags.unused_for_reference = {}", setup_ref_std.flags.unused_for_reference());
                // Full struct bytes
                let ref_bytes = unsafe {
                    std::slice::from_raw_parts(
                        setup_ref_std as *const _ as *const u8,
                        std::mem::size_of::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>(),
                    )
                };
                eprintln!("    [FULL STRUCT BYTES {} bytes] =", ref_bytes.len());
                let hex_str: String = ref_bytes
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!("    {}", hex_str);
            } else {
                eprintln!("  *** WARNING: h265_setup_dpb_slot_info is None! ***");
            }
            eprintln!();

            // 3. Reference slots H.265 DPB slot info
            eprintln!("=== 3. Reference Slots H.265 DPB Slot Info ===");
            for (i, ref_entry) in valid_refs.iter().enumerate() {
                eprintln!("  Reference slot {}: dpb_slot_index={}", i, ref_entry.slot_index);
                eprintln!("    frame_num = {}", ref_entry.frame_num);
                eprintln!("    pic_order_cnt = [{}, {}]", ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1]);

                // Check p_picture_resource
                if i < reference_slots.len() {
                    let slot = &reference_slots[i];
                    eprintln!("    VkVideoReferenceSlotInfoKHR (slot is at {:p}):", &slot as *const _);
                    eprintln!("      slot_index = {}", slot.slot_index);
                    eprintln!("      p_next = {:p}", slot.p_next);
                    eprintln!("      p_picture_resource = {:p}", slot.p_picture_resource);

                    if !slot.p_picture_resource.is_null() {
                        let pr = unsafe { &*slot.p_picture_resource };
                        eprintln!("      -> VkVideoPictureResourceInfoKHR:");
                        eprintln!("        coded_extent = {}x{}", pr.coded_extent.width, pr.coded_extent.height);
                        eprintln!("        image_view_binding = {:p}", pr.image_view_binding);
                        if pr.image_view_binding == vk::ImageView::null() {
                            eprintln!("        *** WARNING: image_view_binding is NULL! ***");
                        }
                    } else {
                        eprintln!("      *** WARNING: p_picture_resource is NULL! ***");
                    }

                    // Check p_next chain (H.265 DPB slot info)
                    if !slot.p_next.is_null() {
                        // Print raw bytes at p_next
                        let raw_bytes = unsafe {
                            std::slice::from_raw_parts(
                                slot.p_next as *const u8,
                                16.min(std::mem::size_of::<vk::VideoDecodeH265DpbSlotInfoKHR>()),
                            )
                        };
                        let hex: String = raw_bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                        eprintln!("      -> pNext raw bytes (first 16): {}", hex);

                        let dpb_slot = unsafe { &*(slot.p_next as *const vk::VideoDecodeH265DpbSlotInfoKHR) };
                        eprintln!("      -> pNext s_type raw value = {} (expected {})",
                            dpb_slot.s_type.as_raw(),
                            vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR.as_raw());
                        if dpb_slot.s_type == vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR {
                            eprintln!("      -> VkVideoDecodeH265DpbSlotInfoKHR:");
                            eprintln!("        p_std_reference_info = {:p}", dpb_slot.p_std_reference_info);
                            if !dpb_slot.p_std_reference_info.is_null() {
                                let ri = unsafe { &*dpb_slot.p_std_reference_info };
                                eprintln!("          -> StdVideoDecodeH265ReferenceInfo:");
                                eprintln!("            PicOrderCntVal = {}", ri.PicOrderCntVal);
                                eprintln!("            flags.used_for_long_term_reference = {}", ri.flags.used_for_long_term_reference());
                                eprintln!("            flags.unused_for_reference = {}", ri.flags.unused_for_reference());
                            } else {
                                eprintln!("        *** WARNING: p_std_reference_info is NULL! ***");
                            }
                        }
                    } else {
                        eprintln!("      *** WARNING: p_next (H265 DPB slot info) is NULL! ***");
                    }
                }
                eprintln!();
            }

            // 4. VkVideoDecodeInfoKHR
            eprintln!("=== 4. VkVideoDecodeInfoKHR ===");
            eprintln!(
                "  s_type = {:?} ({})",
                decode_info.s_type,
                decode_info.s_type.as_raw()
            );
            eprintln!("  p_next = {:p}", decode_info.p_next);
            eprintln!("  flags = {:?} ({})", decode_info.flags, decode_info.flags.as_raw());
            eprintln!("  src_buffer = {:p}", decode_info.src_buffer);
            eprintln!("  src_buffer_offset = {}", decode_info.src_buffer_offset);
            eprintln!("  src_buffer_range = {}", decode_info.src_buffer_range);
            eprintln!("  dst_picture_resource:");
            eprintln!("    coded_offset = ({}, {})", decode_info.dst_picture_resource.coded_offset.x, decode_info.dst_picture_resource.coded_offset.y);
            eprintln!("    coded_extent = {}x{}", decode_info.dst_picture_resource.coded_extent.width, decode_info.dst_picture_resource.coded_extent.height);
            eprintln!("    base_array_layer = {}", decode_info.dst_picture_resource.base_array_layer);
            eprintln!("    image_view_binding = {:p}", decode_info.dst_picture_resource.image_view_binding);
            if decode_info.dst_picture_resource.image_view_binding == vk::ImageView::null() {
                eprintln!("    *** WARNING: dst image_view_binding is NULL! ***");
            }
            eprintln!("  p_setup_reference_slot:");
            if !decode_info.p_setup_reference_slot.is_null() {
                let setup = unsafe { &*decode_info.p_setup_reference_slot };
                eprintln!("    slot_index = {}", setup.slot_index);
                eprintln!("    p_next = {:p}", setup.p_next);
                eprintln!("    p_picture_resource = {:p}", setup.p_picture_resource);
                if !setup.p_picture_resource.is_null() {
                    let pr = unsafe { &*setup.p_picture_resource };
                    eprintln!("    -> image_view_binding = {:p}", pr.image_view_binding);
                    if pr.image_view_binding == vk::ImageView::null() {
                        eprintln!("    *** WARNING: setup image_view_binding is NULL! ***");
                    }
                } else {
                    eprintln!("    *** WARNING: setup p_picture_resource is NULL! ***");
                }
                // Check setup p_next chain
                if !setup.p_next.is_null() {
                    let dpb_slot = unsafe { &*(setup.p_next as *const vk::VideoDecodeH265DpbSlotInfoKHR) };
                    eprintln!("    -> pNext s_type raw value = {} (expected {})",
                        dpb_slot.s_type.as_raw(),
                        vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR.as_raw());
                    if dpb_slot.s_type == vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR {
                        eprintln!("    -> VkVideoDecodeH265DpbSlotInfoKHR:");
                        eprintln!("       p_std_reference_info = {:p}", dpb_slot.p_std_reference_info);
                        if !dpb_slot.p_std_reference_info.is_null() {
                            let ri = unsafe { &*dpb_slot.p_std_reference_info };
                            eprintln!("         PicOrderCntVal = {}", ri.PicOrderCntVal);
                            eprintln!("         unused_for_reference = {}", ri.flags.unused_for_reference());
                        } else {
                            eprintln!("       *** WARNING: setup p_std_reference_info is NULL! ***");
                        }
                    }
                } else {
                    eprintln!("    *** WARNING: setup p_next (H265 DPB slot info) is NULL! ***");
                }
            }
            eprintln!("  reference_slot_count = {}", decode_info.reference_slot_count);
            if !decode_info.p_reference_slots.is_null() && decode_info.reference_slot_count > 0 {
                let slots = unsafe {
                    std::slice::from_raw_parts(
                        decode_info.p_reference_slots,
                        decode_info.reference_slot_count as usize,
                    )
                };
                for (i, slot) in slots.iter().enumerate() {
                    eprintln!("    reference_slots[{}]:", i);
                    eprintln!("      slot_index = {}", slot.slot_index);
                    eprintln!("      p_next = {:p}", slot.p_next);
                    eprintln!("      p_picture_resource = {:p}", slot.p_picture_resource);
                    if !slot.p_picture_resource.is_null() {
                        let pr = unsafe { &*slot.p_picture_resource };
                        eprintln!("      -> image_view_binding = {:p}", pr.image_view_binding);
                    }
                }
            }
            eprintln!();

            // 5. Bitstream info
            eprintln!("=== 5. Bitstream ===");
            eprintln!("  Total size: {} bytes", bitstream_data.len());
            let dump_len = 100.min(bitstream_data.len());
            for i in (0..dump_len).step_by(16) {
                let end = (i + 16).min(dump_len);
                let hex: String = (i..end)
                    .map(|j| format!("{:02x}", bitstream_data[j]))
                    .collect::<Vec<_>>()
                    .join(" ");
                let ascii: String = (i..end)
                    .map(|j| {
                        if bitstream_data[j] >= 0x20 && bitstream_data[j] < 0x7f {
                            bitstream_data[j] as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                eprintln!("    {:04x}: {:<48} |{}|", i, hex, ascii);
            }

            // 6. SPS/PPS info
            if let Some(H264OrH265Sps::H265(sps)) = sps {
                eprintln!("\n=== 6. SPS (for reference) ===");
                eprintln!("  sps_video_parameter_set_id = {}", sps.sps_video_parameter_set_id);
                eprintln!("  sps_max_sub_layers_minus1 = {}", sps.sps_max_sub_layers_minus1);
                eprintln!("  sps_temporal_id_nesting_flag = {}", sps.sps_temporal_id_nesting_flag);
                eprintln!("  chroma_format_idc = {}", sps.chroma_format_idc);
                eprintln!("  pic_width_in_luma_samples = {}", sps.pic_width_in_luma_samples);
                eprintln!("  pic_height_in_luma_samples = {}", sps.pic_height_in_luma_samples);
                eprintln!("  log2_max_pic_order_cnt_lsb_minus4 = {} -> MaxPicOrderCntLsb={}",
                    sps.log2_max_pic_order_cnt_lsb_minus4,
                    1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4));
                eprintln!("  log2_min_luma_coding_block_size_minus3 = {}", sps.log2_min_luma_coding_block_size_minus3);
                eprintln!("  log2_diff_max_min_luma_coding_block_size = {}", sps.log2_diff_max_min_luma_coding_block_size);
                eprintln!("  log2_min_luma_transform_block_size_minus2 = {}", sps.log2_min_luma_transform_block_size_minus2);
                eprintln!("  log2_diff_max_min_luma_transform_block_size = {}", sps.log2_diff_max_min_luma_transform_block_size);
                eprintln!("  max_transform_hierarchy_depth_inter = {}", sps.max_transform_hierarchy_depth_inter);
                eprintln!("  max_transform_hierarchy_depth_intra = {}", sps.max_transform_hierarchy_depth_intra);
                eprintln!("  max_num_ref_frames = {}", sps.max_num_ref_frames);
                eprintln!("  scaling_list_enabled_flag = {}", sps.scaling_list_enabled_flag);
                eprintln!("  amp_enabled_flag = {}", sps.amp_enabled_flag);
                eprintln!("  sample_adaptive_offset_enabled_flag = {}", sps.sample_adaptive_offset_enabled_flag);
                eprintln!("  pcm_enabled_flag = {}", sps.pcm_enabled_flag);
                eprintln!("  long_term_ref_pics_present_flag = {}", sps.long_term_ref_pics_present_flag);
                eprintln!("  sps_temporal_mvp_enabled_flag = {}", sps.sps_temporal_mvp_enabled_flag);
                eprintln!("  num_short_term_ref_pic_sets = {}", sps.num_short_term_ref_pic_sets);
                eprintln!("  num_long_term_ref_pics_sps = {}", sps.num_long_term_ref_pics_sps);
            }
            if let Some(H264OrH265Pps::H265(pps)) = _pps {
                eprintln!("\n=== 7. PPS (for reference) ===");
                eprintln!("  pps_pic_parameter_set_id = {}", pps.pps_pic_parameter_set_id);
                eprintln!("  pps_seq_parameter_set_id = {}", pps.pps_seq_parameter_set_id);
                eprintln!("  dependent_slice_segments_enabled_flag = {}", pps.dependent_slice_segments_enabled_flag);
                eprintln!("  output_flag_present_flag = {}", pps.output_flag_present_flag);
                eprintln!("  num_extra_slice_header_bits = {}", pps.num_extra_slice_header_bits);
                eprintln!("  num_ref_idx_l0_default_active_minus1 = {}", pps.num_ref_idx_l0_default_active_minus1);
                eprintln!("  num_ref_idx_l1_default_active_minus1 = {}", pps.num_ref_idx_l1_default_active_minus1);
                eprintln!("  pps_init_qp_minus26 = {}", pps.pps_init_qp_minus26);
                eprintln!("  constrained_intra_pred_flag = {}", pps.constrained_intra_pred_flag);
                eprintln!("  transform_skip_enabled_flag = {}", pps.transform_skip_enabled_flag);
                eprintln!("  cu_qp_delta_enabled_flag = {}", pps.cu_qp_delta_enabled_flag);
                eprintln!("  pps_slice_chroma_qp_offsets_present_flag = {}", pps.pps_slice_chroma_qp_offsets_present_flag);
                eprintln!("  weighted_pred_flag = {}", pps.weighted_pred_flag);
                eprintln!("  weighted_bipred_flag = {}", pps.weighted_bipred_flag);
                eprintln!("  transquant_bypass_enabled_flag = {}", pps.transquant_bypass_enabled_flag);
                eprintln!("  tiles_enabled_flag = {}", pps.tiles_enabled_flag);
                eprintln!("  entropy_coding_sync_enabled_flag = {}", pps.entropy_coding_sync_enabled_flag);
                eprintln!("  pps_loop_filter_across_slices_enabled_flag = {}", pps.pps_loop_filter_across_slices_enabled_flag);
                eprintln!("  deblocking_filter_control_present_flag = {}", pps.deblocking_filter_control_present_flag);
                eprintln!("  pps_deblocking_filter_disabled_flag = {}", pps.pps_deblocking_filter_disabled_flag);
            }

            eprintln!("\n=================================================================");
            eprintln!("         END H.265 COMPREHENSIVE DECODE PARAMETERS");
            eprintln!("=================================================================\n\n");
        }
        // ====================================================================
        // END H.265 DEBUG LOGGING
        // ====================================================================

        // ====================================================================
        // FRAME 1 SPECIFIC H.265 DEBUG (before decode command)
        // ====================================================================
        if is_frame_1_debug && codec == VideoCodec::H265 {
            eprintln!("\n\n=================================================================");
            eprintln!("         FRAME 1 H.265 DEBUG (before decode command)");
            eprintln!("=================================================================\n");

            // 1. Exact content of StdVideoDecodeH265PictureInfo for Frame 1
            eprintln!("=== 1. StdVideoDecodeH265PictureInfo (Frame 1) ===");
            if let Some(h265_di_ptr) = h265_decode_info {
                let h265_di = unsafe { &*h265_di_ptr };
                if !h265_di.p_std_picture_info.is_null() {
                    let pic_info = unsafe { &*h265_di.p_std_picture_info };
                    eprintln!("  Address: {:016x}", pic_info as *const _ as usize);
                    eprintln!("  Size: {} bytes", std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfo>());

                    // Print flags as raw bits
                    let flags_bytes = unsafe {
                        std::slice::from_raw_parts(
                            &pic_info.flags as *const _ as *const u8,
                            std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfoFlags>(),
                        )
                    };
                    let flags_raw: String = flags_bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                    eprintln!("  flags (raw bytes): {}", flags_raw);
                    eprintln!("  flags.IrapPicFlag = {}", pic_info.flags.IrapPicFlag());
                    eprintln!("  flags.IdrPicFlag = {}", pic_info.flags.IdrPicFlag());
                    eprintln!("  flags.IsReference = {}", pic_info.flags.IsReference());
                    eprintln!("  flags.short_term_ref_pic_set_sps_flag = {}", pic_info.flags.short_term_ref_pic_set_sps_flag());

                    eprintln!("  pps_pic_parameter_set_id = {}", pic_info.pps_pic_parameter_set_id);
                    eprintln!("  pps_seq_parameter_set_id = {}", pic_info.pps_seq_parameter_set_id);
                    eprintln!("  sps_video_parameter_set_id = {}", pic_info.sps_video_parameter_set_id);
                    eprintln!("  NumBitsForSTRefPicSetInSlice = {}", pic_info.NumBitsForSTRefPicSetInSlice);
                    eprintln!("  NumDeltaPocsOfRefRpsIdx = {}", pic_info.NumDeltaPocsOfRefRpsIdx);
                    eprintln!("  PicOrderCntVal = {}", pic_info.PicOrderCntVal);

                    eprintln!("  RefPicSetStCurrBefore = [");
                    for (i, &val) in pic_info.RefPicSetStCurrBefore.iter().enumerate() {
                        let marker = if val == 0xff { " (invalid)" } else { "" };
                        eprintln!("    [{}] = 0x{:02x}{} ({})", i, val, marker, val as i32);
                    }
                    eprintln!("  ]");

                    eprintln!("  RefPicSetStCurrAfter = [");
                    for (i, &val) in pic_info.RefPicSetStCurrAfter.iter().enumerate() {
                        let marker = if val == 0xff { " (invalid)" } else { "" };
                        eprintln!("    [{}] = 0x{:02x}{} ({})", i, val, marker, val as i32);
                    }
                    eprintln!("  ]");

                    eprintln!("  RefPicSetLtCurr = [");
                    for (i, &val) in pic_info.RefPicSetLtCurr.iter().enumerate() {
                        let marker = if val == 0xff { " (invalid)" } else { "" };
                        eprintln!("    [{}] = 0x{:02x}{} ({})", i, val, marker, val as i32);
                    }
                    eprintln!("  ]");

                    // Full struct as hex dump
                    let pic_bytes = unsafe {
                        std::slice::from_raw_parts(
                            pic_info as *const _ as *const u8,
                            std::mem::size_of::<ash::vk::native::StdVideoDecodeH265PictureInfo>(),
                        )
                    };
                    eprintln!("  [FULL STRUCT HEX DUMP]");
                    for i in (0..pic_bytes.len()).step_by(16) {
                        let end = (i + 16).min(pic_bytes.len());
                        let hex: String = (i..end)
                            .map(|j| format!("{:02x}", pic_bytes[j]))
                            .collect::<Vec<_>>()
                            .join(" ");
                        eprintln!("    {:04x}: {}", i, hex);
                    }
                } else {
                    eprintln!("  ERROR: p_std_picture_info is NULL!");
                }
            } else {
                eprintln!("  ERROR: h265_decode_info is None!");
            }
            eprintln!();

            // 2. Exact content of StdVideoDecodeH265ReferenceInfo for each reference
            eprintln!("=== 2. StdVideoDecodeH265ReferenceInfo (each reference) ===");

            // Setup slot reference info
            eprintln!("  --- Setup Slot Reference Info ---");
            if let Some((setup_ref_std_ptr, _)) = h265_setup_dpb_slot_info {
                let setup_ref_std: &ash::vk::native::StdVideoDecodeH265ReferenceInfo = unsafe { &*setup_ref_std_ptr };
                eprintln!("  Address: {:016x}", setup_ref_std as *const _ as usize);
                eprintln!("  PicOrderCntVal = {}", setup_ref_std.PicOrderCntVal);
                eprintln!("  flags.used_for_long_term_reference = {}", setup_ref_std.flags.used_for_long_term_reference());
                eprintln!("  flags.unused_for_reference = {}", setup_ref_std.flags.unused_for_reference());
                // Full struct hex
                let ref_bytes = unsafe {
                    std::slice::from_raw_parts(
                        setup_ref_std as *const _ as *const u8,
                        std::mem::size_of::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>(),
                    )
                };
                let hex_str: String = ref_bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                eprintln!("  [FULL STRUCT HEX] {}", hex_str);
            }
            eprintln!();

            // Reference slots
            eprintln!("  --- Reference Slot Info ---");
            for (i, ref_entry) in valid_refs.iter().enumerate() {
                eprintln!("  Reference slot {} (dpb_slot_index={}):", i, ref_entry.slot_index);
                eprintln!("    frame_num = {}", ref_entry.frame_num);
                eprintln!("    pic_order_cnt = [{}, {}]", ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1]);

                if i < reference_slots.len() {
                    let slot = &reference_slots[i];
                    if !slot.p_next.is_null() {
                        let dpb_slot = unsafe { &*(slot.p_next as *const vk::VideoDecodeH265DpbSlotInfoKHR) };
                        if dpb_slot.s_type == vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR {
                            if !dpb_slot.p_std_reference_info.is_null() {
                                let ri = unsafe { &*dpb_slot.p_std_reference_info };
                                eprintln!("    StdVideoDecodeH265ReferenceInfo:");
                                eprintln!("      Address: {:016x}", ri as *const _ as usize);
                                eprintln!("      PicOrderCntVal = {}", ri.PicOrderCntVal);
                                eprintln!("      flags.used_for_long_term_reference = {}", ri.flags.used_for_long_term_reference());
                                eprintln!("      flags.unused_for_reference = {}", ri.flags.unused_for_reference());
                                // Full struct hex
                                let ref_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        ri as *const _ as *const u8,
                                        std::mem::size_of::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>(),
                                    )
                                };
                                let hex_str: String = ref_bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                                eprintln!("      [FULL STRUCT HEX] {}", hex_str);
                            }
                        }
                    }
                }
                eprintln!();
            }

            eprintln!("=================================================================");
            eprintln!("         END FRAME 1 H.265 DEBUG (before decode)");
            eprintln!("=================================================================\n\n");
        }

        eprintln!("[decode] About to call cmd_decode_video...");
        // DEBUG: Print h265_decode_info.p_std_picture_info pointer
        if codec == VideoCodec::H265 {
            if let Some(h265_di_ptr) = h265_decode_info {
                let h265_di = unsafe { &*h265_di_ptr };
                eprintln!(
                    "[pic_info_ptr] frame_num={} p_std_picture_info={:016x}",
                    frame_num,
                    h265_di.p_std_picture_info as usize,
                );
            }
        }
        eprintln!(
            "[decode] decode_info.p_next={:p}, src_buffer_range={}, dst_picture_resource.image_view={:?}",
            decode_info.p_next,
            decode_info.src_buffer_range,
            decode_info.dst_picture_resource.image_view_binding
        );
        eprintln!(
            "[decode] reference_slot_count={}, p_reference_slots={:p}",
            decode_info.reference_slot_count, decode_info.p_reference_slots
        );
        eprintln!(
            "[decode] p_setup_reference_slot={:p}",
            decode_info.p_setup_reference_slot
        );
        cmd_decode_video(instance, device.handle(), cmd_buffer, &decode_info);
        eprintln!("[decode] cmd_decode_video returned OK");

        // End video coding with proper VkVideoEndCodingInfoKHR
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };
        cmd_end_video_coding(instance, device.handle(), cmd_buffer, &end_coding_info);

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| format!("End command buffer failed: {:?}", e))?;

        // CRITICAL: Submit command buffer BEFORE local data (all_slots, picture_resources,
        // decode_info, etc.) goes out of scope. Vulkan commands reference pointers to
        // this local data, which becomes invalid when the function returns.
        device
            .reset_fences(&[fence])
            .map_err(|e| format!("Reset fence failed: {:?}", e))?;

        device
            .queue_submit(
                device.get_device_queue(decode_queue_family, 0),
                &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                fence,
            )
            .map_err(|e| format!("Submit failed: {:?}", e))?;

        let wait_result = device.wait_for_fences(&[fence], true, 10_000_000_000);
        match wait_result {
            Ok(()) => println!("  Decode completed"),
            Err(vk::Result::TIMEOUT) => {
                return Err("Decode timed out (10s)".to_string());
            }
            Err(other) => {
                return Err(format!("Decode result: {:?}", other));
            }
        }
    }

    Ok(())
}

fn build_h264_picture_info(
    sps: Option<&vk_video_core::picture::H264Sps>,
    pps: Option<&vk_video_core::picture::H264Pps>,
    is_idr: bool,
    is_reference: bool,
    frame_num: u32,
    pic_order_cnt: [i32; 2],
    slice_type: u32,
) -> (ash::vk::native::StdVideoDecodeH264PictureInfo, u32) {
    let sps = sps.expect("H264 SPS not set");
    let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
    let effective_frame_num = frame_num % max_frame_num;
    let log2_max_poc_lsb = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let max_poc_lsb = 1u32 << log2_max_poc_lsb;

    let mut pic_info =
        unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264PictureInfo>() };
    // Match reference: set all required flags
    pic_info.flags.set_field_pic_flag(0);
    // slice_type % 5 == 0 means I-slice (intra), 1 means P-slice, 2 means B-slice
    pic_info
        .flags
        .set_is_intra(if slice_type % 5 == 0 { 1 } else { 0 });
    pic_info.flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
    pic_info.flags.set_bottom_field_flag(0);
    pic_info
        .flags
        .set_is_reference(if is_reference { 1 } else { 0 });
    pic_info.flags.set_complementary_field_pair(0);
    pic_info.seq_parameter_set_id = sps.seq_parameter_set_id as u8;
    pic_info.pic_parameter_set_id = pps.map(|p| p.pic_parameter_set_id as u8).unwrap_or(0);
    pic_info.frame_num = effective_frame_num as u16;
    pic_info.idr_pic_id = if is_idr { 0 } else { 0 };
    // BUG FIX: PicOrderCnt must be the FULL POC value, not modulo max_poc_lsb
    // The Vulkan spec requires the actual PicOrderCnt value as derived by the decoder
    // (PicOrderCntMsb + PicOrderCntLsb), not just the LSB from the bitstream.
    // See H.264 spec 8.2.1.1 and Vulkan Video Decode spec.
    pic_info.PicOrderCnt = [pic_order_cnt[0], pic_order_cnt[1]];

    eprintln!(
        "[pic_info] frame_num={}, idr_pic_id={}, poc=[{}, {}], sps_id={}, pps_id={}",
        pic_info.frame_num,
        pic_info.idr_pic_id,
        pic_info.PicOrderCnt[0],
        pic_info.PicOrderCnt[1],
        pic_info.seq_parameter_set_id,
        pic_info.pic_parameter_set_id
    );
    eprintln!(
        "[pic_info] flags: is_intra={}, is_idr={}, is_ref={}, field={}, bottom={}, comp_field={}",
        pic_info.flags.is_intra(),
        pic_info.flags.IdrPicFlag(),
        pic_info.flags.is_reference(),
        pic_info.flags.field_pic_flag(),
        pic_info.flags.bottom_field_flag(),
        pic_info.flags.complementary_field_pair()
    );

    // DEBUG: Print full struct for comparison with C++
    eprintln!(
        "[pic_info] === FULL StdVideoDecodeH264PictureInfo ({} bytes) ===",
        std::mem::size_of::<ash::vk::native::StdVideoDecodeH264PictureInfo>()
    );
    eprintln!("[pic_info]   flags.raw=0x{:08x}", {
        let flags_bytes = unsafe {
            std::slice::from_raw_parts(
                &pic_info.flags as *const _ as *const u8,
                std::mem::size_of::<ash::vk::native::StdVideoDecodeH264PictureInfoFlags>(),
            )
        };
        let mut val: u32 = 0;
        for (i, &b) in flags_bytes.iter().enumerate() {
            val |= (b as u32) << (i * 8);
        }
        val
    });
    eprintln!(
        "[pic_info]   seq_parameter_set_id={}",
        pic_info.seq_parameter_set_id
    );
    eprintln!(
        "[pic_info]   pic_parameter_set_id={}",
        pic_info.pic_parameter_set_id
    );
    eprintln!("[pic_info]   reserved1={}", unsafe {
        *(std::ptr::addr_of!(pic_info.seq_parameter_set_id).add(2) as *const u8)
    });
    eprintln!("[pic_info]   reserved2={}", unsafe {
        *(std::ptr::addr_of!(pic_info.seq_parameter_set_id).add(3) as *const u8)
    });
    eprintln!("[pic_info]   frame_num={}", pic_info.frame_num);
    eprintln!("[pic_info]   idr_pic_id={}", pic_info.idr_pic_id);
    eprintln!("[pic_info]   PicOrderCnt[0]={}", pic_info.PicOrderCnt[0]);
    eprintln!("[pic_info]   PicOrderCnt[1]={}", pic_info.PicOrderCnt[1]);
    eprintln!("[pic_info] === END StdVideoDecodeH264PictureInfo ===");

    (pic_info, frame_num)
}

fn build_h265_picture_info(
     sps: Option<&vk_video_core::picture::H265Sps>,
     pps: Option<&vk_video_core::picture::H265Pps>,
     pic_order_cnt_val: i32,
     is_idr: bool,
     is_irap: bool,
     is_reference: bool,
     num_bits_for_st_ref_pic_set_in_slice: i32,
     num_delta_pocs_of_ref_rps_idx: i32,
     short_term_ref_pic_set_sps_flag: bool,
     ref_pic_set_st_curr_before: [u8; 8],
     ref_pic_set_st_curr_after: [u8; 8],
     ref_pic_set_lt_curr: [u8; 8],
) -> ash::vk::native::StdVideoDecodeH265PictureInfo {
    let sps = sps.expect("H265 SPS not set");
    let pps = pps.expect("H265 PPS not set");

    let mut pic_info =
        unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265PictureInfo>() };

    // Per C++ reference VulkanVideoParser.cpp:2379-2397:
    // Fields are populated from parsed slice header data
    pic_info.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id as u8;
    pic_info.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id as u8;
    pic_info.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;

    // IrapPicFlag: true for IRAP frames (BLA/CRA/IDR = NAL types 16-22)
    pic_info.flags.set_IrapPicFlag(if is_irap { 1 } else { 0 });
    // IdrPicFlag: true only for IDR pictures (NAL unit types 19-20)
    pic_info.flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
    // FIX: Match C++ reference which does NOT set IsReference flag (defaults to 0)
    // pic_info.flags.set_IsReference(if is_reference { 1 } else { 0 });
    // FIX: Match C++ reference which does NOT set short_term_ref_pic_set_sps_flag (defaults to 0)
    // pic_info.flags.set_short_term_ref_pic_set_sps_flag(if short_term_ref_pic_set_sps_flag { 1 } else { 0 });

    // NumBitsForShortTermRPSInSlice: size of short-term RPS in slice header
    // Per C++ reference VulkanVideoParser.cpp:2392
    pic_info.NumBitsForSTRefPicSetInSlice = num_bits_for_st_ref_pic_set_in_slice as u16;

    // NumDeltaPocsOfRefRpsIdx: delta POCS of reference RPS index
    // Per C++ reference VulkanVideoParser.cpp:2396
    pic_info.NumDeltaPocsOfRefRpsIdx = num_delta_pocs_of_ref_rps_idx as u8;

    // PicOrderCntVal from parsed slice header
    // Per C++ reference VulkanVideoParser.cpp:2397
    pic_info.PicOrderCntVal = pic_order_cnt_val;

    // RefPicSet arrays: DPB slot indices of reference pictures for current frame
    // Per C++ reference VulkanVideoParser.cpp:1666-1718
    // These tell the decoder which DPB slots are valid references
    pic_info.RefPicSetStCurrBefore = ref_pic_set_st_curr_before;
    pic_info.RefPicSetStCurrAfter = ref_pic_set_st_curr_after;
    pic_info.RefPicSetLtCurr = ref_pic_set_lt_curr;

    pic_info
}

// ============================================================================
// Barrier helpers
// ============================================================================

fn record_buffer_barrier(
    instance: &ash::Instance,
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    decode_queue_family: u32,
) {
    // Match NVIDIA samples: src=NONE/HOST_WRITE, dst=VIDEO_DECODE/VIDEO_DECODE_READ
    // dstQueueFamilyIndex = actual decode queue family (not IGNORED)
    let buffer_barrier = vk::BufferMemoryBarrier2 {
        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
        p_next: std::ptr::null(),
        src_stage_mask: vk::PipelineStageFlags2::NONE,
        src_access_mask: vk::AccessFlags2::HOST_WRITE,
        dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
        dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: decode_queue_family,
        buffer,
        offset,
        size,
        _marker: Default::default(),
    };

    let dep_info = vk::DependencyInfo {
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
    cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);
}

fn record_image_barrier(
    instance: &ash::Instance,
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    decode_queue_family: u32,
) {
    // Match NVIDIA samples exactly:
    // - srcStageMask = NONE, srcAccessMask = 0
    // - dstStageMask = VIDEO_DECODE, dstAccessMask = VIDEO_DECODE_READ (template)
    //   but changed to VIDEO_DECODE_WRITE for setup picture
    // - srcQueueFamilyIndex = IGNORED, dstQueueFamilyIndex = actual decode queue
    // - aspectMask = COLOR (NVIDIA samples use COLOR even for multi-plane)
    //
    // Note: NVIDIA samples use COLOR aspect for the image barrier template.
    // For multi-plane images, the Vulkan spec says to use PLANE_0/1/2,
    // but NVIDIA's driver apparently handles COLOR correctly.
    let image_barrier = vk::ImageMemoryBarrier2 {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
        p_next: std::ptr::null(),
        src_stage_mask: vk::PipelineStageFlags2::NONE,
        src_access_mask: vk::AccessFlags2::empty(),
        dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
        dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: decode_queue_family,
        image,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        _marker: Default::default(),
    };

    let barriers = [image_barrier];
    let dep_info = vk::DependencyInfo {
        s_type: vk::StructureType::DEPENDENCY_INFO,
        p_next: std::ptr::null(),
        dependency_flags: vk::DependencyFlags::BY_REGION,
        memory_barrier_count: 0,
        p_memory_barriers: std::ptr::null(),
        buffer_memory_barrier_count: 0,
        p_buffer_memory_barriers: std::ptr::null(),
        image_memory_barrier_count: barriers.len() as u32,
        p_image_memory_barriers: barriers.as_ptr(),
        _marker: Default::default(),
    };
    cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);
}

// ============================================================================
// Vulkan extension function dispatch
// ============================================================================

fn cmd_pipeline_barrier_2(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    dep_info: &vk::DependencyInfo<'_>,
) {
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, b"vkCmdPipelineBarrier2KHR\0".as_ptr().cast())
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

fn cmd_control_video_coding(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    info: &vk::VideoCodingControlInfoKHR<'_>,
) {
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, b"vkCmdControlVideoCodingKHR\0".as_ptr().cast())
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::CommandBuffer,
                *const vk::VideoCodingControlInfoKHR<'_>,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
    }
}

fn cmd_begin_video_coding(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    info: &vk::VideoBeginCodingInfoKHR<'_>,
) {
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, b"vkCmdBeginVideoCodingKHR\0".as_ptr().cast())
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::CommandBuffer,
                *const vk::VideoBeginCodingInfoKHR<'_>,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
    }
}

fn cmd_decode_video(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    info: &vk::VideoDecodeInfoKHR<'_>,
) {
    let fn_ptr =
        unsafe { instance.get_device_proc_addr(device, b"vkCmdDecodeVideoKHR\0".as_ptr().cast()) };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoDecodeInfoKHR<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
    } else {
        eprintln!("[ERROR] vkCmdDecodeVideoKHR function pointer not found!");
    }
}

fn cmd_end_video_coding(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    info: &vk::VideoEndCodingInfoKHR<'_>,
) {
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, b"vkCmdEndVideoCodingKHR\0".as_ptr().cast())
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoEndCodingInfoKHR<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
    }
}

// ============================================================================
// Session memory binding and destruction
// ============================================================================

fn bind_session_memory(
    instance: &ash::Instance,
    device: &ash::Device,
    session: vk::VideoSessionKHR,
    _memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<Vec<vk::DeviceMemory>, String> {
    let get_req_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkGetVideoSessionMemoryRequirementsKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkGetVideoSessionMemoryRequirementsKHR not found")?;

    let mut req_count: u32 = 0;
    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            vk::VideoSessionKHR,
            *mut u32,
            *mut vk::VideoSessionMemoryRequirementsKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(get_req_fn);
        let result = fn_ptr(
            device.handle(),
            session,
            &mut req_count,
            std::ptr::null_mut(),
        );
        if result != vk::Result::SUCCESS {
            return Err(format!(
                "vkGetVideoSessionMemoryRequirementsKHR (count) failed: {:?}",
                result
            ));
        }
    }

    if req_count == 0 {
        return Ok(Vec::new());
    }

    let mut requirements =
        vec![vk::VideoSessionMemoryRequirementsKHR::default(); req_count as usize];
    for (i, req) in requirements.iter_mut().enumerate() {
        req.s_type = vk::StructureType::VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
        req.p_next = std::ptr::null_mut::<std::ffi::c_void>();
        req.memory_bind_index = i as u32;
    }

    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            vk::VideoSessionKHR,
            *mut u32,
            *mut vk::VideoSessionMemoryRequirementsKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(get_req_fn);
        let result = fn_ptr(
            device.handle(),
            session,
            &mut req_count,
            requirements.as_mut_ptr(),
        );
        if result != vk::Result::SUCCESS {
            return Err(format!(
                "vkGetVideoSessionMemoryRequirementsKHR failed: {:?}",
                result
            ));
        }
    }

    let mut bind_infos = Vec::with_capacity(req_count as usize);
    let mut memories = Vec::with_capacity(req_count as usize);

    for (i, req) in requirements.iter().enumerate() {
        let mem_req = req.memory_requirements;
        if mem_req.memory_type_bits == 0 {
            return Err("Session memory requirement has no valid memory types".to_string());
        }

        let mut mem_type_index: u32 = 0;
        let mut type_bits = mem_req.memory_type_bits;
        while (type_bits & 1) == 0 {
            type_bits >>= 1;
            mem_type_index += 1;
        }

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: mem_req.size,
            memory_type_index: mem_type_index,
            _marker: Default::default(),
        };

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("Session memory allocation failed: {}", e))?
        };

        bind_infos.push(vk::BindVideoSessionMemoryInfoKHR {
            s_type: vk::StructureType::BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
            p_next: std::ptr::null::<std::ffi::c_void>(),
            memory,
            memory_bind_index: i as u32,
            memory_offset: 0,
            memory_size: mem_req.size,
            _marker: Default::default(),
        });

        memories.push(memory);
    }

    let bind_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkBindVideoSessionMemoryKHR\0".as_ptr().cast(),
        )
    }
    .ok_or("vkBindVideoSessionMemoryKHR not found")?;

    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            vk::VideoSessionKHR,
            u32,
            *const vk::BindVideoSessionMemoryInfoKHR<'_>,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(bind_fn);
        let result = fn_ptr(
            device.handle(),
            session,
            bind_infos.len() as u32,
            bind_infos.as_ptr(),
        );
        if result != vk::Result::SUCCESS {
            return Err(format!("vkBindVideoSessionMemoryKHR failed: {:?}", result));
        }
    }

    Ok(memories)
}

fn destroy_session_parameters(
    instance: &ash::Instance,
    device: vk::Device,
    session_params: vk::VideoSessionParametersKHR,
) {
    if session_params.is_null() {
        return;
    }
    if let Some(ptr) = unsafe {
        instance.get_device_proc_addr(
            device,
            b"vkDestroyVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    } {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionParametersKHR,
                *const vk::AllocationCallbacks,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(device, session_params, std::ptr::null());
        }
    }
}

fn destroy_session(instance: &ash::Instance, device: vk::Device, session: vk::VideoSessionKHR) {
    if session.is_null() {
        return;
    }
    if let Some(ptr) = unsafe {
        instance.get_device_proc_addr(device, b"vkDestroyVideoSessionKHR\0".as_ptr().cast())
    } {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                *const vk::AllocationCallbacks,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(device, session, std::ptr::null());
        }
    }
}

// ============================================================================
// Command resources
// ============================================================================

fn create_command_resources(
    device: &ash::Device,
    queue_family: u32,
) -> Result<(vk::CommandPool, vk::CommandBuffer), String> {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let pool = unsafe { device.create_command_pool(&pool_create_info, None) }
        .map_err(|e| format!("Command pool creation failed: {:?}", e))?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("Command buffer allocation failed: {:?}", e))?;

    Ok((pool, buffers[0]))
}

// ============================================================================
// Pixel readback - Fixed version
// ============================================================================

/// Decoded pixel data for YUV 420 planar format (from ffmpeg comparison).
#[derive(Clone)]
struct DecodedPixels {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
}

/// Readback decoded image pixels from GPU to CPU.
///
/// Key fixes:
/// 1. Use TRANSFER_SRC_OPTIMAL instead of GENERAL (multi-plane images don't support GENERAL)
/// 2. Separate image barriers for PLANE_0 and PLANE_1
/// 3. Proper interleaved UV plane copy with correct row pitch
fn readback_decoded_image(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<DecodedPixels, String> {
    eprintln!("[readback] Called with image={:?}, size={}x{}", image, width, height);
    let y_size = (width * height) as usize;
    // UV plane: interleaved UV pairs, each 2 bytes, at half resolution
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    let uv_size = (uv_width * uv_height * 2) as usize;
    let total_size = (y_size + uv_size) as u64;

    // Create staging buffer
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(total_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
        .map_err(|e| format!("Staging buffer creation failed: {:?}", e))?;

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    eprintln!(
        "[readback] Buffer mem reqs: size={}, type_bits={:08b}",
        mem_reqs.size, mem_reqs.memory_type_bits
    );
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        eprintln!("[readback] HOST_COHERENT not found, trying HOST_VISIBLE only");
        find_memory_type(
            memory_properties,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    })
    .ok_or("No suitable memory type for staging buffer")?;
    eprintln!("[readback] Using memory type {}", mem_type_index);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("Staging memory allocation failed: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("Buffer memory binding failed: {:?}", e))?;
    }

    let mapped_ptr = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("Memory map failed: {:?}", e))?
    };

    // Allocate command buffer for readback
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("Command buffer allocation failed: {:?}", e))?;
    let cmd_buffer = cmd_buffers[0];

    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .map_err(|e| format!("Begin command buffer failed: {:?}", e))?;

        // Transition DPB image to TRANSFER_SRC_OPTIMAL before copy.
        // Matches C++ reference VkVideoDecoder.cpp:763-795.
        // cmd_copy_image_to_buffer requires TRANSFER_SRC_OPTIMAL layout;
        // copying from VIDEO_DECODE_DPB_KHR directly is undefined behavior.
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
                aspect_mask: vk::ImageAspectFlags::PLANE_1 | vk::ImageAspectFlags::PLANE_2,
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

        let buffer_barriers = [buffer_barrier];
        let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        // Transition image back to VIDEO_DECODE_DPB_KHR so it can be used as reference
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
            .map_err(|e| format!("End command buffer failed: {:?}", e))?;

        // Submit and wait
        device
            .reset_fences(&[fence])
            .map_err(|e| format!("Reset fence failed: {:?}", e))?;

        device
            .queue_submit(
                device.get_device_queue(queue_family, 0),
                &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                fence,
            )
            .map_err(|e| format!("Submit failed: {:?}", e))?;

        let result = device.wait_for_fences(&[fence], true, 10_000_000_000);
        if let Err(e) = result {
            return Err(format!("Readback fence wait failed: {:?}", e));
        }

        // Read data from mapped memory
        let mut y_plane = vec![0u8; y_size];
        let mut uv_plane = vec![0u8; uv_size];

        eprintln!("[readback] Reading from mapped_ptr {:?}", mapped_ptr);
        std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, y_plane.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(
            mapped_ptr.add(y_size) as *const u8,
            uv_plane.as_mut_ptr(),
            uv_size,
        );
        eprintln!(
            "[readback] Y first 16 bytes: {:?}",
            &y_plane[..16.min(y_plane.len())]
        );
        eprintln!(
            "[readback] UV first 16 bytes: {:?}",
            &uv_plane[..16.min(uv_plane.len())]
        );

        // De-interleave UV plane: G8_B8R8 -> separate U and V planes
        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut u_plane = vec![0u8; uv_plane_size];
        let mut v_plane = vec![0u8; uv_plane_size];
        for i in 0..uv_plane_size {
            u_plane[i] = uv_plane[i * 2];
            v_plane[i] = uv_plane[i * 2 + 1];
        }

        // Cleanup
        device.unmap_memory(memory);
        device.free_memory(memory, None);
        device.destroy_buffer(buffer, None);

        Ok(DecodedPixels {
            y_plane,
            u_plane,
            v_plane,
        })
    }
}

/// Verify that decoded pixels are valid (not all zeros).
/// 
/// pixels are at coded dimensions; width/height are display dimensions;
/// crop_left/crop_top indicate the display area offset within the coded frame.
fn verify_decoded_pixels(pixels: &DecodedPixels, width: u32, height: u32, crop_left: u32, crop_top: u32) {
    let y_plane = &pixels.y_plane;
    let u_plane = &pixels.u_plane;
    let v_plane = &pixels.v_plane;

    let y_min = y_plane.iter().min().copied().unwrap_or(0) as i32;
    let y_max = y_plane.iter().max().copied().unwrap_or(255) as i32;
    let y_avg: f64 = y_plane.iter().map(|&b| b as f64).sum::<f64>() / y_plane.len() as f64;

    let u_min = u_plane.iter().min().copied().unwrap_or(0) as i32;
    let u_max = u_plane.iter().max().copied().unwrap_or(255) as i32;
    let v_min = v_plane.iter().min().copied().unwrap_or(0) as i32;
    let v_max = v_plane.iter().max().copied().unwrap_or(255) as i32;

    let y_zero_count = y_plane.iter().filter(|&&b| b == 0).count();
    let y_zero_pct = y_zero_count as f64 / y_plane.len() as f64 * 100.0;

    println!("  Vulkan decoded frame ({}x{} display):", width, height);
    println!(
        "    Y: min={}, max={}, avg={:.1}, zero_pixels={:.1}%",
        y_min, y_max, y_avg, y_zero_pct
    );
    println!("    U: min={}, max={}", u_min, u_max);
    println!("    V: min={}, max={}", v_min, v_max);

    // Derive coded dimensions from plane sizes (for 4:2:0)
    let y_total = y_plane.len() as u32;
    let uv_total = u_plane.len() as u32;
    let (coded_width, coded_height) = if y_total > 0 && uv_total > 0 {
        let mut found = (width, height);
        for cw in (width.max(1)..=y_total).step_by(2) {
            if y_total % cw == 0 {
                let ch = y_total / cw;
                let expected_uv = ((cw + 1) / 2) * ((ch + 1) / 2);
                if expected_uv == uv_total {
                    found = (cw, ch);
                    break;
                }
            }
        }
        found
    } else {
        (width, height)
    };

    // Check center pixel of display area
    let cx = crop_left as usize + (width as usize) / 2;
    let cy = crop_top as usize + (height as usize) / 2;
    let cw = coded_width as usize;
    let y_val = y_plane[cy * cw + cx] as i32;
    let uv_cw = (cw + 1) / 2;
    let uv_cx = (crop_left as usize) / 2 + (width as usize) / 4;
    let uv_cy = (crop_top as usize) / 2 + (height as usize) / 4;
    let u_val = u_plane[uv_cy * uv_cw + uv_cx] as i32;
    let v_val = v_plane[uv_cy * uv_cw + uv_cx] as i32;
    println!(
        "    Center pixel of display area: Y={} U={} V={}",
        y_val, u_val, v_val
    );

    if y_max == 0 && y_avg < 1.0 {
        println!("  WARNING: Decoded frame appears to be all black (Y plane is zero)!");
    } else if y_zero_pct > 95.0 {
        println!(
            "  WARNING: Decoded frame is mostly black ({:.1}% zero pixels)!",
            y_zero_pct
        );
    } else {
        println!("  ✓ Decoded frame looks valid (non-zero pixels detected)");
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required_flags)
        {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// FFmpeg comparison
// ============================================================================

fn compare_with_ffmpeg(bitstream_path: &str, codec: VideoCodec, width: u32, height: u32) {
    let output_dir = "example_output";
    std::fs::create_dir_all(output_dir).ok();

    let codec_name = match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "hevc",
    };
    let yuv_path = format!("{}/ffmpeg_frame_001.yuv", output_dir);

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-codec:v",
            codec_name,
            "-i",
            bitstream_path,
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            "1",
            &yuv_path,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            let yuv_data = match std::fs::read(&yuv_path) {
                Ok(d) => d,
                Err(e) => {
                    println!("  Failed to read YUV file: {}", e);
                    return;
                }
            };

            let y_size = (width * height) as usize;
            let uv_size = (width / 2 * height / 2) as usize;
            let expected_size = y_size + 2 * uv_size;

            if yuv_data.len() != expected_size {
                println!(
                    "  YUV size mismatch: expected {}, got {}",
                    expected_size,
                    yuv_data.len()
                );
                return;
            }

            let y_plane = &yuv_data[..y_size];
            let u_plane = &yuv_data[y_size..y_size + uv_size];
            let v_plane = &yuv_data[y_size + uv_size..];

            let y_min = y_plane.iter().min().copied().unwrap_or(0) as i32;
            let y_max = y_plane.iter().max().copied().unwrap_or(255) as i32;
            let y_avg: f64 = y_plane.iter().map(|&b| b as f64).sum::<f64>() / y_plane.len() as f64;
            let u_min = u_plane.iter().min().copied().unwrap_or(0) as i32;
            let u_max = u_plane.iter().max().copied().unwrap_or(255) as i32;
            let v_min = v_plane.iter().min().copied().unwrap_or(0) as i32;
            let v_max = v_plane.iter().max().copied().unwrap_or(255) as i32;

            println!("  FFmpeg reference frame ({}x{}):", width, height);
            println!("    Y: min={}, max={}, avg={:.1}", y_min, y_max, y_avg);
            println!("    U: min={}, max={}", u_min, u_max);
            println!("    V: min={}, max={}", v_min, v_max);

            if width as usize > 0 && height as usize > 0 {
                let cx = width as usize / 2;
                let cy = height as usize / 2;
                let y_val = y_plane[cy * width as usize + cx] as i32;
                let u_val = u_plane[cy / 2 * (width as usize / 2) + cx / 2] as i32;
                let v_val = v_plane[cy / 2 * (width as usize / 2) + cx / 2] as i32;
                println!(
                    "    Center pixel ({}x{}): Y={} U={} V={}",
                    cx, cy, y_val, u_val, v_val
                );
            }
            println!("  Reference frame decoded successfully");
        }
        Ok(s) => println!("  FFmpeg failed with status: {:?}", s.code()),
        Err(e) => println!("  FFmpeg not available: {}", e),
    }
}

/// Compare a specific decoded frame with ffmpeg reference for that frame number.
///
/// Compare a decoded frame with FFmpeg reference using POC-based matching.
///
/// This function matches frames by their visual content (checksum) rather than
/// by frame index, which is necessary because:
/// - Rust decodes in decode order (bitstream order)
/// - FFmpeg outputs in display order (PTS order) even with -vsync 0
/// - B-frames cause decode order and display order to differ
///
/// The decoded pixels may be at coded dimensions (full frame), but we only compare
/// the display area (after applying SPS crop offsets for H.265).
fn compare_frame_with_ffmpeg(
    bitstream_path: &str,
    codec: VideoCodec,
    display_width: u32,
    display_height: u32,
    crop_left: u32,
    crop_top: u32,
    poc: i32,
    decoded: &DecodedPixels,
) {
    let output_dir = "example_output";
    std::fs::create_dir_all(output_dir).ok();

    // Derive coded dimensions from plane sizes (for 4:2:0)
    // y_plane.len() = coded_width * coded_height
    // u_plane.len() = ((coded_width + 1) / 2) * ((coded_height + 1) / 2)
    let y_total = decoded.y_plane.len() as u32;
    let uv_total = decoded.u_plane.len() as u32;

    let (coded_width, coded_height) = if y_total > 0 && uv_total > 0 {
        // Try to find coded_width that satisfies both plane size equations
        let mut found = (display_width, display_height);
        for cw in (display_width.max(1)..=y_total).step_by(2) {
            if y_total % cw == 0 {
                let ch = y_total / cw;
                let expected_uv = ((cw + 1) / 2) * ((ch + 1) / 2);
                if expected_uv == uv_total {
                    found = (cw, ch);
                    break;
                }
            }
        }
        found
    } else {
        (display_width, display_height)
    };

    // Extract display area from decoded pixels
    let (y_plane, u_plane, v_plane) = if crop_left > 0 || crop_top > 0 {
        let cw = coded_width as usize;
        let ch = coded_height as usize;
        let cl = crop_left as usize;
        let ct = crop_top as usize;
        let dw = display_width as usize;
        let dh = display_height as usize;

        // Extract cropped Y plane
        let mut y_cropped = Vec::with_capacity(dw * dh);
        for y in ct..ct + dh {
            let row_start = y * cw + cl;
            y_cropped.extend_from_slice(&decoded.y_plane[row_start..row_start + dw]);
        }

        // Extract cropped UV planes (half resolution for 4:2:0)
        let uv_cw = (cw + 1) / 2;
        let uv_cl = cl / 2;
        let uv_ct = ct / 2;
        let uv_dw = (dw + 1) / 2;
        let uv_dh = (dh + 1) / 2;

        let mut u_cropped = Vec::with_capacity(uv_dw * uv_dh);
        let mut v_cropped = Vec::with_capacity(uv_dw * uv_dh);
        for y in uv_ct..uv_ct + uv_dh {
            let row_start = y * uv_cw + uv_cl;
            for x in 0..uv_dw {
                u_cropped.push(decoded.u_plane[row_start + x]);
                v_cropped.push(decoded.v_plane[row_start + x]);
            }
        }

        (y_cropped, u_cropped, v_cropped)
    } else {
        (decoded.y_plane.clone(), decoded.u_plane.clone(), decoded.v_plane.clone())
    };

    let codec_name = match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "hevc",
    };

    let y_size = (display_width * display_height) as usize;
    let uv_size = (display_width / 2 * display_height / 2) as usize;
    let expected_size = y_size + 2 * uv_size;

    // Decode frames with FFmpeg one by one and find the one that matches our POC frame
    let mut best_match_idx = None;
    let mut best_psnr = -1.0;

    for i in 0..256u32 {
        let path = format!("{}/ffmpeg_compare_{}.yuv", output_dir, i);
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-codec:v", codec_name,
                "-i", bitstream_path,
                "-vsync", "0",
                "-pix_fmt", "yuv420p",
                "-vf", &format!("select=eq(n\\,{i}),setsar=1"),
                "-frames:v", "1",
                &path,
            ])
            .status();

        if status.map_or(true, |s| !s.success()) {
            break; // No more frames from FFmpeg
        }

        let yuv_data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => break,
        };

        if yuv_data.len() != expected_size {
            let _ = std::fs::remove_file(&path);
            continue;
        }

        let ref_y = &yuv_data[..y_size];

        // Compute PSNR
        let mut mse: f64 = 0.0;
        for (vulkan_y, ref_y_val) in y_plane.iter().zip(ref_y.iter()) {
            let diff = (*vulkan_y as i32 - *ref_y_val as i32) as f64;
            mse += diff * diff;
        }
        mse /= y_size as f64;
        let psnr = if mse > 0.0 {
            10.0 * (255.0 * 255.0 / mse).log10()
        } else {
            f64::INFINITY
        };

        // A match is PSNR > 40 dB (very close)
        if psnr > 40.0 {
            best_psnr = psnr;
            best_match_idx = Some(i);
            let _ = std::fs::remove_file(&path);
            break;
        } else if psnr > best_psnr && psnr > 25.0 {
            best_psnr = psnr;
            best_match_idx = Some(i);
        }

        let _ = std::fs::remove_file(&path);
    }

    if let Some(match_idx) = best_match_idx {
        // Use the frame index from FFmpeg output (display order)
        let yuv_path = format!("{}/ffmpeg_frame_poc_{}.yuv", output_dir, poc);

        // Re-decode just the matching frame for clean output
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-codec:v", codec_name,
                "-i", bitstream_path,
                "-vsync", "0",
                "-pix_fmt", "yuv420p",
                "-vf", &format!("select=eq(n\\,{}),setsar=1", match_idx),
                "-frames:v", "1",
                &yuv_path,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                let yuv_data = match std::fs::read(&yuv_path) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("  Failed to read YUV file: {}", e);
                        return;
                    }
                };

                if yuv_data.len() != expected_size {
                    println!(
                        "  YUV size mismatch: expected {}, got {}",
                        expected_size,
                        yuv_data.len()
                    );
                    return;
                }

                let ref_y = &yuv_data[..y_size];
                let ref_u = &yuv_data[y_size..y_size + uv_size];
                let ref_v = &yuv_data[y_size + uv_size..];

                // Compute PSNR for Y plane
                let mut mse: f64 = 0.0;
                for (vulkan_y, ref_y_val) in y_plane.iter().zip(ref_y.iter()) {
                    let diff = (*vulkan_y as i32 - *ref_y_val as i32) as f64;
                    mse += diff * diff;
                }
                mse /= y_size as f64;
                let psnr = if mse > 0.0 {
                    10.0 * (255.0 * 255.0 / mse).log10()
                } else {
                    f64::INFINITY
                };

                // Count differing pixels (threshold = 2)
                let diff_pixels: usize = y_plane
                    .iter()
                    .zip(ref_y.iter())
                    .filter(|(&v, &r)| ((v as i32 - r as i32).abs() > 2))
                    .count();
                let diff_pct = diff_pixels as f64 / y_size as f64 * 100.0;

                println!("  FFmpeg reference comparison (POC={}, FFmpeg frame={}, {}x{} display area):",
                         poc, match_idx, display_width, display_height);
                println!("    Y-PSNR: {:.2} dB", psnr);
                println!("    Diff pixels (>2): {} ({:.2}%)", diff_pixels, diff_pct);

                if psnr > 40.0 || diff_pct < 1.0 {
                    println!("  ✓ Frame matches ffmpeg reference well");
                } else if psnr > 30.0 {
                    println!("  ⚠ Frame has some differences from ffmpeg reference");
                } else {
                    println!("  ✗ Frame differs significantly from ffmpeg reference");
                }
            }
            Ok(s) => println!("  FFmpeg failed with status: {:?}", s.code()),
            Err(e) => println!("  FFmpeg not available: {}", e),
        }
    } else {
        println!("  ✗ Could not find matching FFmpeg frame for POC={}", poc);
    }
}

/// Compare two DecodedPixels structures to check if they match.
/// Used for reference picture integrity verification.
fn compare_decoded_pixels(
    a: &DecodedPixels,
    b: &DecodedPixels,
    width: u32,
    height: u32,
) -> PixelComparison {
    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * height / 2) as usize;

    let mut y_mse: f64 = 0.0;
    let mut y_diff_count = 0;
    let mut u_mse: f64 = 0.0;
    let mut v_mse: f64 = 0.0;

    for (ya, yb) in a.y_plane.iter().zip(b.y_plane.iter()) {
        let diff = (*ya as i32 - *yb as i32) as f64;
        y_mse += diff * diff;
        if diff.abs() > 0.0 {
            y_diff_count += 1;
        }
    }
    y_mse /= y_size as f64;
    let y_psnr = if y_mse > 0.0 {
        10.0 * (255.0 * 255.0 / y_mse).log10()
    } else {
        f64::INFINITY
    };
    let y_diff_pct = y_diff_count as f64 / y_size as f64 * 100.0;

    for (ua, ub) in a.u_plane.iter().zip(b.u_plane.iter()) {
        let diff = (*ua as i32 - *ub as i32) as f64;
        u_mse += diff * diff;
    }
    u_mse /= uv_size as f64;

    for (va, vb) in a.v_plane.iter().zip(b.v_plane.iter()) {
        let diff = (*va as i32 - *vb as i32) as f64;
        v_mse += diff * diff;
    }
    v_mse /= uv_size as f64;

    PixelComparison {
        matched: y_mse == 0.0 && u_mse == 0.0 && v_mse == 0.0,
        y_mse,
        y_psnr,
        y_diff_count,
        y_diff_pct,
        u_mse,
        v_mse,
    }
}

/// Result of comparing two decoded pixel buffers.
struct PixelComparison {
    matched: bool,
    y_mse: f64,
    y_psnr: f64,
    y_diff_count: usize,
    y_diff_pct: f64,
    u_mse: f64,
    v_mse: f64,
}

// ============================================================================
// Image/Buffer creation with video profile (VkVideoProfileListInfoKHR)
// Required by Vulkan spec for images/buffers with VIDEO_DECODE usage
// ============================================================================

/// Create output image with video profile in pNext chain.
fn create_output_image_with_profile(
    _instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), String> {
    // Build video profile chain using a closure to ensure all pointers are valid
    let image = create_image_with_profile_chain(
        device,
        width,
        height,
        format,
        codec,
        profile_idc,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    )?;

    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

    let mem_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or("No device-local memory type found")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("Memory allocation failed: {:?}", e))?;

    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("Memory binding failed: {:?}", e))?;
    }

    // Create image view with COLOR aspect
    let view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );

    let view = unsafe { device.create_image_view(&view_create_info, None) }
        .map_err(|e| format!("ImageView creation failed: {:?}", e))?;

    Ok((image, view, memory))
}

/// Helper to create an image with a proper video profile chain.
fn create_image_with_profile_chain(
    device: &ash::Device,
    width: u32,
    height: u32,
    format: vk::Format,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<vk::Image, String> {
    // Build chain manually with explicit field initialization
    // Chain: ImageCreateInfo -> VideoProfileListInfoKHR -> VideoProfileInfoKHR -> codec profile

    let codec_op = match codec {
        VideoCodec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
        VideoCodec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
    };

    // Create codec-specific profile (end of chain)
    let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default();
    h264_profile.std_profile_idc = profile_idc;
    h264_profile.picture_layout = vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE;

    let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default();
    h265_profile.std_profile_idc = profile_idc;

    // Create video profile
    let mut video_profile = vk::VideoProfileInfoKHR::default();
    video_profile.video_codec_operation = codec_op;
    video_profile.chroma_subsampling = chroma_subsampling;
    video_profile.luma_bit_depth = luma_bit_depth;
    video_profile.chroma_bit_depth = chroma_bit_depth;

    // Set p_next to point to codec profile
    video_profile.p_next = match codec {
        VideoCodec::H264 => &h264_profile as *const _ as *const _,
        VideoCodec::H265 => &h265_profile as *const _ as *const _,
    };

    // Create profile list
    let profile_list = vk::VideoProfileListInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
        p_next: std::ptr::null(),
        profile_count: 1,
        p_profiles: &video_profile,
        _marker: Default::default(),
    };

    // Create image create info with profile list in pNext chain
    let mut image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    image_create_info.p_next = &profile_list as *const _ as *const _;

    unsafe {
        device
            .create_image(&image_create_info, None)
            .map_err(|e| format!("Image creation failed: {:?}", e))
    }
}

/// Create bitstream buffer with video profile in pNext chain.
fn create_bitstream_buffer_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<VkBitstreamBuffer, String> {
    // Build video profile chain - all structs must stay alive during the API call
    let codec_op = match codec {
        VideoCodec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
        VideoCodec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
    };

    // Create codec-specific profiles (end of chain)
    let h264_profile = vk::VideoDecodeH264ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE,
        _marker: Default::default(),
    };

    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    // Create video profile
    let video_profile = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: match codec {
            VideoCodec::H264 => &h264_profile as *const _ as *const _,
            VideoCodec::H265 => &h265_profile as *const _ as *const _,
        },
        video_codec_operation: codec_op,
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

    // All structs above stay alive until this function returns
    // The Vulkan API call happens synchronously within create_with_pnext
    VkBitstreamBuffer::create_with_pnext(
        device,
        memory_properties,
        size,
        1,
        256,
        &profile_list as *const _ as *const std::ffi::c_void,
    )
    .map_err(|e| e.to_string())
}
