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
    bitstream::BitstreamPacket, h264::H264Parser, h265::H265Parser, DetectedVideoFormat,
    ParseResult, VideoParser,
};
use vk_video_vulkan::{
    buffer::BitstreamBuffer as VkBitstreamBuffer, image::create_output_image, VideoDeviceBuilder,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.h264|h265>", args[0]);
        eprintln!("Available: born_trailer.h264, big_buck_bunney.h265");
        std::process::exit(1);
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
    let vulkan = match VideoDeviceBuilder::new().with_validation(true).build() {
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

    // Align coded extent to picture access granularity from capabilities
    // Using: ((width + alignWidth - 1) & ~(alignWidth - 1))
    let align_width = video_caps.picture_access_granularity.width;
    let align_height = video_caps.picture_access_granularity.height;
    let coded_extent = vk::Extent2D {
        width: (parsed.coded_width + align_width - 1) & !(align_width - 1),
        height: (parsed.coded_height + align_height - 1) & !(align_height - 1),
    };
    println!(
        "  Coded extent aligned: {}x{} -> {}x{}\n",
        parsed.coded_width, parsed.coded_height, coded_extent.width, coded_extent.height
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

    let mut dpb_manager = DpbManager::new(dpb_slots);

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

    for (frame_idx, au) in access_units.iter().enumerate() {
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

        // Flush the mapped memory
        let flush_size = ((au.data.len() as u64 + 63) / 64 * 64).max(64);
        bs_buffer.flush_range(0, flush_size).ok();

        // Select DPB slot for this frame
        let output_slot;
        if au.is_idr {
            // IDR: invalidate all DPB entries, use slot 0
            dpb_manager.invalidate_all();
            output_slot = 0;
        } else {
            // P/B frame: find or recycle a slot
            output_slot = dpb_manager.find_or_recycle_slot().unwrap_or(0);
        }

        let output_view = dpb_views[output_slot as usize];
        let output_img = dpb_image_handles[output_slot as usize];

        // Record decode command (pass DPB manager for reference slot setup)
        // CRITICAL: Use actual bitstream size, NOT aligned.
        // The Vulkan spec requires minBitstreamBufferSizeAlignment for BUFFER SIZE,
        // not for the decode range. Using aligned range causes the decoder to read
        // garbage bytes from previous frames that weren't overwritten.
        let actual_bs_size = au.data.len() as u64;
        let result = record_decode_command(
            &vulkan.instance,
            &vulkan.device,
            command_buffer,
            decode_qf,
            session,
            session_params,
            bs_buffer.buffer(),
            0,
            actual_bs_size,
            output_view,
            output_img,
            coded_extent,
            codec,
            parsed.sps.as_ref(),
            parsed.pps.as_ref(),
            &au.slice_offsets,
            au.frame_num,
            au.pic_order_cnt,
            au.is_idr,
            au.is_reference,
            au.slice_type,
            &dpb_manager,
            output_slot,
            is_first_frame,
            &au.data,
            decoder_reset_done,
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

        // Submit and wait
        unsafe {
            vulkan
                .device
                .reset_fences(&[fence])
                .expect("Failed to reset fence");
            vulkan
                .device
                .queue_submit(
                    vulkan.device.get_device_queue(decode_qf, 0),
                    &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                    fence,
                )
                .expect("Failed to submit");

            let wait_result = vulkan
                .device
                .wait_for_fences(&[fence], true, 10_000_000_000);
            match wait_result {
                Ok(()) => println!("  Decode completed"),
                Err(vk::Result::TIMEOUT) => {
                    eprintln!("  Decode timed out (10s)");
                    break;
                }
                Err(other) => {
                    eprintln!("  Decode result: {:?}", other);
                    break;
                }
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
                "[dpb] Registered frame {} in slot {} as reference",
                au.frame_num, output_slot
            );

            // Apply sliding window reference picture marking after registering new reference
            dpb_manager.apply_sliding_window(au.frame_num);
        }

        // Readback and verify decoded pixels
        println!("  Reading back decoded pixels...");
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
        // NOTE: Do NOT change last_access here - DecodeWrite is the important access
        // that needs to be visible for subsequent decodes using this as reference.
        dpb_manager.set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

        match &decoded_pixels_result {
            Ok(pixels) => {
                verify_decoded_pixels(pixels, parsed.coded_width, parsed.coded_height);

                // Compare with ffmpeg reference
                compare_frame_with_ffmpeg(
                    bitstream_path,
                    codec,
                    parsed.coded_width,
                    parsed.coded_height,
                    frame_idx,
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
            max_dpb_slots,
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
    fn find_or_recycle_slot(&mut self) -> Option<u32> {
        for i in 0..self.max_dpb_slots as usize {
            if !self.entries[i].is_valid {
                return Some(i as u32);
            }
        }
        // Recycle the oldest short-term reference (smallest FrameNumWrap)
        let mut oldest_idx = None;
        let mut oldest_wrap = u32::MAX;
        for i in 0..self.max_dpb_slots as usize {
            if self.entries[i].is_valid {
                let wrap = self.entries[i].frame_num;
                if wrap < oldest_wrap {
                    oldest_wrap = wrap;
                    oldest_idx = Some(i as u32);
                }
            }
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

/// Minimal bit reader for slice header parsing (no EPB removal needed for slice headers).
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
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()?;
        }
        Some(val)
    }

    fn read_ue(&mut self) -> Option<u32> {
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
        Some((1 << leading_zeros) - 1 + value)
    }

    fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        if ue & 1 != 0 {
            Some((ue + 1) as i32 / 2)
        } else {
            Some(-((ue as i32) / 2))
        }
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
) -> Option<(u32, u32, i32, i32, [i32; 2])> {
    // Returns: (first_mb_in_slice, frame_num, pic_order_cnt_lsb, pic_order_cnt_msb, pic_order_cnt)
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

    Some((
        first_mb_in_slice,
        frame_num,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
    ))
}

/// Parse H.265 slice header to extract frame boundary info.
///
/// Based on VulkanH265Parser.cpp:2119-2217 for slice header parsing,
/// and lines 2757-2799 for POC computation.
///
/// Returns: (first_slice_in_pic, pic_order_cnt_lsb, pic_order_cnt_msb, pic_order_cnt, is_idr, is_reference, slice_type)
fn parse_h265_slice_header(
    nal_data: &[u8],
    sps: &vk_video_core::picture::H265Sps,
    pps: &vk_video_core::picture::H265Pps,
    nal_unit_type: u8,
    nuh_temporal_id_plus1: u8,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
) -> Option<(bool, i32, i32, [i32; 2], bool, bool, u32)> {
    if nal_data.len() < 3 {
        return None;
    }

    // H.265 NAL header is 2 bytes, payload starts at byte 2
    let payload = &nal_data[2..];
    let mut r = SliceBitReader::new(payload);

    // first_slice_segment_in_pic_flag
    let first_slice_segment_in_pic_flag = r.read_bit()? == 1;

    // For RAP pictures: no_output_of_prior_pics_flag
    let is_rap = nal_unit_type >= 16 && nal_unit_type <= 23;
    if is_rap {
        let _no_output_of_prior_pics_flag = r.read_bit().unwrap_or(0);
    }

    // pic_parameter_set_id (ue)
    let _pps_id = r.read_ue().unwrap_or(0);

    // If this is a dependent slice segment, we don't parse full header
    // Use info from first slice - for frame detection we only care about first slices
    if !first_slice_segment_in_pic_flag {
        return None;
    }

    // Skip num_extra_slice_header_bits if present
    if pps.num_extra_slice_header_bits > 0 {
        let _extra_bits = r.read_bits(pps.num_extra_slice_header_bits as u32).unwrap_or(0);
    }

    // slice_type (ue): 0=B, 1=P, 2=I
    let slice_type = r.read_ue().unwrap_or(0);

    // pic_output_flag (if present)
    if pps.output_flag_present_flag {
        let _pic_output_flag = r.read_bit().unwrap_or(0);
    }

    // colour_plane_id (if separate_colour_plane_flag)
    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.read_bits(2).unwrap_or(0);
    }

    // Determine is_idr from NAL type
    let is_idr = nal_unit_type == 19 || nal_unit_type == 20; // IDR_W_RADL or IDR_N_LP

    // Parse pic_order_cnt_lsb (skip for IDR)
    let pic_order_cnt_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let pic_order_cnt_lsb = if is_idr {
        0i32
    } else {
        r.read_bits(pic_order_cnt_lsb_bits)? as i32
    };

    // Compute PicOrderCntMsb per H.265 spec 8.3.1
    let pic_order_cnt_msb = if is_rap {
        // For IRAP pictures with NoRaslOutputFlag (BLA or IDR), MSB is 0
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

    Some((
        first_slice_segment_in_pic_flag,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        is_idr,
        is_reference,
        slice_type,
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

        let (nal_type, is_irap, is_au_delimiter, is_slice, is_params, is_trailing) = match codec {
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
                        false,
                    )
                } else {
                    (0, false, false, false, false, false)
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
                    let is_trailing_type = matches!(t, 0..=1 | 40..=41);
                    (
                        t as usize,
                        is_irap_type,
                        is_aud,
                        is_slice_type,
                        is_params_type,
                        is_trailing_type,
                    )
                } else {
                    (0, false, false, false, false, false)
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
                      num_bits_for_st_ref_pic_set_in_slice: 0,
                      num_delta_pocs_of_ref_rps_idx: 0,
                  });
                current_au_data.clear();
                current_slice_offsets.clear();
                in_frame = false;
            }
            offset = end;
            continue;
        }

        // For H.265: end current AU at next IRAP after trailing
        if codec == VideoCodec::H265 && found_first_frame && in_frame && is_trailing {
            if !current_au_data.is_empty() {
                  access_units.push(AccessUnit {
                      data: current_au_data.clone(),
                      slice_offsets: current_slice_offsets.clone(),
                      frame_num: current_frame_num,
                      pic_order_cnt: current_poc,
                      is_idr: current_is_idr,
                      is_reference: current_is_reference,
                      slice_type: current_slice_type,
                      num_bits_for_st_ref_pic_set_in_slice: 0,
                      num_delta_pocs_of_ref_rps_idx: 0,
                  });
                current_au_data.clear();
                current_slice_offsets.clear();
                in_frame = false;
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
                        if let Some((first_mb, frame_num, poc_lsb, poc_msb, poc)) =
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
                                          num_bits_for_st_ref_pic_set_in_slice: 0,
                                          num_delta_pocs_of_ref_rps_idx: 0,
                                      });
                                    current_au_data.clear();
                                    current_slice_offsets.clear();
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
                                          num_bits_for_st_ref_pic_set_in_slice: 0,
                                          num_delta_pocs_of_ref_rps_idx: 0,
                                      });
                                    current_au_data.clear();
                                    current_slice_offsets.clear();
                                }

                                // First slice of a frame - set frame properties from parsed header
                                current_is_idr = slice_is_idr;
                                current_is_reference = slice_is_reference;
                                current_poc = poc;
                                current_slice_type = slice_type;
                                prev_frame_num += 1;
                                current_frame_num = prev_frame_num;

                                // Update POC tracking
                                prev_pic_order_cnt_lsb = poc_lsb;
                                prev_pic_order_cnt_msb = poc_msb;

                                // For IDR frames, reset POC tracking
                                if slice_is_idr {
                                    prev_pic_order_cnt_lsb = 0;
                                    prev_pic_order_cnt_msb = 0;
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
                          num_bits_for_st_ref_pic_set_in_slice: 0,
                          num_delta_pocs_of_ref_rps_idx: 0,
                      });
                    current_au_data.clear();
                    current_slice_offsets.clear();
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

            // Add slice to current AU with start code
            let slice_offset = current_au_data.len();
            current_au_data.extend_from_slice(&[0x00, 0x00, 0x01]);
            current_au_data.extend_from_slice(nal_data);
            current_slice_offsets.push(slice_offset as u32);
        } else if in_frame {
            // Include other NALs of the current frame (SEI, etc.)
            current_au_data.extend_from_slice(&[0x00, 0x00, 0x01]);
            current_au_data.extend_from_slice(nal_data);
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
             num_bits_for_st_ref_pic_set_in_slice: 0,
             num_delta_pocs_of_ref_rps_idx: 0,
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
        .map(|s| s.pic_width_in_luma_samples as u32)
        .unwrap_or(0);
    let coded_height = sps
        .as_ref()
        .map(|s| s.pic_height_in_luma_samples as u32)
        .unwrap_or(0);
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

fn convert_h265_sps(
    sps: &vk_video_core::picture::H265Sps,
) -> ash::vk::native::StdVideoH265SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoH265SpsFlags>() };
    flags.set_sps_temporal_id_nesting_flag(if sps.sps_temporal_id_nesting_flag {
        1
    } else {
        0
    });
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });

    ash::vk::native::StdVideoH265SequenceParameterSet {
        flags,
        chroma_format_idc: sps.chroma_format_idc as u32,
        pic_width_in_luma_samples: sps.pic_width_in_luma_samples as u32,
        pic_height_in_luma_samples: sps.pic_height_in_luma_samples as u32,
        sps_video_parameter_set_id: sps.sps_video_parameter_set_id,
        sps_max_sub_layers_minus1: sps.sps_max_sub_layers_minus1,
        sps_seq_parameter_set_id: sps.sps_seq_parameter_set_id as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        log2_min_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_luma_coding_block_size: 0,
        log2_min_luma_transform_block_size_minus2: 0,
        log2_diff_max_min_luma_transform_block_size: 0,
        max_transform_hierarchy_depth_inter: 0,
        max_transform_hierarchy_depth_intra: 0,
        num_short_term_ref_pic_sets: 0,
        num_long_term_ref_pics_sps: 0,
        pcm_sample_bit_depth_luma_minus1: 0,
        pcm_sample_bit_depth_chroma_minus1: 0,
        log2_min_pcm_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_pcm_luma_coding_block_size: 0,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: 0,
        conf_win_right_offset: 0,
        conf_win_top_offset: 0,
        conf_win_bottom_offset: 0,
        pProfileTierLevel: std::ptr::null(),
        pDecPicBufMgr: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: std::ptr::null(),
        pLongTermRefPicsSps: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }
}

fn convert_h265_pps(
    pps: &vk_video_core::picture::H265Pps,
) -> ash::vk::native::StdVideoH265PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoH265PpsFlags>() };
    flags.set_dependent_slice_segments_enabled_flag(if pps.dependent_slice_segments_enabled_flag {
        1
    } else {
        0
    });
    flags.set_output_flag_present_flag(if pps.output_flag_present_flag { 1 } else { 0 });
    flags.set_sign_data_hiding_enabled_flag(if pps.sign_data_hiding_enabled_flag {
        1
    } else {
        0
    });
    flags.set_cabac_init_present_flag(if pps.cabac_init_present_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag {
        1
    } else {
        0
    });
    flags.set_transform_skip_enabled_flag(if pps.transform_skip_enabled_flag {
        1
    } else {
        0
    });
    flags.set_cu_qp_delta_enabled_flag(if pps.cu_qp_delta_enabled_flag { 1 } else { 0 });
    flags.set_pps_slice_chroma_qp_offsets_present_flag(
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_weighted_bipred_flag(if pps.weighted_bipred_flag { 1 } else { 0 });
    flags.set_transquant_bypass_enabled_flag(if pps.transquant_bypass_enabled_flag {
        1
    } else {
        0
    });
    flags.set_tiles_enabled_flag(if pps.tiles_enabled_flag { 1 } else { 0 });
    flags.set_entropy_coding_sync_enabled_flag(if pps.entropy_coding_sync_enabled_flag {
        1
    } else {
        0
    });

    ash::vk::native::StdVideoH265PictureParameterSet {
        flags,
        pps_pic_parameter_set_id: pps.pps_pic_parameter_set_id as u8,
        pps_seq_parameter_set_id: pps.pps_seq_parameter_set_id as u8,
        sps_video_parameter_set_id: 0,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as u8,
        init_qp_minus26: pps.pps_init_qp_minus26 as i8,
        diff_cu_qp_delta_depth: 0,
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        pps_beta_offset_div2: 0,
        pps_tc_offset_div2: 0,
        log2_parallel_merge_level_minus2: 0,
        log2_max_transform_skip_block_size_minus2: 0,
        diff_cu_chroma_qp_offset_depth: 0,
        chroma_qp_offset_list_len_minus1: 0,
        cb_qp_offset_list: [0; 6],
        cr_qp_offset_list: [0; 6],
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        reserved3: 0,
        pScalingLists: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }
}

fn convert_h265_vps(
    vps: &vk_video_core::picture::H265Vps,
) -> ash::vk::native::StdVideoH265VideoParameterSet {
    use ash::vk::native::*;

    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265VpsFlags>() };
    flags.set_vps_temporal_id_nesting_flag(if vps.vps_temporal_id_nesting_flag {
        1
    } else {
        0
    });

    let mut ptl = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevel>() };
    ptl.general_profile_idc = StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN;
    ptl.general_level_idc = StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1;

    ash::vk::native::StdVideoH265VideoParameterSet {
        flags,
        vps_video_parameter_set_id: vps.vps_video_parameter_set_id,
        vps_max_sub_layers_minus1: vps.vps_max_sub_layers_minus1,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: 0,
        vps_time_scale: 0,
        vps_num_ticks_poc_diff_one_minus1: 0,
        reserved3: 0,
        pDecPicBufMgr: std::ptr::null(),
        pHrdParameters: std::ptr::null(),
        pProfileTierLevel: &ptl as *const _,
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
    slice_offsets: &[u32],
    frame_num: u32,
    pic_order_cnt: [i32; 2],
    is_idr: bool,
    is_reference: bool,
    slice_type: u32,
    dpb_manager: &DpbManager,
    current_slot_index: u32,
    is_frame_1_debug: bool,
    bitstream_data: &[u8],
    decoder_reset_done: bool,
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
    // Align bitstream_range to minBitstreamBufferSizeAlignment (256 bytes).
    // The Vulkan spec requires src_buffer_range to be aligned to
    // minBitstreamBufferSizeAlignment for H.265 decode.
    let bs_range = (bitstream_range + 255) & !255;
    eprintln!("[decode] Bitstream range: {} bytes -> aligned to {} bytes, slice_count={}, frame_num={}, is_idr={}", 
               bitstream_range, bs_range, slice_offsets.len(), frame_num, is_idr);

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

            // Build RefPicSet arrays from valid DPB references
            // Per C++ reference VulkanVideoParser.cpp:1666-1718
            // These arrays contain DPB slot indices of reference pictures
            // that are used as references for the current frame
            let mut ref_pic_set_st_curr_before = [0xffu8; 8];
            let mut ref_pic_set_st_curr_after = [0xffu8; 8];
            let mut ref_pic_set_lt_curr = [0xffu8; 8];

            // For now, populate RefPicSetStCurrBefore with DPB slot indices of
            // valid reference pictures. This tells the decoder which DPB slots
            // contain valid reference frames for motion compensation.
            // Per C++ reference, these should be filled from parsed RefPicSet data.
            // For a simplified approach: use all valid DPB references as StCurrBefore.
            let valid_refs = dpb_manager.get_references();
            for (i, ref_entry) in valid_refs.iter().enumerate() {
                if i < 8 {
                    // Store DPB slot index (masked to 4 bits like C++ reference)
                    ref_pic_set_st_curr_before[i] = (ref_entry.slot_index as u8) & 0xf;
                }
            }

            let pic_info = build_h265_picture_info(
                sps_h265,
                pps_h265,
                pic_order_cnt[0], // Use actual POC from parsed slice header
                is_idr,           // IdrPicFlag based on NAL unit type (19-20)
                is_irap,          // IrapPicFlag for IRAP frames (BLA/CRA/IDR)
                is_reference,
                0,                // NumBitsForSTRefPicSetInSlice (from slice header)
                0,                // NumDeltaPocsOfRefRpsIdx (from slice header)
                ref_pic_set_st_curr_before,
                ref_pic_set_st_curr_after,
                ref_pic_set_lt_curr,
            );
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
    let (h264_decode_info, h264_ref_info, h264_dpb_slot_info) = match &h264_pic_info {
        Some(pic_info) => {
            let frame_num = h264_frame_num.unwrap_or(0);
            let poc = h264_poc.unwrap_or([0, 0]);
            let h264_decode_info = vk::VideoDecodeH264PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: pic_info as *const _,
                slice_count: slice_offsets.len() as u32,
                p_slice_offsets: slice_offsets.as_ptr(),
                _marker: Default::default(),
            };

            let mut ref_info =
                unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>() };
            ref_info.FrameNum = frame_num as u16;
            ref_info.PicOrderCnt = poc;
            // For frame pictures: top_field_flag=0, bottom_field_flag=0
            // Validation layer IsFrame() requires both flags to be 0
            ref_info.flags.set_top_field_flag(0);
            ref_info.flags.set_bottom_field_flag(0);
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

            // Store ref_info first, then take pointer to it
            let h264_ref_info = Some(ref_info);
            let ref_info_ptr = h264_ref_info.as_ref().unwrap() as *const _;

            let dpb_slot_info = vk::VideoDecodeH264DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: ref_info_ptr,
                _marker: Default::default(),
            };
            eprintln!("[dpb]   ref_info_ptr={:p}", ref_info_ptr);
            eprintln!("[dpb]   dpb_slot_info_ptr={:p}", &dpb_slot_info as *const _);
            eprintln!(
                "[dpb]   p_std_reference_info={:p}",
                dpb_slot_info.p_std_reference_info
            );
            (Some(h264_decode_info), h264_ref_info, Some(dpb_slot_info))
        }
        None => (None, None, None),
    };

    let (h265_decode_info, h265_ref_info, h265_dpb_slot_info) = match &h265_pic_info {
        Some(pic_info) => {
            let h265_decode_info = vk::VideoDecodeH265PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: pic_info as *const _,
                slice_segment_count: slice_offsets.len() as u32,
                p_slice_segment_offsets: slice_offsets.as_ptr(),
                _marker: Default::default(),
            };

            let mut ref_info =
                unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>() };
            ref_info.PicOrderCntVal = 0;
            ref_info.flags.set_used_for_long_term_reference(0);
            ref_info.flags.set_unused_for_reference(0);

            let dpb_slot_info = vk::VideoDecodeH265DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: &ref_info as *const _,
                _marker: Default::default(),
            };
            (Some(h265_decode_info), Some(ref_info), Some(dpb_slot_info))
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
    // For H.264, setup slot has pNext=NULL per C++ reference.
    let (setup_reference_slot, decode_info_pnext, h265_setup_dpb_slot_info) = match codec {
        VideoCodec::H264 => {
            let decode_info_pnext = h264_decode_info.as_ref().unwrap() as *const _ as *const _;
            eprintln!(
                "[dpb] Setup slot: vulkan_slot_index={}, dpb_slot={}",
                current_slot_index, current_slot_index
            );
            let setup_reference_slot = vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                slot_index: current_slot_index as i32,
                p_picture_resource: &dpb_setup_picture_resource,
                _marker: Default::default(),
            };
            (setup_reference_slot, decode_info_pnext, None)
        }
        VideoCodec::H265 => {
            let decode_info_pnext = h265_decode_info.as_ref().unwrap() as *const _ as *const _;
            eprintln!(
                "[dpb] Setup slot: vulkan_slot_index={}, dpb_slot={}",
                current_slot_index, current_slot_index
            );

            // H.265 setup slot requires VkVideoDecodeH265DpbSlotInfoKHR in pNext chain
            // (VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07163)
            let mut setup_ref_std_info =
                unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>() };
            setup_ref_std_info.PicOrderCntVal = pic_order_cnt[0];
            setup_ref_std_info.flags.set_used_for_long_term_reference(0);
            setup_ref_std_info.flags.set_unused_for_reference(0);

            let setup_dpb_slot_info = vk::VideoDecodeH265DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: &setup_ref_std_info as *const _,
                _marker: Default::default(),
            };

            let setup_reference_slot = vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: &setup_dpb_slot_info as *const _ as *const _,
                slot_index: current_slot_index as i32,
                p_picture_resource: &dpb_setup_picture_resource,
                _marker: Default::default(),
            };
            eprintln!(
                "[dpb] H265 setup slot pNext chain: VideoReferenceSlotInfo -> VideoDecodeH265DpbSlotInfoKHR({:p}) -> StdVideoDecodeH265ReferenceInfo({:p})",
                &setup_dpb_slot_info as *const _,
                setup_dpb_slot_info.p_std_reference_info
            );
            (setup_reference_slot, decode_info_pnext, Some((setup_ref_std_info, setup_dpb_slot_info)))
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
    let valid_refs_len = valid_refs.len();
    let mut reference_slots: Vec<vk::VideoReferenceSlotInfoKHR> =
        Vec::with_capacity(valid_refs_len);
    let mut ref_picture_resources: Vec<vk::VideoPictureResourceInfoKHR> =
        Vec::with_capacity(valid_refs_len);

    match codec {
        VideoCodec::H264 => {
            let mut ref_std_infos: Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo> =
                Vec::with_capacity(valid_refs_len);
            let mut ref_dpb_slot_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> =
                Vec::with_capacity(valid_refs_len);

            // First pass: create all ref_std_infos
            for ref_entry in valid_refs.iter() {
                let mut ref_std_info = unsafe {
                    std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>()
                };
                ref_std_info.FrameNum = ref_entry.frame_num as u16;
                ref_std_info.PicOrderCnt = ref_entry.pic_order_cnt;
                // For frame pictures: top_field_flag=0, bottom_field_flag=0
                // Validation layer IsFrame() requires both flags to be 0
                ref_std_info.flags.set_top_field_flag(0);
                ref_std_info.flags.set_bottom_field_flag(0);
                ref_std_info.flags.set_used_for_long_term_reference(0);
                ref_std_info.flags.set_is_non_existing(0);
                ref_std_infos.push(ref_std_info);
            }

            // Second pass: create dpb_slot_infos with stable pointers to ref_std_infos
            for (i, _ref_entry) in valid_refs.iter().enumerate() {
                let ref_info_ptr = &ref_std_infos[i] as *const _;
                let dpb_slot_info = vk::VideoDecodeH264DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_info_ptr,
                    _marker: Default::default(),
                };
                ref_dpb_slot_infos.push(dpb_slot_info);
            }

            // Third pass: create picture resources and reference slots
            for (i, ref_entry) in valid_refs.iter().enumerate() {
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
                    p_next: &ref_dpb_slot_infos[i] as *const _ as *const _,
                    slot_index: ref_entry.slot_index as i32,
                    p_picture_resource: &ref_picture_resources[i],
                    _marker: Default::default(),
                };
                reference_slots.push(ref_slot);

                eprintln!("[dpb] Ref slot {}: vulkan_slot_index={}, dpb_slot={}, frame_num={}, poc=[{}, {}]",
                    i, ref_entry.slot_index, ref_entry.slot_index, ref_entry.frame_num,
                    ref_entry.pic_order_cnt[0], ref_entry.pic_order_cnt[1]);
            }
        }
        VideoCodec::H265 => {
            let mut ref_std_infos: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo> =
                Vec::with_capacity(valid_refs_len);
            let mut ref_dpb_slot_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> =
                Vec::with_capacity(valid_refs_len);

            // First pass: create all ref_std_infos for H.265
            // StdVideoDecodeH265ReferenceInfo only has:
            // - flags (with used_for_long_term_reference and unused_for_reference)
            // - PicOrderCntVal
            for ref_entry in valid_refs.iter() {
                let mut ref_std_info = unsafe {
                    std::mem::zeroed::<ash::vk::native::StdVideoDecodeH265ReferenceInfo>()
                };
                ref_std_info.PicOrderCntVal = ref_entry.pic_order_cnt[0];
                ref_std_info.flags.set_used_for_long_term_reference(0);
                ref_std_info.flags.set_unused_for_reference(0);
                ref_std_infos.push(ref_std_info);
            }

            // Second pass: create dpb_slot_infos with stable pointers to ref_std_infos
            for (i, _ref_entry) in valid_refs.iter().enumerate() {
                let ref_info_ptr = &ref_std_infos[i] as *const _;
                let dpb_slot_info = vk::VideoDecodeH265DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_info_ptr,
                    _marker: Default::default(),
                };
                ref_dpb_slot_infos.push(dpb_slot_info);
            }

            // Third pass: create picture resources and reference slots
            for (i, ref_entry) in valid_refs.iter().enumerate() {
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

                // Chain VkVideoDecodeH265DpbSlotInfoKHR after VkVideoReferenceSlotInfoKHR
                let ref_slot = vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: &ref_dpb_slot_infos[i] as *const _ as *const _,
                    slot_index: ref_entry.slot_index as i32,
                    p_picture_resource: &ref_picture_resources[i],
                    _marker: Default::default(),
                };
                reference_slots.push(ref_slot);

                eprintln!("[dpb] H265 Ref slot {}: vulkan_slot_index={}, dpb_slot={}, frame_num={}, poc={}",
                    i, ref_entry.slot_index, ref_entry.slot_index, ref_entry.frame_num,
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
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .map_err(|e| format!("Begin command buffer failed: {:?}", e))?;

        // === SEPARATE BARRIER FOR REFERENCE PICTURES BEFORE vkCmdBeginVideoCodingKHR ===
        // Match C++ reference VkVideoDecoder.cpp:1186-1200:
        // Reference pictures must be in VIDEO_DECODE_DPB layout before BeginVideoCoding.
        // This ensures memory visibility from previous decode's VIDEO_DECODE_WRITE
        // to current decode's VIDEO_DECODE_READ.
        let mut ref_barriers_before_coding: Vec<vk::ImageMemoryBarrier2> =
            Vec::with_capacity(valid_refs.len());
        for ref_entry in &valid_refs {
            let ref_slot_layout = dpb_manager.get_slot_layout(ref_entry.slot_index);
            let ref_last_access = dpb_manager.get_slot_last_access(ref_entry.slot_index);
            let (src_stage, src_access, old_layout) =
                if ref_slot_layout == vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
                    // Already in correct layout - use src_access based on last access type
                    match ref_last_access {
                        LastAccessType::DecodeWrite => (
                            vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                            vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                        ),
                        LastAccessType::TransferRead => (
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_READ,
                            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                        ),
                        LastAccessType::None => (
                            vk::PipelineStageFlags2::NONE,
                            vk::AccessFlags2::NONE,
                            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                        ),
                    }
                } else {
                    // Layout transition needed
                    (
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        ref_slot_layout,
                    )
                };
            let ref_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: src_stage,
                src_access_mask: src_access,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: ref_entry.image,
                old_layout: old_layout,
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
            ref_barriers_before_coding.push(ref_barrier);
            if src_stage == vk::PipelineStageFlags2::VIDEO_DECODE_KHR {
                eprintln!("[barrier-before-coding] Created ref VISIBILITY barrier for slot {} (already DPB, decode write->read)", 
                    ref_entry.slot_index);
            } else {
                eprintln!("[barrier-before-coding] Created ref barrier for slot {}: {:?} -> VIDEO_DECODE_DPB_KHR",
                    ref_entry.slot_index, ref_slot_layout);
            }
        }
        if !ref_barriers_before_coding.is_empty() {
            let ref_dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 0,
                p_buffer_memory_barriers: std::ptr::null(),
                image_memory_barrier_count: ref_barriers_before_coding.len() as u32,
                p_image_memory_barriers: ref_barriers_before_coding.as_ptr(),
                _marker: Default::default(),
            };
            cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &ref_dep_info);
            eprintln!(
                "[barrier-before-coding] Executed reference picture barrier with {} barriers",
                ref_barriers_before_coding.len()
            );
        }

        // RESET is REQUIRED before the first decode per Vulkan spec.
        // Must be INSIDE a video coding block (between BeginVideoCoding and EndVideoCoding)
        // to satisfy VUID-vkCmdControlVideoCodingKHR-videocoding.
        // Also initializes the session and activates DPB slots (VkVideoDecoder.cpp:1205-1213).
        //
        // The RESET is done INSIDE the same video coding block as the decode, matching
        // the C++ reference. DPB slots become active when referenced in BeginVideoCodingKHR,
        // so we include all slots in the first BeginVideoCodingKHR.
        let all_slots: Vec<vk::VideoReferenceSlotInfoKHR> = reference_slots
            .iter()
            .cloned()
            .chain(std::iter::once(setup_reference_slot))
            .collect();
        eprintln!(
            "[dpb] VkVideoBeginCodingInfoKHR: {} total slots ({} refs + 1 setup)",
            all_slots.len(),
            reference_slots.len()
        );

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

        // Combined barrier AFTER vkCmdBeginVideoCodingKHR: buffer + output image only
        // Reference picture barriers are now done BEFORE vkCmdBeginVideoCodingKHR.
        // srcStageMask=HOST required when srcAccessMask=HOST_WRITE per Vulkan spec
        // Using QUEUE_FAMILY_IGNORED for both src and dst since images use EXCLUSIVE sharing mode
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

        // Build image barriers: output image only (no reference pictures - those are before coding)
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::with_capacity(1);

        // Output image barrier (WRITE access - decoder writes decoded frame here)
        // Match C++ reference VkVideoDecoder.cpp:841-863: use COLOR aspect (not PLANE_0/PLANE_1)
        let output_slot_layout = dpb_manager.get_slot_layout(current_slot_index);
        eprintln!(
            "[barrier-after-coding] Output slot {} current layout: {:?}",
            current_slot_index, output_slot_layout
        );
        if output_slot_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
            let output_barrier = vk::ImageMemoryBarrier2 {
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
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                _marker: Default::default(),
            };
            image_barriers.push(output_barrier);
            eprintln!("[barrier-after-coding] Created output barrier (COLOR): {:?} -> VIDEO_DECODE_DPB_KHR (WRITE)", output_slot_layout);
        } else {
            eprintln!(
                "[barrier-after-coding] Skipping output barrier - already in VIDEO_DECODE_DPB_KHR"
            );
        }

        eprintln!("[barrier-after-coding] Total image barriers: {} (output only, refs handled before coding)", 
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
            if let Some(h264_decode_info_ref) = h264_decode_info.as_ref() {
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
            if let Some(dpb_info) = h264_dpb_slot_info.as_ref() {
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
        // END FRAME 1 DEBUG LOGGING
        // ====================================================================

        // DEBUG: Print setup reference slot's reference info flags right before decode
        if let Some(dpb_info) = h264_dpb_slot_info.as_ref() {
            if !dpb_info.p_std_reference_info.is_null() {
                let ref_info_ptr = unsafe { &*dpb_info.p_std_reference_info };
                eprintln!("[decode] Setup slot ref_info flags: top={}, bottom={}, ltr={}, non_existing={}",
                    ref_info_ptr.flags.top_field_flag(),
                    ref_info_ptr.flags.bottom_field_flag(),
                    ref_info_ptr.flags.used_for_long_term_reference(),
                    ref_info_ptr.flags.is_non_existing());
                eprintln!(
                    "[decode] Setup slot ref_info reserved={:04x}",
                    ref_info_ptr.reserved
                );
                eprintln!(
                    "[decode] Setup slot ref_info FrameNum={}",
                    ref_info_ptr.FrameNum
                );
                // Print individual bytes of flags
                let flags_bytes = unsafe {
                    std::slice::from_raw_parts(&ref_info_ptr.flags as *const _ as *const u8, 4)
                };
                eprintln!(
                    "[decode] Setup slot ref_info flags_bytes=[{:02x}, {:02x}, {:02x}, {:02x}]",
                    flags_bytes[0], flags_bytes[1], flags_bytes[2], flags_bytes[3]
                );
            }
        }
        cmd_decode_video(instance, device.handle(), cmd_buffer, &decode_info);

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
    pic_info
        .flags
        .set_IsReference(if is_reference { 1 } else { 0 });

    // NumBitsForShortTermRPSInSlice: size of short-term RPS in slice header
    // Per C++ reference VulkanVideoParser.cpp:2392
    pic_info.NumBitsForSTRefPicSetInSlice = num_bits_for_st_ref_pic_set_in_slice;

    // NumDeltaPocsOfRefRpsIdx: delta POCS of reference RPS index
    // Per C++ reference VulkanVideoParser.cpp:2396
    pic_info.NumDeltaPocsOfRefRpsIdx = num_delta_pocs_of_ref_rps_idx;

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

        // Transition image planes: VIDEO_DECODE_DPB_KHR -> SHADER_READ_ONLY_OPTIMAL
        // For multi-plane images, use PLANE_0 and PLANE_1 separately.
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
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
        // UV plane has dimensions uv_width x uv_height, each pixel is 2 bytes (UV pair)
        // buffer_row_length=0 means tightly packed: row_pitch = uv_width * 2 bytes
        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
fn verify_decoded_pixels(pixels: &DecodedPixels, width: u32, height: u32) {
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

    println!("  Vulkan decoded frame ({}x{}):", width, height);
    println!(
        "    Y: min={}, max={}, avg={:.1}, zero_pixels={:.1}%",
        y_min, y_max, y_avg, y_zero_pct
    );
    println!("    U: min={}, max={}", u_min, u_max);
    println!("    V: min={}, max={}", v_min, v_max);

    // Check center pixel
    let cx = width as usize / 2;
    let cy = height as usize / 2;
    let y_val = y_plane[cy * width as usize + cx] as i32;
    let uv_width = (width as usize + 1) / 2;
    let uv_height = (height as usize + 1) / 2;
    let u_val = u_plane[cy / 2 * uv_width + cx / 2] as i32;
    let v_val = v_plane[cy / 2 * uv_width + cx / 2] as i32;
    println!(
        "    Center pixel ({}x{}): Y={} U={} V={}",
        cx, cy, y_val, u_val, v_val
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
fn compare_frame_with_ffmpeg(
    bitstream_path: &str,
    codec: VideoCodec,
    width: u32,
    height: u32,
    frame_index: usize,
    decoded: &DecodedPixels,
) {
    let output_dir = "example_output";
    std::fs::create_dir_all(output_dir).ok();

    // Save Rust-decoded frame to file for byte-identical comparison
    let rust_yuv_path = format!("{}/rust_frame_{:03}.yuv", output_dir, frame_index);
    let mut rust_yuv_data =
        Vec::with_capacity(decoded.y_plane.len() + decoded.u_plane.len() + decoded.v_plane.len());
    rust_yuv_data.extend_from_slice(&decoded.y_plane);
    rust_yuv_data.extend_from_slice(&decoded.u_plane);
    rust_yuv_data.extend_from_slice(&decoded.v_plane);
    if let Err(e) = std::fs::write(&rust_yuv_path, &rust_yuv_data) {
        println!(
            "  Failed to save Rust-decoded frame to {}: {}",
            rust_yuv_path, e
        );
    } else {
        println!("  Saved Rust-decoded frame to {}", rust_yuv_path);
    }

    let codec_name = match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "hevc",
    };
    let yuv_path = format!("{}/ffmpeg_frame_{:03}.yuv", output_dir, frame_index);

    // Use ffmpeg to extract the specific frame (0-indexed)
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-codec:v",
            codec_name,
            "-i",
            bitstream_path,
            "-pix_fmt",
            "yuv420p",
            "-vf",
            &format!("select=eq(n\\,{})", frame_index),
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

            let ref_y = &yuv_data[..y_size];
            let ref_u = &yuv_data[y_size..y_size + uv_size];
            let ref_v = &yuv_data[y_size + uv_size..];

            // Compute PSNR for Y plane
            let mut mse: f64 = 0.0;
            for (vulkan_y, ref_y_val) in decoded.y_plane.iter().zip(ref_y.iter()) {
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
            let diff_pixels: usize = decoded
                .y_plane
                .iter()
                .zip(ref_y.iter())
                .filter(|(&v, &r)| ((v as i32 - r as i32).abs() > 2))
                .count();
            let diff_pct = diff_pixels as f64 / y_size as f64 * 100.0;

            println!("  FFmpeg reference comparison:");
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
