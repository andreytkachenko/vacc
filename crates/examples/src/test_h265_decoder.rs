//! Simple test using vk-video-vulkan's H265Decoder to verify it works.
//!
//! This test:
//! 1. Parses big_buck_bunney.h265 to get VPS/SPS/PPS
//! 2. Creates a Vulkan device with video decode support
//! 3. Creates an H265Decoder
//! 4. Creates a video session and session parameters
//! 5. Decodes the first IDR frame
//! 6. Reads back the output and checks if it's non-zero
//!
//! Based on vulkan_decode.rs but uses H265Decoder directly instead of hand-rolled code.

use ash::vk::{self, Handle};
use vk_video_parser::{
    bitstream::BitstreamPacket,
    h265::H265Parser,
    nal::{find_next_start_code, parse_h265_nal_header},
    DetectedVideoFormat, ParseResult, VideoParser,
};
use vk_video_vulkan::{
    buffer::BitstreamBuffer,
    image::create_output_image,
    h265::{H265Decoder, H265RefPictureInfo},
    VideoDeviceBuilder,
};

fn main() {
    let bitstream_path = "big_buck_bunney.h265";

    println!("=== H265Decoder Direct Test ===");
    println!("File: {}\n", bitstream_path);

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Step 1: Parse VPS/SPS/PPS
    println!("--- Step 1: Parse VPS/SPS/PPS ---");
    let parsed = parse_h265(&data);
    println!("  VPS: {:?}, SPS: {:?}, PPS: {:?}", parsed.vps.is_some(), parsed.sps.is_some(), parsed.pps.is_some());
    println!("  Resolution: {}x{}", parsed.coded_width, parsed.coded_height);
    println!("  Profile: {}\n", parsed.profile_idc);

    // Step 2: Initialize Vulkan
    println!("--- Step 2: Vulkan initialization ---");
    let vulkan = VideoDeviceBuilder::new()
        .with_validation(false)
        .build()
        .expect("Failed to init Vulkan");
    let decode_qf = vulkan.queue_families.video_decode.expect("No decode queue");
    println!("  GPU: decode queue family = {}\n", decode_qf);

    // Step 3: Query capabilities
    println!("--- Step 3: Query capabilities ---");
    let (video_caps, _decode_caps) = query_video_decode_capabilities(
        &vulkan,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    );
    println!("  maxDPBSlots: {}", video_caps.max_dpb_slots);
    println!(
        "  pictureAccessGranularity: {}x{}",
        video_caps.picture_access_granularity.width,
        video_caps.picture_access_granularity.height
    );
    println!(
        "  minBitstreamBufferSizeAlignment: {}\n",
        video_caps.min_bitstream_buffer_size_alignment
    );

    // Align coded extent
    let align_w = video_caps.picture_access_granularity.width;
    let align_h = video_caps.picture_access_granularity.height;
    let coded_extent = vk::Extent2D {
        width: (parsed.coded_width + align_w - 1) & !(align_w - 1),
        height: (parsed.coded_height + align_h - 1) & !(align_h - 1),
    };
    println!("  Aligned extent: {}x{}\n", coded_extent.width, coded_extent.height);

    // Step 4: Create video session
    println!("--- Step 4: Create video session ---");
    let max_dpb_slots = 4u32;
    let (session, session_params, _session_mems) = create_video_session(
        &vulkan,
        decode_qf,
        parsed.profile_idc,
        coded_extent,
        max_dpb_slots,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
        parsed.vps.as_ref(),
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    );
    println!("  Session created\n");

    // Step 5: Create H265Decoder
    println!("--- Step 5: Create H265Decoder ---");
    let mut decoder = H265Decoder::new(vulkan.device.clone(), vulkan.instance.clone());
    if let Some(ref v) = parsed.vps {
        decoder.set_vps(v.clone());
    }
    if let Some(ref s) = parsed.sps {
        decoder.set_sps(s.clone());
    }
    if let Some(ref p) = parsed.pps {
        decoder.set_pps(p.clone());
    }
    println!("  H265Decoder created with VPS/SPS/PPS\n");

    // Step 6: Update session parameters with VPS/SPS/PPS using H265Decoder
    println!("--- Step 5b: Update session parameters ---");
    decoder.update_session_parameters(
        session_params,
        parsed.vps.as_ref(),
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    ).expect("Failed to update session parameters");
    println!("  Session parameters updated with VPS/SPS/PPS\n");

    // Step 6: Create output image with video profile
    println!("--- Step 6: Create output image ---");
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
    let (output_image, output_image_view, output_memory) = create_output_image_with_profile(
        &vulkan.device,
        &vulkan.memory_properties,
        coded_extent.width,
        coded_extent.height,
        output_format,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    )
    .expect("Failed to create output image");
    println!("  Output image: {}x{} format={:?}\n", coded_extent.width, coded_extent.height, output_format);

    // Step 7: Create bitstream buffer with video profile
    println!("--- Step 7: Create bitstream buffer ---");
    let max_bs_size = 1_000_000u64;
    let mut bs_buffer = create_bitstream_buffer_with_profile(
        &vulkan.device,
        &vulkan.memory_properties,
        max_bs_size,
        parsed.profile_idc,
        parsed.chroma_subsampling,
        parsed.luma_bit_depth,
        parsed.chroma_bit_depth,
    )
    .expect("Failed to create bitstream buffer");
    println!("  Bitstream buffer: {} bytes\n", max_bs_size);

    // Step 8: Find first IDR frame
    println!("--- Step 8: Find first IDR frame ---");
    let (idr_data, slice_offsets) = find_first_idr(&data);
    println!("  IDR frame size: {} bytes", idr_data.len());
    println!("  Slice count: {}\n", slice_offsets.len());

    // Step 9: Write bitstream data
    bs_buffer.write(&idr_data).expect("Failed to write bitstream");
    
    // Align bitstream range to minBitstreamBufferSizeAlignment
    let bs_range = ((idr_data.len() as u64 + 255) / 256 * 256).max(256);
    bs_buffer.flush_range(0, bs_range).ok();

    // Step 10: Create command resources
    println!("--- Step 9: Create command resources ---");
    let (command_pool, command_buffer) = create_command_resources(&vulkan.device, decode_qf);
    let fence = unsafe {
        vulkan
            .device
            .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
            .expect("Failed to create fence")
    };
    println!("  Command buffer and fence created\n");

    // Step 11: Decode first IDR frame using H265Decoder
    println!("--- Step 10: Decode first IDR frame ---");
    let pic_order_cnt = 0i32;

    // For the first frame, provide dpb_setup_picture pointing to the same image
    // This marks the decoded frame as a reference picture in DPB slot 0
    let picture_resource = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: output_image_view,
        _marker: Default::default(),
    };

    let dpb_setup = H265RefPictureInfo {
        slot_index: 0,
        pic_order_cnt,
        picture_resource,
    };

    let result = decoder.record_decode_command(
        command_buffer,
        session,
        session_params,
        bs_buffer.buffer(),
        0,
        bs_range, // Use aligned bitstream range
        output_image_view,
        output_image,
        coded_extent,
        Some(dpb_setup), // dpb_setup_picture for first frame
        &[],  // dpb_ref_pictures (no refs for IDR)
        &slice_offsets,
        Some(pic_order_cnt),
        Some(true),  // is_intra
        Some(true),  // is_reference
        Some(true),  // is_idr
    );

    match &result {
        Ok(()) => println!("  Decode command recorded successfully"),
        Err(e) => {
            eprintln!("  Decode command failed: {}", e);
            cleanup_all(
                &vulkan,
                session,
                session_params,
                output_image,
                output_image_view,
                output_memory,
                command_pool,
                fence,
            );
            std::process::exit(1);
        }
    }

    // Step 12: Submit and wait
    println!("\n--- Step 11: Submit and wait ---");
    unsafe {
        vulkan.device.reset_fences(&[fence]).expect("Failed to reset fence");
        vulkan.device.queue_submit(
            vulkan.device.get_device_queue(decode_qf, 0),
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        ).expect("Failed to submit");
        vulkan.device.wait_for_fences(&[fence], true, u64::MAX).expect("Failed to wait");
    }
    println!("  Decode complete\n");

    // Step 13: Readback and verify
    println!("--- Step 12: Readback and verify ---");
    let pixels = readback_decoded_image(
        &vulkan.instance,
        &vulkan.device,
        &vulkan.memory_properties,
        decode_qf,
        command_pool,
        fence,
        output_image,
        coded_extent.width,
        coded_extent.height,
    );

    match pixels {
        Ok(pixels) => {
            let y_plane_size = (parsed.coded_width * parsed.coded_height) as usize;
            let non_zero_y = pixels.y_plane[..y_plane_size]
                .iter()
                .filter(|&&v| v != 0)
                .count();
            let uv_size = (parsed.coded_width * parsed.coded_height / 4) as usize;
            let non_zero_u = pixels.u_plane[..uv_size]
                .iter()
                .filter(|&&v| v != 0)
                .count();
            let non_zero_v = pixels.v_plane[..uv_size]
                .iter()
                .filter(|&&v| v != 0)
                .count();

            println!("  Y plane: {} non-zero pixels out of {}", non_zero_y, y_plane_size);
            println!("  U plane: {} non-zero pixels out of {}", non_zero_u, uv_size);
            println!("  V plane: {} non-zero pixels out of {}", non_zero_v, uv_size);

            let all_zero = non_zero_y == 0 && non_zero_u == 0 && non_zero_v == 0;
            if all_zero {
                eprintln!("\n*** FAIL: Output is all zeros! Decoder did not produce valid output. ***");
                std::process::exit(1);
            } else {
                println!("\n*** SUCCESS: H265Decoder produced non-zero output! ***");
            }

            // Print some sample pixels
            println!("\n  Sample Y pixels (top-left 8x8):");
            for row in 0..8 {
                for col in 0..8 {
                    let idx = row as usize * parsed.coded_width as usize + col as usize;
                    print!("{:3} ", pixels.y_plane[idx]);
                }
                println!();
            }
        }
        Err(e) => {
            eprintln!("  Readback failed: {}", e);
        }
    }

    cleanup_all(
        &vulkan,
        session,
        session_params,
        output_image,
        output_image_view,
        output_memory,
        command_pool,
        fence,
    );
}

// ============================================================================
// Helper types and functions
// ============================================================================

struct ParsedInfo {
    vps: Option<vk_video_core::picture::H265Vps>,
    sps: Option<vk_video_core::picture::H265Sps>,
    pps: Option<vk_video_core::picture::H265Pps>,
    coded_width: u32,
    coded_height: u32,
    profile_idc: u32,
    max_dpb_slots: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
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
        .max(4);

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
        sps,
        pps,
        coded_width,
        coded_height,
        profile_idc,
        max_dpb_slots,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    }
}

fn find_first_idr(data: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut offset = 0;
    let mut au_data: Vec<u8> = Vec::new();
    let mut slice_offsets: Vec<u32> = Vec::new();
    let mut in_au = false;
    let mut is_idr_au = false;

    while offset < data.len() {
        let Some((start, code_len)) = find_next_start_code(data, offset) else {
            break;
        };
        let next_start = find_next_start_code(data, start + code_len);
        let end = next_start.map(|(s, _)| s).unwrap_or(data.len());
        let nal_data = &data[start + code_len..end];

        if let Some((_, nal_type, _, _)) = parse_h265_nal_header(nal_data) {
            let is_idr_slice = matches!(nal_type, 19 | 20);
            let is_slice = matches!(nal_type, 0..=31);
            let is_params = matches!(nal_type, 32..=34);
            let is_aud = nal_type == 38;

            if is_aud {
                if in_au && is_idr_au && !au_data.is_empty() {
                    return (au_data, slice_offsets);
                }
                if in_au {
                    au_data.clear();
                    slice_offsets.clear();
                    in_au = false;
                    is_idr_au = false;
                }
            } else if is_slice && !is_params {
                if is_idr_slice {
                    is_idr_au = true;
                    if !in_au {
                        au_data.clear();
                        slice_offsets.clear();
                        in_au = true;
                    }
                } else if in_au && is_idr_au {
                    return (au_data, slice_offsets);
                }
                if in_au && is_idr_au {
                    let au_start = au_data.len();
                    au_data.extend_from_slice(&data[start..end]);
                    slice_offsets.push(au_start as u32);
                }
            }
        }
        offset = end;
    }

    if in_au && is_idr_au && !au_data.is_empty() {
        return (au_data, slice_offsets);
    }

    // Fallback
    eprintln!("Warning: Could not find clean IDR frame, using first available slice");
    offset = 0;
    while offset < data.len() {
        let Some((start, code_len)) = find_next_start_code(data, offset) else {
            break;
        };
        let next_start = find_next_start_code(data, start + code_len);
        let end = next_start.map(|(s, _)| s).unwrap_or(data.len());
        let nal_data = &data[start + code_len..end];

        if let Some((_, nal_type, _, _)) = parse_h265_nal_header(nal_data) {
            if matches!(nal_type, 0..=31) && !matches!(nal_type, 32..=34) {
                return (data[start..end].to_vec(), vec![0]);
            }
        }
        offset = end;
    }

    panic!("No slice NAL found in bitstream");
}

/// Query video capabilities with decode-specific capabilities chained via pNext.
fn query_video_decode_capabilities(
    vulkan: &vk_video_vulkan::VulkanDevice,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> (vk::VideoCapabilitiesKHR<'_>, vk::VideoDecodeCapabilitiesKHR<'_>) {
    use ash::vk::Handle;

    let codec_op = vk::VideoCodecOperationFlagsKHR::DECODE_H265;

    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    let codec_profile_ptr = &h265_profile as *const _ as *const _;

    let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
    decode_caps.s_type = vk::StructureType::VIDEO_DECODE_CAPABILITIES_KHR;
    decode_caps.p_next = std::ptr::null_mut();

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: codec_profile_ptr,
        video_codec_operation: codec_op,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    let mut h265_decode_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
    h265_decode_caps.s_type = vk::StructureType::VIDEO_DECODE_H265_CAPABILITIES_KHR;
    h265_decode_caps.p_next = &mut decode_caps as *mut _ as *mut _;

    let get_caps_fn = unsafe {
        vulkan.entry.get_instance_proc_addr(
            vulkan.instance.handle(),
            b"vkGetPhysicalDeviceVideoCapabilitiesKHR\0".as_ptr().cast(),
        )
    }
    .expect("vkGetPhysicalDeviceVideoCapabilitiesKHR not found");

    let mut caps = vk::VideoCapabilitiesKHR::default();
    caps.p_next = &mut h265_decode_caps as *mut _ as *mut _;

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
        panic!("vkGetPhysicalDeviceVideoCapabilitiesKHR failed: {:?}", result);
    }

    (caps, decode_caps)
}

fn create_video_session(
    vulkan: &vk_video_vulkan::VulkanDevice,
    decode_qf: u32,
    profile_idc: u32,
    coded_extent: vk::Extent2D,
    max_dpb_slots: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    _vps: Option<&vk_video_core::picture::H265Vps>,
    _sps: Option<&vk_video_core::picture::H265Sps>,
    _pps: Option<&vk_video_core::picture::H265Pps>,
) -> (vk::VideoSessionKHR, vk::VideoSessionParametersKHR, Vec<vk::DeviceMemory>) {
    use ash::vk::Handle;

    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &h265_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::DECODE_H265,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    let mut std_header_version = vk::ExtensionProperties {
        extension_name: [0i8; 256],
        spec_version: (1u32 << 22) | (0u32 << 12) | 0u32, // VK_STD_VIDEO_SPEC_VERSION 1.0.0
    };
    let name_bytes = b"VK_STD_vulkan_video_codec_h265_decode\0";
    for (i, &b) in name_bytes.iter().enumerate() {
        std_header_version.extension_name[i] = b as i8;
    }

    let session_create_info = vk::VideoSessionCreateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
        p_next: std::ptr::null(),
        queue_family_index: decode_qf,
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
        vulkan.instance.get_device_proc_addr(
            vulkan.device.handle(),
            b"vkCreateVideoSessionKHR\0".as_ptr().cast(),
        )
    }
    .expect("vkCreateVideoSessionKHR not found");

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
            vulkan.device.handle(),
            &session_create_info,
            std::ptr::null(),
            &mut handle,
        );
        assert_eq!(result, vk::Result::SUCCESS, "vkCreateVideoSessionKHR failed: {:?}", result);
        handle
    };

    // Bind session memory
    let session_mems = bind_session_memory(&vulkan.instance, &vulkan.device, &vulkan.memory_properties, session);

    // Create session parameters WITHOUT VPS/SPS/PPS - decoder will update them
    let session_params = create_session_params_empty(
        &vulkan.instance,
        &vulkan.device,
        session,
    );

    (session, session_params, session_mems)
}

fn bind_session_memory(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    session: vk::VideoSessionKHR,
) -> Vec<vk::DeviceMemory> {
    use ash::vk::Handle;

    let get_req_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkGetVideoSessionMemoryRequirementsKHR\0".as_ptr().cast(),
        )
    }
    .expect("vkGetVideoSessionMemoryRequirementsKHR not found");

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
        assert_eq!(result, vk::Result::SUCCESS);
    }

    if req_count == 0 {
        return Vec::new();
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
        assert_eq!(result, vk::Result::SUCCESS);
    }

    let mut bind_infos = Vec::with_capacity(req_count as usize);
    let mut memories = Vec::with_capacity(req_count as usize);

    for (i, req) in requirements.iter().enumerate() {
        let mem_req = req.memory_requirements;

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
                .expect("Session memory allocation failed")
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
    .expect("vkBindVideoSessionMemoryKHR not found");

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
        assert_eq!(result, vk::Result::SUCCESS, "vkBindVideoSessionMemoryKHR failed: {:?}", result);
    }

    memories
}

fn create_session_params_empty(
    instance: &ash::Instance,
    device: &ash::Device,
    session: vk::VideoSessionKHR,
) -> vk::VideoSessionParametersKHR {
    use ash::vk::Handle;

    let h265_params = vk::VideoDecodeH265SessionParametersCreateInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
        p_next: std::ptr::null(),
        max_std_vps_count: 16,
        max_std_sps_count: 32,
        max_std_pps_count: 256,
        p_parameters_add_info: std::ptr::null(),
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

    let create_fn = unsafe {
        instance.get_device_proc_addr(
            device.handle(),
            b"vkCreateVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    }
    .expect("vkCreateVideoSessionParametersKHR not found");

    unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device,
            *const vk::VideoSessionParametersCreateInfoKHR,
            *const vk::AllocationCallbacks,
            *mut vk::VideoSessionParametersKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(create_fn);
        let mut params = vk::VideoSessionParametersKHR::null();
        let result = fn_ptr(
            device.handle(),
            &params_create_info,
            std::ptr::null(),
            &mut params,
        );
        assert_eq!(result, vk::Result::SUCCESS, "vkCreateVideoSessionParametersKHR failed: {:?}", result);
        params
    }
}

fn create_command_resources(
    device: &ash::Device,
    queue_family: u32,
) -> (vk::CommandPool, vk::CommandBuffer) {
    let pool = unsafe {
        device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .expect("Failed to create command pool")
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let buffers = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .expect("Failed to allocate command buffer")
    };

    (pool, buffers[0])
}

struct DecodedPixels {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..32 {
        if (type_bits & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & required_flags) == required_flags
        {
            return Some(i as u32);
        }
    }
    None
}

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
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    let uv_size = (uv_width * uv_height * 2) as usize;
    let total_size = (y_size + uv_size) as u64;

    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(total_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
        .map_err(|e| format!("Staging buffer creation failed: {:?}", e))?;

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        find_memory_type(
            memory_properties,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    })
    .ok_or("No suitable memory type for staging buffer")?;

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

        // Transition image planes
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

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| format!("End command buffer failed: {:?}", e))?;
    }

    // Submit and wait
    unsafe {
        device.reset_fences(&[fence]).map_err(|e| format!("Reset fence failed: {:?}", e))?;
        device.queue_submit(
            device.get_device_queue(queue_family, 0),
            &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
            fence,
        ).map_err(|e| format!("Submit failed: {:?}", e))?;
        device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("Wait failed: {:?}", e))?;
    }

    // Read data
    let y_plane = unsafe {
        let mut data = vec![0u8; y_size];
        std::ptr::copy_nonoverlapping(mapped_ptr, data.as_mut_ptr() as *mut _, y_size);
        data
    };

    let uv_data = unsafe {
        let mut data = vec![0u8; uv_size];
        std::ptr::copy_nonoverlapping(
            mapped_ptr.add(y_size),
            data.as_mut_ptr() as *mut _,
            uv_size,
        );
        data
    };

    unsafe { device.unmap_memory(memory) };

    // Deinterleave UV
    let mut u_plane = vec![0u8; uv_size / 2];
    let mut v_plane = vec![0u8; uv_size / 2];
    for i in 0..uv_size / 2 {
        u_plane[i] = uv_data[i * 2];
        v_plane[i] = uv_data[i * 2 + 1];
    }

    // Cleanup
    unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
    }

    Ok(DecodedPixels {
        y_plane,
        u_plane,
        v_plane,
    })
}

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

fn cleanup_all(
    vulkan: &vk_video_vulkan::VulkanDevice,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    output_image: vk::Image,
    output_image_view: vk::ImageView,
    output_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
) {
    unsafe {
        vulkan.device.destroy_fence(fence, None);
        vulkan.device.destroy_command_pool(command_pool, None);
        vulkan.device.destroy_image_view(output_image_view, None);
        vulkan.device.destroy_image(output_image, None);
        vulkan.device.free_memory(output_memory, None);
    }

    // Destroy session parameters
    if let Some(ptr) = unsafe {
        vulkan.instance.get_device_proc_addr(
            vulkan.device.handle(),
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
            f(vulkan.device.handle(), session_params, std::ptr::null());
        }
    }

    // Destroy session
    if let Some(ptr) = unsafe {
        vulkan.instance.get_device_proc_addr(
            vulkan.device.handle(),
            b"vkDestroyVideoSessionKHR\0".as_ptr().cast(),
        )
    } {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                *const vk::AllocationCallbacks,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(vulkan.device.handle(), session, std::ptr::null());
        }
    }
}

/// Create output image with video profile in pNext chain.
fn create_output_image_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), String> {
    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    let video_profile = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &h265_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::DECODE_H265,
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

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
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

    let mut image_create_info = image_create_info;
    image_create_info.p_next = &profile_list as *const _ as *const _;

    let image = unsafe {
        device
            .create_image(&image_create_info, None)
            .map_err(|e| format!("Image creation failed: {:?}", e))?
    };

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

/// Create bitstream buffer with video profile in pNext chain.
fn create_bitstream_buffer_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> Result<BitstreamBuffer, String> {
    let h265_profile = vk::VideoDecodeH265ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: profile_idc,
        _marker: Default::default(),
    };

    let video_profile = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &h265_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::DECODE_H265,
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

    BitstreamBuffer::create_with_pnext(
        device,
        memory_properties,
        size,
        1,
        256,
        &profile_list as *const _ as *const std::ffi::c_void,
    )
    .map_err(|e| e.to_string())
}
