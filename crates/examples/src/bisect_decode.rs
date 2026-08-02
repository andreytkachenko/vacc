//! Bisecting test to find root cause of all-black pixels.
//!
//! This test splits the decode pipeline into stages and verifies each stage:
//! 1. Bitstream parsing (SPS/PPS extraction)
//! 2. Access unit extraction with start codes
//! 3. Slice offset calculation (with start codes vs without)
//! 4. Decode command recording
//! 5. Output readback

use ash::vk::{self, Handle};
use vk_video_parser::{
    bitstream::BitstreamPacket,
    h264::H264Parser,
    DetectedVideoFormat, ParseResult, VideoParser,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.h264>", args[0]);
        std::process::exit(1);
    };

    println!("=== Bisecting Decode Test ===");
    println!("File: {}\n", bitstream_path);

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Stage 1: Parse SPS/PPS
    println!("--- Stage 1: Parse SPS/PPS ---");
    let (sps, pps, coded_width, coded_height) = parse_sps_pps(&data);
    println!("  SPS: profile={}, width={}, height={}", 
        sps.profile_idc, coded_width, coded_height);
    println!("  PPS: id={}", pps.pic_parameter_set_id);
    println!("  ✓ Stage 1 OK\n");

    // Stage 2: Extract first access unit with start codes
    println!("--- Stage 2: Extract first AU (with start codes) ---");
    let (au_data, slice_offsets_with_start_codes, slice_offsets_after_start_codes) = 
        extract_first_au_with_offsets(&data);
    println!("  AU size: {} bytes", au_data.len());
    println!("  Slice offsets (WITH start codes): {:?}", slice_offsets_with_start_codes);
    println!("  Slice offsets (AFTER start codes): {:?}", slice_offsets_after_start_codes);
    println!("  First 32 bytes of AU:");
    println!("    {:?}", &au_data[..32.min(au_data.len())]);
    println!("  ✓ Stage 2 OK\n");

    // Stage 3: Verify slice offset points to valid NAL header
    println!("--- Stage 3: Verify slice NAL headers ---");
    for (i, &offset_with) in slice_offsets_with_start_codes.iter().enumerate() {
        let offset_with = offset_with as usize;
        if offset_with + 5 <= au_data.len() {
            let nal_header = &au_data[offset_with + 3..offset_with + 5];
            println!("  Slice {} at offset {}: start_code=0x{:06x}, nal_header=0x{:02x}0x{:02x}",
                i, offset_with,
                u32::from_be_bytes([0, au_data[offset_with], au_data[offset_with+1], au_data[offset_with+2]]),
                nal_header[0], nal_header[1]);
        }
    }
    println!("  ✓ Stage 3 OK\n");

    // Stage 4: Initialize Vulkan and create session
    println!("--- Stage 4: Vulkan init ---");
    let vulkan = init_vulkan().expect("Failed to init Vulkan");
    let decode_qf = vulkan.queue_families.video_decode.expect("No decode queue");
    println!("  GPU: {}", get_gpu_name(&vulkan));
    println!("  ✓ Stage 4 OK\n");

    // Stage 5: Test decode with slice offsets INCLUDING start codes
    println!("--- Stage 5: Decode test with offsets INCLUDING start codes ---");
    let result_with = test_decode(
        &vulkan, decode_qf, &au_data, &slice_offsets_with_start_codes,
        &sps, &pps, coded_width, coded_height,
        "WITH_START_CODES",
    );
    println!("  Result: {}", result_with);

    // Stage 6: Test decode with slice offsets AFTER start codes
    println!("\n--- Stage 6: Decode test with offsets AFTER start codes ---");
    let result_after = test_decode(
        &vulkan, decode_qf, &au_data, &slice_offsets_after_start_codes,
        &sps, &pps, coded_width, coded_height,
        "AFTER_START_CODES",
    );
    println!("  Result: {}", result_after);

    // Summary
    println!("\n=== Summary ===");
    println!("WITH_START_CODES: {}", result_with);
    println!("AFTER_START_CODES: {}", result_after);
    
    if result_with.contains("valid") && !result_after.contains("valid") {
        println!("\nROOT CAUSE FOUND: Slice offsets must INCLUDE start codes!");
    } else if result_after.contains("valid") && !result_with.contains("valid") {
        println!("\nROOT CAUSE FOUND: Slice offsets must be AFTER start codes!");
    } else {
        println!("\nBoth tests gave same result - need to investigate further");
    }
}

fn parse_sps_pps(data: &[u8]) -> (vk_video_core::picture::H264Sps, vk_video_core::picture::H264Pps, u32, u32) {
    let mut parser = H264Parser::new();
    parser.init(&DetectedVideoFormat::new(vk_video_core::codec::VideoCodec::DecodeH264)).ok();

    let packet = BitstreamPacket::new(data.to_vec());
    if let Ok(ParseResult::ParameterSet { sps: s, pps: p, .. }) = parser.parse(&packet) {
        let sps = s.and_then(|s| s.downcast_ref::<vk_video_core::picture::H264Sps>().cloned())
            .expect("No H264 SPS found");
        let pps = p.and_then(|p| p.downcast_ref::<vk_video_core::picture::H264Pps>().cloned())
            .expect("No H264 PPS found");
        let coded_width = ((sps.pic_width_in_mbs_minus1 as u32 + 1) * 16);
        let coded_height = if sps.frame_mbs_only_flag {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
        } else {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
        };
        return (sps, pps, coded_width, coded_height);
    }
    panic!("Failed to parse SPS/PPS");
}

fn extract_first_au_with_offsets(data: &[u8]) -> (Vec<u8>, Vec<u32>, Vec<u32>) {
    use vk_video_parser::nal::find_next_start_code;

    let mut result = Vec::new();
    let mut offset = 0;
    let mut found_idr = false;
    let mut has_sps = false;
    let mut has_pps = false;
    let mut offsets_with_start_code: Vec<u32> = Vec::new();
    let mut offsets_after_start_code: Vec<u32> = Vec::new();

    while offset < data.len() {
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

        let nal_type = (nal_data[0] & 0x1F) as usize;
        let is_idr = nal_type == 5;
        let is_slice = matches!(nal_type, 1..=5);
        let is_sps = nal_type == 7;
        let is_pps = nal_type == 8;
        let is_params = nal_type == 7 || nal_type == 8;

        // Stop at next IDR after first frame
        if found_idr && is_idr {
            break;
        }
        // Stop at non-IDR slice after first IDR (next frame)
        if found_idr && is_slice && !is_idr {
            break;
        }

        if is_idr && !found_idr {
            found_idr = true;
        }

        // Include SPS/PPS with start codes
        if is_sps && !has_sps {
            result.extend_from_slice(&data[start..end]);
            has_sps = true;
        } else if is_pps && !has_pps {
            result.extend_from_slice(&data[start..end]);
            has_pps = true;
        } else if is_params {
            // Skip duplicate params
            offset = end;
            continue;
        }

        // Record slice offsets
        if is_slice {
            let offset_in_result = result.len() as u32;
            let offset_with_sc = offset_in_result; // Points to start code
            let offset_after_sc = offset_in_result + code_len as u32; // Points after start code
            
            result.extend_from_slice(&data[start..end]);
            offsets_with_start_code.push(offset_with_sc);
            offsets_after_start_code.push(offset_after_sc);
        } else if found_idr && !is_params {
            result.extend_from_slice(&data[start..end]);
        }

        offset = end;
    }

    if result.is_empty() {
        return (data.to_vec(), vec![0], vec![0]);
    }

    (result, offsets_with_start_code, offsets_after_start_code)
}

fn init_vulkan() -> Result<VulkanDevice, String> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("Entry load failed: {}", e))?;

        let app_name = std::ffi::CString::new("bisect-decode").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_2);

        let instance_exts = vec![
            std::ffi::CString::new("VK_KHR_surface").unwrap(),
            std::ffi::CString::new("VK_KHR_get_physical_device_properties2").unwrap(),
        ];
        let ext_ptrs: Vec<_> = instance_exts.iter().map(|c| c.as_ptr()).collect();

        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&ext_ptrs),
            None,
        ).map_err(|e| format!("Instance creation failed: {}", e))?;

        let physical_devices = instance.enumerate_physical_devices()
            .map_err(|e| format!("Enumerate devices failed: {}", e))?;
        if physical_devices.is_empty() {
            return Err("No physical devices".to_string());
        }
        let pd = physical_devices[0];

        let queue_families = instance.get_physical_device_queue_family_properties(pd);
        let decode_qf = queue_families.iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
            .ok_or("No decode queue family")? as u32;

        let memory_properties = instance.get_physical_device_memory_properties(pd);

        let device_exts = vec![
            std::ffi::CString::new("VK_KHR_video_decode_queue").unwrap(),
            std::ffi::CString::new("VK_KHR_video_decode_h264").unwrap(),
            std::ffi::CString::new("VK_KHR_sampler_ycbcr_conversion").unwrap(),
            std::ffi::CString::new("VK_KHR_synchronization2").unwrap(),
        ];
        let ext_ptrs: Vec<_> = device_exts.iter().map(|c| c.as_ptr()).collect();

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(decode_qf)
            .queue_priorities(&[1.0]);

        let device = instance.create_device(
            pd,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[queue_create_info])
                .enabled_extension_names(&ext_ptrs),
            None,
        ).map_err(|e| format!("Device creation failed: {}", e))?;

        Ok(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            memory_properties,
            queue_families: vk_video_vulkan::QueueFamilies {
                video_decode: Some(decode_qf),
                ..Default::default()
            },
            enabled_extensions: device_exts.into_iter()
                .map(|c| c.to_string_lossy().into_owned())
                .collect(),
        })
    }
}

fn get_gpu_name(vulkan: &VulkanDevice) -> String {
    unsafe {
        let props = vulkan.instance.get_physical_device_properties(vulkan.physical_device);
        std::ffi::CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy().to_string()
    }
}

struct VulkanDevice {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_families: vk_video_vulkan::QueueFamilies,
    enabled_extensions: Vec<String>,
}

fn test_decode(
    vulkan: &VulkanDevice,
    decode_qf: u32,
    au_data: &[u8],
    slice_offsets: &[u32],
    sps: &vk_video_core::picture::H264Sps,
    pps: &vk_video_core::picture::H264Pps,
    width: u32,
    height: u32,
    label: &str,
) -> String {
    let coded_extent = vk::Extent2D {
        width: ((width + 15) / 16 * 16),
        height: ((height + 15) / 16 * 16),
    };

    // Create session
    let h264_profile = vk::VideoDecodeH264ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR,
        p_next: std::ptr::null(),
        std_profile_idc: sps.profile_idc as u32,
        picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE,
        _marker: Default::default(),
    };

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &h264_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::DECODE_H264,
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        _marker: Default::default(),
    };

    let session_create_info = vk::VideoSessionCreateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
        p_next: std::ptr::null(),
        queue_family_index: decode_qf,
        flags: vk::VideoSessionCreateFlagsKHR::empty(),
        p_video_profile: &profile_info as *const _,
        picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
        max_coded_extent: coded_extent,
        reference_picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
        max_dpb_slots: 4,
        max_active_reference_pictures: 4,
        p_std_header_version: std::ptr::null(),
        _marker: Default::default(),
    };

    let create_fn = unsafe {
        vulkan.instance.get_device_proc_addr(
            vulkan.device.handle(),
            b"vkCreateVideoSessionKHR\0".as_ptr().cast(),
        )
    };

    let session = unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device, *const vk::VideoSessionCreateInfoKHR,
            *const vk::AllocationCallbacks, *mut vk::VideoSessionKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(create_fn.unwrap());
        let mut handle = vk::VideoSessionKHR::null();
        fn_ptr(vulkan.device.handle(), &session_create_info, std::ptr::null(), &mut handle);
        handle
    };

    if session.is_null() {
        return "Session creation failed".to_string();
    }

    // Create session parameters with SPS/PPS
    let std_sps = convert_h264_sps(sps);
    let std_pps = convert_h264_pps(pps);

    let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        p_next: std::ptr::null(),
        std_sps_count: 1,
        p_std_sp_ss: &std_sps as *const _,
        std_pps_count: 1,
        p_std_pp_ss: &std_pps as *const _,
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

    let create_params_fn = unsafe {
        vulkan.instance.get_device_proc_addr(
            vulkan.device.handle(),
            b"vkCreateVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    };

    let session_params = unsafe {
        type FnType = unsafe extern "system" fn(
            vk::Device, *const vk::VideoSessionParametersCreateInfoKHR<'_>,
            *const vk::AllocationCallbacks, *mut vk::VideoSessionParametersKHR,
        ) -> vk::Result;
        let fn_ptr: FnType = std::mem::transmute(create_params_fn.unwrap());
        let mut params = vk::VideoSessionParametersKHR::null();
        fn_ptr(vulkan.device.handle(), &params_create_info, std::ptr::null(), &mut params);
        params
    };

    if session_params.is_null() {
        return "Session params creation failed".to_string();
    }

    // Create output image
    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
        .extent(vk::Extent3D { width: coded_extent.width, height: coded_extent.height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
            | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
            | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let output_image = unsafe {
        vulkan.device.create_image(&image_create_info, None).unwrap()
    };

    let mem_reqs = unsafe { vulkan.device.get_image_memory_requirements(output_image) };
    let mem_type_idx = find_memory_type(&vulkan.memory_properties, mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();

    let output_memory = unsafe {
        vulkan.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_idx),
            None,
        ).unwrap()
    };

    unsafe {
        vulkan.device.bind_image_memory(output_image, output_memory, 0).unwrap();
    }

    let output_view = unsafe {
        vulkan.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(output_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
                .subresource_range(vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)),
            None,
        ).unwrap()
    };

    // Create bitstream buffer
    let bs_size = ((au_data.len() as u64 + 255) / 256 * 256).max(256);
    let bs_buffer = unsafe {
        vulkan.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(bs_size)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR),
            None,
        ).unwrap()
    };

    let bs_mem_reqs = unsafe { vulkan.device.get_buffer_memory_requirements(bs_buffer) };
    let bs_mem_type_idx = find_memory_type(&vulkan.memory_properties, bs_mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();

    let bs_memory = unsafe {
        vulkan.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(bs_mem_reqs.size)
                .memory_type_index(bs_mem_type_idx),
            None,
        ).unwrap()
    };

    unsafe {
        vulkan.device.bind_buffer_memory(bs_buffer, bs_memory, 0).unwrap();
        let ptr = vulkan.device.map_memory(bs_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
        std::ptr::copy_nonoverlapping(au_data.as_ptr(), ptr as *mut u8, au_data.len());
        vulkan.device.unmap_memory(bs_memory);
    }

    // Create command resources
    let cmd_pool = unsafe {
        vulkan.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(decode_qf)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ).unwrap()
    };

    let cmd_buffer = unsafe {
        vulkan.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        ).unwrap()[0]
    };

    let fence = unsafe {
        vulkan.device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        ).unwrap()
    };

    // Record decode command
    let pic_info = build_h264_picture_info(sps, pps);
    let h264_decode_info = vk::VideoDecodeH264PictureInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_PICTURE_INFO_KHR,
        p_next: std::ptr::null(),
        p_std_picture_info: &pic_info as *const _,
        slice_count: slice_offsets.len() as u32,
        p_slice_offsets: slice_offsets.as_ptr(),
        _marker: Default::default(),
    };

    let dpb_setup_resource = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: output_view,
        _marker: Default::default(),
    };

    let mut std_ref_info = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264ReferenceInfo>() };
    std_ref_info.FrameNum = 0;
    std_ref_info.PicOrderCnt = [0, 0];

    let dpb_slot_info = vk::VideoDecodeH264DpbSlotInfoKHR {
        s_type: vk::StructureType::VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
        p_next: std::ptr::null(),
        p_std_reference_info: &std_ref_info as *const _,
        _marker: Default::default(),
    };

    let begin_ref_slot = vk::VideoReferenceSlotInfoKHR {
        s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
        p_next: &dpb_slot_info as *const _ as *const _,
        slot_index: -1,
        p_picture_resource: &dpb_setup_resource,
        _marker: Default::default(),
    };

    let decode_ref_slot = vk::VideoReferenceSlotInfoKHR {
        s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
        p_next: &dpb_slot_info as *const _ as *const _,
        slot_index: 0,
        p_picture_resource: &dpb_setup_resource,
        _marker: Default::default(),
    };

    unsafe {
        vulkan.device.begin_command_buffer(
            cmd_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        ).unwrap();

        // Begin coding
        let begin_coding_info = vk::VideoBeginCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoBeginCodingFlagsKHR::empty(),
            video_session: session,
            video_session_parameters: session_params,
            reference_slot_count: 1,
            p_reference_slots: &begin_ref_slot,
            _marker: Default::default(),
        };
        cmd_begin_video_coding(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &begin_coding_info);

        // Reset decoder
        let control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };
        cmd_control_video_coding(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &control_info);

        // Barriers
        let buffer_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::HOST,
            src_access_mask: vk::AccessFlags2::HOST_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: bs_buffer,
            offset: 0,
            size: bs_size,
            _marker: Default::default(),
        };

        let image_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::NONE,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: output_image,
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

        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buffer_barrier,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &image_barrier,
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &dep_info);

        // Decode
        let decode_info = vk::VideoDecodeInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
            p_next: &h264_decode_info as *const _ as *const _,
            flags: vk::VideoDecodeFlagsKHR::empty(),
            src_buffer: bs_buffer,
            src_buffer_offset: 0,
            src_buffer_range: bs_size,
            dst_picture_resource: dpb_setup_resource,
            p_setup_reference_slot: &decode_ref_slot,
            reference_slot_count: 0,
            p_reference_slots: std::ptr::null(),
            _marker: Default::default(),
        };
        cmd_decode_video(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &decode_info);

        // End coding
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };
        cmd_end_video_coding(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &end_coding_info);

        vulkan.device.end_command_buffer(cmd_buffer).unwrap();
    }

    // Submit and wait
    unsafe {
        vulkan.device.reset_fences(&[fence]).unwrap();
        vulkan.device.queue_submit(
            vulkan.device.get_device_queue(decode_qf, 0),
            &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
            fence,
        ).unwrap();

        let result = vulkan.device.wait_for_fences(&[fence], true, 10_000_000_000);
        if let Err(e) = result {
            return format!("Decode failed: {:?}", e);
        }
    }

    // Readback
    let y_size = (coded_extent.width * coded_extent.height) as usize;
    let uv_width = (coded_extent.width + 1) / 2;
    let uv_height = (coded_extent.height + 1) / 2;
    let uv_size = (uv_width * uv_height * 2) as usize;
    let total_size = (y_size + uv_size) as u64;

    let staging_buffer = unsafe {
        vulkan.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(total_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        ).unwrap()
    };

    let staging_mem_reqs = unsafe { vulkan.device.get_buffer_memory_requirements(staging_buffer) };
    let staging_mem_type_idx = find_memory_type(&vulkan.memory_properties, staging_mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();

    let staging_memory = unsafe {
        vulkan.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(staging_mem_reqs.size)
                .memory_type_index(staging_mem_type_idx),
            None,
        ).unwrap()
    };

    unsafe {
        vulkan.device.bind_buffer_memory(staging_buffer, staging_memory, 0).unwrap();
    }

    let mapped_ptr = unsafe {
        vulkan.device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap()
    };

    unsafe {
        vulkan.device.begin_command_buffer(
            cmd_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        ).unwrap();

        let plane0_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: output_image,
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
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
            image: output_image,
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
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
        cmd_pipeline_barrier_2(&vulkan.instance, vulkan.device.handle(), cmd_buffer, &dep_info);

        vulkan.device.cmd_copy_image_to_buffer(
            cmd_buffer, output_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1))
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D { width: coded_extent.width, height: coded_extent.height, depth: 1 })],
        );

        vulkan.device.cmd_copy_image_to_buffer(
            cmd_buffer, output_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1))
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D { width: uv_width, height: uv_height, depth: 1 })],
        );

        vulkan.device.end_command_buffer(cmd_buffer).unwrap();

        vulkan.device.reset_fences(&[fence]).unwrap();
        vulkan.device.queue_submit(
            vulkan.device.get_device_queue(decode_qf, 0),
            &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
            fence,
        ).unwrap();
        vulkan.device.wait_for_fences(&[fence], true, 10_000_000_000).unwrap();

        let mut y_plane = vec![0u8; y_size];
        std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, y_plane.as_mut_ptr(), y_size);

        let y_min = y_plane.iter().min().copied().unwrap_or(0) as i32;
        let y_max = y_plane.iter().max().copied().unwrap_or(255) as i32;
        let y_avg: f64 = y_plane.iter().map(|&b| b as f64).sum::<f64>() / y_plane.len() as f64;
        let y_zero_pct = y_plane.iter().filter(|&&b| b == 0).count() as f64 / y_plane.len() as f64 * 100.0;

        vulkan.device.unmap_memory(staging_memory);
        vulkan.device.free_memory(staging_memory, None);
        vulkan.device.destroy_buffer(staging_buffer, None);

        if y_max == 0 && y_avg < 1.0 {
            format!("ALL BLACK (Y: min={} max={} avg={:.1} zero={:.1}%)", y_min, y_max, y_avg, y_zero_pct)
        } else if y_zero_pct > 95.0 {
            format!("MOSTLY BLACK (Y: min={} max={} avg={:.1} zero={:.1}%)", y_min, y_max, y_avg, y_zero_pct)
        } else {
            format!("VALID (Y: min={} max={} avg={:.1} zero={:.1}%)", y_min, y_max, y_avg, y_zero_pct)
        }
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(required_flags)
        {
            return Some(i);
        }
    }
    None
}

fn build_h264_picture_info(
    sps: &vk_video_core::picture::H264Sps,
    pps: &vk_video_core::picture::H264Pps,
) -> ash::vk::native::StdVideoDecodeH264PictureInfo {
    let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
    let frame_num = 0u32 % max_frame_num;
    let log2_max_poc_lsb = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let max_poc_lsb = 1u32 << log2_max_poc_lsb;
    let poc_lsb = 0u32 % max_poc_lsb;

    let mut pic_info = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeH264PictureInfo>() };
    pic_info.flags.set_field_pic_flag(0);
    pic_info.flags.set_is_intra(1);
    pic_info.flags.set_IdrPicFlag(1);
    pic_info.flags.set_bottom_field_flag(0);
    pic_info.flags.set_is_reference(1);
    pic_info.flags.set_complementary_field_pair(0);
    pic_info.seq_parameter_set_id = sps.seq_parameter_set_id as u8;
    pic_info.pic_parameter_set_id = pps.pic_parameter_set_id as u8;
    pic_info.frame_num = frame_num as u16;
    pic_info.idr_pic_id = 0;
    pic_info.PicOrderCnt = [poc_lsb as i32, poc_lsb as i32];
    pic_info
}

fn convert_h264_sps(sps: &vk_video_core::picture::H264Sps) -> ash::vk::native::StdVideoH264SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoH264SpsFlags>() };
    flags.set_frame_mbs_only_flag(if sps.frame_mbs_only_flag { 1 } else { 0 });
    flags.set_direct_8x8_inference_flag(if sps.direct_8x8_inference_flag { 1 } else { 0 });

    ash::vk::native::StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: sps.profile_idc as u32,
        level_idc: h264_level_idc_to_vulkan(sps.level_idc, sps.constraint_set3_flag),
        chroma_format_idc: sps.chroma_format_idc as u32,
        seq_parameter_set_id: sps.seq_parameter_set_id as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type: sps.pic_order_cnt_type as u32,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        max_num_ref_frames: sps.max_num_ref_frames as u8,
        pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1 as u32,
        pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1 as u32,
        ..unsafe { std::mem::zeroed() }
    }
}

fn convert_h264_pps(pps: &vk_video_core::picture::H264Pps) -> ash::vk::native::StdVideoH264PictureParameterSet {
    ash::vk::native::StdVideoH264PictureParameterSet {
        flags: unsafe { std::mem::zeroed() },
        seq_parameter_set_id: pps.seq_parameter_set_id as u8,
        pic_parameter_set_id: pps.pic_parameter_set_id as u8,
        ..unsafe { std::mem::zeroed() }
    }
}

fn h264_level_idc_to_vulkan(raw_level_idc: u8, _constraint_set3_flag: bool) -> u32 {
    match raw_level_idc {
        10 => 0, 11 => 1, 12 => 2, 13 => 3, 20 => 4, 21 => 5, 22 => 6,
        30 => 7, 31 => 8, 32 => 9, 40 => 10, 41 => 11, 42 => 12,
        50 => 13, 51 => 14, 52 => 15, 60 => 16, 61 => 17, 62 => 18,
        _ => 18,
    }
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
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
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
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoCodingControlInfoKHR<'_>);
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
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoBeginCodingInfoKHR<'_>);
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
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, b"vkCmdDecodeVideoKHR\0".as_ptr().cast())
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoDecodeInfoKHR<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
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
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoEndCodingInfoKHR<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, info);
        }
    }
}
