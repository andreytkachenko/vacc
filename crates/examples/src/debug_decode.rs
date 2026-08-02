//! Debug decode example - dumps detailed info about the decode setup.
//!
//! This example helps diagnose Vulkan Video decode issues by:
//! 1. Printing SPS/PPS fields parsed from the bitstream
//! 2. Printing converted Vulkan native SPS/PPS fields
//! 3. Comparing with ffprobe output
//! 4. Trying different decode configurations

use std::ffi::CString;
use ash::vk;
use vk_video_parser::{
    bitstream::BitstreamPacket,
    h264::H264Parser,
    h265::H265Parser,
    DetectedVideoFormat, ParseResult, VideoParser,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 { &args[1] } else { "born_trailer.h264" };

    println!("=== Debug Decode Example ===");
    println!("File: {}", bitstream_path);

    let data = std::fs::read(bitstream_path).expect("Failed to read file");
    let is_h265 = bitstream_path.ends_with(".h265") || bitstream_path.ends_with(".265");

    // Step 1: Parse bitstream
    println!("\n--- Step 1: Parse SPS/PPS ---");
    if is_h265 {
        parse_h265_debug(&data);
    } else {
        parse_h264_debug(&data);
    }

    // Step 2: Compare with ffprobe
    println!("\n--- Step 2: ffprobe comparison ---");
    compare_with_ffprobe(bitstream_path, is_h265);

    // Step 3: Try Vulkan decode with different configurations
    println!("\n--- Step 3: Vulkan decode test ---");
    test_vulkan_decode(bitstream_path, is_h265);
}

fn parse_h264_debug(data: &[u8]) {
    let mut parser = H264Parser::new();
    parser.init(&DetectedVideoFormat::new(
        vk_video_core::codec::VideoCodec::DecodeH264,
    )).ok();

    let packet = BitstreamPacket::new(data.to_vec());
    if let Ok(ParseResult::ParameterSet { sps, pps, .. }) = parser.parse(&packet) {
        if let Some(sps_boxed) = sps {
            if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                println!("H.264 SPS:");
                println!("  profile_idc: {} ({})", sps.profile_idc, profile_name_h264(sps.profile_idc));
                println!("  level_idc: {} ({})", sps.level_idc, level_name_h264(sps.level_idc));
                println!("  seq_parameter_set_id: {}", sps.seq_parameter_set_id);
                println!("  chroma_format_idc: {}", sps.chroma_format_idc);
                println!("  bit_depth_luma_minus8: {}", sps.bit_depth_luma_minus8);
                println!("  bit_depth_chroma_minus8: {}", sps.bit_depth_chroma_minus8);
                println!("  log2_max_frame_num_minus4: {}", sps.log2_max_frame_num_minus4);
                println!("  pic_order_cnt_type: {}", sps.pic_order_cnt_type);
                println!("  log2_max_pic_order_cnt_lsb_minus4: {}", sps.log2_max_pic_order_cnt_lsb_minus4);
                println!("  max_num_ref_frames: {}", sps.max_num_ref_frames);
                println!("  pic_width_in_mbs_minus1: {} -> {} pixels", 
                    sps.pic_width_in_mbs_minus1, (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16);
                println!("  pic_height_in_map_units_minus1: {} -> {} pixels",
                    sps.pic_height_in_map_units_minus1, 
                    if sps.frame_mbs_only_flag {
                        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                    } else {
                        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                    });
                println!("  frame_mbs_only_flag: {}", sps.frame_mbs_only_flag);
                println!("  frame_cropping_flag: {}", sps.frame_cropping_flag);
            }
        }
        if let Some(pps_boxed) = pps {
            if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                println!("H.264 PPS:");
                println!("  pic_parameter_set_id: {}", pps.pic_parameter_set_id);
                println!("  seq_parameter_set_id: {}", pps.seq_parameter_set_id);
            }
        }
    }
}

fn parse_h265_debug(data: &[u8]) {
    let mut parser = H265Parser::new();
    parser.init(&DetectedVideoFormat::new(
        vk_video_core::codec::VideoCodec::DecodeH265,
    )).ok();

    let packet = BitstreamPacket::new(data.to_vec());
    if let Ok(ParseResult::ParameterSet { sps, pps, vps }) = parser.parse(&packet) {
        if let Some(vps_boxed) = vps {
            if let Some(vps) = vps_boxed.downcast_ref::<vk_video_core::picture::H265Vps>() {
                println!("H.265 VPS:");
                println!("  vps_video_parameter_set_id: {}", vps.vps_video_parameter_set_id);
            }
        }
        if let Some(sps_boxed) = sps {
            if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H265Sps>() {
                println!("H.265 SPS:");
                println!("  sps_seq_parameter_set_id: {}", sps.sps_seq_parameter_set_id);
                println!("  chroma_format_idc: {}", sps.chroma_format_idc);
                println!("  pic_width_in_luma_samples: {}", sps.pic_width_in_luma_samples);
                println!("  pic_height_in_luma_samples: {}", sps.pic_height_in_luma_samples);
                println!("  bit_depth_luma_minus8: {}", sps.bit_depth_luma_minus8);
                println!("  log2_max_pic_order_cnt_lsb_minus4: {}", sps.log2_max_pic_order_cnt_lsb_minus4);
            }
        }
        if let Some(pps_boxed) = pps {
            if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H265Pps>() {
                println!("H.265 PPS:");
                println!("  pps_pic_parameter_set_id: {}", pps.pps_pic_parameter_set_id);
                println!("  pps_seq_parameter_set_id: {}", pps.pps_seq_parameter_set_id);
            }
        }
    }
}

fn compare_with_ffprobe(path: &str, _is_h265: bool) {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path])
        .output();
    
    match output {
        Ok(o) if o.status.success() => {
            let json = String::from_utf8_lossy(&o.stdout);
            // Extract key fields
            if let Some(width_start) = json.find("width") {
                let width_end = json[width_start..].find('\"').unwrap_or(10) + width_start;
                let width = &json[width_start + 6..width_end];
                println!("ffprobe width: {}", width.trim());
            }
            if let Some(height_start) = json.find("height") {
                let height_end = json[height_start..].find('\"').unwrap_or(10) + height_start;
                let height = &json[height_start + 7..height_end];
                println!("ffprobe height: {}", height.trim());
            }
            if let Some(profile_start) = json.find("profile") {
                let profile_end = json[profile_start..].find('\"').unwrap_or(20) + profile_start;
                let profile = &json[profile_start + 8..profile_end];
                println!("ffprobe profile: {}", profile.trim());
            }
        }
        _ => println!("ffprobe not available or failed"),
    }
}

fn test_vulkan_decode(path: &str, is_h265: bool) {
    // Initialize Vulkan
    unsafe {
        let entry = match ash::Entry::load() {
            Ok(e) => e,
            Err(e) => {
                println!("Failed to load Vulkan entry: {}", e);
                return;
            }
        };

        let app_name = CString::new("debug-decode").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_2);

        // Enable validation layer
        let layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layers = vec![layer_name.as_ptr()];
        
        let instance_ext = CString::new("VK_KHR_surface").unwrap();
        let instance_ext2 = CString::new("VK_KHR_get_physical_device_properties2").unwrap();
        let exts = vec![instance_ext.as_ptr(), instance_ext2.as_ptr()];

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&exts)
            .enabled_layer_names(&layers);

        let instance = match entry.create_instance(&instance_create_info, None) {
            Ok(i) => i,
            Err(e) => {
                println!("Failed to create instance: {:?}", e);
                return;
            }
        };

        let physical_devices = match instance.enumerate_physical_devices() {
            Ok(d) => d,
            Err(e) => {
                println!("Failed to enumerate devices: {}", e);
                return;
            }
        };

        if physical_devices.is_empty() {
            println!("No physical devices found");
            return;
        }

        let pd = physical_devices[0];
        let props = instance.get_physical_device_properties(pd);
        let gpu_name = std::ffi::CStr::from_ptr(props.device_name.as_ptr());
        println!("GPU: {}", gpu_name.to_string_lossy());

        // Find video decode queue
        let qfs = instance.get_physical_device_queue_family_properties(pd);
        let mut decode_qf: Option<u32> = None;
        for (i, qf) in qfs.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) {
                decode_qf = Some(i as u32);
                break;
            }
        }

        match decode_qf {
            Some(qf) => println!("Video decode queue family: {}", qf),
            None => {
                println!("No video decode queue found");
                return;
            }
        }

        // Check available extensions
        let extensions = instance.enumerate_device_extension_properties(pd).unwrap();
        println!("\nAvailable video extensions:");
        for ext in &extensions {
            let name_bytes: Vec<u8> = ext.extension_name.iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as u8)
                .collect();
            let name = String::from_utf8_lossy(&name_bytes);
            if name.contains("video_decode") {
                println!("  {}", name);
            }
        }

        // Query video capabilities for H.264
        println!("\nQuerying H.264 decode capabilities...");
        query_h264_capabilities(&entry, &instance, pd, decode_qf.unwrap());

        // Cleanup
        instance.destroy_instance(None);
    }
}

fn query_h264_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    _decode_qf: u32,
) {
    unsafe {
        // Build profile info for H.264 Baseline
        let h264_profile = vk::VideoDecodeH264ProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR,
            p_next: std::ptr::null(),
            std_profile_idc: 66, // Baseline
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

        // Get function pointer
        let get_caps_fn = entry.get_instance_proc_addr(
            instance.handle(),
            b"vkGetPhysicalDeviceVideoCapabilitiesKHR\0".as_ptr().cast(),
        );

        if let Some(ptr) = get_caps_fn {
            type FnType = unsafe extern "system" fn(
                vk::PhysicalDevice,
                *const vk::VideoProfileInfoKHR<'_>,
                *mut vk::VideoCapabilitiesKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(ptr);
            
            let mut caps = vk::VideoCapabilitiesKHR::default();
            let result = fn_ptr(pd, &profile_info, &mut caps);
            
            if result == vk::Result::SUCCESS {
                println!("  H.264 Baseline decode: SUPPORTED");
                println!("  minBitstreamBufferSizeAlignment: {}", caps.min_bitstream_buffer_size_alignment);
                println!("  maxDPBSlots: {}", caps.max_dpb_slots);
                println!("  pictureAccessGranularity: {}x{}", 
                    caps.picture_access_granularity.width,
                    caps.picture_access_granularity.height);
                println!("  minCodedExtent: {}x{}",
                    caps.min_coded_extent.width,
                    caps.min_coded_extent.height);
                println!("  maxCodedExtent: {}x{}",
                    caps.max_coded_extent.width,
                    caps.max_coded_extent.height);
            } else {
                println!("  H.264 Baseline decode: NOT SUPPORTED ({:?})", result);
            }
        } else {
            println!("  vkGetPhysicalDeviceVideoCapabilitiesKHR not found");
        }
    }
}

fn profile_name_h264(idc: u8) -> &'static str {
    match idc {
        66 => "Baseline",
        77 => "Main",
        88 => "Extended",
        100 => "High",
        110 => "High10",
        122 => "High422",
        _ => "Unknown",
    }
}

fn level_name_h264(idc: u8) -> &'static str {
    match idc {
        30 => "3.0",
        31 => "3.1",
        32 => "3.2",
        40 => "4.0",
        41 => "4.1",
        42 => "4.2",
        50 => "5.0",
        51 => "5.1",
        52 => "5.2",
        _ => "&unknown",
    }
}
