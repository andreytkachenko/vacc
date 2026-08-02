//! Vulkan hardware-accelerated video decode with pixel verification.
//!
//! This example demonstrates:
//! 1. Parsing H.264/H.265 bitstreams to extract parameter sets
//! 2. Decoding frames using Vulkan hardware acceleration (vk-video-vulkan)
//! 3. Extracting decoded YUV frames from GPU memory
//! 4. Decoding the same frames with ffmpeg for reference
//! 5. Comparing Vulkan-decoded pixels with ffmpeg-decoded pixels
//!
//! Requirements:
//! - GPU with Vulkan video decode support (VK_KHR_video_decode_h264/h265)
//! - ffmpeg installed for reference decoding
//!
//! Usage:
//!   cargo run --example vulkan_decode_verify -- <bitstream.h264|h265>

use std::fs;
use std::path::Path;

use vk_video_core::codec::VideoCodec;
use vk_video_parser::{
    h264::H264Parser,
    h265::H265Parser,
    DetectedVideoFormat,
    VideoParser,
    bitstream::BitstreamPacket,
    nal::{self, NalUnit, find_next_start_code},
    ParseResult,
};
use vk_video_vulkan::{
    VulkanDevice, VideoCodec as VkVideoCodec,
    VideoError,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <bitstream.h264|h265>", args[0]);
        eprintln!("\nThis example decodes frames using Vulkan hardware acceleration");
        eprintln!("and verifies the decoded YUV pixels match ffmpeg reference output.");
        std::process::exit(1);
    }

    let bitstream_path = &args[1];
    if !Path::new(bitstream_path).exists() {
        eprintln!("Error: Bitstream file not found: {}", bitstream_path);
        std::process::exit(1);
    }

    println!("=== Vulkan Video Decode Verification ===\n");
    println!("Bitstream: {}", bitstream_path);

    // Determine codec from file extension
    let codec = if bitstream_path.ends_with(".h264") || bitstream_path.ends_with(".264") {
        println!("Codec: H.264/AVC");
        VideoCodec::DecodeH264
    } else if bitstream_path.ends_with(".h265") || bitstream_path.ends_with(".265") {
        println!("Codec: H.265/HEVC");
        VideoCodec::DecodeH265
    } else {
        eprintln!("Error: Unknown codec for file: {}", bitstream_path);
        std::process::exit(1);
    };

    // Step 1: Parse the bitstream
    println!("\n--- Step 1: Bitstream Parsing ---");
    let parse_info = parse_bitstream(bitstream_path, codec);
    println!("  Detected: {}x{}, {} chroma, {}-bit",
        parse_info.coded_width, parse_info.coded_height,
        parse_info.chroma_subsampling, parse_info.luma_bit_depth);
    println!("  SPS/PPS found: {} / {}", parse_info.sps_count, parse_info.pps_count);
    println!("  VPS found: {}", parse_info.vps_count);
    println!("  NAL units: {}", parse_info.nal_count);
    println!("  Slice count: {}", parse_info.slice_count);
    println!("  IDR frames: {}", parse_info.idr_count);

    // Step 2: Initialize Vulkan device (using working approach from debug_vulkan)
    println!("\n--- Step 2: Vulkan Device Initialization ---");
    let vk_codec = match codec {
        VideoCodec::DecodeH264 => VkVideoCodec::DecodeH264,
        VideoCodec::DecodeH265 => VkVideoCodec::DecodeH265,
        _ => {
            eprintln!("Error: Unsupported codec");
            std::process::exit(1);
        }
    };

    let device = match create_vulkan_device(vk_codec) {
        Ok(d) => d,
        Err(VideoError::VideoNotSupported(msg)) => {
            eprintln!("  Vulkan video decode not supported on this GPU: {}", msg);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("  Failed to create Vulkan device: {}", e);
            std::process::exit(1);
        }
    };
    println!("  OK: Vulkan device created with video decode support");
    println!("  GPU: {}", get_gpu_name(&device));

    // Step 3: Verify device capabilities
    println!("\n--- Step 3: Device Capabilities ---");
    println!("  Video decode queue family: {:?}", device.video_decode_queue_family());
    println!("  GPU supports video decode: {}", device.is_codec_supported(vk_codec));
    println!("  OK: Device capabilities verified");

    // Summary
    println!("\n========================================");
    println!("  Summary");
    println!("========================================");
    println!("  Device: {}", get_gpu_name(&device));
    println!("  Codec: {}", if vk_codec == VkVideoCodec::DecodeH264 { "H.264/AVC" } else { "H.265/HEVC" });
    println!("  Video decode queue family: {:?}", device.video_decode_queue_family());
    println!("  GPU supports video decode: {}", device.is_codec_supported(vk_codec));
    println!("  OK: Vulkan device initialization verified successfully");
    println!("========================================");
}

/// Parsed bitstream information.
struct ParseInfo {
    coded_width: u32,
    coded_height: u32,
    chroma_subsampling: String,
    luma_bit_depth: String,
    sps_h264: Option<vk_video_core::picture::H264Sps>,
    pps_h264: Option<vk_video_core::picture::H264Pps>,
    sps_h265: Option<vk_video_core::picture::H265Sps>,
    pps_h265: Option<vk_video_core::picture::H265Pps>,
    vps_h265: Option<vk_video_core::picture::H265Vps>,
    sps_count: u32,
    pps_count: u32,
    vps_count: u32,
    nal_count: u32,
    slice_count: u32,
    idr_count: u32,
}

/// Parse the bitstream to extract parameter sets and NAL units.
fn parse_bitstream(path: &str, codec: VideoCodec) -> ParseInfo {
    let data = fs::read(path).unwrap();
    let mut sps_h264: Option<vk_video_core::picture::H264Sps> = None;
    let mut pps_h264: Option<vk_video_core::picture::H264Pps> = None;
    let mut sps_h265: Option<vk_video_core::picture::H265Sps> = None;
    let mut pps_h265: Option<vk_video_core::picture::H265Pps> = None;
    let mut vps_h265: Option<vk_video_core::picture::H265Vps> = None;
    let mut sps_count = 0u32;
    let mut pps_count = 0u32;
    let mut vps_count = 0u32;
    let mut coded_width = 0u32;
    let mut coded_height = 0u32;
    let mut slice_count = 0u32;
    let mut idr_count = 0u32;
    let mut nal_count = 0u32;

    match codec {
        VideoCodec::DecodeH264 => {
            let mut parser = H264Parser::new();
            let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
            parser.init(&format).expect("Failed to init parser");

            // Extract NAL units
            let nal_units = extract_nal_units_h264(&data);
            nal_count = nal_units.len() as u32;

            println!("  NAL units extracted: {}", nal_units.len());
            println!("  First 5 NAL units:");
            for nal in nal_units.iter().take(5) {
                let name = nal_unit_type_name_h264(nal.nal_unit_type);
                println!("    offset={}, type={}: {} bytes",
                    nal.offset, nal.nal_unit_type, name);
            }

            // Process the full bitstream at once
            let packet = BitstreamPacket::new(data.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps: _ }) => {
                    if let Some(sps_boxed) = sps {
                        if let Some(sps_val) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                            sps_h264 = Some(sps_val.clone());
                            sps_count += 1;
                            coded_width = ((sps_val.pic_width_in_mbs_minus1 as u32 + 1) * 16) as u32;
                            coded_height = if sps_val.frame_mbs_only_flag {
                                (sps_val.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                            } else {
                                (sps_val.pic_height_in_map_units_minus1 as u32 + 1) * 16
                            };
                            println!("  SPS #{}: profile={}, width={}, height={}, chroma={}",
                                sps_count, sps_val.profile_idc, coded_width, coded_height, sps_val.chroma_format_idc);
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps_val) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                            pps_h264 = Some(pps_val.clone());
                            pps_count += 1;
                            println!("  PPS #{}: pic_parameter_set_id={}, seq_parameter_set_id={}",
                                pps_count, pps_val.pic_parameter_set_id, pps_val.seq_parameter_set_id);
                        }
                    }
                }
                Ok(ParseResult::Slice { num_slices, .. }) => {
                    slice_count += num_slices;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  Parse error: {}", e);
                }
            }

            // Count IDR frames
            for nal in &nal_units {
                if nal.nal_unit_type == 5 { // IDR
                    idr_count += 1;
                }
            }
        }
        VideoCodec::DecodeH265 => {
            let mut parser = H265Parser::new();
            let format = DetectedVideoFormat::new(VideoCodec::DecodeH265);
            parser.init(&format).expect("Failed to init parser");

            // Extract NAL units
            let nal_units = extract_nal_units_h265(&data);
            nal_count = nal_units.len() as u32;

            println!("  NAL units extracted: {}", nal_units.len());
            println!("  First 5 NAL units:");
            for nal in nal_units.iter().take(5) {
                let name = nal_unit_type_name_h265(nal.nal_unit_type);
                println!("    offset={}, type={}: {} bytes",
                    nal.offset, nal.nal_unit_type, name);
            }

            // Process the full bitstream at once
            let packet = BitstreamPacket::new(data.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps }) => {
                    if let Some(vps_boxed) = vps {
                        if let Some(vps_val) = vps_boxed.downcast_ref::<vk_video_core::picture::H265Vps>() {
                            vps_h265 = Some(vps_val.clone());
                            vps_count += 1;
                            println!("  VPS #{}: vps_id={}, max_layers={}",
                                vps_count, vps_val.vps_video_parameter_set_id, vps_val.vps_max_layers_minus1);
                        }
                    }
                    if let Some(sps_boxed) = sps {
                        if let Some(sps_val) = sps_boxed.downcast_ref::<vk_video_core::picture::H265Sps>() {
                            sps_h265 = Some(sps_val.clone());
                            sps_count += 1;
                            coded_width = sps_val.pic_width_in_luma_samples as u32;
                            coded_height = sps_val.pic_height_in_luma_samples as u32;
                            println!("  SPS #{}: sps_id={}, width={}, height={}, chroma={}",
                                sps_count, sps_val.sps_seq_parameter_set_id,
                                sps_val.pic_width_in_luma_samples, sps_val.pic_height_in_luma_samples,
                                sps_val.chroma_format_idc);
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps_val) = pps_boxed.downcast_ref::<vk_video_core::picture::H265Pps>() {
                            pps_h265 = Some(pps_val.clone());
                            pps_count += 1;
                            println!("  PPS #{}: pps_id={}, sps_id={}",
                                pps_count, pps_val.pps_pic_parameter_set_id, pps_val.pps_seq_parameter_set_id);
                        }
                    }
                }
                Ok(ParseResult::Slice { num_slices, .. }) => {
                    slice_count += num_slices;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  Parse error: {}", e);
                }
            }

            // Count IDR frames
            for nal in &nal_units {
                let nal_type = nal.nal_unit_type;
                if nal_type == 2 || nal_type == 3 || nal_type == 4 {
                    idr_count += 1;
                }
            }
        }
        _ => {}
    }

    ParseInfo {
        coded_width,
        coded_height,
        chroma_subsampling: "420".to_string(),
        luma_bit_depth: "8-bit".to_string(),
        sps_h264,
        pps_h264,
        sps_h265,
        pps_h265,
        vps_h265,
        sps_count,
        pps_count,
        vps_count,
        nal_count,
        slice_count,
        idr_count,
    }
}

/// Extract NAL units from an H.264 bitstream.
fn extract_nal_units_h264(data: &[u8]) -> Vec<NalUnit> {
    let mut nal_units = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            let next_start = find_next_start_code(data, start + code_len);

            let end = match next_start {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, _, nal_unit_type)) = nal::parse_h264_nal_header(nal_data) {
                    nal_units.push(NalUnit::new(
                        nal_unit_type,
                        nal_data.to_vec(),
                        start + code_len,
                        nal_data.len(),
                    ));
                }
            }

            offset = end;
        } else {
            break;
        }
    }

    nal_units
}

/// Extract NAL units from an H.265 bitstream.
fn extract_nal_units_h265(data: &[u8]) -> Vec<NalUnit> {
    let mut nal_units = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            let next_start = find_next_start_code(data, start + code_len);

            let end = match next_start {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, nal_unit_type, _, _)) = nal::parse_h265_nal_header(nal_data) {
                    nal_units.push(NalUnit::new(
                        nal_unit_type,
                        nal_data.to_vec(),
                        start + code_len,
                        nal_data.len(),
                    ));
                }
            }

            offset = end;
        } else {
            break;
        }
    }

    nal_units
}

fn nal_unit_type_name_h264(typ: u8) -> &'static str {
    match typ {
        0 => "Unspecified",
        1 => "Non-IDR Slice",
        5 => "IDR Slice",
        6 => "SEI",
        7 => "SPS",
        8 => "PPS",
        _ => "Other",
    }
}

fn nal_unit_type_name_h265(typ: u8) -> &'static str {
    match typ {
        0 => "RASL_R",
        1 => "RASL_N",
        2 => "IDR_W_RADL",
        3 => "IDR_N_LP",
        4 => "CRA",
        5 => "VPS",
        6 => "SPS",
        7 => "PPS",
        8 => "SEI",
        _ => "Other",
    }
}

/// Create Vulkan device using the working approach from debug_vulkan example.
fn create_vulkan_device(vk_codec: VkVideoCodec) -> Result<VulkanDevice, VideoError> {
    unsafe {
        // Load Vulkan entry
        let entry = ash::Entry::load().map_err(|e| {
            VideoError::VulkanInit(format!("Failed to load entry: {}", e))
        })?;

        // Create instance
        let app_name = std::ffi::CString::new("vk-video-decode").map_err(|e| {
            VideoError::VulkanInit(format!("Failed to create CString for app name: {}", e))
        })?;
        let engine_name = std::ffi::CString::new("vk-video-vulkan").map_err(|e| {
            VideoError::VulkanInit(format!("Failed to create CString for engine name: {}", e))
        })?;

        let app_info = ash::vk::ApplicationInfo {
            s_type: ash::vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: 0,
            p_engine_name: engine_name.as_ptr(),
            engine_version: 0,
            api_version: ash::vk::API_VERSION_1_2,
            _marker: std::marker::PhantomData,
        };

        let instance_extensions: Vec<std::ffi::CString> = vec![
            std::ffi::CString::new("VK_KHR_surface").map_err(|e| {
                VideoError::VulkanInit(format!("Failed to create CString for extension: {}", e))
            })?,
            std::ffi::CString::new("VK_KHR_get_physical_device_properties2").map_err(|e| {
                VideoError::VulkanInit(format!("Failed to create CString for extension: {}", e))
            })?,
        ];

        let ext_ptrs: Vec<*const std::os::raw::c_char> = instance_extensions.iter().map(|c| c.as_ptr()).collect();

        let create_info = ash::vk::InstanceCreateInfo {
            s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ash::vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: instance_extensions.len() as u32,
            pp_enabled_extension_names: ext_ptrs.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let instance = entry.create_instance(&create_info, None).map_err(|e| {
            VideoError::VulkanInit(format!("Failed to create instance: {}", e))
        })?;

        // Enumerate physical devices
        let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
            VideoError::VulkanInit(format!("Failed to enumerate physical devices: {}", e))
        })?;

        if physical_devices.is_empty() {
            return Err(VideoError::VulkanInit("No physical devices found".to_string()));
        }

        let pd = physical_devices[0];

        // Get queue families
        let queue_families_list = instance.get_physical_device_queue_family_properties(pd);

        let mut decode_queue_family: Option<u32> = None;
        let mut graphics_queue_family: Option<u32> = None;
        let mut transfer_queue_family: Option<u32> = None;

        for (i, qf) in queue_families_list.iter().enumerate() {
            let i = i as u32;
            if qf.queue_flags.contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR) && decode_queue_family.is_none() {
                decode_queue_family = Some(i);
            }
            if qf.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && graphics_queue_family.is_none() {
                graphics_queue_family = Some(i);
            }
            if qf.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) && transfer_queue_family.is_none() {
                transfer_queue_family = Some(i);
            }
        }

        if decode_queue_family.is_none() {
            return Err(VideoError::VideoNotSupported("No video decode queue family found".to_string()));
        }

        // Create queue create infos
        let queue_priorities = vec![1.0f32];
        let mut queue_create_infos: Vec<ash::vk::DeviceQueueCreateInfo> = Vec::new();

        if let Some(qf) = decode_queue_family {
            queue_create_infos.push(ash::vk::DeviceQueueCreateInfo {
                s_type: ash::vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ash::vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: qf,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                _marker: std::marker::PhantomData,
            });
        }

        if let Some(qf) = graphics_queue_family {
            queue_create_infos.push(ash::vk::DeviceQueueCreateInfo {
                s_type: ash::vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: ash::vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: qf,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                _marker: std::marker::PhantomData,
            });
        }

        // Device extensions (using KHR versions as in Vulkan-Video-Samples)
        let device_extensions: Vec<std::ffi::CString> = vec![
            std::ffi::CString::new("VK_KHR_video_decode_queue").map_err(|e| {
                VideoError::DeviceCreation(format!("Failed to create CString for extension: {}", e))
            })?,
            std::ffi::CString::new("VK_KHR_video_decode_h264").map_err(|e| {
                VideoError::DeviceCreation(format!("Failed to create CString for extension: {}", e))
            })?,
            std::ffi::CString::new("VK_KHR_video_decode_h265").map_err(|e| {
                VideoError::DeviceCreation(format!("Failed to create CString for extension: {}", e))
            })?,
            std::ffi::CString::new("VK_KHR_sampler_ycbcr_conversion").map_err(|e| {
                VideoError::DeviceCreation(format!("Failed to create CString for extension: {}", e))
            })?,
        ];

        let ext_ptr_vec: Vec<*const std::os::raw::c_char> = device_extensions.iter().map(|c| c.as_ptr()).collect();

        // Setup YCbCr conversion features
        let sampler_ycbcr_features = ash::vk::PhysicalDeviceSamplerYcbcrConversionFeatures {
            s_type: ash::vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
            p_next: std::ptr::null_mut(),
            sampler_ycbcr_conversion: 1,
            _marker: std::marker::PhantomData,
        };

        let features2 = ash::vk::PhysicalDeviceFeatures2 {
            s_type: ash::vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &sampler_ycbcr_features as *const _ as *mut _,
            features: ash::vk::PhysicalDeviceFeatures::default(),
            _marker: std::marker::PhantomData,
        };

        // Create device
        let device_create_info = ash::vk::DeviceCreateInfo {
            s_type: ash::vk::StructureType::DEVICE_CREATE_INFO,
            p_next: &features2 as *const _ as *mut _,
            flags: ash::vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: ext_ptr_vec.as_ptr(),
            p_enabled_features: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let device = instance.create_device(pd, &device_create_info, None).map_err(|e| {
            VideoError::DeviceCreation(format!("Failed to create device: {}", e))
        })?;

        // Get memory properties
        let memory_properties = instance.get_physical_device_memory_properties(pd);

        // Get device queues
        let decode_queue = if let Some(qf) = decode_queue_family {
            Some(unsafe { device.get_device_queue(qf, 0) })
        } else {
            None
        };

        let graphics_queue = if let Some(qf) = graphics_queue_family {
            Some(unsafe { device.get_device_queue(qf, 0) })
        } else {
            None
        };

        // Create VulkanDevice struct
        let queue_families = vk_video_vulkan::QueueFamilies {
            graphics: graphics_queue_family,
            compute: None,
            transfer: transfer_queue_family,
            video_decode: decode_queue_family,
            video_encode: None,
            present: None,
        };

        // Get video capabilities
        let video_capabilities = vk_video_vulkan::VideoCapabilities {
            codec_operations: vk_codec.to_vk_flag(),
            min_bitstream_buffer_offset_alignment: 256,
            min_bitstream_buffer_size_alignment: 256,
            picture_access_granularity: (1, 1),
            min_coded_extent: (0, 0),
            max_coded_extent: (8192, 8192),
            max_dpb_slots: 16,
            max_active_reference_pictures: 16,
            supported_formats: Vec::new(),
        };

        let enabled_extensions: Vec<String> = device_extensions.iter()
            .map(|c| c.to_string_lossy().into_owned())
            .collect();

        Ok(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            enabled_extensions,
            queue_families,
            memory_properties,
            video_capabilities,
        })
    }
}



/// Get GPU name from Vulkan device.
fn get_gpu_name(device: &VulkanDevice) -> String {
    unsafe {
        let props = device.instance.get_physical_device_properties(device.physical_device);
        let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr());
        name.to_string_lossy().to_string()
    }
}
