//! Test program to enumerate all supported VP9 decode formats.
//!
//! Usage:
//!   cargo run --example test_vp9_formats

use ash::vk;
use vk_video_vulkan::{
    vp9::{vp9_vk_constants, VideoDecodeVP9ProfileInfoKHR},
    VideoCodec, VideoDeviceBuilder,
};

fn main() {
    println!("=== VP9 Decode Format Enumeration ===\n");

    let device = match VideoDeviceBuilder::new()
        .with_validation(false)
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::from_raw(
            vp9_vk_constants::DECODE_VP9,
        ))
        .build()
    {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to create device: {}", e);
            std::process::exit(1);
        }
    };

    // VP9 profiles to test. Per the VP9 bitstream spec:
    //   Profile 0: 8-bit, 4:2:0 (implicit)
    //   Profile 1: 8-bit, explicit subsampling
    //   Profile 2: 10/12-bit, 4:2:0 (implicit)
    //   Profile 3: 10/12-bit, explicit subsampling
    let profiles: [(u32, vk::VideoComponentBitDepthFlagsKHR, &str); 6] = [
        (0, vk::VideoComponentBitDepthFlagsKHR::TYPE_8, "Profile 0 (4:2:0, 8-bit)"),
        (1, vk::VideoComponentBitDepthFlagsKHR::TYPE_8, "Profile 1 (explicit subsampling, 8-bit)"),
        (2, vk::VideoComponentBitDepthFlagsKHR::TYPE_10, "Profile 2 (4:2:0, 10-bit)"),
        (2, vk::VideoComponentBitDepthFlagsKHR::TYPE_12, "Profile 2 (4:2:0, 12-bit)"),
        (3, vk::VideoComponentBitDepthFlagsKHR::TYPE_10, "Profile 3 (explicit subsampling, 10-bit)"),
        (3, vk::VideoComponentBitDepthFlagsKHR::TYPE_12, "Profile 3 (explicit subsampling, 12-bit)"),
    ];

    let mut any_supported = false;

    for (profile_idc, bit_depth, profile_name) in &profiles {
        println!("\nTesting VP9 {}:", profile_name);
        println!("----------------------------------------");

        // Build VP9-specific profile info
        let vp9_profile = VideoDecodeVP9ProfileInfoKHR {
            s_type: vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR),
            p_next: std::ptr::null(),
            std_profile: *profile_idc,
            _marker: Default::default(),
        };

        // Build VideoProfileInfoKHR with codec-specific info chained
        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: &vp9_profile as *const _ as *const _,
            video_codec_operation: vk::VideoCodecOperationFlagsKHR::from_raw(vp9_vk_constants::DECODE_VP9),
            chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            luma_bit_depth: *bit_depth,
            chroma_bit_depth: *bit_depth,
            _marker: Default::default(),
        };

        // Query using VkVideoProfileListInfoKHR (same as C++ reference)
        let profile_list = vk::VideoProfileListInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
            p_next: std::ptr::null(),
            profile_count: 1,
            p_profiles: &profile_info,
            _marker: Default::default(),
        };

        let format_info = vk::PhysicalDeviceVideoFormatInfoKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
            p_next: &profile_list as *const _ as *const _,
            image_usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
            _marker: Default::default(),
        };

        // Get function pointer
        let get_format_props_fn = unsafe {
            device.entry.get_instance_proc_addr(
                device.instance.handle(),
                c"vkGetPhysicalDeviceVideoFormatPropertiesKHR".as_ptr(),
            )
        };

        let Some(fn_ptr_raw) = get_format_props_fn else {
            eprintln!("  vkGetPhysicalDeviceVideoFormatPropertiesKHR not found!");
            continue;
        };

        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::PhysicalDevice,
                *const vk::PhysicalDeviceVideoFormatInfoKHR<'_>,
                *mut u32,
                *mut vk::VideoFormatPropertiesKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(fn_ptr_raw);

            // First call to get count
            let mut count: u32 = 0;
            let result = fn_ptr(
                device.physical_device,
                &format_info,
                &mut count,
                std::ptr::null_mut(),
            );

            if result != vk::Result::SUCCESS {
                eprintln!("  Query failed with result: {:?}", result);
                continue;
            }

            if count == 0 {
                println!("  No supported formats for this profile");
                continue;
            }

            any_supported = true;
            println!("  Found {} supported format(s):", count);

            // Second call to get actual properties
            let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
            for p in props.iter_mut() {
                p.s_type = vk::StructureType::VIDEO_FORMAT_PROPERTIES_KHR;
            }

            let result = fn_ptr(
                device.physical_device,
                &format_info,
                &mut count,
                props.as_mut_ptr(),
            );

            if result != vk::Result::SUCCESS {
                eprintln!("  Failed to get format properties: {:?}", result);
                continue;
            }

            props.truncate(count as usize);

            for (i, p) in props.iter().enumerate() {
                println!(
                    "    [{}] format={:?}, usage={:?}",
                    i, p.format, p.image_usage_flags
                );
            }
        }
    }

    if !any_supported {
        println!("\n=== No VP9 decode formats supported on this device ===");
    } else {
        println!("\n=== Done ===");
    }
}
