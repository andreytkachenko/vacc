//! Minimal VP9 capability diagnostic.

use ash::vk;
use std::ffi::{CStr, CString};

fn main() {
    println!("=== Minimal VP9 Diagnostic ===\n");

    let entry = unsafe { ash::Entry::load() }.expect("Entry load failed");

    let app_name = CString::new("vp9_min").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .api_version(vk::API_VERSION_1_2);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None).expect("Instance failed") };

    let phys_devs = unsafe { instance.enumerate_physical_devices().expect("Enumerate failed") };
    let phys_dev = phys_devs[0];

    let props = unsafe { instance.get_physical_device_properties(phys_dev) };
    let gpu = unsafe { CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy() };
    println!("GPU: {}\n", gpu);

    // Check queue family video properties
    println!("--- Queue Video Properties ---");
    let qf_props = unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

    let mut qf_props2: Vec<vk::QueueFamilyProperties2> = Vec::new();
    let mut video_props: Vec<vk::QueueFamilyVideoPropertiesKHR> = Vec::new();

    for _ in 0..qf_props.len() {
        let vp = vk::QueueFamilyVideoPropertiesKHR::default();
        let mut qfp2 = vk::QueueFamilyProperties2::default();
        // Chain: qfp2.p_next -> vp
        video_props.push(vp);
        qf_props2.push(qfp2);
    }

    // Set up chains manually
    for (i, vp) in video_props.iter_mut().enumerate() {
        qf_props2[i].p_next = std::ptr::from_mut::<vk::QueueFamilyVideoPropertiesKHR>(vp) as *mut _;
    }

    unsafe {
        instance.get_physical_device_queue_family_properties2(phys_dev, &mut qf_props2);
    }

    for (i, vp) in video_props.iter().enumerate() {
        let ops = vp.video_codec_operations;
        let raw = ops.as_raw();
        println!("  Queue {}: ops=0x{:08x}", i, raw);
        if raw & 1 != 0 { println!("    DECODE_H264"); }
        if raw & 2 != 0 { println!("    DECODE_H265"); }
        if raw & 4 != 0 { println!("    DECODE_AV1"); }
        if raw & 8 != 0 { println!("    DECODE_VP9  <-- VP9 DECODE SUPPORTED"); }
        if raw & 16 != 0 { println!("    ENCODE_H264"); }
        if raw & 32 != 0 { println!("    ENCODE_H265"); }
        if raw & 64 != 0 { println!("    ENCODE_AV1"); }
    }

    // Check device extensions
    println!("\n--- Device Extensions ---");
    let exts = unsafe { instance.enumerate_device_extension_properties(phys_dev) }.unwrap_or_default();
    for ext in &exts {
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr().cast()).to_string_lossy() };
        if name.contains("vp9") || name.contains("VP9") {
            println!("  {} (spec version: {})", name, ext.spec_version);
        }
    }

    // Find decode queue
    let decode_qf = video_props.iter().position(|vp| vp.video_codec_operations.as_raw() & 8 != 0)
        .map(|i| i as u32).or_else(|| {
            qf_props.iter().position(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
                .map(|i| i as u32)
        });

    let Some(decode_qf) = decode_qf else {
        eprintln!("No decode queue found");
        std::process::exit(1);
    };

    // Create device with VP9 decode extension
    let device_extensions: Vec<CString> = vec![
        CString::new("VK_KHR_video_queue").unwrap(),
        CString::new("VK_KHR_video_decode_queue").unwrap(),
        CString::new("VK_KHR_video_decode_vp9").unwrap(),
    ];
    let ext_ptrs: Vec<*const std::os::raw::c_char> =
        device_extensions.iter().map(|c| c.as_ptr()).collect();

    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(decode_qf)
        .queue_priorities(&[1.0]);
    let queue_infos = vec![queue_info];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs);

    let _device = unsafe {
        instance.create_device(phys_dev, &device_info, None).expect("Device creation failed")
    };
    println!("\nDevice created on queue {}\n", decode_qf);

    // Capability query
    println!("--- Capability Query ---");

    const VP9_PROFILE_INFO_STYPE: i32 = 1000514003;
    const VP9_CAPABILITIES_STYPE: i32 = 1000514001;

    #[repr(C)]
    struct VideoDecodeVP9ProfileInfoKHR {
        s_type: vk::StructureType,
        p_next: *const std::os::raw::c_void,
        std_profile: u32,
    }

    #[repr(C)]
    struct VideoDecodeVP9CapabilitiesKHR {
        s_type: vk::StructureType,
        p_next: *mut std::os::raw::c_void,
        max_level: u32,
    }

    let get_caps_fn = unsafe {
        entry.get_instance_proc_addr(
            instance.handle(),
            b"vkGetPhysicalDeviceVideoCapabilitiesKHR\0".as_ptr().cast(),
        )
    }.expect("Fn not found");

    type GetCapsFn = unsafe extern "system" fn(
        vk::PhysicalDevice,
        *const vk::VideoProfileInfoKHR<'_>,
        *mut vk::VideoCapabilitiesKHR,
    ) -> vk::Result;
    let fn_ptr: GetCapsFn = unsafe { std::mem::transmute(get_caps_fn) };

    let vp9_profile_info = VideoDecodeVP9ProfileInfoKHR {
        s_type: vk::StructureType::from_raw(VP9_PROFILE_INFO_STYPE),
        p_next: std::ptr::null(),
        std_profile: 0,
    };

    println!("  Input: ProfileInfoKHR -> VP9ProfileInfoKHR(stdProfile=0)");

    let mut profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::from_raw(8))
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
    profile_info.p_next = &vp9_profile_info as *const _ as *const _;

    let mut vp9_caps = VideoDecodeVP9CapabilitiesKHR {
        s_type: vk::StructureType::from_raw(VP9_CAPABILITIES_STYPE),
        p_next: std::ptr::null_mut(),
        max_level: 0,
    };

    let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
    decode_caps.p_next = &mut vp9_caps as *mut _ as *mut _;

    let mut caps = vk::VideoCapabilitiesKHR::default();
    caps.p_next = &mut decode_caps as *mut _ as *mut _;

    println!("  Output: CapsKHR -> DecodeCapsKHR -> VP9CapsKHR");

    let result = unsafe { fn_ptr(phys_dev, &profile_info, &mut caps) };
    println!("\n  Result: {:?}", result);

    if result == vk::Result::SUCCESS {
        println!("  *** SUCCESS ***");
        println!("  maxDPB={}", caps.max_dpb_slots);
        println!("  maxExtent={}x{}", caps.max_coded_extent.width, caps.max_coded_extent.height);
        println!("  minExtent={}x{}", caps.min_coded_extent.width, caps.min_coded_extent.height);
        println!("  grain={}x{}", caps.picture_access_granularity.width, caps.picture_access_granularity.height);
        println!("  vp9Caps.maxLevel={}", vp9_caps.max_level);
    } else {
        println!("  *** FAILED ***");
        // Dump raw bytes of profile_info
        println!("\n  Raw profile_info bytes:");
        let ptr = &profile_info as *const _ as *const u8;
        for i in (0..40).step_by(16) {
            let bytes: Vec<String> = (0..16).map(|j| {
                unsafe { format!("{:02x}", *ptr.add(i + j)) }
            }).collect();
            println!("    {:02x}: {}", i, bytes.join(" "));
        }
    }

    println!("\n--- Done ---");
}
