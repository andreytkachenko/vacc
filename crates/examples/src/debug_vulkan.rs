//! Step-by-step Vulkan video decode debugging, aligned with Vulkan-Video-Samples C++ examples.
//!
//! Reference: https://github.com/KhronosGroup/Vulkan-Video-Samples

fn main() {
    println!("=== Step-by-step Vulkan Video Decode Debug ===\n");
    
    // Step 1: Load Vulkan entry
    println!("Step 1: Loading Vulkan entry...");
    let entry = unsafe { ash::Entry::load() }.expect("Failed to load entry");
    println!("  OK: Entry loaded\n");
    
    // Step 2: Create instance (aligned with Vulkan-Video-Samples)
    println!("Step 2: Creating Vulkan instance...");
    let app_name = std::ffi::CString::new("vk-video-debug").unwrap();
    let engine_name = std::ffi::CString::new("vk-video-vulkan").unwrap();
    
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
    
    // Instance extensions from Vulkan-Video-Samples
    let instance_extensions: Vec<std::ffi::CString> = vec![
        std::ffi::CString::new("VK_KHR_surface").unwrap(),
        std::ffi::CString::new("VK_KHR_get_physical_device_properties2").unwrap(),
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
    
    let instance = unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");
    println!("  OK: Instance created\n");
    
    // Step 3: Enumerate physical devices
    println!("Step 3: Enumerating physical devices...");
    let devices = unsafe { instance.enumerate_physical_devices() }.expect("Failed to enumerate");
    println!("  OK: Found {} physical device(s)\n", devices.len());
    
    let pd = devices[0];
    
    // Step 4: Get device properties and queue families
    println!("Step 4: Getting device properties and queue families...");
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
    println!("  Device: {}", name.to_string_lossy());
    
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    println!("  {} queue families:", queue_families.len());
    
    // Find queue families (aligned with Vulkan-Video-Samples)
    let mut decode_queue_family: Option<u32> = None;
    let mut graphics_queue_family: Option<u32> = None;
    let mut transfer_queue_family: Option<u32> = None;
    
    for (i, qf) in queue_families.iter().enumerate() {
        let i = i as u32;
        let mut flags_str = Vec::new();
        if qf.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) { flags_str.push("GRAPHICS"); }
        if qf.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) { flags_str.push("COMPUTE"); }
        if qf.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) { flags_str.push("TRANSFER"); }
        if qf.queue_flags.contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR) { flags_str.push("VIDEO_DECODE"); }
        if qf.queue_flags.contains(ash::vk::QueueFlags::VIDEO_ENCODE_KHR) { flags_str.push("VIDEO_ENCODE"); }
        println!("    Family {}: {} queues, flags=[{}]", i, qf.queue_count, flags_str.join(", "));
        
        // Track queue families (aligned with Vulkan-Video-Samples approach)
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
    
    println!("\n  Selected queue families:");
    println!("    Decode: {:?}", decode_queue_family);
    println!("    Graphics: {:?}", graphics_queue_family);
    println!("    Transfer: {:?}", transfer_queue_family);
    
    if decode_queue_family.is_none() {
        eprintln!("\n  ERROR: No video decode queue family found!");
        std::process::exit(1);
    }
    println!();
    
    // Step 5: Query available device extensions (aligned with Vulkan-Video-Samples)
    println!("Step 5: Querying available device extensions...");
    let device_extensions_available = unsafe {
        instance.enumerate_device_extension_properties(pd)
    }.expect("Failed to enumerate device extensions");
    println!("  {} extensions available:", device_extensions_available.len());
    
    // Check which video decode extensions are available
    let has_decode_queue = device_extensions_available.iter().any(|e| {
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_string_lossy() == "VK_KHR_video_decode_queue"
    });
    let has_h264_ext = device_extensions_available.iter().any(|e| {
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_string_lossy() == "VK_EXT_video_decode_h264"
    });
    let has_h265_ext = device_extensions_available.iter().any(|e| {
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_string_lossy() == "VK_EXT_video_decode_h265"
    });
    let has_h264_khr = device_extensions_available.iter().any(|e| {
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_string_lossy() == "VK_KHR_video_decode_h264"
    });
    let has_h265_khr = device_extensions_available.iter().any(|e| {
        let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        name.to_string_lossy() == "VK_KHR_video_decode_h265"
    });
    
    println!("  VK_KHR_video_decode_queue: {}", has_decode_queue);
    println!("  VK_EXT_video_decode_h264: {}", has_h264_ext);
    println!("  VK_EXT_video_decode_h265: {}", has_h265_ext);
    println!("  VK_KHR_video_decode_h264: {}", has_h264_khr);
    println!("  VK_KHR_video_decode_h265: {}", has_h265_khr);
    
    // List all video-related extensions
    println!("  Video-related extensions:");
    for ext in &device_extensions_available {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        let name_str = name.to_string_lossy();
        if name_str.contains("video") || name_str.contains("decode") || name_str.contains("encode") {
            println!("    {}", name_str);
        }
    }
    println!();
    
    // Step 6: Query YCbCr conversion support
    println!("Step 6: Querying YCbCr conversion support...");
    let mut ycbcr_features = ash::vk::PhysicalDeviceSamplerYcbcrConversionFeatures {
        s_type: ash::vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
        p_next: std::ptr::null_mut(),
        sampler_ycbcr_conversion: 1,
        _marker: std::marker::PhantomData,
    };
    
    let mut features2 = ash::vk::PhysicalDeviceFeatures2 {
        s_type: ash::vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
        p_next: &ycbcr_features as *const _ as *mut _,
        features: ash::vk::PhysicalDeviceFeatures::default(),
        _marker: std::marker::PhantomData,
    };
    
    unsafe {
        instance.get_physical_device_features2(pd, &mut features2);
    }
    
    println!("  YCbCr conversion supported: {}", ycbcr_features.sampler_ycbcr_conversion != 0);
    println!("  OK: YCbCr conversion support queried\n");
    
    // Step 7: Create Vulkan device (aligned with Vulkan-Video-Samples)
    println!("Step 7: Creating Vulkan device...");
    
    // Create queue create infos (aligned with Vulkan-Video-Samples)
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
    
    println!("  Created {} queue create infos", queue_create_infos.len());
    
    // Device extensions (aligned with Vulkan-Video-Samples)
    // First try without video decode extensions to test basic device creation
    let mut device_extensions: Vec<std::ffi::CString> = vec![
        std::ffi::CString::new("VK_KHR_sampler_ycbcr_conversion").unwrap(),
    ];
    
    // Try adding video decode extensions
    println!("  Trying device creation without video decode extensions first...");
    let ext_ptr_vec: Vec<*const std::os::raw::c_char> = device_extensions.iter().map(|c| c.as_ptr()).collect();
    
    let device_create_info_no_video = ash::vk::DeviceCreateInfo {
        s_type: ash::vk::StructureType::DEVICE_CREATE_INFO,
        p_next: &ycbcr_features as *const _ as *mut _,
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
    
    let device_result = unsafe { instance.create_device(pd, &device_create_info_no_video, None) };
    
    let device = match device_result {
        Ok(_dev) => {
            println!("  OK: Basic device created (without video decode extensions)");
            println!("    Now trying with video decode extensions...");
            
            // Now try with video decode extensions (using KHR versions as in Vulkan-Video-Samples)
            device_extensions.clear();
            device_extensions.extend_from_slice(&[
                std::ffi::CString::new("VK_KHR_video_decode_queue").unwrap(),
                std::ffi::CString::new("VK_KHR_video_decode_h264").unwrap(),
                std::ffi::CString::new("VK_KHR_video_decode_h265").unwrap(),
                std::ffi::CString::new("VK_KHR_sampler_ycbcr_conversion").unwrap(),
            ]);
            
            let ext_ptr_vec2: Vec<*const std::os::raw::c_char> = device_extensions.iter().map(|c| c.as_ptr()).collect();
            
            let device_create_info_with_video = ash::vk::DeviceCreateInfo {
                s_type: ash::vk::StructureType::DEVICE_CREATE_INFO,
                p_next: &ycbcr_features as *const _ as *mut _,
                flags: ash::vk::DeviceCreateFlags::empty(),
                queue_create_info_count: queue_create_infos.len() as u32,
                p_queue_create_infos: queue_create_infos.as_ptr(),
                enabled_layer_count: 0,
                pp_enabled_layer_names: std::ptr::null(),
                enabled_extension_count: device_extensions.len() as u32,
                pp_enabled_extension_names: ext_ptr_vec2.as_ptr(),
                p_enabled_features: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };
            
            match unsafe { instance.create_device(pd, &device_create_info_with_video, None) } {
                Ok(dev) => {
                    println!("  OK: Device created WITH video decode extensions!");
                    println!("    Device handle: {:?}\n", dev.handle());
                    dev
                }
                Err(e) => {
                    println!("  ERROR: Failed with video decode extensions: {:?}\n", e);
                    eprintln!("  Note: Video decode extensions may not be available on this GPU.");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            println!("  ERROR: Failed to create basic device: {:?}\n", e);
            eprintln!("  This is unexpected - even basic device creation failed.");
            std::process::exit(1);
        }
    };
    
    // Step 8: Get device queues
    println!("Step 8: Getting device queues...");
    let decode_queue = if let Some(qf) = decode_queue_family {
        let queue = unsafe { device.get_device_queue(qf, 0) };
        println!("  Decode queue: {:?}", queue);
        Some(queue)
    } else {
        None
    };
    
    if let Some(qf) = graphics_queue_family {
        let queue = unsafe { device.get_device_queue(qf, 0) };
        println!("  Graphics queue: {:?}", queue);
    }
    println!();
    
    // Step 9: Get memory properties
    println!("Step 9: Getting memory properties...");
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(pd) };
    println!("  {} memory types", memory_properties.memory_type_count);
    println!("  {} memory heaps", memory_properties.memory_heap_count);
    println!("  OK: Memory properties retrieved\n");
    
    // Summary
    println!("=== Debug Summary ===");
    println!("  Device: {}", name.to_string_lossy());
    println!("  Decode queue family: {:?}", decode_queue_family);
    println!("  Graphics queue family: {:?}", graphics_queue_family);
    println!("  Transfer queue family: {:?}", transfer_queue_family);
    println!("  Device extensions: {}", device_extensions.len());
    println!("  YCbCr conversion: enabled");
    println!("  OK: All steps completed successfully!");
}
