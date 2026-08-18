//! Vulkan device initialization for video decode.

use ash::vk;
use std::ffi::CString;

use super::vp9::{vp9_vk_constants, VideoDecodeVP9CapabilitiesKHR, VideoDecodeVP9ProfileInfoKHR};
use super::{AppInfo, VideoError, VideoResult};

/// PhysicalDeviceVideoDecodeFeaturesKHR - not available in ash 0.38, define manually.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct PhysicalDeviceVideoDecodeFeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    video_decode_h264: u32,
    video_decode_h265: u32,
    video_decode_av1: u32,
    video_decode_vp9: u32,
}

const PHYSICAL_DEVICE_VIDEO_DECODE_FEATURES_KHR: vk::StructureType =
    vk::StructureType::from_raw(1000346000);

/// PhysicalDeviceVideoMaintenance2FeaturesKHR - not available in ash 0.38, define manually.
/// The videoMaintenance2 feature makes videoSessionParameters = VK_NULL_HANDLE legal in
/// vkCmdBeginVideoCodingKHR for AV1/VP9/H.264/H.265 decode
/// (VUID-VkVideoBeginCodingInfoKHR-videoSession-09261).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct PhysicalDeviceVideoMaintenance2FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    video_maintenance2: u32,
}

const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR: vk::StructureType =
    vk::StructureType::from_raw(1000586000);

/// PhysicalDeviceVideoMaintenance1FeaturesKHR - not re-exported by ash 0.38, define manually.
/// The videoMaintenance1 feature is REQUIRED for VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR
/// (VUID-VkVideoSessionCreateInfoKHR-flags-09236). The C++ reference (VulkanDeviceContext.cpp:816-841)
/// enables it: GetPhysicalDeviceFeatures2 fills the chain with supported values, so
/// videoMaintenance1 ends up VK_TRUE when the device is created. Without it, the NVIDIA driver
/// silently skips every decode (all-zero DPB) even though vkCreateVideoSessionKHR succeeds.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct PhysicalDeviceVideoMaintenance1FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    video_maintenance1: u32,
}

const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR: vk::StructureType =
    vk::StructureType::from_raw(1000515000);

/// Queue family indices found during device selection.
#[derive(Debug, Clone, Default)]
pub struct QueueFamilies {
    pub graphics: Option<u32>,
    pub compute: Option<u32>,
    pub transfer: Option<u32>,
    pub video_decode: Option<u32>,
    pub video_encode: Option<u32>,
    pub present: Option<u32>,
}

/// Video capabilities queried from the GPU.
#[derive(Debug, Clone)]
pub struct VideoCapabilities {
    pub codec_operations: vk::VideoCodecOperationFlagsKHR,
    pub min_bitstream_buffer_offset_alignment: u32,
    pub min_bitstream_buffer_size_alignment: u32,
    pub picture_access_granularity: vk::Extent2D,
    pub min_coded_extent: vk::Extent2D,
    pub max_coded_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub supported_formats: Vec<VideoFormatProperties>,
}

#[derive(Debug, Clone)]
pub struct VideoFormatProperties {
    pub format: vk::Format,
    pub image_tiling: vk::ImageTiling,
    pub image_usage_flags: vk::ImageUsageFlags,
    pub sample_count: vk::SampleCountFlags,
}

/// Video codec type for the Rust API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    DecodeH264,
    DecodeH265,
    DecodeAv1,
    DecodeVp9,
}

impl VideoCodec {
    pub fn to_vk_flag(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            Self::DecodeH264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Self::DecodeH265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            Self::DecodeAv1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            Self::DecodeVp9 => {
                vk::VideoCodecOperationFlagsKHR::from_raw(vp9_vk_constants::DECODE_VP9)
            }
        }
    }
}

/// Vulkan device wrapper for video decode.
pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub enabled_extensions: Vec<String>,
    pub queue_families: QueueFamilies,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub has_validation: bool,
}

pub struct VideoDeviceBuilder {
    app_info: AppInfo,
    enable_validation: bool,
    video_codecs: vk::VideoCodecOperationFlagsKHR,
    num_decode_queues: usize,
    create_graphics_queue: bool,
    create_transfer_queue: bool,
}

impl VideoDeviceBuilder {
    pub fn new() -> Self {
        Self {
            app_info: AppInfo::default(),
            enable_validation: false,
            video_codecs: vk::VideoCodecOperationFlagsKHR::DECODE_H264
                | vk::VideoCodecOperationFlagsKHR::DECODE_H265
                | vk::VideoCodecOperationFlagsKHR::DECODE_AV1
                | vk::VideoCodecOperationFlagsKHR::from_raw(vp9_vk_constants::DECODE_VP9),
            num_decode_queues: 1,
            create_graphics_queue: false,
            create_transfer_queue: false,
        }
    }

    pub fn with_app_info(mut self, info: AppInfo) -> Self {
        self.app_info = info;
        self
    }

    pub fn with_validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    pub fn with_video_codecs(mut self, codecs: vk::VideoCodecOperationFlagsKHR) -> Self {
        self.video_codecs = codecs;
        self
    }

    pub fn with_num_decode_queues(mut self, count: usize) -> Self {
        self.num_decode_queues = count;
        self
    }

    pub fn with_graphics_queue(mut self, enable: bool) -> Self {
        self.create_graphics_queue = enable;
        self
    }

    pub     fn build(self) -> VideoResult<VulkanDevice> {
        eprintln!("[VideoDeviceBuilder] Creating entry...");
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| VideoError::VulkanInit(e.to_string()))?;
        eprintln!("[VideoDeviceBuilder] Creating instance...");
        let (instance, has_validation) = Self::create_instance(&entry, &self)?;
        eprintln!("[VideoDeviceBuilder] Selecting physical device...");
        let (physical_device, queue_families) = Self::select_physical_device(&instance, &self)?;
        eprintln!("[VideoDeviceBuilder] Creating device...");
        let (device, enabled_extensions) =
            Self::create_device(&entry, &instance, &physical_device, &queue_families, &self)?;
        eprintln!("[VideoDeviceBuilder] Getting memory properties...");
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let debug_messenger = if has_validation {
            Self::create_debug_messenger(&entry, &instance)?
        } else {
            vk::DebugUtilsMessengerEXT::null()
        };

        Ok(VulkanDevice {
            entry,
            instance,
            physical_device,
            device,
            enabled_extensions,
            queue_families,
            memory_properties,
            debug_messenger,
            has_validation,
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        builder: &VideoDeviceBuilder,
    ) -> VideoResult<(ash::Instance, bool)> {
        let app_name = CString::new(&builder.app_info.name[..]).map_err(|e| {
            VideoError::VulkanInit(format!("Failed to create CString for app name: {}", e))
        })?;
        let engine_name = CString::new(&builder.app_info.engine_name[..]).map_err(|e| {
            VideoError::VulkanInit(format!("Failed to create CString for engine name: {}", e))
        })?;

        let api_version = std::cmp::min(builder.app_info.api_version, vk::API_VERSION_1_2);

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(api_version);

        // Check available instance layers
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.map_err(|e| {
                VideoError::VulkanInit(format!("Failed to enumerate instance layers: {}", e))
            })?;

        let available_layer_names: Vec<String> = available_layers
            .iter()
            .map(|layer| {
                let name_bytes: Vec<u8> = layer
                    .layer_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as u8)
                    .collect();
                String::from_utf8_lossy(&name_bytes).into_owned()
            })
            .collect();

        // Check available instance extensions
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|e| {
            VideoError::VulkanInit(format!("Failed to enumerate instance extensions: {}", e))
        })?;

        let available_ext_names: Vec<String> = available_extensions
            .iter()
            .map(|ext| {
                let name_bytes: Vec<u8> = ext
                    .extension_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as u8)
                    .collect();
                String::from_utf8_lossy(&name_bytes).into_owned()
            })
            .collect();

        let mut instance_extensions: Vec<CString> = vec![
            CString::new("VK_KHR_surface").unwrap(),
            CString::new("VK_KHR_get_physical_device_properties2").unwrap(),
        ];
        let mut layers: Vec<CString> = Vec::new();
        let mut has_validation = false;
        if builder.enable_validation {
            // Check if validation layer is available
            let validation_layer = "VK_LAYER_KHRONOS_validation";
            if available_layer_names.contains(&validation_layer.to_string()) {
                layers.push(CString::new(validation_layer).unwrap());
                has_validation = true;
            } else {
                eprintln!(
                    "[VideoDeviceBuilder] WARNING: Validation layer {} not available",
                    validation_layer
                );
            }

            // Check if debug utils extension is available
            let debug_ext = "VK_EXT_debug_utils";
            if available_ext_names.contains(&debug_ext.to_string()) {
                instance_extensions.push(CString::new(debug_ext).unwrap());
            } else {
                eprintln!(
                    "[VideoDeviceBuilder] WARNING: Debug extension {} not available",
                    debug_ext
                );
                has_validation = false;
            }
        }
        let layer_ptrs: Vec<*const std::os::raw::c_char> =
            layers.iter().map(|c| c.as_ptr()).collect();
        let ext_ptrs: Vec<*const std::os::raw::c_char> =
            instance_extensions.iter().map(|c| c.as_ptr()).collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_ptrs)
            .enabled_layer_names(&layer_ptrs);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| VideoError::VulkanInit(e.to_string()))?;

        Ok((instance, has_validation))
    }

    fn create_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> VideoResult<vk::DebugUtilsMessengerEXT> {
        unsafe extern "system" fn debug_callback(
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
            _user_data: *mut std::os::raw::c_void,
        ) -> u32 {
            if p_callback_data.is_null() {
                return 0;
            }
            let data = *p_callback_data;
            let message = if !data.p_message.is_null() {
                std::ffi::CStr::from_ptr(data.p_message)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };

            // Format severity level
            let severity = if message_severity
                .contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            {
                "ERROR"
            } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
                "WARN"
            } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
                "INFO"
            } else {
                "VERBOSE"
            };

            // Format message type
            let msg_type = if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
            {
                "PERF"
            } else if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
                "VALID"
            } else {
                "GEN"
            };

            eprintln!(
                "[Vulkan Validation] [{}] [{}] {}",
                severity, msg_type, message
            );
            0 // VK_FALSE - don't abort
        }

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(std::ptr::null_mut());

        let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);
        unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .map_err(|e| {
                    VideoError::VulkanInit(format!("Failed to create debug messenger: {}", e))
                })
        }
    }

    fn select_physical_device(
        instance: &ash::Instance,
        _builder: &VideoDeviceBuilder,
    ) -> VideoResult<(vk::PhysicalDevice, QueueFamilies)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
            VideoError::VulkanInit(format!("Failed to enumerate physical devices: {}", e))
        })?;

        if physical_devices.is_empty() {
            return Err(VideoError::VulkanInit(
                "No physical devices found".to_string(),
            ));
        }

        // Find a physical device with video decode support
        for &pd in &physical_devices {
            let queue_families_list =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };

            // TEMP DIAGNOSTIC (iteration 5): which device, and does it have AV1?
            {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let name: String = props
                    .device_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|b| *b as u8 as char)
                    .collect();
                let exts =
                    unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
                let ext_names: Vec<String> = exts
                    .iter()
                    .map(|e| {
                        e.extension_name
                            .iter()
                            .take_while(|&&b| b != 0)
                            .map(|b| *b as u8 as char)
                            .collect()
                    })
                    .collect();
                let has_av1 = ext_names.iter().any(|n| n == "VK_KHR_video_decode_av1");
                let has_decode = ext_names.iter().any(|n| n == "VK_KHR_video_decode_queue");
                eprintln!(
                    "[DEV-DIAG] candidate: {} | decode_queue_ext={} av1={}",
                    name, has_decode, has_av1
                );
            }

            let mut decode_queue_family: Option<u32> = None;
            let mut graphics_queue_family: Option<u32> = None;
            let mut transfer_queue_family: Option<u32> = None;

            for (i, qf) in queue_families_list.iter().enumerate() {
                let i = i as u32;
                if qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                    && decode_queue_family.is_none()
                {
                    decode_queue_family = Some(i);
                }
                if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && graphics_queue_family.is_none()
                {
                    graphics_queue_family = Some(i);
                }
                if qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && transfer_queue_family.is_none()
                {
                    transfer_queue_family = Some(i);
                }
            }

            if let Some(decode_qf) = decode_queue_family {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let name: String = props
                    .device_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|b| *b as u8 as char)
                    .collect();
                eprintln!("[DEV-DIAG] SELECTED: {}", name);
                let queue_families = QueueFamilies {
                    graphics: graphics_queue_family,
                    compute: None,
                    transfer: transfer_queue_family.or(Some(decode_qf)),
                    video_decode: decode_queue_family,
                    video_encode: None,
                    present: None,
                };
                return Ok((pd, queue_families));
            }
        }

        Err(VideoError::VideoNotSupported(
            "No physical device with video decode queue found".to_string(),
        ))
    }

    fn create_device(
        _entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        queue_families: &QueueFamilies,
        builder: &VideoDeviceBuilder,
    ) -> VideoResult<(ash::Device, Vec<String>)> {
        // Query available device extensions
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(*physical_device) }.map_err(
                |e| VideoError::DeviceCreation(format!("Failed to enumerate extensions: {}", e)),
            )?;

        let available_names: Vec<String> = available_extensions
            .iter()
            .map(|ext| {
                let name_bytes: Vec<u8> = ext
                    .extension_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as u8)
                    .collect();
                String::from_utf8_lossy(&name_bytes).into_owned()
            })
            .collect();

        eprintln!("[VideoDeviceBuilder] Available video extensions:");
        for name in &available_names {
            if name.contains("video") {
                eprintln!("  - {}", name);
            }
        }

        // Collect device extensions (only those that are actually available)
        // NOTE: VK_KHR_video_maintenance1 is used for session auto-initialization.
        // VK_KHR_video_maintenance2 (optional) enables the videoMaintenance2 feature,
        // which makes NULL videoSessionParameters legal in vkCmdBeginVideoCodingKHR
        // for AV1 decode (VUID-VkVideoBeginCodingInfoKHR-videoSession-09261).
        let mut extensions: Vec<&str> = Vec::new();
        let required = [
            "VK_KHR_video_queue",
            "VK_KHR_video_decode_queue",
            "VK_KHR_video_maintenance1",
            "VK_KHR_sampler_ycbcr_conversion",
            "VK_KHR_synchronization2",
        ];
        for ext in &required {
            if available_names.iter().any(|n| n.as_str() == *ext) {
                extensions.push(ext);
            } else {
                eprintln!(
                    "[VideoDeviceBuilder] WARNING: Required extension {} not available",
                    ext
                );
            }
        }

        // VK_KHR_video_maintenance2 (optional but required for NULL session params on AV1)
        if available_names.iter().any(|n| n.as_str() == "VK_KHR_video_maintenance2") {
            extensions.push("VK_KHR_video_maintenance2");
        } else {
            eprintln!("[VideoDeviceBuilder] WARNING: VK_KHR_video_maintenance2 not available (NULL session params invalid for AV1 decode)");
        }

        // Add codec-specific extensions only if available
        if builder
            .video_codecs
            .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            && available_names
                .iter()
                .any(|n| n.as_str() == "VK_KHR_video_decode_h264")
            {
                extensions.push("VK_KHR_video_decode_h264");
            }
        if builder
            .video_codecs
            .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
            && available_names
                .iter()
                .any(|n| n.as_str() == "VK_KHR_video_decode_h265")
            {
                extensions.push("VK_KHR_video_decode_h265");
            }
        if builder
            .video_codecs
            .contains(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
        {
            if available_names
                .iter()
                .any(|n| n.as_str() == "VK_KHR_video_decode_av1")
            {
                extensions.push("VK_KHR_video_decode_av1");
            } else {
                eprintln!("[VideoDeviceBuilder] WARNING: VK_KHR_video_decode_av1 not available (AV1 decode not supported)");
            }
        }
        if builder
            .video_codecs
            .contains(vk::VideoCodecOperationFlagsKHR::from_raw(
                vp9_vk_constants::DECODE_VP9,
            ))
        {
            if available_names
                .iter()
                .any(|n| n.as_str() == "VK_KHR_video_decode_vp9")
            {
                extensions.push("VK_KHR_video_decode_vp9");
            } else {
                eprintln!("[VideoDeviceBuilder] WARNING: VK_KHR_video_decode_vp9 not available (VP9 decode not supported)");
            }
        }

        let c_extensions: Vec<CString> = extensions
            .iter()
            .map(|e| CString::new(*e).unwrap())
            .collect();
        let ext_ptrs: Vec<*const std::os::raw::c_char> =
            c_extensions.iter().map(|c| c.as_ptr()).collect();

        let mut sync2_features = vk::PhysicalDeviceSynchronization2FeaturesKHR::default();
        sync2_features.s_type = vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
        sync2_features.synchronization2 = 1;

        let mut video_decode_features = PhysicalDeviceVideoDecodeFeaturesKHR::default();
        video_decode_features.s_type = PHYSICAL_DEVICE_VIDEO_DECODE_FEATURES_KHR;
        video_decode_features.p_next = &mut sync2_features as *mut _ as *mut _;
        video_decode_features.video_decode_h264 = 1;
        video_decode_features.video_decode_h265 = 1;
        video_decode_features.video_decode_av1 = 1;
        video_decode_features.video_decode_vp9 = 1;

        let mut sampler_ycbcr_features =
            vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        sampler_ycbcr_features.s_type =
            vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
        sampler_ycbcr_features.p_next = &mut video_decode_features as *mut _ as *mut _;
        sampler_ycbcr_features.sampler_ycbcr_conversion = 1;

        let mut video_maintenance1_features = PhysicalDeviceVideoMaintenance1FeaturesKHR::default();
        video_maintenance1_features.s_type = PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR;
        video_maintenance1_features.p_next = &mut sampler_ycbcr_features as *mut _ as *mut _;
        video_maintenance1_features.video_maintenance1 = 1;

        let mut video_maintenance2_features = PhysicalDeviceVideoMaintenance2FeaturesKHR::default();
        video_maintenance2_features.s_type = PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR;
        video_maintenance2_features.p_next = &mut video_maintenance1_features as *mut _ as *mut _;
        video_maintenance2_features.video_maintenance2 = 1;

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        features2.s_type = vk::StructureType::PHYSICAL_DEVICE_FEATURES_2;
        features2.p_next = &mut video_maintenance2_features as *mut _ as *mut std::ffi::c_void;

        let queue_priorities = vec![1.0f32; builder.num_decode_queues];
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();

        if let Some(qf) = queue_families.video_decode {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qf)
                    .queue_priorities(&queue_priorities),
            );
        }

        if builder.create_graphics_queue {
            if let Some(qf) = queue_families.graphics {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(qf)
                        .queue_priorities(&[1.0f32]),
                );
            }
        }

        if builder.create_transfer_queue {
            if let Some(qf) = queue_families.transfer {
                // Only add if different from decode queue
                if queue_families.video_decode != Some(qf) {
                    queue_create_infos.push(
                        vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(qf)
                            .queue_priorities(&[1.0f32]),
                    );
                }
            }
        }

        #[allow(deprecated)]
        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: &features2 as *const _ as *const _,
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: ext_ptrs.len() as u32,
            pp_enabled_extension_names: ext_ptrs.as_ptr(),
            p_enabled_features: std::ptr::null(),
            _marker: Default::default(),
        };

        eprintln!("[VideoDeviceBuilder] Creating device...");
        let device = unsafe { instance.create_device(*physical_device, &device_create_info, None) }
            .map_err(|e| VideoError::DeviceCreation(e.to_string()))?;
        eprintln!("[VideoDeviceBuilder] Device created successfully");

        let ext_names: Vec<String> = c_extensions
            .iter()
            .map(|c| c.to_string_lossy().into_owned())
            .collect();

        Ok((device, ext_names))
    }
}

impl Default for VideoDeviceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanDevice {
    pub fn video_decode_queue(&self, index: u32) -> vk::Queue {
        let qf = self
            .queue_families
            .video_decode
            .expect("No video decode queue family");
        unsafe { self.device.get_device_queue(qf, index) }
    }

    pub fn video_decode_queue_family(&self) -> Option<u32> {
        self.queue_families.video_decode
    }

    /// Query video decode capabilities for a given codec profile.
    pub fn query_video_capabilities(
        &self,
        codec: VideoCodec,
        profile_idc: u32,
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    ) -> VideoResult<vk::VideoCapabilitiesKHR<'_>> {
        let codec_op = codec.to_vk_flag();

        // Build profile info chain - structs must live for entire function duration
        // to avoid dangling pointers when passed to Vulkan.
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
        let av1_profile = vk::VideoDecodeAV1ProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_AV1_PROFILE_INFO_KHR,
            p_next: std::ptr::null(),
            std_profile: profile_idc,
            film_grain_support: 0,
            _marker: Default::default(),
        };
        let vp9_profile = VideoDecodeVP9ProfileInfoKHR {
            s_type: vk::StructureType::from_raw(
                vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR,
            ),
            p_next: std::ptr::null(),
            std_profile: profile_idc,
            _marker: Default::default(),
        };

        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: match codec {
                VideoCodec::DecodeH264 => &h264_profile as *const _ as *const _,
                VideoCodec::DecodeH265 => &h265_profile as *const _ as *const _,
                VideoCodec::DecodeAv1 => &av1_profile as *const _ as *const _,
                VideoCodec::DecodeVp9 => &vp9_profile as *const _ as *const _,
            },
            video_codec_operation: codec_op,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            _marker: Default::default(),
        };
        let profile_ptr: *const vk::VideoProfileInfoKHR = &profile_info;

        // Get function pointer
        let get_caps_fn = unsafe {
            self.entry.get_instance_proc_addr(
                self.instance.handle(),
                c"vkGetPhysicalDeviceVideoCapabilitiesKHR".as_ptr(),
            )
        }
        .ok_or_else(|| {
            VideoError::CapabilityNotAvailable(
                "vkGetPhysicalDeviceVideoCapabilitiesKHR not found".to_string(),
            )
        })?;

        // Build output pNext chain per codec:
        // VideoCapabilitiesKHR -> <codec-specific> -> VideoDecodeCapabilitiesKHR
        let caps = unsafe {
            type FnType = unsafe extern "system" fn(
                vk::PhysicalDevice,
                *const vk::VideoProfileInfoKHR<'_>,
                *mut vk::VideoCapabilitiesKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(get_caps_fn);

            // Codec-specific capabilities structs
            let mut h264_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
            let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
            let mut av1_caps = vk::VideoDecodeAV1CapabilitiesKHR::default();
            let mut vp9_caps = VideoDecodeVP9CapabilitiesKHR::default();

            // Decode capabilities (intermediate)
            let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
            decode_caps.s_type = vk::StructureType::VIDEO_DECODE_CAPABILITIES_KHR;

            // Chain: decode_caps -> codec-specific
            // Vulkan spec: VkVideoCapabilitiesKHR -> VkVideoDecodeCapabilitiesKHR -> VkVideoDecode<Codec>CapabilitiesKHR
            match codec {
                VideoCodec::DecodeH264 => {
                    decode_caps.p_next = &mut h264_caps as *mut _ as *mut _;
                }
                VideoCodec::DecodeH265 => {
                    decode_caps.p_next = &mut h265_caps as *mut _ as *mut _;
                }
                VideoCodec::DecodeAv1 => {
                    decode_caps.p_next = &mut av1_caps as *mut _ as *mut _;
                }
                VideoCodec::DecodeVp9 => {
                    vp9_caps.s_type = vk::StructureType::from_raw(
                        vp9_vk_constants::VIDEO_DECODE_VP9_CAPABILITIES_KHR,
                    );
                    decode_caps.p_next = &mut vp9_caps as *mut _ as *mut _;
                }
            }

            // Top-level capabilities: caps -> decode_caps
            let mut caps = vk::VideoCapabilitiesKHR::default();
            caps.p_next = &mut decode_caps as *mut _ as *mut _;

            let result = fn_ptr(self.physical_device, profile_ptr, &mut caps);
            if result != vk::Result::SUCCESS {
                return Err(VideoError::CapabilityNotAvailable(format!(
                    "vkGetPhysicalDeviceVideoCapabilitiesKHR failed: {:?}",
                    result
                )));
            }
            caps
        };

        Ok(caps)
    }

    /// Query supported video formats for a codec.
    pub fn query_supported_formats(&self, codec: VideoCodec) -> Vec<vk::VideoFormatPropertiesKHR<'_>> {
        let codec_op = codec.to_vk_flag();

        // Common semi-planar 420 formats
        let candidate_formats = [
            vk::Format::G8_B8R8_2PLANE_420_UNORM,
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
            vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
        ];

        let mut formats = Vec::new();
        eprintln!(
            "[VideoDeviceBuilder] Querying supported video formats for codec {:?}",
            codec_op
        );
        for fmt in candidate_formats {
            eprintln!("  Trying format: {:?}", fmt);
            let format_props = self.get_physical_device_video_format_properties(
                codec_op,
                vk::ImageTiling::OPTIMAL,
                fmt,
                vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            );
            formats.extend(format_props);
        }

        formats
    }

    fn get_physical_device_video_format_properties(
        &self,
        video_operation: vk::VideoCodecOperationFlagsKHR,
        _image_tiling: vk::ImageTiling,
        _image_format: vk::Format,
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    ) -> Vec<vk::VideoFormatPropertiesKHR<'_>> {
        // Chain: PhysicalDeviceVideoFormatInfoKHR -> VideoProfileInfoKHR
        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: std::ptr::null(),
            video_codec_operation: video_operation,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            _marker: Default::default(),
        };

        let format_info = vk::PhysicalDeviceVideoFormatInfoKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
            p_next: &profile_info as *const _ as *const _,
            image_usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
            _marker: Default::default(),
        };

        let get_format_props_fn = unsafe {
            self.entry.get_instance_proc_addr(
                self.instance.handle(),
                c"vkGetPhysicalDeviceVideoFormatPropertiesKHR"
                    .as_ptr(),
            )
        };

        let Some(fn_ptr_raw) = get_format_props_fn else {
            return Vec::new();
        };

        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::PhysicalDevice,
                *const vk::PhysicalDeviceVideoFormatInfoKHR<'_>,
                *mut u32,
                *mut vk::VideoFormatPropertiesKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(fn_ptr_raw);

            let mut count: u32 = 0;
            let result = fn_ptr(
                self.physical_device,
                &format_info,
                &mut count,
                std::ptr::null_mut(),
            );
            if result != vk::Result::SUCCESS {
                return Vec::new();
            }

            let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
            let result = fn_ptr(
                self.physical_device,
                &format_info,
                &mut count,
                props.as_mut_ptr(),
            );
            if result != vk::Result::SUCCESS {
                return Vec::new();
            }

            props.truncate(count as usize);
            for p in &props {
                eprintln!(
                    "    Supported: format={:?}, usage={:?}",
                    p.format, p.image_usage_flags
                );
            }
            props
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Destroy debug messenger BEFORE instance (if not already destroyed)
        if self.has_validation && self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
            let debug_utils = ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
            unsafe {
                // Ignore errors - instance may already be destroyed
                debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.debug_messenger = vk::DebugUtilsMessengerEXT::null();
        }
    }
}
