//! Vulkan device initialization for video decode.

use ash::vk;
use std::ffi::CString;

use super::{Error, Result};

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
            Self::DecodeVp9 => vk::VideoCodecOperationFlagsKHR::from_raw(0x00000008),
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

    pub fn graphics_queue(&self, index: u32) -> Option<vk::Queue> {
        self.queue_families.graphics.map(|qf| {
            unsafe { self.device.get_device_queue(qf, index) }
        })
    }

    pub fn transfer_queue(&self, index: u32) -> vk::Queue {
        let qf = self
            .queue_families
            .transfer
            .unwrap_or(self.queue_families.video_decode.unwrap());
        unsafe { self.device.get_device_queue(qf, index) }
    }
}

/// Builder for VulkanDevice.
pub struct DeviceBuilder {
    enable_validation: bool,
    video_codecs: vk::VideoCodecOperationFlagsKHR,
    num_decode_queues: usize,
    create_graphics_queue: bool,
    create_transfer_queue: bool,
}

impl DeviceBuilder {
    pub fn new() -> Self {
        Self {
            enable_validation: false,
            video_codecs: vk::VideoCodecOperationFlagsKHR::DECODE_H264
                | vk::VideoCodecOperationFlagsKHR::DECODE_H265
                | vk::VideoCodecOperationFlagsKHR::DECODE_AV1
                | vk::VideoCodecOperationFlagsKHR::from_raw(0x00000008),
            num_decode_queues: 1,
            create_graphics_queue: false,
            create_transfer_queue: false,
        }
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

    pub fn with_transfer_queue(mut self, enable: bool) -> Self {
        self.create_transfer_queue = enable;
        self
    }

    pub fn build(self) -> Result<VulkanDevice> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Init(e.to_string()))?;
        let instance = Self::create_instance(&entry, &self)?;
        let (physical_device, queue_families) =
            Self::select_physical_device(&instance, &self)?;
        let (device, enabled_extensions) = Self::create_device(
            &entry, &instance, &physical_device, &queue_families, &self,
        )?;
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        Ok(VulkanDevice {
            entry,
            instance,
            physical_device,
            device,
            enabled_extensions,
            queue_families,
            memory_properties,
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        builder: &DeviceBuilder,
    ) -> Result<ash::Instance> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2);

        let mut instance_extensions: Vec<CString> = vec![
            CString::new("VK_KHR_surface").unwrap(),
            CString::new("VK_KHR_get_physical_device_properties2").unwrap(),
        ];

        let mut layers: Vec<CString> = Vec::new();
        if builder.enable_validation {
            layers.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
            instance_extensions.push(CString::new("VK_EXT_debug_utils").unwrap());
        }

        let layer_ptrs: Vec<*const std::os::raw::c_char> =
            layers.iter().map(|c| c.as_ptr()).collect();
        let ext_ptrs: Vec<*const std::os::raw::c_char> =
            instance_extensions.iter().map(|c| c.as_ptr()).collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_ptrs)
            .enabled_layer_names(&layer_ptrs);

        unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| Error::Init(e.to_string()))
    }

    fn select_physical_device(
        instance: &ash::Instance,
        _builder: &DeviceBuilder,
    ) -> Result<(vk::PhysicalDevice, QueueFamilies)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| Error::Init(e.to_string()))?;

        if physical_devices.is_empty() {
            return Err(Error::NoSuitableDevice);
        }

        for &pd in &physical_devices {
            let queue_families_list =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };

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

        Err(Error::NoSuitableDevice)
    }

    fn create_device(
        _entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        queue_families: &QueueFamilies,
        builder: &DeviceBuilder,
    ) -> Result<(ash::Device, Vec<String>)> {
        let available_extensions = unsafe {
            instance.enumerate_device_extension_properties(*physical_device)
        }.map_err(|e| Error::Device(e.to_string()))?;

        let available_names: Vec<String> = available_extensions
            .iter()
            .map(|ext| {
                let name_bytes: Vec<u8> = ext.extension_name.iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as u8)
                    .collect();
                String::from_utf8_lossy(&name_bytes).into_owned()
            })
            .collect();

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
            }
        }

        if builder.video_codecs.contains(vk::VideoCodecOperationFlagsKHR::DECODE_H264) {
            if available_names.iter().any(|n| n.as_str() == "VK_KHR_video_decode_h264") {
                extensions.push("VK_KHR_video_decode_h264");
            }
        }
        if builder.video_codecs.contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265) {
            if available_names.iter().any(|n| n.as_str() == "VK_KHR_video_decode_h265") {
                extensions.push("VK_KHR_video_decode_h265");
            }
        }
        if builder.video_codecs.contains(vk::VideoCodecOperationFlagsKHR::DECODE_AV1) {
            if available_names.iter().any(|n| n.as_str() == "VK_KHR_video_decode_av1") {
                extensions.push("VK_KHR_video_decode_av1");
            }
        }
        if builder.video_codecs.contains(vk::VideoCodecOperationFlagsKHR::from_raw(0x00000008)) {
            if available_names.iter().any(|n| n.as_str() == "VK_KHR_video_decode_vp9") {
                extensions.push("VK_KHR_video_decode_vp9");
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

        let mut sampler_ycbcr_features = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        sampler_ycbcr_features.s_type = vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
        sampler_ycbcr_features.p_next = &mut sync2_features as *mut _ as *mut _;
        sampler_ycbcr_features.sampler_ycbcr_conversion = 1;

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        features2.s_type = vk::StructureType::PHYSICAL_DEVICE_FEATURES_2;
        features2.p_next = &mut sampler_ycbcr_features as *mut _ as *mut std::ffi::c_void;

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
                if queue_families.video_decode != Some(qf) {
                    queue_create_infos.push(
                        vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(qf)
                            .queue_priorities(&[1.0f32]),
                    );
                }
            }
        }

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

        let device = unsafe {
            instance
                .create_device(*physical_device, &device_create_info, None)
        }
        .map_err(|e| Error::Device(e.to_string()))?;

        let ext_names: Vec<String> = c_extensions
            .iter()
            .map(|c| c.to_string_lossy().into_owned())
            .collect();

        Ok((device, ext_names))
    }
}

impl Default for DeviceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
