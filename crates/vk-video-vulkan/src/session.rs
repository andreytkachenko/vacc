//! Video session management with proper profile chaining.

use ash::vk;
use ash::vk::Handle;

use super::{device::VideoCodec, VideoError, VideoResult};
use super::vp9::{VideoDecodeVP9ProfileInfoKHR, VideoDecodeVP9SessionParametersCreateInfoKHR, vp9_vk_constants};

/// Codec-specific profile information for session creation.
#[derive(Debug, Clone)]
pub enum CodecProfileInfo {
    H264 {
        std_profile_idc: u32,
        picture_layout: u32,
    },
    H265 {
        std_profile_idc: u32,
    },
    Av1 {
        std_profile: u32,
        film_grain_support: bool,
    },
    Vp9 {
        std_profile: u32,
    },
}

/// Video session parameters for creating a video session.
#[derive(Debug, Clone)]
pub struct VideoSessionParams {
    pub queue_family_index: u32,
    pub picture_format: vk::Format,
    pub reference_picture_format: vk::Format,
    pub max_coded_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub codec: VideoCodec,
    pub codec_profile_info: CodecProfileInfo,
}

/// A Vulkan video session.
pub struct VideoSession {
    session: vk::VideoSessionKHR,
    device: ash::Device,
    instance: ash::Instance,
}

impl VideoSession {
    /// Bind memory to a video session (required after session creation).
    ///
    /// Calls vkGetVideoSessionMemoryRequirementsKHR + vkBindVideoSessionMemoryKHR.
    /// Returns allocated memory handles that must be freed before device destruction.
    fn bind_session_memory(
        instance: &ash::Instance,
        device: &ash::Device,
        session: vk::VideoSessionKHR,
    ) -> VideoResult<Vec<vk::DeviceMemory>> {
        let get_req_fn = unsafe {
            instance.get_device_proc_addr(
                device.handle(),
                b"vkGetVideoSessionMemoryRequirementsKHR\0".as_ptr().cast(),
            )
        }
        .ok_or_else(|| {
            VideoError::SessionCreation(
                "vkGetVideoSessionMemoryRequirementsKHR not found".to_string(),
            )
        })?;

        let mut req_count: u32 = 0;
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                *mut u32,
                *mut vk::VideoSessionMemoryRequirementsKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(get_req_fn);
            let result = fn_ptr(device.handle(), session, &mut req_count, std::ptr::null_mut());
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkGetVideoSessionMemoryRequirementsKHR (count) failed: {:?}",
                    result
                )));
            }
        }

        if req_count == 0 {
            return Ok(Vec::new());
        }

        let mut requirements = vec![vk::VideoSessionMemoryRequirementsKHR::default(); req_count as usize];
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
            let result = fn_ptr(device.handle(), session, &mut req_count, requirements.as_mut_ptr());
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkGetVideoSessionMemoryRequirementsKHR failed: {:?}",
                    result
                )));
            }
        }

        let mut bind_infos = Vec::with_capacity(req_count as usize);
        let mut memories = Vec::with_capacity(req_count as usize);

        for (i, req) in requirements.iter().enumerate() {
            let mem_req = req.memory_requirements;
            if mem_req.memory_type_bits == 0 {
                return Err(VideoError::SessionCreation(
                    "Session memory requirement has no valid memory types".to_string(),
                ));
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
                device.allocate_memory(&alloc_info, None)
                    .map_err(|e| VideoError::SessionCreation(e.to_string()))?
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
        .ok_or_else(|| {
            VideoError::SessionCreation("vkBindVideoSessionMemoryKHR not found".to_string())
        })?;

        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                u32,
                *const vk::BindVideoSessionMemoryInfoKHR<'_>,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(bind_fn);
            let result = fn_ptr(device.handle(), session, bind_infos.len() as u32, bind_infos.as_ptr());
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkBindVideoSessionMemoryKHR failed: {:?}",
                    result
                )));
            }
        }

        Ok(memories)
    }

    /// Create a video session with proper profile chain.
    pub fn create(
        instance: &ash::Instance,
        device: &ash::Device,
        params: &VideoSessionParams,
        std_header_version: &vk::ExtensionProperties,
    ) -> VideoResult<(Self, Vec<vk::DeviceMemory>)> {
        let codec_op = params.codec.to_vk_flag();

        // Build profile info chain using raw structs
        let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default();
        let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default();
        let mut av1_profile = vk::VideoDecodeAV1ProfileInfoKHR::default();
        let mut vp9_profile = VideoDecodeVP9ProfileInfoKHR::default();

        let profile_next: *const std::ffi::c_void = match &params.codec_profile_info {
            CodecProfileInfo::H264 {
                std_profile_idc,
                picture_layout,
            } => {
                h264_profile.s_type = vk::StructureType::VIDEO_DECODE_H264_PROFILE_INFO_KHR;
                h264_profile.p_next = std::ptr::null();
                h264_profile.std_profile_idc = *std_profile_idc;
                h264_profile.picture_layout = vk::VideoDecodeH264PictureLayoutFlagsKHR::from_raw(*picture_layout);
                &h264_profile as *const _ as *const std::ffi::c_void
            }
            CodecProfileInfo::H265 { std_profile_idc } => {
                h265_profile.s_type = vk::StructureType::VIDEO_DECODE_H265_PROFILE_INFO_KHR;
                h265_profile.p_next = std::ptr::null();
                h265_profile.std_profile_idc = *std_profile_idc;
                &h265_profile as *const _ as *const std::ffi::c_void
            }
            CodecProfileInfo::Av1 {
                std_profile,
                film_grain_support,
            } => {
                av1_profile.s_type = vk::StructureType::VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
                av1_profile.p_next = std::ptr::null();
                av1_profile.std_profile = *std_profile;
                av1_profile.film_grain_support = if *film_grain_support { 1 } else { 0 };
                &av1_profile as *const _ as *const std::ffi::c_void
            }
            CodecProfileInfo::Vp9 { std_profile } => {
                vp9_profile.s_type = vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR);
                vp9_profile.p_next = std::ptr::null();
                vp9_profile.std_profile = *std_profile;
                &vp9_profile as *const _ as *const std::ffi::c_void
            }
        };

        // Build VkVideoProfileInfoKHR with p_next chain
        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: profile_next as *const _,
            video_codec_operation: codec_op,
            chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            _marker: Default::default(),
        };

        // Session create info
        let session_create_info = vk::VideoSessionCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            queue_family_index: params.queue_family_index,
            flags: vk::VideoSessionCreateFlagsKHR::empty(),
            p_video_profile: &profile_info as *const _,
            picture_format: params.picture_format,
            max_coded_extent: params.max_coded_extent,
            reference_picture_format: params.reference_picture_format,
            max_dpb_slots: params.max_dpb_slots,
            max_active_reference_pictures: params.max_active_reference_pictures,
            p_std_header_version: std_header_version as *const _,
            _marker: Default::default(),
        };

        // Get function pointer
        let create_fn = unsafe {
            instance.get_device_proc_addr(
                device.handle(),
                b"vkCreateVideoSessionKHR\0".as_ptr().cast(),
            )
        }
        .ok_or_else(|| {
            VideoError::SessionCreation("vkCreateVideoSessionKHR not found".to_string())
        })?;

        let session = unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                *const vk::VideoSessionCreateInfoKHR,
                *const vk::AllocationCallbacks,
                *mut vk::VideoSessionKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(create_fn);

            let mut session_handle = vk::VideoSessionKHR::null();
            let result = fn_ptr(
                device.handle(),
                &session_create_info,
                std::ptr::null(),
                &mut session_handle,
            );
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkCreateVideoSessionKHR failed: {:?}",
                    result
                )));
            }
            session_handle
        };

        // Bind session memory (required after session creation)
        let session_memories = Self::bind_session_memory(instance, device, session)?;

        Ok((Self {
            session,
            device: device.clone(),
            instance: instance.clone(),
        }, session_memories))
    }

    pub fn handle(&self) -> vk::VideoSessionKHR {
        self.session
    }

    pub fn is_valid(&self) -> bool {
        !self.session.is_null()
    }
}

impl Drop for VideoSession {
    fn drop(&mut self) {
        if self.session.is_null() {
            return;
        }

        let destroy_fn = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkDestroyVideoSessionKHR\0".as_ptr().cast(),
                )
        };

        if let Some(ptr) = destroy_fn {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::Device, vk::VideoSessionKHR, *const vk::AllocationCallbacks);
                let fn_ptr: FnType = std::mem::transmute(ptr);
                fn_ptr(
                    self.device.handle(),
                    self.session,
                    std::ptr::null(),
                );
            }
        }
    }
}

/// Video session parameters object.
pub struct VideoSessionParameters {
    parameters: vk::VideoSessionParametersKHR,
    device: ash::Device,
    instance: ash::Instance,
}

impl std::fmt::Debug for VideoSessionParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoSessionParameters")
            .field("is_valid", &self.is_valid())
            .finish()
    }
}

impl VideoSessionParameters {
    /// Create session parameters with codec-specific info and initial SPS/PPS/VPS.
    pub fn create(
        instance: &ash::Instance,
        device: &ash::Device,
        session: vk::VideoSessionKHR,
        codec: VideoCodec,
        sps: Option<&vk_video_core::picture::H264Sps>,
        pps: Option<&vk_video_core::picture::H264Pps>,
        vps: Option<&vk_video_core::picture::H265Vps>,
        sps_h265: Option<&vk_video_core::picture::H265Sps>,
        pps_h265: Option<&vk_video_core::picture::H265Pps>,
    ) -> VideoResult<Self> {
        use super::codec_types::*;

        let params_create_info = match codec {
            VideoCodec::DecodeH264 => {
                let std_sps: Option<StdVideoH264SequenceParameterSet> =
                    sps.map(super::h264::convert_h264_sps);
                let std_pps: Option<StdVideoH264PictureParameterSet> =
                    pps.map(super::h264::convert_h264_pps);

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
                    p_parameters_add_info: &add_info as *const _ as *const _,
                    _marker: Default::default(),
                };
                vk::VideoSessionParametersCreateInfoKHR {
                    s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                    p_next: &h264_params as *const _ as *const _,
                    flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                    video_session: session,
                    video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                    _marker: Default::default(),
                }
            }
            VideoCodec::DecodeH265 => {
                let std_vps: Option<StdVideoH265VideoParameterSet> =
                    vps.map(super::h265::convert_h265_vps);
                let std_sps: Option<StdVideoH265SequenceParameterSet> =
                    sps_h265.map(super::h265::convert_h265_sps);
                let std_pps: Option<StdVideoH265PictureParameterSet> =
                    pps_h265.map(super::h265::convert_h265_pps);

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
                    p_parameters_add_info: &add_info as *const _ as *const _,
                    _marker: Default::default(),
                };
                vk::VideoSessionParametersCreateInfoKHR {
                    s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                    p_next: &h265_params as *const _ as *const _,
                    flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                    video_session: session,
                    video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                    _marker: Default::default(),
                }
            }
            VideoCodec::DecodeAv1 => {
                let av1_params = vk::VideoDecodeAV1SessionParametersCreateInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_sequence_header: std::ptr::null(),
                    _marker: Default::default(),
                };
                vk::VideoSessionParametersCreateInfoKHR {
                    s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                    p_next: &av1_params as *const _ as *const _,
                    flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                    video_session: session,
                    video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                    _marker: Default::default(),
                }
            }
            VideoCodec::DecodeVp9 => {
                // VP9 doesn't need codec-specific session parameters create info
                // (unlike H264/H265 which need SPS/PPS counts, or AV1 which needs SPS)
                vk::VideoSessionParametersCreateInfoKHR {
                    s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                    p_next: std::ptr::null(),
                    flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
                    video_session: session,
                    video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
                    _marker: Default::default(),
                }
            }
        };

        let create_fn = unsafe {
            instance.get_device_proc_addr(
                device.handle(),
                b"vkCreateVideoSessionParametersKHR\0".as_ptr().cast(),
            )
        }
        .ok_or_else(|| {
            VideoError::SessionCreation(
                "vkCreateVideoSessionParametersKHR not found".to_string(),
            )
        })?;

        let parameters = unsafe {
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
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkCreateVideoSessionParametersKHR failed: {:?}",
                    result
                )));
            }
            params
        };

        Ok(Self {
            parameters,
            device: device.clone(),
            instance: instance.clone(),
        })
    }

    pub fn handle(&self) -> vk::VideoSessionParametersKHR {
        self.parameters
    }

    pub fn is_valid(&self) -> bool {
        !self.parameters.is_null()
    }

    /// Initialize the video session with the given session parameters.
    ///
    /// Calls vkUpdateVideoSessionKHR/vkUpdateVideoSession if available.
    /// Falls back to vkUpdateVideoSessionParametersKHR with empty update entries.
    /// This is required to transition the session from uninitialized to initialized state.
    pub fn update_session(
        &self,
        session: vk::VideoSessionKHR,
    ) -> VideoResult<()> {
        // Try vkUpdateVideoSessionKHR first (VK_KHR_video_maintenance2)
        let update_fn = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkUpdateVideoSessionKHR\0".as_ptr().cast(),
            )
        };

        // Fall back to core vkUpdateVideoSession (Vulkan 1.4+)
        let update_fn = update_fn.or_else(|| unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkUpdateVideoSession\0".as_ptr().cast(),
            )
        });

        if let Some(fn_ptr) = update_fn {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::Device,
                    *const VkVideoSessionUpdateInfoKHR,
                );
                let fn_ptr: FnType = std::mem::transmute(fn_ptr);

                // VIDEO_SESSION_UPDATE_INFO_KHR = 1000348000 (VK_KHR_video_maintenance2)
                const VIDEO_SESSION_UPDATE_INFO_KHR: u32 = 1000348000;

                let update_info = VkVideoSessionUpdateInfoKHR {
                    s_type: vk::StructureType::from_raw(VIDEO_SESSION_UPDATE_INFO_KHR as i32),
                    p_next: std::ptr::null(),
                    video_session: session,
                    video_session_parameters: self.parameters,
                };

                fn_ptr(self.device.handle(), &update_info);
            }
            eprintln!("[session] vkUpdateVideoSessionKHR called successfully");
            return Ok(());
        }

        // vkUpdateVideoSessionKHR not available - vkCmdBeginVideoCodingKHR will initialize
        // the session with the session parameters when first called.
        eprintln!("[session] vkUpdateVideoSessionKHR not available, relying on vkCmdBeginVideoCodingKHR for initialization");

        Ok(())
    }
}

/// VkVideoSessionUpdateInfoKHR structure for vkUpdateVideoSessionKHR.
/// Defined manually since ash doesn't expose it.
#[repr(C)]
struct VkVideoSessionUpdateInfoKHR {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    video_session: vk::VideoSessionKHR,
    video_session_parameters: vk::VideoSessionParametersKHR,
}

impl Drop for VideoSessionParameters {
    fn drop(&mut self) {
        if self.parameters.is_null() {
            return;
        }

        let destroy_fn = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkDestroyVideoSessionParametersKHR\0".as_ptr().cast(),
                )
        };

        if let Some(ptr) = destroy_fn {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::Device,
                    vk::VideoSessionParametersKHR,
                    *const vk::AllocationCallbacks,
                );
                let fn_ptr: FnType = std::mem::transmute(ptr);
                fn_ptr(
                    self.device.handle(),
                    self.parameters,
                    std::ptr::null(),
                );
            }
        }
    }
}
