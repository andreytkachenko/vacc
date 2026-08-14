//! Video session management with proper profile chaining.

use ash::vk;
use ash::vk::Handle;

use super::vp9::{
    vp9_vk_constants, VideoDecodeVP9ProfileInfoKHR, VideoDecodeVP9SessionParametersCreateInfoKHR,
};
use super::{device::VideoCodec, VideoError, VideoResult};

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
    pub chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    pub luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    pub chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
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
            let result = fn_ptr(
                device.handle(),
                session,
                &mut req_count,
                std::ptr::null_mut(),
            );
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
                device
                    .allocate_memory(&alloc_info, None)
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
            let result = fn_ptr(
                device.handle(),
                session,
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            );
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
                h264_profile.picture_layout =
                    vk::VideoDecodeH264PictureLayoutFlagsKHR::from_raw(*picture_layout);
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
                vp9_profile.s_type = vk::StructureType::from_raw(
                    vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR,
                );
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
            chroma_subsampling: params.chroma_subsampling,
            luma_bit_depth: params.luma_bit_depth,
            chroma_bit_depth: params.chroma_bit_depth,
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

        Ok((
            Self {
                session,
                device: device.clone(),
                instance: instance.clone(),
            },
            session_memories,
        ))
    }

    pub fn handle(&self) -> vk::VideoSessionKHR {
        self.session
    }

    pub fn is_valid(&self) -> bool {
        !self.session.is_null()
    }

    /// Mark this session as destroyed (null out the handle).
    /// Call after manually destroying via vkDestroyVideoSessionKHR
    /// to prevent double-free in Drop.
    pub fn reset(&mut self) {
        self.session = vk::VideoSessionKHR::null();
    }
}

impl Drop for VideoSession {
    fn drop(&mut self) {
        if self.session.is_null() {
            return;
        }

        let destroy_fn = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkDestroyVideoSessionKHR\0".as_ptr().cast(),
            )
        };

        if let Some(ptr) = destroy_fn {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::Device,
                    vk::VideoSessionKHR,
                    *const vk::AllocationCallbacks,
                );
                let fn_ptr: FnType = std::mem::transmute(ptr);
                fn_ptr(self.device.handle(), self.session, std::ptr::null());
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

        // All structs must live for the entire function duration to avoid dangling pointers
        // when passed to Vulkan. Declare them here, initialize based on codec below.
        let std_sps_h264: Option<StdVideoH264SequenceParameterSet> =
            sps.map(super::h264::convert_h264_sps);
        let std_pps_h264: Option<StdVideoH264PictureParameterSet> =
            pps.map(super::h264::convert_h264_pps);
        let std_vps_h265: Option<StdVideoH265VideoParameterSet> =
            vps.map(super::h265::convert_h265_vps);
        let std_sps_h265: Option<StdVideoH265SequenceParameterSet> =
            sps_h265.map(super::h265::convert_h265_sps);
        let std_pps_h265: Option<StdVideoH265PictureParameterSet> =
            pps_h265.map(super::h265::convert_h265_pps);

        let mut h264_add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR::default();
        let mut h264_params = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default();
        let mut h265_add_info = vk::VideoDecodeH265SessionParametersAddInfoKHR::default();
        let mut h265_params = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default();
        let mut av1_params = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default();

        // Initialize codec-specific structs
        match codec {
            VideoCodec::DecodeH264 => {
                h264_add_info.s_type =
                    vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
                h264_add_info.p_next = std::ptr::null();
                h264_add_info.std_sps_count = std_sps_h264.is_some() as u32;
                h264_add_info.p_std_sp_ss = std_sps_h264
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s as *const _);
                h264_add_info.std_pps_count = std_pps_h264.is_some() as u32;
                h264_add_info.p_std_pp_ss = std_pps_h264
                    .as_ref()
                    .map_or(std::ptr::null(), |p| p as *const _);

                h264_params.s_type =
                    vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR;
                h264_params.p_next = std::ptr::null();
                h264_params.max_std_sps_count = 32;
                h264_params.max_std_pps_count = 256;
                h264_params.p_parameters_add_info = &h264_add_info as *const _ as *const _;
            }
            VideoCodec::DecodeH265 => {
                h265_add_info.s_type =
                    vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
                h265_add_info.p_next = std::ptr::null();
                h265_add_info.std_vps_count = std_vps_h265.is_some() as u32;
                h265_add_info.p_std_vp_ss = std_vps_h265
                    .as_ref()
                    .map_or(std::ptr::null(), |v| v as *const _);
                h265_add_info.std_sps_count = std_sps_h265.is_some() as u32;
                h265_add_info.p_std_sp_ss = std_sps_h265
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s as *const _);
                h265_add_info.std_pps_count = std_pps_h265.is_some() as u32;
                h265_add_info.p_std_pp_ss = std_pps_h265
                    .as_ref()
                    .map_or(std::ptr::null(), |p| p as *const _);

                h265_params.s_type =
                    vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR;
                h265_params.p_next = std::ptr::null();
                h265_params.max_std_vps_count = 16;
                h265_params.max_std_sps_count = 32;
                h265_params.max_std_pps_count = 256;
                h265_params.p_parameters_add_info = &h265_add_info as *const _ as *const _;
            }
            VideoCodec::DecodeAv1 => {
                av1_params.s_type =
                    vk::StructureType::VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR;
                av1_params.p_next = std::ptr::null();
                av1_params.p_std_sequence_header = std::ptr::null();
            }
            VideoCodec::DecodeVp9 => {
                // VP9 doesn't need codec-specific session parameters create info
            }
        }

        let params_create_info = vk::VideoSessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: match codec {
                VideoCodec::DecodeH264 => &h264_params as *const _ as *const _,
                VideoCodec::DecodeH265 => &h265_params as *const _ as *const _,
                VideoCodec::DecodeAv1 => &av1_params as *const _ as *const _,
                VideoCodec::DecodeVp9 => std::ptr::null(),
            },
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
        .ok_or_else(|| {
            VideoError::SessionCreation("vkCreateVideoSessionParametersKHR not found".to_string())
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

    /// Mark these parameters as destroyed (null out the handle).
    /// Call after manually destroying via vkDestroyVideoSessionParametersKHR
    /// to prevent double-free in Drop.
    pub fn reset(&mut self) {
        self.parameters = vk::VideoSessionParametersKHR::null();
    }

    /// Initialize the video session with the given session parameters.
    ///
    /// With VK_KHR_video_maintenance1 (not maintenance2), vkCmdBeginVideoCodingKHR
    /// will automatically initialize the session when first called with session parameters.
    /// This matches the NVIDIA Vulkan-Video-Samples behavior.
    pub fn update_session(&self, _session: vk::VideoSessionKHR) -> VideoResult<()> {
        // With VK_KHR_video_maintenance1, the session is auto-initialized by
        // vkCmdBeginVideoCodingKHR when called with session parameters.
        // No explicit vkUpdateVideoSessionKHR call needed.
        Ok(())
    }
}

impl Drop for VideoSessionParameters {
    fn drop(&mut self) {
        if self.parameters.is_null() {
            return;
        }

        let destroy_fn = unsafe {
            self.instance.get_device_proc_addr(
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
                fn_ptr(self.device.handle(), self.parameters, std::ptr::null());
            }
        }
    }
}
