//! Video session management with proper profile chaining.

use ash::vk;
use ash::vk::Handle;

use super::vp9::{vp9_vk_constants, VideoDecodeVP9ProfileInfoKHR};
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
                c"vkGetVideoSessionMemoryRequirementsKHR".as_ptr(),
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
            instance.get_device_proc_addr(device.handle(), c"vkBindVideoSessionMemoryKHR".as_ptr())
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

        // Session create flags.
        //
        // VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR (0x4, VK_KHR_video_maintenance1),
        // matching the C++ reference (VkVideoDecoder.cpp:282-289, set whenever
        // maintenance1 is supported). Both known-working NVIDIA H265 decoders use a
        // non-zero session flag (FFmpeg: INLINE_SESSION_PARAMETERS 0x20; C++ ref:
        // INLINE_QUERIES 0x4); with empty() flags the NVIDIA driver traps in
        // vkCmdDecodeVideoKHR on frame 0. ash 0.38 lacks the constant, so use raw bits.
        let session_flags = vk::VideoSessionCreateFlagsKHR::from_raw(0x4);

        let session_create_info = vk::VideoSessionCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            queue_family_index: params.queue_family_index,
            flags: session_flags,
            p_video_profile: &profile_info as *const _,
            picture_format: params.picture_format,
            max_coded_extent: params.max_coded_extent,
            reference_picture_format: params.reference_picture_format,
            max_dpb_slots: params.max_dpb_slots,
            max_active_reference_pictures: params.max_active_reference_pictures,
            p_std_header_version: std_header_version as *const _,
            _marker: Default::default(),
        };

        // DEBUG (iteration 17): comprehensive session creation parameters
        eprintln!("[SESSION-CREATE] ===== VkVideoSessionCreateInfoKHR =====");
        eprintln!(
            "[SESSION-CREATE]   flags                          = {:?}",
            session_create_info.flags
        );
        eprintln!(
            "[SESSION-CREATE]   queueFamilyIndex               = {}",
            session_create_info.queue_family_index
        );
        eprintln!(
            "[SESSION-CREATE]   pictureFormat                  = {:?}",
            session_create_info.picture_format
        );
        eprintln!(
            "[SESSION-CREATE]   referencePictureFormat         = {:?}",
            session_create_info.reference_picture_format
        );
        eprintln!(
            "[SESSION-CREATE]   maxCodedExtent                 = {}x{}",
            session_create_info.max_coded_extent.width, session_create_info.max_coded_extent.height
        );
        eprintln!(
            "[SESSION-CREATE]   maxDpbSlots                    = {}",
            session_create_info.max_dpb_slots
        );
        eprintln!(
            "[SESSION-CREATE]   maxActiveReferencePictures     = {}",
            session_create_info.max_active_reference_pictures
        );
        eprintln!(
            "[SESSION-CREATE]   VkVideoProfileInfoKHR: codecOp={:?} chromaSub={:?} lumaBit={:?} chromaBit={:?}",
            profile_info.video_codec_operation,
            profile_info.chroma_subsampling,
            profile_info.luma_bit_depth,
            profile_info.chroma_bit_depth,
        );
        // Print codec-specific profile info
        match &params.codec_profile_info {
            CodecProfileInfo::H264 {
                std_profile_idc,
                picture_layout,
            } => {
                eprintln!("[SESSION-CREATE]   VkVideoDecodeH264ProfileInfoKHR: stdProfileIdc={} pictureLayout={}", std_profile_idc, picture_layout);
            }
            CodecProfileInfo::H265 { std_profile_idc } => {
                eprintln!(
                    "[SESSION-CREATE]   VkVideoDecodeH265ProfileInfoKHR: stdProfileIdc={}",
                    std_profile_idc
                );
            }
            CodecProfileInfo::Av1 {
                std_profile,
                film_grain_support,
            } => {
                eprintln!("[SESSION-CREATE]   VkVideoDecodeAV1ProfileInfoKHR: stdProfile={} filmGrainSupport={}", std_profile, film_grain_support);
            }
            CodecProfileInfo::Vp9 { std_profile } => {
                eprintln!(
                    "[SESSION-CREATE]   VkVideoDecodeVP9ProfileInfoKHR: stdProfile={}",
                    std_profile
                );
            }
        }
        // Print std header version
        unsafe {
            eprintln!(
                "[SESSION-CREATE]   pStdHeaderVersion->extensionName = \"{}\"",
                std::ffi::CStr::from_ptr((*std_header_version).extension_name.as_ptr())
                    .to_string_lossy()
            );
            eprintln!(
                "[SESSION-CREATE]   pStdHeaderVersion->specVersion   = {} (0x{:08X})",
                std_header_version.spec_version, std_header_version.spec_version
            );
        }
        eprintln!("[SESSION-CREATE] ===========================================");

        // Get function pointer
        let create_fn = unsafe {
            instance.get_device_proc_addr(device.handle(), c"vkCreateVideoSessionKHR".as_ptr())
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

            eprintln!("[Session] Calling vkCreateVideoSessionKHR...");
            let mut session_handle = vk::VideoSessionKHR::null();
            let result = fn_ptr(
                device.handle(),
                &session_create_info,
                std::ptr::null(),
                &mut session_handle,
            );
            eprintln!("[Session] vkCreateVideoSessionKHR returned: {:?}", result);
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkCreateVideoSessionKHR failed: {:?}",
                    result
                )));
            }
            session_handle
        };

        // Bind session memory (required after session creation)
        eprintln!("[Session] Binding session memory...");
        let session_memories = Self::bind_session_memory(instance, device, session)?;
        eprintln!("[Session] Session memory bound");

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
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkDestroyVideoSessionKHR".as_ptr())
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
        sps_av1: Option<&vk_video_core::picture::Av1Sps>,
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
        // AV1: the sequence header (and the color config / timing info it
        // points to) must remain valid for the lifetime of the session
        // parameters object. The driver retains these pointers (it does not
        // copy the data), so we leak them to keep them alive past create().
        let std_color_config_av1: Option<*const StdVideoAV1ColorConfig> = sps_av1.map(|sps| {
            Box::into_raw(Box::new(super::av1::convert_av1_color_config(sps))) as *const _
        });
        let std_timing_info_av1: Option<*const StdVideoAV1TimingInfo> = sps_av1.map(|sps| {
            Box::into_raw(Box::new(super::av1::convert_av1_timing_info(sps))) as *const _
        });
        let std_sps_av1: Option<*const StdVideoAV1SequenceHeader> = sps_av1.map(|sps| {
            let mut header = super::av1::convert_av1_sps(sps);
            header.pColorConfig = std_color_config_av1.unwrap_or(std::ptr::null());
            header.pTimingInfo = std_timing_info_av1.unwrap_or(std::ptr::null());
            // === FULL SPS DIAGNOSTIC DUMP ===
            eprintln!("[SPS-DUMP] ===== StdVideoAV1SequenceHeader =====");
            eprintln!(
                "[SPS-DUMP] flags.still_picture                       = {}",
                header.flags.still_picture()
            );
            eprintln!(
                "[SPS-DUMP] flags.reduced_still_picture_header        = {}",
                header.flags.reduced_still_picture_header()
            );
            eprintln!(
                "[SPS-DUMP] flags.use_128x128_superblock              = {}",
                header.flags.use_128x128_superblock()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_filter_intra                 = {}",
                header.flags.enable_filter_intra()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_intra_edge_filter            = {}",
                header.flags.enable_intra_edge_filter()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_interintra_compound          = {}",
                header.flags.enable_interintra_compound()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_masked_compound              = {}",
                header.flags.enable_masked_compound()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_warped_motion                = {}",
                header.flags.enable_warped_motion()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_dual_filter                  = {}",
                header.flags.enable_dual_filter()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_order_hint                   = {}",
                header.flags.enable_order_hint()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_jnt_comp                     = {}",
                header.flags.enable_jnt_comp()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_ref_frame_mvs                = {}",
                header.flags.enable_ref_frame_mvs()
            );
            eprintln!(
                "[SPS-DUMP] flags.frame_id_numbers_present_flag       = {}",
                header.flags.frame_id_numbers_present_flag()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_superres                     = {}",
                header.flags.enable_superres()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_cdef                         = {}",
                header.flags.enable_cdef()
            );
            eprintln!(
                "[SPS-DUMP] flags.enable_restoration                  = {}",
                header.flags.enable_restoration()
            );
            eprintln!(
                "[SPS-DUMP] flags.film_grain_params_present           = {}",
                header.flags.film_grain_params_present()
            );
            eprintln!(
                "[SPS-DUMP] flags.timing_info_present_flag            = {}",
                header.flags.timing_info_present_flag()
            );
            eprintln!(
                "[SPS-DUMP] flags.initial_display_delay_present_flag  = {}",
                header.flags.initial_display_delay_present_flag()
            );
            eprintln!(
                "[SPS-DUMP] seq_profile                               = {}",
                header.seq_profile
            );
            eprintln!(
                "[SPS-DUMP] frame_width_bits_minus_1                  = {}",
                header.frame_width_bits_minus_1
            );
            eprintln!(
                "[SPS-DUMP] frame_height_bits_minus_1                 = {}",
                header.frame_height_bits_minus_1
            );
            eprintln!(
                "[SPS-DUMP] max_frame_width_minus_1                   = {}",
                header.max_frame_width_minus_1
            );
            eprintln!(
                "[SPS-DUMP] max_frame_height_minus_1                  = {}",
                header.max_frame_height_minus_1
            );
            eprintln!(
                "[SPS-DUMP] delta_frame_id_length_minus_2             = {}",
                header.delta_frame_id_length_minus_2
            );
            eprintln!(
                "[SPS-DUMP] additional_frame_id_length_minus_1        = {}",
                header.additional_frame_id_length_minus_1
            );
            eprintln!(
                "[SPS-DUMP] order_hint_bits_minus_1                   = {}",
                header.order_hint_bits_minus_1
            );
            eprintln!(
                "[SPS-DUMP] seq_force_integer_mv                      = {} (SELECT=2)",
                header.seq_force_integer_mv
            );
            eprintln!(
                "[SPS-DUMP] seq_force_screen_content_tools            = {} (SELECT=2)",
                header.seq_force_screen_content_tools
            );
            if !header.pColorConfig.is_null() {
                let cc = unsafe { &*header.pColorConfig };
                eprintln!("[SPS-DUMP] --- ColorConfig ---");
                eprintln!(
                    "[SPS-DUMP] cc.flags.mono_chrome                     = {}",
                    cc.flags.mono_chrome()
                );
                eprintln!(
                    "[SPS-DUMP] cc.flags.color_range                    = {}",
                    cc.flags.color_range()
                );
                eprintln!(
                    "[SPS-DUMP] cc.flags.separate_uv_delta_q            = {}",
                    cc.flags.separate_uv_delta_q()
                );
                eprintln!(
                    "[SPS-DUMP] cc.flags.color_description_present_flag = {}",
                    cc.flags.color_description_present_flag()
                );
                eprintln!(
                    "[SPS-DUMP] cc.BitDepth                             = {}",
                    cc.BitDepth
                );
                eprintln!(
                    "[SPS-DUMP] cc.subsampling_x                        = {}",
                    cc.subsampling_x
                );
                eprintln!(
                    "[SPS-DUMP] cc.subsampling_y                        = {}",
                    cc.subsampling_y
                );
                eprintln!(
                    "[SPS-DUMP] cc.color_primaries                      = {}",
                    cc.color_primaries
                );
                eprintln!(
                    "[SPS-DUMP] cc.transfer_characteristics             = {}",
                    cc.transfer_characteristics
                );
                eprintln!(
                    "[SPS-DUMP] cc.matrix_coefficients                  = {}",
                    cc.matrix_coefficients
                );
                eprintln!(
                    "[SPS-DUMP] cc.chroma_sample_position               = {}",
                    cc.chroma_sample_position
                );
            } else {
                eprintln!("[SPS-DUMP] --- ColorConfig: NULL ---");
            }
            if !header.pTimingInfo.is_null() {
                let ti = unsafe { &*header.pTimingInfo };
                eprintln!("[SPS-DUMP] --- TimingInfo ---");
                eprintln!(
                    "[SPS-DUMP] ti.flags.equal_picture_interval         = {}",
                    ti.flags.equal_picture_interval()
                );
                eprintln!(
                    "[SPS-DUMP] ti.num_units_in_display_tick            = {}",
                    ti.num_units_in_display_tick
                );
                eprintln!(
                    "[SPS-DUMP] ti.time_scale                           = {}",
                    ti.time_scale
                );
                eprintln!(
                    "[SPS-DUMP] ti.num_ticks_per_picture_minus_1        = {}",
                    ti.num_ticks_per_picture_minus_1
                );
            } else {
                eprintln!("[SPS-DUMP] --- TimingInfo: NULL ---");
            }
            eprintln!("[SPS-DUMP] ============================================");
            Box::into_raw(Box::new(header)) as *const _
        });

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
                // Match the C++ reference (Vulkan-Video-Samples) exactly: create the
                // session parameters with the VPS only, then add SPS/PPS afterwards
                // via vkUpdateVideoSessionParametersKHR (done below). Passing all
                // three at create time leaves the driver's SPS/PPS tables empty on
                // this NVIDIA driver -> NULL-deref SIGSEGV in vkCmdDecodeVideoKHR.
                h265_add_info.s_type =
                    vk::StructureType::VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
                h265_add_info.p_next = std::ptr::null();
                h265_add_info.std_vps_count = std_vps_h265.is_some() as u32;
                h265_add_info.p_std_vp_ss = std_vps_h265
                    .as_ref()
                    .map_or(std::ptr::null(), |v| v as *const _);
                h265_add_info.std_sps_count = 0;
                h265_add_info.p_std_sp_ss = std::ptr::null();
                h265_add_info.std_pps_count = 0;
                h265_add_info.p_std_pp_ss = std::ptr::null();

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
                av1_params.p_std_sequence_header = std_sps_av1.unwrap_or(std::ptr::null());
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

        // DEBUG (iteration 17): comprehensive session parameters creation dump
        eprintln!("[SESSION-PARAMS-CREATE] ===== VkVideoSessionParametersCreateInfoKHR =====");
        eprintln!(
            "[SESSION-PARAMS-CREATE]   flags                                  = {:?}",
            params_create_info.flags
        );
        eprintln!(
            "[SESSION-PARAMS-CREATE]   videoSession                           = {:?} (valid={})",
            params_create_info.video_session,
            !params_create_info.video_session.is_null()
        );
        eprintln!(
            "[SESSION-PARAMS-CREATE]   videoSessionParametersTemplate         = {:?} (valid={})",
            params_create_info.video_session_parameters_template,
            !params_create_info
                .video_session_parameters_template
                .is_null()
        );
        match codec {
            VideoCodec::DecodeAv1 => {
                eprintln!(
                    "[SESSION-PARAMS-CREATE]   p_next -> VkVideoDecodeAV1SessionParametersCreateInfoKHR"
                );
                eprintln!(
                    "[SESSION-PARAMS-CREATE]     p_std_sequence_header          = {:?} (valid={})",
                    av1_params.p_std_sequence_header,
                    !av1_params.p_std_sequence_header.is_null()
                );
                if !av1_params.p_std_sequence_header.is_null() {
                    unsafe {
                        let sps = &*av1_params.p_std_sequence_header;
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->seq_profile             = {}",
                            sps.seq_profile
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->frame_width_bits_minus_1 = {}",
                            sps.frame_width_bits_minus_1
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->frame_height_bits_minus_1= {}",
                            sps.frame_height_bits_minus_1
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->max_frame_width_minus_1  = {}",
                            sps.max_frame_width_minus_1
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->max_frame_height_minus_1 = {}",
                            sps.max_frame_height_minus_1
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->order_hint_bits_minus_1  = {}",
                            sps.order_hint_bits_minus_1
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->pColorConfig             = {:?}",
                            sps.pColorConfig
                        );
                        eprintln!(
                            "[SESSION-PARAMS-CREATE]       sps->pTimingInfo              = {:?}",
                            sps.pTimingInfo
                        );
                    }
                }
            }
            _ => {
                eprintln!(
                    "[SESSION-PARAMS-CREATE]   p_next -> codec-specific params for {:?}",
                    codec
                );
            }
        }
        eprintln!("[SESSION-PARAMS-CREATE] ===========================================");

        let create_fn = unsafe {
            instance.get_device_proc_addr(
                device.handle(),
                c"vkCreateVideoSessionParametersKHR".as_ptr(),
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

            eprintln!("[SessionParams] Calling vkCreateVideoSessionParametersKHR...");
            let mut params = vk::VideoSessionParametersKHR::null();
            let result = fn_ptr(
                device.handle(),
                &params_create_info,
                std::ptr::null(),
                &mut params,
            );
            eprintln!(
                "[SessionParams] vkCreateVideoSessionParametersKHR returned: {:?}",
                result
            );
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkCreateVideoSessionParametersKHR failed: {:?}",
                    result
                )));
            }
            params
        };

        // Add SPS/PPS via vkUpdateVideoSessionParametersKHR, matching the C++
        // reference sequence (create with VPS only, then update SPS, then PPS).
        // The NVIDIA driver does not process SPS/PPS supplied at create time.
        if matches!(codec, VideoCodec::DecodeH265) {
            let update_fn = unsafe {
                instance.get_device_proc_addr(
                    device.handle(),
                    c"vkUpdateVideoSessionParametersKHR".as_ptr(),
                )
            };
            match update_fn {
                Some(f) => {
                    type UpdFn = unsafe extern "system" fn(
                        vk::Device,
                        vk::VideoSessionParametersKHR,
                        *const vk::VideoSessionParametersUpdateInfoKHR<'_>,
                    ) -> vk::Result;
                    let upd: UpdFn = unsafe { std::mem::transmute(f) };
                    // Vulkan spec (VUID-...-07215): update_sequence_count must be the
                    // object's current counter + 1 on every update (starts at 0).
                    // A zero count made the NVIDIA driver drop the SPS/PPS updates.
                    let mut seq = 0u32;
                    if let Some(sps) = std_sps_h265 {
                        // The driver may retain this pointer (it does not copy the
                        // data), so leak it like the AV1 sequence header above.
                        let sps_ptr = Box::leak(Box::new(sps)) as *const StdVideoH265SequenceParameterSet;
                        let mut add_info =
                            vk::VideoDecodeH265SessionParametersAddInfoKHR::default();
                        add_info.std_sps_count = 1;
                        add_info.p_std_sp_ss = sps_ptr;
                        seq += 1;
                        let update_info = vk::VideoSessionParametersUpdateInfoKHR {
                            p_next: &add_info as *const _ as *const _,
                            update_sequence_count: seq,
                            ..Default::default()
                        };
                        let r = unsafe { upd(device.handle(), parameters, &update_info) };
                        eprintln!("[SessionParams] vkUpdateVideoSessionParametersKHR (SPS) -> {:?}", r);
                    }
                    if let Some(pps) = std_pps_h265 {
                        let pps_ptr = Box::leak(Box::new(pps)) as *const StdVideoH265PictureParameterSet;
                        let mut add_info =
                            vk::VideoDecodeH265SessionParametersAddInfoKHR::default();
                        add_info.std_pps_count = 1;
                        add_info.p_std_pp_ss = pps_ptr;
                        seq += 1;
                        let update_info = vk::VideoSessionParametersUpdateInfoKHR {
                            p_next: &add_info as *const _ as *const _,
                            update_sequence_count: seq,
                            ..Default::default()
                        };
                        let r = unsafe { upd(device.handle(), parameters, &update_info) };
                        eprintln!("[SessionParams] vkUpdateVideoSessionParametersKHR (PPS) -> {:?}", r);
                    }
                }
                None => eprintln!(
                    "[SessionParams] WARNING: vkUpdateVideoSessionParametersKHR not found; SPS/PPS will be missing from session parameters"
                ),
            }
        }

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
    /// Explicitly calls vkUpdateVideoSessionKHR so the session is initialized
    /// with the codec-specific parameters (e.g. the AV1 SPS) before the first
    /// decode. The function pointer is loaded via the device proc addr, with a
    /// fallback to the instance proc addr (some drivers do not expose
    /// core-promoted video commands via vkGetDeviceProcAddr on a 1.2 device).
    pub fn update_session(&self, session: vk::VideoSessionKHR) -> VideoResult<()> {
        // Try both the KHR-suffixed extension name and the core (non-KHR) name.
        // Some drivers only expose core-promoted video commands under one or the other.
        let update_fn = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkUpdateVideoSessionKHR".as_ptr())
                .or_else(|| {
                    self.instance.get_device_proc_addr(
                        self.device.handle(),
                        c"vkUpdateVideoSession".as_ptr(),
                    )
                })
        };
        let update_fn = match update_fn {
            Some(f) => f,
            None => {
                // vkUpdateVideoSessionKHR not loadable via vkGetDeviceProcAddr on this
                // driver (API 1.2 device). VK_KHR_video_maintenance1 is enabled, so the
                // session is auto-initialized by vkCmdBeginVideoCodingKHR with the session
                // parameters. Rely on that (matches the C++ reference, which never calls
                // vkUpdateVideoSessionKHR).
                eprintln!("[SessionParams] vkUpdateVideoSessionKHR not found (tried KHR + core names); relying on maintenance1 auto-init");
                return Ok(());
            }
        };

        let result = unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                vk::VideoSessionParametersKHR,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(update_fn);
            fn_ptr(self.device.handle(), session, self.parameters)
        };

        eprintln!(
            "[SessionParams] vkUpdateVideoSessionKHR returned: {:?}",
            result
        );
        if result != vk::Result::SUCCESS {
            return Err(VideoError::SessionCreation(format!(
                "vkUpdateVideoSessionKHR failed: {:?}",
                result
            )));
        }
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
                c"vkDestroyVideoSessionParametersKHR".as_ptr(),
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
