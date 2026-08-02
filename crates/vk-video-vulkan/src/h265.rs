//! H.265/HEVC Vulkan video decoder.

use ash::vk;
use ash::vk::native::*;

use super::codec_types::*;
use super::{VideoError, VideoResult};

use vk_video_core::picture::{H265Vps, H265Sps, H265Pps, H265ShortTermRefPicSet};

/// H.265 decoder state.
pub struct H265Decoder {
    device: ash::Device,
    instance: ash::Instance,
    /// Cached VPS for session parameters.
    vps: Option<H265Vps>,
    /// Cached SPS for decode info construction.
    sps: Option<H265Sps>,
    /// Cached PPS for decode info construction.
    pps: Option<H265Pps>,
    /// Frame counter for POC tracking.
    frame_count: u32,
    /// Previous POC LSB.
    prev_pic_order_cnt_lsb: u32,
}

impl H265Decoder {
    pub fn new(device: ash::Device, instance: ash::Instance) -> Self {
        Self {
            device,
            instance,
            vps: None,
            sps: None,
            pps: None,
            frame_count: 0,
            prev_pic_order_cnt_lsb: 0,
        }
    }

    pub fn set_vps(&mut self, vps: H265Vps) {
        self.vps = Some(vps);
    }

    pub fn set_sps(&mut self, sps: H265Sps) {
        self.sps = Some(sps);
    }

    pub fn set_pps(&mut self, pps: H265Pps) {
        self.pps = Some(pps);
    }

    /// Update session parameters with VPS/SPS/PPS data.
    pub fn update_session_parameters(
        &self,
        session_params: vk::VideoSessionParametersKHR,
        vps: Option<&H265Vps>,
        sps: Option<&H265Sps>,
        pps: Option<&H265Pps>,
    ) -> VideoResult<()> {
        let std_vps: Option<StdVideoH265VideoParameterSet> = vps.map(convert_h265_vps);
        let std_sps: Option<StdVideoH265SequenceParameterSet> = sps.map(convert_h265_sps);
        let std_pps: Option<StdVideoH265PictureParameterSet> = pps.map(convert_h265_pps);

        // Per C++ reference (VulkanVideoParser.cpp:2346-2351), all three counts should be set
        // when the corresponding parameter sets are available.
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

        let update_info = vk::VideoSessionParametersUpdateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
            p_next: &add_info as *const _ as *const _,
            update_sequence_count: 0,
            _marker: Default::default(),
        };

        self.update_session_parameters_raw(session_params, &update_info)
    }

    fn update_session_parameters_raw(
        &self,
        session_params: vk::VideoSessionParametersKHR,
        update_info: &vk::VideoSessionParametersUpdateInfoKHR<'_>,
    ) -> VideoResult<()> {
        let update_fn = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkUpdateVideoSessionParametersKHR\0".as_ptr().cast(),
            )
        }
        .ok_or_else(|| {
            VideoError::SessionCreation("vkUpdateVideoSessionParametersKHR not found".to_string())
        })?;

        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionParametersKHR,
                *const vk::VideoSessionParametersUpdateInfoKHR<'_>,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(update_fn);

            let result = fn_ptr(self.device.handle(), session_params, update_info);
            if result != vk::Result::SUCCESS {
                return Err(VideoError::SessionCreation(format!(
                    "vkUpdateVideoSessionParametersKHR failed: {:?}",
                    result
                )));
            }
        }

        Ok(())
    }

    /// Record a decode command.
    pub fn record_decode_command(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        session: vk::VideoSessionKHR,
        session_params: vk::VideoSessionParametersKHR,
        bitstream_buffer: vk::Buffer,
        bitstream_offset: u64,
        bitstream_range: u64,
        output_image_view: vk::ImageView,
        output_image: vk::Image,
        coded_extent: vk::Extent2D,
        dpb_setup_picture: Option<vk::VideoPictureResourceInfoKHR<'static>>,
        dpb_ref_pictures: &[vk::VideoPictureResourceInfoKHR<'static>],
        slice_offsets: &[u32],
        pic_order_cnt: Option<i32>,
        is_intra: Option<bool>,
        is_reference: Option<bool>,
        is_idr: Option<bool>,
    ) -> VideoResult<()> {
        let (effective_poc, effective_is_intra, effective_is_ref, effective_is_idr) =
            if let (Some(poc), Some(intra), Some(ref_), Some(idr)) =
                (pic_order_cnt, is_intra, is_reference, is_idr)
            {
                (poc, intra, ref_, idr)
            } else {
                let sps = self.sps.as_ref().expect("H265 SPS not set before decode");
                let log2_max_poc_lsb = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
                let max_poc_lsb = 1u32 << log2_max_poc_lsb;
                ((self.frame_count % max_poc_lsb) as i32, false, true, false)
            };

        let pic_info = self.build_picture_info(
            coded_extent,
            effective_poc,
            effective_is_intra,
            effective_is_ref,
            effective_is_idr,
        );

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(|e| {
                    VideoError::CommandBufferRecording(format!("Begin failed: {:?}", e))
                })?;

            // Bitstream buffer barrier
            let buffer_barrier = vk::BufferMemoryBarrier2 {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::HOST,
                src_access_mask: vk::AccessFlags2::HOST_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: bitstream_buffer,
                offset: bitstream_offset,
                size: bitstream_range,
                _marker: Default::default(),
            };

            let dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 1,
                p_buffer_memory_barriers: &buffer_barrier,
                image_memory_barrier_count: 0,
                p_image_memory_barriers: std::ptr::null(),
                _marker: Default::default(),
            };
            self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info);

            // Output image barrier
            // PLANE_0 for semi-planar YUV images (G8_B8R8_2PLANE_420_UNORM)
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let image_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::HOST,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: output_image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::VIDEO_DECODE_DST_KHR,
                subresource_range,
                _marker: Default::default(),
            };

            let dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 0,
                p_buffer_memory_barriers: std::ptr::null(),
                image_memory_barrier_count: 1,
                p_image_memory_barriers: &image_barrier,
                _marker: Default::default(),
            };
            self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info);

            // Begin video coding
            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: 0,
                p_reference_slots: std::ptr::null(),
                _marker: Default::default(),
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);

            // Build H.265 picture info
            let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: output_image_view,
                _marker: Default::default(),
            };

            let pic_ptr = &pic_info as *const StdVideoDecodeH265PictureInfo;

            let h265_decode_info = vk::VideoDecodeH265PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: pic_ptr,
                slice_segment_count: slice_offsets.len() as u32,
                p_slice_segment_offsets: if slice_offsets.is_empty() {
                    std::ptr::null()
                } else {
                    slice_offsets.as_ptr()
                },
                _marker: Default::default(),
            };

            // Build reference slots
            let setup_slot = dpb_setup_picture
                .as_ref()
                .map(|res| vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    slot_index: -1,
                    p_picture_resource: res as *const _,
                    _marker: Default::default(),
                });

            let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = dpb_ref_pictures
                .iter()
                .enumerate()
                .map(|(i, res)| vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    slot_index: i as i32,
                    p_picture_resource: &*res as *const _,
                    _marker: Default::default(),
                })
                .collect();

            let decode_info = vk::VideoDecodeInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
                p_next: &h265_decode_info as *const _ as *const _,
                flags: vk::VideoDecodeFlagsKHR::empty(),
                src_buffer: bitstream_buffer,
                src_buffer_offset: bitstream_offset,
                src_buffer_range: bitstream_range,
                dst_picture_resource: dst_picture_resource,
                p_setup_reference_slot: setup_slot
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s as *const _),
                reference_slot_count: ref_slots.len() as u32,
                p_reference_slots: ref_slots.as_ptr(),
                _marker: Default::default(),
            };

            self.cmd_decode_video(cmd_buffer, &decode_info);

            // End video coding
            self.cmd_end_video_coding(cmd_buffer);

            // End command buffer
            self.device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| VideoError::CommandBufferRecording(format!("End failed: {:?}", e)))?;
        }

        self.frame_count += 1;
        Ok(())
    }

    fn build_picture_info(
        &self,
        _coded_extent: vk::Extent2D,
        pic_order_cnt_val: i32,
        is_intra: bool,
        is_reference: bool,
        is_idr: bool,
    ) -> StdVideoDecodeH265PictureInfo {
        let sps = self.sps.as_ref().expect("H265 SPS not set before decode");
        let pps = self.pps.as_ref().expect("H265 PPS not set before decode");

        let mut pic_info = unsafe { std::mem::zeroed::<StdVideoDecodeH265PictureInfo>() };

        // Per C++ reference VulkanVideoParser.cpp:2379-2397:
        // Fields are populated from parsed slice header data
        pic_info.sps_video_parameter_set_id = sps.sps_video_parameter_set_id as u8;
        pic_info.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id as u8;
        pic_info.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id as u8;

        // IrapPicFlag: true for IRAP frames (BLA, CRA, IDR = NAL types 16-23)
        // is_intra indicates this is an IRAP frame
        pic_info.flags.set_IrapPicFlag(if is_intra { 1 } else { 0 });
        // IdrPicFlag: true only for IDR pictures (NAL unit types 19-20: IDR_W_RADL/IDR_N_LP)
        pic_info.flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
        pic_info
            .flags
            .set_IsReference(if is_reference { 1 } else { 0 });

        // NumBitsForShortTermRPSInSlice: size of short-term RPS in slice header
        // Per C++ reference VulkanVideoParser.cpp:2392
        // Set to 0 when not available from parser
        pic_info.NumBitsForSTRefPicSetInSlice = 0;

        // NumDeltaPocsOfRefRpsIdx: delta POCS of reference RPS index
        // Per C++ reference VulkanVideoParser.cpp:2396
        // Set to 0 when not available from parser
        pic_info.NumDeltaPocsOfRefRpsIdx = 0;

        // PicOrderCntVal from parsed slice header (CurrPicOrderCntVal)
        // Per C++ reference VulkanVideoParser.cpp:2397
        pic_info.PicOrderCntVal = pic_order_cnt_val;

        // RefPicSet arrays: DPB slot indices of reference pictures for current frame
        // Per C++ reference VulkanVideoParser.cpp:1666-1718
        // Initialize with 0xff (invalid) - should be filled from DPB reference list
        pic_info.RefPicSetStCurrBefore = [0xffu8; 8];
        pic_info.RefPicSetStCurrAfter = [0xffu8; 8];
        pic_info.RefPicSetLtCurr = [0xffu8; 8];

        pic_info
    }

    // Helper: dispatch cmdPipelineBarrier2
    fn cmd_pipeline_barrier_2(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dep_info: &vk::DependencyInfo<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdPipelineBarrier2KHR\0".as_ptr().cast(),
            )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, dep_info);
            }
        }
    }

    fn cmd_begin_video_coding(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoBeginCodingInfoKHR<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdBeginVideoCodingKHR\0".as_ptr().cast(),
            )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoBeginCodingInfoKHR<'_>,
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    fn cmd_decode_video(&self, cmd_buffer: vk::CommandBuffer, info: &vk::VideoDecodeInfoKHR<'_>) {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdDecodeVideoKHR\0".as_ptr().cast(),
            )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoDecodeInfoKHR<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdEndVideoCodingKHR\0".as_ptr().cast(),
            )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(vk::CommandBuffer);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer);
            }
        }
    }
}

fn convert_h265_sps(sps: &H265Sps) -> StdVideoH265SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265SpsFlags>() };
    flags.set_sps_temporal_id_nesting_flag(if sps.sps_temporal_id_nesting_flag { 1 } else { 0 });
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });
    flags.set_conformance_window_flag(if sps.conformance_window_flag { 1 } else { 0 });
    flags.set_sps_sub_layer_ordering_info_present_flag(
        if sps.sps_sub_layer_ordering_info_present_flag { 1 } else { 0 },
    );
    flags.set_scaling_list_enabled_flag(if sps.scaling_list_enabled_flag { 1 } else { 0 });
    flags.set_sps_scaling_list_data_present_flag(
        if sps.sps_scaling_list_data_present_flag { 1 } else { 0 },
    );
    flags.set_amp_enabled_flag(if sps.amp_enabled_flag { 1 } else { 0 });
    flags.set_sample_adaptive_offset_enabled_flag(
        if sps.sample_adaptive_offset_enabled_flag { 1 } else { 0 },
    );
    flags.set_sps_temporal_mvp_enabled_flag(
        if sps.sps_temporal_mvp_enabled_flag { 1 } else { 0 },
    );
    flags.set_strong_intra_smoothing_enabled_flag(
        if sps.strong_intra_smoothing_enabled_flag { 1 } else { 0 },
    );
    flags.set_long_term_ref_pics_present_flag(
        if sps.long_term_ref_pics_present_flag { 1 } else { 0 },
    );

    // DecPicBufMgr - always set per C++ reference (VulkanH265Parser.cpp:499)
    let max_latency_increase_plus1: [u32; 7] = sps
        .max_latency_increase_plus1
        .map(|v| v as u32);
    let dec_pic_buf_mgr = Box::leak(Box::new(StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: sps.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: sps.max_num_reorder_pics,
    }));

    // ShortTermRefPicSet array - per C++ reference (VulkanH265Parser.cpp:596-597)
    let short_term_ref_pic_set: *const StdVideoH265ShortTermRefPicSet =
        if !sps.short_term_ref_pic_sets.is_empty() {
            let std_strps: Vec<StdVideoH265ShortTermRefPicSet> = sps
                .short_term_ref_pic_sets
                .iter()
                .map(|strps| convert_h265_short_term_ref_pic_set(strps))
                .collect();
            Box::leak(std_strps.into_boxed_slice()).as_ptr()
        } else {
            std::ptr::null()
        };

    // LongTermRefPicsSps - per C++ reference (VulkanH265Parser.cpp:600+)
    let long_term_ref_pics_sps = if sps.long_term_ref_pics_present_flag
        && sps.num_long_term_ref_pics_sps > 0
    {
        let ltrp = Box::leak(Box::new(StdVideoH265LongTermRefPicsSps {
            used_by_curr_pic_lt_sps_flag: sps.used_by_curr_pic_lt_sps_flag,
            lt_ref_pic_poc_lsb_sps: sps.lt_ref_pic_poc_lsb_sps,
        }));
        ltrp
    } else {
        std::ptr::null()
    };

    StdVideoH265SequenceParameterSet {
        flags,
        chroma_format_idc: sps.chroma_format_idc as StdVideoH265ChromaFormatIdc,
        pic_width_in_luma_samples: sps.pic_width_in_luma_samples as u32,
        pic_height_in_luma_samples: sps.pic_height_in_luma_samples as u32,
        sps_video_parameter_set_id: sps.sps_video_parameter_set_id,
        sps_max_sub_layers_minus1: sps.sps_max_sub_layers_minus1,
        sps_seq_parameter_set_id: sps.sps_seq_parameter_set_id as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2: sps.log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size: sps.log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        num_short_term_ref_pic_sets: sps.num_short_term_ref_pic_sets,
        num_long_term_ref_pics_sps: sps.num_long_term_ref_pics_sps,
        pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3: sps.log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: sps.log2_diff_max_min_pcm_luma_coding_block_size,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: sps.palette_max_size,
        delta_palette_max_predictor_size: sps.delta_palette_max_predictor_size,
        motion_vector_resolution_control_idc: sps.motion_vector_resolution_control_idc,
        sps_num_palette_predictor_initializers_minus1: sps.sps_num_palette_predictor_initializers_minus1,
        conf_win_left_offset: sps.conf_win_left_offset,
        conf_win_right_offset: sps.conf_win_right_offset,
        conf_win_top_offset: sps.conf_win_top_offset,
        conf_win_bottom_offset: sps.conf_win_bottom_offset,
        pProfileTierLevel: std::ptr::null(),
        pDecPicBufMgr: dec_pic_buf_mgr,
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: short_term_ref_pic_set,
        pLongTermRefPicsSps: long_term_ref_pics_sps,
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }
}

fn convert_h265_short_term_ref_pic_set(
    strps: &H265ShortTermRefPicSet,
) -> StdVideoH265ShortTermRefPicSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265ShortTermRefPicSetFlags>() };
    flags.set_inter_ref_pic_set_prediction_flag(
        if strps.inter_ref_pic_set_prediction_flag { 1 } else { 0 },
    );

    StdVideoH265ShortTermRefPicSet {
        flags,
        delta_idx_minus1: strps.delta_idx_minus1,
        use_delta_flag: strps.use_delta_flag,
        abs_delta_rps_minus1: strps.abs_delta_rps_minus1,
        used_by_curr_pic_flag: strps.used_by_curr_pic_flag,
        used_by_curr_pic_s0_flag: strps.used_by_curr_pic_s0_flag,
        used_by_curr_pic_s1_flag: strps.used_by_curr_pic_s1_flag,
        reserved1: 0,
        reserved2: 0,
        reserved3: 0,
        num_negative_pics: strps.num_negative_pics,
        num_positive_pics: strps.num_positive_pics,
        delta_poc_s0_minus1: strps.delta_poc_s0_minus1,
        delta_poc_s1_minus1: strps.delta_poc_s1_minus1,
    }
}

fn convert_h265_pps(pps: &H265Pps) -> StdVideoH265PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265PpsFlags>() };
    flags.set_dependent_slice_segments_enabled_flag(
        if pps.dependent_slice_segments_enabled_flag { 1 } else { 0 },
    );
    flags.set_output_flag_present_flag(if pps.output_flag_present_flag { 1 } else { 0 });
    flags.set_sign_data_hiding_enabled_flag(if pps.sign_data_hiding_enabled_flag { 1 } else { 0 });
    flags.set_cabac_init_present_flag(if pps.cabac_init_present_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag { 1 } else { 0 });
    flags.set_transform_skip_enabled_flag(if pps.transform_skip_enabled_flag { 1 } else { 0 });
    flags.set_cu_qp_delta_enabled_flag(if pps.cu_qp_delta_enabled_flag { 1 } else { 0 });
    flags.set_pps_slice_chroma_qp_offsets_present_flag(
        if pps.pps_slice_chroma_qp_offsets_present_flag { 1 } else { 0 },
    );
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_weighted_bipred_flag(if pps.weighted_bipred_flag { 1 } else { 0 });
    flags.set_transquant_bypass_enabled_flag(
        if pps.transquant_bypass_enabled_flag { 1 } else { 0 },
    );
    flags.set_tiles_enabled_flag(if pps.tiles_enabled_flag { 1 } else { 0 });
    flags.set_entropy_coding_sync_enabled_flag(
        if pps.entropy_coding_sync_enabled_flag { 1 } else { 0 },
    );

    StdVideoH265PictureParameterSet {
        flags,
        pps_pic_parameter_set_id: pps.pps_pic_parameter_set_id as u8,
        pps_seq_parameter_set_id: pps.pps_seq_parameter_set_id as u8,
        sps_video_parameter_set_id: pps.sps_video_parameter_set_id,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        init_qp_minus26: pps.pps_init_qp_minus26 as i8,
        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        pps_beta_offset_div2: pps.pps_beta_offset_div2,
        pps_tc_offset_div2: pps.pps_tc_offset_div2,
        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
        log2_max_transform_skip_block_size_minus2: pps.log2_max_transform_skip_block_size_minus2,
        diff_cu_chroma_qp_offset_depth: pps.diff_cu_chroma_qp_offset_depth,
        chroma_qp_offset_list_len_minus1: pps.chroma_qp_offset_list_len_minus1,
        cb_qp_offset_list: pps.cb_qp_offset_list,
        cr_qp_offset_list: pps.cr_qp_offset_list,
        log2_sao_offset_scale_luma: pps.log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma: pps.log2_sao_offset_scale_chroma,
        pps_act_y_qp_offset_plus5: pps.pps_act_y_qp_offset_plus5,
        pps_act_cb_qp_offset_plus5: pps.pps_act_cb_qp_offset_plus5,
        pps_act_cr_qp_offset_plus3: pps.pps_act_cr_qp_offset_plus3,
        pps_num_palette_predictor_initializers: pps.pps_num_palette_predictor_initializers,
        luma_bit_depth_entry_minus8: pps.luma_bit_depth_entry_minus8,
        chroma_bit_depth_entry_minus8: pps.chroma_bit_depth_entry_minus8,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: pps.column_width_minus1,
        row_height_minus1: pps.row_height_minus1,
        reserved3: 0,
        pScalingLists: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }
}

fn convert_h265_vps(vps: &H265Vps) -> StdVideoH265VideoParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265VpsFlags>() };
    flags.set_vps_temporal_id_nesting_flag(if vps.vps_temporal_id_nesting_flag { 1 } else { 0 });

    // DecPicBufMgr is set when vps_max_sub_layers_minus1 != 0
    // Per C++ reference: VulkanH265Parser.cpp lines 943-963
    let dec_pic_buf_mgr = if vps.vps_max_sub_layers_minus1 != 0 {
        let max_latency_increase_plus1: [u32; 7] = vps
            .max_latency_increase_plus1
            .map(|v| v as u32);
        let mgr = StdVideoH265DecPicBufMgr {
            max_latency_increase_plus1,
            max_dec_pic_buffering_minus1: vps.max_dec_pic_buffering_minus1,
            max_num_reorder_pics: vps.max_num_reorder_pics,
        };
        Box::leak(Box::new(mgr))
    } else {
        std::ptr::null()
    };

    StdVideoH265VideoParameterSet {
        flags,
        vps_video_parameter_set_id: vps.vps_video_parameter_set_id,
        vps_max_sub_layers_minus1: vps.vps_max_sub_layers_minus1,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: vps.vps_num_units_in_tick,
        vps_time_scale: vps.vps_time_scale,
        vps_num_ticks_poc_diff_one_minus1: vps.vps_num_ticks_poc_diff_one_minus1,
        reserved3: 0,
        pDecPicBufMgr: dec_pic_buf_mgr,
        pHrdParameters: std::ptr::null(),
        pProfileTierLevel: std::ptr::null(),
    }
}
