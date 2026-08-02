//! H.264/AVC Vulkan video decoder.

use ash::vk;
use ash::vk::native::*;

use super::codec_types::*;
use super::{VideoError, VideoResult};

/// H.264 decoder state.
pub struct H264Decoder {
    device: ash::Device,
    instance: ash::Instance,
    /// Cached SPS for decode info construction.
    sps: Option<vk_video_core::picture::H264Sps>,
    /// Cached PPS for decode info construction.
    pps: Option<vk_video_core::picture::H264Pps>,
    /// Frame counter for POC tracking.
    frame_count: u32,
    /// Previous frame num for POC computation.
    prev_frame_num: u32,
    /// Previous POC LSB.
    prev_pic_order_cnt_lsb: u32,
}

impl H264Decoder {
    pub fn new(device: ash::Device, instance: ash::Instance) -> Self {
        Self {
            device,
            instance,
            sps: None,
            pps: None,
            frame_count: 0,
            prev_frame_num: 0,
            prev_pic_order_cnt_lsb: 0,
        }
    }

    pub fn set_sps(&mut self, sps: vk_video_core::picture::H264Sps) {
        self.sps = Some(sps);
    }

    pub fn set_pps(&mut self, pps: vk_video_core::picture::H264Pps) {
        self.pps = Some(pps);
    }

    /// Update session parameters with SPS/PPS data.
    pub fn update_session_parameters(
        &self,
        session_params: vk::VideoSessionParametersKHR,
        sps: Option<&vk_video_core::picture::H264Sps>,
        pps: Option<&vk_video_core::picture::H264Pps>,
    ) -> VideoResult<()> {
        let std_sps: Option<StdVideoH264SequenceParameterSet> =
            sps.map(convert_h264_sps);
        let std_pps: Option<StdVideoH264PictureParameterSet> =
            pps.map(convert_h264_pps);

        let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: std::ptr::null(),
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
            VideoError::SessionCreation(
                "vkUpdateVideoSessionParametersKHR not found".to_string(),
            )
        })?;

        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionParametersKHR,
                *const vk::VideoSessionParametersUpdateInfoKHR<'_>,
            ) -> vk::Result;
            let fn_ptr: FnType = std::mem::transmute(update_fn);

            let result = fn_ptr(
                self.device.handle(),
                session_params,
                update_info,
            );
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
    ///
    /// When frame_num/pic_order_cnt/is_intra/is_reference are None, the decoder
    /// computes them from internal state. For correct multi-frame decode, callers
    /// should provide these values parsed from the bitstream.
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
        frame_num: Option<u32>,
        pic_order_cnt: Option<[i32; 2]>,
        is_intra: Option<bool>,
        is_reference: Option<bool>,
    ) -> VideoResult<()> {
        // Use provided values or compute from internal state
        let (effective_frame_num, effective_poc, effective_is_intra, effective_is_ref) = 
            if let (Some(fn_), Some(poc), Some(intra), Some(ref_)) = 
                (frame_num, pic_order_cnt, is_intra, is_reference) {
                (fn_, poc, intra, ref_)
            } else {
                let sps = self.sps.as_ref().expect("H264 SPS not set before decode");
                let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
                let computed_frame_num = self.frame_count % max_frame_num;
                let log2_max_poc_lsb = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
                let max_poc_lsb = 1u32 << log2_max_poc_lsb;
                let poc_lsb = self.frame_count % max_poc_lsb;
                (
                    computed_frame_num,
                    [poc_lsb as i32, poc_lsb as i32],
                    false,
                    true,
                )
            };

        // Build picture info outside unsafe block so frame_num/poc are in scope
        let pic_info = self.build_picture_info(
            coded_extent,
            effective_frame_num,
            effective_poc,
            effective_is_intra,
            effective_is_ref,
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

            // Bitstream buffer barrier: HOST_WRITE -> VIDEO_DECODE_READ
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

            // Output image barrier: UNDEFINED -> VIDEO_DECODE_DPB_KHR
            // PLANE_0 and PLANE_1 for semi-planar YUV images (G8_B8R8_2PLANE_420_UNORM)
            // When old_layout is UNDEFINED, src_stage_mask must be NONE
            let image_barriers = [
                vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    src_access_mask: vk::AccessFlags2::NONE,
                    dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: output_image,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::PLANE_0,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    _marker: Default::default(),
                },
                vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    src_access_mask: vk::AccessFlags2::NONE,
                    dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: output_image,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::PLANE_1,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    _marker: Default::default(),
                },
            ];

            let dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 0,
                p_buffer_memory_barriers: std::ptr::null(),
                image_memory_barrier_count: image_barriers.len() as u32,
                p_image_memory_barriers: image_barriers.as_ptr(),
                _marker: Default::default(),
            };
            self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info);

            // Begin video coding
            // Include setup reference slot in begin coding's reference slots
            let total_begin_slots = dpb_setup_picture.as_ref().map_or(0, |_| 1) + dpb_ref_pictures.len();
            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: total_begin_slots as u32,
                p_reference_slots: std::ptr::null(), // Will be set below
                _marker: Default::default(),
            };

            // Build all reference slots for begin coding (setup + references)
            let mut all_begin_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
            if let Some(res) = dpb_setup_picture.as_ref() {
                all_begin_slots.push(vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    slot_index: 0,
                    p_picture_resource: res as *const _,
                    _marker: Default::default(),
                });
            }
            for (i, res) in dpb_ref_pictures.iter().enumerate() {
                all_begin_slots.push(vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    slot_index: i as i32,
                    p_picture_resource: &*res as *const _,
                    _marker: Default::default(),
                });
            }

            // Override p_reference_slots if we have slots
            let begin_coding_info = if all_begin_slots.is_empty() {
                begin_coding_info
            } else {
                vk::VideoBeginCodingInfoKHR {
                    p_reference_slots: all_begin_slots.as_ptr(),
                    ..begin_coding_info
                }
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);

            // Build H.264 picture info
            let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: output_image_view,
                _marker: Default::default(),
            };

            let pic_ptr = &pic_info as *const StdVideoDecodeH264PictureInfo;

            let h264_decode_info = vk::VideoDecodeH264PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H264_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: pic_ptr,
                slice_count: slice_offsets.len() as u32,
                p_slice_offsets: if slice_offsets.is_empty() {
                    std::ptr::null()
                } else {
                    slice_offsets.as_ptr()
                },
                _marker: Default::default(),
            };

            // Build reference slots - use same indices as in begin coding
            let setup_slot = dpb_setup_picture.as_ref().map(|res| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                slot_index: dpb_ref_pictures.len() as i32, // Setup slot after reference slots
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
                p_next: &h264_decode_info as *const _ as *const _,
                flags: vk::VideoDecodeFlagsKHR::empty(),
                src_buffer: bitstream_buffer,
                src_buffer_offset: bitstream_offset,
                src_buffer_range: bitstream_range,
                dst_picture_resource: dst_picture_resource,
                p_setup_reference_slot: setup_slot.as_ref().map_or(std::ptr::null(), |s| s as *const _),
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
                .map_err(|e| {
                    VideoError::CommandBufferRecording(format!("End failed: {:?}", e))
                })?;
        }

        // Update POC tracking
        self.prev_frame_num = effective_frame_num;
        self.prev_pic_order_cnt_lsb = effective_poc[0] as u32;
        self.frame_count += 1;

        Ok(())
    }

    /// Build StdVideoDecodeH264PictureInfo from parsed SPS/PPS.
    fn build_picture_info(
        &self,
        _coded_extent: vk::Extent2D,
        frame_num: u32,
        pic_order_cnt: [i32; 2],
        is_intra: bool,
        is_reference: bool,
    ) -> StdVideoDecodeH264PictureInfo {
        let sps = self
            .sps
            .as_ref()
            .expect("H264 SPS not set before decode");
        let pps = self
            .pps
            .as_ref()
            .expect("H264 PPS not set before decode");

        let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
        let effective_frame_num = frame_num % max_frame_num;

        let mut pic_info = unsafe { std::mem::zeroed::<StdVideoDecodeH264PictureInfo>() };
        pic_info.frame_num = effective_frame_num as u16;
        pic_info.PicOrderCnt = pic_order_cnt;
        pic_info.seq_parameter_set_id = sps.seq_parameter_set_id as u8;
        pic_info.pic_parameter_set_id = pps.pic_parameter_set_id as u8;

        // Set flags based on frame properties
        pic_info.flags.set_is_intra(if is_intra { 1 } else { 0 });
        pic_info.flags.set_is_reference(if is_reference { 1 } else { 0 });
        pic_info.flags.set_field_pic_flag(0);
        pic_info.flags.set_IdrPicFlag(0);
        pic_info.flags.set_bottom_field_flag(0);
        pic_info.flags.set_complementary_field_pair(0);

        pic_info
    }

    // Helper: dispatch cmdPipelineBarrier2
    fn cmd_pipeline_barrier_2(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dep_info: &vk::DependencyInfo<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
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

    // Helper: dispatch cmdBeginVideoCodingKHR
    fn cmd_begin_video_coding(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoBeginCodingInfoKHR<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdBeginVideoCodingKHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoBeginCodingInfoKHR<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    // Helper: dispatch cmdDecodeVideoKHR
    fn cmd_decode_video(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoDecodeInfoKHR<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
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

    // Helper: dispatch cmdEndVideoCodingKHR
    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
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

/// Convert our H264Sps to StdVideoH264SequenceParameterSet.
fn convert_h264_sps(sps: &vk_video_core::picture::H264Sps) -> StdVideoH264SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH264SpsFlags>() };
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });
    flags.set_qpprime_y_zero_transform_bypass_flag(if sps.qpprime_y_zero_transform_bypass_flag { 1 } else { 0 });
    flags.set_frame_mbs_only_flag(if sps.frame_mbs_only_flag { 1 } else { 0 });
    flags.set_direct_8x8_inference_flag(if sps.direct_8x8_inference_flag { 1 } else { 0 });
    flags.set_frame_cropping_flag(if sps.frame_cropping_flag { 1 } else { 0 });
    flags.set_vui_parameters_present_flag(if sps.vui_parameters_present_flag { 1 } else { 0 });

    // Convert VUI parameters if present
    let vui_data = if let Some(vui) = &sps.vui {
        let mut vui_flags = unsafe { std::mem::zeroed::<StdVideoH264SpsVuiFlags>() };
        vui_flags.set_aspect_ratio_info_present_flag(if vui.aspect_ratio_info_present_flag { 1 } else { 0 });
        vui_flags.set_overscan_info_present_flag(if vui.overscan_info_present_flag { 1 } else { 0 });
        vui_flags.set_overscan_appropriate_flag(if vui.overscan_appropriate_flag { 1 } else { 0 });
        vui_flags.set_video_signal_type_present_flag(if vui.video_signal_type_present_flag { 1 } else { 0 });
        vui_flags.set_video_full_range_flag(if vui.video_full_range_flag { 1 } else { 0 });
        vui_flags.set_color_description_present_flag(if vui.color_description_present_flag { 1 } else { 0 });
        vui_flags.set_chroma_loc_info_present_flag(if vui.chroma_loc_info_present_flag { 1 } else { 0 });
        vui_flags.set_timing_info_present_flag(if vui.timing_info_present_flag { 1 } else { 0 });
        vui_flags.set_fixed_frame_rate_flag(if vui.fixed_frame_rate_flag { 1 } else { 0 });
        vui_flags.set_bitstream_restriction_flag(if vui.bitstream_restriction_flag { 1 } else { 0 });
        vui_flags.set_nal_hrd_parameters_present_flag(if vui.nal_hrd_parameters_present_flag { 1 } else { 0 });
        vui_flags.set_vcl_hrd_parameters_present_flag(if vui.vcl_hrd_parameters_present_flag { 1 } else { 0 });

        StdVideoH264SequenceParameterSetVui {
            flags: vui_flags,
            aspect_ratio_idc: vui.aspect_ratio_idc as u32,
            sar_width: vui.sar_width,
            sar_height: vui.sar_height,
            video_format: vui.video_format,
            colour_primaries: vui.colour_primaries,
            transfer_characteristics: vui.transfer_characteristics,
            matrix_coefficients: vui.matrix_coefficients,
            num_units_in_tick: vui.num_units_in_tick,
            time_scale: vui.time_scale,
            max_num_reorder_frames: vui.max_num_reorder_frames,
            max_dec_frame_buffering: vui.max_dec_frame_buffering,
            chroma_sample_loc_type_top_field: vui.chroma_sample_loc_type_top_field,
            chroma_sample_loc_type_bottom_field: vui.chroma_sample_loc_type_bottom_field,
            reserved1: 0,
            pHrdParameters: std::ptr::null(), // HRD not implemented
        }
    } else {
        unsafe { std::mem::zeroed::<StdVideoH264SequenceParameterSetVui>() }
    };

    // Leak the Box to get a &'static pointer. Vulkan copies the data, so this is safe.
    let vui_ptr = Box::leak(Box::new(vui_data)) as *const StdVideoH264SequenceParameterSetVui;

    StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: sps.profile_idc as u32,
        level_idc: sps.level_idc as u32,
        chroma_format_idc: sps.chroma_format_idc as u32,
        seq_parameter_set_id: sps.seq_parameter_set_id as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type: sps.pic_order_cnt_type as u32,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: sps.max_num_ref_frames as u8,
        reserved1: 0,
        pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1 as u32,
        pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1 as u32,
        frame_crop_left_offset: sps.frame_crop_left_offset,
        frame_crop_right_offset: sps.frame_crop_right_offset,
        frame_crop_top_offset: sps.frame_crop_top_offset,
        frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: vui_ptr,
    }
}

/// Convert our H264Pps to StdVideoH264PictureParameterSet.
fn convert_h264_pps(pps: &vk_video_core::picture::H264Pps) -> StdVideoH264PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH264PpsFlags>() };
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_deblocking_filter_control_present_flag(if pps.deblocking_filter_control_present_flag { 1 } else { 0 });
    flags.set_redundant_pic_cnt_present_flag(if pps.redundant_pic_cnt_present_flag { 1 } else { 0 });
    flags.set_transform_8x8_mode_flag(if pps.transform_8x8_mode_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag { 1 } else { 0 });

    StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: pps.seq_parameter_set_id as u8,
        pic_parameter_set_id: pps.pic_parameter_set_id as u8,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as u8,
        weighted_bipred_idc: pps.weighted_bipred_idc as u32,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as i8,
        pScalingLists: std::ptr::null(),
    }
}
