//! H.265/HEVC Vulkan video decoder.

use ash::vk;
use ash::vk::Handle;
use ash::vk::native::*;

use super::{VideoError, VideoResult};

use vacc_core::picture::{H265Pps, H265ShortTermRefPicSet, H265Sps, H265SpsVui, H265Vps};

/// Reference picture info for H.265 DPB slot setup.
pub struct H265RefPictureInfo<'a> {
    /// Vulkan DPB slot index
    pub slot_index: u32,
    /// Picture order count
    pub pic_order_cnt: i32,
    /// Picture resource info
    pub picture_resource: vk::VideoPictureResourceInfoKHR<'a>,
    /// Vulkan image handle (for memory barriers)
    pub image: vk::Image,
    /// Current image layout (for memory barriers)
    pub current_layout: vk::ImageLayout,
    /// Absolute array layer of this slot's subresource within its image
    /// (slot index when the DPB is a layered image, 0 otherwise). Image
    /// memory barriers address the image directly, so they need the absolute
    /// layer (picture-resource baseArrayLayer stays view-relative: 0).
    pub image_base_layer: u32,
    /// The slot's picture is a long-term reference of the current picture
    /// (StdVideoDecodeH265ReferenceInfoFlags.used_for_long_term_reference).
    pub used_for_long_term_reference: bool,
}

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
    #[allow(dead_code)]
    prev_pic_order_cnt_lsb: u32,
    /// Monotonically increasing counter for session parameter updates.
    update_sequence_count: u32,
    /// Whether the mandatory one-time codec reset (vkCmdControlVideoCodingKHR
    /// with RESET) has been issued for this session. The Vulkan spec REQUIRES a
    /// reset before the first video coding operation on a newly created session;
    /// the NVIDIA driver traps in vkCmdDecodeVideoKHR without it.
    reset_done: bool,
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
            update_sequence_count: 0,
            reset_done: false,
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
        &mut self,
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

        // Increment update_sequence_count for each session parameter update.
        // Vulkan spec: must be monotonically increasing.
        self.update_sequence_count += 1;

        let update_info = vk::VideoSessionParametersUpdateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
            p_next: &add_info as *const _ as *const _,
            update_sequence_count: self.update_sequence_count,
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
                c"vkUpdateVideoSessionParametersKHR".as_ptr(),
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
    pub fn record_decode_command<'a>(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        session: vk::VideoSessionKHR,
        session_params: vk::VideoSessionParametersKHR,
        bitstream_buffer: vk::Buffer,
        bitstream_offset: u64,
        bitstream_range: u64,
        output_image_view: vk::ImageView,
        output_image: vk::Image,
        output_format: vk::Format,
        coded_extent: vk::Extent2D,
        dpb_setup_picture: Option<H265RefPictureInfo<'a>>,
        dpb_ref_pictures: &[H265RefPictureInfo<'a>],
        slice_offsets: &[u32],
        h265_info: &vacc_parser::h265::SliceHeaderInfo,
        rps_slots: (&[i32], &[i32], &[i32]),
        query_pool: vk::QueryPool,
        frame_index: u32,
    ) -> VideoResult<()> {
        // POC / RPS / list data come from the common parser + common DPB
        // (single source of truth shared across backends).
        let pic_info = self.build_picture_info(coded_extent, h265_info, rps_slots);

        if super::vacc_debug() && !dpb_ref_pictures.is_empty() {
            eprintln!(
                "[PIC-INFO] sps_vps_id={} pps_sps_id={} pps_id={} numDeltaPocsRefRpsIdx={} numBitsStrpsInSlice={} poc={} stBefore=[{:02x?}] stAfter=[{:02x?}] ltCurr=[{:02x?}] irap={} idr={} isref={} strps_sps_flag={}",
                pic_info.sps_video_parameter_set_id,
                pic_info.pps_seq_parameter_set_id,
                pic_info.pps_pic_parameter_set_id,
                pic_info.NumDeltaPocsOfRefRpsIdx,
                pic_info.NumBitsForSTRefPicSetInSlice,
                pic_info.PicOrderCntVal,
                pic_info.RefPicSetStCurrBefore,
                pic_info.RefPicSetStCurrAfter,
                pic_info.RefPicSetLtCurr,
                pic_info.flags.IrapPicFlag(),
                pic_info.flags.IdrPicFlag(),
                pic_info.flags.IsReference(),
                pic_info.flags.short_term_ref_pic_set_sps_flag()
            );
            eprintln!(
                "[PIC-INFO] begin_slots: setup={} refs={:?}",
                dpb_setup_picture.as_ref().map(|s| s.slot_index).unwrap_or(u32::MAX),
                dpb_ref_pictures.iter().map(|r| (r.slot_index, r.pic_order_cnt, r.used_for_long_term_reference)).collect::<Vec<_>>()
            );
        }

        // Video result-status queries must be explicitly reset after pool
        // creation and between uses (VUID-vkCmdDecodeVideoKHR-pNext-08366).
        // Matches C++ ref VkVideoDecoder.cpp: CmdResetQueryPool is recorded
        // right after BeginCommandBuffer, before CmdBeginVideoCodingKHR.
        if query_pool != vk::QueryPool::null() {
            self.cmd_reset_query_pool(cmd_buffer, query_pool, frame_index, 1);
        }

        // H.265 requires VkVideoDecodeH265DpbSlotInfoKHR in the pNext chain of
        // each reference slot (VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07163).
        // Build all structs in correct order to ensure stable pointers.

        // First: create StdVideoDecodeH265ReferenceInfo for setup slot
        let setup_ref_std_info = dpb_setup_picture.as_ref().map(|info| {
            let mut ref_std_info = unsafe { std::mem::zeroed::<StdVideoDecodeH265ReferenceInfo>() };
            ref_std_info.PicOrderCntVal = info.pic_order_cnt;
            ref_std_info.flags.set_used_for_long_term_reference(0);
            ref_std_info.flags.set_unused_for_reference(0);
            ref_std_info
        });

        // LT reference flag per reference slot (common DPB marking).
        let lt_flags: Vec<u32> = dpb_ref_pictures
            .iter()
            .map(|info| u32::from(info.used_for_long_term_reference))
            .collect();

        // Second: create VkVideoDecodeH265DpbSlotInfoKHR for setup slot
        let setup_dpb_slot_info =
            setup_ref_std_info
                .as_ref()
                .map(|ref_std_info| vk::VideoDecodeH265DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_std_info as *const _,
                    _marker: Default::default(),
                });

        // Third: create VkVideoReferenceSlotInfoKHR for setup slot
        // (DecodeVideo's pSetupReferenceSlot — actual slot index)
        let setup_slot = dpb_setup_picture.as_ref().map(|info| {
            vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: setup_dpb_slot_info
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s as *const _ as *const _),
                slot_index: info.slot_index as i32, // Use actual slot index
                p_picture_resource: &info.picture_resource as *const _,
                _marker: Default::default(),
            }
        });

        // Third-bis: copy of the setup slot for BeginVideoCoding.
        // Spec (vkCmdBeginVideoCodingKHR): a non-negative slotIndex must
        // identify a DPB slot in the ACTIVE state at execution time
        // (VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239). The output slot is
        // still INACTIVE when Begin executes — it only becomes active when the
        // decode activates it. So it must be bound with
        // slotIndex = VK_VIDEO_DECODE_SLOT_INDEX_NONE (-1) + picture resource
        // ("added to the set of bound reference picture resources without an
        // associated DPB slot"; the decode's pSetupReferenceSlot then
        // associates it with the actual slot). Matches FFmpeg
        // vulkan_decode.c ff_vk_decode_frame ("The current decoding reference
        // has to be bound as an inactive reference"). Declaring the actual
        // (inactive) index here makes the NVIDIA driver trap in
        // vkCmdDecodeVideoKHR.
        let setup_slot_for_begin = setup_slot.as_ref().map(|s| vk::VideoReferenceSlotInfoKHR {
            s_type: s.s_type,
            p_next: s.p_next,
            slot_index: -1,
            p_picture_resource: s.p_picture_resource,
            _marker: Default::default(),
        });

        // Fourth: create StdVideoDecodeH265ReferenceInfo for each ref picture
        let ref_std_infos: Vec<StdVideoDecodeH265ReferenceInfo> = dpb_ref_pictures
            .iter()
            .zip(lt_flags.iter())
            .map(|(info, lt_flag)| {
                let mut ref_std_info =
                    unsafe { std::mem::zeroed::<StdVideoDecodeH265ReferenceInfo>() };
                ref_std_info.PicOrderCntVal = info.pic_order_cnt;
                ref_std_info.flags.set_used_for_long_term_reference(*lt_flag);
                ref_std_info.flags.set_unused_for_reference(0);
                ref_std_info
            })
            .collect();

        // Fifth: create VkVideoDecodeH265DpbSlotInfoKHR for each ref picture
        let ref_dpb_slot_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = ref_std_infos
            .iter()
            .map(|ref_std_info| vk::VideoDecodeH265DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: ref_std_info as *const _,
                _marker: Default::default(),
            })
            .collect();

        // Sixth: create VkVideoReferenceSlotInfoKHR for each ref picture
        let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = dpb_ref_pictures
            .iter()
            .zip(ref_dpb_slot_infos.iter())
            .map(|(info, dpb_slot_info)| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: dpb_slot_info as *const _ as *const _,
                slot_index: info.slot_index as i32,
                p_picture_resource: &info.picture_resource as *const _,
                _marker: Default::default(),
            })
            .collect();

        // BeginVideoCoding reference slots: Match C++ reference VkVideoDecoder.cpp:1048-1052
        //
        // CRITICAL: When refs exist, BeginVideoCoding MUST include BOTH refs AND setup.
        // The count is ref_count + (setup ? 1 : 0), and pointer points to refs.
        // The driver reads count elements starting from refs pointer, so setup must be
        // placed immediately after refs in memory.
        //
        // C++ pattern:
        //   decodeBeginInfo.referenceSlotCount = decodeFrameInfo.referenceSlotCount +
        //       (decodeFrameInfo.pSetupReferenceSlot ? 1 : 0);
        //   decodeBeginInfo.pReferenceSlots = (decodeFrameInfo.referenceSlotCount > 0) ?
        //       decodeFrameInfo.pReferenceSlots : decodeFrameInfo.pSetupReferenceSlot;
        //
        // When refs exist: BeginVideoCoding uses refs + setup combined (count = refs + 1 if setup).
        // When no refs: BeginVideoCoding uses only setup slot.
        let setup_slot_for_decode = setup_slot;

        // Build combined slots array for BeginVideoCoding.
        // MUST keep this alive until after cmd_begin_video_coding is called!
        let begin_video_coding_slots: Vec<vk::VideoReferenceSlotInfoKHR> = if !ref_slots.is_empty()
        {
            let mut combined = ref_slots.clone();
            if let Some(ref setup) = setup_slot_for_begin {
                combined.push(*setup);
            }
            combined
        } else {
            setup_slot_for_begin.into_iter().collect()
        };

        let begin_slot_count = begin_video_coding_slots.len() as u32;
        let begin_slot_ptr = if begin_video_coding_slots.is_empty() {
            std::ptr::null()
        } else {
            begin_video_coding_slots.as_ptr()
        };

        // Begin video coding with reference slots
        let begin_coding_info = vk::VideoBeginCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoBeginCodingFlagsKHR::empty(),
            video_session: session,
            video_session_parameters: session_params,
            reference_slot_count: begin_slot_count,
            p_reference_slots: begin_slot_ptr,
            _marker: Default::default(),
        };

        self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);

        // MANDATORY codec reset before the first frame (Vulkan spec: a newly
        // created video session must be reset before any video coding op). Matches
        // the C++ reference (VkVideoDecoder.cpp:1187-1196) and our H264 path.
        if !self.reset_done {
            self.cmd_control_video_coding(cmd_buffer);
            self.reset_done = true;
        }

        // Barriers AFTER BeginVideoCoding and BEFORE DecodeVideo
        // This matches C++ reference VkVideoDecoder.cpp:1216-1227
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

        // Output image barrier
        // Multi-planar formats must not use COLOR in the aspect mask (the
        // NVIDIA driver loses the device when decoding P010 with COLOR).
        // When dpb_setup_picture points to the same image as dstPictureResource,
        // use VIDEO_DECODE_DPB_KHR layout (per Vulkan spec).
        // With a layered DPB image the barrier must target the output slot's
        // own array layer (C++ ref VkVideoDecoder.cpp:841-861); targeting
        // layer 0 of the shared image invalidates another frame's contents.
        let output_image_base_layer = dpb_setup_picture
            .as_ref()
            .map(|s| s.image_base_layer)
            .unwrap_or(0);
        let output_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: super::profile_chain::aspect_mask_for_format(output_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: output_image_base_layer,
            layer_count: 1,
        };

        let new_layout = if dpb_setup_picture.is_some() {
            // dpb_setup_picture points to the same image, so use DPB layout
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        } else {
            vk::ImageLayout::VIDEO_DECODE_DST_KHR
        };

        let image_barrier = vk::ImageMemoryBarrier2 {
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
            new_layout,
            subresource_range: output_subresource_range,
            _marker: Default::default(),
        };

        // Add barriers for reference images (matches C++ VkVideoDecoder.cpp:1044-1056)
        let mut all_image_barriers: Vec<vk::ImageMemoryBarrier2> = vec![image_barrier];
        for ref_pic in dpb_ref_pictures.iter() {
            // Only add barrier if image is valid and not already in DPB layout
            if ref_pic.image != vk::Image::null()
                && ref_pic.current_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR
            {
                all_image_barriers.push(vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    src_access_mask: vk::AccessFlags2::empty(),
                    dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: ref_pic.image,
                    old_layout: ref_pic.current_layout,
                    new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: super::profile_chain::aspect_mask_for_format(output_format),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: ref_pic.image_base_layer,
                        layer_count: 1,
                    },
                    _marker: Default::default(),
                });
            }
        }

        // Single DependencyInfo with ALL barriers (matches C++ VkVideoDecoder.cpp:1229-1240)
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buffer_barrier,
            image_memory_barrier_count: all_image_barriers.len() as u32,
            p_image_memory_barriers: all_image_barriers.as_ptr(),
            _marker: Default::default(),
        };
        self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info);

        // Build H.265 decode info
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

        // Match the C++ reference (VkVideoDecoder.cpp:1211-1230): the session is
        // created with VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR and every
        // decode command carries a VkVideoInlineQueryInfoKHR. The reference uses
        // one RESULT_STATUS_ONLY query per frame; queryPool=NULL + queryCount=0
        // means "no queries" (legal per spec) when no pool is available.
        let inline_query = if query_pool != vk::QueryPool::null() {
            super::inline_queries::VideoInlineQueryInfoKHR {
                s_type: super::inline_queries::VIDEO_INLINE_QUERY_INFO_KHR,
                p_next: &h265_decode_info as *const _ as *const _,
                query_pool: query_pool.as_raw(),
                first_query: frame_index,
                query_count: 1,
            }
        } else {
            super::inline_queries::empty_inline_queries(
                &h265_decode_info as *const _ as *const _,
            )
        };

        let decode_info = vk::VideoDecodeInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
            p_next: &inline_query as *const _ as *const _,
            flags: vk::VideoDecodeFlagsKHR::empty(),
            src_buffer: bitstream_buffer,
            src_buffer_offset: bitstream_offset,
            src_buffer_range: bitstream_range,
            dst_picture_resource,
            // Always include setup slot (VUID-vkCmdDecodeVideoKHR-pDecodeInfo-08376
            // requires pSetupReferenceSlot must not be NULL).
            p_setup_reference_slot: setup_slot_for_decode
                .as_ref()
                .map_or(std::ptr::null(), |s| s as *const _),
            reference_slot_count: ref_slots.len() as u32,
            p_reference_slots: if ref_slots.is_empty() {
                std::ptr::null()
            } else {
                ref_slots.as_ptr()
            },
            _marker: Default::default(),
        };

        self.cmd_decode_video(cmd_buffer, &decode_info);

        // End video coding
        self.cmd_end_video_coding(cmd_buffer);

        self.frame_count += 1;
        Ok(())
    }

    fn build_picture_info(
        &self,
        _coded_extent: vk::Extent2D,
        info: &vacc_parser::h265::SliceHeaderInfo,
        rps_slots: (&[i32], &[i32], &[i32]),
    ) -> StdVideoDecodeH265PictureInfo {
        let sps = self.sps.as_ref().expect("H265 SPS not set before decode");
        let pps = self.pps.as_ref().expect("H265 PPS not set before decode");

        let mut pic_info = unsafe { std::mem::zeroed::<StdVideoDecodeH265PictureInfo>() };

        pic_info.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        pic_info.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id as u8;
        pic_info.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id as u8;

        // IrapPicFlag: IRAP picture (BLA/CRA/IDR, NAL types 16-23) — NOT the
        // intra slice type: a CRA may carry an inter slice.
        pic_info.flags.set_IrapPicFlag(u32::from(info.is_rap));
        // IdrPicFlag: IDR pictures only (NAL unit types 19-20).
        pic_info.flags.set_IdrPicFlag(u32::from(info.is_idr));
        pic_info.flags.set_IsReference(u32::from(info.is_reference));
        pic_info
            .flags
            .set_short_term_ref_pic_set_sps_flag(u32::from(info.short_term_ref_pic_set_sps_flag));

        // RPS reconstruction hints for the driver (IDR pictures carry no RPS):
        // - SPS-flagged RPS: NumDeltaPocs[RefRpsIdx] = number of DeltaPoc
        //   entries (S0 + S1) in that SPS short-term reference picture set;
        // - in-slice RPS: its SizeInBits in the slice header.
        if !info.is_idr && info.short_term_ref_pic_set_sps_flag {
            let rps = sps
                .short_term_ref_pic_sets
                .get(info.short_term_ref_pic_set_idx as usize)
                .expect("SPS STRPS index out of range");
            pic_info.NumDeltaPocsOfRefRpsIdx =
                (rps.num_negative_pics + rps.num_positive_pics) as u8;
            pic_info.NumBitsForSTRefPicSetInSlice = 0;
        } else if !info.is_idr {
            pic_info.NumDeltaPocsOfRefRpsIdx = 0;
            pic_info.NumBitsForSTRefPicSetInSlice = info.num_bits_for_strps_in_slice;
        }

        // CurrPicOrderCntVal from the common parser.
        pic_info.PicOrderCntVal = info.curr_pic_order_cnt_val;

        // RefPicSet arrays: DPB slot of each RPS entry (common DPB match),
        // STD_VIDEO_H265_NO_REFERENCE_PICTURE (0xff) when the entry's picture
        // is not in the DPB or has no entry.
        pic_info.RefPicSetStCurrBefore = [0xff; 8];
        pic_info.RefPicSetStCurrAfter = [0xff; 8];
        pic_info.RefPicSetLtCurr = [0xff; 8];
        let fill = |arr: &mut [u8; 8], slots: &[i32]| {
            for (i, s) in slots.iter().enumerate().take(8) {
                arr[i] = if *s >= 0 { *s as u8 } else { 0xff };
            }
        };
        fill(&mut pic_info.RefPicSetStCurrBefore, rps_slots.0);
        fill(&mut pic_info.RefPicSetStCurrAfter, rps_slots.1);
        fill(&mut pic_info.RefPicSetLtCurr, rps_slots.2);

        pic_info
    }

    // Helper: dispatch vkCmdResetQueryPool (video result-status queries must
    // be reset after pool creation and between uses, VUID-08366)
    fn cmd_reset_query_pool(
        &self,
        cmd_buffer: vk::CommandBuffer,
        query_pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdResetQueryPool".as_ptr(),
            )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, vk::QueryPool, u32, u32);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, query_pool, first_query, query_count);
            }
        } else if super::vacc_debug() {
            eprintln!("[H265] WARNING: vkCmdResetQueryPool not found; queries may be invalid");
        }
    }

    // Helper: dispatch cmdPipelineBarrier2
    fn cmd_pipeline_barrier_2(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dep_info: &vk::DependencyInfo<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdPipelineBarrier2KHR".as_ptr())
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
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdBeginVideoCodingKHR".as_ptr())
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoBeginCodingInfoKHR<'_>,
                ) -> i64;
                let f: FnType = std::mem::transmute(ptr);
                let rc = f(cmd_buffer, info);
                if std::env::var("VACC_DBG_H265").is_ok() {
                    eprintln!("[H265-DEC] vkCmdBeginVideoCodingKHR rc={rc} (0x{rc:x})");
                }
            }
        }
    }

    fn cmd_decode_video(&self, cmd_buffer: vk::CommandBuffer, info: &vk::VideoDecodeInfoKHR<'_>) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdDecodeVideoKHR".as_ptr())
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(
                        vk::CommandBuffer,
                        *const vk::VideoDecodeInfoKHR<'_>,
                    ) -> i64;
                let f: FnType = std::mem::transmute(ptr);
                let rc = f(cmd_buffer, info);
                if std::env::var("VACC_DBG_H265").is_ok() {
                    eprintln!("[H265-DEC] vkCmdDecodeVideoKHR rc={rc} (0x{rc:x})");
                }
            }
        }
    }

    fn cmd_control_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let coding_control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdControlVideoCodingKHR".as_ptr())
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoCodingControlInfoKHR<'_>,
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, &coding_control_info);
            }
            if super::vacc_debug() {
                eprintln!("[H265] issued codec RESET (vkCmdControlVideoCodingKHR)");
            }
        } else {
            eprintln!("[H265] WARNING: vkCmdControlVideoCodingKHR not found; skipping mandatory reset");
        }
    }

    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };

        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdEndVideoCodingKHR".as_ptr())
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoEndCodingInfoKHR<'_>,
                ) -> i64;
                let f: FnType = std::mem::transmute(ptr);
                let r = f(cmd_buffer, &end_coding_info);
                if std::env::var("VACC_DBG_H265").is_ok() {
                    eprintln!("[H265-DEC] vkCmdEndVideoCodingKHR rc={r} (0x{r:x})");
                }
            }
        }
    }
}

/// Convert raw H.265 level_idc to Vulkan StdVideoH265LevelIdc enum.
///
/// H.265 level_idc = level_number * 30 (e.g., 4.1 -> 123, 3.1 -> 93).
/// Matches C++ reference VulkanH265Parser.cpp:generalLevelIdcToVulkanLevelIdcEnum().
fn h265_level_idc_to_vulkan(raw_level_idc: u8) -> StdVideoH265LevelIdc {
    match raw_level_idc {
        30 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        60 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        63 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        90 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        93 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        120 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        123 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        150 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        153 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        156 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        180 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        183 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        186 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
        _ => {
            eprintln!(
                "[H265] WARNING: Unknown level_idc={}, defaulting to 5.1",
                raw_level_idc
            );
            StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1
        }
    }
}

/// Convert H.265 SPS VUI to Vulkan StdVideoH265SequenceParameterSetVui.
/// This is critical for correct chroma output - the video_full_range_flag
/// tells the decoder whether to use full range (0-255) or limited range (16-235).
pub fn convert_h265_vui(vui: &H265SpsVui) -> StdVideoH265SequenceParameterSetVui {
    let mut vui_flags = unsafe { std::mem::zeroed::<StdVideoH265SpsVuiFlags>() };
    vui_flags.set_aspect_ratio_info_present_flag(if vui.aspect_ratio_info_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_overscan_info_present_flag(if vui.overscan_info_present_flag { 1 } else { 0 });
    vui_flags.set_overscan_appropriate_flag(if vui.overscan_appropriate_flag { 1 } else { 0 });
    vui_flags.set_video_signal_type_present_flag(if vui.video_signal_type_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_video_full_range_flag(if vui.video_full_range_flag { 1 } else { 0 });
    vui_flags.set_colour_description_present_flag(if vui.colour_description_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_chroma_loc_info_present_flag(if vui.chroma_loc_info_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_neutral_chroma_indication_flag(if vui.neutral_chroma_indication_flag {
        1
    } else {
        0
    });
    vui_flags.set_field_seq_flag(if vui.field_seq_flag { 1 } else { 0 });
    vui_flags.set_frame_field_info_present_flag(if vui.frame_field_info_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_default_display_window_flag(if vui.default_display_window_flag {
        1
    } else {
        0
    });
    vui_flags.set_vui_timing_info_present_flag(if vui.vui_timing_info_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_vui_poc_proportional_to_timing_flag(if vui.vui_poc_proportional_to_timing_flag {
        1
    } else {
        0
    });
    vui_flags.set_vui_hrd_parameters_present_flag(if vui.vui_hrd_parameters_present_flag {
        1
    } else {
        0
    });
    vui_flags.set_bitstream_restriction_flag(if vui.bitstream_restriction_flag { 1 } else { 0 });
    vui_flags.set_tiles_fixed_structure_flag(if vui.tiles_fixed_structure_flag { 1 } else { 0 });
    vui_flags.set_motion_vectors_over_pic_boundaries_flag(
        if vui.motion_vectors_over_pic_boundaries_flag {
            1
        } else {
            0
        },
    );
    vui_flags.set_restricted_ref_pic_lists_flag(if vui.restricted_ref_pic_lists_flag {
        1
    } else {
        0
    });

    StdVideoH265SequenceParameterSetVui {
        flags: vui_flags,
        aspect_ratio_idc: vui.aspect_ratio_idc as StdVideoH265AspectRatioIdc,
        sar_width: vui.sar_width,
        sar_height: vui.sar_height,
        video_format: vui.video_format,
        colour_primaries: vui.colour_primaries,
        transfer_characteristics: vui.transfer_characteristics,
        matrix_coeffs: vui.matrix_coeffs,
        chroma_sample_loc_type_top_field: vui.chroma_sample_loc_type_top_field as u8,
        chroma_sample_loc_type_bottom_field: vui.chroma_sample_loc_type_bottom_field as u8,
        reserved1: 0,
        reserved2: 0,
        def_disp_win_left_offset: vui.def_disp_win_left_offset as u16,
        def_disp_win_right_offset: vui.def_disp_win_right_offset as u16,
        def_disp_win_top_offset: vui.def_disp_win_top_offset as u16,
        def_disp_win_bottom_offset: vui.def_disp_win_bottom_offset as u16,
        vui_num_units_in_tick: vui.vui_num_units_in_tick,
        vui_time_scale: vui.vui_time_scale,
        vui_num_ticks_poc_diff_one_minus1: vui.vui_num_ticks_poc_diff_one_minus1,
        min_spatial_segmentation_idc: vui.min_spatial_segmentation_idc as u16,
        reserved3: 0,
        max_bytes_per_pic_denom: vui.max_bytes_per_pic_denom as u8,
        max_bits_per_min_cu_denom: vui.max_bits_per_min_cu_denom as u8,
        log2_max_mv_length_horizontal: vui.log2_max_mv_length_horizontal as u8,
        log2_max_mv_length_vertical: vui.log2_max_mv_length_vertical as u8,
        pHrdParameters: std::ptr::null(),
    }
}

pub fn convert_h265_sps(sps: &H265Sps) -> StdVideoH265SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265SpsFlags>() };
    flags.set_sps_temporal_id_nesting_flag(if sps.sps_temporal_id_nesting_flag {
        1
    } else {
        0
    });
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });
    flags.set_conformance_window_flag(if sps.conformance_window_flag { 1 } else { 0 });
    flags.set_sps_sub_layer_ordering_info_present_flag(
        if sps.sps_sub_layer_ordering_info_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_scaling_list_enabled_flag(if sps.scaling_list_enabled_flag { 1 } else { 0 });
    flags.set_sps_scaling_list_data_present_flag(if sps.sps_scaling_list_data_present_flag {
        1
    } else {
        0
    });
    flags.set_amp_enabled_flag(if sps.amp_enabled_flag { 1 } else { 0 });
    flags.set_sample_adaptive_offset_enabled_flag(if sps.sample_adaptive_offset_enabled_flag {
        1
    } else {
        0
    });
    flags.set_sps_temporal_mvp_enabled_flag(if sps.sps_temporal_mvp_enabled_flag {
        1
    } else {
        0
    });
    flags.set_strong_intra_smoothing_enabled_flag(if sps.strong_intra_smoothing_enabled_flag {
        1
    } else {
        0
    });
    flags.set_long_term_ref_pics_present_flag(if sps.long_term_ref_pics_present_flag {
        1
    } else {
        0
    });
    flags.set_pcm_enabled_flag(if sps.pcm_enabled_flag { 1 } else { 0 });
    flags.set_pcm_loop_filter_disabled_flag(if sps.pcm_loop_filter_disabled_flag {
        1
    } else {
        0
    });
    flags.set_vui_parameters_present_flag(if sps.vui_parameters_present_flag {
        1
    } else {
        0
    });
    flags.set_sps_extension_present_flag(if sps.sps_extension_present_flag { 1 } else { 0 });
    flags.set_sps_range_extension_flag(if sps.sps_range_extension_flag { 1 } else { 0 });
    flags.set_intra_smoothing_disabled_flag(if sps.intra_smoothing_disabled_flag {
        1
    } else {
        0
    });
    flags.set_palette_mode_enabled_flag(if sps.palette_mode_enabled_flag { 1 } else { 0 });

    // DecPicBufMgr - always set per C++ reference (VulkanH265Parser.cpp:499)
    let max_latency_increase_plus1: [u32; 7] = sps.max_latency_increase_plus1.map(|v| v as u32);
    let dec_pic_buf_mgr_data = StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: sps.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: sps.max_num_reorder_pics,
    };
    let dec_pic_buf_mgr = Box::leak(Box::new(dec_pic_buf_mgr_data));

    // ShortTermRefPicSet array - per C++ reference (VulkanH265Parser.cpp:596-597)
    let short_term_ref_pic_set: *const StdVideoH265ShortTermRefPicSet =
        if !sps.short_term_ref_pic_sets.is_empty() {
            let std_strps: Vec<StdVideoH265ShortTermRefPicSet> = sps
                .short_term_ref_pic_sets
                .iter()
                .map(convert_h265_short_term_ref_pic_set)
                .collect();
            Box::leak(std_strps.into_boxed_slice()).as_ptr()
        } else {
            std::ptr::null()
        };

    // LongTermRefPicsSps - per C++ reference (VulkanH265Parser.cpp:600+)
    let long_term_ref_pics_sps =
        if sps.long_term_ref_pics_present_flag && sps.num_long_term_ref_pics_sps > 0 {
            let ltrp = Box::leak(Box::new(StdVideoH265LongTermRefPicsSps {
                used_by_curr_pic_lt_sps_flag: sps.used_by_curr_pic_lt_sps_flag,
                lt_ref_pic_poc_lsb_sps: sps.lt_ref_pic_poc_lsb_sps,
            }));
            ltrp
        } else {
            std::ptr::null()
        };

    // ProfileTierLevel - REQUIRED by Vulkan spec
    // Use actual profile/level from parsed SPS, matching C++ reference
    let mut ptl = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevel>() };
    ptl.flags
        .set_general_tier_flag(if sps.tier_flag { 1 } else { 0 });
    ptl.general_profile_idc = sps.profile_idc as StdVideoH265ProfileIdc;
    ptl.general_level_idc = h265_level_idc_to_vulkan(sps.level_idc);
    let profile_tier_level = Box::leak(Box::new(ptl));

    let p_sequence_parameter_set_vui = if sps.vui_parameters_present_flag {
        let vui_data = convert_h265_vui(&sps.vui);
        Box::leak(Box::new(vui_data)) as *const StdVideoH265SequenceParameterSetVui
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
        log2_diff_max_min_luma_transform_block_size: sps
            .log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        num_short_term_ref_pic_sets: sps.num_short_term_ref_pic_sets,
        num_long_term_ref_pics_sps: sps.num_long_term_ref_pics_sps,
        pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3: sps.log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: sps
            .log2_diff_max_min_pcm_luma_coding_block_size,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: sps.palette_max_size,
        delta_palette_max_predictor_size: sps.delta_palette_max_predictor_size,
        motion_vector_resolution_control_idc: sps.motion_vector_resolution_control_idc,
        sps_num_palette_predictor_initializers_minus1: sps
            .sps_num_palette_predictor_initializers_minus1,
        conf_win_left_offset: sps.conf_win_left_offset,
        conf_win_right_offset: sps.conf_win_right_offset,
        conf_win_top_offset: sps.conf_win_top_offset,
        conf_win_bottom_offset: sps.conf_win_bottom_offset,
        pProfileTierLevel: profile_tier_level,
        pDecPicBufMgr: dec_pic_buf_mgr,
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: short_term_ref_pic_set,
        pLongTermRefPicsSps: long_term_ref_pics_sps,
        pSequenceParameterSetVui: p_sequence_parameter_set_vui,
        pPredictorPaletteEntries: std::ptr::null(),
    }
}

pub fn convert_h265_short_term_ref_pic_set(
    strps: &H265ShortTermRefPicSet,
) -> StdVideoH265ShortTermRefPicSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265ShortTermRefPicSetFlags>() };
    flags.set_inter_ref_pic_set_prediction_flag(if strps.inter_ref_pic_set_prediction_flag {
        1
    } else {
        0
    });

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

pub fn convert_h265_pps(pps: &H265Pps) -> StdVideoH265PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265PpsFlags>() };
    flags.set_dependent_slice_segments_enabled_flag(if pps.dependent_slice_segments_enabled_flag {
        1
    } else {
        0
    });
    flags.set_output_flag_present_flag(if pps.output_flag_present_flag { 1 } else { 0 });
    flags.set_sign_data_hiding_enabled_flag(if pps.sign_data_hiding_enabled_flag {
        1
    } else {
        0
    });
    flags.set_cabac_init_present_flag(if pps.cabac_init_present_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag {
        1
    } else {
        0
    });
    flags.set_transform_skip_enabled_flag(if pps.transform_skip_enabled_flag {
        1
    } else {
        0
    });
    flags.set_cu_qp_delta_enabled_flag(if pps.cu_qp_delta_enabled_flag { 1 } else { 0 });
    flags.set_pps_slice_chroma_qp_offsets_present_flag(
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_weighted_bipred_flag(if pps.weighted_bipred_flag { 1 } else { 0 });
    flags.set_transquant_bypass_enabled_flag(if pps.transquant_bypass_enabled_flag {
        1
    } else {
        0
    });
    flags.set_tiles_enabled_flag(if pps.tiles_enabled_flag { 1 } else { 0 });
    flags.set_entropy_coding_sync_enabled_flag(if pps.entropy_coding_sync_enabled_flag {
        1
    } else {
        0
    });
    flags.set_uniform_spacing_flag(if pps.uniform_spacing_flag { 1 } else { 0 });
    flags.set_loop_filter_across_tiles_enabled_flag(if pps.loop_filter_across_tiles_enabled_flag {
        1
    } else {
        0
    });
    flags.set_pps_loop_filter_across_slices_enabled_flag(
        if pps.pps_loop_filter_across_slices_enabled_flag {
            1
        } else {
            0
        },
    );
    flags.set_deblocking_filter_control_present_flag(
        if pps.deblocking_filter_control_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_deblocking_filter_override_enabled_flag(
        if pps.deblocking_filter_override_enabled_flag {
            1
        } else {
            0
        },
    );
    flags.set_pps_deblocking_filter_disabled_flag(if pps.pps_deblocking_filter_disabled_flag {
        1
    } else {
        0
    });
    flags.set_pps_scaling_list_data_present_flag(if pps.pps_scaling_list_data_present_flag {
        1
    } else {
        0
    });
    flags.set_lists_modification_present_flag(if pps.lists_modification_present_flag {
        1
    } else {
        0
    });

    // TEMPORARY debug override for PPS field isolation testing
    if let Ok(v) = std::env::var("VACC_PPS_OVR") {
        for pair in v.split(',') {
            if let Some((k, val)) = pair.split_once('=') {
                let v: u32 = val.parse().unwrap_or(0);
                match k {
                    "across_slices" => flags.set_pps_loop_filter_across_slices_enabled_flag(v),
                    "lists_mod" => flags.set_lists_modification_present_flag(v),
                    "log2_par_merge" => {} // handled below via struct field
                    _ => {}
                }
            }
        }
    }

    let mut log2_par_merge = pps.log2_parallel_merge_level_minus2;
    if let Ok(v) = std::env::var("VACC_PPS_OVR") {
        for pair in v.split(',') {
            if let Some(("log2_par_merge", val)) = pair.split_once('=') {
                log2_par_merge = val.parse().unwrap_or(log2_par_merge);
            }
        }
    }

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
        log2_parallel_merge_level_minus2: log2_par_merge,
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

pub fn convert_h265_vps(vps: &H265Vps) -> StdVideoH265VideoParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265VpsFlags>() };
    flags.set_vps_temporal_id_nesting_flag(if vps.vps_temporal_id_nesting_flag {
        1
    } else {
        0
    });
    flags.set_vps_sub_layer_ordering_info_present_flag(
        if vps.vps_sub_layer_ordering_info_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_vps_timing_info_present_flag(if vps.vps_timing_info_present_flag {
        1
    } else {
        0
    });

    // DecPicBufMgr - always set per C++ reference
    let max_latency_increase_plus1: [u32; 7] = vps.max_latency_increase_plus1.map(|v| v as u32);
    let mgr = StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: vps.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: vps.max_num_reorder_pics,
    };
    let dec_pic_buf_mgr = Box::leak(Box::new(mgr));

    // ProfileTierLevel - REQUIRED by Vulkan spec
    // Use actual profile/level from parsed VPS, matching C++ reference
    let mut ptl = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevel>() };
    ptl.flags
        .set_general_tier_flag(if vps.tier_flag { 1 } else { 0 });
    ptl.general_profile_idc = vps.profile_idc as StdVideoH265ProfileIdc;
    ptl.general_level_idc = h265_level_idc_to_vulkan(vps.level_idc);
    let profile_tier_level = Box::leak(Box::new(ptl));

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
        pProfileTierLevel: profile_tier_level,
    }
}
