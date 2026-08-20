//! H.264/AVC Vulkan video decoder.

use ash::vk;
use ash::vk::native::*;

use super::dpb::LastAccessType;
use super::{VideoError, VideoResult};

/// H.264 DPB reference picture with metadata for Vulkan decode.
#[derive(Debug, Clone)]
pub struct H264DpbRefPicture<'a> {
    pub slot_index: u32,
    pub picture_resource: vk::VideoPictureResourceInfoKHR<'a>,
    pub image: vk::Image,
    pub frame_num: u32,
    pub pic_order_cnt: [i32; 2],
    pub current_layout: vk::ImageLayout,
    pub last_access: LastAccessType,
}

/// H.264 setup picture info (output picture being decoded into).
#[derive(Debug)]
pub struct H264SetupPictureInfo<'a> {
    pub slot_index: u32,
    pub picture_resource: vk::VideoPictureResourceInfoKHR<'a>,
}

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
    /// Monotonically increasing counter for session parameter updates.
    update_sequence_count: u32,
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
            update_sequence_count: 0,
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
        &mut self,
        session_params: vk::VideoSessionParametersKHR,
        sps: Option<&vk_video_core::picture::H264Sps>,
        pps: Option<&vk_video_core::picture::H264Pps>,
    ) -> VideoResult<()> {
        let std_sps: Option<StdVideoH264SequenceParameterSet> = sps.map(convert_h264_sps);
        let std_pps: Option<StdVideoH264PictureParameterSet> = pps.map(convert_h264_pps);

        let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: std::ptr::null(),
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
    ///
    /// When frame_num/pic_order_cnt/is_intra/is_reference are None, the decoder
    /// computes them from internal state. For correct multi-frame decode, callers
    /// should provide these values parsed from the bitstream.
    ///
    /// On the first frame (is_first_frame=true), ALL DPB slots are activated
    /// to satisfy VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239.
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
        dpb_setup_picture: Option<H264SetupPictureInfo<'static>>,
        dpb_ref_pictures: &[H264DpbRefPicture<'_>],
        slice_offsets: &[u32],
        frame_num: Option<u32>,
        pic_order_cnt: Option<[i32; 2]>,
        is_intra: Option<bool>,
        is_reference: Option<bool>,
        is_idr: Option<bool>,
        is_first_frame: bool,
        _max_dpb_slots: u32,
        _dpb_images: &[vk::Image],
        dpb_views: &[vk::ImageView],
    ) -> VideoResult<()> {
        // Use provided values or compute from internal state
        let (
            effective_frame_num,
            effective_poc,
            effective_is_intra,
            effective_is_ref,
            effective_is_idr,
        ) = if let (Some(fn_), Some(poc), Some(intra), Some(ref_), Some(idr)) =
            (frame_num, pic_order_cnt, is_intra, is_reference, is_idr)
        {
            (fn_, poc, intra, ref_, idr)
        } else {
            let sps = self.sps.as_ref().ok_or(VideoError::InvalidState(
                "H264 SPS not set before decode".into(),
            ))?;
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
                false,
            )
        };

        // Build picture info outside unsafe block so frame_num/poc are in scope
        let pic_info = self.build_picture_info(
            coded_extent,
            effective_frame_num,
            effective_poc,
            effective_is_intra,
            effective_is_ref,
            effective_is_idr,
        );

        // Begin video coding (BEFORE memory barriers, matches C++ reference)
        //
        // CRITICAL: On first frame (RESET), activate ALL DPB slots to satisfy
        // VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239. Subsequent frames
        // reference slots that were activated on the RESET frame.
        //
        // Matches working example: vulkan_decode.rs lines 4061-4140

        let max_frame_num = 1u32
            << (self
                .sps
                .as_ref()
                .ok_or(VideoError::InvalidState("H264 SPS not set".into()))?
                .log2_max_frame_num_minus4 as u32
                + 4);

        // Build reference slots for begin coding
        //
        // CRITICAL FIX: Match C++ reference VkVideoDecoder.cpp:1079-1084 exactly:
        //   decodeBeginInfo.referenceSlotCount = decodeFrameInfo.referenceSlotCount +
        //       (decodeFrameInfo.pSetupReferenceSlot ? 1 : 0);
        //   decodeBeginInfo.pReferenceSlots = (decodeFrameInfo.referenceSlotCount > 0) ?
        //       decodeFrameInfo.pReferenceSlots : decodeFrameInfo.pSetupReferenceSlot;
        //
        // When refs exist: BeginVideoCoding uses SAME slots as DecodeVideo's p_reference_slots.
        // When no refs: BeginVideoCoding uses only the setup slot.
        //
        // The setup picture is NOT in the reference slots array when refs exist;
        // it's passed separately via DecodeVideo's p_setup_reference_slot.
        let all_begin_slots: Vec<vk::VideoReferenceSlotInfoKHR>;

        if is_first_frame {
            // RESET frame: activate ALL DPB slots

            // Build picture resources for all slots
            let all_picture_resources: Vec<vk::VideoPictureResourceInfoKHR> = (0..dpb_views.len()
                as u32)
                .map(|slot_idx| vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent,
                    base_array_layer: 0,
                    image_view_binding: dpb_views[slot_idx as usize],
                    _marker: Default::default(),
                })
                .collect();

            // Build setup slot with DPB info
            let setup_slot_idx = dpb_setup_picture
                .as_ref()
                .map_or(0, |s| s.slot_index as usize);

            let mut setup_ref_info =
                unsafe { std::mem::zeroed::<StdVideoDecodeH264ReferenceInfo>() };
            setup_ref_info.FrameNum = (effective_frame_num % max_frame_num) as u16;
            setup_ref_info.PicOrderCnt = effective_poc;
            // For progressive frames: both fields are available for prediction
            setup_ref_info.flags.set_top_field_flag(1);
            setup_ref_info.flags.set_bottom_field_flag(1);
            let setup_dpb_slot_info =
                vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_ref_info);

            // Build all slots
            all_begin_slots = (0..dpb_views.len() as u32)
                .map(|slot_idx| {
                    let is_setup = slot_idx as usize == setup_slot_idx;
                    let pr = &all_picture_resources[slot_idx as usize];
                    let p_next = if is_setup {
                        &setup_dpb_slot_info as *const _ as *const _
                    } else {
                        // Non-setup slots on RESET: no DPB slot info needed for H.264
                        std::ptr::null()
                    };
                    vk::VideoReferenceSlotInfoKHR {
                        s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                        p_next,
                        slot_index: slot_idx as i32,
                        p_picture_resource: pr,
                        _marker: Default::default(),
                    }
                })
                .collect();

            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: all_begin_slots.len() as u32,
                p_reference_slots: all_begin_slots.as_ptr(),
                _marker: Default::default(),
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);
        } else {
            // Non-RESET frame: Build DPB slot info for reference pictures
            //
            // Match C++ pattern: BeginVideoCoding gets only the ref slots (same as DecodeVideo),
            // NOT the setup picture. Setup picture is only in DecodeVideo's p_setup_reference_slot.
            let mut ref_infos: Vec<StdVideoDecodeH264ReferenceInfo> = Vec::new();
            for ref_pic in dpb_ref_pictures.iter() {
                let mut ref_info = unsafe { std::mem::zeroed::<StdVideoDecodeH264ReferenceInfo>() };
                ref_info.FrameNum = (ref_pic.frame_num % max_frame_num) as u16;
                ref_info.PicOrderCnt = ref_pic.pic_order_cnt;
                // For progressive frames: both fields are available for prediction
                ref_info.flags.set_top_field_flag(1);
                ref_info.flags.set_bottom_field_flag(1);
                ref_infos.push(ref_info);
            }
            let ref_dpb_slot_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = ref_infos
                .iter()
                .map(|ref_info| {
                    vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(ref_info)
                })
                .collect();

            // Build reference slots with DPB slot info in pNext chain
            // These are the SAME slots that will be used in DecodeVideo's p_reference_slots
            all_begin_slots = dpb_ref_pictures
                .iter()
                .zip(ref_dpb_slot_infos.iter())
                .map(|(ref_pic, dpb_slot_info)| vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: dpb_slot_info as *const _ as *const _,
                    slot_index: ref_pic.slot_index as i32,
                    p_picture_resource: &ref_pic.picture_resource as *const _,
                    _marker: Default::default(),
                })
                .collect();

            // C++ pattern: count = ref_count + (setup ? 1 : 0), ptr = refs when refs exist
            // Setup must be adjacent to refs in memory (driver reads count elements from ptr)
            //
            // CRITICAL: All structs must stay alive until after cmd_begin_video_coding!
            // We use Box::leak to ensure pointers remain valid.

            // Build setup DPB info (leaked to ensure pointer validity)
            let setup_ref_info: *const StdVideoDecodeH264ReferenceInfo =
                if dpb_setup_picture.is_some() {
                    let mut info = unsafe { std::mem::zeroed::<StdVideoDecodeH264ReferenceInfo>() };
                    info.FrameNum = (effective_frame_num % max_frame_num) as u16;
                    info.PicOrderCnt = effective_poc;
                    info.flags.set_top_field_flag(1);
                    info.flags.set_bottom_field_flag(1);
                    Box::leak(Box::new(info))
                } else {
                    std::ptr::null()
                };

            let setup_dpb_slot_info: *const vk::VideoDecodeH264DpbSlotInfoKHR =
                if !setup_ref_info.is_null() {
                    let dpb_info = vk::VideoDecodeH264DpbSlotInfoKHR::default()
                        .std_reference_info(unsafe { &*setup_ref_info });
                    Box::leak(Box::new(dpb_info))
                } else {
                    std::ptr::null()
                };

            // Build setup slot info (if setup picture exists)
            let setup_slot_info = if !setup_dpb_slot_info.is_null() {
                Some(vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: setup_dpb_slot_info as *const _ as *const _,
                    slot_index: dpb_setup_picture
                        .as_ref()
                        .map_or(0, |s| s.slot_index as i32),
                    p_picture_resource: dpb_setup_picture
                        .as_ref()
                        .map_or(std::ptr::null(), |s| &s.picture_resource as *const _),
                    _marker: Default::default(),
                })
            } else {
                None
            };

            // Build combined slots array for BeginVideoCoding
            // MUST keep this alive until after cmd_begin_video_coding is called!
            let begin_video_coding_slots: Vec<vk::VideoReferenceSlotInfoKHR> =
                if all_begin_slots.is_empty() {
                    // No refs: use only setup slot
                    setup_slot_info.into_iter().collect()
                } else {
                    // Has refs: combine refs + setup
                    let mut combined = all_begin_slots.clone();
                    if let Some(slot) = setup_slot_info {
                        combined.push(slot);
                    }
                    combined
                };

            let slot_count = begin_video_coding_slots.len() as u32;
            let slot_ptr = if begin_video_coding_slots.is_empty() {
                std::ptr::null()
            } else {
                begin_video_coding_slots.as_ptr()
            };

            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: slot_count,
                p_reference_slots: slot_ptr,
                _marker: Default::default(),
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);
        }

        // RESET decoder before first frame (required by Vulkan spec)
        // Must be INSIDE video coding block (after Begin, before Decode)
        if is_first_frame {
            self.cmd_control_video_coding(cmd_buffer);
        }

        // Barriers AFTER BeginVideoCoding and BEFORE DecodeVideo
        // This matches C++ reference VkVideoDecoder.cpp:1216-1227
        // Bitstream buffer barrier
        let buffer_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::NONE,
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
        // Use COLOR aspect (matches C++ reference VkVideoDecoder.cpp:857)
        // When dpb_setup_picture points to the same image as dstPictureResource,
        // use VIDEO_DECODE_DPB_KHR layout (per Vulkan spec).
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
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
            subresource_range,
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
                    subresource_range,
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
        // Each slot needs a VkVideoDecodeH264DpbSlotInfoKHR in its pNext chain
        let max_frame_num =
            1u32 << (self.sps.as_ref().expect("SPS").log2_max_frame_num_minus4 as u32 + 4);

        // Setup slot DPB info
        let mut setup_ref_info = unsafe { std::mem::zeroed::<StdVideoDecodeH264ReferenceInfo>() };
        setup_ref_info.FrameNum = (effective_frame_num % max_frame_num) as u16;
        setup_ref_info.PicOrderCnt = effective_poc;
        // For progressive frames: both fields are available for prediction
        setup_ref_info.flags.set_top_field_flag(1);
        setup_ref_info.flags.set_bottom_field_flag(1);
        let setup_dpb_slot_info = dpb_setup_picture.as_ref().map(|_| {
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_ref_info)
        });

        let setup_slot = dpb_setup_picture
            .as_ref()
            .map(|info| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: setup_dpb_slot_info
                    .as_ref()
                    .map(|s| s as *const _ as *const _)
                    .unwrap_or(std::ptr::null()),
                slot_index: info.slot_index as i32,
                p_picture_resource: &info.picture_resource as *const _,
                _marker: Default::default(),
            });

        // Reference slots with their DPB info - store everything in Vecs so pointers remain valid
        let mut ref_infos: Vec<StdVideoDecodeH264ReferenceInfo> = Vec::new();
        for ref_pic in dpb_ref_pictures.iter() {
            let mut ref_info = unsafe { std::mem::zeroed::<StdVideoDecodeH264ReferenceInfo>() };
            ref_info.FrameNum = (ref_pic.frame_num % max_frame_num) as u16;
            ref_info.PicOrderCnt = ref_pic.pic_order_cnt;
            // For progressive frames: both fields are available for prediction
            ref_info.flags.set_top_field_flag(1);
            ref_info.flags.set_bottom_field_flag(1);
            ref_infos.push(ref_info);
        }

        // Build DPB slot infos first (so pointers remain valid)
        let ref_dpb_slot_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR> = ref_infos
            .iter()
            .map(|ref_info| {
                vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(ref_info)
            })
            .collect();

        let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = dpb_ref_pictures
            .iter()
            .zip(ref_dpb_slot_infos.iter())
            .map(|(ref_pic, dpb_slot_info)| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: dpb_slot_info as *const _ as *const _,
                slot_index: ref_pic.slot_index as i32,
                p_picture_resource: &ref_pic.picture_resource as *const _,
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
            dst_picture_resource,
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
        is_idr: bool,
    ) -> StdVideoDecodeH264PictureInfo {
        let sps = self.sps.as_ref().expect("H264 SPS not set before decode");
        let pps = self.pps.as_ref().expect("H264 PPS not set before decode");

        let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
        let effective_frame_num = frame_num % max_frame_num;

        let mut pic_info = unsafe { std::mem::zeroed::<StdVideoDecodeH264PictureInfo>() };
        pic_info.frame_num = effective_frame_num as u16;
        pic_info.PicOrderCnt = pic_order_cnt;
        pic_info.seq_parameter_set_id = sps.seq_parameter_set_id as u8;
        pic_info.pic_parameter_set_id = pps.pic_parameter_set_id as u8;

        // Set flags based on frame properties
        pic_info.flags.set_is_intra(if is_intra { 1 } else { 0 });
        pic_info
            .flags
            .set_is_reference(if is_reference { 1 } else { 0 });
        pic_info.flags.set_field_pic_flag(0);
        pic_info.flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
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

    // Helper: dispatch cmdBeginVideoCodingKHR (void return)
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
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    // Helper: dispatch cmdDecodeVideoKHR
    fn cmd_decode_video(&self, cmd_buffer: vk::CommandBuffer, info: &vk::VideoDecodeInfoKHR<'_>) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), c"vkCmdDecodeVideoKHR".as_ptr())
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
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, &end_coding_info);
            }
        }
    }

    // Helper: dispatch cmdControlVideoCodingKHR (for RESET)
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
        }
    }
}

/// Convert our H264Sps to StdVideoH264SequenceParameterSet.
pub fn convert_h264_sps(sps: &vk_video_core::picture::H264Sps) -> StdVideoH264SequenceParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH264SpsFlags>() };
    flags.set_separate_colour_plane_flag(if sps.separate_colour_plane_flag { 1 } else { 0 });
    flags.set_qpprime_y_zero_transform_bypass_flag(if sps.qpprime_y_zero_transform_bypass_flag {
        1
    } else {
        0
    });
    flags.set_frame_mbs_only_flag(if sps.frame_mbs_only_flag { 1 } else { 0 });
    flags.set_direct_8x8_inference_flag(if sps.direct_8x8_inference_flag { 1 } else { 0 });
    flags.set_frame_cropping_flag(if sps.frame_cropping_flag { 1 } else { 0 });
    flags.set_vui_parameters_present_flag(if sps.vui_parameters_present_flag {
        1
    } else {
        0
    });

    // Convert VUI parameters if present
    let vui_data = if let Some(vui) = &sps.vui {
        let mut vui_flags = unsafe { std::mem::zeroed::<StdVideoH264SpsVuiFlags>() };
        vui_flags.set_aspect_ratio_info_present_flag(if vui.aspect_ratio_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_overscan_info_present_flag(if vui.overscan_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_overscan_appropriate_flag(if vui.overscan_appropriate_flag { 1 } else { 0 });
        vui_flags.set_video_signal_type_present_flag(if vui.video_signal_type_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_video_full_range_flag(if vui.video_full_range_flag { 1 } else { 0 });
        vui_flags.set_color_description_present_flag(if vui.color_description_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_chroma_loc_info_present_flag(if vui.chroma_loc_info_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_timing_info_present_flag(if vui.timing_info_present_flag { 1 } else { 0 });
        vui_flags.set_fixed_frame_rate_flag(if vui.fixed_frame_rate_flag { 1 } else { 0 });
        vui_flags.set_bitstream_restriction_flag(if vui.bitstream_restriction_flag {
            1
        } else {
            0
        });
        vui_flags.set_nal_hrd_parameters_present_flag(if vui.nal_hrd_parameters_present_flag {
            1
        } else {
            0
        });
        vui_flags.set_vcl_hrd_parameters_present_flag(if vui.vcl_hrd_parameters_present_flag {
            1
        } else {
            0
        });

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
pub fn convert_h264_pps(pps: &vk_video_core::picture::H264Pps) -> StdVideoH264PictureParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH264PpsFlags>() };
    flags.set_weighted_pred_flag(if pps.weighted_pred_flag { 1 } else { 0 });
    flags.set_deblocking_filter_control_present_flag(
        if pps.deblocking_filter_control_present_flag {
            1
        } else {
            0
        },
    );
    flags.set_redundant_pic_cnt_present_flag(if pps.redundant_pic_cnt_present_flag {
        1
    } else {
        0
    });
    flags.set_transform_8x8_mode_flag(if pps.transform_8x8_mode_flag { 1 } else { 0 });
    flags.set_constrained_intra_pred_flag(if pps.constrained_intra_pred_flag {
        1
    } else {
        0
    });

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
