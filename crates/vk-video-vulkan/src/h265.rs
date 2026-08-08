//! H.265/HEVC Vulkan video decoder.

use ash::vk;
use ash::vk::native::*;

use super::codec_types::*;
use super::{VideoError, VideoResult};
use super::dpb::LastAccessType;

use vk_video_core::picture::{H265Vps, H265Sps, H265Pps, H265ShortTermRefPicSet, H265SpsVui};

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
            update_sequence_count: 1, // First update must be 1
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
        coded_extent: vk::Extent2D,
        dpb_setup_picture: Option<H265RefPictureInfo<'a>>,
        dpb_ref_pictures: &[H265RefPictureInfo<'a>],
        slice_offsets: &[u32],
        pic_order_cnt: Option<i32>,
        is_intra: Option<bool>,
        is_reference: Option<bool>,
        is_idr: Option<bool>,
    ) -> VideoResult<()> {
        eprintln!("[DEBUG] H265Decoder::record_decode_command: bitstream_range={}, refs={}, slice_offsets={:?}", 
                  bitstream_range, dpb_ref_pictures.len(), slice_offsets);
        eprintln!("[DEBUG] H265Decoder::record_decode_command: output_view={:?}, output_img={:?}", 
                  output_image_view, output_image);
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

            // Second: create VkVideoDecodeH265DpbSlotInfoKHR for setup slot
            let setup_dpb_slot_info = setup_ref_std_info.as_ref().map(|ref_std_info| {
                vk::VideoDecodeH265DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: ref_std_info as *const _,
                    _marker: Default::default(),
                }
            });

            // Third: create VkVideoReferenceSlotInfoKHR for setup slot
            // Use actual slot index (matches C++ reference implementation)
            let setup_slot = dpb_setup_picture.as_ref().map(|info| {
                let pnext = setup_dpb_slot_info.as_ref().map_or(std::ptr::null(), |s| s as *const _ as *const _);
                vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: pnext,
                    slot_index: info.slot_index as i32, // Use actual slot index
                    p_picture_resource: &info.picture_resource as *const _,
                    _marker: Default::default(),
                }
            });

            // Fourth: create StdVideoDecodeH265ReferenceInfo for each ref picture
            let ref_std_infos: Vec<StdVideoDecodeH265ReferenceInfo> = dpb_ref_pictures
                .iter()
                .map(|info| {
                    let mut ref_std_info = unsafe { std::mem::zeroed::<StdVideoDecodeH265ReferenceInfo>() };
                    ref_std_info.PicOrderCntVal = info.pic_order_cnt;
                    ref_std_info.flags.set_used_for_long_term_reference(0);
                    ref_std_info.flags.set_unused_for_reference(0);
                    ref_std_info
                })
                .collect();

            // Fifth: create VkVideoDecodeH265DpbSlotInfoKHR for each ref picture
            let ref_dpb_slot_infos: Vec<vk::VideoDecodeH265DpbSlotInfoKHR> = ref_std_infos
                .iter()
                .map(|ref_std_info| {
                    vk::VideoDecodeH265DpbSlotInfoKHR {
                        s_type: vk::StructureType::VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
                        p_next: std::ptr::null(),
                        p_std_reference_info: ref_std_info as *const _,
                        _marker: Default::default(),
                    }
                })
                .collect();

            // Sixth: create VkVideoReferenceSlotInfoKHR for each ref picture
            let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = dpb_ref_pictures
                .iter()
                .zip(ref_dpb_slot_infos.iter())
                .map(|(info, dpb_slot_info)| {
                    vk::VideoReferenceSlotInfoKHR {
                        s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                        p_next: dpb_slot_info as *const _ as *const _,
                        slot_index: info.slot_index as i32,
                        p_picture_resource: &info.picture_resource as *const _,
                        _marker: Default::default(),
                    }
                })
                .collect();

            // BeginVideoCoding reference slots: Match C++ reference VkVideoDecoder.cpp:1079-1084
            //
            // CRITICAL: When refs exist, BeginVideoCoding uses the SAME slots as DecodeVideo's
            // p_reference_slots (NOT including the setup picture). The setup picture is only
            // in DecodeVideo's p_setup_reference_slot.
            //
            // C++ pattern:
            //   decodeBeginInfo.referenceSlotCount = decodeFrameInfo.referenceSlotCount +
            //       (decodeFrameInfo.pSetupReferenceSlot ? 1 : 0);
            //   decodeBeginInfo.pReferenceSlots = (decodeFrameInfo.referenceSlotCount > 0) ?
            //       decodeFrameInfo.pReferenceSlots : decodeFrameInfo.pSetupReferenceSlot;
            //
            // When refs exist: BeginVideoCoding uses refs only (same ptr as DecodeVideo refs).
            // When no refs: BeginVideoCoding uses only setup slot.
            let setup_slot_for_decode = setup_slot.clone();
            let (begin_slot_count, begin_slot_ptr) = if !ref_slots.is_empty() {
                // Has refs: use refs only for BeginVideoCoding (matches C++)
                (ref_slots.len() as u32, ref_slots.as_ptr())
            } else {
                // No refs: use only setup slot for BeginVideoCoding
                setup_slot.as_ref()
                    .map(|s| (1u32, s as *const _))
                    .unwrap_or((0u32, std::ptr::null()))
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

            eprintln!("[DEBUG] H265Decoder: BeginVideoCodingKHR: slot_count={}, has_refs={}, has_setup={}",
                      begin_slot_count, !dpb_ref_pictures.is_empty(), dpb_setup_picture.is_some());
            for (i, rs) in ref_slots.iter().enumerate() {
                if let Some(pr) = rs.p_picture_resource.as_ref() {
                    eprintln!("[DEBUG]   BeginVideoCoding slot[{}] index={}, view={:?}",
                              i, rs.slot_index, pr.image_view_binding);
                }
            }
            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);

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
                if ref_pic.image != vk::Image::null() && ref_pic.current_layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR {
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
            eprintln!("[DEBUG] H265Decoder: PipelineBarrier2: 1 buffer barrier, {} image barriers", all_image_barriers.len());
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

             let decode_info = vk::VideoDecodeInfoKHR {
                 s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
                 p_next: &h265_decode_info as *const _ as *const _,
                 flags: vk::VideoDecodeFlagsKHR::empty(),
                 src_buffer: bitstream_buffer,
                 src_buffer_offset: bitstream_offset,
                 src_buffer_range: bitstream_range,
                 dst_picture_resource: dst_picture_resource,
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

             // Comprehensive debug dump of VkVideoDecodeInfoKHR
             eprintln!("[DEBUG] H265Decoder: === VkVideoDecodeInfoKHR ===");
             eprintln!("[DEBUG]   src_buffer={:?}, offset={}, range={}",
                       decode_info.src_buffer, decode_info.src_buffer_offset, decode_info.src_buffer_range);
             eprintln!("[DEBUG]   dst_picture_resource.view={:?}",
                       decode_info.dst_picture_resource.image_view_binding);
             eprintln!("[DEBUG]   p_setup_reference_slot={:?}",
                       decode_info.p_setup_reference_slot);
             if let Some(ref s) = setup_slot_for_decode {
                 eprintln!("[DEBUG]     setup_slot.index={}, view={:?}",
                           s.slot_index, s.p_picture_resource.as_ref().map(|pr| pr.image_view_binding));
             }
             eprintln!("[DEBUG]   reference_slot_count={}", decode_info.reference_slot_count);
             eprintln!("[DEBUG]   p_reference_slots={:?}", decode_info.p_reference_slots);
             for (i, rs) in ref_slots.iter().enumerate() {
                 if let Some(pr) = rs.p_picture_resource.as_ref() {
                     eprintln!("[DEBUG]     ref_slot[{}].index={}, view={:?}",
                               i, rs.slot_index, pr.image_view_binding);
                 }
             }
             eprintln!("[DEBUG]   h265_decode_info.slice_segment_count={}", h265_decode_info.slice_segment_count);
            if let Some(sps) = &self.sps {
                eprintln!("[DEBUG]   pic_info.PicOrderCntVal={}, is_reference={}, is_idr={}",
                          pic_info.PicOrderCntVal,
                          pic_info.flags.IsReference(), pic_info.flags.IdrPicFlag());
            }
             eprintln!("[DEBUG] H265Decoder: ===============================");

             eprintln!("[DEBUG] H265Decoder: calling vkCmdDecodeVideoKHR");
             self.cmd_decode_video(cmd_buffer, &decode_info);
             eprintln!("[DEBUG] H265Decoder: vkCmdDecodeVideoKHR returned");

            // End video coding
            self.cmd_end_video_coding(cmd_buffer);
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

    fn cmd_control_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let coding_control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdControlVideoCodingKHR\0".as_ptr().cast(),
            )
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

    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };

        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                b"vkCmdEndVideoCodingKHR\0".as_ptr().cast(),
            )
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
            eprintln!("[H265] WARNING: Unknown level_idc={}, defaulting to 5.1", raw_level_idc);
            StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1
        }
    }
}

/// Convert H.265 SPS VUI to Vulkan StdVideoH265SequenceParameterSetVui.
/// This is critical for correct chroma output - the video_full_range_flag
/// tells the decoder whether to use full range (0-255) or limited range (16-235).
pub fn convert_h265_vui(vui: &H265SpsVui) -> StdVideoH265SequenceParameterSetVui {
    let mut vui_flags = unsafe { std::mem::zeroed::<StdVideoH265SpsVuiFlags>() };
    vui_flags.set_aspect_ratio_info_present_flag(if vui.aspect_ratio_info_present_flag { 1 } else { 0 });
    vui_flags.set_overscan_info_present_flag(if vui.overscan_info_present_flag { 1 } else { 0 });
    vui_flags.set_overscan_appropriate_flag(if vui.overscan_appropriate_flag { 1 } else { 0 });
    vui_flags.set_video_signal_type_present_flag(if vui.video_signal_type_present_flag { 1 } else { 0 });
    vui_flags.set_video_full_range_flag(if vui.video_full_range_flag { 1 } else { 0 });
    vui_flags.set_colour_description_present_flag(if vui.colour_description_present_flag { 1 } else { 0 });
    vui_flags.set_chroma_loc_info_present_flag(if vui.chroma_loc_info_present_flag { 1 } else { 0 });
    vui_flags.set_neutral_chroma_indication_flag(if vui.neutral_chroma_indication_flag { 1 } else { 0 });
    vui_flags.set_field_seq_flag(if vui.field_seq_flag { 1 } else { 0 });
    vui_flags.set_frame_field_info_present_flag(if vui.frame_field_info_present_flag { 1 } else { 0 });
    vui_flags.set_default_display_window_flag(if vui.default_display_window_flag { 1 } else { 0 });
    vui_flags.set_vui_timing_info_present_flag(if vui.vui_timing_info_present_flag { 1 } else { 0 });
    vui_flags.set_vui_poc_proportional_to_timing_flag(if vui.vui_poc_proportional_to_timing_flag { 1 } else { 0 });
    vui_flags.set_vui_hrd_parameters_present_flag(if vui.vui_hrd_parameters_present_flag { 1 } else { 0 });
    vui_flags.set_bitstream_restriction_flag(if vui.bitstream_restriction_flag { 1 } else { 0 });
    vui_flags.set_tiles_fixed_structure_flag(if vui.tiles_fixed_structure_flag { 1 } else { 0 });
    vui_flags.set_motion_vectors_over_pic_boundaries_flag(if vui.motion_vectors_over_pic_boundaries_flag { 1 } else { 0 });
    vui_flags.set_restricted_ref_pic_lists_flag(if vui.restricted_ref_pic_lists_flag { 1 } else { 0 });

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
    flags.set_pcm_enabled_flag(if sps.pcm_enabled_flag { 1 } else { 0 });
    flags.set_pcm_loop_filter_disabled_flag(
        if sps.pcm_loop_filter_disabled_flag { 1 } else { 0 },
    );
    flags.set_vui_parameters_present_flag(
        if sps.vui_parameters_present_flag { 1 } else { 0 },
    );
    flags.set_sps_extension_present_flag(
        if sps.sps_extension_present_flag { 1 } else { 0 },
    );
    flags.set_sps_range_extension_flag(
        if sps.sps_range_extension_flag { 1 } else { 0 },
    );
    flags.set_intra_smoothing_disabled_flag(
        if sps.intra_smoothing_disabled_flag { 1 } else { 0 },
    );
    flags.set_palette_mode_enabled_flag(
        if sps.palette_mode_enabled_flag { 1 } else { 0 },
    );

     // DecPicBufMgr - always set per C++ reference (VulkanH265Parser.cpp:499)
    let max_latency_increase_plus1: [u32; 7] = sps
        .max_latency_increase_plus1
        .map(|v| v as u32);
    let dec_pic_buf_mgr_data = StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: sps.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: sps.max_num_reorder_pics,
    };
    eprintln!("[H265-SPS] DecPicBufMgr: max_dec_pic_buffering_minus1={:?}, max_num_reorder_pics={:?}",
        sps.max_dec_pic_buffering_minus1, sps.max_num_reorder_pics);
    let dec_pic_buf_mgr = Box::leak(Box::new(dec_pic_buf_mgr_data));

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

    // ProfileTierLevel - REQUIRED by Vulkan spec
    // Use actual profile/level from parsed SPS, matching C++ reference
    let mut ptl = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevel>() };
    ptl.flags.set_general_tier_flag(if sps.tier_flag { 1 } else { 0 });
    ptl.general_profile_idc = sps.profile_idc as StdVideoH265ProfileIdc;
    ptl.general_level_idc = h265_level_idc_to_vulkan(sps.level_idc);
    eprintln!(
        "[H265 SPS convert] profile_idc={}, level_idc={}, vulkan_level={:?}",
        sps.profile_idc, sps.level_idc, ptl.general_level_idc
    );
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

pub fn convert_h265_pps(pps: &H265Pps) -> StdVideoH265PictureParameterSet {
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
     flags.set_uniform_spacing_flag(if pps.uniform_spacing_flag { 1 } else { 0 });
     flags.set_loop_filter_across_tiles_enabled_flag(
         if pps.loop_filter_across_tiles_enabled_flag { 1 } else { 0 },
     );
     flags.set_pps_loop_filter_across_slices_enabled_flag(
         if pps.pps_loop_filter_across_slices_enabled_flag { 1 } else { 0 },
     );
     flags.set_deblocking_filter_control_present_flag(
         if pps.deblocking_filter_control_present_flag { 1 } else { 0 },
     );
     flags.set_deblocking_filter_override_enabled_flag(
         if pps.deblocking_filter_override_enabled_flag { 1 } else { 0 },
     );
     flags.set_pps_deblocking_filter_disabled_flag(
         if pps.pps_deblocking_filter_disabled_flag { 1 } else { 0 },
     );
     flags.set_pps_scaling_list_data_present_flag(
         if pps.pps_scaling_list_data_present_flag { 1 } else { 0 },
     );
     flags.set_lists_modification_present_flag(
         if pps.lists_modification_present_flag { 1 } else { 0 },
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

pub fn convert_h265_vps(vps: &H265Vps) -> StdVideoH265VideoParameterSet {
    let mut flags = unsafe { std::mem::zeroed::<StdVideoH265VpsFlags>() };
    flags.set_vps_temporal_id_nesting_flag(if vps.vps_temporal_id_nesting_flag { 1 } else { 0 });
    flags.set_vps_sub_layer_ordering_info_present_flag(if vps.vps_sub_layer_ordering_info_present_flag { 1 } else { 0 });
    flags.set_vps_timing_info_present_flag(if vps.vps_timing_info_present_flag { 1 } else { 0 });

    // DecPicBufMgr - always set per C++ reference
    let max_latency_increase_plus1: [u32; 7] = vps
        .max_latency_increase_plus1
        .map(|v| v as u32);
    let mgr = StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: vps.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: vps.max_num_reorder_pics,
    };
    let dec_pic_buf_mgr = Box::leak(Box::new(mgr));

    // ProfileTierLevel - REQUIRED by Vulkan spec
    // Use actual profile/level from parsed VPS, matching C++ reference
    let mut ptl = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevel>() };
    ptl.flags.set_general_tier_flag(if vps.tier_flag { 1 } else { 0 });
    ptl.general_profile_idc = vps.profile_idc as StdVideoH265ProfileIdc;
    ptl.general_level_idc = h265_level_idc_to_vulkan(vps.level_idc);
    eprintln!(
        "[H265 VPS convert] profile_idc={}, level_idc={}, vulkan_level={:?}",
        vps.profile_idc, vps.level_idc, ptl.general_level_idc
    );
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
