//! AV1 Vulkan video decoder.

use ash::vk;
use ash::vk::native::*;

use super::codec_types::*;
use super::{VideoError, VideoResult};

/// AV1 decoder state.
pub struct Av1Decoder {
    device: ash::Device,
    instance: ash::Instance,
    /// Cached SPS for decode info construction.
    sps: Option<vk_video_core::picture::Av1Sps>,
    /// Frame counter.
    frame_count: u32,
}

impl Av1Decoder {
    pub fn new(device: ash::Device, instance: ash::Instance) -> Self {
        Self {
            device,
            instance,
            sps: None,
            frame_count: 0,
        }
    }

    pub fn set_sps(&mut self, sps: vk_video_core::picture::Av1Sps) {
        self.sps = Some(sps);
    }

    /// Update session parameters with SPS data.
    /// For AV1, the sequence header is passed directly in the create info,
    /// not via a separate add_info struct.
    pub fn update_session_parameters(
        &self,
        session_params: vk::VideoSessionParametersKHR,
        sps: Option<&vk_video_core::picture::Av1Sps>,
    ) -> VideoResult<()> {
        let _ = session_params;
        let _ = sps;
        // AV1 session parameters update is handled differently.
        // The sequence header is set at creation time.
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
    ) -> VideoResult<()> {
        let pic_info = self.build_picture_info(coded_extent);

        unsafe {
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
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
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

            // Build AV1 picture info
            let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: output_image_view,
                _marker: Default::default(),
            };

            let pic_ptr = &pic_info as *const StdVideoDecodeAV1PictureInfo;

            let av1_decode_info = vk::VideoDecodeAV1PictureInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_AV1_PICTURE_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_picture_info: pic_ptr,
                reference_name_slot_indices: [-1; vk::MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR],
                frame_header_offset: 0,
                tile_count: 0,
                p_tile_offsets: std::ptr::null(),
                p_tile_sizes: std::ptr::null(),
                _marker: Default::default(),
            };

            let decode_info = vk::VideoDecodeInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
                p_next: &av1_decode_info as *const _ as *const _,
                flags: vk::VideoDecodeFlagsKHR::empty(),
                src_buffer: bitstream_buffer,
                src_buffer_offset: bitstream_offset,
                src_buffer_range: bitstream_range,
                dst_picture_resource: dst_picture_resource,
                p_setup_reference_slot: std::ptr::null(),
                reference_slot_count: 0,
                p_reference_slots: std::ptr::null(),
                _marker: Default::default(),
            };

            self.cmd_decode_video(cmd_buffer, &decode_info);

            // End video coding
            self.cmd_end_video_coding(cmd_buffer);
        }

        self.frame_count += 1;
        Ok(())
    }

    fn build_picture_info(&self, coded_extent: vk::Extent2D) -> StdVideoDecodeAV1PictureInfo {
        let mut pic_info = unsafe { std::mem::zeroed::<StdVideoDecodeAV1PictureInfo>() };
        pic_info.frame_type = StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY;
        pic_info.current_frame_id = self.frame_count;
        pic_info.OrderHint = self.frame_count as u8;
        pic_info.primary_ref_frame = 0;
        pic_info.refresh_frame_flags = 1; // Refresh key frame
        pic_info.interpolation_filter = 0; // EIGHTTAP
        pic_info.TxMode = 0; // TX_MODE_ONLY_4X4
        pic_info.coded_denom = 1;
        pic_info.OrderHints = [self.frame_count as u8; 8];
        pic_info.expectedFrameId = [0; 8];
        pic_info.pTileInfo = std::ptr::null();
        pic_info.pQuantization = std::ptr::null();
        pic_info.pSegmentation = std::ptr::null();
        pic_info.pLoopFilter = std::ptr::null();
        pic_info.pCDEF = std::ptr::null();
        pic_info.pLoopRestoration = std::ptr::null();
        pic_info.pGlobalMotion = std::ptr::null();
        pic_info.pFilmGrain = std::ptr::null();
        pic_info
    }

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
