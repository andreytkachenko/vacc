//! High-level video decoder that wraps H.264/H.265 Vulkan decode.

use ash::vk::{self, Handle};
use std::ffi::CString;

use super::{
    access_unit::{AccessUnit, H264OrH265Pps, H264OrH265Sps, VideoCodec as AccessUnitCodec},
    buffer::BitstreamBuffer,
    device::{VideoCodec, VulkanDevice},
    dpb::{DpbEntry, DpbManager, LastAccessType},
    h264::{H264Decoder, H264DpbRefPicture, H264SetupPictureInfo},
    h265::H265Decoder,
    profile_chain::{create_bitstream_buffer_with_profile, create_output_image_with_profile},
    readback::DecodedPixels,
    session::{CodecProfileInfo, VideoSession, VideoSessionParams, VideoSessionParameters},
    VideoError, VideoResult,
};

/// Decoded frame with metadata for presentation ordering.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub poc: i32,
    pub frame_num: u32,
    pub is_idr: bool,
    pub is_reference: bool,
    pub pixels: DecodedPixels,
    pub coded_width: u32,
    pub coded_height: u32,
}

/// Parsed bitstream information.
#[derive(Debug, Clone)]
pub struct ParsedInfo {
    pub vps: Option<vk_video_core::picture::H265Vps>,
    pub sps: Option<H264OrH265Sps>,
    pub pps: Option<H264OrH265Pps>,
    pub coded_width: u32,
    pub coded_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub crop_left: u32,
    pub crop_top: u32,
    pub profile_idc: u32,
    pub max_dpb_slots: u32,
    pub chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    pub luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    pub chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
}

/// High-level video decoder for H.264 and H.265.
pub struct VideoDecoder {
    vulkan: VulkanDevice,
    codec: VideoCodec,
    decoded_codec: AccessUnitCodec,
    parsed: ParsedInfo,
    session: VideoSession,
    session_params: VideoSessionParameters,
    session_memories: Vec<vk::DeviceMemory>,
    decode_queue_family: u32,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    bs_buffer: BitstreamBuffer,
    coded_extent: vk::Extent2D,
    dpb_manager: DpbManager,
    dpb_views: Vec<vk::ImageView>,
    dpb_images: Vec<vk::Image>,
    dpb_memories: Vec<vk::DeviceMemory>,
    output_format: vk::Format,
    bitstream_data: Vec<u8>,
    /// Persistent storage for codec-specific Vulkan structs.
    h265_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH265PictureInfo>,
    h265_decode_info_vec: Vec<vk::VideoDecodeH265PictureInfoKHR<'static>>,
    h265_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo>,
    h265_dpb_slot_info_vec: Vec<vk::VideoDecodeH265DpbSlotInfoKHR<'static>>,
    h264_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH264PictureInfo>,
    h264_decode_info_vec: Vec<vk::VideoDecodeH264PictureInfoKHR<'static>>,
    h264_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo>,
    h264_dpb_slot_info_vec: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'static>>,
    decoder_reset_done: bool,
    /// Bitstream buffer size alignment from device capabilities.
    bs_buffer_size_alignment: u64,
    /// Maximum coded extent supported by the device for this codec/profile.
    max_coded_extent: vk::Extent2D,
}

impl VideoDecoder {
    /// Create a new video decoder from bitstream data.
    pub fn new(data: Vec<u8>, max_frames: usize) -> VideoResult<Self> {
        let decoded_codec = detect_codec_from_data(&data);

        let parsed = match decoded_codec {
            AccessUnitCodec::H264 => parse_h264(&data)?,
            AccessUnitCodec::H265 => parse_h265(&data)?,
        };

        if parsed.coded_width == 0 || parsed.coded_height == 0 {
            return Err(VideoError::DecoderInit("Failed to parse video dimensions".to_string()));
        }

        let vulkan = super::VideoDeviceBuilder::new()
            .with_validation(true)
            .build()?;

        let decode_queue_family = vulkan
            .queue_families
            .video_decode
            .ok_or_else(|| VideoError::VideoNotSupported("No decode queue".to_string()))?;

        let coded_extent = vk::Extent2D {
            width: parsed.coded_width,
            height: parsed.coded_height,
        };
        eprintln!("[VideoDecoder] coded_extent: {}x{}", coded_extent.width, coded_extent.height);

        let session_dpb_slots = parsed.max_dpb_slots.min(4) + 1;

        let codec = match decoded_codec {
            AccessUnitCodec::H264 => VideoCodec::DecodeH264,
            AccessUnitCodec::H265 => VideoCodec::DecodeH265,
        };

        // Query device capabilities for this codec/profile
        eprintln!("[VideoDecoder] codec={:?}, profile_idc={}, chroma={:?}, luma={:?}, chroma_bd={:?}",
                  codec, parsed.profile_idc, parsed.chroma_subsampling, parsed.luma_bit_depth, parsed.chroma_bit_depth);
        let caps = vulkan.query_video_capabilities(
            codec,
            parsed.profile_idc,
            parsed.chroma_subsampling,
            parsed.luma_bit_depth,
            parsed.chroma_bit_depth,
        )?;

        // Validate stream dimensions against hardware limits
        if coded_extent.width > caps.max_coded_extent.width
            || coded_extent.height > caps.max_coded_extent.height
        {
            return Err(VideoError::DecoderInit(format!(
                "Stream resolution {}x{} exceeds hardware max_coded_extent {}x{}",
                coded_extent.width,
                coded_extent.height,
                caps.max_coded_extent.width,
                caps.max_coded_extent.height
            )));
        }

        // Extract owned values from capabilities before moving vulkan
        let bs_buffer_size_alignment = caps.min_bitstream_buffer_size_alignment;
        let max_coded_extent = caps.max_coded_extent;

        let (session, session_params, session_memories) = create_video_session(
            &vulkan,
            codec,
            &parsed,
            coded_extent,
            session_dpb_slots,
        )?;

        let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

        let max_au_size = extract_max_au_size(&data, decoded_codec, max_frames, &parsed);

        let bs_buffer = create_bitstream_buffer_with_profile(
            &vulkan.device,
            &vulkan.memory_properties,
            max_au_size as u64,
            codec,
            parsed.profile_idc,
            parsed.chroma_subsampling,
            parsed.luma_bit_depth,
            parsed.chroma_bit_depth,
        )?;

        let command_pool = create_command_pool(&vulkan.device, decode_queue_family)?;
        let fence = create_fence(&vulkan.device)?;

        let dpb_slots = parsed.max_dpb_slots.min(4);
        let mut dpb_views = Vec::new();
        let mut dpb_images = Vec::new();
        let mut dpb_memories = Vec::new();

        for _ in 0..dpb_slots {
            let (img, view, mem) = create_output_image_with_profile(
                &vulkan.device,
                &vulkan.memory_properties,
                coded_extent.width,
                coded_extent.height,
                output_format,
                codec,
                parsed.profile_idc,
                parsed.chroma_subsampling,
                parsed.luma_bit_depth,
                parsed.chroma_bit_depth,
            )?;
            dpb_views.push(view);
            dpb_images.push(img);
            dpb_memories.push(mem);
        }

        let mut dpb_manager = DpbManager::new(session_dpb_slots);

        if let Some(H264OrH265Sps::H264(sps)) = &parsed.sps {
            dpb_manager.set_max_num_ref_frames(sps.max_num_ref_frames);
            dpb_manager.set_max_frame_num(1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4));
        }

        Ok(Self {
            vulkan,
            codec,
            decoded_codec,
            parsed,
            session,
            session_params,
            session_memories,
            decode_queue_family,
            command_pool,
            fence,
            bs_buffer,
            coded_extent,
            dpb_manager,
            dpb_views,
            dpb_images,
            dpb_memories,
            output_format,
            h265_pic_info_vec: Vec::new(),
            h265_decode_info_vec: Vec::new(),
            h265_ref_info_vec: Vec::new(),
            h265_dpb_slot_info_vec: Vec::new(),
            h264_pic_info_vec: Vec::new(),
            h264_decode_info_vec: Vec::new(),
            h264_ref_info_vec: Vec::new(),
              h264_dpb_slot_info_vec: Vec::new(),
              decoder_reset_done: false,
              bitstream_data: data,
              bs_buffer_size_alignment,
              max_coded_extent,
          })
      }

      /// Decode all frames from the bitstream.
    ///
    /// Returns frames in decoding order. Use `reorder_to_presentation` to
    /// reorder by presentation order (POC).
    pub fn decode_all(&mut self, max_frames: usize) -> VideoResult<Vec<DecodedFrame>> {
        let access_units = super::access_unit::extract_all_access_units(
            self.bitstream_data(),
            self.decoded_codec,
            max_frames,
            self.parsed.sps.as_ref(),
            self.parsed.pps.as_ref(),
        );

        if access_units.is_empty() {
            return Err(VideoError::DecoderInit("No access units found".to_string()));
        }

        let mut frames = Vec::new();
        let mut is_first_frame = true;

        for (frame_idx, au) in access_units.iter().enumerate().take(max_frames) {
            eprintln!("[DEBUG] === Decoding frame {} ===", frame_idx);
            eprintln!("[DEBUG]   POC={}, frame_num={}, is_idr={}, is_ref={}, adaptive_ref_pic_marking={}, mmco_count={}",
                      au.pic_order_cnt[0], au.frame_num, au.is_idr, au.is_reference,
                      au.adaptive_ref_pic_marking_mode_flag, au.mmco_commands.len());
            if !au.mmco_commands.is_empty() {
                eprintln!("[DEBUG]   MMCO commands: {:?}", au.mmco_commands);
            }
            self.bs_buffer.write(&au.data)?;

            let alignment = self.bs_buffer_size_alignment.max(1);
            let aligned_size = ((au.data.len() as u64 + alignment - 1) & !(alignment - 1)).max(alignment);
            let padding_start = au.data.len() as u64;
            let padding_size = aligned_size - padding_start;
            if padding_size > 0 {
                self.bs_buffer.zero_range(padding_start, padding_size);
            }
            self.bs_buffer.flush_range(0, aligned_size).ok();

            let output_slot = if au.is_idr {
                self.dpb_manager.invalidate_all();
                0
            } else {
                // For H.264: ref_pocs from access_unit is empty (no RPS concept).
                // Use all valid DPB entries as protected references since any could be needed.
                // For H.265: use ref_pocs from RPS parsing.
                let protected_pocs: Vec<i32> = if self.decoded_codec == AccessUnitCodec::H264 {
                    self.dpb_manager
                        .entries
                        .iter()
                        .filter(|e| e.is_valid)
                        .flat_map(|e| {
                            if e.pic_order_cnt[0] == e.pic_order_cnt[1] {
                                vec![e.pic_order_cnt[0]]
                            } else {
                                vec![e.pic_order_cnt[0], e.pic_order_cnt[1]]
                            }
                        })
                        .collect()
                } else {
                    au.ref_pocs.clone()
                };
                eprintln!("[DEBUG]   DPB state before find_or_recycle_slot: protected_pocs={:?}", protected_pocs);
                for (i, e) in self.dpb_manager.entries.iter().enumerate() {
                    eprintln!("[DEBUG]     slot {}: valid={}, frame_num={}, poc={:?}", i, e.is_valid, e.frame_num, e.pic_order_cnt);
                }
                self.dpb_manager
                    .find_or_recycle_slot(&protected_pocs)
                    .unwrap_or(0)
            };

            let output_view = self.dpb_views[output_slot as usize];
            let output_img = self.dpb_images[output_slot as usize];
            eprintln!("[DEBUG]   output_slot={}, output_view={:?}, output_img={:?}", output_slot, output_view, output_img);

            let actual_bs_size = aligned_size;

            eprintln!("[DEBUG]   Calling record_decode_command...");
            self.record_decode_command(
                au,
                output_view,
                output_img,
                actual_bs_size,
                output_slot,
                is_first_frame,
            )?;
            eprintln!("[DEBUG]   record_decode_command completed");

            if is_first_frame {
                is_first_frame = false;
                self.decoder_reset_done = true;
            }

            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
            self.dpb_manager
                .set_slot_last_access(output_slot, LastAccessType::DecodeWrite);

            // Always update DPB entry for reference frames, regardless of MMCO flag.
            // When adaptive_ref_pic_marking_mode_flag is true, MMCO commands are present
            // in the bitstream and take precedence over sliding window.
            // When false, sliding window handles cleanup.
            if au.is_reference {
                eprintln!("[DEBUG]   Updating DPB entry for slot {}: frame_num={}, poc={:?}, adaptive_ref_pic_marking={}",
                          output_slot, au.frame_num, au.pic_order_cnt, au.adaptive_ref_pic_marking_mode_flag);
                self.dpb_manager.entries[output_slot as usize] = DpbEntry {
                    frame_num: au.frame_num,
                    pic_order_cnt: au.pic_order_cnt,
                    slot_index: output_slot,
                    is_valid: true,
                    image_view: self.dpb_views[output_slot as usize],
                    image: self.dpb_images[output_slot as usize],
                    current_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    last_access: LastAccessType::DecodeWrite,
                };

                // Apply reference picture marking AFTER updating the current frame's DPB entry.
                // When adaptive_ref_pic_marking_mode_flag is true, use MMCO commands.
                // When false, use sliding window.
                if au.adaptive_ref_pic_marking_mode_flag && !au.mmco_commands.is_empty() {
                    eprintln!("[DEBUG]   Applying MMCO commands ({} commands)", au.mmco_commands.len());
                    self.dpb_manager.apply_mmco(au.frame_num, output_slot, &au.mmco_commands);
                } else {
                    eprintln!("[DEBUG]   Applying sliding window");
                    self.dpb_manager.apply_sliding_window(au.frame_num);
                }

                eprintln!("[DEBUG]   DPB state after marking:");
                for (i, e) in self.dpb_manager.entries.iter().enumerate() {
                    eprintln!("[DEBUG]     slot {}: valid={}, frame_num={}, poc={:?}", i, e.is_valid, e.frame_num, e.pic_order_cnt);
                }
            } else {
                eprintln!("[DEBUG]   NOT updating DPB entry: is_ref={}", au.is_reference);
            }

            eprintln!("[DEBUG]   Calling readback_decoded_image for output_img={:?}", output_img);
            let pixels = super::readback::readback_decoded_image(
                &self.vulkan.instance,
                &self.vulkan.device,
                &self.vulkan.memory_properties,
                self.decode_queue_family,
                self.command_pool,
                self.fence,
                output_img,
                self.coded_extent.width,
                self.coded_extent.height,
            )?;
            eprintln!("[DEBUG]   readback done, Y plane bytes={}, U plane bytes={}", pixels.y_plane.len(), pixels.u_plane.len());

            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);
            self.dpb_manager
                .set_slot_last_access(output_slot, LastAccessType::TransferRead);

            frames.push(DecodedFrame {
                poc: au.pic_order_cnt[0],
                frame_num: au.frame_num,
                is_idr: au.is_idr,
                is_reference: au.is_reference,
                pixels,
                coded_width: self.coded_extent.width,
                coded_height: self.coded_extent.height,
            });
        }

        Ok(frames)
    }

    fn bitstream_data(&self) -> &[u8] {
        &self.bitstream_data
    }

    fn record_decode_command(
        &mut self,
        au: &AccessUnit,
        output_view: vk::ImageView,
        output_img: vk::Image,
        bs_size: u64,
        output_slot: u32,
        is_first_frame: bool,
    ) -> VideoResult<()> {
        eprintln!("[DEBUG] record_decode_command: codec={:?}", self.codec);
        let cmd_buffer = allocate_command_buffer(&self.vulkan.device, self.command_pool)?;
        eprintln!("[DEBUG] record_decode_command: allocated cmd_buffer={:?}", cmd_buffer);

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.vulkan
                .device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;
        }

        match self.codec {
            VideoCodec::DecodeH264 => {
                self.record_h264_decode(cmd_buffer, au, output_view, output_img, bs_size, output_slot, is_first_frame)?;
            }
            VideoCodec::DecodeH265 => {
                self.record_h265_decode(cmd_buffer, au, output_view, output_img, bs_size, output_slot, is_first_frame)?;
            }
            _ => {}
        }

        unsafe {
            self.vulkan
                .device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;
            eprintln!("[DEBUG] record_decode_command: command buffer ended");

            self.vulkan
                .device
                .reset_fences(&[self.fence])
                .map_err(|e| VideoError::FenceWait(e.to_string()))?;

            let queue = self.vulkan.device.get_device_queue(self.decode_queue_family, 0);
            self.vulkan
                .device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                    self.fence,
                )
                .map_err(|e| VideoError::QueueSubmission(e.to_string()))?;
            eprintln!("[DEBUG] record_decode_command: queue_submit completed");

            self.vulkan
                .device
                .wait_for_fences(&[self.fence], true, 10_000_000_000)
                .map_err(|e| VideoError::FenceWait(e.to_string()))?;
            eprintln!("[DEBUG] record_decode_command: wait_for_fences completed");
        }

        Ok(())
    }

    fn record_h264_decode(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        au: &AccessUnit,
        output_view: vk::ImageView,
        output_img: vk::Image,
        bs_size: u64,
        _output_slot: u32,
        is_first_frame: bool,
    ) -> VideoResult<()> {
        eprintln!("[DEBUG] record_h264_decode: frame_num={}, slice_type={}, bs_size={}", au.frame_num, au.slice_type, bs_size);
        eprintln!("[DEBUG] record_h264_decode: output_view={:?}, output_img={:?}", output_view, output_img);
        let sps = match &self.parsed.sps {
            Some(H264OrH265Sps::H264(s)) => s,
            _ => return Err(VideoError::DecoderInit("H264 SPS not found".to_string())),
        };
        let pps = match &self.parsed.pps {
            Some(H264OrH265Pps::H264(p)) => p,
            _ => return Err(VideoError::DecoderInit("H264 PPS not found".to_string())),
        };

        let mut h264_decoder = H264Decoder::new(
            self.vulkan.device.clone(),
            self.vulkan.instance.clone(),
        );
        h264_decoder.set_sps(sps.clone());
        h264_decoder.set_pps(pps.clone());

        let dpb_ref_pictures: Vec<H264DpbRefPicture<'_>> = self
            .dpb_manager
            .get_references()
            .iter()
            .filter(|e| e.slot_index != _output_slot)
            .map(|entry| H264DpbRefPicture {
                slot_index: entry.slot_index,
                picture_resource: vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent: self.coded_extent,
                    base_array_layer: 0,
                    image_view_binding: entry.image_view,
                    _marker: Default::default(),
                },
                image: entry.image,
                frame_num: entry.frame_num,
                pic_order_cnt: entry.pic_order_cnt,
                current_layout: entry.current_layout,
                last_access: entry.last_access,
            })
            .collect();
        eprintln!("[DEBUG] record_h264_decode: using {} refs:", dpb_ref_pictures.len());
        for r in &dpb_ref_pictures {
            eprintln!("[DEBUG]   ref slot={}, frame_num={}, poc={:?}, layout={:?}, last_access={:?}",
                      r.slot_index, r.frame_num, r.pic_order_cnt, r.current_layout, r.last_access);
        }

        let setup_picture = Some(H264SetupPictureInfo {
            slot_index: _output_slot,
            picture_resource: vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent: self.coded_extent,
                base_array_layer: 0,
                image_view_binding: output_view,
                _marker: Default::default(),
            },
        });

        h264_decoder.record_decode_command(
            cmd_buffer,
            self.session.handle(),
            self.session_params.handle(),
            self.bs_buffer.buffer(),
            0,
            bs_size,
            output_view,
            output_img,
            self.coded_extent,
            setup_picture,
            &dpb_ref_pictures,
            &au.slice_offsets,
            Some(au.frame_num),
            Some(au.pic_order_cnt),
             Some(matches!(au.slice_type, 0 | 4 | 8)), // I or SI slices are intra
            Some(au.is_reference),
            Some(au.is_idr),
            is_first_frame,
            self.parsed.max_dpb_slots,
            &self.dpb_images,
            &self.dpb_views,
        )?;

        Ok(())
    }

    fn record_h265_decode(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        au: &AccessUnit,
        output_view: vk::ImageView,
        output_img: vk::Image,
        bs_size: u64,
        output_slot: u32,
        is_first_frame: bool,
    ) -> VideoResult<()> {
        eprintln!("[DEBUG] record_h265_decode: frame_num={}, slice_type={}, bs_size={}", au.frame_num, au.slice_type, bs_size);
        eprintln!("[DEBUG] record_h265_decode: output_view={:?}, output_img={:?}", output_view, output_img);
        let vps = &self.parsed.vps;
        let sps = match &self.parsed.sps {
            Some(H264OrH265Sps::H265(s)) => s,
            _ => return Err(VideoError::DecoderInit("H265 SPS not found".to_string())),
        };
        let pps = match &self.parsed.pps {
            Some(H264OrH265Pps::H265(p)) => p,
            _ => return Err(VideoError::DecoderInit("H265 PPS not found".to_string())),
        };

        let mut h265_decoder = H265Decoder::new(
            self.vulkan.device.clone(),
            self.vulkan.instance.clone(),
        );
        if let Some(vps) = vps {
            h265_decoder.set_vps(vps.clone());
        }
        h265_decoder.set_sps(sps.clone());
        h265_decoder.set_pps(pps.clone());

        let dpb_ref_pictures: Vec<super::h265::H265RefPictureInfo> = self
            .dpb_manager
            .get_references()
            .iter()
            .filter(|e| e.slot_index != output_slot)
            .map(|entry| super::h265::H265RefPictureInfo {
                slot_index: entry.slot_index,
                pic_order_cnt: entry.pic_order_cnt[0],
                picture_resource: vk::VideoPictureResourceInfoKHR {
                    s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                    p_next: std::ptr::null(),
                    coded_offset: vk::Offset2D::default(),
                    coded_extent: self.coded_extent,
                    base_array_layer: 0,
                    image_view_binding: entry.image_view,
                    _marker: Default::default(),
                },
                image: entry.image,
                current_layout: entry.current_layout,
            })
            .collect();

        let setup_picture = super::h265::H265RefPictureInfo {
            slot_index: output_slot,
            pic_order_cnt: au.pic_order_cnt[0],
            picture_resource: vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent: self.coded_extent,
                base_array_layer: 0,
                image_view_binding: output_view,
                _marker: Default::default(),
            },
            image: output_img,
            current_layout: vk::ImageLayout::UNDEFINED,
        };

        h265_decoder.record_decode_command(
            cmd_buffer,
            self.session.handle(),
            self.session_params.handle(),
            self.bs_buffer.buffer(),
            0,
            bs_size,
            output_view,
            output_img,
            self.coded_extent,
            Some(setup_picture),
            &dpb_ref_pictures,
            &au.slice_offsets,
            Some(au.pic_order_cnt[0]),
            Some(au.is_idr || au.slice_type == 2),
            Some(au.is_reference),
            Some(au.is_idr),
        )?;

        Ok(())
    }
}

 impl Drop for VideoDecoder {
     fn drop(&mut self) {
         unsafe {
             self.vulkan
                 .device
                 .device_wait_idle()
                 .ok();

             // Destroy command resources first
             self.vulkan.device.destroy_command_pool(self.command_pool, None);
             self.vulkan.device.destroy_fence(self.fence, None);

             // Destroy DPB resources
             for ((img, view), mem) in self.dpb_images
                 .drain(..)
                 .zip(self.dpb_views.drain(..))
                 .zip(self.dpb_memories.drain(..))
             {
                 self.vulkan.device.destroy_image_view(view, None);
                 self.vulkan.device.destroy_image(img, None);
                 self.vulkan.device.free_memory(mem, None);
             }

             // Destroy bitstream buffer BEFORE device
             self.bs_buffer = BitstreamBuffer::null(&self.vulkan.device);

              // Destroy session resources
              destroy_session_parameters(
                  &self.vulkan.instance,
                  self.vulkan.device.handle(),
                  self.session_params.handle(),
              );
              destroy_session(
                  &self.vulkan.instance,
                  self.vulkan.device.handle(),
                  self.session.handle(),
              );

              for mem in self.session_memories.drain(..) {
                  self.vulkan.device.free_memory(mem, None);
              }

               // Destroy debug messenger BEFORE destroying device/instance
               if self.vulkan.has_validation && self.vulkan.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                   let debug_utils = ash::ext::debug_utils::Instance::new(&self.vulkan.entry, &self.vulkan.instance);
                   let _ = unsafe {
                       debug_utils.destroy_debug_utils_messenger(self.vulkan.debug_messenger, None);
                   };
                   self.vulkan.debug_messenger = vk::DebugUtilsMessengerEXT::null();
               }

              self.vulkan.device.destroy_device(None);
              self.vulkan.instance.destroy_instance(None);
         }
     }
 }

// ============================================================================
// Helper functions
// ============================================================================

fn detect_codec_from_data(data: &[u8]) -> AccessUnitCodec {
    // First pass: prioritize SPS/VPS detection for unambiguous identification
    // H.264: SPS=7, PPS=8; H.265: VPS=32, SPS=33, PPS=34
    for i in 0..data.len().min(4096) {
        let start = if i + 4 <= data.len() && data[i..i+4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i+3] == [0x00, 0x00, 0x01] {
            i + 3
        } else {
            continue;
        };
        if start >= data.len() {
            continue;
        }
        let nal_type = data[start] & 0x1F;
        // H.265 VPS/SPS/PPS are unambiguous
        if nal_type == 32 || nal_type == 33 || nal_type == 34 {
            return AccessUnitCodec::H265;
        }
        // H.264 SPS/PPS are unambiguous
        if nal_type == 7 || nal_type == 8 {
            return AccessUnitCodec::H264;
        }
    }

    // Second pass: detect from other NAL types
    for i in 0..data.len().min(1024) {
        let start = if i + 4 <= data.len() && data[i..i+4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i+3] == [0x00, 0x00, 0x01] {
            i + 3
        } else {
            continue;
        };
        if start >= data.len() {
            continue;
        }
        let nal_type = data[start] & 0x1F;
        // H.264: slice types (1-5), SEI (6), filler (9)
        if (1..=6).contains(&nal_type) || nal_type == 9 {
            return AccessUnitCodec::H264;
        }
        // H.265: TRAIL/RASL (0-1), TSA/Sample (2-3), STSA (4), RADL (5), RASL (6),
        //       BLA/IDR/CRA (16-21), VCL extension (12-15), etc.
        // Note: types 7-15 are H.265 non-VCL extension types
        if (0..=6).contains(&nal_type) || (12..=21).contains(&nal_type) {
            return AccessUnitCodec::H265;
        }
    }
    // Default to H.264 for legacy compatibility
    AccessUnitCodec::H264
}

fn parse_h264(data: &[u8]) -> VideoResult<ParsedInfo> {
    use vk_video_parser::{
        bitstream::BitstreamPacket, h264::H264Parser,
        DetectedVideoFormat, ParseResult, VideoParser,
    };

    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH264,
        ))
        .map_err(|e| VideoError::DecoderInit(e.to_string()))?;

    let packet = BitstreamPacket::new(data.to_vec());
    let mut sps: Option<vk_video_core::picture::H264Sps> = None;
    let mut pps: Option<vk_video_core::picture::H264Pps> = None;

    if let Ok(ParseResult::ParameterSet { sps: s, pps: p, .. }) = parser.parse(&packet) {
        if let Some(s) = s {
            sps = s.downcast_ref::<vk_video_core::picture::H264Sps>().cloned();
        }
        if let Some(p) = p {
            pps = p.downcast_ref::<vk_video_core::picture::H264Pps>().cloned();
        }
    }

    let coded_width = sps
        .as_ref()
        .map(|s| (s.pic_width_in_mbs_minus1 as u32 + 1) * 16)
        .unwrap_or(0);
    let coded_height = sps
        .as_ref()
        .map(|s| {
            if s.frame_mbs_only_flag {
                (s.pic_height_in_map_units_minus1 as u32 + 1) * 16
            } else {
                (s.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
            }
        })
        .unwrap_or(0);
    let raw_profile_idc = sps.as_ref().map(|s| s.profile_idc as u32).unwrap_or(100);
    let profile_idc = match raw_profile_idc {
        41 => 66,
        66 | 77 | 88 | 100 | 110 | 122 | 244 => raw_profile_idc,
        _ => 100,
    };
    let max_dpb_slots = 16;

    let chroma_subsampling = match sps.as_ref().map(|s| s.chroma_format_idc) {
        Some(0) => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
        Some(1) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        Some(2) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        Some(3) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        _ => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
    };

    let luma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_luma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    let chroma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_chroma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    Ok(ParsedInfo {
        vps: None,
        sps: sps.map(H264OrH265Sps::H264),
        pps: pps.map(H264OrH265Pps::H264),
        coded_width,
        coded_height,
        display_width: coded_width,
        display_height: coded_height,
        crop_left: 0,
        crop_top: 0,
        profile_idc,
        max_dpb_slots,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    })
}

fn parse_h265(data: &[u8]) -> VideoResult<ParsedInfo> {
    use vk_video_parser::{
        bitstream::BitstreamPacket, h265::H265Parser,
        DetectedVideoFormat, ParseResult, VideoParser,
    };

    let mut parser = H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH265,
        ))
        .map_err(|e| VideoError::DecoderInit(e.to_string()))?;

    let packet = BitstreamPacket::new(data.to_vec());
    let mut vps: Option<vk_video_core::picture::H265Vps> = None;
    let mut sps: Option<vk_video_core::picture::H265Sps> = None;
    let mut pps: Option<vk_video_core::picture::H265Pps> = None;

    if let Ok(ParseResult::ParameterSet {
        vps: v,
        sps: s,
        pps: p,
        ..
    }) = parser.parse(&packet)
    {
        if let Some(v) = v {
            vps = v.downcast_ref::<vk_video_core::picture::H265Vps>().cloned();
        }
        if let Some(s) = s {
            sps = s.downcast_ref::<vk_video_core::picture::H265Sps>().cloned();
        }
        if let Some(p) = p {
            pps = p.downcast_ref::<vk_video_core::picture::H265Pps>().cloned();
        }
    }

    let coded_width = sps
        .as_ref()
        .map(|s| ((s.pic_width_in_luma_samples as u32) + 15) & !15)
        .unwrap_or(0);
    let coded_height = sps
        .as_ref()
        .map(|s| ((s.pic_height_in_luma_samples as u32) + 15) & !15)
        .unwrap_or(0);

    let (display_width, display_height, crop_left, crop_top) = sps
        .as_ref()
        .map(|s| {
            let chroma_format_idc = s.chroma_format_idc;
            let (log2_sub_width_c, log2_sub_height_c) = match chroma_format_idc {
                0 => (0, 0),
                1 | 2 => (1, 1),
                _ => (0, 0),
            };
            let sub_width_c = 1u32 << log2_sub_width_c;
            let sub_height_c = 1u32 << log2_sub_height_c;

            let pic_width = s.pic_width_in_luma_samples as u32;
            let pic_height = s.pic_height_in_luma_samples as u32;

            let (crop_left, crop_top) = if s.conformance_window_flag {
                (
                    s.conf_win_left_offset * sub_width_c,
                    s.conf_win_top_offset * sub_height_c,
                )
            } else {
                (0, 0)
            };

            let display_width = if s.conformance_window_flag {
                let left_right = (s.conf_win_left_offset + s.conf_win_right_offset) * sub_width_c;
                pic_width.saturating_sub(left_right)
            } else {
                pic_width
            };

            let display_height = if s.conformance_window_flag {
                let top_bottom = (s.conf_win_top_offset + s.conf_win_bottom_offset) * sub_height_c;
                pic_height.saturating_sub(top_bottom)
            } else {
                pic_height
            };

            (display_width, display_height, crop_left, crop_top)
        })
        .unwrap_or((coded_width, coded_height, 0, 0));

    let profile_idc = 1;
    let max_dpb_slots = sps
        .as_ref()
        .map(|s| (s.max_num_ref_frames as u32).max(1))
        .unwrap_or(16)
        .max(4);

    let chroma_subsampling = match sps.as_ref().map(|s| s.chroma_format_idc) {
        Some(0) => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
        Some(1) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        Some(2) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
        Some(3) => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        _ => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
    };

    let luma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_luma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    let chroma_bit_depth = match sps.as_ref().map(|s| s.bit_depth_chroma_minus8) {
        Some(0) => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        Some(2) => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        Some(4) => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };

    Ok(ParsedInfo {
        vps,
        sps: sps.map(H264OrH265Sps::H265),
        pps: pps.map(H264OrH265Pps::H265),
        coded_width,
        coded_height,
        display_width,
        display_height,
        crop_left,
        crop_top,
        profile_idc,
        max_dpb_slots,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
    })
}

fn create_video_session(
    vulkan: &VulkanDevice,
    codec: VideoCodec,
    parsed: &ParsedInfo,
    coded_extent: vk::Extent2D,
    max_dpb_slots: u32,
) -> VideoResult<(VideoSession, VideoSessionParameters, Vec<vk::DeviceMemory>)> {
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let codec_profile_info = match codec {
        VideoCodec::DecodeH264 => CodecProfileInfo::H264 {
            std_profile_idc: parsed.profile_idc,
            picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE.as_raw(),
        },
        VideoCodec::DecodeH265 => CodecProfileInfo::H265 {
            std_profile_idc: parsed.profile_idc,
        },
        _ => return Err(VideoError::CodecNotSupported(format!("{:?}", codec))),
    };

    let session_params = VideoSessionParams {
        queue_family_index: vulkan.queue_families.video_decode.unwrap(),
        picture_format: output_format,
        reference_picture_format: output_format,
        max_coded_extent: coded_extent,
        max_dpb_slots,
        max_active_reference_pictures: max_dpb_slots,
        codec,
        codec_profile_info,
    };

    let std_header_version = build_std_header_version(match codec {
        VideoCodec::DecodeH264 => "VK_STD_vulkan_video_codec_h264_decode",
        VideoCodec::DecodeH265 => "VK_STD_vulkan_video_codec_h265_decode",
        _ => return Err(VideoError::CodecNotSupported(format!("{:?}", codec))),
    });

    let (session, session_memories) =
        VideoSession::create(&vulkan.instance, &vulkan.device, &session_params, &std_header_version)?;

    // Extract SPS/PPS/VPS for session parameters creation
    let h264_sps = match codec {
        VideoCodec::DecodeH264 => match &parsed.sps {
            Some(H264OrH265Sps::H264(s)) => Some(s),
            _ => None,
        },
        _ => None,
    };
    let h264_pps = match codec {
        VideoCodec::DecodeH264 => match &parsed.pps {
            Some(H264OrH265Pps::H264(p)) => Some(p),
            _ => None,
        },
        _ => None,
    };
    let h265_vps = match codec {
        VideoCodec::DecodeH265 => parsed.vps.as_ref(),
        _ => None,
    };
    let h265_sps = match codec {
        VideoCodec::DecodeH265 => match &parsed.sps {
            Some(H264OrH265Sps::H265(s)) => Some(s),
            _ => None,
        },
        _ => None,
    };
    let h265_pps = match codec {
        VideoCodec::DecodeH265 => match &parsed.pps {
            Some(H264OrH265Pps::H265(p)) => Some(p),
            _ => None,
        },
        _ => None,
    };

    let session_parameters = VideoSessionParameters::create(
        &vulkan.instance,
        &vulkan.device,
        session.handle(),
        codec,
        h264_sps,
        h264_pps,
        h265_vps,
        h265_sps,
        h265_pps,
    )?;

    // Initialize the session with the session parameters via vkUpdateVideoSessionKHR
    session_parameters.update_session(session.handle())?;

    Ok((session, session_parameters, session_memories))
}

fn build_std_header_version(extension_name: &str) -> vk::ExtensionProperties {
    let mut props = vk::ExtensionProperties::default();
    let bytes = format!("{}\0", extension_name).into_bytes();
    props
        .extension_name
        .iter_mut()
        .zip(bytes.iter())
        .for_each(|(c, &b)| *c = b as std::os::raw::c_char);
    props.spec_version = (1u32 << 22) | (0u32 << 12) | 0u32;
    props
}

fn create_command_pool(
    device: &ash::Device,
    queue_family: u32,
) -> VideoResult<vk::CommandPool> {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    unsafe {
        device
            .create_command_pool(&pool_create_info, None)
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))
    }
}

fn allocate_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> VideoResult<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    unsafe {
        let buffers = device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;
        Ok(buffers[0])
    }
}

fn create_fence(device: &ash::Device) -> VideoResult<vk::Fence> {
    unsafe {
        device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .map_err(|e| VideoError::FenceWait(e.to_string()))
    }
}

fn destroy_session_parameters(
    instance: &ash::Instance,
    device: vk::Device,
    session_params: vk::VideoSessionParametersKHR,
) {
    if session_params.is_null() {
        return;
    }
    if let Some(ptr) = unsafe {
        instance.get_device_proc_addr(
            device,
            b"vkDestroyVideoSessionParametersKHR\0".as_ptr().cast(),
        )
    } {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionParametersKHR,
                *const vk::AllocationCallbacks,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(device, session_params, std::ptr::null());
        }
    }
}

fn destroy_session(
    instance: &ash::Instance,
    device: vk::Device,
    session: vk::VideoSessionKHR,
) {
    if session.is_null() {
        return;
    }
    if let Some(ptr) = unsafe {
        instance.get_device_proc_addr(device, b"vkDestroyVideoSessionKHR\0".as_ptr().cast())
    } {
        unsafe {
            type FnType = unsafe extern "system" fn(
                vk::Device,
                vk::VideoSessionKHR,
                *const vk::AllocationCallbacks,
            );
            let f: FnType = std::mem::transmute(ptr);
            f(device, session, std::ptr::null());
        }
    }
}

fn extract_max_au_size(
    data: &[u8],
    codec: AccessUnitCodec,
    max_frames: usize,
    parsed: &ParsedInfo,
) -> usize {
    let access_units = super::access_unit::extract_all_access_units(
        data,
        codec,
        max_frames,
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    );

    access_units
        .iter()
        .map(|au| au.data.len())
        .max()
        .unwrap_or(0)
}
