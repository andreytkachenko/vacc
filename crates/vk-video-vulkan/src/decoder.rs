//! High-level video decoder that wraps H.264/H.265/VP9/AV1 Vulkan decode.

use ash::vk::{self, Handle};

use super::{
    access_unit::{
        AccessUnit, ExtractedItem, H264OrH265Pps, H264OrH265Sps, H265VpsOpt,
        InBandParameterSet, VideoCodec as AccessUnitCodec,
    },
    av1::{
        Av1Decoder, Av1PictureInfoContainer, VideoDecodeAV1PictureInfoKHR,
    },
    buffer::BitstreamBuffer,
    device::{VideoCodec, VulkanDevice},
    dpb::{DpbEntry, DpbManager, LastAccessType},
    h264::{H264Decoder, H264DpbRefPicture, H264SetupPictureInfo},
    h265::H265Decoder,
    profile_chain::{
        create_bitstream_buffer_with_profile, create_dpb_image_array_with_profile,
        create_output_image_with_profile,
    },
    readback::DecodedPixels,
    session::{CodecProfileInfo, VideoSession, VideoSessionParameters, VideoSessionParams},
    vp9::{
        convert_vp9_picture_info, VideoDecodeVP9PictureInfoKHR, Vp9Decoder,
    },
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
    pub display_width: u32,
    pub display_height: u32,
    pub crop_left: u32,
    pub crop_top: u32,
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

/// High-level video decoder for H.264, H.265, VP9, and AV1.
pub struct VideoDecoder {
    vulkan: VulkanDevice,
    codec: VideoCodec,
    decoded_codec: AccessUnitCodec,
    parsed: ParsedInfo,
    session: VideoSession,
    session_params: Option<VideoSessionParameters>,
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
    /// True when the DPB is a SINGLE image with array layers (one layer per
    /// slot) instead of one image per slot. This is required when the device
    /// does not support `VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR`
    /// (VUID-VkVideoBeginCodingInfoKHR-flags-07244), matching the C++ reference
    /// `m_useImageArray` (VkVideoDecoder.cpp:349-353). When true, every
    /// `dpb_images[i]` is the same image handle and slot `i` lives in array
    /// layer `i`, so barriers/readback must use `base_array_layer = slot`.
    dpb_use_image_array: bool,
    #[allow(dead_code)]
    output_format: vk::Format,
    bitstream_data: Vec<u8>,
    /// Persistent storage for codec-specific Vulkan structs.
    #[allow(dead_code)]
    h265_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH265PictureInfo>,
    #[allow(dead_code)]
    h265_decode_info_vec: Vec<vk::VideoDecodeH265PictureInfoKHR<'static>>,
    #[allow(dead_code)]
    h265_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH265ReferenceInfo>,
    #[allow(dead_code)]
    h265_dpb_slot_info_vec: Vec<vk::VideoDecodeH265DpbSlotInfoKHR<'static>>,
    #[allow(dead_code)]
    h264_pic_info_vec: Vec<ash::vk::native::StdVideoDecodeH264PictureInfo>,
    #[allow(dead_code)]
    h264_decode_info_vec: Vec<vk::VideoDecodeH264PictureInfoKHR<'static>>,
    #[allow(dead_code)]
    h264_ref_info_vec: Vec<ash::vk::native::StdVideoDecodeH264ReferenceInfo>,
    #[allow(dead_code)]
    h264_dpb_slot_info_vec: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'static>>,
    decoder_reset_done: bool,
    /// Bitstream buffer size alignment from device capabilities.
    bs_buffer_size_alignment: u64,
    /// Picture access granularity for alignment.
    picture_access_granularity: vk::Extent2D,
    /// VP9-specific parser state
    vp9_parser: Option<vk_video_parser::vp9::Vp9Parser>,
    /// AV1-specific parser state
    av1_parser: Option<vk_video_parser::av1::Av1Parser>,
    /// AV1 sequence header (SPS)
    av1_sps: Option<vk_video_core::picture::Av1Sps>,
}

impl VideoDecoder {
    /// Base array layer to use for readback/barriers of DPB slot `slot`.
    /// In image-array mode slot `slot` lives in array layer `slot` of the
    /// shared image; in separate-image mode each slot is its own single-layer
    /// image, so the layer is always 0.
    fn dpb_base_layer(&self, slot: u32) -> u32 {
        if self.dpb_use_image_array {
            slot
        } else {
            0
        }
    }

    /// Create a new video decoder from bitstream data.
    pub fn new(data: Vec<u8>, max_frames: usize) -> VideoResult<Self> {
        let decoded_codec = detect_codec_from_data(&data);

        let (
            parsed,
            codec,
            vulkan,
            session_dpb_slots,
            _dpb_slots,
            coded_extent,
            av1_sps,
        ) =
            match decoded_codec {
                AccessUnitCodec::H264 => {
                    let parsed = parse_h264(&data)?;
                    if parsed.coded_width == 0 || parsed.coded_height == 0 {
                        return Err(VideoError::DecoderInit(
                            "Failed to parse video dimensions".to_string(),
                        ));
                    }
                    let vulkan = super::VideoDeviceBuilder::new()
                        .with_validation(false)
                        .build()?;
                    let coded_extent = vk::Extent2D {
                        width: parsed.coded_width,
                        height: parsed.coded_height,
                    };
                    let session_dpb_slots = parsed.max_dpb_slots.min(4) + 1;
                    let codec = VideoCodec::DecodeH264;
                    let dpb_slots = parsed.max_dpb_slots.min(4);
                    (
                        parsed,
                        codec,
                        vulkan,
                        session_dpb_slots,
                        dpb_slots,
                        coded_extent,
                        None,
                    )
                }
                AccessUnitCodec::H265 => {
                    let parsed = parse_h265(&data)?;
                    if parsed.coded_width == 0 || parsed.coded_height == 0 {
                        return Err(VideoError::DecoderInit(
                            "Failed to parse video dimensions".to_string(),
                        ));
                    }
                    let vulkan = super::VideoDeviceBuilder::new()
                        .with_validation(false)
                        .build()?;
                    let coded_extent = vk::Extent2D {
                        width: parsed.coded_width,
                        height: parsed.coded_height,
                    };
                    let session_dpb_slots = parsed.max_dpb_slots.min(4) + 1;
                    let codec = VideoCodec::DecodeH265;
                    let dpb_slots = parsed.max_dpb_slots.min(4);
                    (
                        parsed,
                        codec,
                        vulkan,
                        session_dpb_slots,
                        dpb_slots,
                        coded_extent,
                        None,
                    )
                }
                AccessUnitCodec::Vp9 => {
                    let (parsed, vulkan, session_dpb_slots, dpb_slots, coded_extent) =
                        parse_vp9_init(&data)?;
                    let codec = VideoCodec::DecodeVp9;
                    (
                        parsed,
                        codec,
                        vulkan,
                        session_dpb_slots,
                        dpb_slots,
                        coded_extent,
                        None,
                    )
                }
                AccessUnitCodec::Av1 => {
                    let (
                        parsed,
                        vulkan,
                        session_dpb_slots,
                        dpb_slots,
                        coded_extent,
                        av1_sps,
                    ) = parse_av1_init(&data)?;
                    let codec = VideoCodec::DecodeAv1;
                    (
                        parsed,
                        codec,
                        vulkan,
                        session_dpb_slots,
                        dpb_slots,
                        coded_extent,
                        av1_sps,
                    )
                }
            };

        let decode_queue_family = vulkan
            .queue_families
            .video_decode
            .ok_or_else(|| VideoError::VideoNotSupported("No decode queue".to_string()))?;

        // Query device capabilities
        eprintln!("[Decoder] Querying video capabilities...");
        let caps = vulkan.query_video_capabilities(
            codec,
            parsed.profile_idc,
            parsed.chroma_subsampling,
            parsed.luma_bit_depth,
            parsed.chroma_bit_depth,
        )?;
        eprintln!("[Decoder] Video capabilities queried successfully");

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

        let bs_buffer_size_alignment = caps.min_bitstream_buffer_size_alignment;
        let picture_access_granularity = caps.picture_access_granularity;

        // Align coded_extent for session max_coded_extent (required by Vulkan spec).
        // But use raw dimensions for images and per-frame decode commands.
        let session_coded_extent = {
            let align_width = picture_access_granularity.width;
            let align_height = picture_access_granularity.height;
            vk::Extent2D {
                width: (coded_extent.width + align_width - 1) & !(align_width - 1),
                height: (coded_extent.height + align_height - 1) & !(align_height - 1),
            }
        };

        let (session, session_params, session_memories) = create_video_session(
            &vulkan,
            codec,
            &parsed,
            session_coded_extent,
            session_dpb_slots,
            av1_sps.as_ref(),
        )?;
        eprintln!("[Decoder] Video session created successfully");

        let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

        let max_frame_size = extract_max_frame_size(&data, decoded_codec, max_frames);

        let bs_buffer = create_bitstream_buffer_with_profile(
            &vulkan.device,
            &vulkan.memory_properties,
            max_frame_size as u64,
            codec,
            parsed.profile_idc,
            parsed.chroma_subsampling,
            parsed.luma_bit_depth,
            parsed.chroma_bit_depth,
            decode_queue_family,
        )?;
        eprintln!("[Decoder] Bitstream buffer created");

        let command_pool = create_command_pool(&vulkan.device, decode_queue_family)?;
        eprintln!("[Decoder] Command pool created");

        let fence = create_fence(&vulkan.device)?;
        eprintln!("[Decoder] Fence created");

        let mut dpb_views = Vec::new();
        let mut dpb_images = Vec::new();
        let mut dpb_memories = Vec::new();
        let mut dpb_use_image_array = false;

        // Create DPB images aligned to pictureAccessGranularity, matching the C++
        // reference (VkVideoDecoder.cpp:259-262). The image extent must be a multiple
        // of the granularity for the NVIDIA driver to accept the DPB image; a raw
        // 1080 (not a multiple of 16) made the driver silently skip the decode.
        // The per-frame decode-command codedExtent stays raw (1920x1080) — the image
        // only needs to be >= it.
        //
        // When the device does NOT support VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_
        // IMAGES_BIT_KHR, the spec (VUID-VkVideoBeginCodingInfoKHR-flags-07244)
        // requires ALL reference imageViews to come from the SAME image. The C++
        // reference handles this with m_useImageArray (a single image with array
        // layers, one per DPB slot) — see VkVideoDecoder.cpp:349-353 and :544.
        // We do the same: one image with `session_dpb_slots` layers and one
        // per-slot image view selecting layer `slot`.
        let supports_separate_reference_images = caps
            .flags
            .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);
        // Iteration 19: VACC_SEPARATE_DPB=1 forces the SEPARATE-image DPB path
        // (one VkImage per slot, baseArrayLayer=0 everywhere) even when the device
        // does not support SEPARATE_REFERENCE_IMAGES, to exactly match the C++
        // reference and test whether the array-layer reference binding is the
        // cause of the multi-ref bug (VUID-07244 is violated; the driver
        // tolerates it — the C++ reference is pixel-perfect with separate images).
        let force_separate = std::env::var("VACC_SEPARATE_DPB")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        eprintln!(
            "[Decoder] Creating {} DPB slots ({}x{} aligned to granularity), separate_reference_images={} force_separate={}",
            session_dpb_slots,
            session_coded_extent.width,
            session_coded_extent.height,
            supports_separate_reference_images,
            force_separate
        );

        if !supports_separate_reference_images && !force_separate {
            let (img, views, mem) = create_dpb_image_array_with_profile(
                &vulkan.device,
                &vulkan.memory_properties,
                session_coded_extent.width,
                session_coded_extent.height,
                output_format,
                codec,
                parsed.profile_idc,
                parsed.chroma_subsampling,
                parsed.luma_bit_depth,
                parsed.chroma_bit_depth,
                decode_queue_family,
                session_dpb_slots,
            )?;
            eprintln!(
                "[Decoder] DPB image array created ({} layers, single image)",
                session_dpb_slots
            );
            for view in views {
                dpb_views.push(view);
                dpb_images.push(img);
            }
            dpb_memories.push(mem);
            dpb_use_image_array = true;
        } else {
            for i in 0..session_dpb_slots {
                let (img, view, mem) = create_output_image_with_profile(
                    &vulkan.device,
                    &vulkan.memory_properties,
                    session_coded_extent.width,
                    session_coded_extent.height,
                    output_format,
                    codec,
                    parsed.profile_idc,
                    parsed.chroma_subsampling,
                    parsed.luma_bit_depth,
                    parsed.chroma_bit_depth,
                    decode_queue_family,
                )?;
                eprintln!("[Decoder] DPB slot {} created", i);
                dpb_views.push(view);
                dpb_images.push(img);
                dpb_memories.push(mem);
            }
        }

        let mut dpb_manager = DpbManager::new(session_dpb_slots);

        if let Some(H264OrH265Sps::H264(sps)) = &parsed.sps {
            dpb_manager.set_max_num_ref_frames(sps.max_num_ref_frames);
            dpb_manager.set_max_frame_num(1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4));
        }

        let vp9_parser = if decoded_codec == AccessUnitCodec::Vp9 {
            let mut parser = vk_video_parser::vp9::Vp9Parser::new();
            vk_video_parser::VideoParser::init(
                &mut parser,
                &vk_video_parser::DetectedVideoFormat::new(
                    vk_video_core::codec::VideoCodec::DecodeVp9,
                ),
            )
            .map_err(|e| VideoError::DecoderInit(format!("VP9 parser init error: {e}")))?;
            Some(parser)
        } else {
            None
        };

        let (av1_parser, av1_sps) = if decoded_codec == AccessUnitCodec::Av1 {
            let mut parser = vk_video_parser::av1::Av1Parser::new();
            vk_video_parser::VideoParser::init(
                &mut parser,
                &vk_video_parser::DetectedVideoFormat::new(
                    vk_video_core::codec::VideoCodec::DecodeAv1,
                ),
            )
            .map_err(|e| VideoError::DecoderInit(format!("AV1 parser init error: {e}")))?;
            // Parse SPS from first frame data
            let av1_sps = parse_av1_sps_from_data(&data, &mut parser);
            (Some(parser), av1_sps)
        } else {
            (None, None)
        };

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
            dpb_use_image_array,
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
            picture_access_granularity,
            vp9_parser,
            av1_parser,
            av1_sps,
        })
    }

    /// Decode all frames from the bitstream.
    ///
    /// Returns frames in decoding order. Use `reorder_to_presentation` to
    /// reorder by presentation order (POC).
    pub fn decode_all(&mut self, max_frames: usize) -> VideoResult<Vec<DecodedFrame>> {
        match self.decoded_codec {
            AccessUnitCodec::H264 | AccessUnitCodec::H265 => self.decode_all_h26x(max_frames),
            AccessUnitCodec::Vp9 => self.decode_all_vp9(max_frames),
            AccessUnitCodec::Av1 => self.decode_all_av1(max_frames),
        }
    }

    /// Reorder frames from decoding order to presentation order (by POC).
    ///
    /// H.264/H.265 use B-frames which are decoded out of order. This method
    /// sorts frames by their picture order count (POC) for correct display.
    pub fn reorder_to_presentation(frames: Vec<DecodedFrame>) -> Vec<DecodedFrame> {
        let mut sorted = frames;
        sorted.sort_by_key(|f| f.poc);
        sorted
    }

    fn decode_all_h26x(&mut self, max_frames: usize) -> VideoResult<Vec<DecodedFrame>> {
        let items = super::access_unit::extract_all_access_units(
            self.bitstream_data(),
            self.decoded_codec,
            max_frames,
            self.parsed.sps.as_ref(),
            self.parsed.pps.as_ref(),
        );

        if items.is_empty() {
            return Err(VideoError::DecoderInit("No access units found".to_string()));
        }

        let items: Vec<_> = items.into_iter().take(max_frames * 2).collect();

        let mut frames = Vec::new();
        let mut is_first_frame = true;
        let mut access_unit_count = 0;

        for (idx, item) in items.iter().enumerate() {
            match item {
                ExtractedItem::ParameterSet(ps) => {
                    // Handle in-band parameter set update
                    self.handle_inband_parameter_set(ps)?;
                }
                ExtractedItem::AccessUnit(au) => {
                    if access_unit_count >= max_frames {
                        break;
                    }
                    access_unit_count += 1;

                    self.bs_buffer.write(&au.data)?;

                    let alignment = self.bs_buffer_size_alignment.max(1);
                    let aligned_size =
                        ((au.data.len() as u64 + alignment - 1) & !(alignment - 1)).max(alignment);
                    let padding_start = au.data.len() as u64;
                    let padding_size = aligned_size - padding_start;
                    if padding_size > 0 {
                        self.bs_buffer.zero_range(padding_start, padding_size);
                    }
                    self.bs_buffer.flush_range(0, aligned_size).ok();

                    let output_slot = if au.is_idr || au.no_output_of_prior_pics_flag {
                        self.dpb_manager.invalidate_all();
                        0
                    } else {
                        // For H.264: ref_pocs from access_unit is empty (no RPS concept).
                        // Use all valid DPB entries as protected references since any could be needed.
                        // For H.265: collect ref_pocs from ALL remaining access units to protect
                        // frames needed by future frames, not just the current one.
                        let protected_pocs: Vec<i32> =
                            if self.decoded_codec == AccessUnitCodec::H264 {
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
                                // Collect all POCs referenced by current and future access units
                                let mut all_ref_pocs = std::collections::HashSet::new();
                                for future_item in items[idx..].iter() {
                                    if let ExtractedItem::AccessUnit(future_au) = future_item {
                                        for poc in &future_au.ref_pocs {
                                            all_ref_pocs.insert(*poc);
                                        }
                                    }
                                }
                                all_ref_pocs.into_iter().collect()
                            };
                        self.dpb_manager
                            .find_or_recycle_slot(&protected_pocs)
                            .unwrap_or(0)
                    };

                    let output_view = self.dpb_views[output_slot as usize];
                    let output_img = self.dpb_images[output_slot as usize];

                    let actual_bs_size = aligned_size;

                    self.record_decode_command(
                        au,
                        output_view,
                        output_img,
                        actual_bs_size,
                        output_slot,
                        is_first_frame,
                    )?;

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
                            self.dpb_manager.apply_mmco(
                                au.frame_num,
                                output_slot,
                                &au.mmco_commands,
                            );
                        } else {
                            self.dpb_manager.apply_sliding_window(au.frame_num);
                        }
                    }

                    let pixels = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        output_img,
                        self.dpb_base_layer(output_slot as u32),
                        self.coded_extent.width,
                        self.coded_extent.height,
                    )?;

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
                        display_width: self.parsed.display_width,
                        display_height: self.parsed.display_height,
                        crop_left: self.parsed.crop_left,
                        crop_top: self.parsed.crop_top,
                    });
                }
            }
        }

        Ok(frames)
    }

    /// Handle in-band parameter set updates.
    /// Updates cached parameter sets and calls vkUpdateVideoSessionParametersKHR.
    fn handle_inband_parameter_set(&mut self, ps: &InBandParameterSet) -> VideoResult<()> {
        if ps.vps.is_none() && ps.sps.is_none() && ps.pps.is_none() {
            return Ok(());
        }

        // Update cached parameter sets
        if let Some(ref vps_opt) = ps.vps {
            let H265VpsOpt::H265(vps) = vps_opt;
            self.parsed.vps = Some(vps.clone());
        }
        if let Some(ref sps) = ps.sps {
            self.parsed.sps = Some(sps.clone());
        }
        if let Some(ref pps) = ps.pps {
            self.parsed.pps = Some(pps.clone());
        }

        // Call update_session_parameters with the new parameter sets
        let session_params = match &self.session_params {
            Some(p) => p.handle(),
            None => return Ok(()), // No session params to update
        };

        match self.codec {
            VideoCodec::DecodeH264 => {
                let sps_ref = match &self.parsed.sps {
                    Some(H264OrH265Sps::H264(s)) => Some(s),
                    _ => None,
                };
                let pps_ref = match &self.parsed.pps {
                    Some(H264OrH265Pps::H264(p)) => Some(p),
                    _ => None,
                };

                let mut h264_decoder =
                    H264Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
                h264_decoder.update_session_parameters(session_params, sps_ref, pps_ref)?;
            }
            VideoCodec::DecodeH265 => {
                let vps_ref = self.parsed.vps.as_ref();
                let sps_ref = match &self.parsed.sps {
                    Some(H264OrH265Sps::H265(s)) => Some(s),
                    _ => None,
                };
                let pps_ref = match &self.parsed.pps {
                    Some(H264OrH265Pps::H265(p)) => Some(p),
                    _ => None,
                };

                let mut h265_decoder =
                    H265Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
                h265_decoder.update_session_parameters(
                    session_params,
                    vps_ref,
                    sps_ref,
                    pps_ref,
                )?;
            }
            _ => {}
        }

        Ok(())
    }

    fn decode_all_vp9(&mut self, max_frames: usize) -> VideoResult<Vec<DecodedFrame>> {
        let frames = super::access_unit::extract_vp9_frames(&self.bitstream_data, max_frames);

        if frames.is_empty() {
            return Err(VideoError::DecoderInit("No VP9 frames found".to_string()));
        }

        let mut vp9_decoder =
            Vp9Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
        vp9_decoder.set_session(&self.session);
        vp9_decoder.set_max_dpb_slots(self.parsed.max_dpb_slots);

        let mut decoded_frames = Vec::new();
        let mut is_first_frame = true;
        let mut frame_count: u32 = 0;

        let _align_width = self.picture_access_granularity.width;
        let _align_height = self.picture_access_granularity.height;

        for (frame_idx, vp9_frame) in frames.iter().enumerate().take(max_frames) {
            // Parse frame header
            let parser = self
                .vp9_parser
                .as_mut()
                .ok_or_else(|| VideoError::DecoderInit("VP9 parser not initialized".to_string()))?;
            let parsed = parser.parse_frame(&vp9_frame.data).map_err(|e| {
                VideoError::DecoderInit(format!("Failed to parse VP9 frame {}: {:?}", frame_idx, e))
            })?;

            // Use raw frame dimensions for coded extent (not aligned).
            // Vulkan spec: codedExtent is the actual coded dimensions of the frame.
            // picture_access_granularity alignment is only required for session max_coded_extent.
            let frame_coded_extent = vk::Extent2D {
                width: parsed.frame_width,
                height: parsed.frame_height,
            };

            // Handle show_existing_frame
            if parsed.show_existing_frame {
                let frame_buffer_idx = parsed.frame_to_show_map_idx as usize;
                let pic_idx = vp9_decoder.get_pic_idx_for_frame_buffer(frame_buffer_idx);
                if pic_idx >= 0 {
                    let slot = pic_idx as usize;
                    let img = self.dpb_images[slot];
                    let pixels = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        img,
                        self.dpb_base_layer(slot as u32),
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    )?;
                    decoded_frames.push(DecodedFrame {
                        poc: frame_count as i32,
                        frame_num: frame_count,
                        is_idr: false,
                        is_reference: false,
                        pixels,
                        coded_width: frame_coded_extent.width,
                        coded_height: frame_coded_extent.height,
                        display_width: self.parsed.display_width,
                        display_height: self.parsed.display_height,
                        crop_left: self.parsed.crop_left,
                        crop_top: self.parsed.crop_top,
                    });
                }
                continue;
            }

            let is_key_frame =
                parsed.picture_info.frame_type == vk_video_core::picture::Vp9FrameType::Key;

            // Write bitstream data
            let bs_align = self.bs_buffer_size_alignment.max(1);
            let actual_size = vp9_frame.data.len() as u64;
            let aligned_size = ((actual_size + bs_align - 1) & !(bs_align - 1)).max(bs_align);
            self.bs_buffer.zero_range(0, aligned_size);
            self.bs_buffer.write(&vp9_frame.data)?;
            self.bs_buffer.flush_range(0, aligned_size).ok();

            // Compute reference name slot indices
            let reference_name_slot_indices = vp9_decoder
                .compute_reference_name_slot_indices(is_key_frame, &parsed.ref_frame_idx);

            // Select DPB slot
            let output_slot = if is_key_frame || is_first_frame {
                if is_key_frame {
                    self.dpb_manager.invalidate_all();
                    vp9_decoder.reset_dpb();
                }
                0
            } else {
                let exclude_slots: Vec<i32> = reference_name_slot_indices
                    .iter()
                    .filter(|&&s| s >= 0)
                    .copied()
                    .collect();
                self.dpb_manager
                    .find_or_recycle_slot_excluding(&exclude_slots)
                    .unwrap_or(0)
            };

            let output_view = self.dpb_views[output_slot as usize];
            let output_img = self.dpb_images[output_slot as usize];

            // Build DPB picture resources
            let (dpb_setup_picture, dpb_ref_pictures, dpb_ref_slot_indices) =
                build_vp9_dpb_picture_resources(
                    &self.dpb_manager,
                    &self.dpb_views,
                    frame_coded_extent,
                    output_slot,
                    is_key_frame,
                    &reference_name_slot_indices,
                );

            let dpb_ref_images: Vec<vk::Image> = dpb_ref_slot_indices
                .iter()
                .map(|&slot_idx| self.dpb_images[slot_idx as usize])
                .collect();

            // Get actual layouts of reference slots for proper memory barriers.
            // This ensures we use correct old_layout in barriers (not always UNDEFINED).
            let dpb_ref_slot_layouts: Vec<vk::ImageLayout> = dpb_ref_slot_indices
                .iter()
                .map(|&slot_idx| self.dpb_manager.get_slot_layout(slot_idx as u32))
                .collect();

            // Convert to Vulkan picture info
            let mut picture_info_container = Box::new(convert_vp9_picture_info(
                &parsed.picture_info,
                &parsed.color_config,
                &parsed.loop_filter,
                &parsed.segmentation,
            ));
            picture_info_container.init_pointers();

            let vp9_decode_info = Box::new(VideoDecodeVP9PictureInfoKHR::new(
                picture_info_container.std_picture_info(),
                reference_name_slot_indices,
                parsed.uncompressed_header_offset,
                parsed.compressed_header_offset,
                parsed.tiles_offset,
            ));

            // Record decode command
            let cmd_buffer = allocate_command_buffer(&self.vulkan.device, self.command_pool)?;
            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.vulkan
                    .device
                    .begin_command_buffer(cmd_buffer, &begin_info)
                    .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;
            }

            let output_slot_old_layout = self.dpb_manager.get_slot_layout(output_slot);
            let result = vp9_decoder.record_decode_command(
                cmd_buffer,
                self.session.handle(),
                self.session_params
                    .as_ref()
                    .map(|p| p.handle())
                    .unwrap_or(vk::VideoSessionParametersKHR::null()),
                self.bs_buffer.buffer(),
                0,
                aligned_size,
                output_view,
                output_img,
                frame_coded_extent,
                dpb_setup_picture,
                &dpb_ref_pictures,
                &dpb_ref_slot_indices,
                &dpb_ref_images,
                &dpb_ref_slot_layouts,
                &picture_info_container,
                &vp9_decode_info,
                is_first_frame,
                output_slot as i32,
                output_slot_old_layout,
            );

            let _picture_info_guard = picture_info_container;
            let _vp9_decode_guard = vp9_decode_info;

            if is_first_frame {
                is_first_frame = false;
            }

            result.map_err(|e| VideoError::DecoderInit(format!("VP9 decode failed: {}", e)))?;

            // Submit
            unsafe {
                self.vulkan
                    .device
                    .end_command_buffer(cmd_buffer)
                    .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;
                self.vulkan
                    .device
                    .reset_fences(&[self.fence])
                    .map_err(|e| VideoError::FenceWait(e.to_string()))?;
                let queue = self
                    .vulkan
                    .device
                    .get_device_queue(self.decode_queue_family, 0);
                self.vulkan
                    .device
                    .queue_submit(
                        queue,
                        &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                        self.fence,
                    )
                    .map_err(|e| VideoError::QueueSubmission(e.to_string()))?;
                self.vulkan
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| VideoError::FenceWait(e.to_string()))?;
            }

            // Record which DPB slot contains this frame buffer
            let current_frame_buffer_idx = output_slot as i32;
            vp9_decoder.set_frame_buffer_dpb_slot(output_slot as usize, current_frame_buffer_idx);

            self.dpb_manager.register_frame(output_slot, frame_count);
            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

            // Readback
            let pixels = super::readback::readback_decoded_image(
                &self.vulkan.instance,
                &self.vulkan.device,
                &self.vulkan.memory_properties,
                self.decode_queue_family,
                self.command_pool,
                self.fence,
                output_img,
                self.dpb_base_layer(output_slot as u32),
                frame_coded_extent.width,
                frame_coded_extent.height,
            )?;

            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

            decoded_frames.push(DecodedFrame {
                poc: frame_count as i32,
                frame_num: frame_count,
                is_idr: is_key_frame,
                is_reference: true,
                pixels,
                coded_width: frame_coded_extent.width,
                coded_height: frame_coded_extent.height,
                display_width: self.parsed.display_width,
                display_height: self.parsed.display_height,
                crop_left: self.parsed.crop_left,
                crop_top: self.parsed.crop_top,
            });

            frame_count += 1;
        }

        Ok(decoded_frames)
    }

    fn decode_all_av1(&mut self, max_frames: usize) -> VideoResult<Vec<DecodedFrame>> {
        let frames = super::access_unit::extract_av1_frames(&self.bitstream_data, max_frames);

        if frames.is_empty() {
            return Err(VideoError::DecoderInit("No AV1 frames found".to_string()));
        }

        let mut av1_decoder =
            Av1Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
        av1_decoder.set_session(&self.session);

        let sps = self.av1_sps.as_ref().ok_or_else(|| {
            VideoError::DecoderInit("AV1 SPS not available".to_string())
        })?;

        let mut decoded_frames = Vec::new();
        let mut is_first_frame = true;
        let mut frame_count: u32 = 0;
        let mut display_count: usize = 0;

        eprintln!(
            "[AV1] session_params present: {}, handle_is_null: {}",
            self.session_params.is_some(),
            self.session_params
                .as_ref()
                .map(|p| p.handle().is_null())
                .unwrap_or(true)
        );
        // AV1 frame buffer indices: INTRA=0, LAST=1, LAST2=2, LAST3=3,
        // GOLDEN=4, BWDREF=5, ALTREF2=6, ALTREF=7
        // These map to ref_frame_idx[0..6] in the parser

        for (frame_idx, av1_frame) in frames.iter().enumerate() {
            // Parse frame header
            let parser = self
                .av1_parser
                .as_mut()
                .ok_or_else(|| VideoError::DecoderInit("AV1 parser not initialized".to_string()))?;

            // Use the per-Frame-OBU payload (extracted in extract_av1_frames).
            // The bitstream written to the GPU is the full IVF packet; the tile
            // offsets below point into it for THIS Frame OBU (C++ behavior).
            let frame_obu_payload: &[u8] = &av1_frame.frame_obu_payload[..];

            // Parse frame header from the Frame OBU payload
            let fh = match parser.parse_frame_header(frame_obu_payload, sps) {
                Ok(fh) => fh,
                Err(e) => {
                    eprintln!("[AV1] Failed to parse frame {} header: {:?}", frame_idx, e);
                    // Skip this frame
                    continue;
                }
            };

              eprintln!(
                  "[AV1] Frame {} (disp#{}): type={:?}, show_frame={}, show_existing={}, show_map_idx={}, order_hint={}, primary_ref={}, refresh_flags={:08b}, ref_idx={:?}, payload_start={}",
                  frame_idx,
                  display_count,
                  match fh.frame_type {
                      0 => "KEY",
                      1 => "INTER",
                      2 => "INTRA_ONLY",
                      3 => "SWITCH",
                      _ => "UNKNOWN",
                  },
                  fh.show_frame,
                  fh.show_existing_frame,
                  fh.frame_to_show_map_idx,
                  fh.order_hint,
                  fh.primary_ref_frame,
                  fh.refresh_frame_flags,
                  fh.ref_frame_idx,
                  av1_frame.payload_start,
              );

            // Handle show_existing_frame: no new decode, output the already-decoded
            // buffer of the referenced frame buffer as this display frame.
            if fh.show_existing_frame {
                let frame_buffer_idx = fh.frame_to_show_map_idx as usize;
                let pic_idx = av1_decoder.get_pic_idx_for_frame_buffer(frame_buffer_idx);
                eprintln!(
                    "[AV1]   show_existing: frame_buffer_idx={} -> dpb_slot={}",
                    frame_buffer_idx, pic_idx
                );
                if pic_idx >= 0 {
                    let slot = pic_idx as usize;
                    let img = self.dpb_images[slot];
                    // The show_existing_frame header carries no size fields
                    // (fh.frame_width/height == 0); use the coded dimensions
                    // recorded when the referenced frame buffer was refreshed.
                    let (ref_width, ref_height) =
                        av1_decoder.get_frame_buffer_dims(frame_buffer_idx);
                    let (ref_width, ref_height) = if ref_width == 0 || ref_height == 0 {
                        // Fallback: session coded extent.
                        (self.coded_extent.width, self.coded_extent.height)
                    } else {
                        (ref_width, ref_height)
                    };
                    let pixels = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        img,
                        self.dpb_base_layer(slot as u32),
                        ref_width,
                        ref_height,
                    )?;
                    decoded_frames.push(DecodedFrame {
                        poc: frame_count as i32,
                        frame_num: frame_count,
                        is_idr: false,
                        is_reference: false,
                        pixels,
                        coded_width: ref_width,
                        coded_height: ref_height,
                        display_width: self.parsed.display_width,
                        display_height: self.parsed.display_height,
                        crop_left: self.parsed.crop_left,
                        crop_top: self.parsed.crop_top,
                    });
                    display_count += 1;
                    if display_count >= max_frames {
                        break;
                    }
                }
                continue;
            }

            // Use actual frame dimensions
            let frame_coded_extent = vk::Extent2D {
                width: fh.frame_width,
                height: fh.frame_height,
            };

            let is_key_frame = fh.frame_type == 0; // KEY_FRAME

            // Write bitstream data
            let bs_align = self.bs_buffer_size_alignment.max(1);
            // TEST (iter 22): VACC_SINGLE_OBU=1 writes only the current frame's
            // Frame OBU payload to the bitstream buffer (not the whole IVF packet
            // containing all 5 Frame OBUs). If the driver picks the wrong frame
            // header from the multi-OBU buffer, this isolates the single OBU so the
            // frame header is unambiguously at offset 0.
            let single_obu = std::env::var("VACC_SINGLE_OBU")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false);
            let bs_data: &[u8] = if single_obu {
                &av1_frame.frame_obu_payload
            } else {
                &av1_frame.data
            };
            let actual_size = bs_data.len() as u64;
            let aligned_size = ((actual_size + bs_align - 1) & !(bs_align - 1)).max(bs_align);
            // TEMP DIAGNOSTIC (iteration 5): dump first bytes of frame 0 bitstream
            if frame_idx == 0 {
                eprintln!(
                    "[AV1-DIAG] frame0 bitstream: size={}, first32={:02x?} frame_header_offset={}",
                    actual_size,
                    &av1_frame.data[..av1_frame.data.len().min(32)],
                    find_av1_frame_header_offset(&av1_frame.data),
                );
                eprintln!(
                    "[FH-DIAG] frame0: frame_type={} frame_w={}x{} tile_cols_log2={} tile_rows_log2={} order_hint={} refresh_flags={:08b} primary_ref={} base_q={} interp_filter={} tx_mode={} superres={} render_diff={} film_grain={} enable_order_hint_sps={}",
                    fh.frame_type, fh.frame_width, fh.frame_height,
                    fh.tile_cols_log2, fh.tile_rows_log2, fh.order_hint,
                    fh.refresh_frame_flags, fh.primary_ref_frame, fh.base_q_index,
                    fh.interpolation_filter, fh.tx_mode, fh.use_superres,
                    fh.render_and_frame_size_different, fh.apply_grain,
                    0
                );
                eprintln!(
                    "[AV1-TILE-DIAG] frame0: frame_header_offset={} frame_header_size={} tile_offset={} tile_size={} frame_obu_payload_len={}",
                    find_av1_frame_header_offset(&av1_frame.data),
                    fh.frame_header_size,
                    find_av1_frame_header_offset(&av1_frame.data) + fh.frame_header_size,
                    (frame_obu_payload.len() as u32).saturating_sub(fh.frame_header_size),
                    frame_obu_payload.len(),
                );
            }
            // DEBUG (iteration 10): picture-info GAP fields for frames 1-2
            if frame_idx == 1 || frame_idx == 2 || frame_idx == 3 {
                eprintln!(
                    "[AV1-TILE-DBG] frame{}: uniform_tile_spacing={} tile_count={} tile_cols={} tile_rows={} tile_cols_log2={} tile_rows_log2={} tile_size_bytes_minus_1={} context_update_tile_id={} diff_uv_delta={} separate_uv_delta_q={} base_q={} using_qmatrix={} w_sbs={:?} h_sbs={:?} mi_col={:?} mi_row={:?}",
                    frame_idx,
                    fh.uniform_tile_spacing_flag,
                    fh.tile_count,
                    fh.tile_cols,
                    fh.tile_rows,
                    fh.tile_cols_log2,
                    fh.tile_rows_log2,
                    fh.tile_size_bytes_minus_1,
                    fh.context_update_tile_id,
                    fh.diff_uv_delta,
                    self.av1_sps.as_ref().map(|s| s.separate_uv_delta_q).unwrap_or(false),
                    fh.base_q_index,
                    fh.using_qmatrix,
                    &fh.tile_width_in_sbs_minus_1[..fh.tile_cols.min(64) as usize],
                    &fh.tile_height_in_sbs_minus_1[..fh.tile_rows.min(64) as usize],
                    &fh.tile_mi_col_starts[..fh.tile_cols.min(64) as usize],
                    &fh.tile_mi_row_starts[..fh.tile_rows.min(64) as usize],
                );
                eprintln!(
                    "[AV1-LF-DBG] frame{}: lf_delta_enabled={} lf_delta_update={} lf_ref_deltas={:?} lf_mode_deltas={:?} lf_level=[{},{},{},{}] lf_sharp={} primary_ref={} ref_idx={:?} frss={} order_hint={}",
                    frame_idx,
                    fh.loop_filter_delta_enabled,
                    fh.loop_filter_delta_update,
                    fh.loop_filter_ref_deltas,
                    fh.loop_filter_mode_deltas,
                    fh.loop_filter_level[0],
                    fh.loop_filter_level[1],
                    fh.loop_filter_level_uv[0],
                    fh.loop_filter_level_uv[1],
                    fh.loop_filter_sharpness,
                    fh.primary_ref_frame,
                    fh.ref_frame_idx,
                    fh.frame_refs_short_signaling,
                    fh.order_hint,
                );
            }
            self.bs_buffer.zero_range(0, aligned_size);
            self.bs_buffer.write(bs_data)?;
            self.bs_buffer.flush_range(0, aligned_size).ok();

            // Compute reference name slot indices
            let reference_name_slot_indices = av1_decoder
                .compute_reference_name_slot_indices(is_key_frame, &fh.ref_frame_idx, fh.primary_ref_frame);

            eprintln!(
                "[AV1]   reference_name_slot_indices={:?}",
                reference_name_slot_indices
            );

            // Select DPB slot
            let output_slot = if is_key_frame || is_first_frame {
                if is_key_frame {
                    self.dpb_manager.invalidate_all();
                    av1_decoder.reset_dpb();
                }
                0
            } else {
                let exclude_slots: Vec<i32> = reference_name_slot_indices
                    .iter()
                    .filter(|&&s| s >= 0)
                    .copied()
                    .collect();
                self.dpb_manager
                    .find_or_recycle_slot_excluding(&exclude_slots)
                    .unwrap_or(0)
            };

            let output_view = self.dpb_views[output_slot as usize];
            let output_img = self.dpb_images[output_slot as usize];

            // Build DPB picture resources
            let (dpb_setup_picture, dpb_ref_pictures, dpb_ref_slot_indices) =
                build_av1_dpb_picture_resources(
                     &self.dpb_manager,
                     &self.dpb_views,
                     frame_coded_extent,
                     output_slot,
                     is_key_frame,
                     &reference_name_slot_indices,
                     &av1_decoder,
                 );

            let dpb_ref_images: Vec<vk::Image> = dpb_ref_slot_indices
                .iter()
                .map(|&slot_idx| self.dpb_images[slot_idx as usize])
                .collect();

            // Order hint of the picture in each reference slot (for the
            // VkVideoDecodeAV1DpbSlotInfoKHR pNext chain).
            let dpb_ref_order_hints: Vec<u32> = dpb_ref_slot_indices
                .iter()
                .map(|&slot_idx| {
                    reference_name_slot_indices
                        .iter()
                        .zip(fh.ref_frame_idx.iter())
                        .find(|(s, _)| **s == slot_idx)
                        .map(|(_, &fb_idx)| {
                            av1_decoder.get_frame_buffer_order_hint(fb_idx as usize)
                        })
                        .unwrap_or(0)
                })
                .collect();

            let dpb_ref_slot_layouts: Vec<vk::ImageLayout> = dpb_ref_slot_indices
                .iter()
                .map(|&slot_idx| self.dpb_manager.get_slot_layout(slot_idx as u32))
                .collect();

            // Build AV1 picture info container
            let mut picture_info_container = Av1PictureInfoContainer::default();
            let pic_info = &mut picture_info_container.std_picture_info;

            // Fill in picture info from parsed frame header
            pic_info.frame_type = match fh.frame_type {
                0 => ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY,
                1 => ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER,
                2 => ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY,
                3 => ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_SWITCH,
                _ => ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY,
            };
            // TEST (iter 21): force base_q_index=0 for ext3 (frame_idx==3) to check
            // if the driver uses our picture info. If output != keyframe after this,
            // the driver IS decoding (using our picture info). If still keyframe, the
            // driver is copying the reference (ignoring our picture info).
            if frame_idx == 3 && std::env::var("VACC_FORCE_Q0_F3")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false)
            {
                eprintln!("[TEST] forcing base_q_index=0 for ext3 (was {})", fh.base_q_index);
            }
            pic_info.current_frame_id = 0;
            pic_info.OrderHint = (fh.order_hint & 0xFF) as u8;
            pic_info.primary_ref_frame = fh.primary_ref_frame;
            pic_info.refresh_frame_flags = fh.refresh_frame_flags;
            pic_info.coded_denom = fh.coded_denom;

            // Picture-level flags (AV1 spec 7.10 uncompressed_header)
            pic_info.flags.set_error_resilient_mode(fh.error_resilient_mode as u32);
            pic_info.flags.set_disable_cdf_update(fh.disable_cdf_update as u32);
            pic_info.flags.set_use_superres(fh.use_superres as u32);
            pic_info.flags.set_render_and_frame_size_different(
                fh.render_and_frame_size_different as u32,
            );
            pic_info.flags.set_allow_screen_content_tools(fh.allow_screen_content_tools as u32);
            pic_info.flags.set_is_filter_switchable(fh.is_filter_switchable as u32);
            pic_info.flags.set_force_integer_mv(fh.force_integer_mv as u32);
            pic_info.flags.set_frame_size_override_flag(fh.frame_size_override_flag as u32);
            // SPS use_buffer_removal_time = false for our stream
            pic_info.flags.set_buffer_removal_time_present_flag(0);
            pic_info.flags.set_allow_intrabc(fh.allow_intrabc as u32);
            pic_info.flags.set_frame_refs_short_signaling(fh.frame_refs_short_signaling as u32);
            pic_info.flags.set_allow_high_precision_mv(fh.allow_high_precision_mv as u32);
            pic_info.flags.set_is_motion_mode_switchable(fh.is_motion_mode_switchable as u32);
            pic_info.flags.set_use_ref_frame_mvs(fh.use_ref_frame_mvs as u32);
            pic_info.flags.set_disable_frame_end_update_cdf(fh.disable_frame_end_update_cdf as u32);
            pic_info.flags.set_allow_warped_motion(fh.allow_warped_motion as u32);
            pic_info.flags.set_reduced_tx_set(fh.reduced_tx_set as u32);
            pic_info.flags.set_reference_select(fh.reference_select as u32);
            pic_info.flags.set_skip_mode_present(fh.skip_mode_present as u32);
            pic_info.flags.set_delta_q_present(fh.delta_q_present as u32);
            pic_info.flags.set_delta_lf_present(fh.delta_lf_present as u32);
            pic_info.flags.set_delta_lf_multi(fh.delta_lf_multi as u32);
            pic_info.flags.set_segmentation_enabled(fh.segmentation_enabled as u32);
            pic_info.flags.set_segmentation_update_map(fh.segmentation_update_map as u32);
            pic_info.flags.set_segmentation_temporal_update(fh.segmentation_temporal_update as u32);
            pic_info.flags.set_segmentation_update_data(fh.segmentation_update_data as u32);
            pic_info.flags.set_UsesLr(fh.uses_lr as u32);
            pic_info.flags.set_usesChromaLr(
                (fh.loop_restoration_type[1] != 0 || fh.loop_restoration_type[2] != 0) as u32,
            );
            pic_info.flags.set_apply_grain(fh.apply_grain as u32);

            // Interpolation filter and tx mode (parser already stores Vulkan enum values)
            pic_info.interpolation_filter = fh.interpolation_filter as u32;
            pic_info.TxMode = fh.tx_mode as u32;
            pic_info.delta_q_res = fh.delta_q_res;
            pic_info.delta_lf_res = fh.delta_lf_res;
            // SkipModeFrame: reference name indices (1-based) of the nearest
            // forward/backward references, computed by the parser (C++
            // VulkanAV1Decoder.cpp IsSkipModeAllowed). The driver uses these
            // for skip-mode block motion compensation.
            pic_info.SkipModeFrame = fh.skip_mode_frame;

            // Order hints for all 8 reference names: the order hint of the frame
            // buffer that reference name i references.
            // ref_frame_idx order (parser): [0]=LAST, [1]=LAST2, [2]=LAST3, [3]=GOLDEN,
            // [4]=BWDREF, [5]=ALTREF2, [6]=ALTREF
            // The OrderHints array is indexed by the AV1 frame buffer index
            // (0=INTRA, 1=LAST, 2=LAST2, 3=LAST3, 4=GOLDEN, 5=BWDREF,
            // 6=ALTREF2, 7=ALTREF), matching referenceNameSlotIndices.
            let ref_name_to_ref_idx: [Option<usize>; 8] = [
                None,    // INTRA (0)
                Some(0), // LAST (1)
                Some(1), // LAST2 (2)
                Some(2), // LAST3 (3)
                Some(3), // GOLDEN (4)
                Some(4), // BWDREF (5)
                Some(5), // ALTREF2 (6)
                Some(6), // ALTREF (7)
            ];
            for i in 0..8usize {
                pic_info.expectedFrameId[i] = 0; // frame_id_numbers = false
                if let Some(ri) = ref_name_to_ref_idx[i] {
                    let fb = fh.ref_frame_idx[ri] as usize;
                    if fb < 8 {
                        pic_info.OrderHints[i] =
                            (av1_decoder.get_frame_buffer_order_hint(fb) & 0xFF) as u8;
                    }
                }
            }

            // Tile info (mirrors C++ VulkanAV1Decoder.cpp:1185-1289)
            let tile_info = &mut picture_info_container.tile_info;
            tile_info.flags.set_uniform_tile_spacing_flag(fh.uniform_tile_spacing_flag as u32);
            tile_info.TileCols = fh.tile_cols.min(255) as u8;
            tile_info.TileRows = fh.tile_rows.min(255) as u8;
            tile_info.context_update_tile_id = fh.context_update_tile_id as u16;
            // C++ (VulkanAV1Decoder.cpp:1281-1289): the driver-facing struct field
            // is value-initialized to 0 and only assigned u(2) when TileRows*TileCols>1.
            // The class-member default of 3 (line 62) is NOT what the driver sees.
            tile_info.tile_size_bytes_minus_1 = if fh.tile_count > 1 {
                fh.tile_size_bytes_minus_1.min(255) as u8
            } else {
                0
            };

            // Per-tile size arrays from the parser (C++ VulkanVideoParser.cpp:2549-2552).
            // width/col arrays are per tile COLUMN, height/row arrays per tile ROW.
            let tile_cols_n = fh.tile_cols.min(64) as usize;
            let tile_rows_n = fh.tile_rows.min(64) as usize;
            picture_info_container.tile_width_in_sbs_minus_1 =
                fh.tile_width_in_sbs_minus_1[..tile_cols_n].to_vec();
            picture_info_container.tile_height_in_sbs_minus_1 =
                fh.tile_height_in_sbs_minus_1[..tile_rows_n].to_vec();
            picture_info_container.tile_mi_col_starts =
                fh.tile_mi_col_starts[..tile_cols_n].to_vec();
            picture_info_container.tile_mi_row_starts =
                fh.tile_mi_row_starts[..tile_rows_n].to_vec();

            // Quantization
            let quant = &mut picture_info_container.quantization;
            quant.flags.set_using_qmatrix(fh.using_qmatrix as u32);
            quant.flags.set_diff_uv_delta(fh.diff_uv_delta as u32);
            quant.base_q_idx = if frame_idx == 3
                && std::env::var("VACC_FORCE_Q0_F3")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                0
            } else {
                fh.base_q_index
            };
            quant.DeltaQYDc = fh.delta_q_y_dc;
            quant.DeltaQUDc = fh.delta_q_u_dc;
            quant.DeltaQUAc = fh.delta_q_u_ac;
            quant.DeltaQVDc = fh.delta_q_v_dc;
            quant.DeltaQVAc = fh.delta_q_v_ac;
            quant.qm_y = fh.qm_y;
            quant.qm_u = fh.qm_u;
            quant.qm_v = fh.qm_v;

            // Loop filter
            let lf = &mut picture_info_container.loop_filter;
            lf.flags.set_loop_filter_delta_enabled(fh.loop_filter_delta_enabled as u32);
            lf.flags.set_loop_filter_delta_update(fh.loop_filter_delta_update as u32);
            lf.loop_filter_level = [
                fh.loop_filter_level[0],
                fh.loop_filter_level[1],
                fh.loop_filter_level_uv[0],
                fh.loop_filter_level_uv[1],
            ];
            lf.loop_filter_sharpness = fh.loop_filter_sharpness;
            lf.update_ref_delta = fh.loop_filter_delta_update as u8;
            lf.loop_filter_ref_deltas = fh.loop_filter_ref_deltas;
            // GAP: parser does not store update_mode_delta
            lf.update_mode_delta = 0;
            lf.loop_filter_mode_deltas = fh.loop_filter_mode_deltas;

            // Segmentation
            let seg = &mut picture_info_container.segmentation;
            seg.FeatureEnabled = fh.segment_feature_enabled;
            seg.FeatureData = fh.segment_feature_data;

            // CDEF
            let cdef = &mut picture_info_container.cdef;
            cdef.cdef_damping_minus_3 = fh.cdef_damping;
            cdef.cdef_bits = fh.cdef_bits;
            cdef.cdef_y_pri_strength = fh.cdef_y_pri_strength;
            cdef.cdef_y_sec_strength = fh.cdef_y_sec_strength;
            cdef.cdef_uv_pri_strength = fh.cdef_uv_pri_strength;
            cdef.cdef_uv_sec_strength = fh.cdef_uv_sec_strength;

            // Loop restoration (parser already stores remapped StdVideo values)
            let lr = &mut picture_info_container.loop_restoration;
            for i in 0..3 {
                lr.FrameRestorationType[i] = fh.loop_restoration_type[i] as u32;
                lr.LoopRestorationSize[i] = fh.loop_restoration_size[i];
            }

            // Global motion: index 0 = identity, 1..7 from parser models 0..6
            let gm = &mut picture_info_container.global_motion;
            gm.GmType[0] = 0;
            gm.gm_params[0] = [0, 0, 65536, 0, 0, 65536];
            for i in 1..8 {
                gm.GmType[i] = fh.global_motion_type[i - 1];
                gm.gm_params[i] = fh.global_motion_params[i - 1];
            }

            // Film grain: not present in our SPS (pFilmGrain stays null via init_pointers)

            picture_info_container.init_pointers();

            // Tile offsets/sizes: the C++ reference (verified working on this
            // NVIDIA driver) ALWAYS sets tileCount=1 with tileOffsets[0] and
            // tileSizes[0], even for single-tile frames. Without them the driver
            // doesn't know where the tile data is -> decodes nothing -> zeros.
            //   tileOffsets[0] = frame header payload offset + consumed header bytes
            //   tileSizes[0]   = Frame OBU payload size - consumed header bytes
            let tile_offset = if single_obu {
                fh.frame_header_size
            } else {
                av1_frame.payload_start + fh.frame_header_size
            };
            let tile_size = av1_frame.payload_size.saturating_sub(fh.frame_header_size);
            picture_info_container.tile_offsets[0] = tile_offset;
            picture_info_container.tile_sizes[0] = tile_size;

            // Frame header offset: points to the start of the frame header data
            // (the Frame OBU payload) within the bitstream buffer. The driver
            // parses the frame header from the bitstream (it ignores the picture
            // info we pass for the decode — verified by forcing base_q=0 with no
            // effect). With multiple Frame OBUs in the buffer and offset=0, the
            // driver finds the wrong frame header → ext3/ext4 decode to keyframe.
            // (iter 21: was hardcoded 0, matching C++ which also uses 0 but works
            // because... TBD. Testing the per-frame payload_start value.)
            let frame_header_offset: u32 = if std::env::var("VACC_FH_OFF")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false)
            {
                av1_frame.payload_start
            } else {
                0
            };

            // DEBUG (iteration 26): comprehensive StdVideoDecodeAV1PictureInfo dump
            // for EVERY decoded frame, to diff field-by-field vs C++ [CPP-PI].
            {
                let pi = &picture_info_container.std_picture_info;
                let f = &pi.flags;
                eprintln!(
                    "[RUST-PI-ALL] fc={} type={} oh={} primref={} refresh={:08x} refidx={:?}",
                    frame_idx, pi.frame_type as u32, pi.OrderHint, pi.primary_ref_frame,
                    pi.refresh_frame_flags, fh.ref_frame_idx
                );
                eprintln!(
                    "[RUST-PI-ALL]   flags: superres={} renderdiff={} screencontent={} filterswitch={} intmv={} intrabc={} frss={} highprec={} mmodesw={} refrf_mvs={} warp={} reductx={} refsel={} skipmode={} deltaq={} delf={} delfmulti={} segen={} segmap={} segtemp={} segdata={} useslr={} chromalr={} grain={}",
                    f.use_superres(), f.render_and_frame_size_different(), f.allow_screen_content_tools(),
                    f.is_filter_switchable(), f.force_integer_mv(), f.allow_intrabc(), f.frame_refs_short_signaling(),
                    f.allow_high_precision_mv(), f.is_motion_mode_switchable(), f.use_ref_frame_mvs(),
                    f.allow_warped_motion(), f.reduced_tx_set(), f.reference_select(), f.skip_mode_present(),
                    f.delta_q_present(), f.delta_lf_present(), f.delta_lf_multi(), f.segmentation_enabled(),
                    f.segmentation_update_map(), f.segmentation_temporal_update(), f.segmentation_update_data(),
                    f.UsesLr(), f.usesChromaLr(), f.apply_grain()
                );
                eprintln!(
                    "[RUST-PI-ALL]   flags2: errres={} discdf={} fso={} brt={}",
                    f.error_resilient_mode(), f.disable_cdf_update(),
                    f.frame_size_override_flag(), f.buffer_removal_time_present_flag()
                );
                eprintln!(
                    "[RUST-PI-ALL]   interp={} txmode={} deltaqres={} delfres={}",
                    pi.interpolation_filter, pi.TxMode, pi.delta_q_res, pi.delta_lf_res
                );
                let q = &picture_info_container.quantization;
                eprintln!(
                    "[RUST-PI-ALL]   quant: using_qmatrix={} diff_uv_delta={} base_q={} dQYdc={} dQUdc={} dQUac={} dQVdc={} dQVac={} qm_y={} qm_u={} qm_v={}",
                    q.flags.using_qmatrix(), q.flags.diff_uv_delta(), q.base_q_idx, q.DeltaQYDc, q.DeltaQUDc, q.DeltaQUAc, q.DeltaQVDc, q.DeltaQVAc, q.qm_y, q.qm_u, q.qm_v
                );
                let lf = &picture_info_container.loop_filter;
                eprintln!(
                    "[RUST-PI-ALL]   lf: delta_en={} delta_upd={} level=[{},{},{},{}] sharp={} updrefd={} refd=[{},{},{},{},{},{},{},{}] updmodes={} moded=[{},{}]",
                    lf.flags.loop_filter_delta_enabled(), lf.flags.loop_filter_delta_update(),
                    lf.loop_filter_level[0], lf.loop_filter_level[1], lf.loop_filter_level[2], lf.loop_filter_level[3],
                    lf.loop_filter_sharpness, lf.update_ref_delta,
                    lf.loop_filter_ref_deltas[0], lf.loop_filter_ref_deltas[1], lf.loop_filter_ref_deltas[2], lf.loop_filter_ref_deltas[3],
                    lf.loop_filter_ref_deltas[4], lf.loop_filter_ref_deltas[5], lf.loop_filter_ref_deltas[6], lf.loop_filter_ref_deltas[7],
                    lf.update_mode_delta, lf.loop_filter_mode_deltas[0], lf.loop_filter_mode_deltas[1]
                );
                let c = &picture_info_container.cdef;
                eprintln!(
                    "[RUST-PI-ALL]   cdef: damping={} bits={} ypri=[{},{},{},{}] ysec=[{},{},{},{}] uvprim=[{},{},{},{}] uvsec=[{},{},{},{}]",
                    c.cdef_damping_minus_3, c.cdef_bits,
                    c.cdef_y_pri_strength[0], c.cdef_y_pri_strength[1], c.cdef_y_pri_strength[2], c.cdef_y_pri_strength[3],
                    c.cdef_y_sec_strength[0], c.cdef_y_sec_strength[1], c.cdef_y_sec_strength[2], c.cdef_y_sec_strength[3],
                    c.cdef_uv_pri_strength[0], c.cdef_uv_pri_strength[1], c.cdef_uv_pri_strength[2], c.cdef_uv_pri_strength[3],
                    c.cdef_uv_sec_strength[0], c.cdef_uv_sec_strength[1], c.cdef_uv_sec_strength[2], c.cdef_uv_sec_strength[3]
                );
                let lr = &picture_info_container.loop_restoration;
                eprintln!(
                    "[RUST-PI-ALL]   lr: type=[{},{},{}] size=[{},{},{}]",
                    lr.FrameRestorationType[0], lr.FrameRestorationType[1], lr.FrameRestorationType[2],
                    lr.LoopRestorationSize[0], lr.LoopRestorationSize[1], lr.LoopRestorationSize[2]
                );
                let gm = &picture_info_container.global_motion;
                eprintln!(
                    "[RUST-PI-ALL]   gm: type=[{},{},{},{},{},{},{},{}]",
                    gm.GmType[0], gm.GmType[1], gm.GmType[2], gm.GmType[3], gm.GmType[4], gm.GmType[5], gm.GmType[6], gm.GmType[7]
                );
                for i in 0..8 {
                    eprintln!(
                        "[RUST-PI-ALL]   gm_params[{}]=[{},{},{},{},{},{}]",
                        i, gm.gm_params[i][0], gm.gm_params[i][1], gm.gm_params[i][2],
                        gm.gm_params[i][3], gm.gm_params[i][4], gm.gm_params[i][5]
                    );
                }
                let sg = &picture_info_container.segmentation;
                eprintln!(
                    "[RUST-PI-ALL]   seg: enabled=[{},{},{},{},{},{},{},{}]",
                    sg.FeatureEnabled[0], sg.FeatureEnabled[1], sg.FeatureEnabled[2], sg.FeatureEnabled[3],
                    sg.FeatureEnabled[4], sg.FeatureEnabled[5], sg.FeatureEnabled[6], sg.FeatureEnabled[7]
                );
                for i in 0..8 {
                    eprintln!(
                        "[RUST-PI-ALL]   segdata[{}]=[{},{},{},{},{},{},{},{}]",
                        i, sg.FeatureData[i][0], sg.FeatureData[i][1], sg.FeatureData[i][2], sg.FeatureData[i][3],
                        sg.FeatureData[i][4], sg.FeatureData[i][5], sg.FeatureData[i][6], sg.FeatureData[i][7]
                    );
                }
                eprintln!(
                    "[RUST-PI-ALL]   refNameSlotIndices={:?} tileCount=1",
                    reference_name_slot_indices
                );
                eprintln!(
                    "[RUST-PI-ALL]   orderHints={:?} expectedFrameId={:?} skipModeFrame={:?} coded_denom={} current_frame_id={}",
                    pi.OrderHints, pi.expectedFrameId, pi.SkipModeFrame, pi.coded_denom, pi.current_frame_id
                );
                eprintln!(
                    "[RUST-PI-ALL]   tile: tileOffsets[0]={} tileSizes[0]={}",
                    picture_info_container.tile_offsets[0], picture_info_container.tile_sizes[0]
                );
                let ti = &picture_info_container.tile_info;
                eprintln!(
                    "[RUST-PI-ALL]   tileinfo: uniform={} cols={} rows={} ctxid={} tsbm1={}",
                    ti.flags.uniform_tile_spacing_flag(), ti.TileCols, ti.TileRows,
                    ti.context_update_tile_id, ti.tile_size_bytes_minus_1
                );
                eprintln!(
                    "[RUST-PI-ALL]   tilewidth_sbs_m1={:?} tileheight_sbs_m1={:?}",
                    &picture_info_container.tile_width_in_sbs_minus_1,
                    &picture_info_container.tile_height_in_sbs_minus_1
                );
                eprintln!(
                    "[RUST-PI-ALL]   tilemicol={:?} tilerow={:?}",
                    &picture_info_container.tile_mi_col_starts,
                    &picture_info_container.tile_mi_row_starts
                );
            }

            // Capture the current frame's reference info BEFORE picture_info_container
            // is moved, so the refresh loop below can record it per frame buffer
            // (C++ VulkanAV1Decoder.cpp:390-394).
            let cur_order_hints = picture_info_container.std_picture_info.OrderHints;
            let cur_order_hint = fh.order_hint;
            let cur_frame_type = fh.frame_type;
            let cur_disable_cdf = fh.disable_cdf_update as u8;
            let cur_seg_enabled = fh.segmentation_enabled as u8;
            let cur_ohb = sps.order_hint_bits_minus1 as u32;

            let av1_decode_info = Box::new(VideoDecodeAV1PictureInfoKHR::new(
                picture_info_container.std_picture_info(),
                reference_name_slot_indices,
                frame_header_offset,
                1,
                picture_info_container.tile_offsets.as_ptr(),
                picture_info_container.tile_sizes.as_ptr(),
            ));

            let _picture_info_guard = picture_info_container;
            let _av1_decode_guard = av1_decode_info;
            // EXPERIMENT B (iteration 12): VACC_DUMP_REF_SLOT=1 captures the
            // reference slot (slot 0 / layer 0) IMMEDIATELY BEFORE frame 1's decode
            // command is recorded (i.e. after frame 0's readback, before frame 1
            // decode). Comparing this to the known-good frame 0 tells us whether
            // slot 0 was corrupted by frame 0's readback.
            if frame_idx == 1
                && std::env::var("VACC_DUMP_REF_SLOT")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                if let Some(&ref_slot) = reference_name_slot_indices.iter().find(|&&s| s >= 0) {
                    let ref_img = self.dpb_images[ref_slot as usize];
                    let (rw, rh) = (frame_coded_extent.width, frame_coded_extent.height);
                    let pixels = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        ref_img,
                        self.dpb_base_layer(ref_slot as u32),
                        rw,
                        rh,
                    )?;
                    // Save as planar YUV420P cropped to the conformance window
                    // (same layout as the example's frame_to_yuv420p).
                    let cw = rw as usize;
                    let dw = self.parsed.display_width as usize;
                    let dh = self.parsed.display_height as usize;
                    let cx = self.parsed.crop_left as usize;
                    let cy = self.parsed.crop_top as usize;
                    let uv_cx = cx / 2;
                    let uv_cy = cy / 2;
                    let uv_dw = dw / 2;
                    let uv_dh = dh / 2;
                    let mut out = Vec::with_capacity(dw * dh + uv_dw * uv_dh * 2);
                    for y in cy..cy + dh {
                        out.extend_from_slice(&pixels.y_plane[y * cw + cx..y * cw + cx + dw]);
                    }
                    for y in uv_cy..uv_cy + uv_dh {
                        out.extend_from_slice(&pixels.u_plane[y * (cw / 2) + uv_cx..y * (cw / 2) + uv_cx + uv_dw]);
                    }
                    for y in uv_cy..uv_cy + uv_dh {
                        out.extend_from_slice(&pixels.v_plane[y * (cw / 2) + uv_cx..y * (cw / 2) + uv_cx + uv_dw]);
                    }
                    let path = "/tmp/diag/slot0_before_f1.yuv";
                    std::fs::create_dir_all("/tmp/diag").ok();
                    std::fs::write(path, &out).ok();
                    eprintln!(
                        "[EXP-B] dumped reference slot {} ({}x{}) before frame 1 decode -> {} ({} bytes)",
                        ref_slot, rw, rh, path, out.len()
                    );
                }
            }

            // DEBUG (iteration 23): dump mean Y of ALL DPB slots before fc6 decode
            // (the first error frame) to check the DPB state.
            if frame_idx == 6
                && std::env::var("VACC_ALLSLOTS_PROBE")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                for slot in 0..self.dpb_images.len() as u32 {
                    let px = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        self.dpb_images[slot as usize],
                        self.dpb_base_layer(slot),
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    )?;
                    let mean_y = px.y_plane.iter().map(|&b| b as u32).sum::<u32>() as f64
                        / px.y_plane.len().max(1) as f64;
                    eprintln!("[ALLSLOTS-PROBE] before fc6 slot {}: meanY={:.3}", slot, mean_y);
                }
            }

            // Record decode command
            let cmd_buffer = allocate_command_buffer(&self.vulkan.device, self.command_pool)?;

            let output_slot_old_layout = self.dpb_manager.get_slot_layout(output_slot);
            let result = av1_decoder.record_decode_command(
                cmd_buffer,
                self.session.handle(),
                self.session_params
                    .as_ref()
                    .map(|p| p.handle())
                    .unwrap_or(vk::VideoSessionParametersKHR::null()),
                self.bs_buffer.buffer(),
                0,
                actual_size,
                output_view,
                output_img,
                frame_coded_extent,
                dpb_setup_picture,
                &dpb_ref_pictures,
                &dpb_ref_slot_indices,
                &dpb_ref_order_hints,
                &dpb_ref_images,
                &dpb_ref_slot_layouts,
                &_picture_info_guard,
                &_av1_decode_guard,
                is_first_frame,
                output_slot as i32,
                output_slot_old_layout,
                self.dpb_use_image_array,
            );

            if is_first_frame {
                is_first_frame = false;
            }

            result.map_err(|e| VideoError::DecoderInit(format!("AV1 decode failed: {}", e)))?;

            // Submit
            unsafe {
                self.vulkan
                    .device
                    .reset_fences(&[self.fence])
                    .map_err(|e| VideoError::FenceWait(e.to_string()))?;
                let queue = self
                    .vulkan
                    .device
                    .get_device_queue(self.decode_queue_family, 0);
                self.vulkan
                    .device
                    .queue_submit(
                        queue,
                        &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                        self.fence,
                    )
                    .map_err(|e| VideoError::QueueSubmission(e.to_string()))?;
                self.vulkan
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| VideoError::FenceWait(e.to_string()))?;
            }

            // Update frame buffer to DPB slot mapping based on refresh_frame_flags
            // AV1 frame buffer indices:
            // 0 = INTRA_FRAME (never refreshed)
            // 1 = LAST_FRAME
            // 2 = LAST2_FRAME
            // 3 = LAST3_FRAME
            // 4 = GOLDEN_FRAME
            // 5 = BWDREF_FRAME
            // 6 = ALTREF2_FRAME
            // 7 = ALTREF_FRAME
            //
            // refresh_frame_flags bit i corresponds to frame buffer (i+1):
            // bit0=LAST(fb1), bit1=LAST2(fb2), ..., bit6=ALTREF(fb7).
            // The mapping from ref_frame_idx to frame buffers:
            // ref_frame_idx[0] -> LAST_FRAME -> frame buffer 1
            // ref_frame_idx[1] -> LAST2_FRAME -> frame buffer 2
            // ref_frame_idx[2] -> LAST3_FRAME -> frame buffer 3
            // ref_frame_idx[3] -> GOLDEN_FRAME -> frame buffer 4
            // ref_frame_idx[4] -> BWDREF_FRAME -> frame buffer 5
            // ref_frame_idx[5] -> ALTREF2_FRAME -> frame buffer 6
            // ref_frame_idx[6] -> ALTREF_FRAME -> frame buffer 7

            // Record which DPB slot contains each refreshed frame buffer.
            // Matches C++ VulkanAV1Decoder::UpdateFramePointers (line 379-417):
            // `ref_index` starts at 0 and increments per bit, so refresh_frame_flags
            // bit i -> frame buffer i (NOT i+1). This means frame buffer 0 is
            // refreshed too (for a KEY frame with refresh_frame_flags=0xFF, all
            // 8 frame buffers 0..7 point to the key frame). The bitstream's
            // ref_frame_idx uses the same indexing (e.g. frame 1 has
            // ref_frame_idx=[0,0,0,0,0,0,0] -> all reference frame buffer 0).
            for i in 0..8usize {
                if (fh.refresh_frame_flags & (1 << i)) != 0 {
                    let fb = i; // bit i -> frame buffer i (C++ UpdateFramePointers)
                    av1_decoder.set_frame_buffer_dpb_slot(fb, output_slot as i32);
                    av1_decoder.set_frame_buffer_order_hint(fb, fh.order_hint);
                    av1_decoder.set_frame_buffer_dims(
                        fb,
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    );
                    av1_decoder.set_frame_buffer_ref_info(
                        fb,
                        &cur_order_hints,
                        cur_order_hint,
                        cur_ohb,
                        cur_frame_type,
                        cur_disable_cdf,
                        cur_seg_enabled,
                    );
                }
            }

            self.dpb_manager.register_frame(output_slot, frame_count);
            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

            // DEBUG: print frame buffer -> DPB slot state after this frame
            {
                let mut fb_state = String::new();
                for i in 0..8usize {
                    let slot = av1_decoder.get_pic_idx_for_frame_buffer(i);
                    fb_state.push_str(&format!("{}:{}, ", i, slot));
                }
                eprintln!(
                    "[DPB] after frame {} (out_slot={}): {}",
                    frame_idx, output_slot, fb_state
                );
            }

            // DEBUG (iteration 18): dump mean Y of each reference slot BEFORE decode
            // for multi-ref INTER frames to check if references differ.
            // Extended (iter 21): probe fc=3/4/5 (the broken + working multi-ref frames).
            if (frame_idx == 3 || frame_idx == 4 || frame_idx == 5)
                && std::env::var("VACC_REF_PROBE")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                for &slot in &dpb_ref_slot_indices {
                    if slot < 0 {
                        continue;
                    }
                    let px = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        self.dpb_images[slot as usize],
                        self.dpb_base_layer(slot as u32),
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    )?;
                    let mean_y = px.y_plane.iter().map(|&b| b as u32).sum::<u32>() as f64
                        / px.y_plane.len().max(1) as f64;
                    eprintln!("[REF-PROBE] ext{} ref slot {}: meanY={:.3}", frame_idx, slot, mean_y);
                }
            }

            // DEBUG (iteration 20b): AFTER the keyframe (fc=0) decode, read back
            // ALL DPB slots to see their initial content. If slots 3/4 already
            // contain the keyframe before ext3 is decoded, the fc=3/fc=4 decodes
            // are not writing (or something pre-fills the DPB with the keyframe).
            if frame_idx == 0
                && std::env::var("VACC_OUT_PROBE")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                for slot in 0..self.dpb_images.len() as u32 {
                    let px = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        self.dpb_images[slot as usize],
                        self.dpb_base_layer(slot),
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    )?;
                    let mean_y = px.y_plane.iter().map(|&b| b as u32).sum::<u32>() as f64
                        / px.y_plane.len().max(1) as f64;
                    eprintln!("[OUT-PROBE] fc0 slot {}: meanY={:.3}", slot, mean_y);
                }
            }

            // DEBUG (iteration 20): AFTER ext2/ext3/ext4 decode (multi-ref
            // frames), read back the output slot and slot 0 (keyframe) and
            // compute full MAD to see which frames still decode to the first
            // reference's content.
            if (frame_idx == 2 || frame_idx == 3 || frame_idx == 4)
                && std::env::var("VACC_OUT_PROBE")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                let probe_slots: [i32; 2] = [output_slot as i32, 0];
                let mut pixels: Vec<super::readback::DecodedPixels> = Vec::new();
                for &slot in &probe_slots {
                    let px = super::readback::readback_decoded_image(
                        &self.vulkan.instance,
                        &self.vulkan.device,
                        &self.vulkan.memory_properties,
                        self.decode_queue_family,
                        self.command_pool,
                        self.fence,
                        self.dpb_images[slot as usize],
                        self.dpb_base_layer(slot as u32),
                        frame_coded_extent.width,
                        frame_coded_extent.height,
                    )?;
                    let mean_y = px.y_plane.iter().map(|&b| b as u32).sum::<u32>() as f64
                        / px.y_plane.len().max(1) as f64;
                    eprintln!(
                        "[OUT-PROBE] ext{} after decode slot {}: meanY={:.3}",
                        frame_idx, slot, mean_y
                    );
                    pixels.push(px);
                }
                let (out, key) = (&pixels[0], &pixels[1]);
                let n = out.y_plane.len();
                let mad_y = out.y_plane.iter().zip(key.y_plane.iter())
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u32)
                    .sum::<u32>() as f64 / n.max(1) as f64;
                let maxd = out.y_plane.iter().zip(key.y_plane.iter())
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                    .max().unwrap_or(0);
                let nz = out.y_plane.iter().zip(key.y_plane.iter())
                    .filter(|(a, b)| a != b).count();
                eprintln!(
                    "[OUT-PROBE] ext{} out(slot {}) vs key(slot 0): MAD_y={:.4} maxdiff={} nonzero={}",
                    frame_idx, output_slot, mad_y, maxd, nz
                );
            }

            // Only read back + output display frames (show_frame=1). Non-display
            // frames are decoded (to update DPB state) but not output (C++ behavior).
            if !fh.show_frame {
                frame_count += 1;
                continue;
            }

            // Readback. VACC_STAGING_READBACK=1 routes the readback through a
            // separate staging image so the DPB slot is never transitioned out of
            // VIDEO_DECODE_DPB_KHR (tests whether the direct readback's layout
            // transition corrupts reference data for future frames).
            let use_staging = std::env::var("VACC_STAGING_READBACK")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false);
            // EXPERIMENT A (iteration 12): VACC_SKIP_READBACK=1 skips the readback
            // of frame 0 entirely (decode frame 0 into slot 0, do NOT read it back,
            // then decode frame 1 referencing slot 0). Tests whether the act of
            // reading back frame 0 corrupts the reference used by frame 1.
            let skip_readback = std::env::var("VACC_SKIP_READBACK")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false)
                && frame_idx == 0;
            let pixels = if skip_readback {
                eprintln!("[EXP-A] skipping readback of frame 0 (slot 0 left untouched)");
                let y_size = (frame_coded_extent.width * frame_coded_extent.height) as usize;
                let uv_w = frame_coded_extent.width.div_ceil(2) as usize;
                let uv_h = frame_coded_extent.height.div_ceil(2) as usize;
                super::readback::DecodedPixels {
                    y_plane: vec![0u8; y_size],
                    u_plane: vec![0u8; uv_w * uv_h],
                    v_plane: vec![0u8; uv_w * uv_h],
                }
            } else if use_staging {
                super::readback::readback_decoded_image_via_staging(
                    &self.vulkan.instance,
                    &self.vulkan.device,
                    &self.vulkan.memory_properties,
                    self.decode_queue_family,
                    self.command_pool,
                    self.fence,
                    output_img,
                    self.dpb_base_layer(output_slot as u32),
                    frame_coded_extent.width,
                    frame_coded_extent.height,
                )?
            } else {
                super::readback::readback_decoded_image(
                    &self.vulkan.instance,
                    &self.vulkan.device,
                    &self.vulkan.memory_properties,
                    self.decode_queue_family,
                    self.command_pool,
                    self.fence,
                    output_img,
                    self.dpb_base_layer(output_slot as u32),
                    frame_coded_extent.width,
                    frame_coded_extent.height,
                )?
            };

            // VACC_LAYER_PROBE=1: after frame 3's decode, read layer 0 and layer 2
            // of the DPB image and print the mean Y of each (plus the output slot),
            // to diagnose WHERE the decode writes: does frame 3's content land in its
            // own output slot, or does the decode overwrite layer 0 / leave layer 2
            // holding frame 0's content?
            if frame_idx == 3
                && std::env::var("VACC_LAYER_PROBE")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
            {
                let mean_y = |px: &super::readback::DecodedPixels| -> f64 {
                    px.y_plane
                        .iter()
                        .map(|&b| b as u32)
                        .sum::<u32>() as f64
                        / px.y_plane.len().max(1) as f64
                };
                let out_mean = mean_y(&pixels);
                let l0 = super::readback::readback_decoded_image(
                    &self.vulkan.instance,
                    &self.vulkan.device,
                    &self.vulkan.memory_properties,
                    self.decode_queue_family,
                    self.command_pool,
                    self.fence,
                    output_img,
                    0,
                    frame_coded_extent.width,
                    frame_coded_extent.height,
                )?;
                let l0_mean = mean_y(&l0);
                let l2 = super::readback::readback_decoded_image(
                    &self.vulkan.instance,
                    &self.vulkan.device,
                    &self.vulkan.memory_properties,
                    self.decode_queue_family,
                    self.command_pool,
                    self.fence,
                    output_img,
                    2,
                    frame_coded_extent.width,
                    frame_coded_extent.height,
                )?;
                let l2_mean = mean_y(&l2);
                eprintln!(
                    "[LAYER_PROBE] frame 3: output_slot={} | out(layer {}) meanY={:.3} | layer0 meanY={:.3} | layer2 meanY={:.3}",
                    output_slot, output_slot, out_mean, l0_mean, l2_mean
                );
            }

            self.dpb_manager
                .set_slot_layout(output_slot, vk::ImageLayout::VIDEO_DECODE_DPB_KHR);

            decoded_frames.push(DecodedFrame {
                poc: frame_count as i32,
                frame_num: frame_count,
                is_idr: is_key_frame,
                is_reference: true,
                pixels,
                coded_width: frame_coded_extent.width,
                coded_height: frame_coded_extent.height,
                display_width: self.parsed.display_width,
                display_height: self.parsed.display_height,
                crop_left: self.parsed.crop_left,
                crop_top: self.parsed.crop_top,
            });

            frame_count += 1;
            display_count += 1;
            if display_count >= max_frames {
                break;
            }
        }

        Ok(decoded_frames)
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
        let cmd_buffer = allocate_command_buffer(&self.vulkan.device, self.command_pool)?;

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
                self.record_h264_decode(
                    cmd_buffer,
                    au,
                    output_view,
                    output_img,
                    bs_size,
                    output_slot,
                    is_first_frame,
                )?;
            }
            VideoCodec::DecodeH265 => {
                self.record_h265_decode(
                    cmd_buffer,
                    au,
                    output_view,
                    output_img,
                    bs_size,
                    output_slot,
                    is_first_frame,
                )?;
            }
            _ => {}
        }

        unsafe {
            self.vulkan
                .device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;

            self.vulkan
                .device
                .reset_fences(&[self.fence])
                .map_err(|e| VideoError::FenceWait(e.to_string()))?;

            let queue = self
                .vulkan
                .device
                .get_device_queue(self.decode_queue_family, 0);
            self.vulkan
                .device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                    self.fence,
                )
                .map_err(|e| VideoError::QueueSubmission(e.to_string()))?;

            self.vulkan
                .device
                .wait_for_fences(&[self.fence], true, 10_000_000_000)
                .map_err(|e| VideoError::FenceWait(e.to_string()))?;
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
        let sps = match &self.parsed.sps {
            Some(H264OrH265Sps::H264(s)) => s,
            _ => return Err(VideoError::DecoderInit("H264 SPS not found".to_string())),
        };
        let pps = match &self.parsed.pps {
            Some(H264OrH265Pps::H264(p)) => p,
            _ => return Err(VideoError::DecoderInit("H264 PPS not found".to_string())),
        };

        let mut h264_decoder =
            H264Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
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
            self.session_params
                .as_ref()
                .map(|p| p.handle())
                .unwrap_or(vk::VideoSessionParametersKHR::null()),
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
        _is_first_frame: bool,
    ) -> VideoResult<()> {
        let vps = &self.parsed.vps;
        let sps = match &self.parsed.sps {
            Some(H264OrH265Sps::H265(s)) => s,
            _ => return Err(VideoError::DecoderInit("H265 SPS not found".to_string())),
        };
        let pps = match &self.parsed.pps {
            Some(H264OrH265Pps::H265(p)) => p,
            _ => return Err(VideoError::DecoderInit("H265 PPS not found".to_string())),
        };

        let mut h265_decoder =
            H265Decoder::new(self.vulkan.device.clone(), self.vulkan.instance.clone());
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
            self.session_params
                .as_ref()
                .map(|p| p.handle())
                .unwrap_or(vk::VideoSessionParametersKHR::null()),
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
            au.num_bits_for_st_ref_pic_set_in_slice,
            au.num_delta_pocs_of_ref_rps_idx,
            &au.ref_pocs,
            &self.dpb_manager.get_references(),
        )?;

        Ok(())
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        unsafe {
            self.vulkan.device.device_wait_idle().ok();

            // Destroy command resources first
            self.vulkan
                .device
                .destroy_command_pool(self.command_pool, None);
            self.vulkan.device.destroy_fence(self.fence, None);

            // Destroy DPB resources. In image-array mode (dpb_use_image_array)
            // every slot shares ONE image and ONE memory allocation, so the
            // image/memory must be destroyed only once (dedup by handle).
            for view in self.dpb_views.drain(..) {
                self.vulkan.device.destroy_image_view(view, None);
            }
            let mut destroyed_images = std::collections::HashSet::new();
            for img in self.dpb_images.drain(..) {
                if destroyed_images.insert(img) {
                    self.vulkan.device.destroy_image(img, None);
                }
            }
            for mem in self.dpb_memories.drain(..) {
                self.vulkan.device.free_memory(mem, None);
            }

            // Destroy bitstream buffer BEFORE device
            self.bs_buffer = BitstreamBuffer::null(&self.vulkan.device);

            // Destroy session resources
            destroy_session_parameters(
                &self.vulkan.instance,
                self.vulkan.device.handle(),
                self.session_params
                    .as_ref()
                    .map(|p| p.handle())
                    .unwrap_or(vk::VideoSessionParametersKHR::null()),
            );
            if let Some(ref mut params) = self.session_params {
                params.reset();
            }
            destroy_session(
                &self.vulkan.instance,
                self.vulkan.device.handle(),
                self.session.handle(),
            );
            self.session.reset();

            for mem in self.session_memories.drain(..) {
                self.vulkan.device.free_memory(mem, None);
            }

            // Destroy debug messenger BEFORE destroying device/instance
            if self.vulkan.has_validation
                && self.vulkan.debug_messenger != vk::DebugUtilsMessengerEXT::null()
            {
                let debug_utils =
                    ash::ext::debug_utils::Instance::new(&self.vulkan.entry, &self.vulkan.instance);
                debug_utils.destroy_debug_utils_messenger(self.vulkan.debug_messenger, None);
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
    // Check for IVF container
    if data.len() >= 32 && data[0..4] == *b"DKIF" {
        // Check codec FourCC at offset 8-11 (4 bytes)
        // "VP90" for VP9, "AV01" for AV1
        if data.len() >= 12 {
            let codec = &data[8..12];
            if codec == b"AV01" {
                return AccessUnitCodec::Av1;
            }
            if codec == b"VP90" {
                return AccessUnitCodec::Vp9;
            }
        }
        // Default IVF to VP9 for backwards compatibility
        return AccessUnitCodec::Vp9;
    }

    // Check for VP9 frame marker (top 2 bits = 0b10)
    // Skip leading zeros and check for frame marker
    for &byte in &data[..data.len().min(256)] {
        if byte == 0 {
            continue;
        }
        if (byte & 0xC0) == 0x80 {
            return AccessUnitCodec::Vp9;
        }
        break;
    }

    // First pass: prioritize SPS/VPS detection for unambiguous identification
    // H.264: SPS=7, PPS=8; H.265: VPS=32, SPS=33, PPS=34
    for i in 0..data.len().min(4096) {
        let start = if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i + 3] == [0x00, 0x00, 0x01] {
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
        let start = if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i + 3] == [0x00, 0x00, 0x01] {
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
        bitstream::BitstreamPacket, h264::H264Parser, DetectedVideoFormat, ParseResult, VideoParser,
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
        bitstream::BitstreamPacket, h265::H265Parser, DetectedVideoFormat, ParseResult, VideoParser,
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
            let _sub_width_c = 1u32 << log2_sub_width_c;
            let _sub_height_c = 1u32 << log2_sub_height_c;

            let ctb_size = 1u32 << (s.log2_min_luma_coding_block_size_minus3 as u32 + 3);

            let pic_width = s.pic_width_in_luma_samples as u32;
            let pic_height = s.pic_height_in_luma_samples as u32;

            let (crop_left, crop_top) = if s.conformance_window_flag {
                (
                    s.conf_win_left_offset * ctb_size,
                    s.conf_win_top_offset * ctb_size,
                )
            } else {
                (0, 0)
            };

            let display_width = if s.conformance_window_flag {
                let left_right = (s.conf_win_left_offset + s.conf_win_right_offset) * ctb_size;
                pic_width.saturating_sub(left_right)
            } else {
                pic_width
            };

            let display_height = if s.conformance_window_flag {
                let top_bottom = (s.conf_win_top_offset + s.conf_win_bottom_offset) * ctb_size;
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
    av1_sps: Option<&vk_video_core::picture::Av1Sps>,
) -> VideoResult<(
    VideoSession,
    Option<VideoSessionParameters>,
    Vec<vk::DeviceMemory>,
)> {
    let output_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let (codec_profile_info, std_header_name) = match codec {
        VideoCodec::DecodeH264 => (
            CodecProfileInfo::H264 {
                std_profile_idc: parsed.profile_idc,
                picture_layout: vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE.as_raw(),
            },
            "VK_STD_vulkan_video_codec_h264_decode",
        ),
        VideoCodec::DecodeH265 => (
            CodecProfileInfo::H265 {
                std_profile_idc: parsed.profile_idc,
            },
            "VK_STD_vulkan_video_codec_h265_decode",
        ),
        VideoCodec::DecodeVp9 => (
            CodecProfileInfo::Vp9 {
                std_profile: parsed.profile_idc,
            },
            "VK_STD_vulkan_video_codec_vp9_decode",
        ),
        VideoCodec::DecodeAv1 => (
            CodecProfileInfo::Av1 {
                std_profile: parsed.profile_idc,
                film_grain_support: false,
            },
            "VK_STD_vulkan_video_codec_av1_decode",
        ),
    };

    let session_params = VideoSessionParams {
        queue_family_index: vulkan.queue_families.video_decode.unwrap(),
        picture_format: output_format,
        reference_picture_format: output_format,
        max_coded_extent: coded_extent,
        max_dpb_slots,
        // VUID-VkVideoSessionCreateInfoKHR-maxActiveReferencePictures-04831:
        // maxActiveReferencePictures MUST be strictly less than maxDpbSlots,
        // leaving room for the setup/current slot. The C++ reference uses
        // maxDpbSlots - 1. Setting them equal made the NVIDIA driver silently
        // skip the decode (no room for the current picture) -> all-zero output.
        max_active_reference_pictures: max_dpb_slots.saturating_sub(1),
        codec,
        codec_profile_info,
        chroma_subsampling: parsed.chroma_subsampling,
        luma_bit_depth: parsed.luma_bit_depth,
        chroma_bit_depth: parsed.chroma_bit_depth,
    };

    let std_header_version = build_std_header_version(std_header_name);

    eprintln!("[Session] Creating video session...");
    let (session, session_memories) = VideoSession::create(
        &vulkan.instance,
        &vulkan.device,
        &session_params,
        &std_header_version,
    )?;
    eprintln!("[Session] Video session created, binding memory...");

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

    // VP9 doesn't use session parameters (per Vulkan spec with maintenance1).
    // AV1 DOES need session parameters: the SPS is passed via
    // VkVideoDecodeAV1SessionParametersCreateInfoKHR::p_std_sequence_header.
    // The session is then initialized with the parameters (see update_session).
    let session_parameters = if codec == VideoCodec::DecodeVp9 {
        None
    } else {
        eprintln!("[Session] Creating session parameters...");
        let params = VideoSessionParameters::create(
            &vulkan.instance,
            &vulkan.device,
            session.handle(),
            codec,
            h264_sps,
            h264_pps,
            h265_vps,
            h265_sps,
            h265_pps,
            av1_sps,
        )?;
        eprintln!("[Session] Session parameters created, updating session...");
        // Initialize the session with the session parameters via vkUpdateVideoSessionKHR
        params.update_session(session.handle())?;
        eprintln!("[Session] Session updated");
        Some(params)
    };

    eprintln!("[Session] create_video_session completed");

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
    props.spec_version = 1u32 << 22;
    props
}

fn create_command_pool(device: &ash::Device, queue_family: u32) -> VideoResult<vk::CommandPool> {
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
            c"vkDestroyVideoSessionParametersKHR".as_ptr(),
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

fn destroy_session(instance: &ash::Instance, device: vk::Device, session: vk::VideoSessionKHR) {
    if session.is_null() {
        return;
    }
    if let Some(ptr) = unsafe {
        instance.get_device_proc_addr(device, c"vkDestroyVideoSessionKHR".as_ptr())
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

#[allow(dead_code)]
fn extract_max_au_size(
    data: &[u8],
    codec: AccessUnitCodec,
    max_frames: usize,
    parsed: &ParsedInfo,
) -> usize {
    let items = super::access_unit::extract_all_access_units(
        data,
        codec,
        max_frames,
        parsed.sps.as_ref(),
        parsed.pps.as_ref(),
    );

    items
        .iter()
        .filter_map(|item| {
            if let ExtractedItem::AccessUnit(au) = item {
                Some(au.data.len())
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
}

fn extract_max_frame_size(data: &[u8], codec: AccessUnitCodec, max_frames: usize) -> usize {
    match codec {
        AccessUnitCodec::H264 | AccessUnitCodec::H265 => {
            // For H264/H265, we need parsed info which is computed elsewhere
            // This is a fallback - the actual size is computed in the codec-specific path
            data.len()
        }
        AccessUnitCodec::Vp9 => {
            let frames = super::access_unit::extract_vp9_frames(data, max_frames);
            frames.iter().map(|f| f.data.len()).max().unwrap_or(0)
        }
        AccessUnitCodec::Av1 => {
            let frames = super::access_unit::extract_av1_frames(data, max_frames);
            frames.iter().map(|f| f.data.len()).max().unwrap_or(0)
        }
    }
}

// ============================================================================
// VP9-specific helpers
// ============================================================================

fn parse_vp9_init(data: &[u8]) -> VideoResult<(ParsedInfo, VulkanDevice, u32, u32, vk::Extent2D)> {
    use vk_video_core::codec::VideoCodec as CoreCodec;
    use vk_video_parser::vp9::Vp9Parser;
    use vk_video_parser::{DetectedVideoFormat, VideoParser};

    // Extract first frame for parsing
    let raw_frames = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        match super::access_unit::parse_ivf_container(data) {
            Ok(frames) => frames,
            Err(_) => vec![data.to_vec()],
        }
    } else {
        let expanded = super::access_unit::expand_superframes(&[data.to_vec()]);
        super::access_unit::split_vp9_bitstream(&expanded)
    };

    if raw_frames.is_empty() {
        return Err(VideoError::DecoderInit("No VP9 frames found".to_string()));
    }

    // Parse first frame
    let mut parser = Vp9Parser::new();
    parser
        .init(&DetectedVideoFormat::new(CoreCodec::DecodeVp9))
        .map_err(|e| VideoError::DecoderInit(e.to_string()))?;

    let parsed = parser.parse_frame(&raw_frames[0]).map_err(|e| {
        VideoError::DecoderInit(format!("Failed to parse first VP9 frame: {:?}", e))
    })?;

    let coded_width = parsed.frame_width;
    let coded_height = parsed.frame_height;
    let display_width = parsed.render_width;
    let display_height = parsed.render_height;
    let profile = parsed.picture_info.profile as u32;
    let bit_depth = parsed.color_config.bit_depth;

    if coded_width == 0 || coded_height == 0 {
        return Err(VideoError::DecoderInit(
            "Failed to parse VP9 dimensions".to_string(),
        ));
    }

    // Initialize Vulkan with VP9 decode support
    let vulkan = super::VideoDeviceBuilder::new()
        .with_validation(false)
        .build()?;

    let luma_bit_depth = match bit_depth {
        8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };
    let chroma_bit_depth = luma_bit_depth;
    let chroma_subsampling = vk::VideoChromaSubsamplingFlagsKHR::TYPE_420;

    // VP9 has 8 DPB slots
    let max_dpb_slots = 8u32;
    let session_dpb_slots = max_dpb_slots + 1;
    let dpb_slots = max_dpb_slots;

    let coded_extent = vk::Extent2D {
        width: coded_width,
        height: coded_height,
    };

    Ok((
        ParsedInfo {
            vps: None,
            sps: None,
            pps: None,
            coded_width,
            coded_height,
            display_width,
            display_height,
            crop_left: 0,
            crop_top: 0,
            profile_idc: profile,
            max_dpb_slots,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
        },
        vulkan,
        session_dpb_slots,
        dpb_slots,
        coded_extent,
    ))
}

fn parse_av1_init(
    data: &[u8],
) -> VideoResult<(
    ParsedInfo,
    VulkanDevice,
    u32,
    u32,
    vk::Extent2D,
    Option<vk_video_core::picture::Av1Sps>,
)> {
    use vk_video_core::codec::VideoCodec as CoreCodec;
    use vk_video_parser::av1::Av1Parser;
    use vk_video_parser::{DetectedVideoFormat, VideoParser};

    // Extract first frame for parsing
    let raw_frames = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        match super::access_unit::parse_ivf_container(data) {
            Ok(frames) => frames,
            Err(_) => vec![data.to_vec()],
        }
    } else {
        vec![data.to_vec()]
    };

    if raw_frames.is_empty() {
        return Err(VideoError::DecoderInit("No AV1 frames found".to_string()));
    }

    // Parse SPS from first frame
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(CoreCodec::DecodeAv1))
        .map_err(|e| VideoError::DecoderInit(e.to_string()))?;

    // Try to parse sequence header from first frame
    let sps = parse_av1_sps_from_data(&raw_frames[0], &mut parser);

    // TEMP DIAGNOSTIC (iteration 5): print parsed SPS fields
    if let Some(ref s) = sps {
        eprintln!(
            "[SPS-DIAG] profile={} high_bitdepth={} twelve_bit={} max_w={} max_h={} mono_chrome={} subsampling_x={} subsampling_y={}",
            s.profile, s.high_bitdepth, s.twelve_bit,
            s.max_frame_width_minus_1, s.max_frame_height_minus_1,
            s.mono_chrome, s.subsampling_x, s.subsampling_y
        );
    } else {
        eprintln!("[SPS-DIAG] SPS parse FAILED");
    }

    let (coded_width, coded_height, profile, bit_depth) = if let Some(ref s) = sps {
        (
            s.max_frame_width_minus_1 as u32 + 1,
            s.max_frame_height_minus_1 as u32 + 1,
            s.profile as u32,
            if s.high_bitdepth {
                if s.twelve_bit { 12 } else { 10 }
            } else {
                8
            },
        )
    } else {
        (0, 0, 0, 8)
    };

    if coded_width == 0 || coded_height == 0 {
        return Err(VideoError::DecoderInit(
            "Failed to parse AV1 dimensions".to_string(),
        ));
    }

    // Initialize Vulkan with AV1 decode support
    let vulkan = super::VideoDeviceBuilder::new()
        .with_validation(false)
        .build()?;

    let luma_bit_depth = match bit_depth {
        8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        _ => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
    };
    let chroma_bit_depth = luma_bit_depth;
    let chroma_subsampling = vk::VideoChromaSubsamplingFlagsKHR::TYPE_420;

    // AV1 DPB sizing — must match the C++ reference exactly:
    //   BeginSequence: maxDpbSlots = STD_VIDEO_AV1_NUM_REF_FRAMES + 1 = 9
    //   VulkanVideoSession::Create: createInfo.maxDpbSlots = maxDpbSlots + 1 = 10
    //   createInfo.maxActiveReferencePictures = 9
    // The NVIDIA driver silently skips every decode (all-zero output) when the
    // session is created with maxDpbSlots=9 / maxActiveReferencePictures=8.
    let max_dpb_slots = 9u32;
    let session_dpb_slots = max_dpb_slots + 1;
    let dpb_slots = max_dpb_slots;

    let coded_extent = vk::Extent2D {
        width: coded_width,
        height: coded_height,
    };

    Ok((
        ParsedInfo {
            vps: None,
            sps: None,
            pps: None,
            coded_width,
            coded_height,
            display_width: coded_width,
            display_height: coded_height,
            crop_left: 0,
            crop_top: 0,
            profile_idc: profile,
            max_dpb_slots,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
        },
        vulkan,
        session_dpb_slots,
        dpb_slots,
        coded_extent,
        sps,
    ))
}

/// Parse AV1 SPS from frame data using the parser.
fn parse_av1_sps_from_data(
    data: &[u8],
    parser: &mut vk_video_parser::av1::Av1Parser,
) -> Option<vk_video_core::picture::Av1Sps> {
    use vk_video_parser::{bitstream::BitstreamPacket, ParseResult, VideoParser};

    if data.is_empty() {
        return None;
    }

    // Extract raw AV1 frame data from IVF container if needed
    let frame_data = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        match crate::access_unit::parse_ivf_container(data) {
            Ok(frames) => frames.into_iter().next().unwrap_or_else(|| data.to_vec()),
            Err(_) => data.to_vec(),
        }
    } else {
        data.to_vec()
    };

    let packet = BitstreamPacket::new(frame_data);
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps: s, .. }) => {
            s.and_then(|sp| sp.downcast_ref::<vk_video_core::picture::Av1Sps>().cloned())
        }
        Ok(r) => {
            None
        }
        Err(e) => {
            None
        }
    }
}

fn build_vp9_dpb_picture_resources(
    dpb_manager: &DpbManager,
    dpb_views: &[vk::ImageView],
    coded_extent: vk::Extent2D,
    output_slot: u32,
    is_key_frame: bool,
    reference_name_slot_indices: &[i32; 3],
) -> (
    Option<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<i32>,
) {
    let mut ref_pictures = Vec::new();
    let mut ref_slot_indices = Vec::new();

    if !is_key_frame {
        let mut seen_slots: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &slot_idx in reference_name_slot_indices.iter() {
            if slot_idx < 0 || seen_slots.contains(&slot_idx) {
                continue;
            }
            let slot = slot_idx as usize;
            if slot >= dpb_manager.entries.len() || (slot as u32) == output_slot {
                continue;
            }
            let entry = &dpb_manager.entries[slot];
            if !entry.is_valid {
                continue;
            }
            seen_slots.insert(slot_idx);
            let view = dpb_views[slot];
            let picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: view,
                _marker: Default::default(),
            };
            ref_pictures.push(picture_resource);
            ref_slot_indices.push(slot_idx);
        }
    }

    let setup_picture = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: dpb_views[output_slot as usize],
        _marker: Default::default(),
    };

    (Some(setup_picture), ref_pictures, ref_slot_indices)
}

// ============================================================================
// AV1-specific helpers
// ============================================================================

fn build_av1_dpb_picture_resources(
    dpb_manager: &DpbManager,
    dpb_views: &[vk::ImageView],
    coded_extent: vk::Extent2D,
    output_slot: u32,
    is_key_frame: bool,
    reference_name_slot_indices: &[i32; 7],
    av1_decoder: &super::av1::Av1Decoder,
) -> (
    Option<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<vk::VideoPictureResourceInfoKHR<'static>>,
    Vec<i32>,
) {
    let mut ref_pictures = Vec::new();
    let mut ref_slot_indices = Vec::new();

    if !is_key_frame {
        let mut seen_slots: std::collections::HashSet<i32> = std::collections::HashSet::new();
        // ORDER MATTERS (iteration 19): the C++ reference (VulkanVideoParser.cpp
        // FillDpbAV1State, lines 1812-1839) builds pReferenceSlots by iterating
        // FRAME BUFFER index 0..8 (INTRA, LAST, LAST2, LAST3, GOLDEN, BWDREF,
        // ALTREF) and adding a reference slot for each frame buffer whose picture
        // is an active reference. The driver evidently maps reference names to
        // reference slots by this position, so we must match it exactly.
        // (Previously we iterated reference_name_slot_indices in reference-name
        // order LAST..INTRA, which placed INTRA last instead of first.)
        let active_slots: std::collections::HashSet<i32> = reference_name_slot_indices
            .iter()
            .copied()
            .filter(|&s| s >= 0)
            .collect();
        for fb in 0..8 {
            let slot_idx = av1_decoder.get_pic_idx_for_frame_buffer(fb);
            if slot_idx < 0 || seen_slots.contains(&slot_idx) {
                continue;
            }
            if !active_slots.contains(&slot_idx) {
                continue;
            }
            let slot = slot_idx as usize;
            if slot >= dpb_manager.entries.len() || (slot as u32) == output_slot {
                continue;
            }
            let entry = &dpb_manager.entries[slot];
            if !entry.is_valid {
                continue;
            }
            seen_slots.insert(slot_idx);
            let view = dpb_views[slot];
            let picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: view,
                _marker: Default::default(),
            };
            ref_pictures.push(picture_resource);
            ref_slot_indices.push(slot_idx);
        }
    }

    let setup_picture = vk::VideoPictureResourceInfoKHR {
        s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
        p_next: std::ptr::null(),
        coded_offset: vk::Offset2D::default(),
        coded_extent,
        base_array_layer: 0,
        image_view_binding: dpb_views[output_slot as usize],
        _marker: Default::default(),
    };

    (Some(setup_picture), ref_pictures, ref_slot_indices)
}

/// Find the offset of the frame header OBU in an AV1 frame.
/// This is used to set the frame_header_offset in VideoDecodeAV1PictureInfoKHR.
fn find_av1_frame_header_offset(data: &[u8]) -> u32 {
    // For low-overhead format, the frame header is at the start of the Frame OBU
    // The OBU header is 1-2 bytes, followed by the size field (1-2 bytes for leb128)
    // Then the frame header data starts
    
    if data.is_empty() {
        return 0;
    }

    // Try to find the Frame OBU (type 6) or FrameHeader OBU (type 3)
    // In low-overhead format, the first OBU is typically the Frame OBU
    
    // Parse OBU header
    if data.len() < 1 {
        return 0;
    }
    
    let first_byte = data[0];
    let obu_type = (first_byte >> 3) & 0x0F;
    let extension_flag = (first_byte >> 2) & 1 != 0;
    
    // OBU header size: 1 byte + 1 byte if extension_flag
    let header_size = 1 + usize::from(extension_flag);
    
    if data.len() <= header_size {
        return 0;
    }
    
    // For Frame OBU (type 6) or FrameHeader OBU (type 3), the frame header
    // starts right after the OBU header and size field
    if obu_type == 6 || obu_type == 3 {
        // Skip OBU header
        let mut offset = header_size;
        
        // Check if there's a size field
        let has_size_field = (first_byte >> 1) & 1 != 0;
        if has_size_field {
            // Read leb128 size
            while offset < data.len() {
                if data[offset] & 0x80 == 0 {
                    offset += 1;
                    break;
                }
                offset += 1;
            }
        }
        
        return offset as u32;
    }
    
    // For other OBUs (e.g., temporal delimiter), try to find the frame OBU
    let mut offset = 0;
    while offset < data.len().saturating_sub(1) {
        let first_byte = data[offset];
        let obu_type = (first_byte >> 3) & 0x0F;
        let extension_flag = (first_byte >> 2) & 1 != 0;
        let has_size_field = (first_byte >> 1) & 1 != 0;
        
        let header_size = 1 + usize::from(extension_flag);
        let mut size_offset = offset + header_size;
        
        if has_size_field && size_offset < data.len() {
            // Read leb128 size
            while size_offset < data.len() {
                if data[size_offset] & 0x80 == 0 {
                    size_offset += 1;
                    break;
                }
                size_offset += 1;
            }
            
            // Read the size value
            let mut size: usize = 0;
            let mut shift = 0;
            let start = offset + header_size;
            let mut pos = start;
            while pos < data.len() {
                size |= (data[pos] as usize & 0x7F) << shift;
                shift += 7;
                pos += 1;
                if data[pos - 1] & 0x80 == 0 {
                    break;
                }
            }
            
            if obu_type == 6 || obu_type == 3 {
                // Frame OBU found - frame header starts after header + size
                return (offset + header_size + (pos - start)) as u32;
            }
            
            // Skip this OBU
            offset = pos + size;
        } else {
            offset += 1;
        }
    }
    
    0
}

/// Extract the Frame OBU (type 6) or Frame Header OBU (type 3) payload from AV1 frame data.
/// The frame data may contain multiple OBUs (TemporalDelimiter, SequenceHeader, etc.).
/// This function iterates through OBUs, finds the Frame OBU (preferred) or Frame Header OBU,
/// and returns its payload.
fn extract_av1_frame_obu_payload(data: &[u8]) -> &[u8] {
    if data.is_empty() {
        return data;
    }

    // Debug: print first few bytes and OBU types found
    eprintln!(
        "[AV1] extract_av1_frame_obu_payload: data_len={}, first_bytes={:02x?}",
        data.len(),
        &data[..data.len().min(8)]
    );

    let mut offset = 0;
    let mut obu_count = 0;
    let mut frame_header_payload: Option<&[u8]> = None;

    while offset < data.len().saturating_sub(1) {
        let first_byte = data[offset];
        let obu_type = (first_byte >> 3) & 0x0F;
        let extension_flag = (first_byte >> 2) & 1 != 0;
        let has_size_field = (first_byte >> 1) & 1 != 0;

        let header_size = 1 + usize::from(extension_flag);
        let size_start = offset + header_size;

        if has_size_field && size_start < data.len() {
            // Read EB128 size
            let mut size: usize = 0;
            let mut shift = 0;
            let mut pos = size_start;
            loop {
                if pos >= data.len() {
                    break;
                }
                size |= (data[pos] as usize & 0x7F) << shift;
                shift += 7;
                pos += 1;
                if data[pos - 1] & 0x80 == 0 {
                    break;
                }
            }

            let obu_name = match obu_type {
                1 => "SEQUENCE_HEADER",
                2 => "TEMPORAL_DELIMITER",
                3 => "FRAME_HEADER",
                4 => "TILE_GROUP",
                5 => "METADATA",
                6 => "FRAME",
                7 => "REDUNDANT_FRAME_HEADER",
                8 => "TILE_LIST",
                15 => "PADDING",
                _ => "UNKNOWN",
            };
            eprintln!(
                "[AV1]   OBU[{}]: offset={}, type={}({}), has_size={}, size={}, payload_start={}",
                obu_count, offset, obu_type, obu_name, has_size_field, size, pos
            );
            obu_count += 1;

            if obu_type == 6 {
                // Frame OBU found - return payload
                let payload_start = pos;
                let payload_end = (payload_start + size).min(data.len());

                // Debug: print first bytes of payload (frame_type is first 2 bits)
                if payload_start < payload_end && payload_end - payload_start > 1 {
                    let payload_slice = &data[payload_start..(payload_start + 4).min(payload_end)];
                    let frame_type = (payload_slice[0] >> 6) & 0x03;
                    eprintln!(
                        "[AV1]   Frame OBU payload: first_bytes={:02x?}, frame_type={}",
                        payload_slice,
                        frame_type
                    );
                }

                return &data[payload_start..payload_end];
            }

            if obu_type == 3 || obu_type == 7 {
                // Frame Header OBU (type 3) or Redundant Frame Header OBU (type 7) found
                // remember as fallback
                // In IVF containers, frames may use FRAME_HEADER OBU (type 3) instead of
                // FRAME OBU (type 6). Store its payload for later use.
                let obu_name = if obu_type == 3 { "FRAME_HEADER" } else { "REDUNDANT_FRAME_HEADER" };
                let payload_start = pos;
                let payload_end = (payload_start + size).min(data.len());

                if frame_header_payload.is_none() {
                    frame_header_payload = Some(&data[payload_start..payload_end]);
                    if payload_start < payload_end && payload_end - payload_start > 1 {
                        let payload_slice = &data[payload_start..(payload_start + 4).min(payload_end)];
                        let frame_type = (payload_slice[0] >> 6) & 0x03;
                        eprintln!(
                            "[AV1]   {} OBU payload: first_bytes={:02x?}, frame_type={}",
                            obu_name, payload_slice, frame_type
                        );
                    }
                }
            }

            // Skip this OBU
            offset = pos + size;
        } else if !has_size_field {
            // No size field - this is likely a TemporalDelimiter (type 7) which has no payload
            let obu_name = match obu_type {
                2 => "TEMPORAL_DELIMITER",
                _ => "NO_SIZE",
            };
            eprintln!(
                "[AV1]   OBU[{}]: offset={}, type={}({}), has_size=false",
                obu_count, offset, obu_type, obu_name
            );
            obu_count += 1;
            offset += header_size;
        } else {
            offset += 1;
        }
    }

    // Frame OBU (type 6) not found - try FRAME_HEADER OBU (type 3) fallback
    if let Some(payload) = frame_header_payload {
        eprintln!("[AV1]   Frame OBU (type 6) not found, using FRAME_HEADER OBU (type 3) payload");
        return payload;
    }

    eprintln!("[AV1]   Frame OBU (type 6) not found, returning original data as fallback");
    // Neither Frame OBU nor Frame Header OBU found, return original data
    data
}
