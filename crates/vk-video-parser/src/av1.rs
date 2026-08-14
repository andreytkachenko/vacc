//! AV1 bitstream parser.
//!
//! Parses AV1 bitstreams with OBU support to extract sequence headers (SPS equivalent).
//! Based on cros-codecs AV1 parser implementation.
//!
//! AV1 uses a different structure than H.264/H.265 - it has OBUs (Open Bitstream Units)
//! that contain sequence headers, frame headers, and frame data.

use crate::bitreader::BitReader;
use crate::{DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser};

/// AV1 OBU types as defined in the AV1 spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObuType {
    Reserved = 0,
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
    RedundantFrameHeader = 7,
    TileList = 8,
    Reserved2 = 9,
    Reserved3 = 10,
    Reserved4 = 11,
    Reserved5 = 12,
    Reserved6 = 13,
    Reserved7 = 14,
    Padding = 15,
}

impl TryFrom<u8> for ObuType {
    type Error = ParserError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ObuType::Reserved),
            1 => Ok(ObuType::SequenceHeader),
            2 => Ok(ObuType::TemporalDelimiter),
            3 => Ok(ObuType::FrameHeader),
            4 => Ok(ObuType::TileGroup),
            5 => Ok(ObuType::Metadata),
            6 => Ok(ObuType::Frame),
            7 => Ok(ObuType::RedundantFrameHeader),
            8 => Ok(ObuType::TileList),
            9 => Ok(ObuType::Reserved2),
            10 => Ok(ObuType::Reserved3),
            11 => Ok(ObuType::Reserved4),
            12 => Ok(ObuType::Reserved5),
            13 => Ok(ObuType::Reserved6),
            14 => Ok(ObuType::Reserved7),
            15 => Ok(ObuType::Padding),
            _ => Err(ParserError::InvalidBitstream),
        }
    }
}

/// OBU header parsed from the bitstream.
#[derive(Debug, Clone)]
struct ObuHeader {
    obu_type: ObuType,
    extension_flag: bool,
    has_size_field: bool,
    temporal_id: u32,
    spatial_id: u32,
}

/// AV1 frame header parsed from Frame/FrameHeader OBU.
/// Contains fields needed for hardware decode (Vulkan/VAAPI).
#[derive(Debug, Clone, Default)]
pub struct Av1FrameHeader {
    /// Whether this is a show-existing-frame command.
    pub show_existing_frame: bool,
    /// Frame to show (when show_existing_frame is true).
    pub frame_to_show_map_idx: u8,
    /// Frame type: 0=KEY, 1=INTER, 2=INTRA_ONLY, 3=SWITCH
    pub frame_type: u8,
    /// Primary reference frame index.
    pub primary_ref_frame: u8,
    /// Reference frame indices [7] (LAST..ALTREF per AV1 spec).
    pub ref_frame_idx: [u8; 7],
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Render width (may differ from coded width).
    pub render_width: u32,
    /// Render height (may differ from coded height).
    pub render_height: u32,
    /// Tile columns log2.
    pub tile_cols_log2: u8,
    /// Tile rows log2.
    pub tile_rows_log2: u8,
    /// Order hint for reference picture management.
    pub order_hint: u32,
    /// Whether frame is error-resilient.
    pub error_resilient_mode: bool,
    /// refresh_frame_flags from frame header.
    pub refresh_frame_flags: u8,
}

/// Annex B state for tracking temporal and frame units.
#[derive(Debug, Clone)]
#[derive(Default)]
struct AnnexBState {
    temporal_unit_size: u32,
    frame_unit_size: u32,
    temporal_unit_consumed: u32,
    frame_unit_consumed: u32,
}


/// Stream format detected for the bitstream.
#[derive(Debug, Clone)]
enum StreamFormat {
    LowOverhead,
    AnnexB(AnnexBState),
}

/// AV1 parser state.
pub struct Av1Parser {
    /// Sequence header (SPS equivalent).
    active_sps: Option<vk_video_core::picture::Av1Sps>,
    /// Detected format.
    detected_format: DetectedVideoFormat,
    /// Frame counter.
    frame_count: u32,
    /// Stream format (low-overhead or Annex B).
    stream_format: StreamFormat,
    /// Whether we should probe for Annex B format.
    should_probe_for_annexb: bool,
    /// Reference frame sizes for frame size inheritance (per AV1 spec 7.20).
    ref_frame_sizes: [(u32, u32); 8],
    /// Reference frame order hints for short signaling derivation.
    ref_frame_order_hints: [u32; 8],
}

impl Default for Av1Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1Parser {
    pub fn new() -> Self {
        Self {
            active_sps: None,
            detected_format: DetectedVideoFormat::new(vk_video_core::codec::VideoCodec::DecodeAv1),
            frame_count: 0,
            stream_format: StreamFormat::LowOverhead,
            should_probe_for_annexb: true,
            ref_frame_sizes: [(0, 0); 8],
            ref_frame_order_hints: [0; 8],
        }
    }

    /// Probe the input data for the Annex B format.
    fn annexb_probe(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        let mut r = BitReader::new(data, false);

        // Try reading the first TU and frame unit size
        let temporal_unit_size = match r.read_leb128() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if temporal_unit_size == 0 {
            return false;
        }

        let frame_unit_size = match r.read_leb128() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if frame_unit_size == 0 || frame_unit_size > temporal_unit_size {
            return false;
        }

        let obu_length = match r.read_leb128() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if obu_length == 0 || obu_length > frame_unit_size {
            return false;
        }

        // The first OBU in the first frame_unit of each temporal_unit must
        // be a temporal delimiter OBU
        let header = match Self::parse_obu_header(&mut r) {
            Ok(h) => h,
            Err(_) => return false,
        };
        if header.obu_type != ObuType::TemporalDelimiter {
            return false;
        }

        // Try identifying a sequence and a frame.
        if obu_length > 0 {
            let _ = r.skip_bits((obu_length as u8).min(31) * 8);
        }

        let mut num_bytes_read = 0;
        let mut seen_sequence = false;
        let mut seen_frame = false;

        loop {
            let obu_length = match r.read_leb128() {
                Ok(v) => v,
                Err(_) => return false,
            };

            num_bytes_read += obu_length;

            if !seen_sequence {
                let mut obu_reader = BitReader::new(&data[r.position() as usize..], false);
                if let Ok(header) = Self::parse_obu_header(&mut obu_reader) {
                    seen_sequence = header.obu_type == ObuType::SequenceHeader;
                }
            }

            if !seen_frame {
                let mut obu_reader = BitReader::new(&data[r.position() as usize..], false);
                if let Ok(header) = Self::parse_obu_header(&mut obu_reader) {
                    seen_frame = matches!(header.obu_type, ObuType::Frame | ObuType::FrameHeader);
                }
            }

            if seen_sequence && seen_frame {
                return true;
            }

            if num_bytes_read >= frame_unit_size {
                return false;
            }

            if obu_length > 0 {
                let _ = r.skip_bits((obu_length as u8).min(31) * 8);
            }
        }
    }

    /// Parse an OBU header from the bitstream.
    fn parse_obu_header(r: &mut BitReader) -> Result<ObuHeader, ParserError> {
        let _obu_forbidden_bit = r.read_bit()?;

        let obu_type = ObuType::try_from(r.read_bits(4)? as u8)?;
        let extension_flag = r.read_bit()?;
        let has_size_field = r.read_bit()?;
        let _obu_reserved_1bit = r.read_bit()?;

        let mut header = ObuHeader {
            obu_type,
            extension_flag,
            has_size_field,
            temporal_id: 0,
            spatial_id: 0,
        };

        if extension_flag {
            header.temporal_id = r.read_bits(3)?;
            header.spatial_id = r.read_bits(2)?;
            let _ = r.read_bits(3)?;
        }

        Ok(header)
    }

    /// Read one OBU from the bitstream, handling both Annex B and low-overhead formats.
    ///
    /// Returns (header, obu_data_start, obu_size) on success.
    fn read_obu(&mut self, data: &[u8]) -> Result<Option<(ObuHeader, usize, usize)>, ParserError> {
        if data.is_empty() {
            return Ok(None);
        }

        let mut reader = BitReader::new(data, false);

        if self.should_probe_for_annexb {
            self.stream_format = if Self::annexb_probe(data) {
                StreamFormat::AnnexB(AnnexBState::default())
            } else {
                StreamFormat::LowOverhead
            };
            self.should_probe_for_annexb = false;
        }

        let obu_length: Option<usize> = match &mut self.stream_format {
            StreamFormat::AnnexB(annexb_state) => {
                Self::current_annexb_obu_length(&mut reader, annexb_state)?
            }
            _ => None,
        };

        let header = Self::parse_obu_header(&mut reader)?;

        if matches!(self.stream_format, StreamFormat::LowOverhead)
            && !header.has_size_field {
                return Err(ParserError::InvalidBitstream);
            }

        let obu_size: usize = if header.has_size_field {
            reader.read_leb128()? as usize
        } else {
            obu_length
                .ok_or(ParserError::InvalidBitstream)?
                .checked_sub(1)
                .and_then(|v| v.checked_sub(usize::from(header.extension_flag)))
                .ok_or(ParserError::InvalidBitstream)?
        };

        // Update Annex B state if applicable
        if let StreamFormat::AnnexB(ref mut annexb_state) = self.stream_format {
            annexb_state.temporal_unit_consumed += (reader.position() / 8) as u32;
            annexb_state.frame_unit_consumed += (reader.position() / 8) as u32;
            annexb_state.temporal_unit_consumed += obu_size as u32;
            annexb_state.frame_unit_consumed += obu_size as u32;
        }

        let start_offset = (reader.position() / 8) as usize;

        Ok(Some((header, start_offset, obu_size)))
    }

    /// Get the length of the current OBU in Annex B format.
    fn current_annexb_obu_length(
        reader: &mut BitReader,
        annexb_state: &mut AnnexBState,
    ) -> Result<Option<usize>, ParserError> {
        if !reader.has_more_data() {
            return Ok(None);
        }

        if annexb_state.temporal_unit_consumed == annexb_state.temporal_unit_size {
            annexb_state.temporal_unit_size = 0;
        }

        if annexb_state.temporal_unit_size == 0 {
            annexb_state.temporal_unit_size = reader.read_leb128()?;
            if annexb_state.temporal_unit_size == 0 {
                return Ok(None);
            }
        }

        let start_pos = (reader.position() / 8) as u32;

        if annexb_state.frame_unit_consumed == annexb_state.frame_unit_size {
            annexb_state.frame_unit_size = 0;
        }

        if annexb_state.frame_unit_size == 0 {
            annexb_state.frame_unit_size = reader.read_leb128()?;
            if annexb_state.frame_unit_size == 0 {
                return Ok(None);
            }
            annexb_state.temporal_unit_consumed += (reader.position() / 8) as u32 - start_pos;
        }

        let start_pos = (reader.position() / 8) as u32;
        let obu_length = reader.read_leb128()?;
        let consumed = (reader.position() / 8) as u32 - start_pos;

        annexb_state.temporal_unit_consumed += consumed;
        annexb_state.frame_unit_consumed += consumed;

        Ok(Some(obu_length as usize))
    }

    /// Parse the Sequence Header OBU and populate Av1Sps.
    fn parse_sequence_header_obu(
        &mut self,
        obu_data: &[u8],
    ) -> ParserResult<vk_video_core::picture::Av1Sps> {
        let mut sps = vk_video_core::picture::Av1Sps::new();

        if obu_data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        let mut r = BitReader::new(obu_data, false);

        // seq_profile (3 bits)
        sps.profile = r.read_bits(3)? as u8;

        // still_picture (1 bit)
        sps.still_picture = r.read_bit()?;

        // reduced_still_picture_header (1 bit)
        sps.reduced_still_picture_header = r.read_bit()?;

        if sps.reduced_still_picture_header {
            // For reduced still picture header, many fields are implicit
            // timing_info_present_flag = false (implicit)
            // decoder_model_info_present_flag = false (implicit)
            // initial_display_delay_present_flag = false (implicit)
            // operating_points_cnt_minus_1 = 0 (implicit)

            // seq_level_idx (5 bits)
            sps.level = r.read_bits(5)? as u8;
        } else {
            // timing_info_present_flag (1 bit)
            sps.timing_info_present_flag = r.read_bit()?;
            let decoder_model_info_present_flag = if sps.timing_info_present_flag {
                // Parse timing info
                Self::parse_timing_info(&mut sps, &mut r)?;

                // decoder_model_info_present_flag (1 bit)
                r.read_bit()?
            } else {
                false
            };
            sps.decoder_model_info_present_flag = decoder_model_info_present_flag;
            if decoder_model_info_present_flag {
                Self::parse_decoder_model_info(&mut sps, &mut r)?;
            }

            // initial_display_delay_present_flag (1 bit)
            sps.initial_display_delay_present_flag = r.read_bit()?;

            // operating_points_cnt_minus_1 (5 bits)
            let operating_points_cnt_minus_1 = r.read_bits(5)? as usize;

            // Parse each operating point
            for i in 0..=operating_points_cnt_minus_1 {
                // operating_point_idc (12 bits)
                let _operating_point_idc = r.read_bits(12)?;

                // seq_level_idx (5 bits)
                let seq_level_idx = r.read_bits(5)?;
                if i == 0 {
                    sps.level = seq_level_idx as u8;
                }

                // seq_tier (1 bit) if seq_level_idx > 7
                if seq_level_idx > 7 {
                    let _seq_tier = r.read_bit()?;
                }

                // decoder_model_present_for_this_op if decoder_model_info_present_flag
                if decoder_model_info_present_flag {
                    let decoder_model_present_for_this_op = r.read_bit()?;
                    if decoder_model_present_for_this_op {
                        // Parse operating parameters info
                        // Use buffer_delay_length_minus_1 from decoder_model_info per AV1 spec 5.3.2
                        let n = sps.buffer_delay_length_minus_1 + 1;
                        let _decoder_buffer_delay = r.read_bits(n)?;
                        let _encoder_buffer_delay = r.read_bits(n)?;
                        let _low_delay_mode_flag = r.read_bit()?;
                    }
                }

                // initial_display_delay_present_for_this_op if initial_display_delay_present_flag
                if sps.initial_display_delay_present_flag {
                    let initial_display_delay_present_for_this_op = r.read_bit()?;
                    if initial_display_delay_present_for_this_op {
                        let _initial_display_delay_minus_1 = r.read_bits(4)?;
                    }
                }
            }
        }

        // frame_width_bits_minus_1 (4 bits)
        let frame_width_bits_minus_1 = r.read_bits(4)? as u8;
        sps.frame_width_bits = frame_width_bits_minus_1 + 1;

        // frame_height_bits_minus_1 (4 bits)
        let frame_height_bits_minus_1 = r.read_bits(4)? as u8;
        sps.frame_height_bits = frame_height_bits_minus_1 + 1;

        // max_frame_width_minus_1 (frame_width_bits_minus_1 + 1 bits)
        sps.max_frame_width_minus_1 = r.read_bits(frame_width_bits_minus_1 + 1)? as u16;

        // max_frame_height_minus_1 (frame_height_bits_minus_1 + 1 bits)
        sps.max_frame_height_minus_1 = r.read_bits(frame_height_bits_minus_1 + 1)? as u16;

        // frame_id_numbers_present_flag (1 bit) - implicit in reduced_still_picture_header
        let frame_id_numbers_present_flag = if sps.reduced_still_picture_header {
            false
        } else {
            r.read_bit()?
        };

        if frame_id_numbers_present_flag {
            // delta_frame_id_length_minus2 (4 bits)
            sps.delta_frame_id_length_minus2 = r.read_bits(4)? as u8;

            // additional_frame_id_length_minus1 (3 bits)
            sps.additional_frame_id_length_minus1 = r.read_bits(3)? as u8;

            let frame_id_length = sps.additional_frame_id_length_minus1 as u32
                + sps.delta_frame_id_length_minus2 as u32
                + 3;
            if frame_id_length > 16 {
                return Err(ParserError::InvalidBitstream);
            }
        }

        // use_128x128_superblock (1 bit)
        sps.use_128x128_superblock = r.read_bit()?;

        // enable_filter_intra (1 bit)
        sps.enable_filter_intra = r.read_bit()?;

        // enable_intra_edge_filter (1 bit)
        sps.enable_intra_edge_filter = r.read_bit()?;

        if sps.reduced_still_picture_header {
            // For reduced still picture header, these are implicit
            sps.enable_interintra_compound = false;
            sps.enable_masked_compound = false;
            sps.enable_warped_motion = false;
            sps.enable_dual_filter = false;
            sps.enable_order_hint = false;
            sps.enable_jnt_motion = false;
            sps.enable_second_ref_frame = false;
            sps.order_hint_bits_minus1 = 0;
        } else {
            // enable_interintra_compound (1 bit)
            sps.enable_interintra_compound = r.read_bit()?;

            // enable_masked_compound (1 bit)
            sps.enable_masked_compound = r.read_bit()?;

            // enable_warped_motion (1 bit)
            sps.enable_warped_motion = r.read_bit()?;

            // enable_dual_filter (1 bit)
            sps.enable_dual_filter = r.read_bit()?;

            // enable_order_hint (1 bit)
            sps.enable_order_hint = r.read_bit()?;

            if sps.enable_order_hint {
                // enable_jnt_comp (1 bit)
                sps.enable_jnt_motion = r.read_bit()?;

                // enable_ref_frame_mvs (1 bit)
                sps.enable_second_ref_frame = r.read_bit()?;
            } else {
                sps.enable_jnt_motion = false;
                sps.enable_second_ref_frame = false;
            }

            // seq_choose_screen_content_tools (1 bit)
            let seq_choose_screen_content_tools = r.read_bit()?;
            if !seq_choose_screen_content_tools {
                // seq_force_screen_content_tools (1 bit)
                let seq_force_screen_content_tools = r.read_bit()?;
                if seq_force_screen_content_tools {
                    // seq_choose_integer_mv (1 bit)
                    let seq_choose_integer_mv = r.read_bit()?;
                    if !seq_choose_integer_mv {
                        // seq_force_integer_mv (1 bit)
                        let _seq_force_integer_mv = r.read_bit()?;
                    }
                }
            }

            if sps.enable_order_hint {
                // order_hint_bits_minus1 (3 bits)
                sps.order_hint_bits_minus1 = r.read_bits(3)? as u8;
            } else {
                sps.order_hint_bits_minus1 = 0;
            }
        }

        // enable_superres (1 bit)
        sps.enable_superres = r.read_bit()?;

        // enable_cdef (1 bit)
        sps.enable_cdef = r.read_bit()?;

        // enable_restoration (1 bit)
        sps.enable_restoration = r.read_bit()?;

        // Parse color config
        Self::parse_color_config(&mut sps, &mut r)?;

        // film_grain_params_present (1 bit)
        sps.film_grain_params_present = r.read_bit()?;

        // Parse trailing bits
        Self::parse_trailing_bits(&mut r, obu_data.len())?;

        // Update detected format
        self.update_format_from_sps(&sps);

        self.active_sps = Some(sps.clone());
        Ok(sps)
    }

    /// Parse timing_info syntax element.
    fn parse_timing_info(
        sps: &mut vk_video_core::picture::Av1Sps,
        r: &mut BitReader,
    ) -> ParserResult<()> {
        // num_units_in_display_tick (32 bits)
        sps.num_units_in_display_tick = r.read_bits(32)?;

        // time_scale (32 bits)
        sps.time_scale = r.read_bits(32)?;

        // equal_picture_interval (1 bit)
        sps.equal_picture_interval = r.read_bit()?;
        if sps.equal_picture_interval {
            // num_ticks_per_picture_minus_1 (uvlc)
            let _num_ticks_per_picture_minus_1 = r.read_uvlc()?;
        }

        Ok(())
    }

    /// Parse decoder_model_info syntax element.
    fn parse_decoder_model_info(
        sps: &mut vk_video_core::picture::Av1Sps,
        r: &mut BitReader,
    ) -> ParserResult<()> {
        // buffer_delay_length_minus_1 (5 bits)
        sps.buffer_delay_length_minus_1 = r.read_bits(5)? as u8;

        // num_units_in_decoding_tick (32 bits)
        let _num_units_in_decoding_tick = r.read_bits(32)?;

        // buffer_removal_time_length_minus_1 (5 bits)
        let _buffer_removal_time_length_minus_1 = r.read_bits(5)?;

        // frame_presentation_time_length_minus_1 (5 bits)
        let _frame_presentation_time_length_minus_1 = r.read_bits(5)?;

        Ok(())
    }

    /// Parse color_config syntax element.
    fn parse_color_config(
        sps: &mut vk_video_core::picture::Av1Sps,
        r: &mut BitReader,
    ) -> ParserResult<()> {
        let seq_profile = sps.profile as u32;

        // high_bitdepth (1 bit)
        sps.high_bitdepth = r.read_bit()?;

        // twelve_bit (1 bit) - only for profile 2 with high_bitdepth
        sps.twelve_bit = if seq_profile == 2 && sps.high_bitdepth {
            r.read_bit()?
        } else {
            false
        };

        // mono_chrome (1 bit) - not present for profile 1
        sps.mono_chrome = if seq_profile == 1 {
            false
        } else {
            r.read_bit()?
        };

        // color_description_present_flag (1 bit)
        sps.color_description_present = r.read_bit()?;
        let (color_primaries, transfer_characteristics, matrix_coefficients) =
            if sps.color_description_present {
                // color_primaries (8 bits)
                let color_primaries = r.read_bits(8)? as u8;

                // transfer_characteristics (8 bits)
                let transfer_characteristics = r.read_bits(8)? as u8;

                // matrix_coefficients (8 bits)
                let matrix_coefficients = r.read_bits(8)? as u8;

                (
                    color_primaries,
                    transfer_characteristics,
                    matrix_coefficients,
                )
            } else {
                (2, 2, 2) // Default: BT.709
            };
        sps.color_primaries = color_primaries;
        sps.transfer_characteristics = transfer_characteristics;
        sps.matrix_coefficients = matrix_coefficients;

        if sps.mono_chrome {
            // color_range (1 bit)
            sps.color_range = r.read_bit()?;
        } else {
            // Check for sRGB color space per AV1 spec:
            // is_srgb = (color_primaries == 1) && (transfer_characteristics == 13) && (matrix_coefficients == 1)
            let is_srgb = (color_primaries == 1)
                && (transfer_characteristics == 13)
                && (matrix_coefficients == 1);
            if !is_srgb {
                // color_range (1 bit)
                sps.color_range = r.read_bit()?;

                // subsampling_x, subsampling_y - profile 2 only, and only when twelve_bit is true
                // Per AV1 spec 5.3.3: subsampling_x/y present when seq_profile==2 AND twelve_bit==1
                if seq_profile == 2 && sps.twelve_bit {
                    // subsampling_x (1 bit)
                    sps.subsampling_x = r.read_bit()? as u8;
                    if sps.subsampling_x != 0 {
                        // subsampling_y (1 bit)
                        sps.subsampling_y = r.read_bit()? as u8;

                        // chroma_sample_position (2 bits) - only if subsampled in both dimensions
                        if sps.subsampling_y != 0 {
                            sps.chroma_sample_position = r.read_bits(2)? as u8;
                        }
                    }
                }
            }
        }

        // separate_uv_delta_q (1 bit)
        let _separate_uv_delta_q = r.read_bit()?;

        Ok(())
    }

    /// Parse trailing bits (trailing_one_bit + trailing_zero_bits).
    fn parse_trailing_bits(r: &mut BitReader, data_len: usize) -> ParserResult<()> {
        let total_bits = (data_len * 8) as u64;
        let current_pos = r.position();
        let remaining_bits = total_bits - current_pos;

        if remaining_bits == 0 {
            return Ok(());
        }

        // trailing_one_bit (1 bit)
        let trailing_one_bit = r.read_bit()?;
        if !trailing_one_bit {
            return Err(ParserError::InvalidBitstream);
        }

        // trailing_zero_bits (remaining bits should be zero)
        let remaining = remaining_bits - 1;
        if remaining > 0 {
            let remaining_u8 = (remaining.min(31)) as u8;
            let trailing_zeros = r.read_bits(remaining_u8)?;
            if trailing_zeros != 0 {
                return Err(ParserError::InvalidBitstream);
            }
        }

        Ok(())
    }

    /// Parse AV1 uncompressed frame header from Frame/FrameHeader OBU data.
    /// Extracts fields needed for hardware decode (Vulkan Video / VAAPI).
    pub fn parse_frame_header(
        &self,
        obu_data: &[u8],
        sps: &vk_video_core::picture::Av1Sps,
    ) -> ParserResult<Av1FrameHeader> {
        if obu_data.is_empty() || self.active_sps.is_none() {
            return Err(ParserError::InvalidBitstream);
        }

        let mut r = BitReader::new(obu_data, false);
        let mut fh = Av1FrameHeader::default();

        // For reduced still picture header, frame header is empty
        if sps.reduced_still_picture_header {
            fh.frame_type = 0; // KEY_FRAME
            fh.primary_ref_frame = 0;
            return Ok(fh);
        }

        // frame_context_idx (2 bits)
        let _frame_context_idx = r.read_bits(2)?;

        // show_frame (1 bit) - indicates if this frame should be displayed
        let _show_frame = r.read_bit()?;

        // show_existing_frame (1 bit)
        fh.show_existing_frame = r.read_bit()?;
        if fh.show_existing_frame {
            fh.frame_to_show_map_idx = r.read_bits(3)? as u8;
            return Ok(fh);
        }

        // frame_type (2 bits): 0=KEY, 1=INTER, 2=INTRA_ONLY, 3=SWITCH
        fh.frame_type = r.read_bits(2)? as u8;

        // error_resilient_mode (1 bit)
        fh.error_resilient_mode = r.read_bit()?;

        // frame_id_number_present_flag (1 bit)
        let frame_id_number_present_flag = r.read_bit()?;
        if frame_id_number_present_flag {
            // frame_id (32 bits)
            let _frame_id = r.read_bits(32)?;
        } else {
            // short_frame_id
            let frame_id_length = sps.delta_frame_id_length_minus2 + 2 + 1;
            let _short_frame_id = r.read_bits(frame_id_length)?;
        }

        // export_tile_stream_flag (1 bit)
        let _export_tile_stream_flag = r.read_bit()?;

        // primary_ref_frame (3 bits)
        fh.primary_ref_frame = r.read_bits(3)? as u8;

        // frame_size_override_flag / render_and_frame_size_different
        // Per AV1 spec:
        // - error_resilient_mode=1: bit is render_and_frame_size_different, frame size inherited from ref
        // - error_resilient_mode=0: bit is frame_size_override_flag, if 0 frame size inherited from ref
        let frame_size_override = if fh.error_resilient_mode {
            false // In error resilient mode, frame size is always inherited
        } else {
            r.read_bit()? // frame_size_override_flag
        };

        if frame_size_override {
            fh.frame_width = r.read_bits(sps.frame_width_bits)? + 1;
            fh.frame_height = r.read_bits(sps.frame_height_bits)? + 1;
        } else {
            // Frame size inherited from the primary reference frame (AV1 spec 7.20).
            // In error_resilient_mode, use the first available reference frame.
            let ref_idx = if fh.error_resilient_mode {
                // In error resilient mode, inherit from first valid ref
                self.ref_frame_sizes
                    .iter()
                    .position(|&(w, h)| w > 0 && h > 0)
                    .unwrap_or(0) as u8
            } else {
                fh.primary_ref_frame
            };
            match self.ref_frame_sizes.get(ref_idx as usize) {
                Some(&(w, h)) if w > 0 && h > 0 => {
                    fh.frame_width = w;
                    fh.frame_height = h;
                }
                _ => {
                    // Fallback to max dimensions if reference frame size unavailable
                    fh.frame_width = sps.max_frame_width_minus_1 as u32 + 1;
                    fh.frame_height = sps.max_frame_height_minus_1 as u32 + 1;
                }
            }
        }

        // render_and_frame_size_different
        let render_size_different = r.read_bit()?;

        if render_size_different {
            fh.render_width = r.read_bits(16)? + 1;
            fh.render_height = r.read_bits(16)? + 1;
        } else {
            fh.render_width = fh.frame_width;
            fh.render_height = fh.frame_height;
        }

        // allow_screen_content_tools (1 bit) - only if not error resilient
        let allow_screen_content_tools = if !fh.error_resilient_mode {
            r.read_bit()?
        } else {
            false
        };

        // frame_refs_short_signaling (1 bit) - only if not error resilient and allow_screen_content_tools
        let frame_refs_short_signaling = if !fh.error_resilient_mode && allow_screen_content_tools {
            r.read_bit()?
        } else {
            false
        };

        if frame_refs_short_signaling {
            // Short signaling (AV1 spec 7.21): read last_frame_idx and gold_frame_idx
            let last_frame_idx = r.read_bits(2)? as u8;
            let gold_frame_idx = r.read_bits(2)? as u8;

            // Derive ref_frame array from order hints per AV1 spec 7.21.
            // LAST_FRAME, LAST2_FRAME, LAST3_FRAME are the 3 most recent frames
            // by order hint < current order_hint.
            // GOLDEN_FRAME = gold_frame_idx
            // BWDREF_FRAME = most recent frame by order hint > current order_hint
            // ALTREF2_FRAME, ALTREF_FRAME = 2 most recent frames by order hint > current
            let cur_order_hint = fh.order_hint;
            let candidates: Vec<(u32, u8)> = (0..8)
                .map(|i| (self.ref_frame_order_hints[i], i as u8))
                .filter(|(oh, _)| *oh != cur_order_hint)
                .collect();

            // LAST/LAST2/LAST3: 3 most recent with order_hint < current
            let mut past = candidates
                .iter()
                .filter(|(oh, _)| *oh < cur_order_hint)
                .cloned()
                .collect::<Vec<_>>();
            past.sort_by_key(|(oh, _)| *oh);
            past.reverse();

            // ALTREF/ALTREF2/BWDREF: most recent with order_hint > current
            let mut future = candidates
                .iter()
                .filter(|(oh, _)| *oh > cur_order_hint)
                .cloned()
                .collect::<Vec<_>>();
            future.sort_by_key(|(oh, _)| *oh);
            future.reverse();

            fh.ref_frame_idx[0] = past.first().map(|(_, i)| *i).unwrap_or(last_frame_idx); // LAST
            fh.ref_frame_idx[1] = past.get(1).map(|(_, i)| *i).unwrap_or(last_frame_idx); // LAST2
            fh.ref_frame_idx[2] = past.get(2).map(|(_, i)| *i).unwrap_or(last_frame_idx); // LAST3
            fh.ref_frame_idx[3] = gold_frame_idx; // GOLDEN
            fh.ref_frame_idx[4] = future.first().map(|(_, i)| *i).unwrap_or(gold_frame_idx); // BWDREF
            fh.ref_frame_idx[5] = future.get(1).map(|(_, i)| *i).unwrap_or(gold_frame_idx); // ALTREF2
            fh.ref_frame_idx[6] = future.first().map(|(_, i)| *i).unwrap_or(gold_frame_idx);
        // ALTREF
        } else {
            // Normal signaling: reference_frame (3 bits each for 7 refs)
            for i in 0..7usize {
                let ref_frame = r.read_bits(3)? as u8;
                fh.ref_frame_idx[i] = ref_frame;
            }
        }

        // order_hint (if enable_order_hint)
        if sps.enable_order_hint {
            let order_hint_bits = sps.order_hint_bits_minus1 + 1;
            fh.order_hint = r.read_bits(order_hint_bits)?;
        }

        // refresh_frame_flags (8 bits) - which ref slots this frame refreshes
        fh.refresh_frame_flags = r.read_bits(8)? as u8;

        // inter_frame (if not key frame)
        if fh.frame_type != 0 {
            // allow_high_precision_mv (1 bit)
            let _allow_high_precision_mv = r.read_bit()?;

            // ref_frame_sign_bias (1 bit each for refs 1-7)
            for _ in 1..8usize {
                let _sign_bias = r.read_bit()?;
            }

            // allow_warped_motion (1 bit)
            let _allow_warped_motion = if !fh.error_resilient_mode {
                r.read_bit()?
            } else {
                false
            };

            // compound_type (2 bits)
            let _compound_type = r.read_bits(2)?;
        }

        // tile_info
        let _uniform_tile_spacing_flag = r.read_bit()?;
        fh.tile_cols_log2 = r.read_bits(3)? as u8;
        fh.tile_rows_log2 = r.read_bits(3)? as u8;

        Ok(fh)
    }

    /// Update reference frame tracking state after parsing a frame header (AV1 spec 7.20).
    #[allow(dead_code)]
    fn update_ref_frames(&mut self, fh: &Av1FrameHeader) {
        // Update sizes for refreshed frame slots
        for i in 0..8usize {
            if (fh.refresh_frame_flags & (1 << i)) != 0 {
                self.ref_frame_sizes[i] = (fh.frame_width, fh.frame_height);
                self.ref_frame_order_hints[i] = fh.order_hint;
            }
        }
    }

    /// Update detected format from sequence header.
    fn update_format_from_sps(&mut self, sps: &vk_video_core::picture::Av1Sps) {
        let coded_width = sps.max_frame_width_minus_1 as u32 + 1;
        let coded_height = sps.max_frame_height_minus_1 as u32 + 1;

        self.detected_format.coded_width = coded_width;
        self.detected_format.coded_height = coded_height;

        // AV1 bit depth is determined from profile and color config
        // For now, default to 8-bit (actual bit depth is in frame header)
        self.detected_format.luma_bit_depth = vk_video_core::format::ComponentBitDepth::Bit8;
        self.detected_format.chroma_bit_depth = vk_video_core::format::ComponentBitDepth::Bit8;

        // AV1 chroma subsampling - typically 4:2:0 for profile 0
        self.detected_format.chroma_subsampling = vk_video_core::format::ChromaSubsampling::_420;

        self.detected_format.codec_profile = sps.profile as u32;
        self.detected_format.film_grain_used = sps.film_grain_params_present;
        self.detected_format.progressive_sequence = true;
    }

    /// Check if the data looks like an AV1 bitstream.
    pub fn is_av1(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        // AV1 start code is 0x9E or 0x80 for the first frame
        data[data.len() - 1] == 0x9E || (data.len() >= 2 && data[data.len() - 2] == 0x80)
    }
}

impl VideoParser for Av1Parser {
    fn init(&mut self, format: &DetectedVideoFormat) -> ParserResult<()> {
        if format.codec != vk_video_core::codec::VideoCodec::DecodeAv1 {
            return Err(ParserError::InvalidBitstream);
        }
        self.detected_format = format.clone();
        Ok(())
    }

    fn parse(&mut self, packet: &crate::bitstream::BitstreamPacket) -> ParserResult<ParseResult> {
        if packet.is_eos() {
            return Ok(ParseResult::EndOfStream);
        }

        let data = &packet.payload;

        // Iterate through OBUs in the packet
        let mut offset = 0;
        while offset < data.len() {
            match self.read_obu(&data[offset..]) {
                Ok(Some((header, obu_start, obu_size))) => {
                    let obu_data_offset = obu_start;
                    let obu_data = if obu_data_offset + obu_size <= data.len() - offset {
                        &data[offset + obu_data_offset..offset + obu_data_offset + obu_size]
                    } else {
                        break;
                    };

                    match header.obu_type {
                        ObuType::SequenceHeader => {
                            if self.active_sps.is_none() {
                                let seq_header = self.parse_sequence_header_obu(obu_data)?;
                                return Ok(ParseResult::ParameterSet {
                                    sps: Some(
                                        vk_video_core::picture::BoxedPictureParametersSet::new(
                                            seq_header,
                                        ),
                                    ),
                                    pps: None,
                                    vps: None,
                                });
                            }
                        }
                        ObuType::Frame | ObuType::FrameHeader
                            // If we have a sequence header, treat remaining data as frame data
                            if self.active_sps.is_some() => {
                                self.frame_count += 1;
                                offset += obu_data_offset + obu_size;
                                return Ok(ParseResult::Slice {
                                    slice_data_offset: offset,
                                    slice_data_len: data.len() - offset,
                                    num_slices: 1,
                                    slice_header: None,
                                });
                            }
                        _ => {
                            // Skip other OBU types
                        }
                    }

                    offset += obu_data_offset + obu_size;
                }
                Ok(None) => {
                    break;
                }
                Err(_) => {
                    // If we can't parse an OBU, try to continue from next byte
                    offset += 1;
                }
            }
        }

        // If we have a sequence header and no OBUs were parsed, treat as frame data
        if self.active_sps.is_some() && !data.is_empty() {
            self.frame_count += 1;
            return Ok(ParseResult::Slice {
                slice_data_offset: 0,
                slice_data_len: data.len(),
                num_slices: 1,
                slice_header: None,
            });
        }

        Ok(ParseResult::Nothing)
    }

    fn reset(&mut self) {
        self.active_sps = None;
        self.frame_count = 0;
        self.stream_format = StreamFormat::LowOverhead;
        self.should_probe_for_annexb = true;
        self.ref_frame_sizes = [(0, 0); 8];
        self.ref_frame_order_hints = [0; 8];
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}
