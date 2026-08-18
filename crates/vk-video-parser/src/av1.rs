//! AV1 bitstream parser.
//!
//! Parses AV1 bitstreams with OBU support to extract sequence headers (SPS equivalent).
//! Based on cros-codecs AV1 parser implementation.
//!
//! AV1 uses a different structure than H.264/H.265 - it has OBUs (Open Bitstream Units)
//! that contain sequence headers, frame headers, and frame data.

use crate::bitreader::BitReader;
use crate::{DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser};

/// Returns true when `VACC_DEBUG=1` is set. Gates the verbose per-frame
/// frame-header debug dumps. Off by default.
fn vacc_debug() -> bool {
    std::env::var("VACC_DEBUG").ok().unwrap_or_default() == "1"
}

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

/// Fixed-size tile dimension array (64 entries). Wraps `[u16; 64]` because
/// that array type does not implement `Default` on stable Rust; the newtype
/// provides a manual `Default` plus `Deref`/`DerefMut` so indexing/slicing
/// works exactly like the raw array.
#[derive(Debug, Clone, Copy)]
pub struct TileArray(pub [u16; 64]);
impl Default for TileArray {
    fn default() -> Self {
        TileArray([0; 64])
    }
}
impl std::ops::Deref for TileArray {
    type Target = [u16; 64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for TileArray {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
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
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
    /// uniform_tile_spacing_flag.
    pub uniform_tile_spacing_flag: bool,
    /// tile_size_bytes_minus_1 (only meaningful when tile_count > 1).
    pub tile_size_bytes_minus_1: u32,
    /// context_update_tile_id (only meaningful when tile_count > 1).
    pub context_update_tile_id: u32,
    /// Number of tiles (tile_cols * tile_rows).
    pub tile_count: u32,
    /// Per-tile width in superblocks minus 1 (up to 64 tiles).
    pub tile_width_in_sbs_minus_1: TileArray,
    /// Per-tile height in superblocks minus 1 (up to 64 tiles).
    pub tile_height_in_sbs_minus_1: TileArray,
    /// Per-tile MI column start (up to 64 tiles).
    pub tile_mi_col_starts: TileArray,
    /// Per-tile MI row start (up to 64 tiles).
    pub tile_mi_row_starts: TileArray,
    /// diff_uv_delta (quantization).
    pub diff_uv_delta: bool,
    /// Order hint for reference picture management.
    pub order_hint: u32,
    /// Whether frame is error-resilient.
    pub error_resilient_mode: bool,
    /// refresh_frame_flags from frame header.
    pub refresh_frame_flags: u8,
    /// show_frame (1 bit).
    pub show_frame: bool,
    /// frame_size_override_flag.
    pub frame_size_override_flag: bool,
    /// render_and_frame_size_different.
    pub render_and_frame_size_different: bool,
    /// use_superres (derived from superres_scale_present).
    pub use_superres: bool,
    /// allow_screen_content_tools.
    pub allow_screen_content_tools: bool,
    /// force_integer_mv (from SPS seq_force_integer_mv).
    pub force_integer_mv: bool,
    /// frame_refs_short_signaling.
    pub frame_refs_short_signaling: bool,
    /// allow_intrabc.
    pub allow_intrabc: bool,
    /// allow_high_precision_mv.
    pub allow_high_precision_mv: bool,
    /// is_filter_switchable.
    pub is_filter_switchable: bool,
    /// interpolation_filter: raw bitstream value (0=EIGHTTAP, 1=EIGHTTAP_SMOOTH, 2=EIGHTTAP_SHARP)
    /// or 4=SWITCHABLE when is_filter_switchable. Stored as-is for Vulkan struct.
    pub interpolation_filter: u8,
    /// is_motion_mode_switchable.
    pub is_motion_mode_switchable: bool,
    /// use_ref_frame_mvs.
    pub use_ref_frame_mvs: bool,
    /// disable_cdf_update.
    pub disable_cdf_update: bool,
    /// disable_frame_end_update_cdf.
    pub disable_frame_end_update_cdf: bool,
    /// allow_warped_motion.
    pub allow_warped_motion: bool,
    /// reduced_tx_set.
    pub reduced_tx_set: bool,
    /// reference_select.
    pub reference_select: bool,
    /// skip_mode_present.
    pub skip_mode_present: bool,
    /// SkipModeFrame (Vulkan): reference name indices (1-based: LAST=1..ALTREF=7)
    /// of the nearest forward/backward references, per C++
    /// VulkanAV1Decoder.cpp IsSkipModeAllowed (NOT the bitstream skip_mode_frame
    /// bits). [0,0] when skip mode is not present.
    pub skip_mode_frame: [u8; 2],
    /// tx_mode: 0=ONLY4X4, 1=LARGEST, 2=SELECT.
    pub tx_mode: u8,
    /// delta_q_present.
    pub delta_q_present: bool,
    /// delta_lf_present.
    pub delta_lf_present: bool,
    /// delta_lf_multi.
    pub delta_lf_multi: bool,
    /// delta_q_res (log2 of delta_q resolution).
    pub delta_q_res: u8,
    /// delta_lf_res (log2 of delta_lf resolution).
    pub delta_lf_res: u8,
    /// base_qindex.
    pub base_q_index: u8,
    /// delta_q_y_dc.
    pub delta_q_y_dc: i8,
    /// delta_q_u_dc.
    pub delta_q_u_dc: i8,
    /// delta_q_u_ac.
    pub delta_q_u_ac: i8,
    /// delta_q_v_dc.
    pub delta_q_v_dc: i8,
    /// delta_q_v_ac.
    pub delta_q_v_ac: i8,
    /// using_qmatrix.
    pub using_qmatrix: bool,
    /// qm_y.
    pub qm_y: u8,
    /// qm_u.
    pub qm_u: u8,
    /// qm_v.
    pub qm_v: u8,
    /// segmentation_enabled.
    pub segmentation_enabled: bool,
    /// segmentation_update_map.
    pub segmentation_update_map: bool,
    /// segmentation_temporal_update.
    pub segmentation_temporal_update: bool,
    /// segmentation_update_data.
    pub segmentation_update_data: bool,
    /// segmentation_abs_or_delta_update.
    pub segmentation_abs_or_delta_update: bool,
    /// Per-segment feature enabled bitmask (8 segments).
    pub segment_feature_enabled: [u8; 8],
    /// Per-segment feature data [8 segments][8 features].
    pub segment_feature_data: [[i16; 8]; 8],
    /// loop_filter_level [y, uv].
    pub loop_filter_level: [u8; 2],
    /// loop_filter_level_uv [u, v].
    pub loop_filter_level_uv: [u8; 2],
    /// loop_filter_sharpness.
    pub loop_filter_sharpness: u8,
    /// loop_filter_delta_enabled.
    pub loop_filter_delta_enabled: bool,
    /// loop_filter_delta_update (mode_ref_delta_update).
    pub loop_filter_delta_update: bool,
    /// loop_filter_ref_deltas (8 refs).
    pub loop_filter_ref_deltas: [i8; 8],
    /// loop_filter_mode_deltas (2).
    pub loop_filter_mode_deltas: [i8; 2],
    /// cdef_damping (cdef_damping_minus_3).
    pub cdef_damping: u8,
    /// cdef_bits.
    pub cdef_bits: u8,
    /// cdef_y_pri_strength (8).
    pub cdef_y_pri_strength: [u8; 8],
    /// cdef_y_sec_strength (8).
    pub cdef_y_sec_strength: [u8; 8],
    /// cdef_uv_pri_strength (8).
    pub cdef_uv_pri_strength: [u8; 8],
    /// cdef_uv_sec_strength (8).
    pub cdef_uv_sec_strength: [u8; 8],
    /// Whether the frame is coded lossless (all segments lossless).
    pub coded_lossless: bool,
    /// Whether the frame is all lossless (coded_lossless && no superres).
    pub all_lossless: bool,
    /// apply_grain (film grain applied this frame).
    pub apply_grain: bool,
    /// showable_frame (derived).
    pub showable_frame: bool,
    /// coded_denom (superres denominator; 0 when no superres).
    pub coded_denom: u8,
    /// Order hints for the 8 reference frame names.
    pub order_hints: [u8; 8],
    /// Loop restoration type per plane [y, u, v] (StdVideo enum values).
    pub loop_restoration_type: [u8; 3],
    /// Loop restoration size (log2) per plane [y, u, v].
    pub loop_restoration_size: [u16; 3],
    /// Whether luma loop restoration is used.
    pub uses_lr: bool,
    /// Global motion type per model [7] (IDENTITY=0, TRANSLATION=1, AFFINE=2, ROTZOOM=3).
    pub global_motion_type: [u8; 7],
    /// Global motion parameters [7 models][6 params] (wmmat).
    pub global_motion_params: [[i32; 6]; 7],
    /// Size of the uncompressed frame header in BYTES (rounded up from bits).
    /// Used to compute the tile data offset within the bitstream buffer.
    /// 0 for show_existing_frame / reduced_still_picture frames (not decoded).
    pub frame_header_size: u32,
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
    /// Per-frame-buffer global motion models [8 slots][7 models] of (type, wmmat[6]).
    ref_global_models: [[(u8, [i32; 6]); 7]; 8],
    /// Per-frame-buffer segmentation [8 slots] of (FeatureEnabled[8], FeatureData[8][8]).
    ref_segmentation: [([u8; 8], [[i16; 8]; 8]); 8],
    /// Per-frame-buffer loop filter deltas [8 slots] of (ref_deltas[8], mode_deltas[2]).
    ref_loop_filter: [([i8; 8], [i8; 2]); 8],
    /// Persistent CDEF strengths carried across frames (AV1: levels not re-coded
    /// in the current frame inherit the previous frame's values). Mirrors the
    /// C++ reference's persistent `m_PicData.CDEF`. Format:
    /// (cdef_damping, cdef_bits, y_pri[8], y_sec[8], uv_pri[8], uv_sec[8]).
    last_cdef: (u8, u8, [u8; 8], [u8; 8], [u8; 8], [u8; 8]),
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
            ref_global_models: [Self::default_global_models(); 8],
            ref_segmentation: [(([0; 8]), [[0; 8]; 8]); 8],
            ref_loop_filter: [(([0; 8]), [0; 2]); 8],
            last_cdef: (0, 0, [0; 8], [0; 8], [0; 8], [0; 8]),
        }
    }

    /// Default global motion models: all identity {type=0, wmmat=[0,0,65536,0,0,65536]}.
    fn default_global_models() -> [(u8, [i32; 6]); 7] {
        let identity = (0u8, [0i32, 0, 65536, 0, 0, 65536]);
        [identity; 7]
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

        // OBU size is byte-aligned after the OBU header (1 or 2 bytes).
        // Skip to the correct byte offset directly.
        let header_bytes = 1usize + usize::from(header.extension_flag);
        if data.len() <= header_bytes {
            return Err(ParserError::InvalidBitstream);
        }
        let mut size_reader = BitReader::new(&data[header_bytes..], false);

        let obu_size: usize = if header.has_size_field {
            let size = size_reader.read_leb128()? as usize;

            size
        } else {
            obu_length
                .ok_or(ParserError::InvalidBitstream)?
                .checked_sub(1)
                .and_then(|v| v.checked_sub(usize::from(header.extension_flag)))
                .ok_or(ParserError::InvalidBitstream)?
        };

        // Update Annex B state if applicable
        if let StreamFormat::AnnexB(ref mut annexb_state) = self.stream_format {
            // ...
        }

        let size_bytes = (size_reader.position() / 8) as usize;
        let start_offset = header_bytes + size_bytes;


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
        sps.frame_id_numbers_present_flag = if sps.reduced_still_picture_header {
            false
        } else {
            r.read_bit()?
        };


        if sps.frame_id_numbers_present_flag {
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
                sps.enable_ref_frame_mvs = r.read_bit()?;

            } else {
                sps.enable_jnt_motion = false;
                sps.enable_second_ref_frame = false;
                sps.enable_ref_frame_mvs = false;
            }

            // seq_choose_screen_content_tools (1 bit)
            let seq_choose_screen_content_tools = r.read_bit()?;
            if seq_choose_screen_content_tools {
                sps.seq_force_screen_content_tools = 2; // SELECT (STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS = 2)
            } else {
                // seq_force_screen_content_tools (1 bit)
                sps.seq_force_screen_content_tools = r.read_bit()? as u8;
            }

            if sps.seq_force_screen_content_tools > 0 {
                // seq_choose_integer_mv (1 bit)
                let seq_choose_integer_mv = r.read_bit()?;
                if seq_choose_integer_mv {
                    sps.seq_force_integer_mv = 2; // SELECT (STD_VIDEO_AV1_SELECT_INTEGER_MV = 2)
                } else {
                    // seq_force_integer_mv (1 bit)
                    sps.seq_force_integer_mv = r.read_bit()? as u8;
                }
            } else {
                sps.seq_force_integer_mv = 2; // SELECT (STD_VIDEO_AV1_SELECT_INTEGER_MV = 2)
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
        let start_pos = r.position();

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
        sps.separate_uv_delta_q = r.read_bit()?;

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
            // Some encoders may not include trailing bits - be lenient
            return Ok(());
        }

        // trailing_zero_bits (remaining bits should be zero)
        let remaining = remaining_bits - 1;
        if remaining > 0 {
            let remaining_u8 = (remaining.min(31)) as u8;
            let trailing_zeros = r.read_bits(remaining_u8)?;
            // Be lenient: don't fail on non-zero trailing bits
            // Some encoders may use the remaining bits for other purposes
        }

        Ok(())
    }

    /// Parse AV1 uncompressed frame header from Frame/FrameHeader OBU data.
    /// Extracts fields needed for hardware decode (Vulkan Video / VAAPI).
    pub fn parse_frame_header(
        &mut self,
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
            fh.primary_ref_frame = 7; // NONE
            fh.show_frame = true;
            fh.showable_frame = false;
            fh.error_resilient_mode = true;
            fh.refresh_frame_flags = 0xFF;
            fh.frame_width = sps.max_frame_width_minus_1 as u32 + 1;
            fh.frame_height = sps.max_frame_height_minus_1 as u32 + 1;
            fh.render_width = fh.frame_width;
            fh.render_height = fh.frame_height;
            fh.frame_header_size = 0; // not decoded
            return Ok(fh);
        }

        // 1. show_existing_frame (1 bit)
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] start: obu_data.len={} bitpos=0", obu_data.len());
        }
        fh.show_existing_frame = r.read_bit()?;
        if fh.show_existing_frame {
            fh.frame_to_show_map_idx = r.read_bits(3)? as u8;
            fh.frame_header_size = 0; // not decoded
            return Ok(fh);
        }

        // 2. frame_type (2 bits)
        fh.frame_type = r.read_bits(2)? as u8;
        let frame_is_intra = fh.frame_type == 0 || fh.frame_type == 2; // KEY or INTRA_ONLY

        // 3. show_frame (1 bit)
        fh.show_frame = r.read_bit()?;
        let show_frame = fh.show_frame;
        fh.showable_frame = if show_frame {
            fh.frame_type != 0 // not KEY
        } else {
            r.read_bit()?
        };
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] show_frame={} showable={} bitpos={}", show_frame, fh.showable_frame, r.position());
        }

        // 4. error_resilient_mode (inferred for SWITCH || (KEY && show_frame))
        fh.error_resilient_mode =
            if fh.frame_type == 3 || (fh.frame_type == 0 && show_frame) {
                true
            } else {
                r.read_bit()?
            };
        let error_resilient = fh.error_resilient_mode;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] error_resilient={} bitpos={}", error_resilient, r.position());
        }

        // 5. disable_cdf_update (1 bit)
        fh.disable_cdf_update = r.read_bit()?;
        let disable_cdf = fh.disable_cdf_update;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] disable_cdf={} bitpos={}", disable_cdf, r.position());
        }

        // 6. allow_screen_content_tools
        fh.allow_screen_content_tools = if sps.seq_force_screen_content_tools == 2 {
            r.read_bit()?
        } else {
            sps.seq_force_screen_content_tools != 0
        };
        let allow_sct = fh.allow_screen_content_tools;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] allow_sct={} bitpos={}", allow_sct, r.position());
        }

        // 7. force_integer_mv
        fh.force_integer_mv = if allow_sct {
            if sps.seq_force_integer_mv == 2 {
                r.read_bit()?
            } else {
                sps.seq_force_integer_mv != 0
            }
        } else {
            false
        };
        if frame_is_intra {
            fh.force_integer_mv = true;
        }
        let force_imv = fh.force_integer_mv;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] force_imv={} bitpos={}", force_imv, r.position());
        }

        // 8. frame_id (if frame_id_numbers_present)
        if sps.frame_id_numbers_present_flag {
            let id_len = sps.additional_frame_id_length_minus1 as u8
                + sps.delta_frame_id_length_minus2 as u8
                + 3;
            let _frame_id = r.read_bits(id_len)?;
        }

        // 9. frame_size_override_flag
        fh.frame_size_override_flag = if fh.frame_type == 3 {
            true
        } else {
            r.read_bit()?
        };
        let fso = fh.frame_size_override_flag;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] fso={} bitpos={}", fso, r.position());
        }

        // 10. order_hint
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] before order_hint: bitpos={}", r.position());
        }
        if sps.enable_order_hint {
            fh.order_hint = r.read_bits(sps.order_hint_bits_minus1 + 1)?;
        }
        let order_hint = fh.order_hint;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] order_hint={} bitpos={}", order_hint, r.position());
        }

        // 11. primary_ref_frame
        fh.primary_ref_frame = if frame_is_intra || error_resilient {
            7 // NONE
        } else {
            r.read_bits(3)? as u8
        };
        let primary_ref = fh.primary_ref_frame;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] primary_ref={} bitpos={}", primary_ref, r.position());
        }

        // 13. refresh_frame_flags
        if vacc_debug() {
            eprintln!(
                "[AV1-PARSE-DBG] before refresh_frame_flags: frame_type={} bitpos={}",
                fh.frame_type, r.position()
            );
        }
        fh.refresh_frame_flags =
            if fh.frame_type == 3 || (fh.frame_type == 0 && show_frame) {
                0xFF
            } else {
                r.read_bits(8)? as u8
            };
        let refresh = fh.refresh_frame_flags;
        if vacc_debug() {
            eprintln!("[AV1-PARSE-DBG] refresh_frame_flags={:08b} bitpos={}", refresh, r.position());
        }

        // 14. ref_order_hint (if !intra || refresh != 0xFF, and error_resilient, and order_hint)
        if (!frame_is_intra || refresh != 0xFF)
            && sps.enable_order_hint
            && error_resilient
        {
            for _i in 0..8 {
                let _oh = r.read_bits(sps.order_hint_bits_minus1 + 1)?;
            }
        }

        // Helper closure: read superres + render_size (shared by frame_size paths)
        let mut read_superres_render = |fh: &mut Av1FrameHeader, r: &mut BitReader| -> ParserResult<()> {
            if sps.enable_superres {
                fh.use_superres = r.read_bit()?;
                if fh.use_superres {
                    fh.coded_denom = r.read_bits(3)? as u8; // coded_denom_minus_4 raw (match C++ VulkanAV1Decoder.cpp)
                } else {
                    fh.coded_denom = 0;
                }
            } else {
                fh.use_superres = false;
                fh.coded_denom = 0;
            }
            Ok(())
        };
        let mut read_render_size = |fh: &mut Av1FrameHeader, r: &mut BitReader| -> ParserResult<()> {
            fh.render_and_frame_size_different = r.read_bit()?;
            if fh.render_and_frame_size_different {
                fh.render_width = r.read_bits(16)? + 1;
                fh.render_height = r.read_bits(16)? + 1;
            } else {
                fh.render_width = fh.frame_width;
                fh.render_height = fh.frame_height;
            }
            Ok(())
        };
        let inherit_size = |fh: &mut Av1FrameHeader, primary_ref: u8, sizes: &[(u32, u32); 8], sps: &vk_video_core::picture::Av1Sps| {
            match sizes.get(primary_ref as usize) {
                Some(&(w, h)) if w > 0 && h > 0 => {
                    fh.frame_width = w;
                    fh.frame_height = h;
                }
                _ => {
                    fh.frame_width = sps.max_frame_width_minus_1 as u32 + 1;
                    fh.frame_height = sps.max_frame_height_minus_1 as u32 + 1;
                }
            }
        };

        if frame_is_intra {
            // ---- INTRA (KEY / INTRA_ONLY) ----
            // frame_size()
            if fso {
                fh.frame_width = r.read_bits(sps.frame_width_bits)? + 1;
                fh.frame_height = r.read_bits(sps.frame_height_bits)? + 1;
            } else {
                inherit_size(&mut fh, primary_ref, &self.ref_frame_sizes, sps);
            }
            read_superres_render(&mut fh, &mut r)?;
            read_render_size(&mut fh, &mut r)?;
            // allow_intrabc
            fh.allow_intrabc = if allow_sct {
                r.read_bit()?
            } else {
                false
            };
            fh.use_ref_frame_mvs = false;
        } else {
            // ---- INTER ----
            // frame_refs_short_signaling
            fh.frame_refs_short_signaling = if sps.enable_order_hint {
                r.read_bit()?
            } else {
                false
            };
            let frss = fh.frame_refs_short_signaling;
            if vacc_debug() {
                eprintln!(
                    "[AV1-PARSE-DBG] frame_refs_short_signaling={} bitpos={}",
                    frss, r.position()
                );
            }
            if frss {
                let last_frame_idx = r.read_bits(3)? as u8;
                let golden_frame_idx = r.read_bits(3)? as u8;
                if vacc_debug() {
                    eprintln!(
                        "[AV1-PARSE-DBG] last_frame_idx={} golden_frame_idx={}",
                        last_frame_idx, golden_frame_idx
                    );
                }
                self.set_frame_refs(last_frame_idx, golden_frame_idx, order_hint, &mut fh);
            } else {
                for i in 0..7 {
                    fh.ref_frame_idx[i] = r.read_bits(3)? as u8;
                }
                if vacc_debug() {
                    eprintln!("[AV1-PARSE-DBG] ref_frame_idx={:?}", fh.ref_frame_idx);
                }
            }

            // frame_size (with refs if fso && !error_resilient)
            if fso && !error_resilient {
                // frame_size_with_refs()
                let mut found = false;
                for i in 0..7 {
                    let found_ref = r.read_bit()?;
                    if found_ref {
                        let ref_idx = fh.ref_frame_idx[i];
                        inherit_size(&mut fh, ref_idx, &self.ref_frame_sizes, sps);
                        fh.render_width = fh.frame_width;
                        fh.render_height = fh.frame_height;
                        found = true;
                        break;
                    }
                }
                if found {
                    read_superres_render(&mut fh, &mut r)?;
                } else {
                    if fso {
                        fh.frame_width = r.read_bits(sps.frame_width_bits)? + 1;
                        fh.frame_height = r.read_bits(sps.frame_height_bits)? + 1;
                    } else {
                        inherit_size(&mut fh, primary_ref, &self.ref_frame_sizes, sps);
                    }
                    read_superres_render(&mut fh, &mut r)?;
                    read_render_size(&mut fh, &mut r)?;
                }
            } else {
                // frame_size() + render_size()
                if fso {
                    fh.frame_width = r.read_bits(sps.frame_width_bits)? + 1;
                    fh.frame_height = r.read_bits(sps.frame_height_bits)? + 1;
                } else {
                    inherit_size(&mut fh, primary_ref, &self.ref_frame_sizes, sps);
                }
                read_superres_render(&mut fh, &mut r)?;
                read_render_size(&mut fh, &mut r)?;
            }

            // allow_high_precision_mv
            fh.allow_high_precision_mv = if force_imv {
                false
            } else {
                r.read_bit()?
            };

            // interpolation_filter
            fh.is_filter_switchable = r.read_bit()?;
            if fh.is_filter_switchable {
                fh.interpolation_filter = 4; // SWITCHABLE
            } else {
                fh.interpolation_filter = r.read_bits(2)? as u8;
            }

            // is_motion_mode_switchable
            fh.is_motion_mode_switchable = r.read_bit()?;

            // use_ref_frame_mvs
            fh.use_ref_frame_mvs = if error_resilient || !sps.enable_ref_frame_mvs {
                false
            } else {
                r.read_bit()?
            };

            // order_hints for the 7 refs
            for i in 0..7 {
                let ref_idx = fh.ref_frame_idx[i] as usize;
                fh.order_hints[i] = self
                    .ref_frame_order_hints
                    .get(ref_idx)
                    .copied()
                    .unwrap_or(0) as u8;
            }
        }

        // 16. disable_frame_end_update_cdf
        fh.disable_frame_end_update_cdf = if disable_cdf {
            true
        } else {
            r.read_bit()?
        };

        // 17. tile_info
        self.parse_tile_info(&mut r, &mut fh, sps)?;

        // 18. quantization
        self.parse_quantization(&mut r, &mut fh, sps)?;

        // 19. segmentation
        self.parse_segmentation(&mut r, &mut fh, sps, primary_ref)?;

        // 20. delta_q_params + 21. delta_lf_params
        self.parse_delta_q_lf(&mut r, &mut fh)?;

        // 22. coded_lossless
        let mut coded_lossless = true;
        for i in 0..8 {
            let qindex = if fh.segmentation_enabled && (fh.segment_feature_enabled[i] & (1 << 2)) != 0 {
                (fh.base_q_index as i16 + fh.segment_feature_data[i][2]).clamp(0, 255)
            } else {
                fh.base_q_index as i16
            };
            if qindex != 0
                || fh.delta_q_y_dc != 0
                || fh.delta_q_u_dc != 0
                || fh.delta_q_u_ac != 0
                || fh.delta_q_v_dc != 0
                || fh.delta_q_v_ac != 0
            {
                coded_lossless = false;
            }
        }
        fh.coded_lossless = coded_lossless;
        fh.all_lossless = coded_lossless && !fh.use_superres;

        // 23. loop_filter
        self.parse_loop_filter(&mut r, &mut fh, sps, coded_lossless)?;

        // 24. cdef
        self.parse_cdef(&mut r, &mut fh, sps, coded_lossless)?;

        // 25. loop restoration
        let all_lossless = fh.all_lossless;
        self.parse_loop_restoration(&mut r, &mut fh, sps, all_lossless)?;

        // 26. tx_mode
        fh.tx_mode = if coded_lossless {
            0 // ONLY_4X4
        } else {
            r.read_bits(1)? as u8 + 1 // LARGEST(1) or SELECT(2)
        };

        // 27. reference_select
        fh.reference_select = if !frame_is_intra {
            r.read_bit()?
        } else {
            false
        };

        // 28. skip_mode
        let skip_refs = self.skip_mode_refs(&fh, sps);
        fh.skip_mode_present = if skip_refs.is_some() {
            r.read_bit()?
        } else {
            false
        };
        if let Some((ref0, ref1)) = skip_refs {
            // C++ VulkanAV1Decoder.cpp IsSkipModeAllowed: SkipModeFrame holds
            // the reference name indices (1-based: LAST=1..ALTREF=7, i.e.
            // bitstream ref name + 1) of the nearest forward (ref0) and
            // backward (ref1) references. The NVIDIA driver uses these to pick
            // the references for skip-mode blocks; passing [0,0] made
            // multi-reference frames decode to the first reference's content.
            fh.skip_mode_frame = [
                std::cmp::min(ref0, ref1) as u8 + 1,
                std::cmp::max(ref0, ref1) as u8 + 1,
            ];
        }

        // 29. allow_warped_motion
        fh.allow_warped_motion = if !frame_is_intra && !error_resilient && sps.enable_warped_motion {
            r.read_bit()?
        } else {
            false
        };

        // 30. reduced_tx_set
        fh.reduced_tx_set = r.read_bit()?;

        // 31. global_motion (inter only)
        if !frame_is_intra {
            self.parse_global_motion(&mut r, &mut fh, sps, primary_ref)?;
        }

        // 32. film_grain (not present in our SPS)
        fh.apply_grain = false;

        // Update reference frame tracking state for subsequent frames
        self.update_ref_frames(&fh);

        // Size of the uncompressed frame header in bytes (rounded up from bits).
        fh.frame_header_size = ((r.position() + 7) / 8) as u32;

        Ok(fh)
    }

    /// AV1 spec 7.8: derive the 7 reference frame indices from last/golden indices.
    fn set_frame_refs(
        &self,
        last_frame_idx: u8,
        golden_frame_idx: u8,
        order_hint: u32,
        fh: &mut Av1FrameHeader,
    ) {
        let sps = match &self.active_sps {
            Some(s) => s,
            None => return,
        };
        let ohb = sps.order_hint_bits_minus1 as u32;
        let cur_frame_hint = 1 << ohb;

        let mut ref_idx: [i32; 7] = [-1; 7];
        let mut used = [false; 8];

        ref_idx[0] = last_frame_idx as i32; // LAST
        ref_idx[3] = golden_frame_idx as i32; // GOLDEN
        used[last_frame_idx as usize] = true;
        used[golden_frame_idx as usize] = true;

        // shiftedOrderHints[i] = curFrameHint + GetRelativeDist1(RefOrderHint[i], OrderHint)
        let shifted: Vec<i32> = (0..8)
            .map(|i| {
                cur_frame_hint as i32
                    + Self::get_relative_dist1(
                        self.ref_frame_order_hints[i] as i32,
                        order_hint as i32,
                        ohb,
                    )
            })
            .collect();

        // ALTREF_FRAME (idx 6): unused, hint>=cur, MAX hint
        let mut best = -1i32;
        let mut best_hint = -1i32;
        for i in 0..8 {
            if !used[i] && shifted[i] >= cur_frame_hint as i32 && (best < 0 || shifted[i] >= best_hint) {
                best = i as i32;
                best_hint = shifted[i];
            }
        }
        if best >= 0 {
            ref_idx[6] = best;
            used[best as usize] = true;
        }
        // BWDREF_FRAME (idx 4): unused, hint>=cur, MIN hint
        let mut best = -1i32;
        let mut best_hint = -1i32;
        for i in 0..8 {
            if !used[i] && shifted[i] >= cur_frame_hint as i32 && (best < 0 || shifted[i] < best_hint) {
                best = i as i32;
                best_hint = shifted[i];
            }
        }
        if best >= 0 {
            ref_idx[4] = best;
            used[best as usize] = true;
        }
        // ALTREF2_FRAME (idx 5): unused, hint>=cur, MIN hint
        let mut best = -1i32;
        let mut best_hint = -1i32;
        for i in 0..8 {
            if !used[i] && shifted[i] >= cur_frame_hint as i32 && (best < 0 || shifted[i] < best_hint) {
                best = i as i32;
                best_hint = shifted[i];
            }
        }
        if best >= 0 {
            ref_idx[5] = best;
            used[best as usize] = true;
        }
        // Ref_Frame_List = [LAST2(1), LAST3(2), BWDREF(4), ALTREF2(5), ALTREF(6)]: unused, hint<cur, MAX hint
        for name in [1, 2, 4, 5, 6] {
            if ref_idx[name] < 0 {
                let mut best = -1i32;
                let mut best_hint = -1i32;
                for i in 0..8 {
                    if !used[i] && shifted[i] < cur_frame_hint as i32 && (best < 0 || shifted[i] >= best_hint) {
                        best = i as i32;
                        best_hint = shifted[i];
                    }
                }
                if best >= 0 {
                    ref_idx[name] = best;
                    used[best as usize] = true;
                }
            }
        }
        // Final: fill remaining with argmin over ALL i of shifted
        let mut fill = 0i32;
        let mut fill_hint = i32::MAX;
        for i in 0..8 {
            if shifted[i] < fill_hint {
                fill = i as i32;
                fill_hint = shifted[i];
            }
        }
        for i in 0..7 {
            if ref_idx[i] < 0 {
                ref_idx[i] = fill;
            }
        }

        for i in 0..7 {
            fh.ref_frame_idx[i] = ref_idx[i] as u8;
        }
    }

    /// AV1 spec 7.10: GetRelativeDist1(a, b).
    fn get_relative_dist1(a: i32, b: i32, ohb: u32) -> i32 {
        let bits = ohb + 1;
        let diff = a - b;
        let m = 1 << (bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    /// AV1 spec 7.11: IsSkipModeAllowed. Returns (ref0, ref1) as bitstream
    /// reference name indices (0=LAST..6=INTRA) of the nearest forward and
    /// backward references, or None if skip mode is not allowed.
    fn skip_mode_refs(&self, fh: &Av1FrameHeader, sps: &vk_video_core::picture::Av1Sps) -> Option<(i32, i32)> {
        if !sps.enable_order_hint || fh.frame_type == 0 || fh.frame_type == 2 || !fh.reference_select {
            return None;
        }
        let ohb = sps.order_hint_bits_minus1 as u32;
        let cur = fh.order_hint as i32;

        let mut ref0 = -1i32;
        let mut ref1 = -1i32;
        let mut ref0_off = -1i32;
        let mut ref1_off = -1i32;
        for i in 0..7 {
            let frame_idx = fh.ref_frame_idx[i] as usize;
            let ref_off = self.ref_frame_order_hints.get(frame_idx).copied().unwrap_or(0) as i32;
            let rel_off = Self::get_relative_dist1(ref_off, cur, ohb);
            if rel_off < 0 && (ref0_off == -1 || Self::get_relative_dist1(ref_off, ref0_off, ohb) > 0) {
                ref0 = i as i32;
                ref0_off = ref_off;
            }
            if rel_off > 0 && (ref1_off == -1 || Self::get_relative_dist1(ref_off, ref1_off, ohb) < 0) {
                ref1 = i as i32;
                ref1_off = ref_off;
            }
        }
        if ref0 != -1 && ref1 != -1 {
            return Some((ref0, ref1));
        }
        if ref0 != -1 {
            for i in 0..7 {
                let frame_idx = fh.ref_frame_idx[i] as usize;
                let ref_off = self.ref_frame_order_hints.get(frame_idx).copied().unwrap_or(0) as i32;
                if Self::get_relative_dist1(ref_off, ref0_off, ohb) < 0
                    && (ref1_off == -1 || Self::get_relative_dist1(ref_off, ref1_off, ohb) > 0)
                {
                    ref1 = i as i32;
                    ref1_off = ref_off;
                }
            }
            if ref1 != -1 {
                return Some((ref0, ref1));
            }
        }
        None
    }

    /// AV1 spec 7.22: tile_info.
    fn parse_tile_info(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
    ) -> ParserResult<()> {
        let frame_width = if fh.frame_width > 0 { fh.frame_width } else { sps.max_frame_width_minus_1 as u32 + 1 };
        let frame_height = if fh.frame_height > 0 { fh.frame_height } else { sps.max_frame_height_minus_1 as u32 + 1 };

        let mi_cols = 2 * ((frame_width + 7) >> 3);
        let mi_rows = 2 * ((frame_height + 7) >> 3);
        let use_128 = sps.use_128x128_superblock;
        let sb_cols = if use_128 { (mi_cols + 31) >> 5 } else { (mi_cols + 15) >> 4 };
        let sb_rows = if use_128 { (mi_rows + 31) >> 5 } else { (mi_rows + 15) >> 4 };
        let sb_shift = if use_128 { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let _ = mi_cols;
        let _ = mi_rows;

        let max_tile_width_sb = 4096u32 >> sb_size;
        let max_tile_area_sb = (4096u32 * 2304) >> (2 * sb_size);
        let tile_log2 = |blk_size: u32, target: u32| -> u32 {
            let mut k = 0u32;
            while (blk_size << k) < target {
                k += 1;
            }
            k
        };
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, std::cmp::min(sb_cols, 64));
        let max_log2_tile_rows = tile_log2(1, std::cmp::min(sb_rows, 64));
        let min_log2_tiles = std::cmp::max(min_log2_tile_cols, tile_log2(max_tile_area_sb, sb_rows * sb_cols));

        let uniform = r.read_bit()?;
        fh.uniform_tile_spacing_flag = uniform;
        let mut log2_tile_cols = 0u32;
        let mut log2_tile_rows = 0u32;
        let mut tile_cols = 0u32;
        let mut tile_rows = 0u32;

        if uniform {
            log2_tile_cols = min_log2_tile_cols;
            while log2_tile_cols < max_log2_tile_cols {
                if !r.read_bit()? {
                    break;
                }
                log2_tile_cols += 1;
            }
            let tile_width_sb = (sb_cols + (1 << log2_tile_cols) - 1) >> log2_tile_cols;
            tile_cols = (sb_cols + tile_width_sb - 1) / tile_width_sb;
            let min_log2_tile_rows = std::cmp::max(min_log2_tiles - log2_tile_cols, 0);
            log2_tile_rows = min_log2_tile_rows;
            while log2_tile_rows < max_log2_tile_rows {
                if !r.read_bit()? {
                    break;
                }
                log2_tile_rows += 1;
            }
            let tile_height_sb = (sb_rows + (1 << log2_tile_rows) - 1) >> log2_tile_rows;
            tile_rows = (sb_rows + tile_height_sb - 1) / tile_height_sb;

            // Derive per-tile sizes + MI starts (mirrors C++ VulkanAV1Decoder.cpp:1222-1245).
            for c in 0..tile_cols {
                fh.tile_width_in_sbs_minus_1[c as usize] = if c < tile_cols - 1 {
                    (tile_width_sb - 1) as u16
                } else {
                    (sb_cols - (tile_cols - 1) * tile_width_sb - 1) as u16
                };
            }
            for rr in 0..tile_rows {
                fh.tile_height_in_sbs_minus_1[rr as usize] = if rr < tile_rows - 1 {
                    (tile_height_sb - 1) as u16
                } else {
                    (sb_rows - (tile_rows - 1) * tile_height_sb - 1) as u16
                };
            }
            let mut start_sb = 0u32;
            for i in 0..tile_cols {
                fh.tile_mi_col_starts[i as usize] = (start_sb << sb_shift) as u16;
                start_sb += tile_width_sb;
            }
            let mut start_sb = 0u32;
            for i in 0..tile_rows {
                fh.tile_mi_row_starts[i as usize] = (start_sb << sb_shift) as u16;
                start_sb += tile_height_sb;
            }
        } else {
            // non-uniform: read_ns for each tile width/height (mirrors C++ 1246-1278)
            let mut start_sb = 0u32;
            let mut i = 0u32;
            let mut widest_tile_sb = 0u32;
            while start_sb < sb_cols && i < 64 {
                fh.tile_mi_col_starts[i as usize] = (start_sb << sb_shift) as u16;
                let max_width = std::cmp::min(sb_cols - start_sb, max_tile_width_sb);
                let w = if max_width > 1 { r.read_ns(max_width - 1)? } else { 0 };
                fh.tile_width_in_sbs_minus_1[i as usize] = w as u16;
                let size_sb = w + 1;
                widest_tile_sb = std::cmp::max(size_sb, widest_tile_sb);
                start_sb += size_sb;
                i += 1;
            }
            log2_tile_cols = tile_log2(1, i);
            tile_cols = i;

            let num_sb = sb_cols * sb_rows;
            let max_tile_area_sb = if min_log2_tiles > 0 { num_sb >> (min_log2_tiles + 1) } else { num_sb };
            let max_tile_height_sb = std::cmp::max(max_tile_area_sb / std::cmp::max(1, widest_tile_sb), 1u32);
            let mut start_sb = 0u32;
            let mut i = 0u32;
            while start_sb < sb_rows && i < 64 {
                fh.tile_mi_row_starts[i as usize] = (start_sb << sb_shift) as u16;
                let max_height = std::cmp::min(sb_rows - start_sb, max_tile_height_sb);
                let h = if max_height > 1 { r.read_ns(max_height - 1)? } else { 0 };
                fh.tile_height_in_sbs_minus_1[i as usize] = h as u16;
                let size_sb = h + 1;
                start_sb += size_sb;
                i += 1;
            }
            log2_tile_rows = tile_log2(1, i);
            tile_rows = i;
        }

        fh.tile_cols_log2 = log2_tile_cols as u8;
        fh.tile_rows_log2 = log2_tile_rows as u8;
        fh.tile_cols = tile_cols;
        fh.tile_rows = tile_rows;
        fh.tile_count = tile_rows * tile_cols;

        let mut context_update_tile_id = 0u32;
        let mut tile_size_bytes_minus_1 = 3u32;
        if tile_rows * tile_cols > 1 {
            context_update_tile_id = r.read_bits((log2_tile_rows + log2_tile_cols) as u8)?;
            tile_size_bytes_minus_1 = r.read_bits(2)?;
        }
        fh.context_update_tile_id = context_update_tile_id;
        fh.tile_size_bytes_minus_1 = tile_size_bytes_minus_1;

        Ok(())
    }

    /// AV1 spec 7.23: quantization_params.
    fn parse_quantization(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
    ) -> ParserResult<()> {
        fh.base_q_index = r.read_bits(8)? as u8;

        // ReadDeltaQ(6) = u(1) ? 7-bit signed : 0
        let read_delta_q = |r: &mut BitReader| -> ParserResult<i8> {
            if r.read_bit()? {
                Ok(r.read_signed_bits(7)? as i8)
            } else {
                Ok(0)
            }
        };

        fh.delta_q_y_dc = read_delta_q(r)?;
        if !sps.mono_chrome {
            let diff_uv_delta = if sps.separate_uv_delta_q { r.read_bit()? } else { false };
            fh.diff_uv_delta = diff_uv_delta;
            fh.delta_q_u_dc = read_delta_q(r)?;
            fh.delta_q_u_ac = read_delta_q(r)?;
            if diff_uv_delta {
                fh.delta_q_v_dc = read_delta_q(r)?;
                fh.delta_q_v_ac = read_delta_q(r)?;
            } else {
                fh.delta_q_v_dc = fh.delta_q_u_dc;
                fh.delta_q_v_ac = fh.delta_q_u_ac;
            }
        } else {
            fh.delta_q_u_dc = 0;
            fh.delta_q_u_ac = 0;
            fh.delta_q_v_dc = 0;
            fh.delta_q_v_ac = 0;
        }

        fh.using_qmatrix = r.read_bit()?;
        if fh.using_qmatrix {
            fh.qm_y = r.read_bits(4)? as u8;
            fh.qm_u = r.read_bits(4)? as u8;
            if sps.separate_uv_delta_q {
                fh.qm_v = r.read_bits(4)? as u8;
            } else {
                fh.qm_v = fh.qm_u;
            }
        } else {
            fh.qm_y = 0;
            fh.qm_u = 0;
            fh.qm_v = 0;
        }
        Ok(())
    }

    /// AV1 spec 7.24: segmentation_params.
    fn parse_segmentation(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
        primary_ref: u8,
    ) -> ParserResult<()> {
        let _ = sps;
        fh.segmentation_enabled = r.read_bit()?;
        if !fh.segmentation_enabled {
            fh.segmentation_update_map = false;
            fh.segmentation_update_data = false;
            fh.segmentation_temporal_update = false;
            return Ok(());
        }

        if primary_ref == 7 {
            fh.segmentation_update_map = true;
            fh.segmentation_update_data = true;
            fh.segmentation_temporal_update = false;
        } else {
            fh.segmentation_update_map = r.read_bit()?;
            fh.segmentation_temporal_update = if fh.segmentation_update_map { r.read_bit()? } else { false };
            fh.segmentation_update_data = r.read_bit()?;
        }

    if fh.segmentation_update_data {
        // C++ VulkanAV1Decoder.cpp:1387: reset FeatureEnabled before reading
        // (avoids OR-ing with a stale value if the header is reused).
        for seg in 0..8 {
            fh.segment_feature_enabled[seg] = 0;
        }
        // feature bits: {8,6,6,6,6,3,0,0}, signed: {1,1,1,1,1,0,0,0}
        let feature_bits = [8u8, 6, 6, 6, 6, 3, 0, 0];
        let feature_signed = [true, true, true, true, true, false, false, false];
        for seg in 0..8 {
            for feat in 0..8 {
                let enabled = r.read_bit()?;
                if enabled {
                    fh.segment_feature_enabled[seg] |= 1 << feat;
                    let bits = feature_bits[feat];
                    if bits > 0 {
                        let val = if feature_signed[feat] {
                            r.read_signed_bits(bits)? as i16
                        } else {
                            r.read_bits(bits)? as i16
                        };
                        fh.segment_feature_data[seg][feat] = val;
                    }
                }
            }
        }
    } else if primary_ref != 7 {
        // C++ VulkanAV1Decoder.cpp:1405-1413: inherit segmentation state from
        // the primary reference frame buffer when segmentation_update_data is
        // false (the feature data is not re-signaled in the bitstream).
        let ref_idx = fh.ref_frame_idx[primary_ref as usize] as usize;
        if let Some((feature_enabled, feature_data)) = self.ref_segmentation.get(ref_idx) {
            fh.segment_feature_enabled = *feature_enabled;
            fh.segment_feature_data = *feature_data;
        }
    }
    Ok(())
    }

    /// AV1 spec 7.25/7.26: delta_q_params + delta_lf_params.
    fn parse_delta_q_lf(&self, r: &mut BitReader, fh: &mut Av1FrameHeader) -> ParserResult<()> {
        fh.delta_q_present = if fh.base_q_index > 0 { r.read_bit()? } else { false };
        if fh.delta_q_present {
            fh.delta_q_res = r.read_bits(2)? as u8;
            if !fh.allow_intrabc {
                fh.delta_lf_present = r.read_bit()?;
                if fh.delta_lf_present {
                    fh.delta_lf_res = r.read_bits(2)? as u8;
                    fh.delta_lf_multi = r.read_bit()?;
                }
            }
        }
        Ok(())
    }

    /// AV1 spec 7.27: loop_filter_params.
    fn parse_loop_filter(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
        coded_lossless: bool,
    ) -> ParserResult<()> {
        // C++ VulkanAV1Decoder.cpp:1417,1426: loop_filter_ref_deltas default
        // {1,0,0,0,-1,0,-1,-1} (NOT all zeros). The deltas persist across frames
        // and are inherited from the primary reference buffer when
        // loop_filter_delta_update is false.
        const LF_REF_DELTA_DEFAULT: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
        fh.loop_filter_ref_deltas = LF_REF_DELTA_DEFAULT;
        fh.loop_filter_mode_deltas = [0, 0];

        if fh.allow_intrabc || coded_lossless {
            fh.loop_filter_level[0] = 0;
            fh.loop_filter_level[1] = 0;
            return Ok(());
        }

        // C++ VulkanAV1Decoder.cpp:1434-1441: inherit ref/mode deltas from the
        // primary reference frame buffer (overrides the default).
        if fh.primary_ref_frame != 7 {
            let prim_buf_idx = fh.ref_frame_idx[fh.primary_ref_frame as usize] as usize;
            if prim_buf_idx < 8 {
                let (ref_deltas, mode_deltas) = self.ref_loop_filter[prim_buf_idx];
                fh.loop_filter_ref_deltas = ref_deltas;
                fh.loop_filter_mode_deltas = mode_deltas;
            }
        }

        fh.loop_filter_level[0] = r.read_bits(6)? as u8;
        fh.loop_filter_level[1] = r.read_bits(6)? as u8;
        if !sps.mono_chrome && (fh.loop_filter_level[0] != 0 || fh.loop_filter_level[1] != 0) {
            fh.loop_filter_level_uv[0] = r.read_bits(6)? as u8;
            fh.loop_filter_level_uv[1] = r.read_bits(6)? as u8;
        }
        fh.loop_filter_sharpness = r.read_bits(3)? as u8;

        fh.loop_filter_delta_enabled = r.read_bit()?;
        if fh.loop_filter_delta_enabled {
            fh.loop_filter_delta_update = r.read_bit()?;
            if fh.loop_filter_delta_update {
                for i in 0..8 {
                    if r.read_bit()? {
                        fh.loop_filter_ref_deltas[i] = r.read_signed_bits(7)? as i8;
                    }
                }
                for i in 0..2 {
                    if r.read_bit()? {
                        fh.loop_filter_mode_deltas[i] = r.read_signed_bits(7)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    /// AV1 spec 7.28: cdef_params.
    ///
    /// Levels not re-coded in the current frame (i >= 1 << cdef_bits) inherit
    /// the previous frame's strengths (persistent state, matching the C++
    /// reference's persistent `m_PicData.CDEF`).
    fn parse_cdef(
        &mut self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
        coded_lossless: bool,
    ) -> ParserResult<()> {
        // Carry over the previous frame's CDEF strengths (all 8 levels).
        let (last_damping, last_bits, last_y_pri, last_y_sec, last_uv_pri, last_uv_sec) =
            self.last_cdef;
        fh.cdef_damping = last_damping;
        fh.cdef_bits = last_bits;
        fh.cdef_y_pri_strength = last_y_pri;
        fh.cdef_y_sec_strength = last_y_sec;
        fh.cdef_uv_pri_strength = last_uv_pri;
        fh.cdef_uv_sec_strength = last_uv_sec;

        if coded_lossless || !sps.enable_cdef || fh.allow_intrabc {
            fh.cdef_bits = 0;
            // Strengths remain the carried-over values (C++ does not reset them).
            self.last_cdef = (
                fh.cdef_damping,
                fh.cdef_bits,
                fh.cdef_y_pri_strength,
                fh.cdef_y_sec_strength,
                fh.cdef_uv_pri_strength,
                fh.cdef_uv_sec_strength,
            );
            return Ok(());
        }
        fh.cdef_damping = r.read_bits(2)? as u8;
        fh.cdef_bits = r.read_bits(2)? as u8;
        let n = 1usize << fh.cdef_bits;
        for i in 0..n {
            fh.cdef_y_pri_strength[i] = r.read_bits(4)? as u8;
            fh.cdef_y_sec_strength[i] = r.read_bits(2)? as u8;
            if !sps.mono_chrome {
                fh.cdef_uv_pri_strength[i] = r.read_bits(4)? as u8;
                fh.cdef_uv_sec_strength[i] = r.read_bits(2)? as u8;
            }
        }
        // Update the persistent state for the next frame.
        self.last_cdef = (
            fh.cdef_damping,
            fh.cdef_bits,
            fh.cdef_y_pri_strength,
            fh.cdef_y_sec_strength,
            fh.cdef_uv_pri_strength,
            fh.cdef_uv_sec_strength,
        );
        Ok(())
    }

    /// AV1 spec 7.29: lr_params.
    fn parse_loop_restoration(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
        all_lossless: bool,
    ) -> ParserResult<()> {
        if all_lossless || !sps.enable_restoration || fh.allow_intrabc {
            fh.uses_lr = false;
            return Ok(());
        }
        let n_planes = if sps.mono_chrome { 1 } else { 3 };
        // bitstream->StdVideo remap: [0,3,1,2]
        let remap = [0u8, 3, 1, 2];
        let mut use_lr = false;
        let mut use_chroma_lr = false;
        for pl in 0..n_planes {
            let lr_type = r.read_bits(2)? as u8;
            fh.loop_restoration_type[pl] = remap[lr_type as usize];
            if fh.loop_restoration_type[pl] != 0 {
                use_lr = true;
                if pl > 0 {
                    use_chroma_lr = true;
                }
            }
        }
        fh.uses_lr = use_lr;
        if use_lr {
            let sb_size = if sps.use_128x128_superblock { 2 } else { 1 };
            for pl in 0..n_planes {
                fh.loop_restoration_size[pl] = sb_size as u16;
            }
            let mut lr_unit_shift = 0u16;
            if sps.use_128x128_superblock {
                lr_unit_shift = 1 + r.read_bit()? as u16;
            } else {
                lr_unit_shift = r.read_bit()? as u16;
                if lr_unit_shift != 0 {
                    lr_unit_shift += r.read_bit()? as u16;
                }
            }
            fh.loop_restoration_size[0] = 1 + lr_unit_shift;
        } else {
            for pl in 0..n_planes {
                fh.loop_restoration_size[pl] = 3;
            }
        }
        let mut lr_uv_shift = 0u16;
        if !sps.mono_chrome {
            if use_chroma_lr && sps.subsampling_x != 0 && sps.subsampling_y != 0 {
                lr_uv_shift = r.read_bit()? as u16;
                fh.loop_restoration_size[1] = fh.loop_restoration_size[0] - lr_uv_shift;
                fh.loop_restoration_size[2] = fh.loop_restoration_size[1];
            } else {
                fh.loop_restoration_size[1] = fh.loop_restoration_size[0];
                fh.loop_restoration_size[2] = fh.loop_restoration_size[0];
            }
        }
        fh.loop_restoration_size[1] = fh.loop_restoration_size[1] >> lr_uv_shift >> lr_uv_shift;
        Ok(())
    }

    /// AV1 spec 7.31: global_motion_params (inter frames only).
    fn parse_global_motion(
        &self,
        r: &mut BitReader,
        fh: &mut Av1FrameHeader,
        sps: &vk_video_core::picture::Av1Sps,
        primary_ref: u8,
    ) -> ParserResult<()> {
        let _ = sps;
        // prev models: from primary ref buffer, else identity
        let prev: [(u8, [i32; 6]); 7] = if primary_ref != 7 {
            let ref_idx = fh.ref_frame_idx[primary_ref as usize] as usize;
            self.ref_global_models
                .get(ref_idx)
                .copied()
                .unwrap_or_else(Self::default_global_models)
        } else {
            Self::default_global_models()
        };

        let allow_hp = fh.allow_high_precision_mv;
        for i in 0..7 {
            let (ref_type, ref_params) = prev[i];
            let _ = ref_type;
            let gm_type = r.read_bit()?;
            let gm_type = if gm_type {
                if r.read_bit()? {
                    2 // ROTZOOM (internal AV1_TRANSFORMATION_TYPE: IDENTITY=0, TRANSLATION=1, ROTZOOM=2, AFFINE=3)
                } else {
                    if r.read_bit()? {
                        1 // TRANSLATION
                    } else {
                        3 // AFFINE
                    }
                }
            } else {
                0 // IDENTITY
            };
            let mut wmmat = [0i32; 6];
            wmmat[2] = 65536;
            wmmat[5] = 65536;

            if gm_type >= 2 {
                wmmat[2] = r.read_signed_refsubexpfin(4097, 3, (ref_params[2] >> 1) - 32768)? * 2 + 65536;
                wmmat[3] = r.read_signed_refsubexpfin(4097, 3, ref_params[3] >> 1)? * 2;
            }
            if gm_type >= 3 {
                wmmat[4] = r.read_signed_refsubexpfin(4097, 3, ref_params[4] >> 1)? * 2;
                wmmat[5] = r.read_signed_refsubexpfin(4097, 3, (ref_params[5] >> 1) - 32768)? * 2 + 65536;
            } else {
                wmmat[4] = -wmmat[3];
                wmmat[5] = wmmat[2];
            }
            if gm_type >= 1 {
                let (tb, tf, td) = if gm_type == 1 {
                    let tb = 9 - (!allow_hp) as u32;
                    let tf = 8192 * (1 << (if allow_hp { 0 } else { 1 }));
                    let td = 13 + (!allow_hp) as u32;
                    (tb, tf, td)
                } else {
                    (12, 1024, 10)
                };
                wmmat[0] = r.read_signed_refsubexpfin((1 << tb) + 1, 3, ref_params[0] >> td)? * tf;
                wmmat[1] = r.read_signed_refsubexpfin((1 << tb) + 1, 3, ref_params[1] >> td)? * tf;
            }

            fh.global_motion_type[i] = gm_type as u8;
            fh.global_motion_params[i] = wmmat;
        }
        Ok(())
    }
    

    /// Update reference frame tracking state after parsing a frame header (AV1 spec 7.20).
    fn update_ref_frames(&mut self, fh: &Av1FrameHeader) {
        // Update sizes for refreshed frame slots
        for i in 0..8usize {
            if (fh.refresh_frame_flags & (1 << i)) != 0 {
                self.ref_frame_sizes[i] = (fh.frame_width, fh.frame_height);
                self.ref_frame_order_hints[i] = fh.order_hint;
                // C++ VulkanAV1Decoder.cpp:401-402: save loop filter ref/mode
                // deltas per refreshed frame buffer for later inheritance.
                self.ref_loop_filter[i] = (fh.loop_filter_ref_deltas, fh.loop_filter_mode_deltas);
                // C++ VulkanAV1Decoder.cpp:399: save the current frame's global
                // motion models per refreshed frame buffer for later inheritance.
                // Without this, ref_global_models stays at the identity default
                // and every subsequent frame's global motion params are decoded
                // against the wrong "previous" model (multi-ref INTER frames
                // then decode to the primary reference's content).
                let mut models: [(u8, [i32; 6]); 7] = Default::default();
                for j in 0..7 {
                    models[j] = (fh.global_motion_type[j], fh.global_motion_params[j]);
                }
                self.ref_global_models[i] = models;
                // C++ VulkanAV1Decoder.cpp:404-405: save the current frame's
                // segmentation state per refreshed frame buffer for later
                // inheritance (same class of bug as the iter-22 global-motion
                // fix). Without this, ref_segmentation stays zero and any
                // frame with segmentation_update_data==0 would inherit the
                // wrong (all-zero) feature state instead of the primary ref's.
                self.ref_segmentation[i] = (fh.segment_feature_enabled, fh.segment_feature_data);
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
                                     sps_nal: None,
                                     pps_nal: None,
                                 });
                            }
                        }
                        ObuType::Frame | ObuType::FrameHeader
                            // If we have a sequence header, treat remaining data as frame data
                            if self.active_sps.is_some() => {
                                self.frame_count += 1;
                                offset += obu_data_offset + obu_size;
                                 let bytes_consumed = data.len().saturating_sub(offset);
                                 return Ok(ParseResult::Slice {
                                     slices: vec![crate::SliceEntry {
                                         slice_header: None,
                                         nal_data: Vec::new(),
                                     }],
                                     bytes_consumed,
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
                slices: vec![crate::SliceEntry {
                    slice_header: None,
                    nal_data: Vec::new(),
                }],
                bytes_consumed: data.len(),
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
