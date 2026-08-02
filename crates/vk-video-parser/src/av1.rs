//! AV1 bitstream parser.
//!
//! Parses AV1 bitstreams to extract sequence headers (SPS equivalent).
//! AV1 uses a different structure than H.264/H.265 - it has a single
//! sequence header that contains all the profile-level information.

use crate::nal::{self, NalUnit};
use crate::{
    nal::remove_emulation_prevention_bytes,
    DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser,
};

/// AV1 parser state.
pub struct Av1Parser {
    /// Sequence header (SPS equivalent).
    active_sps: Option<vk_video_core::picture::Av1Sps>,
    /// Detected format.
    detected_format: DetectedVideoFormat,
    /// Frame counter.
    frame_count: u32,
}

impl Av1Parser {
    pub fn new() -> Self {
        Self {
            active_sps: None,
            detected_format: DetectedVideoFormat::new(
                vk_video_core::codec::VideoCodec::DecodeAv1,
            ),
            frame_count: 0,
        }
    }

    /// Parse the AV1 sequence header.
    ///
    /// AV1 sequence headers are embedded in the bitstream and contain
    /// all the profile, level, and format information.
    pub fn parse_sequence_header(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::Av1Sps> {
        let mut sps = vk_video_core::picture::Av1Sps::new();

        if data.len() < 4 {
            return Err(ParserError::InvalidBitstream);
        }

        let mut pos = 0;

        // Read the first byte - profile is in the top 3 bits
        let first_byte = data[0];
        sps.profile = (first_byte >> 4) & 0x07;

        // Check still_picture bit
        sps.still_picture = (first_byte & 0x08) != 0;

        // Check reduced_still_picture_header
        sps.reduced_still_picture_header = (first_byte & 0x04) != 0;

        pos += 1;

        // Level is in the next 5 bits
        if pos < data.len() {
            let second_byte = data[pos];
            sps.level = (second_byte >> 3) & 0x1F;
            pos += 1;
        }

        // If reduced_still_picture_header is set, the frame width/height
        // are encoded differently
        if sps.reduced_still_picture_header {
            // Frame width - 1 (14 bits)
            if pos + 1 < data.len() {
                let fw1 = (((data[pos] & 0x1F) as u16) << 4)
                    | ((data[pos + 1] >> 4) as u16);
                sps.max_frame_width_minus_1 = fw1;
                pos += 2;
            }
            // Frame height - 1 (14 bits)
            if pos + 1 < data.len() {
                let fh1 = (((data[pos] & 0x0F) as u16) << 6)
                    | ((data[pos + 1]) as u16);
                sps.max_frame_height_minus_1 = fh1;
                pos += 2;
            }
        } else {
            // Frame width - 1 (14 bits)
            if pos + 1 < data.len() {
                let fw1 = (((data[pos] & 0x3F) as u16) << 2)
                    | ((data[pos + 1] >> 6) as u16);
                sps.max_frame_width_minus_1 = fw1;
                pos += 2;
            }
            // Frame height - 1 (14 bits)
            if pos + 1 < data.len() {
                let fh1 = (((data[pos] & 0x3F) as u16) << 2)
                    | ((data[pos + 1] >> 6) as u16);
                sps.max_frame_height_minus_1 = fh1;
                pos += 2;
            }
        }

        // Bit depth (2 bits)
        if pos + 1 < data.len() {
            let third_byte = data[pos];
            let fourth_byte = data[pos + 1];
            let bit_depth = ((third_byte >> 1) & 0x06) | ((fourth_byte >> 7) & 0x01);
            sps.frame_width_bits = 8 + ((third_byte & 0x01) << 3);
            sps.frame_height_bits = 8 + ((fourth_byte >> 4) & 0x01);
            pos += 2;
        }

        // Initial display delay (4 bits)
        if pos < data.len() {
            let fifth_byte = data[pos];
            sps.initial_display_delay_present_flag = (fifth_byte >> 4) != 0;
            // initial_display_delay is part of initial_display_delay_present_flag
            pos += 1;
        }

        // Operation point set
        if pos < data.len() {
            let op_point_index = data[pos] & 0x1F;
            let _ = op_point_index;
            pos += 1;
        }

        // Filter params
        if pos < data.len() {
            let filter_byte = data[pos];
            sps.enable_filter_intra = (filter_byte & 0x01) != 0;
            sps.enable_interintra_compound = (filter_byte & 0x02) != 0;
            sps.enable_masked_compound = (filter_byte & 0x04) != 0;
            sps.enable_warped_motion = (filter_byte & 0x08) != 0;
            sps.enable_dual_filter = (filter_byte & 0x10) != 0;
            sps.enable_order_hint = (filter_byte & 0x20) != 0;
            sps.enable_jnt_motion = (filter_byte & 0x40) != 0;
            sps.enable_second_ref_frame = (filter_byte & 0x80) != 0;
            pos += 1;
        }

        // More flags
        if pos < data.len() {
            let flags2 = data[pos];
            sps.enable_offset_unit = (flags2 & 0x01) != 0;
            sps.enable_txfm_32x32 = (flags2 & 0x02) != 0;
            sps.enable_superres = (flags2 & 0x04) != 0;
            sps.enable_cdef = (flags2 & 0x08) != 0;
            sps.enable_restoration = (flags2 & 0x10) != 0;
            sps.film_grain_params_present = (flags2 & 0x20) != 0;
            pos += 1;
        }

        // Update detected format
        self.update_format_from_sps(&sps);

        self.active_sps = Some(sps.clone());
        Ok(sps)
    }

    /// Update detected format from sequence header.
    fn update_format_from_sps(&mut self, sps: &vk_video_core::picture::Av1Sps) {
        let coded_width = sps.max_frame_width_minus_1 as u32 + 1;
        let coded_height = sps.max_frame_height_minus_1 as u32 + 1;

        self.detected_format.coded_width = coded_width;
        self.detected_format.coded_height = coded_height;

        // AV1 supports 8, 10, and 12 bit depth
        // The actual bit depth is stored in the frame header
        self.detected_format.luma_bit_depth = vk_video_core::format::ComponentBitDepth::Bit8;
        self.detected_format.chroma_bit_depth = vk_video_core::format::ComponentBitDepth::Bit8;

        // AV1 chroma subsampling - typically 4:2:0
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
        // AV1 start code is 0x9E (0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x9E)
        // or 0x80 for the first frame
        // The signature byte is 0x9E in AV1
        data[data.len() - 1] == 0x9E
            || (data.len() >= 2 && data[data.len() - 2] == 0x80)
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

        // AV1 uses a different framing - no start codes in the same way
        // Check if we have a sequence header to parse
        if self.active_sps.is_none() {
            // Try to find and parse the sequence header
            // AV1 sequence header starts with 0x02 (profile bits)
            // followed by level, frame dimensions, etc.
            if data.len() >= 4 {
                let seq_header = self.parse_sequence_header(data)?;
                return Ok(ParseResult::ParameterSet {
                    sps: Some(vk_video_core::picture::BoxedPictureParametersSet::new(seq_header)),
                    pps: None,
                    vps: None,
                });
            }
        }

        // If we have a sequence header, treat remaining data as frame data
        if self.active_sps.is_some() {
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
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}
