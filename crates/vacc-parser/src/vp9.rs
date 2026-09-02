//! VP9 bitstream parser.
//!
//! Parses VP9 bitstreams to extract frame headers, color config, loop filter,
//! quantization, segmentation, and tile info. Aligned with NVIDIA's
//! Vulkan-Video-Samples VP9 parser (VulkanVP9Decoder.cpp).
//!
//! VP9 does not use NAL units or start codes like H.264/H.265. Instead, frames
//! begin with a 2-bit frame marker (0b10), followed by the uncompressed header.
//! Superframes may contain a superframe index at the end of the data.

use crate::bitreader::BitReader;
use crate::{DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser};
use vacc_core::picture::{
    Vp9ColorConfig, Vp9ColorSpace, Vp9FrameData, Vp9FrameType, Vp9InterpolationFilter, Vp9Profile,
    VP9_FRAME_MARKER, VP9_FRAME_SYNC_CODE, VP9_LOOP_FILTER_ADJUSTMENTS, VP9_MAX_PROBABILITY,
    VP9_MAX_REF_FRAMES, VP9_MAX_SEGMENTATION_PRED_PROB, VP9_MAX_SEGMENTATION_TREE_PROBS,
    VP9_MAX_SEGMENTS, VP9_MAX_TILE_WIDTH_B64, VP9_MIN_TILE_WIDTH_B64, VP9_NUM_REF_FRAMES,
    VP9_REFS_PER_FRAME, VP9_SEG_LVL_MAX,
};

/// VP9 parser state.
pub struct Vp9Parser {
    /// Detected format.
    detected_format: DetectedVideoFormat,
    /// Frame counter.
    frame_count: u32,
    /// Last frame width for compute_image_size side effects.
    last_frame_width: u32,
    /// Last frame height for compute_image_size side effects.
    last_frame_height: u32,
    /// Last show_frame for compute_image_size side effects.
    last_show_frame: bool,
    /// Last loop filter reference deltas.
    loop_filter_ref_deltas: [i8; VP9_MAX_REF_FRAMES as usize],
    /// Last loop filter mode deltas.
    loop_filter_mode_deltas: [i8; VP9_LOOP_FILTER_ADJUSTMENTS as usize],
    /// Reference frame sizes indexed by DPB slot (for inter frame size inheritance).
    reference_frame_sz: [(u32, u32); VP9_NUM_REF_FRAMES as usize],
    /// Color config carried across frames. Per the VP9 spec the color config is
    /// only signaled on key frames and intra-only frames (profiles 1-3); a
    /// decoder must keep using the last-seen values until refreshed. Without
    /// this carry-over, inter frames of a 10/12-bit stream would report the
    /// default 8-bit color config to the backends.
    last_color_config: Vp9ColorConfig,
    /// Interpolation filter carried across frames. It is only signaled on
    /// non-intra inter frames; key/intra-only frames keep the last value
    /// (FFmpeg `h->filtermode` and the cuvid parser both carry it over).
    last_interpolation_filter: Vp9InterpolationFilter,
}

impl Default for Vp9Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp9Parser {
    pub fn new() -> Self {
        Self {
            detected_format: DetectedVideoFormat::new(vacc_core::codec::VideoCodec::DecodeVp9),
            frame_count: 0,
            last_frame_width: 0,
            last_frame_height: 0,
            last_show_frame: false,
            loop_filter_ref_deltas: [0; VP9_MAX_REF_FRAMES as usize],
            loop_filter_mode_deltas: [0; VP9_LOOP_FILTER_ADJUSTMENTS as usize],
            reference_frame_sz: [(0, 0); VP9_NUM_REF_FRAMES as usize],
            last_color_config: Vp9ColorConfig::default(),
            last_interpolation_filter: Vp9InterpolationFilter::default(),
        }
    }

    /// Parse a VP9 frame from a bitstream packet (without superframe offset).
    pub fn parse_frame(&mut self, data: &[u8]) -> ParserResult<Vp9FrameData> {
        self.parse_frame_with_offset(data, 0)
    }

    /// Parse a VP9 frame from a bitstream packet.
    ///
    /// # Arguments
    /// * `data` - The frame data
    /// * `superframe_offset` - Offset of this frame within a superframe (0 if not from superframe)
    pub fn parse_frame_with_offset(
        &mut self,
        data: &[u8],
        superframe_offset: u32,
    ) -> ParserResult<Vp9FrameData> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        let mut frame_data = Vp9FrameData::default();
        frame_data.superframe_frame_offset = superframe_offset;
        let mut r = BitReader::new(data, false);

        // Check frame marker (2 bits must be 0b10)
        let marker = r.read_bits(2)?;
        if marker != VP9_FRAME_MARKER as u32 {
            return Err(ParserError::InvalidBitstream);
        }

        // Parse profile (2 bits: low first, then high)
        let profile_low = r.read_bits(1)?;
        let profile_high = r.read_bits(1)?;
        let profile = (profile_high << 1) | profile_low;
        frame_data.picture_info.profile = match profile {
            0 => Vp9Profile::Profile0,
            1 => Vp9Profile::Profile1,
            2 => Vp9Profile::Profile2,
            3 => Vp9Profile::Profile3,
            _ => return Err(ParserError::InvalidBitstream),
        };

        // Profile 3: check zero bit
        if profile == 3 && r.read_bits(1)? != 0 {
            return Err(ParserError::InvalidBitstream);
        }

        // show_existing_frame
        frame_data.show_existing_frame = r.read_bit()?;
        if frame_data.show_existing_frame {
            frame_data.frame_to_show_map_idx = r.read_bits(3)? as u8;
            frame_data.uncompressed_header_offset = Self::consumed_bits_to_bytes(&r);
            frame_data.compressed_header_size = 0;
            return Ok(frame_data);
        }

        // frame_type
        frame_data.picture_info.frame_type = match r.read_bit()? {
            true => Vp9FrameType::Inter,
            false => Vp9FrameType::Key,
        };

        // show_frame
        frame_data.picture_info.flags.show_frame = r.read_bits(1)? as u8;

        // error_resilient_mode: read for ALL frames to match cros-codecs bitstream position
        // (for key frames, this bit is part of the data that gets byte-aligned away)
        frame_data.picture_info.flags.error_resilient_mode = r.read_bits(1)? as u8;

        if frame_data.picture_info.frame_type == Vp9FrameType::Key {
            // Key frame: read frame sync code (24 bits)
            let sync_code = r.read_bits(24)?;
            if sync_code != VP9_FRAME_SYNC_CODE {
                eprintln!(
                    "[VP9] Invalid frame sync code: 0x{:06x} (expected 0x{:06x})",
                    sync_code, VP9_FRAME_SYNC_CODE
                );
            }

            self.parse_color_config(
                &mut r,
                &mut frame_data.color_config,
                frame_data.picture_info.profile,
            )?;
            // Key frames refresh the carried color config.
            self.last_color_config = frame_data.color_config;

            self.parse_frame_and_render_size(&mut r, &mut frame_data)?;

            // Key frames implicitly refresh all frame buffers
            frame_data.picture_info.refresh_frame_flags = 0xFF;

            frame_data.frame_is_intra = true;
            for i in 0..VP9_REFS_PER_FRAME as usize {
                frame_data.ref_frame_idx[i] = 0;
            }
        } else {
            // Non-key frame: error_resilient_mode already read above

            // intra_only: only present when show_frame == 0 (per VP9 spec)
            if frame_data.picture_info.flags.show_frame == 0 {
                frame_data.picture_info.flags.intra_only = r.read_bits(1)? as u8;
            }
            // reset_frame_context: only present when !error_resilient_mode
            if frame_data.picture_info.flags.error_resilient_mode == 0 {
                frame_data.picture_info.flags.reset_frame_context = r.read_bits(2)? as u8;
            }

            frame_data.frame_is_intra = frame_data.picture_info.flags.intra_only != 0;

            if frame_data.frame_is_intra {
                // Intra-only non-key frame: read sync code
                let sync_code = r.read_bits(24)?;
                if sync_code != VP9_FRAME_SYNC_CODE {
                    eprintln!(
                        "[VP9] Invalid intra frame sync code: 0x{:06x} (expected 0x{:06x})",
                        sync_code, VP9_FRAME_SYNC_CODE
                    );
                }

                if (frame_data.picture_info.profile as u32) > (Vp9Profile::Profile0 as u32) {
                    self.parse_color_config(
                        &mut r,
                        &mut frame_data.color_config,
                        frame_data.picture_info.profile,
                    )?;
                } else {
                    frame_data.color_config.color_space = Vp9ColorSpace::Bt601;
                    frame_data.color_config.subsampling_x = 1;
                    frame_data.color_config.subsampling_y = 1;
                    frame_data.color_config.bit_depth = 8;
                }
                // Intra-only frames refresh the carried color config.
                self.last_color_config = frame_data.color_config;

                // refresh_frame_flags comes AFTER color_config but BEFORE frame_size
                // for intra-only frames (per VP9 spec section 7.2.4.1 and cros-codecs)
                frame_data.picture_info.refresh_frame_flags = r.read_bits(8)? as u8;

                self.parse_frame_and_render_size(&mut r, &mut frame_data)?;
            } else {
                // Inter frames carry over the last-seen color config (bit depth,
                // subsampling, color space) — it is not re-signaled here.
                frame_data.color_config = self.last_color_config;

                // Inter frame: refresh_frame_flags comes before ref_frame_idx
                frame_data.picture_info.refresh_frame_flags = r.read_bits(8)? as u8;

                frame_data.picture_info.ref_frame_sign_bias_mask = 0;
                // VP9 spec: each frame specifies exactly 3 reference frames
                for i in 0..3usize {
                    frame_data.ref_frame_idx[i] = r.read_bits(3)? as u8;
                    let sign_bias = r.read_bits(1)?;
                    frame_data.picture_info.ref_frame_sign_bias_mask |=
                        (sign_bias as u8) << (i + 1);
                }

                self.parse_frame_and_render_size_with_refs(&mut r, &mut frame_data)?;

                frame_data.picture_info.flags.allow_high_precision_mv = r.read_bits(1)? as u8;

                let is_filter_switchable = r.read_bits(1)?;
                if is_filter_switchable != 0 {
                    frame_data.picture_info.interpolation_filter =
                        Vp9InterpolationFilter::Switchable;
                } else {
                    let filter_literal = r.read_bits(2)?;
                    // Mapping per C++ reference: 0->SMOOTH, 1->EIGHTTAP, 2->SHARP, 3->BILINEAR
                    frame_data.picture_info.interpolation_filter = match filter_literal {
                        0 => Vp9InterpolationFilter::EightTapSmooth,
                        1 => Vp9InterpolationFilter::EightTap,
                        2 => Vp9InterpolationFilter::EightTapSharp,
                        3 => Vp9InterpolationFilter::Bilinear,
                        _ => Vp9InterpolationFilter::EightTapSmooth,
                    };
                }
                self.last_interpolation_filter = frame_data.picture_info.interpolation_filter;
            }
        }

        // Key/intra-only frames do not signal the interpolation filter; carry
        // over the last value (FFmpeg `h->filtermode` / cuvid convention).
        if frame_data.frame_is_intra {
            frame_data.picture_info.interpolation_filter = self.last_interpolation_filter;
        }

        // refresh_frame_context and frame_parallel_decoding_mode
        if frame_data.picture_info.flags.error_resilient_mode == 0 {
            frame_data.picture_info.flags.refresh_frame_context = r.read_bits(1)? as u8;
            frame_data.picture_info.flags.frame_parallel_decoding_mode = r.read_bits(1)? as u8;
        } else {
            frame_data.picture_info.flags.refresh_frame_context = 0;
            frame_data.picture_info.flags.frame_parallel_decoding_mode = 1;
        }

        // frame_context_idx
        frame_data.picture_info.frame_context_idx = r.read_bits(2)? as u8;

        // Reset frame context for intra or error resilient mode
        if frame_data.frame_is_intra || frame_data.picture_info.flags.error_resilient_mode != 0 {
            frame_data.segmentation.feature_enabled.fill(0);
            frame_data
                .segmentation
                .feature_data
                .iter_mut()
                .for_each(|f| f.fill(0));
            frame_data.picture_info.frame_context_idx = 0;
        }

        // Parse loop filter parameters
        self.parse_loop_filter_params(&mut r, &mut frame_data)?;

        // Parse quantization parameters
        self.parse_quantization_params(&mut r, &mut frame_data)?;

        // Lossless frame: all quantization values zero (FFmpeg convention).
        frame_data.picture_info.lossless = frame_data.picture_info.base_q_idx == 0
            && frame_data.picture_info.delta_q_y_dc == 0
            && frame_data.picture_info.delta_q_uv_dc == 0
            && frame_data.picture_info.delta_q_uv_ac == 0;

        // Parse segmentation parameters
        self.parse_segmentation_params(&mut r, &mut frame_data)?;

        // Parse tile info
        self.parse_tile_info(&mut r, &mut frame_data)?;

        // compressed_header_size
        frame_data.compressed_header_size = r.read_bits(16)?;

        // Compute offsets
        frame_data.uncompressed_header_offset = 0;
        frame_data.compressed_header_offset = Self::consumed_bits_to_bytes(&r);
        frame_data.tiles_offset =
            frame_data.compressed_header_offset + frame_data.compressed_header_size;

        // Update reference frame sizes for refreshed frames (per cros-codecs)
        for i in 0..VP9_NUM_REF_FRAMES as usize {
            let flag = 1u8 << i;
            if frame_data.picture_info.refresh_frame_flags & flag != 0 {
                self.reference_frame_sz[i] = (frame_data.frame_width, frame_data.frame_height);
            }
        }

        // Update detected format
        self.update_format(&frame_data);

        self.frame_count += 1;

        Ok(frame_data)
    }

    /// Parse color configuration.
    fn parse_color_config(
        &mut self,
        r: &mut BitReader,
        color_config: &mut Vp9ColorConfig,
        profile: Vp9Profile,
    ) -> ParserResult<()> {
        if (profile as u32) >= (Vp9Profile::Profile2 as u32) {
            color_config.bit_depth = if r.read_bits(1)? != 0 { 12 } else { 10 };
        } else {
            color_config.bit_depth = 8;
        }

        color_config.color_space = match r.read_bits(3)? {
            0 => Vp9ColorSpace::Unknown,
            1 => Vp9ColorSpace::Bt601,
            2 => Vp9ColorSpace::Bt709,
            3 => Vp9ColorSpace::Smpte170,
            4 => Vp9ColorSpace::Smpte240,
            5 => Vp9ColorSpace::Bt2020,
            6 => Vp9ColorSpace::Reserved,
            7 => Vp9ColorSpace::Rgb,
            _ => Vp9ColorSpace::Unknown,
        };

        let is_high_profile = (profile as u32) == (Vp9Profile::Profile1 as u32)
            || (profile as u32) == (Vp9Profile::Profile3 as u32);

        if color_config.color_space != Vp9ColorSpace::Rgb {
            color_config.flags.color_range = r.read_bits(1)? as u8;
            if is_high_profile {
                color_config.subsampling_x = r.read_bits(1)? as u8;
                color_config.subsampling_y = r.read_bits(1)? as u8;
                if r.read_bits(1)? != 0 {
                    return Err(ParserError::InvalidBitstream);
                }
            } else {
                color_config.subsampling_x = 1;
                color_config.subsampling_y = 1;
            }
        } else {
            color_config.flags.color_range = 1;
            if is_high_profile {
                color_config.subsampling_x = 0;
                color_config.subsampling_y = 0;
                if r.read_bits(1)? != 0 {
                    return Err(ParserError::InvalidBitstream);
                }
            }
        }

        Ok(())
    }

    /// Parse frame and render size (key frame or intra-only).
    fn parse_frame_and_render_size(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        frame_data.frame_width = r.read_bits(16)? + 1;
        frame_data.frame_height = r.read_bits(16)? + 1;

        self.compute_image_size(frame_data);

        if r.read_bits(1)? != 0 {
            frame_data.render_width = r.read_bits(16)? + 1;
            frame_data.render_height = r.read_bits(16)? + 1;
        } else {
            frame_data.render_width = frame_data.frame_width;
            frame_data.render_height = frame_data.frame_height;
        }

        Ok(())
    }

    /// Parse frame and render size with references (inter frame).
    fn parse_frame_and_render_size_with_refs(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        // Per VP9 spec and cros-codecs: read frame_size_coding_flag for each of 3 refs.
        // When the first flag is 1, use that ref's size and stop reading flags.
        let mut found_ref = false;

        for i in 0..3usize {
            let frame_size_coding_flag = r.read_bits(1)? != 0;

            if frame_size_coding_flag {
                let idx = frame_data.ref_frame_idx[i] as usize;
                frame_data.frame_width = self.reference_frame_sz[idx].0;
                frame_data.frame_height = self.reference_frame_sz[idx].1;
                found_ref = true;
                break;
            }
        }

        if !found_ref {
            // No reference frame size available, read from bitstream
            frame_data.frame_width = r.read_bits(16)? + 1;
            frame_data.frame_height = r.read_bits(16)? + 1;

            self.compute_image_size(frame_data);
        } else {
            self.compute_image_size(frame_data);
        }

        // Per cros-codecs: always read render_size_flag for inter frames
        if r.read_bits(1)? != 0 {
            frame_data.render_width = r.read_bits(16)? + 1;
            frame_data.render_height = r.read_bits(16)? + 1;
        } else {
            frame_data.render_width = frame_data.frame_width;
            frame_data.render_height = frame_data.frame_height;
        }

        Ok(())
    }

    /// Compute image size and side effects per VP9 spec section 7.2.6.
    fn compute_image_size(&mut self, frame_data: &mut Vp9FrameData) {
        frame_data.mi_cols = (frame_data.frame_width + 7) >> 3;
        frame_data.mi_rows = (frame_data.frame_height + 7) >> 3;
        frame_data.sb64_cols = (frame_data.mi_cols + 7) >> 3;
        frame_data.sb64_rows = (frame_data.mi_rows + 7) >> 3;

        if self.last_frame_height != frame_data.frame_height
            || self.last_frame_width != frame_data.frame_width
        {
            frame_data.picture_info.flags.use_prev_frame_mvs = 0;
        } else {
            let intra_only = frame_data.picture_info.frame_type == Vp9FrameType::Key
                || frame_data.picture_info.flags.intra_only != 0;
            frame_data.picture_info.flags.use_prev_frame_mvs = if self.last_show_frame
                && frame_data.picture_info.flags.error_resilient_mode == 0
                && !intra_only
            {
                1
            } else {
                0
            };
        }

        self.last_frame_height = frame_data.frame_height;
        self.last_frame_width = frame_data.frame_width;
        self.last_show_frame = frame_data.picture_info.flags.show_frame != 0;
    }

    /// Parse loop filter parameters.
    fn parse_loop_filter_params(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        let loop_filter = &mut frame_data.loop_filter;

        if frame_data.frame_is_intra || frame_data.picture_info.flags.error_resilient_mode != 0 {
            self.loop_filter_ref_deltas.fill(0);
            self.loop_filter_mode_deltas.fill(0);
            self.loop_filter_ref_deltas[0] = 1;
            self.loop_filter_ref_deltas[1] = 0;
            self.loop_filter_ref_deltas[2] = -1;
            self.loop_filter_ref_deltas[3] = -1;
        }

        loop_filter.loop_filter_level = r.read_bits(6)? as u8;
        loop_filter.loop_filter_sharpness = r.read_bits(3)? as u8;

        loop_filter.flags.loop_filter_delta_enabled = r.read_bits(1)? as u8;

        if loop_filter.flags.loop_filter_delta_enabled != 0 {
            loop_filter.flags.loop_filter_delta_update = r.read_bits(1)? as u8;

            if loop_filter.flags.loop_filter_delta_update != 0 {
                loop_filter.flags.update_ref_delta = 0;
                for i in 0..VP9_MAX_REF_FRAMES as usize {
                    let update_ref_delta = r.read_bits(1)?;
                    loop_filter.flags.update_ref_delta |= (update_ref_delta as u8) << i;
                    if update_ref_delta != 0 {
                        self.loop_filter_ref_deltas[i] = r.read_bits(6)? as i8;
                        if r.read_bits(1)? != 0 {
                            self.loop_filter_ref_deltas[i] = -self.loop_filter_ref_deltas[i];
                        }
                    }
                }

                loop_filter.flags.update_mode_delta = 0;
                for i in 0..VP9_LOOP_FILTER_ADJUSTMENTS as usize {
                    let update_mode_delta = r.read_bits(1)?;
                    loop_filter.flags.update_mode_delta |= (update_mode_delta as u8) << i;
                    if update_mode_delta != 0 {
                        self.loop_filter_mode_deltas[i] = r.read_bits(6)? as i8;
                        if r.read_bits(1)? != 0 {
                            self.loop_filter_mode_deltas[i] = -self.loop_filter_mode_deltas[i];
                        }
                    }
                }
            }
        }

        loop_filter
            .loop_filter_ref_deltas
            .copy_from_slice(&self.loop_filter_ref_deltas);
        loop_filter
            .loop_filter_mode_deltas
            .copy_from_slice(&self.loop_filter_mode_deltas);

        Ok(())
    }

    /// Parse quantization parameters.
    fn parse_quantization_params(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        frame_data.picture_info.base_q_idx = r.read_bits(8)? as u8;
        frame_data.picture_info.delta_q_y_dc = Self::read_delta_q(r)?;
        frame_data.picture_info.delta_q_uv_dc = Self::read_delta_q(r)?;
        frame_data.picture_info.delta_q_uv_ac = Self::read_delta_q(r)?;
        Ok(())
    }

    /// Read a delta Q value.
    fn read_delta_q(r: &mut BitReader) -> ParserResult<i8> {
        if r.read_bits(1)? != 0 {
            let mut delta = r.read_bits(4)? as i8;
            if r.read_bits(1)? != 0 {
                delta = -delta;
            }
            Ok(delta)
        } else {
            Ok(0)
        }
    }

    /// Parse segmentation parameters.
    fn parse_segmentation_params(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        let segmentation = &mut frame_data.segmentation;

        frame_data.picture_info.flags.segmentation_enabled = r.read_bits(1)? as u8;
        if frame_data.picture_info.flags.segmentation_enabled == 0 {
            return Ok(());
        }

        segmentation.flags.segmentation_update_map = r.read_bits(1)? as u8;

        if segmentation.flags.segmentation_update_map != 0 {
            for i in 0..VP9_MAX_SEGMENTATION_TREE_PROBS as usize {
                let prob_coded = r.read_bits(1)?;
                segmentation.segmentation_tree_probs[i] = if prob_coded != 0 {
                    r.read_bits(8)? as u8
                } else {
                    VP9_MAX_PROBABILITY
                };
            }

            segmentation.flags.segmentation_temporal_update = r.read_bits(1)? as u8;
            for i in 0..VP9_MAX_SEGMENTATION_PRED_PROB as usize {
                if segmentation.flags.segmentation_temporal_update != 0 {
                    let prob_coded = r.read_bits(1)?;
                    segmentation.segmentation_pred_prob[i] = if prob_coded != 0 {
                        r.read_bits(8)? as u8
                    } else {
                        VP9_MAX_PROBABILITY
                    };
                } else {
                    segmentation.segmentation_pred_prob[i] = VP9_MAX_PROBABILITY;
                }
            }
        }

        // segmentation_update_data is read unconditionally when segmentation is enabled
        segmentation.flags.segmentation_update_data = r.read_bits(1)? as u8;
        if segmentation.flags.segmentation_update_data != 0 {
            segmentation.flags.segmentation_abs_or_delta_update = r.read_bits(1)? as u8;

            segmentation.feature_enabled.fill(0);
            segmentation.feature_data.iter_mut().for_each(|f| f.fill(0));

            // VP9 segmentation feature bits and signedness per spec:
            // Feature 0 (ALT_Q): 8 bits magnitude + sign bit
            // Feature 1 (ALT_Y_AC): 6 bits magnitude + sign bit
            // Feature 2 (ALT_Y_DC): 2 bits (unsigned, used as offset)
            // Feature 3 (ALT_UV_AC): 0 bits (always 0)
            let feature_bits: [u8; VP9_SEG_LVL_MAX as usize] = [8, 6, 2, 0];
            let feature_signed: [bool; VP9_SEG_LVL_MAX as usize] = [true, true, false, false];

            for i in 0..VP9_MAX_SEGMENTS as usize {
                for j in 0..VP9_SEG_LVL_MAX as usize {
                    let feature_enabled = r.read_bits(1)?;
                    segmentation.feature_enabled[i] |= (feature_enabled as u8) << j;

                    if feature_enabled != 0 && feature_bits[j] > 0 {
                        let mut feature_value = r.read_bits(feature_bits[j])? as i8;
                        if feature_signed[j] {
                            let feature_sign = r.read_bits(1)?;
                            if feature_sign != 0 {
                                feature_value = -feature_value;
                            }
                        }
                        segmentation.feature_data[i][j] = feature_value;
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse tile information.
    fn parse_tile_info(
        &mut self,
        r: &mut BitReader,
        frame_data: &mut Vp9FrameData,
    ) -> ParserResult<()> {
        let sb64_cols = frame_data.sb64_cols;

        let mut min_log2: u8 = 0;
        while ((VP9_MAX_TILE_WIDTH_B64 as u32) << min_log2) < sb64_cols {
            min_log2 += 1;
        }

        let mut max_log2: u8 = 1;
        while (sb64_cols >> max_log2) >= (VP9_MIN_TILE_WIDTH_B64 as u32) {
            max_log2 += 1;
        }
        max_log2 -= 1;

        frame_data.picture_info.tile_cols_log2 = min_log2;
        while frame_data.picture_info.tile_cols_log2 < max_log2 {
            if r.read_bits(1)? != 0 {
                frame_data.picture_info.tile_cols_log2 += 1;
            } else {
                break;
            }
        }

        frame_data.picture_info.tile_rows_log2 = r.read_bits(1)? as u8;
        if frame_data.picture_info.tile_rows_log2 != 0 {
            frame_data.picture_info.tile_rows_log2 += r.read_bits(1)? as u8;
        }

        frame_data.num_tiles = (1u32 << frame_data.picture_info.tile_rows_log2)
            * (1u32 << frame_data.picture_info.tile_cols_log2);

        Ok(())
    }

    /// Parse superframe index from data.
    pub fn parse_superframe_index(data: &[u8]) -> (u32, Vec<u32>) {
        if data.len() < 2 {
            return (0, Vec::new());
        }

        let final_byte = data[data.len() - 1];

        if (final_byte & 0xE0) != 0xC0 {
            return (0, Vec::new());
        }

        let frames = (final_byte & 0x07) as u32 + 1;
        let mag = ((final_byte >> 3) & 0x03) as u32 + 1;
        let index_sz = 2 + mag * frames;

        if data.len() < index_sz as usize {
            return (0, Vec::new());
        }

        let start = data.len() - index_sz as usize;
        if data[start] != final_byte {
            return (0, Vec::new());
        }

        let mut sizes = Vec::with_capacity(frames as usize);
        let mut x = start + 1;
        for _ in 0..frames {
            let mut this_sz: u32 = 0;
            for j in 0..mag as usize {
                this_sz |= (data[x + j] as u32) << (j * 8);
            }
            sizes.push(this_sz);
            x += mag as usize;
        }

        (frames, sizes)
    }

    /// Skip a superframe index from the beginning of data.
    pub fn skip_superframe_index(data: &[u8]) -> &[u8] {
        if data.is_empty() {
            return data;
        }

        if (data[0] & 0xE0) == 0xC0 {
            let marker = data[0];
            let frames = (marker & 0x07) as usize + 1;
            let mag = ((marker >> 3) & 0x03) as usize + 1;
            let index_sz = 2 + mag * frames;

            if data.len() >= index_sz && data[index_sz - 1] == marker {
                return &data[index_sz..];
            }
        }

        data
    }

    /// Convert consumed bit position to byte offset (round up to next byte).
    /// Matches C++: (consumed_bits() + 7) >> 3
    fn consumed_bits_to_bytes(r: &BitReader) -> u32 {
        r.bytes_consumed() as u32
    }

    /// Update detected format from parsed frame data.
    fn update_format(&mut self, frame_data: &Vp9FrameData) {
        self.detected_format.coded_width = frame_data.frame_width;
        self.detected_format.coded_height = frame_data.frame_height;

        self.detected_format.luma_bit_depth = match frame_data.color_config.bit_depth {
            8 => vacc_core::format::ComponentBitDepth::Bit8,
            10 => vacc_core::format::ComponentBitDepth::Bit10,
            12 => vacc_core::format::ComponentBitDepth::Bit12,
            _ => vacc_core::format::ComponentBitDepth::Bit8,
        };

        self.detected_format.chroma_bit_depth = self.detected_format.luma_bit_depth;

        self.detected_format.chroma_subsampling = if frame_data.color_config.subsampling_x == 1
            && frame_data.color_config.subsampling_y == 1
        {
            vacc_core::format::ChromaSubsampling::_420
        } else {
            vacc_core::format::ChromaSubsampling::_444
        };

        self.detected_format.codec_profile = frame_data.picture_info.profile as u32;
        self.detected_format.progressive_sequence = true;
    }
}

impl VideoParser for Vp9Parser {
    fn init(&mut self, format: &DetectedVideoFormat) -> ParserResult<()> {
        if format.codec != vacc_core::codec::VideoCodec::DecodeVp9 {
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
        if data.is_empty() {
            return Ok(ParseResult::Nothing);
        }

        let data = Self::skip_superframe_index(data);

        match self.parse_frame(data) {
            Ok(frame_data) => Ok(ParseResult::Slice {
                slices: vec![crate::SliceEntry {
                    slice_header: None,
                    nal_data: Vec::new(),
                }],
                bytes_consumed: data
                    .len()
                    .saturating_sub(frame_data.compressed_header_offset as usize),
            }),
            Err(e) => Err(e),
        }
    }

    fn reset(&mut self) {
        self.frame_count = 0;
        self.last_frame_width = 0;
        self.last_frame_height = 0;
        self.last_show_frame = false;
        self.loop_filter_ref_deltas.fill(0);
        self.loop_filter_mode_deltas.fill(0);
        self.reference_frame_sz.fill((0, 0));
        self.last_color_config = Vp9ColorConfig::default();
        self.last_interpolation_filter = Vp9InterpolationFilter::default();
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_marker() {
        let data = [0b10000000u8, 0x00];
        let mut r = BitReader::new(&data, false);
        let marker = r.read_bits(2).unwrap();
        assert_eq!(marker, VP9_FRAME_MARKER as u32);
    }

    #[test]
    fn test_profile_parsing() {
        // Profile 0
        let data = [0b10000000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_bits(2).unwrap(), VP9_FRAME_MARKER as u32);
        assert_eq!(r.read_bits(2).unwrap(), 0);

        // Profile 1
        let data = [0b10010000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_bits(2).unwrap(), VP9_FRAME_MARKER as u32);
        assert_eq!(r.read_bits(2).unwrap(), 1);

        // Profile 2
        let data = [0b10100000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_bits(2).unwrap(), VP9_FRAME_MARKER as u32);
        assert_eq!(r.read_bits(2).unwrap(), 2);

        // Profile 3: marker=10, profile=11, then 0
        let data = [0b10110000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_bits(2).unwrap(), VP9_FRAME_MARKER as u32);
        assert_eq!(r.read_bits(2).unwrap(), 3);
        assert_eq!(r.read_bits(1).unwrap(), 0);
    }

    #[test]
    fn test_show_existing_frame() {
        // marker=10, profile=00, show_existing=1, idx=000
        let data = [0b10001000u8, 0x00];
        let mut parser = Vp9Parser::new();
        let frame_data = parser.parse_frame(&data).unwrap();
        assert!(frame_data.show_existing_frame);
        assert_eq!(frame_data.frame_to_show_map_idx, 0);
        assert_eq!(frame_data.compressed_header_size, 0);
    }

    #[test]
    fn test_delta_q_parsing() {
        // flag=0 -> delta=0
        let data = [0b00000000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(Vp9Parser::read_delta_q(&mut r).unwrap(), 0);

        // flag=1, value=5, sign=0 -> delta=5
        let data = [0b10101000u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(Vp9Parser::read_delta_q(&mut r).unwrap(), 5);

        // flag=1, value=3, sign=1 -> delta=-3
        let data = [0b10011100u8];
        let mut r = BitReader::new(&data, false);
        assert_eq!(Vp9Parser::read_delta_q(&mut r).unwrap(), -3);
    }

    #[test]
    fn test_superframe_index_parsing() {
        // Superframe index: 2 frames, 2-byte magnitude
        // Marker: frames=2 (0b001+1=2), mag=2 (0b01+1=2)
        // Marker byte: 0b11001001 = 0xC9
        let data = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC9, 0x00, 0x03, 0x00, 0x0A, 0xC9,
        ];

        let (count, sizes) = Vp9Parser::parse_superframe_index(&data);
        assert_eq!(count, 2);
        assert_eq!(sizes[0], 0x0300);
        assert_eq!(sizes[1], 0x0A00);
    }

    #[test]
    fn test_superframe_index_not_present() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let (count, sizes) = Vp9Parser::parse_superframe_index(&data);
        assert_eq!(count, 0);
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_skip_superframe_index() {
        let data = [
            0xC1, // Marker: frames=2, mag=1
            0x10, 0x20, 0xC1, 0x10, 0x10,
        ];

        let remaining = Vp9Parser::skip_superframe_index(&data);
        assert_eq!(remaining, &[0x10, 0x10]);
    }

    #[test]
    fn test_compute_image_size() {
        let mut parser = Vp9Parser::new();
        let mut frame_data = Vp9FrameData::default();

        frame_data.frame_width = 1920;
        frame_data.frame_height = 1080;
        frame_data.picture_info.flags.error_resilient_mode = 1;

        parser.compute_image_size(&mut frame_data);

        assert_eq!(frame_data.mi_cols, 240);
        assert_eq!(frame_data.mi_rows, 135);
        assert_eq!(frame_data.sb64_cols, 30);
        assert_eq!(frame_data.sb64_rows, 17);
    }

    #[test]
    fn test_tile_calculation() {
        let mut frame_data = Vp9FrameData::default();
        frame_data.sb64_cols = 31;

        let mut min_log2: u8 = 0;
        while ((VP9_MAX_TILE_WIDTH_B64 as u32) << min_log2) < frame_data.sb64_cols {
            min_log2 += 1;
        }
        assert_eq!(min_log2, 0);

        let mut max_log2: u8 = 1;
        while (frame_data.sb64_cols >> max_log2) >= (VP9_MIN_TILE_WIDTH_B64 as u32) {
            max_log2 += 1;
        }
        max_log2 -= 1;
        assert_eq!(max_log2, 2);
    }

    #[test]
    fn test_loop_filter_initialization() {
        let mut parser = Vp9Parser::new();

        parser.loop_filter_ref_deltas.fill(0);
        parser.loop_filter_mode_deltas.fill(0);
        parser.loop_filter_ref_deltas[0] = 1;
        parser.loop_filter_ref_deltas[1] = 0;
        parser.loop_filter_ref_deltas[2] = -1;
        parser.loop_filter_ref_deltas[3] = -1;

        assert_eq!(parser.loop_filter_ref_deltas, [1, 0, -1, -1]);
    }

    #[test]
    fn test_parser_init() {
        let mut parser = Vp9Parser::new();
        let format = DetectedVideoFormat::new(vacc_core::codec::VideoCodec::DecodeVp9);
        parser.init(&format).unwrap();
        assert_eq!(
            parser.detected_format().codec,
            vacc_core::codec::VideoCodec::DecodeVp9
        );
    }

    #[test]
    fn test_parser_init_wrong_codec() {
        let mut parser = Vp9Parser::new();
        let format = DetectedVideoFormat::new(vacc_core::codec::VideoCodec::DecodeH264);
        assert!(parser.init(&format).is_err());
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = Vp9Parser::new();
        parser.frame_count = 42;
        parser.last_frame_width = 1920;
        parser.last_frame_height = 1080;
        parser.last_show_frame = true;
        parser.loop_filter_ref_deltas[0] = 5;

        parser.reset();

        assert_eq!(parser.frame_count, 0);
        assert_eq!(parser.last_frame_width, 0);
        assert_eq!(parser.last_frame_height, 0);
        assert!(!parser.last_show_frame);
        assert_eq!(parser.loop_filter_ref_deltas[0], 0);
    }

    #[test]
    fn test_interpolation_filter_values() {
        assert_eq!(Vp9InterpolationFilter::EightTap as u32, 0);
        assert_eq!(Vp9InterpolationFilter::EightTapSmooth as u32, 1);
        assert_eq!(Vp9InterpolationFilter::EightTapSharp as u32, 2);
        assert_eq!(Vp9InterpolationFilter::Bilinear as u32, 3);
        assert_eq!(Vp9InterpolationFilter::Switchable as u32, 4);
    }

    #[test]
    fn test_constants() {
        assert_eq!(VP9_FRAME_MARKER, 0b10);
        assert_eq!(VP9_FRAME_SYNC_CODE, 0x498342);
        assert_eq!(VP9_NUM_REF_FRAMES, 8);
        assert_eq!(VP9_REFS_PER_FRAME, 7);
        assert_eq!(VP9_MAX_REF_FRAMES, 4);
        assert_eq!(VP9_LOOP_FILTER_ADJUSTMENTS, 2);
        assert_eq!(VP9_MAX_SEGMENTS, 8);
        assert_eq!(VP9_SEG_LVL_MAX, 4);
        assert_eq!(VP9_MAX_SEGMENTATION_TREE_PROBS, 7);
        assert_eq!(VP9_MAX_SEGMENTATION_PRED_PROB, 3);
    }

    // ------------------------------------------------------------------
    // Synthetic bitstream helpers (MSB-first, matching BitReader).
    // ------------------------------------------------------------------

    struct BitWriter {
        bytes: Vec<u8>,
        bitpos: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bitpos: 0,
            }
        }

        fn bits(&mut self, val: u32, n: u32) {
            for i in (0..n).rev() {
                if (val >> i) & 1 != 0 {
                    let byte_idx = (self.bitpos / 8) as usize;
                    while self.bytes.len() <= byte_idx {
                        self.bytes.push(0);
                    }
                    self.bytes[byte_idx] |= 1 << (7 - (self.bitpos % 8));
                }
                self.bitpos += 1;
            }
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// Profile-0 keyframe, all optional fields absent/zero.
    fn keyframe_bytes(w: u32, h: u32, fps: u16) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.bits(0b10, 2); // marker
        bw.bits(0, 2); // profile 0
        bw.bits(0, 1); // show_existing_frame
        bw.bits(0, 1); // frame_type: key
        bw.bits(1, 1); // show_frame
        bw.bits(0, 1); // error_resilient_mode
        bw.bits(0x498342, 24); // sync code
        bw.bits(1, 3); // color_space: Bt601
        bw.bits(0, 1); // color_range
        bw.bits(w - 1, 16);
        bw.bits(h - 1, 16);
        bw.bits(0, 1); // display size flag
        bw.bits(0, 1); // refresh_frame_context
        bw.bits(0, 1); // frame_parallel_decoding_mode
        bw.bits(0, 2); // frame_context_idx
        bw.bits(0, 6); // filter_level
        bw.bits(0, 3); // sharpness_level
        bw.bits(0, 1); // lf_delta_enabled
        bw.bits(46, 8); // y_ac_qi
        bw.bits(0, 1); // y_dc_delta flag
        bw.bits(0, 1); // uv_dc_delta flag
        bw.bits(0, 1); // uv_ac_delta flag
        bw.bits(0, 1); // segmentation_enabled
        bw.bits(0, 1); // log2_tile_rows (cols: 0 bits for sb_cols <= 7)
        bw.bits(fps as u32, 16); // first_partition_size
        bw.finish()
    }

    /// Profile-0 inter frame. `size_flags` bit i = frame_size_coding_flag[i]
    /// (bit 0 = last, 1 = golden, 2 = alt); the chain stops at the first set
    /// flag. `explicit_w/h` are only emitted when no flag is set.
    fn inter_bytes(
        refidx: [u32; 3],
        size_flags: u8,
        explicit_w: u32,
        explicit_h: u32,
        refresh_mask: u32,
        fps: u16,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.bits(0b10, 2); // marker
        bw.bits(0, 2); // profile 0
        bw.bits(0, 1); // show_existing_frame
        bw.bits(1, 1); // frame_type: inter
        bw.bits(1, 1); // show_frame (intra_only not present when shown)
        bw.bits(0, 1); // error_resilient_mode
        bw.bits(0, 2); // reset_frame_context
        bw.bits(refresh_mask, 8);
        for &ri in refidx.iter() {
            bw.bits(ri, 3);
            bw.bits(0, 1); // sign bias
        }
        let mut inherited = false;
        for i in 0..3u32 {
            let flag = (size_flags >> i) & 1;
            bw.bits(flag as u32, 1);
            if flag != 0 {
                inherited = true;
                break;
            }
        }
        if !inherited {
            bw.bits(explicit_w - 1, 16);
            bw.bits(explicit_h - 1, 16);
        }
        bw.bits(0, 1); // display size flag
        bw.bits(0, 1); // allow_high_precision_mv
        bw.bits(1, 1); // switchable filter (skips mcomp_filter_type)
        bw.bits(0, 1); // refresh_frame_context
        bw.bits(0, 1); // frame_parallel_decoding_mode
        bw.bits(0, 2); // frame_context_idx
        bw.bits(0, 6); // filter_level
        bw.bits(0, 3); // sharpness_level
        bw.bits(0, 1); // lf_delta_enabled
        bw.bits(46, 8); // y_ac_qi
        bw.bits(0, 1); // y_dc_delta flag
        bw.bits(0, 1); // uv_dc_delta flag
        bw.bits(0, 1); // uv_ac_delta flag
        bw.bits(0, 1); // segmentation_enabled
        bw.bits(0, 1); // log2_tile_rows (cols: 0 bits for sb_cols <= 7)
        bw.bits(fps as u32, 16);
        bw.finish()
    }

    /// Inter frames inherit their size from a reference slot. Set up distinct
    /// per-slot sizes, then probe each size-inheritance branch.
    #[test]
    fn test_inter_size_inheritance() {
        // Case (a): flag0 set -> inherit from refidx[0]'s slot.
        let mut parser = Vp9Parser::new();
        parser.parse_frame(&keyframe_bytes(64, 64, 100)).unwrap(); // all slots 64x64
        parser
            .parse_frame(&inter_bytes([0, 0, 0], 0b000, 32, 32, 0b010, 100))
            .unwrap(); // slot 1 := 32x32
        let f = parser
            .parse_frame(&inter_bytes([1, 2, 0], 0b001, 0, 0, 0b000, 100))
            .unwrap();
        assert_eq!((f.frame_width, f.frame_height), (32, 32));
        assert_eq!(f.compressed_header_size, 100);
        assert_eq!(f.compressed_header_offset, 10);

        // Case (b): flag0=0, flag1 set -> inherit from refidx[1]'s slot, and
        // the flag chain must stop after 2 bits.
        let mut parser = Vp9Parser::new();
        parser.parse_frame(&keyframe_bytes(64, 64, 100)).unwrap();
        parser
            .parse_frame(&inter_bytes([0, 0, 0], 0b000, 32, 32, 0b010, 100))
            .unwrap(); // slot 1 := 32x32
        parser
            .parse_frame(&inter_bytes([0, 0, 0], 0b000, 16, 16, 0b100, 100))
            .unwrap(); // slot 2 := 16x16
        let f = parser
            .parse_frame(&inter_bytes([0, 2, 1], 0b010, 0, 0, 0b000, 100))
            .unwrap();
        assert_eq!((f.frame_width, f.frame_height), (16, 16));
        assert_eq!(f.compressed_header_size, 100);
        assert_eq!(f.compressed_header_offset, 10);

        // Case (c): flags 0,0,1 -> inherit from refidx[2]'s slot.
        let mut parser = Vp9Parser::new();
        parser.parse_frame(&keyframe_bytes(64, 64, 100)).unwrap();
        parser
            .parse_frame(&inter_bytes([0, 0, 0], 0b000, 32, 32, 0b010, 100))
            .unwrap();
        let f = parser
            .parse_frame(&inter_bytes([2, 1, 0], 0b100, 0, 0, 0b000, 100))
            .unwrap();
        assert_eq!((f.frame_width, f.frame_height), (64, 64)); // slot 0 unchanged
        assert_eq!(f.compressed_header_size, 100);
        assert_eq!(f.compressed_header_offset, 10);

        // Case (d): no flags -> explicit size from bitstream.
        let mut parser = Vp9Parser::new();
        parser.parse_frame(&keyframe_bytes(64, 64, 100)).unwrap();
        let f = parser
            .parse_frame(&inter_bytes([0, 1, 2], 0b000, 96, 96, 0b000, 100))
            .unwrap();
        assert_eq!((f.frame_width, f.frame_height), (96, 96));
        assert_eq!(f.compressed_header_size, 100);
        assert_eq!(f.compressed_header_offset, 14);
    }

    // ------------------------------------------------------------------
    // Golden header values (first_partition_size + frame_header_length,
    // first 30 frames each) for the bundled samples in assets/samples/,
    // embedded at compile time so the test is self-contained.
    // Verified against FFmpeg's VAAPI pic params and an independent
    // bit-level reference parser. NOTE: anchors correspond to the sample
    // files on this machine; regenerate if the IVF set is re-encoded.
    // ------------------------------------------------------------------

    const VP9_PROFILE0_FPS: [u16; 30] = [
        249, 28, 6, 21, 20, 65, 6, 35, 13, 19, 219, 9, 21, 10, 8, 19, 7, 16, 7, 35, 129, 9, 21, 20,
        6, 23, 6, 20, 6, 34,
    ];
    const VP9_PROFILE0_HDR: [u8; 30] = [
        18, 10, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10,
        10, 10, 10, 10, 10, 10, 10,
    ];
    const VP9_PROFILE1_444_FPS: [u16; 30] = [
        210, 122, 43, 20, 6, 13, 14, 9, 3, 3, 94, 24, 15, 14, 5, 21, 5, 13, 3, 3, 75, 17, 14, 11,
        11, 5, 3, 3, 3, 10,
    ];
    const VP9_PROFILE1_444_HDR: [u8; 30] = [
        18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10,
    ];
    const VP9_PROFILE1_FPS: [u16; 30] = [
        250, 23, 3, 30, 10, 56, 6, 63, 6, 42, 238, 9, 20, 21, 13, 6, 9, 19, 8, 15, 84, 11, 23, 18,
        11, 5, 10, 13, 5, 13,
    ];
    const VP9_PROFILE1_HDR: [u8; 30] = [
        18, 10, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10,
    ];
    const VP9_PROFILE2_FPS: [u16; 30] = [
        252, 13, 4, 33, 19, 65, 9, 34, 10, 47, 201, 11, 17, 22, 12, 5, 3, 16, 8, 20, 99, 9, 25, 6,
        11, 5, 12, 10, 21, 7,
    ];
    const VP9_PROFILE2_HDR: [u8; 30] = [
        18, 10, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10,
        10, 10, 10, 10, 10, 10, 10,
    ];

    #[test]
    fn test_golden_headers_bundled_samples() {
        type GoldenCase = (
            &'static str,
            Vp9Profile,
            u8,
            &'static [u16; 30],
            &'static [u8; 30],
            &'static [u8],
        );
        let cases: [GoldenCase; 4] = [
            (
                "vp9_profile0.ivf",
                Vp9Profile::Profile0,
                8,
                &VP9_PROFILE0_FPS,
                &VP9_PROFILE0_HDR,
                include_bytes!("../../../assets/samples/vp9_profile0.ivf"),
            ),
            (
                "vp9_profile1_444.ivf",
                Vp9Profile::Profile1,
                8,
                &VP9_PROFILE1_444_FPS,
                &VP9_PROFILE1_444_HDR,
                include_bytes!("../../../assets/samples/vp9_profile1_444.ivf"),
            ),
            (
                "vp9_profile1.ivf",
                Vp9Profile::Profile2,
                10,
                &VP9_PROFILE1_FPS,
                &VP9_PROFILE1_HDR,
                include_bytes!("../../../assets/samples/vp9_profile1.ivf"),
            ),
            (
                "vp9_profile2.ivf",
                Vp9Profile::Profile2,
                12,
                &VP9_PROFILE2_FPS,
                &VP9_PROFILE2_HDR,
                include_bytes!("../../../assets/samples/vp9_profile2.ivf"),
            ),
        ];

        for (name, profile, bit_depth, fps_golden, hdr_golden, data) in cases {
            assert_eq!(&data[0..4], b"DKIF", "{name}: not an IVF file");
            let hsz = u16::from_le_bytes([data[6], data[7]]) as usize;

            let mut off = hsz;
            let mut parser = Vp9Parser::new();
            let mut i = 0u32;
            while off + 12 <= data.len() && i < 30 {
                let size =
                    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                        as usize;
                off += 12;
                let pkt = &data[off..off + size];
                off += size;

                let f = parser
                    .parse_frame(pkt)
                    .unwrap_or_else(|e| panic!("{name} f{i}: {e:?}"));
                assert_eq!(
                    f.compressed_header_size, fps_golden[i as usize] as u32,
                    "{name} f{i}: first_partition_size"
                );
                assert_eq!(
                    f.compressed_header_offset as u8, hdr_golden[i as usize],
                    "{name} f{i}: frame_header_length"
                );
                if i == 0 {
                    assert_eq!(f.picture_info.profile, profile, "{name}: profile");
                    assert_eq!(f.color_config.bit_depth, bit_depth, "{name}: bit depth");
                }
                i += 1;
            }
            assert_eq!(i, 30, "{name}: expected 30 frames");
        }
    }

    /// Full-stream frame-type classification for vp9_profile0.ivf (embedded):
    /// all 300 frames must parse and key frames must land exactly at decode
    /// positions 0, 128 and 256 - verified against ffprobe pict_type
    /// (3 I + 297 P, 0 mismatches).
    #[test]
    fn test_full_stream_frame_types_profile0() {
        let data = include_bytes!("../../../assets/samples/vp9_profile0.ivf");
        assert_eq!(&data[0..4], b"DKIF", "not an IVF file");
        let hsz = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut off = hsz;
        let mut parser = Vp9Parser::new();
        let mut i = 0u32;
        let mut key_positions: Vec<u32> = Vec::new();
        while off + 12 <= data.len() {
            let size = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                as usize;
            off += 12;
            let pkt = &data[off..off + size];
            off += size;

            let f = parser
                .parse_frame(pkt)
                .unwrap_or_else(|e| panic!("f{i}: {e:?}"));
            if f.frame_is_intra {
                key_positions.push(i);
            }
            i += 1;
        }
        assert_eq!(i, 300, "expected 300 frames");
        assert_eq!(key_positions, vec![0, 128, 256], "key frame positions");
    }
}
