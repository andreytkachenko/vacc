//! VAAPI H.264 encoder implementing `vacc_core::Encoder`.
//!
//! Ports the validated C probe flow (see memory `h264_encode_probe.md`):
//! packed SPS/PPS/slice headers, driver-internal VME (no per-MB buffer),
//! CAVLC, CQP rate control, single reference.

use libva::buffer::{
    BufferType, EncPackedHeaderParameter, EncPackedHeaderType,
    EncPictureParameterBufferH264, EncPictureParameterBufferHEVC,
    EncSequenceParameterBufferH264, EncSequenceParameterBufferHEVC,
    EncSliceParameterBufferH264, EncSliceParameterBufferHEVC, HEVCEncPicFields,
    HEVCEncSeqFields, HevcEncPicSccFields, HevcEncSeqSccFields, HevcEncSliceFields,
    H264EncPicFields, H264EncSeqFields, MappedCodedBuffer, PictureH264, PictureHEVC,
};
use libva::{
    Buffer, Config, Context, Display, Image, Picture, VAEntrypoint, VAProfile, VAConfigAttrib,
    VAConfigAttribType,
};
use std::rc::Rc;
use vacc_core::encoder::{EncodedFrame, Encoder, EncoderConfig, EncoderInfo, FrameKind, InputFrame};
use vacc_core::session::Extent2D;
use vacc_core::VideoCodec;

use super::Error;
use crate::{h264, h265};

/// Number of surfaces in the reference pool. Must exceed max_ref_frames so
/// a surface is never overwritten before the driver stops using it as a
/// reference.
const NSURFACES: usize = 6;

/// VAAPI H.264 / H.265 encoder.
pub struct VaapiEncoder {
    /// Must outlive the context.
    #[allow(dead_code)]
    config: Config,
    context: Rc<Context>,
    surfaces: Vec<libva::Surface<()>>,

    codec: VideoCodec,
    width: u32,
    height: u32,
    qp: u8,
    intra_period: u32,
    fps_num: u32,
    fps_den: u32,
    frame_index: u32,
}

impl VaapiEncoder {
    /// Open a display and create an H.264 encoder from `config`.
    pub fn open(config: &EncoderConfig) -> Result<Self, Error> {
        let device = crate::device::VaapiEncoderDevice::new()?;
        Self::open_on(device.display(), config)
    }

    /// Create an encoder on an existing display.
    pub fn open_on(display: &Rc<Display>, config: &EncoderConfig) -> Result<Self, Error> {
        config.validate().map_err(Error::InvalidState)?;
        let is_h265 = config.codec == vacc_core::VideoCodec::EncodeH265;
        if config.codec != vacc_core::VideoCodec::EncodeH264 && !is_h265 {
            return Err(Error::CodecNotSupported(format!(
                "vaapi-encode supports EncodeH264/EncodeH265 only (got {:?})",
                config.codec
            )));
        }

        let width = config.width;
        let height = config.height;
        let profile = if is_h265 {
            VAProfile::VAProfileHEVCMain
        } else {
            VAProfile::VAProfileH264Main
        };
        if !display
            .query_config_entrypoints(profile)
            .map(|eps| eps.contains(&VAEntrypoint::VAEntrypointEncSlice))
            .unwrap_or(false)
        {
            return Err(Error::CodecNotSupported(format!(
                "{:?} + VAEntrypointEncSlice not supported",
                profile
            )));
        }

        // Config attributes: CQP rate control, NV12 render format, packed
        // headers. H.264 uses SEQUENCE|PICTURE|SLICE; H.265 packs VPS+SPS+PPS
        // in the Sequence buffer and the slice header in the Slice buffer.
        let packed_headers: u32 = if is_h265 {
            0x1 | 0x4 // VA_ENC_PACKED_HEADER_SEQUENCE | SLICE
        } else {
            0x7 // SEQUENCE|PICTURE|SLICE
        };
        let attrs = vec![
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribRateControl,
                value: 0x0000_0010, // VA_RC_CQP
            },
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribRTFormat,
                value: 1,           // VA_RT_FORMAT_YUV420
            },
            VAConfigAttrib {
                type_: VAConfigAttribType::VAConfigAttribEncPackedHeaders,
                value: packed_headers,
            },
        ];
        let cfg = display
            .create_config(
                attrs,
                profile,
                VAEntrypoint::VAEntrypointEncSlice,
            )
            .map_err(|e| Error::EncoderInit(format!("vaCreateConfig: {}", e)))?;

        let surfaces = display
            .create_surfaces::<()>(
                1,
                Some(0x3231_564E), // VA_FOURCC_NV12 (must match input layout)
                width,
                height,
                None,
                vec![(); NSURFACES],
            )
            .map_err(|e| Error::EncoderInit(format!("vaCreateSurfaces: {}", e)))?;

        let context = display
            .create_context(&cfg, width, height, Some(&surfaces), true)
            .map_err(|e| Error::EncoderInit(format!("vaCreateContext: {}", e)))?;

        Ok(Self {
            config: cfg,
            context,
            surfaces,
            codec: config.codec,
            width,
            height,
            qp: config.qp,
            intra_period: config.intra_period,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            frame_index: 0,
        })
    }

    /// Upload one NV12 frame into a driver surface.
    fn upload_frame(&self, surface: &libva::Surface<()>, frame: &InputFrame) -> Result<(), Error> {
        if frame.width != self.width || frame.height != self.height {
            return Err(Error::InvalidState(format!(
                "input frame {}x{} does not match encoder {}x{}",
                frame.width,
                frame.height,
                self.width,
                self.height
            )));
        }
        let mut img = Image::derive_from(surface, (self.width, self.height))
            .map_err(|e| Error::VaApi(format!("vaDeriveImage: {}", e)))?;
        // Copy plane geometry before taking the mutable data borrow.
        let y_off = img.image().offsets[0] as usize;
        let y_pitch = img.image().pitches[0] as usize;
        let uv_off = img.image().offsets[1] as usize;
        let uv_pitch = img.image().pitches[1] as usize;
        let data = img.as_mut();

        // Y plane.
        for row in 0..self.height as usize {
            let src = &frame.y[row * frame.y_pitch..row * frame.y_pitch + self.width as usize];
            let dst = &mut data[y_off + row * y_pitch..y_off + row * y_pitch + self.width as usize];
            dst.copy_from_slice(src);
        }
        // Interleaved UV plane (width bytes per row).
        for row in 0..(self.height / 2) as usize {
            let src = &frame.uv[row * frame.uv_pitch..row * frame.uv_pitch + self.width as usize];
            let dst = &mut data[uv_off + row * uv_pitch..uv_off + row * uv_pitch + self.width as usize];
            dst.copy_from_slice(src);
        }
        // Drop unmaps the image buffer.
        Ok(())
    }

    /// Build the per-frame sequence parameter buffer (contents are static;
    /// the VA buffer itself must be recreated per picture because `Buffer`
    /// owns its VABufferID).
    fn seq_buffer(&self) -> Buffer {
        let seq_fields = H264EncSeqFields::new(
            1, // chroma_format_idc (4:2:0)
            1, // frame_mbs_only_flag
            0, // mb_adaptive_frame_field_flag
            0, // seq_scaling_matrix_present_flag
            1, // direct_8x8_inference_flag
            0, // log2_max_frame_num_minus4
            0, // pic_order_cnt_type
            4, // log2_max_pic_order_cnt_lsb_minus4
            0, // delta_pic_order_always_zero_flag
        );
        let seq = EncSequenceParameterBufferH264::new(
            0,                        // seq_parameter_set_id
            31,                       // level_idc (3.1)
            self.intra_period,        // intra_period
            self.intra_period,        // intra_idr_period
            1,                        // ip_period
            500_000,        // bits_per_second (nominal; CQP ignores it)
            1,                        // max_num_ref_frames
            (self.width / 16) as u16, // picture_width_in_mbs
            (self.height / 16) as u16,
            &seq_fields,
            0, // bit_depth_luma_minus8
            0, // bit_depth_chroma_minus8
            0, // num_ref_frames_in_pic_order_cnt_cycle
            0, // offset_for_non_ref_pic
            0, // offset_for_top_to_bottom_field
            [0i32; 256],              // offset_for_ref_frame
            None,                     // frame_crop
            None,                     // vui_fields
            0,                        // aspect_ratio_idc
            0,                        // sar_width
            0,                        // sar_height
            0,                        // num_units_in_tick
            0,                        // time_scale
        );
        self.context
            .create_buffer(BufferType::EncSequenceParameter(
                libva::buffer::EncSequenceParameter::H264(seq),
            ))
            .expect("seq param buffer")
    }

    /// Build packed SPS parameter + data buffers.
    fn sps_buffers(&self) -> (Buffer, Buffer) {
        let sps_rbsp = h264::make_sps(self.width / 16, self.height / 16, 31, 1);
        // Packed data: start code + NAL header (ref_idc=3, type=7) + EPB'd RBSP.
        let mut sps_pk = Vec::with_capacity(4 + sps_rbsp.len());
        sps_pk.extend_from_slice(&[0, 0, 1, 0x67]);
        sps_pk.extend_from_slice(&sps_rbsp);
        (
            self.context
                .create_buffer(BufferType::EncPackedHeaderParameter(
                    EncPackedHeaderParameter::new(
                        // H264_SPS == Sequence in current VA API.
                        EncPackedHeaderType::Sequence,
                        (sps_pk.len() as u32) * 8,
                        true,
                    ),
                ))
                .expect("sps param buffer"),
            self.context
                .create_buffer(BufferType::EncPackedHeaderData(sps_pk))
                .expect("sps data buffer"),
        )
    }

    /// Build packed PPS parameter + data buffers.
    fn pps_buffers(&self) -> (Buffer, Buffer) {
        let pps_rbsp = h264::make_pps();
        // Packed data: start code + NAL header (ref_idc=1, type=8) + EPB'd RBSP.
        let mut pps_pk = Vec::with_capacity(4 + pps_rbsp.len());
        pps_pk.extend_from_slice(&[0, 0, 1, 0x28]);
        pps_pk.extend_from_slice(&pps_rbsp);
        (
            self.context
                .create_buffer(BufferType::EncPackedHeaderParameter(
                    EncPackedHeaderParameter::new(
                        // H264_PPS == Picture in current VA API.
                        EncPackedHeaderType::Picture,
                        (pps_pk.len() as u32) * 8,
                        true,
                    ),
                ))
                .expect("pps param buffer"),
            self.context
                .create_buffer(BufferType::EncPackedHeaderData(pps_pk))
                .expect("pps data buffer"),
        )
    }

    /// Collect coded segments from the coded buffer into Annex B bytes.
    fn read_coded(coded: &libva::buffer::EncCodedBuffer) -> Result<Vec<u8>, Error> {
        const VA_CODED_BUF_STATUS_BAD_BITSTREAM: u32 = 1 << 2;
        let mapped = MappedCodedBuffer::new(coded)
            .map_err(|e| Error::VaApi(format!("vaMapBuffer(coded): {}", e)))?;
        let mut out = Vec::new();
        for seg in mapped.iter() {
            if seg.status & VA_CODED_BUF_STATUS_BAD_BITSTREAM != 0 {
                eprintln!(
                    "bad-bitstream segment: status={:#x} bit_offset={} len={} head={:02x?}",
                    seg.status,
                    seg.bit_offset,
                    seg.buf.len(),
                    &seg.buf[..seg.buf.len().min(64)]
                );
                return Err(Error::CodecError("driver reported bad bitstream".into()));
            }
            // Driver segments already contain start codes.
            out.extend_from_slice(seg.buf);
        }
        Ok(out)
    }
    fn encode_h264(
        &self,
        idx: usize,
        coded: &libva::buffer::EncCodedBuffer,
        is_idr: bool,
        poc_lsb: u16,
        timestamp: i64,
    ) -> Result<Vec<u8>, Error> {
        let nmb = (self.width / 16) * (self.height / 16);
        let frame_num = (self.frame_index & 0xF) as u16;

        let curr = PictureH264::new(self.surfaces[idx].id(), self.frame_index, 0, 0, 0);
        // Previous-frame reference (PictureH264 is not Clone; rebuild from
        // primitives where needed).
        let prev: Option<(u32, u32)> = if is_idr {
            None
        } else {
            let prev_idx = ((self.frame_index - 1) as usize) % self.surfaces.len();
            Some((self.surfaces[prev_idx].id(), self.frame_index - 1))
        };
        let refs: [PictureH264; 16] = std::array::from_fn(|i| {
            match (prev, i) {
                (Some((pid, pidx)), 0) => PictureH264::new(pid, pidx, 0x8 /* SHORT_TERM_REFERENCE */, 0, 0),
                _ => PictureH264::new(0, 0, 0, 0, 0),
            }
        });
        let pic_fields = H264EncPicFields::new(
            if is_idr { 1 } else { 0 }, // idr_pic_flag
            1,                          // reference_pic_flag
            0,                          // entropy_coding_mode_flag (CAVLC)
            0,                          // weighted_pred_flag
            0,                          // weighted_bipred_idc
            0,                          // constrained_intra_pred_flag
            0,                          // transform_8x8_mode_flag
            0,                          // deblocking_filter_control_present_flag (matches C probe)
            0,                          // redundant_pic_cnt_present_flag
            1,                          // pic_order_present_flag
            0,                          // pic_scaling_matrix_present_flag
        );
        let pic_param = EncPictureParameterBufferH264::new(
            curr,
            refs,
            coded.id(),
            0, // pic_parameter_set_id
            0, // seq_parameter_set_id
            0, // last_picture
            frame_num,
            self.qp, // pic_init_qp
            0,       // num_ref_idx_l0_active_minus1
            0,       // num_ref_idx_l1_active_minus1
            0,       // chroma_qp_index_offset
            0,       // second_chroma_qp_index_offset
            &pic_fields,
        );

        let rpl0: [PictureH264; 32] = std::array::from_fn(|i| {
            match (prev, i) {
                (Some((pid, pidx)), 0) => PictureH264::new(pid, pidx, 0x8 /* SHORT_TERM_REFERENCE */, 0, 0),
                _ => PictureH264::new(0, 0, 0, 0, 0),
            }
        });
        let slice_param = EncSliceParameterBufferH264::new(
            0,                            // macroblock_address
            nmb,                          // num_macroblocks
            0,                            // macroblock_info (VA_INVALID_ID: driver VME)
            if is_idr { 7 } else { 5 },   // slice_type (I=7, P=5)
            0,                            // pic_parameter_set_id
            if is_idr { 1 } else { 0 },   // idr_pic_id
            poc_lsb,                      // pic_order_cnt_lsb
            0,                            // delta_pic_order_cnt_bottom
            [0; 2],                       // delta_pic_order_cnt
            0,                            // direct_spatial_mv_pred_flag
            0,                            // num_ref_idx_active_override_flag
            0,                            // num_ref_idx_l0_active_minus1
            0,                            // num_ref_idx_l1_active_minus1
            rpl0,                         // ref_pic_list_0
            std::array::from_fn(|_| PictureH264::new(0, 0, 0, 0, 0)), // ref_pic_list_1
            0,                            // luma_log2_weight_denom
            0,                            // chroma_log2_weight_denom
            0,                            // luma_weight_l0_flag
            [0i16; 32],                   // luma_weight_l0
            [0i16; 32],                   // luma_offset_l0
            0,                            // chroma_weight_l0_flag
            [[0i16; 2]; 32],              // chroma_weight_l0
            [[0i16; 2]; 32],              // chroma_offset_l0
            0,                            // luma_weight_l1_flag
            [0i16; 32],                   // luma_weight_l1
            [0i16; 32],                   // luma_offset_l1
            0,                            // chroma_weight_l1_flag
            [[0i16; 2]; 32],              // chroma_weight_l1
            [[0i16; 2]; 32],              // chroma_offset_l1
            0,                            // cabac_init_idc
            0,                            // slice_qp_delta
            0,                            // disable_deblocking_filter_idc
            0,                            // slice_alpha_c0_offset_div2
            0,                            // slice_beta_offset_div2
        );

        let (slh_bytes, slh_bits) = h264::make_slice_hdr(is_idr, self.frame_index, poc_lsb as u32, 1);
        let slh_param = EncPackedHeaderParameter::new(
            // H264_Slice == Slice in current VA API. has_emulation_bytes=0:
            // the driver inserts EPB and appends rbsp_slice_trailing_bits.
            EncPackedHeaderType::Slice,
            slh_bits,
            false,
        );

        let seq_buf = self.seq_buffer();
        let (sps_hdr, sps_data) = self.sps_buffers();
        let (pps_hdr, pps_data) = self.pps_buffers();
        let pic_buf = self
            .context
            .create_buffer(BufferType::EncPictureParameter(
                libva::buffer::EncPictureParameter::H264(pic_param),
            ))
            .map_err(|e| Error::VaApi(format!("picture param buffer: {}", e)))?;
        let slice_buf = self
            .context
            .create_buffer(BufferType::EncSliceParameter(
                libva::buffer::EncSliceParameter::H264(slice_param),
            ))
            .map_err(|e| Error::VaApi(format!("slice param buffer: {}", e)))?;
        let slh_hdr = self
            .context
            .create_buffer(BufferType::EncPackedHeaderParameter(slh_param))
            .map_err(|e| Error::VaApi(format!("slice hdr param buffer: {}", e)))?;
        let slh_data = self
            .context
            .create_buffer(BufferType::EncPackedHeaderData(slh_bytes))
            .map_err(|e| Error::VaApi(format!("slice hdr data buffer: {}", e)))?;

        // Render. Order matters: iHD processes render buffers sequentially
        // and must see the packed slice header BEFORE the slice parameter
        // buffer (sets bHavePackedSliceHdr), matching the C probe layout:
        // seq, packed SPS (param+data), packed PPS (param+data),
        // packed slice hdr (param+data), picture params, slice params.
        let mut picture = Picture::new(timestamp as u64, Rc::clone(&self.context), &self.surfaces[idx]);
        picture.add_buffer(seq_buf);
        picture.add_buffer(sps_hdr);
        picture.add_buffer(sps_data);
        picture.add_buffer(pps_hdr);
        picture.add_buffer(pps_data);
        picture.add_buffer(slh_hdr);
        picture.add_buffer(slh_data);
        picture.add_buffer(pic_buf);
        picture.add_buffer(slice_buf);
        picture
            .begin()
            .and_then(|p| p.render())
            .and_then(|p| p.end())
            .and_then(|p| match p.sync() {
                Ok(p) => Ok(p),
                Err((e, _)) => Err(e),
            })
            .map_err(|e| Error::VaApi(format!("render: {}", e)))?;

        // Read back the coded bitstream (Annex B).
        Self::read_coded(coded)
    }

    /// H.265/HEVC: build picture/slice/packed-header buffers, render, read back.
    fn encode_h265(
        &self,
        idx: usize,
        coded: &libva::buffer::EncCodedBuffer,
        is_idr: bool,
        poc_lsb: u16,
        timestamp: i64,
    ) -> Result<Vec<u8>, Error> {
        const RPS_ST_CURR_BEFORE: u32 = 0x10; // VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
        let nctu = self.num_ctus();
        let poc = self.frame_index as i32;

        let curr = PictureHEVC::new(self.surfaces[idx].id(), poc, 0);
        let prev: Option<u32> = if is_idr {
            None
        } else {
            let prev_idx = ((self.frame_index - 1) as usize) % self.surfaces.len();
            Some(self.surfaces[prev_idx].id())
        };
        let refs: [PictureHEVC; 15] = std::array::from_fn(|i| match (prev, i) {
            (Some(pid), 0) => PictureHEVC::new(pid, poc - 1, RPS_ST_CURR_BEFORE),
            _ => PictureHEVC::new(0, 0, 0),
        });

        let pic_fields = HEVCEncPicFields::new(
            if is_idr { 1 } else { 0 }, // idr_pic_flag
            if is_idr { 1 } else { 2 }, // coding_type (HEVC: I=1, P=2, B=3)
            1,                          // reference_pic_flag
            0,                          // dependent_slice_segments_enabled
            1,                          // sign_data_hiding_enabled
            0,                          // constrained_intra_pred
            0,                          // transform_skip_enabled
            1,                          // cu_qp_delta_enabled
            0,                          // weighted_pred
            0,                          // weighted_bipred
            0,                          // transquant_bypass
            0,                          // tiles_enabled
            0,                          // entropy_coding_sync
            1,                          // loop_filter_across_tiles
            1,                          // pps_loop_filter_across_slices
            0,                          // scaling_list_data_present
            0,                          // screen_content
            0,                          // enable_gpu_weighted_prediction
            0,                          // no_output_of_prior_pics
        );
        let pic_param = EncPictureParameterBufferHEVC::new(
            curr,
            refs,
            coded.id(),
            0, // collocated_ref_pic_index
            0, // last_picture
            self.qp, // pic_init_qp
            0,       // diff_cu_qp_delta_depth
            0,       // pps_cb_qp_offset
            0,       // pps_cr_qp_offset
            0,       // num_tile_columns_minus1
            0,       // num_tile_rows_minus1
            [0; 19], // column_width_minus1
            [0; 21], // row_height_minus1
            0, // log2_parallel_merge_level_minus2
            0, // ctu_max_bitsize_allowed
            0, // num_ref_idx_l0_default_active_minus1
            0, // num_ref_idx_l1_default_active_minus1
            0, // slice_pic_parameter_set_id
            if is_idr { 19 } else { 1 }, // nal_unit_type (IDR_W_RADL / TRAIL_N)
            &pic_fields,
            1, // hierarchical_level_plus1
            0, // va_byte_reserved
            &HevcEncPicSccFields::new(0),
        );

        let rpl0: [PictureHEVC; 15] = std::array::from_fn(|i| match (prev, i) {
            (Some(pid), 0) => PictureHEVC::new(pid, poc - 1, RPS_ST_CURR_BEFORE),
            _ => PictureHEVC::new(0, 0, 0),
        });
        let slice_fields = HevcEncSliceFields::new(
            1, // last_slice_of_pic_flag
            0, // dependent_slice_segment_flag
            0, // colour_plane_id
            0, // slice_temporal_mvp_enabled
            1, // slice_sao_luma_flag
            1, // slice_sao_chroma_flag
            0, // num_ref_idx_active_override
            0, // mvd_l1_zero
            0, // cabac_init
            0, // slice_deblocking_filter_disabled
            1, // slice_loop_filter_across_slices_enabled
            0, // collocated_from_l0
        );
        let slice_param = EncSliceParameterBufferHEVC::new(
            0, // slice_segment_address
            nctu, // num_ctu_in_slice
            if is_idr { 2 } else { 1 }, // slice_type (raw HEVC syntax: I=2, P=1, B=0)
            0, // slice_pic_parameter_set_id
            0, // num_ref_idx_l0_active_minus1
            0, // num_ref_idx_l1_active_minus1
            rpl0, // ref_pic_list0
            std::array::from_fn(|_| PictureHEVC::new(0, 0, 0)), // ref_pic_list1
            0, // luma_log2_weight_denom
            0, // delta_chroma_log2_weight_denom
            [0i8; 15], // delta_luma_weight_l0
            [0i8; 15], // luma_offset_l0
            [[0i8; 2]; 15], // delta_chroma_weight_l0
            [[0i8; 2]; 15], // chroma_offset_l0
            [0i8; 15], // delta_luma_weight_l1
            [0i8; 15], // luma_offset_l1
            [[0i8; 2]; 15], // delta_chroma_weight_l1
            [[0i8; 2]; 15], // chroma_offset_l1
            5, // max_num_merge_cand (matches five_minus_max_num_merge_cand=0)
            0, // slice_qp_delta
            0, // slice_cb_qp_offset
            0, // slice_cr_qp_offset
            0, // slice_beta_offset_div2
            0, // slice_tc_offset_div2
            &slice_fields,
            0, // pred_weight_table_bit_offset
            0, // pred_weight_table_bit_length
        );

        // Packed Slice header (sent every frame). has_emulation_bytes=1: the
        // data already carries start code + EPB'd RBSP.
        let (slh_bytes, slh_bits) = h265::make_slice_hdr(is_idr, poc_lsb as u32);
        let slh_hdr = self
            .context
            .create_buffer(BufferType::EncPackedHeaderParameter(
                EncPackedHeaderParameter::new(EncPackedHeaderType::Slice, slh_bits, true),
            ))
            .map_err(|e| Error::VaApi(format!("slice hdr param buffer: {}", e)))?;
        let slh_data = self
            .context
            .create_buffer(BufferType::EncPackedHeaderData(slh_bytes))
            .map_err(|e| Error::VaApi(format!("slice hdr data buffer: {}", e)))?;

        let pic_buf = self
            .context
            .create_buffer(BufferType::EncPictureParameter(
                libva::buffer::EncPictureParameter::HEVC(pic_param),
            ))
            .map_err(|e| Error::VaApi(format!("picture param buffer: {}", e)))?;
        let slice_buf = self
            .context
            .create_buffer(BufferType::EncSliceParameter(
                libva::buffer::EncSliceParameter::HEVC(slice_param),
            ))
            .map_err(|e| Error::VaApi(format!("slice param buffer: {}", e)))?;

        // Render order mirrors FFmpeg's hevc_vaapi: on IDR, the sequence
        // parameter buffer + packed VPS/SPS/PPS precede the picture buffer.
        let mut picture = Picture::new(timestamp as u64, Rc::clone(&self.context), &self.surfaces[idx]);
        if is_idr {
            picture.add_buffer(self.h265_seq_buffer());
            let (seq_hdr, seq_data) = self.seq_packed_buffers();
            picture.add_buffer(seq_hdr);
            picture.add_buffer(seq_data);
        }
        picture.add_buffer(pic_buf);
        picture.add_buffer(slh_hdr);
        picture.add_buffer(slh_data);
        picture.add_buffer(slice_buf);
        picture
            .begin()
            .and_then(|p| p.render())
            .and_then(|p| p.end())
            .and_then(|p| match p.sync() {
                Ok(p) => Ok(p),
                Err((e, _)) => Err(e),
            })
            .map_err(|e| Error::VaApi(format!("render: {}", e)))?;

        Self::read_coded(coded)
    }

    /// Total CTU count for a 64x64 CTB grid (matches the SPS).
    fn num_ctus(&self) -> u32 {
        ((self.width + 63) / 64) * ((self.height + 63) / 64)
    }

    /// HEVC sequence parameter buffer (static contents; recreated per picture).
    fn h265_seq_buffer(&self) -> Buffer {
        let seq_fields = HEVCEncSeqFields::new(
            1, // chroma_format_idc (4:2:0)
            0, // separate_colour_plane
            0, // bit_depth_luma_minus8
            0, // bit_depth_chroma_minus8
            0, // scaling_list_enabled
            1, // strong_intra_smoothing (matches SPS)
            0, // amp_enabled (matches SPS)
            1, // sample_adaptive_offset (matches SPS)
            0, // pcm_enabled
            0, // pcm_loop_filter_disabled
            0, // sps_temporal_mvp (matches SPS)
            1, // low_delay_seq
            0, // hierarchical_flag
        );
        let seq = EncSequenceParameterBufferHEVC::new(
            1, // general_profile_idc (Main)
            60, // general_level_idc
            0, // general_tier_flag
            self.intra_period,
            self.intra_period,
            1, // ip_period
            500_000, // bits_per_second (nominal; CQP ignores it)
            self.width as u16,
            self.height as u16,
            &seq_fields,
            0, // log2_min_luma_coding_block_size_minus3
            3, // log2_diff_max_min_luma_coding_block_size
            0, // log2_min_transform_block_size_minus2
            3, // log2_diff_max_min_transform_block_size
            0, // max_transform_hierarchy_depth_inter (maxTU = 4x4, matches SPS/x265)
            0, // max_transform_hierarchy_depth_intra
            0, // pcm_sample_bit_depth_luma_minus1
            0, // pcm_sample_bit_depth_chroma_minus1
            0, // log2_min_pcm_luma_coding_block_size_minus3
            0, // log2_max_pcm_luma_coding_block_size_minus3
            None, // vui_fields
            0, // aspect_ratio_idc
            0, // sar_width
            0, // sar_height
            0, // vui_num_units_in_tick
            0, // vui_time_scale
            0, // min_spatial_segmentation_idc
            0, // max_bytes_per_pic_denom
            0, // max_bits_per_min_cu_denom
            &HevcEncSeqSccFields::new(0),
        );
        self.context
            .create_buffer(BufferType::EncSequenceParameter(
                libva::buffer::EncSequenceParameter::HEVC(seq),
            ))
            .expect("h265 seq param buffer")
    }

    /// Packed Sequence buffer: VPS + SPS + PPS NALs (4-byte start-coded).
    fn seq_packed_buffers(&self) -> (Buffer, Buffer) {
        let data = h265::seq_packed_data(self.width, self.height);
        (
            self.context
                .create_buffer(BufferType::EncPackedHeaderParameter(
                    EncPackedHeaderParameter::new(
                        EncPackedHeaderType::Sequence,
                        (data.len() as u32) * 8,
                        true,
                    ),
                ))
                .expect("seq packed param buffer"),
            self.context
                .create_buffer(BufferType::EncPackedHeaderData(data))
                .expect("seq packed data buffer"),
        )
    }

}

impl Encoder for VaapiEncoder {
    type Error = Error;

    fn new(config: &EncoderConfig) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Self::open(config)
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            backend: "vaapi".into(),
            codec: self.codec,
            coded_size: Extent2D::new(self.width, self.height),
            qp: self.qp,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
        }
    }

    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Self::Error> {
        let idx = (self.frame_index as usize) % self.surfaces.len();
        let is_idr = self.frame_index == 0 || self.frame_index.is_multiple_of(self.intra_period);
        let poc_lsb = (self.frame_index & 0xFF) as u16;

        // 1. Upload input pixels.
        self.upload_frame(&self.surfaces[idx], frame)?;

        // 2. Coded buffer.
        let coded = self
            .context
            .create_enc_coded((self.width * self.height * 2) as usize)
            .map_err(|e| Error::VaApi(format!("create coded buffer: {}", e)))?;

        // 3. Build per-frame buffers + render (codec-specific), then read back.
        let data = match self.codec {
            VideoCodec::EncodeH265 => {
                self.encode_h265(idx, &coded, is_idr, poc_lsb, timestamp)
            }
            _ => self.encode_h264(idx, &coded, is_idr, poc_lsb, timestamp),
        }?;

        self.frame_index += 1;

        Ok(vec![EncodedFrame {
            frame_index: self.frame_index - 1,
            timestamp,
            dts: timestamp,
            data,
            is_idr,
            kind: if is_idr { FrameKind::I } else { FrameKind::P },
            poc: poc_lsb as i32,
        }])
    }

    /// H.264: build picture/slice/packed-header buffers, render, read back.
    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error> {
        // Synchronous backend: nothing in flight.
        Ok(Vec::new())
    }
}
