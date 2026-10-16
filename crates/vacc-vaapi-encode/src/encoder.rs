//! VAAPI H.264 encoder implementing `vacc_core::Encoder`.
//!
//! Ports the validated C probe flow (see memory `h264_encode_probe.md`):
//! packed SPS/PPS/slice headers, driver-internal VME (no per-MB buffer),
//! CAVLC, CQP rate control, single reference.

use libva::buffer::{
    BufferType, EncMiscParameter, EncMiscParameterFrameRate, EncPackedHeaderParameter,
    EncPackedHeaderType, EncPictureParameterBufferH264, EncPictureParameterBufferHEVC,
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
    /// Input surfaces: NV12 frames are uploaded here and passed to
    /// `vaBeginPicture`. Indexed by `frame_index % len`.
    surfaces: Vec<libva::Surface<()>>,
    /// Reconstructed (decoded) surfaces for HEVC: `decoded_curr_pic` and all
    /// reference slots point at these, mirroring FFmpeg's separate
    /// input/recon surface model. iHD requires the recon surface to be a
    /// dedicated scratch surface, not the input surface.
    recon_surfaces: Vec<libva::Surface<()>>,

    codec: VideoCodec,
    width: u32,
    height: u32,
    qp: u8,
    intra_period: u32,
    fps_num: u32,
    fps_den: u32,
    frame_index: u32,

    /// HEVC GPB path: input frames buffered until their mini-GOP is complete
    /// (always < 3 entries; drained on IDR boundaries and `flush`).
    pending: Vec<PendingFrame>,
    /// Display index of the last emitted IDR (POC base for poc_lsb).
    last_idr_frame_index: u32,
    /// Display index of the most recent anchor/IDR (L0 reference source).
    last_ref_display: u32,
}

/// An input frame held until its HEVC mini-GOP can be encoded in decode
/// order. Owns copies of the NV12 planes (the caller's `InputFrame` is a
/// borrowed view that may not outlive the call).
struct PendingFrame {
    frame_index: u32,
    timestamp: i64,
    width: u32,
    height: u32,
    /// Packed NV12 planes (pitch == width).
    y: Vec<u8>,
    uv: Vec<u8>,
}

impl PendingFrame {
    fn new(frame_index: u32, timestamp: i64, frame: &InputFrame) -> Self {
        let mut y = Vec::with_capacity(frame.y.len());
        for row in 0..frame.height as usize {
            let start = row * frame.y_pitch;
            y.extend_from_slice(&frame.y[start..start + frame.width as usize]);
        }
        let mut uv = Vec::with_capacity(frame.uv.len());
        for row in 0..(frame.height / 2) as usize {
            let start = row * frame.uv_pitch;
            uv.extend_from_slice(&frame.uv[start..start + frame.width as usize]);
        }
        Self {
            frame_index,
            timestamp,
            width: frame.width,
            height: frame.height,
            y,
            uv,
        }
    }

    fn as_input(&self) -> InputFrame<'_> {
        InputFrame::packed(self.width, self.height, &self.y, &self.uv)
    }
}

/// Coded picture role within the FF-mirrored GOP structure.
#[derive(Clone, Copy)]
enum Role {
    /// IDR_W_RADL I-slice.
    Idr,
    /// B-slice with a single L0 reference (FF's "P emitted as B").
    Anchor,
    /// B-slice with L0 (previous anchor) and L1 (current anchor).
    B,
}

/// Per-frame encoding job: role + reference display indices.
struct FrameJob {
    role: Role,
    /// Display index of the L0 reference (`None` for IDR).
    l0_ref: Option<u32>,
    /// Display index of the L1 reference (B frames only).
    l1_ref: Option<u32>,
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
        let mut attrs = vec![
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
        if std::env::var("VACC_NO_RT").is_ok() {
            attrs.retain(|a| a.type_ != VAConfigAttribType::VAConfigAttribRTFormat);
        }
        let cfg = display
            .create_config(
                attrs,
                profile,
                VAEntrypoint::VAEntrypointEncSlice,
            )
            .map_err(|e| Error::EncoderInit(format!("vaCreateConfig: {}", e)))?;

        // One combined pool of input + recon surfaces (Surface is not Clone,
        // so build a single vec and split it). Registering both halves with
        // the context lets the driver accept recon surfaces as
        // decoded_curr_pic / reference frames.
        let all_surfaces = display
            .create_surfaces::<()>(
                1,
                Some(0x3231_564E), // VA_FOURCC_NV12 (must match input layout)
                width,
                height,
                None,
                vec![(); 2 * NSURFACES],
            )
            .map_err(|e| Error::EncoderInit(format!("vaCreateSurfaces: {}", e)))?;

        let context = display
            .create_context(&cfg, width, height, Some(&all_surfaces), true)
            .map_err(|e| Error::EncoderInit(format!("vaCreateContext: {}", e)))?;

        // Split: first half = input surfaces, second half = recon surfaces.
        let mut surfaces = all_surfaces;
        let recon_surfaces = surfaces.split_off(NSURFACES);

        Ok(Self {
            config: cfg,
            context,
            surfaces,
            recon_surfaces,
            codec: config.codec,
            width,
            height,
            qp: config.qp,
            intra_period: config.intra_period,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            frame_index: 0,
            pending: Vec::new(),
            last_idr_frame_index: 0,
            last_ref_display: 0,
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
        // iHD flags some perfectly valid segments (notably the first bipred
        // B-frame of a mini-GOP) with BAD_BITSTREAM even though the
        // bitstream decodes cleanly, so accept it and warn once per process.
        static BAD_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let mapped = MappedCodedBuffer::new(coded)
            .map_err(|e| Error::VaApi(format!("vaMapBuffer(coded): {}", e)))?;
        let mut out = Vec::new();
        for seg in mapped.iter() {
            if seg.status & VA_CODED_BUF_STATUS_BAD_BITSTREAM != 0
                && !BAD_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "warning: driver flagged segment status={:#x} len={} as bad bitstream; accepting (known iHD false positive)",
                    seg.status,
                    seg.buf.len()
                );
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

    /// QP delta for IDR/anchor slices relative to pic_init_qp (26).
    fn idr_anchor_qp_delta(&self) -> i32 {
        self.qp as i32 - 26
    }

    /// B-slice QP delta: FF's hevc_vaapi uses fixed_qp_b = round(1.2·qp + 0.5)
    /// (b_qfactor 6/5, b_qoffset 0), clamped to [1, 51].
    fn b_qp_delta(&self) -> i32 {
        let b = ((1.2 * self.qp as f64 + 0.5) as i32).clamp(1, 51);
        b - 26
    }

    /// HEVC: build picture/slice/packed-header buffers for one frame of the
    /// FF-mirrored GOP structure, render it, and read back the coded data.
    fn encode_one(&self, pf: &PendingFrame, job: &FrameJob) -> Result<Vec<u8>, Error> {
        const RPS_ST_CURR_BEFORE: u32 = 0x10; // VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
        const RPS_ST_CURR_AFTER: u32 = 0x20; // VA_PICTURE_HEVC_RPS_ST_CURR_AFTER

        let nctu = self.num_ctus();
        let idx = (pf.frame_index as usize) % self.surfaces.len();
        let poc = (pf.frame_index - self.last_idr_frame_index) as i32;
        // References and the reconstructed current picture live in the recon
        // pool (FFmpeg's model); the input frame is uploaded to the input pool.
        let surf_id = |d: u32| self.recon_surfaces[(d as usize) % self.recon_surfaces.len()].id();
        let ref_poc = |d: u32| (d - self.last_idr_frame_index) as i32;

        self.upload_frame(&self.surfaces[idx], &pf.as_input())?;
        // Coded buffer size mirrors FFmpeg's formula (3*w*h + 64KiB); iHD may
        // key internal partitioning off the allocated coded size.
        let coded = self
            .context
            .create_enc_coded((self.width * self.height * 3 + 65536) as usize)
            .map_err(|e| Error::VaApi(format!("create coded buffer: {}", e)))?;

        // Unused reference slots: VA_INVALID_SURFACE + INVALID flag, exactly as
        // in FF's captured buffers (all-zero entries look like valid surface 0).
        let invalid = PictureHEVC::new(u32::MAX, 0, 1);
        let l0 = job.l0_ref;
        let l1 = job.l1_ref;

        // EncPictureParameterBufferHEVC.refs: L0 then L1 (rest INVALID).
        let refs: [PictureHEVC; 15] = std::array::from_fn(|i| match i {
            0 if l0.is_some() => {
                PictureHEVC::new(surf_id(l0.unwrap()), ref_poc(l0.unwrap()), RPS_ST_CURR_BEFORE)
            }
            1 if l1.is_some() => {
                PictureHEVC::new(surf_id(l1.unwrap()), ref_poc(l1.unwrap()), RPS_ST_CURR_AFTER)
            }
            _ => invalid,
        });

        let (idr_pic_flag, coding_type, reference_pic_flag) = match job.role {
            Role::Idr => (1, 1, 1),
            Role::Anchor => (0, 2, 1),
            Role::B => (0, 3, 0),
        };
        // pic_fields mirror FF's captured buffers; the packed PPS is
        // authoritative for the bitstream (iHD ignores these structured flags).
        let pic_fields = HEVCEncPicFields::new(
            idr_pic_flag,
            coding_type,
            reference_pic_flag,
            0, // dependent_slice_segments_enabled
            0, // sign_data_hiding_enabled
            0, // constrained_intra_pred
            1, // transform_skip_enabled (iHD fixed config)
            0, // cu_qp_delta_enabled
            0, // weighted_pred
            0, // weighted_bipred
            0, // transquant_bypass
            0, // tiles_enabled
            0, // entropy_coding_sync
            0, // loop_filter_across_tiles (FF: 0; tiles disabled)
            1, // pps_loop_filter_across_slices (matches our PPS loop_across=1)
            0, // scaling_list_data_present
            0, // screen_content
            0, // enable_gpu_weighted_prediction
            0, // no_output_of_prior_pics
        );
        let pic_param = EncPictureParameterBufferHEVC::new(
            PictureHEVC::new(self.recon_surfaces[idx].id(), poc, 0),
            refs,
            coded.id(),
            0, // collocated_ref_pic_index
            0, // last_picture
            26, // pic_init_qp (matches PPS init_qp_minus26 = 0)
            0, // diff_cu_qp_delta_depth
            0, // pps_cb_qp_offset
            0, // pps_cr_qp_offset
            0, // num_tile_columns_minus1
            0, // num_tile_rows_minus1
            [0; 19], // column_width_minus1
            [0; 21], // row_height_minus1
            0, // log2_parallel_merge_level_minus2
            0, // ctu_max_bitsize_allowed
            0, // num_ref_idx_l0_default_active_minus1
            0, // num_ref_idx_l1_default_active_minus1
            0, // slice_pic_parameter_set_id
            match job.role {
                Role::Idr => 19,   // IDR_W_RADL
                Role::Anchor => 1, // TRAIL_N
                Role::B => 0,      // TRAIL_R
            }, // nal_unit_type
            &pic_fields,
            0, // hierarchical_level_plus1
            0, // va_byte_reserved
            &HevcEncPicSccFields::new(0),
        );

        // Slice reference lists. The anchor keeps its L0 reference in BOTH
        // lists (FF does this even though the packed RPS has an empty L1);
        // B frames use L0 = previous anchor, L1 = current anchor.
        let rpl0: [PictureHEVC; 15] = std::array::from_fn(|i| match (i, l0) {
            (0, Some(d)) => PictureHEVC::new(surf_id(d), ref_poc(d), RPS_ST_CURR_BEFORE),
            _ => invalid,
        });
        let rpl1: [PictureHEVC; 15] = std::array::from_fn(|i| match (i, job.role, l0, l1) {
            (0, Role::Anchor, Some(d), _) => {
                PictureHEVC::new(surf_id(d), ref_poc(d), RPS_ST_CURR_BEFORE)
            }
            (0, _, _, Some(d)) => PictureHEVC::new(surf_id(d), ref_poc(d), RPS_ST_CURR_AFTER),
            _ => invalid,
        });

        // FF: I-slices have no temporal MVP; inter slices enable it.
        let slice_tmvp = if matches!(job.role, Role::Idr) { 0 } else { 1 };
        let slice_fields = HevcEncSliceFields::new(
            1, // last_slice_of_pic_flag
            0, // dependent_slice_segment_flag
            0, // colour_plane_id
            slice_tmvp, // slice_temporal_mvp_enabled (FF: I=0, inter=1)
            1, // slice_sao_luma_flag
            1, // slice_sao_chroma_flag
            0, // num_ref_idx_active_override
            0, // mvd_l1_zero
            0, // cabac_init
            0, // slice_deblocking_filter_disabled
            0, // slice_loop_filter_across_slices (FF: 0)
            slice_tmvp, // collocated_from_l0 (FF: 1 for inter slices with tmvp)
        );
        let slice_qp_delta = match job.role {
            Role::B => self.b_qp_delta(),
            _ => self.idr_anchor_qp_delta(),
        };
        let slice_param = EncSliceParameterBufferHEVC::new(
            0, // slice_segment_address
            nctu, // num_ctu_in_slice
            match job.role {
                Role::Idr => 2, // slice_type (raw HEVC syntax: I=2)
                _ => 0,         // B=0
            },
            0, // slice_pic_parameter_set_id
            0, // num_ref_idx_l0_active_minus1
            0, // num_ref_idx_l1_active_minus1
            rpl0, // ref_pic_list0
            rpl1, // ref_pic_list1
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
            5, // max_num_merge_cand (five_minus_max_num_merge_cand = 0)
            slice_qp_delta as i8,
            0, // slice_cb_qp_offset
            0, // slice_cr_qp_offset
            0, // slice_beta_offset_div2
            0, // slice_tc_offset_div2
            &slice_fields,
            0, // pred_weight_table_bit_offset
            0, // pred_weight_table_bit_length
        );

        // Packed slice header (byte-exact vs the FF ground truth).
        let poc_lsb = poc as u32;
        let kind = match job.role {
            Role::Idr => h265::SliceKind::Idr,
            Role::Anchor => h265::SliceKind::Inter(h265::InterSlice {
                nal_type: 1, // TRAIL_N
                poc_lsb,
                neg_delta_minus1: (poc - ref_poc(l0.expect("anchor has an L0 ref"))) as u32 - 1,
                pos_delta_minus1: None,
                qp_delta: self.idr_anchor_qp_delta(),
            }),
            Role::B => h265::SliceKind::Inter(h265::InterSlice {
                nal_type: 0, // TRAIL_R
                poc_lsb,
                neg_delta_minus1: (poc - ref_poc(l0.expect("B has an L0 ref"))) as u32 - 1,
                pos_delta_minus1: Some((ref_poc(l1.expect("B has an L1 ref")) - poc) as u32 - 1),
                qp_delta: self.b_qp_delta(),
            }),
        };
        let (slh_bytes, slh_bits) = h265::make_slice_hdr(&kind);
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

        // Zero implicit alignment padding (bytes 506-507, between nal_unit_type
        // and the 4-aligned pic_fields union) to match FF's zeroed buffer.
        unsafe {
            let base = &pic_param as *const _ as *mut u8;
            std::ptr::write_bytes(base.add(506), 0, 2);
        }
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

        // Render order mirrors FF's captured attach order:
        // [seq, framerate] (IDR only) -> picture params -> [packed seq]
        // (IDR only) -> packed slice header -> slice params.
        let mut picture = Picture::new(
            pf.timestamp as u64,
            Rc::clone(&self.context),
            &self.surfaces[idx],
        );
        if matches!(job.role, Role::Idr) {
            picture.add_buffer(self.h265_seq_buffer());
            picture.add_buffer(self.frame_rate_buffer());
        }
        picture.add_buffer(pic_buf);
        if matches!(job.role, Role::Idr) {
            let (seq_hdr, seq_data) = self.seq_packed_buffers();
            picture.add_buffer(seq_hdr);
            picture.add_buffer(seq_data);
        }
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

        Self::read_coded(&coded)
    }

    /// Total CTU count for the fixed 64px CTB grid (matches the SPS).
    fn num_ctus(&self) -> u32 {
        self.width.div_ceil(64) * self.height.div_ceil(64)
    }

    /// HEVC sequence parameter buffer (static contents; recreated per picture).
    /// Values mirror FF's captured buffers; the packed SPS is authoritative
    /// for the bitstream, these structured fields are driver hints.
    fn h265_seq_buffer(&self) -> Buffer {
        let seq_fields = HEVCEncSeqFields::new(
            1, // chroma_format_idc (4:2:0)
            0, // separate_colour_plane
            0, // bit_depth_luma_minus8
            0, // bit_depth_chroma_minus8
            0, // scaling_list_enabled
            0, // strong_intra_smoothing
            1, // amp_enabled (iHD fixed config)
            1, // sample_adaptive_offset (iHD fixed config)
            0, // pcm_enabled
            0, // pcm_loop_filter_disabled
            1, // sps_temporal_mvp (matches our SPS tmvp=1; FF sends 1)
            0, // low_delay_seq
            0, // hierarchical_flag
        );
        let seq = EncSequenceParameterBufferHEVC::new(
            1, // general_profile_idc (Main)
            60, // general_level_idc
            0, // general_tier_flag
            120, // intra_period (matches FF hevc_vaapi gopsize)
            120, // intra_idr_period
            3, // ip_period (b_per_p + 1)
            0, // bits_per_second (CQP: no rate control)
            self.width as u16,
            self.height as u16,
            &seq_fields,
            0, // log2_min_luma_coding_block_size_minus3
            3, // log2_diff_max_min_luma_coding_block_size (CTU 64)
            0, // log2_min_transform_block_size_minus2
            3, // log2_diff_max_min_transform_block_size (TU 32)
            2, // max_transform_hierarchy_depth_inter
            2, // max_transform_hierarchy_depth_intra
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

    /// Misc FrameRate buffer (FF's `den << 16 | num` packing), attached on
    /// IDR pictures only.
    fn frame_rate_buffer(&self) -> Buffer {
        let framerate = (self.fps_den << 16) | self.fps_num;
        let fr = EncMiscParameterFrameRate::new(framerate, 0);
        self.context
            .create_buffer(BufferType::EncMiscParameter(
                EncMiscParameter::FrameRate(fr),
            ))
            .expect("framerate param buffer")
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


    /// H.264 path: 1:1 frame encoding (no B-frame reordering).
    fn encode_h264_frame(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Error> {
        let idx = (self.frame_index as usize) % self.surfaces.len();
        let is_idr = self.frame_index == 0 || self.frame_index.is_multiple_of(self.intra_period);
        let poc_lsb = (self.frame_index & 0xFF) as u16;

        // 1. Upload input pixels.
        self.upload_frame(&self.surfaces[idx], frame)?;

        // 2. Coded buffer + per-frame buffers + render, then read back.
        let coded = self
            .context
            .create_enc_coded((self.width * self.height * 2) as usize)
            .map_err(|e| Error::VaApi(format!("create coded buffer: {}", e)))?;
        let data = self.encode_h264(idx, &coded, is_idr, poc_lsb, timestamp)?;

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

    /// HEVC GPB path (FF-mirrored): IDR + [anchor + 2×B] mini-GOPs. Input
    /// frames are buffered until a mini-GOP is complete; output arrives in
    /// decode order with DTS reordering (0, 3, 1, 2, 6, 4, 5, ...).
    fn encode_h265_gop(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Error> {
        let d = self.frame_index;
        let is_idr = d == 0 || (self.intra_period >= 1 && d.is_multiple_of(self.intra_period));

        if is_idr {
            // Drain any pending tail frames (between the previous anchor and
            // this IDR) before starting the new GOP.
            let mut out = self.drain_pending_h265()?;
            let pf = PendingFrame::new(d, timestamp, frame);
            self.last_idr_frame_index = d;
            let job = FrameJob {
                role: Role::Idr,
                l0_ref: None,
                l1_ref: None,
            };
            let data = self.encode_one(&pf, &job)?;
            out.push(EncodedFrame {
                frame_index: d,
                timestamp,
                dts: timestamp,
                data,
                is_idr: true,
                kind: FrameKind::I,
                poc: 0,
            });
            self.last_ref_display = d;
            self.frame_index += 1;
            return Ok(out);
        }

        self.pending.push(PendingFrame::new(d, timestamp, frame));
        self.frame_index += 1;
        if self.pending.len() == 3 {
            self.encode_mini_gop()
        } else {
            Ok(Vec::new())
        }
    }

    /// Encode a complete [a, b, c] mini-GOP in decode order: anchor(c), B(a),
    /// B(b). `c` is an anchor (B-slice with a single L0 reference to the
    /// previous anchor/IDR); `a`/`b` reference the previous anchor (L0) and
    /// `c` (L1).
    fn encode_mini_gop(&mut self) -> Result<Vec<EncodedFrame>, Error> {
        let frames = std::mem::take(&mut self.pending);
        debug_assert_eq!(frames.len(), 3);
        let (a, b, c) = (
            frames[0].frame_index,
            frames[1].frame_index,
            frames[2].frame_index,
        );
        let prev = self.last_ref_display;
        let poc_of = |d: u32| (d - self.last_idr_frame_index) as i32;

        // Decode order: anchor(c), B(a), B(b).
        let jobs = [
            (
                &frames[2],
                FrameJob {
                    role: Role::Anchor,
                    l0_ref: Some(prev),
                    l1_ref: None,
                },
            ),
            (
                &frames[0],
                FrameJob {
                    role: Role::B,
                    l0_ref: Some(prev),
                    l1_ref: Some(c),
                },
            ),
            (
                &frames[1],
                FrameJob {
                    role: Role::B,
                    l0_ref: Some(prev),
                    l1_ref: Some(c),
                },
            ),
        ];
        let mut out = Vec::with_capacity(3);
        for (pf, job) in &jobs {
            out.push(self.encode_one(pf, job)?);
        }
        self.last_ref_display = c;

        // Decode order was anchor(c), B(a), B(b).
        let data_b = out.pop().expect("3 coded frames");
        let data_a = out.pop().expect("3 coded frames");
        let data_c = out.pop().expect("3 coded frames");

        Ok(vec![
            EncodedFrame {
                frame_index: c,
                timestamp: frames[2].timestamp,
                dts: frames[0].timestamp, // earliest pending display time
                data: data_c,
                is_idr: false,
                kind: FrameKind::P,
                poc: poc_of(c),
            },
            EncodedFrame {
                frame_index: a,
                timestamp: frames[0].timestamp,
                dts: frames[0].timestamp,
                data: data_a,
                is_idr: false,
                kind: FrameKind::B,
                poc: poc_of(a),
            },
            EncodedFrame {
                frame_index: b,
                timestamp: frames[1].timestamp,
                dts: frames[1].timestamp,
                data: data_b,
                is_idr: false,
                kind: FrameKind::B,
                poc: poc_of(b),
            },
        ])
    }

    /// Encode any buffered tail frames (1 or 2) before an IDR / at flush. One
    /// frame becomes an anchor-style reference picture; two become [B, anchor]
    /// in decode order.
    fn drain_pending_h265(&mut self) -> Result<Vec<EncodedFrame>, Error> {
        let frames = std::mem::take(&mut self.pending);
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let prev = self.last_ref_display;
        let poc_of = |d: u32| (d - self.last_idr_frame_index) as i32;

        match frames.len() {
            1 => {
                let f = &frames[0];
                let job = FrameJob {
                    role: Role::Anchor,
                    l0_ref: Some(prev),
                    l1_ref: None,
                };
                let data = self.encode_one(f, &job)?;
                self.last_ref_display = f.frame_index;
                Ok(vec![EncodedFrame {
                    frame_index: f.frame_index,
                    timestamp: f.timestamp,
                    dts: f.timestamp,
                    data,
                    is_idr: false,
                    kind: FrameKind::P,
                    poc: poc_of(f.frame_index),
                }])
            }
            2 => {
                let (a, b) = (&frames[0], &frames[1]);
                // Decode order: anchor(b), B(a).
                let job_b = FrameJob {
                    role: Role::Anchor,
                    l0_ref: Some(prev),
                    l1_ref: None,
                };
                let data_b = self.encode_one(b, &job_b)?;
                let job_a = FrameJob {
                    role: Role::B,
                    l0_ref: Some(prev),
                    l1_ref: Some(b.frame_index),
                };
                let data_a = self.encode_one(a, &job_a)?;
                self.last_ref_display = b.frame_index;
                Ok(vec![
                    EncodedFrame {
                        frame_index: b.frame_index,
                        timestamp: b.timestamp,
                        dts: a.timestamp,
                        data: data_b,
                        is_idr: false,
                        kind: FrameKind::P,
                        poc: poc_of(b.frame_index),
                    },
                    EncodedFrame {
                        frame_index: a.frame_index,
                        timestamp: a.timestamp,
                        dts: a.timestamp,
                        data: data_a,
                        is_idr: false,
                        kind: FrameKind::B,
                        poc: poc_of(a.frame_index),
                    },
                ])
            }
            _ => unreachable!("pending holds at most 2 frames before an IDR"),
        }
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
        match self.codec {
            VideoCodec::EncodeH265 => self.encode_h265_gop(frame, timestamp),
            _ => self.encode_h264_frame(frame, timestamp),
        }
    }

    /// HEVC: flush emits the pending tail (display order may lag decode order).
    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error> {
        match self.codec {
            VideoCodec::EncodeH265 => self.drain_pending_h265(),
            _ => Ok(Vec::new()), // Synchronous backend: nothing in flight.
        }
    }
}
