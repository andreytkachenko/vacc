//! Software H.265 (HEVC) decoder backend.
//!
//! Architecture:
//! - Bitstream parsing (NAL/AU splitting, SPS/PPS/slice headers), POC
//!   computation and DPB management run in Rust on the shared vacc-parser
//!   (`H265Parser`, `H265Dpb`) — the same state machines the other backends
//!   use.
//! - Pixel reconstruction (CABAC, intra/inter prediction, transform,
//!   deblocking, SAO) runs in the hevc.js core via FFI (`ffi` /
//!   `hevc_driver`). The C++ core keeps its own DPB of fully decoded pictures
//!   (pixels + motion metadata) and resolves references from the bitstream;
//!   each decoded picture is copied into a Rust-owned padded buffer.
//! - Output is reordered to display order: a frame is emitted once
//!   `max_num_reorder_pics` (SPS) or more pictures have been decoded after it
//!   (spec 7.4.6 bounds the decode/display delay).

use std::collections::BTreeMap;
use std::os::raw::c_int;
use std::ptr::NonNull;

use vacc_core::codec::VideoCodec;
use vacc_core::decoder::{Decoder, DecoderInfo};
use vacc_core::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use vacc_core::frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane};
use vacc_core::picture::H265Sps;
use vacc_core::session::Extent2D;

use vacc_parser::h265::H265Parser;
use vacc_parser::h265_dpb::H265Dpb;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceHeader, VideoParser};

use crate::error::{Error, Result};
use crate::ffi;

/// Safety cap for the Rust-side DPB slot count.
const MAX_DPB_SLOTS: usize = 64;

/// Pixel planes of a decoded picture (Rust-owned, padded buffers; strides in
/// samples, 16-byte tails).
#[derive(Debug, Clone)]
struct Planes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// Decoded but not yet emitted frame (display-order reorder buffer entry).
/// Keyed in `reorder` by (unwrapped POC, decode sequence).
///
/// Owns its pixel planes: DPB slots are recycled as soon as a picture is no
/// longer a reference, which can happen long before this frame reaches the
/// front of the display queue.
#[derive(Debug, Clone)]
struct BufferedFrame {
    poc: i32, // raw POC
    seq: i32, // decode-order sequence number
    is_ref: bool,
    planes: Planes,
}

/// Software H.265 decoder (CPU reconstruction via the hevc.js core).
pub struct SoftwareH265Decoder {
    ctx: NonNull<ffi::hevcdec_context>,
    parser: H265Parser,
    dpb: Option<H265Dpb>,

    sps: Option<H265Sps>,

    // Picture layout (valid once the SPS has been seen).
    layout_ready: bool,
    coded_w: u32,
    coded_h: u32,
    conf_left: u32,
    conf_right: u32,
    conf_top: u32,
    conf_bottom: u32,
    display_w: u32,
    display_h: u32,
    chroma_idc: u8,
    bps: u8,        // bytes per sample (1 or 2)
    ystride: usize, // samples
    cstride: usize, // samples
    chroma_w: u32,
    chroma_h: u32,

    // Pending input.
    pending_data: Vec<u8>,
    parse_offset: usize,
    /// Parameter-set NALs are buffered in `pending_data` until the first
    /// slice of a picture arrives, so the C++ core receives them raw.
    ps_pending: bool,

    // Display-order reorder state.
    reorder: BTreeMap<(i32, i32), BufferedFrame>, // (uw_poc, seq) -> frame
    poc_period: i32,
    poc_cycle: i32,
    prev_decoded_raw_poc: Option<i32>,
    max_seq: i32,
    max_reorder: u32,
    seq_counter: i32,
    frame_index: u32,
}

impl SoftwareH265Decoder {
    /// Scan Annex B data for H.265 VPS/SPS NAL units (codec detection).
    fn detect_h265(data: &[u8]) -> bool {
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                // NAL header byte follows the 3-byte start code.
                let nal_type = (data[i + 3] >> 1) & 0x3f;
                if nal_type == 32 || nal_type == 33 {
                    return true; // VPS or SPS
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        false
    }

    fn chroma_factors(chroma_idc: u8) -> (u32, u32) {
        match chroma_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }

    /// Apply a newly parsed SPS: set up the picture layout and (re)create the
    /// Rust DPB.
    fn on_sps(&mut self, sps: &H265Sps) {
        let changed = self
            .sps
            .as_ref()
            .map(|old| {
                old.pic_width_in_luma_samples != sps.pic_width_in_luma_samples
                    || old.pic_height_in_luma_samples != sps.pic_height_in_luma_samples
                    || old.chroma_format_idc != sps.chroma_format_idc
                    || 8 + old.bit_depth_luma_minus8 != 8 + sps.bit_depth_luma_minus8
            })
            .unwrap_or(false);

        if changed && self.layout_ready {
            // Resolution/format change: wipe all state (C++ core re-parses its
            // own SPS cache from the bitstream).
            unsafe { ffi::hevcdec_reset(self.ctx.as_ptr()) };
            if let Some(dpb) = self.dpb.as_mut() {
                dpb.invalidate_all();
                dpb.clear_display_pending();
            }
            self.reorder.clear();
            self.prev_decoded_raw_poc = None;
            self.poc_cycle = 0;
            self.max_seq = 0;
            self.seq_counter = 0;
        }

        let (sw, sh) = Self::chroma_factors(sps.chroma_format_idc);
        let w = sps.pic_width_in_luma_samples as u32;
        let h = sps.pic_height_in_luma_samples as u32;
        let conf_left = sps.conf_win_left_offset * sw;
        let conf_right = sps.conf_win_right_offset * sw;
        let conf_top = sps.conf_win_top_offset * sh;
        let conf_bottom = sps.conf_win_bottom_offset * sh;

        self.coded_w = w;
        self.coded_h = h;
        self.conf_left = conf_left;
        self.conf_right = conf_right;
        self.conf_top = conf_top;
        self.conf_bottom = conf_bottom;
        self.display_w = w.saturating_sub(conf_left + conf_right);
        self.display_h = h.saturating_sub(conf_top + conf_bottom);
        self.chroma_idc = sps.chroma_format_idc;
        let bd = 8 + sps.bit_depth_luma_minus8 as u32;
        self.bps = if bd > 8 { 2 } else { 1 };
        self.ystride = (w as usize).div_ceil(16) * 16;
        let cw = w.div_ceil(sw);
        let ch = h.div_ceil(sh);
        self.chroma_w = cw;
        self.chroma_h = ch;
        self.cstride = (cw as usize).div_ceil(16) * 16;

        self.max_reorder = sps.max_num_reorder_pics[0] as u32;
        self.poc_period = 1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

        let num_slots = (1 + sps.max_dec_pic_buffering_minus1[0] as usize).clamp(4, MAX_DPB_SLOTS);
        if self.dpb.is_none() || changed {
            let mut dpb = H265Dpb::new(num_slots);
            dpb.set_max_num_reorder_frames(self.max_reorder);
            self.dpb = Some(dpb);
        }

        self.sps = Some(sps.clone());
        self.layout_ready = true;
    }

    /// Allocate the padded output planes for one coded picture.
    fn alloc_planes(&self) -> Planes {
        let y_len = self.coded_h as usize * self.ystride * self.bps as usize + 16;
        let c_len = self.chroma_h as usize * self.cstride * self.bps as usize + 16;
        Planes {
            y: vec![0u8; y_len],
            u: vec![0u8; c_len],
            v: vec![0u8; c_len],
        }
    }

    /// Unwrap the raw POC into a monotonic value across POC periods.
    fn unwrap_poc(&mut self, poc: i32) -> i32 {
        let period = self.poc_period;
        if period > 1
            && let Some(prev) = self.prev_decoded_raw_poc
        {
            if poc < prev - period / 2 {
                self.poc_cycle += 1;
            } else if poc > prev + period / 2 {
                self.poc_cycle -= 1;
            }
        }
        self.prev_decoded_raw_poc = Some(poc);
        poc + self.poc_cycle * period
    }

    /// Decode one picture: stage it in the Rust DPB, inject the reference
    /// planes into the C++ core, run reconstruction, store the result.
    fn decode_picture(
        &mut self,
        au: &[u8],
        first_info: &vacc_parser::h265::SliceHeaderInfo,
    ) -> Result<()> {
        let sps = self
            .sps
            .clone()
            .ok_or_else(|| Error::InvalidState("SPS not available".to_string()))?;
        if !self.layout_ready {
            return Err(Error::InvalidState("picture layout not ready".to_string()));
        }

        // --- Stage the current picture in the Rust DPB (spec 8.3.2) ---
        let slot = {
            let dpb = self
                .dpb
                .as_mut()
                .ok_or_else(|| Error::InvalidState("DPB not initialized".to_string()))?;
            dpb.picture_start(&sps, first_info, first_info.is_reference)
        };

        // --- Run the C++ reconstruction core on the access unit ---
        // Reference resolution happens inside the C++ core (its own DPB of
        // fully decoded pictures, matched by POC from the bitstream).
        let mut planes = self.alloc_planes();
        let rc = unsafe {
            ffi::hevcdec_decode_picture(
                self.ctx.as_ptr(),
                au.as_ptr(),
                au.len(),
                planes.y.as_mut_ptr(),
                self.ystride as c_int,
                planes.u.as_mut_ptr(),
                self.cstride as c_int,
                planes.v.as_mut_ptr(),
                self.cstride as c_int,
            )
        };
        if rc != ffi::HEVCDEC_OK {
            return Err(Error::Core {
                code: rc,
                msg: "hevcdec_decode_picture failed".to_string(),
            });
        }

        // Cross-check the C++ core's POC computation against the Rust side.
        let c_poc = unsafe { ffi::hevcdec_last_pic_poc(self.ctx.as_ptr()) };
        if c_poc != first_info.curr_pic_order_cnt_val {
            log::warn!(
                "POC mismatch: rust={} cpp={}",
                first_info.curr_pic_order_cnt_val,
                c_poc
            );
        }

        // --- Commit to the Rust DPB (display state machine) ---
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.commit_current(slot);
        }

        // --- Track for display-order presentation ---
        self.seq_counter += 1;
        let seq = self.seq_counter;
        self.max_seq = self.max_seq.max(seq);
        if first_info.pic_output_flag {
            let uw = self.unwrap_poc(first_info.curr_pic_order_cnt_val);
            self.reorder.insert(
                (uw, seq),
                BufferedFrame {
                    poc: first_info.curr_pic_order_cnt_val,
                    seq,
                    is_ref: first_info.is_reference,
                    planes,
                },
            );
        }

        Ok(())
    }

    /// Parse pending data and decode the next picture, if a complete access
    /// unit is available. Returns `Ok(true)` when a picture was decoded.
    fn decode_next_picture(&mut self) -> Result<bool> {
        if self.parse_offset >= self.pending_data.len() {
            return Ok(false);
        }

        let remaining = &self.pending_data[self.parse_offset..];
        let packet = BitstreamPacket::new(remaining.to_vec());

        loop {
            match self.parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, .. }) => {
                    if let Some(b) = sps
                        && let Some(s) = b.downcast_ref::<H265Sps>()
                    {
                        self.on_sps(s);
                    }
                    // Raw PS NALs stay in the pending region: they are fed to
                    // the C++ core together with the first slice of the next
                    // picture (bytes_consumed covers them).
                    self.ps_pending = true;
                    continue;
                }
                Ok(ParseResult::Slice {
                    slices,
                    bytes_consumed,
                }) => {
                    if slices.is_empty() {
                        return Ok(false);
                    }
                    let first_info = match &slices[0].slice_header {
                        Some(SliceHeader::H265(i)) => i,
                        _ => return Ok(false),
                    };

                    let au_end = self.parse_offset + bytes_consumed;
                    if au_end > self.pending_data.len() {
                        return Err(Error::Parser(
                            "bytes_consumed exceeds pending data".to_string(),
                        ));
                    }
                    let au = self.pending_data[self.parse_offset..au_end].to_vec();
                    self.parse_offset = au_end;
                    self.ps_pending = false;

                    // Clone the header info out (parser borrow ends here).
                    let info = first_info.clone();
                    self.decode_picture(&au, &info)?;
                    return Ok(true);
                }
                Ok(ParseResult::Nothing) => {
                    if self.ps_pending {
                        // PS NALs buffered but no slice yet: wait for more data.
                        return Ok(false);
                    }
                    self.parse_offset = self.pending_data.len();
                    return Ok(false);
                }
                Ok(ParseResult::EndOfStream) => {
                    self.parse_offset = self.pending_data.len();
                    return Ok(false);
                }
                Err(e) => return Err(Error::Parser(e.to_string())),
            }
        }
    }

    /// Build a `DecodedFrame` for the given reorder entry (crops to display
    /// size, packs planes into one owned buffer).
    fn build_frame(&mut self, key: &(i32, i32)) -> Result<DecodedFrame> {
        let bf = match self.reorder.remove(key) {
            Some(bf) => bf,
            None => return Err(Error::InvalidState("reorder entry vanished".to_string())),
        };

        let planes = &bf.planes;

        let bps = self.bps as usize;
        let (sw, sh) = Self::chroma_factors(self.chroma_idc);
        let cw = self.display_w as usize;
        let ch = self.display_h as usize;
        let x0 = self.conf_left as usize / sw.max(1) as usize;
        let y0 = self.conf_top as usize / sh.max(1) as usize;
        let cwidth = cw.div_ceil(sw as usize);
        let cheight = ch.div_ceil(sh as usize);

        let y_len = cw * ch * bps;
        let c_len = cwidth * cheight * bps;
        let mut buf = Vec::with_capacity(y_len + 2 * c_len);

        // Luma (crop to display window).
        for y in 0..ch {
            let src = &planes.y[((y0 + y) * self.ystride + x0) * bps..];
            buf.extend_from_slice(&src[..cw * bps]);
        }
        // Chroma.
        if self.chroma_idc != 0 {
            for plane in [planes.u.as_slice(), planes.v.as_slice()] {
                for y in 0..cheight {
                    let row = &plane[((y0 + y) * self.cstride + x0) * bps..];
                    buf.extend_from_slice(&row[..cwidth * bps]);
                }
            }
        }

        let y_ptr = buf.as_ptr();
        let u_ptr = unsafe { buf.as_ptr().add(y_len) };
        let v_ptr = unsafe { buf.as_ptr().add(y_len + c_len) };

        // The slot may have been recycled since decode; only mark displayed
        // if a live slot still holds this POC.
        if let Some(dpb) = self.dpb.as_mut()
            && let Some(i) = dpb.slots().iter().position(|s| s.valid && s.poc == bf.poc)
        {
            dpb.mark_displayed(i);
        }

        let frame_idx = self.frame_index;
        self.frame_index += 1;

        let pixel_data = PixelData {
            format: if self.bps > 1 { "I420-10" } else { "I420" }.to_string(),
            y: PixelPlane {
                data: y_ptr,
                pitch: cw * bps,
                width: cw,
                height: ch,
            },
            u: PixelPlane {
                data: u_ptr,
                pitch: cwidth * bps,
                width: cwidth,
                height: cheight,
            },
            v: Some(PixelPlane {
                data: v_ptr,
                pitch: cwidth * bps,
                width: cwidth,
                height: cheight,
            }),
            buffer: buf,
        };

        let is_ref = bf.is_ref;

        Ok(DecodedFrame {
            frame_index: frame_idx,
            timestamp: frame_idx as i64 * 33_333,
            width: self.display_w,
            height: self.display_h,
            skipped: false,
            pts_valid: false,
            poc: bf.poc,
            field_flags: FieldFlags {
                progressive_frame: true,
                ref_pic: is_ref,
                ..Default::default()
            },
            sync_info: Default::default(),
            pixel_data: Some(pixel_data),
        })
    }

    /// Emit the display-ready frame at the front of the reorder buffer, if
    /// any. A frame is ready once `max_reorder` or more pictures have been
    /// decoded after it (spec 7.4.6).
    fn emit_ready(&mut self) -> Option<DecodedFrame> {
        let front = self.reorder.iter().next()?;
        let ready = self.max_seq - front.1.seq >= self.max_reorder as i32;
        if !ready {
            return None;
        }
        // `?` inside a closure won't work with the borrow; build inline.
        let key = *front.0;
        match self.build_frame(&key) {
            Ok(f) => Some(f),
            Err(e) => {
                log::error!("failed to build frame: {e}");
                None
            }
        }
    }

    fn reset_state(&mut self) {
        unsafe { ffi::hevcdec_reset(self.ctx.as_ptr()) };
        self.parser.reset();
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.invalidate_all();
            dpb.clear_display_pending();
        }
        self.reorder.clear();
        self.prev_decoded_raw_poc = None;
        self.poc_cycle = 0;
        self.max_seq = 0;
        self.seq_counter = 0;
        self.ps_pending = false;
        self.parse_offset = self.pending_data.len();
    }
}

impl Decoder for SoftwareH265Decoder {
    type Error = Error;

    fn new(data: Vec<u8>) -> Result<Self> {
        if !Self::detect_h265(&data) {
            return Err(Error::DecoderInit(
                "unsupported bitstream: expected H.265/HEVC (VPS/SPS not found)".to_string(),
            ));
        }

        let ptr = unsafe { ffi::hevcdec_create(software_threads()) };
        let ctx = NonNull::new(ptr).ok_or_else(|| {
            Error::DecoderInit("failed to create C++ decoder context".to_string())
        })?;

        let mut parser = H265Parser::new();
        parser
            .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
            .map_err(|e| Error::Parser(e.to_string()))?;

        Ok(Self {
            ctx,
            parser,
            dpb: None,
            sps: None,
            layout_ready: false,
            coded_w: 0,
            coded_h: 0,
            conf_left: 0,
            conf_right: 0,
            conf_top: 0,
            conf_bottom: 0,
            display_w: 0,
            display_h: 0,
            chroma_idc: 1,
            bps: 1,
            ystride: 0,
            cstride: 0,
            chroma_w: 0,
            chroma_h: 0,
            pending_data: data,
            parse_offset: 0,
            ps_pending: false,
            reorder: BTreeMap::new(),
            poc_period: 1,
            poc_cycle: 0,
            prev_decoded_raw_poc: None,
            max_seq: 0,
            max_reorder: 0,
            seq_counter: 0,
            frame_index: 0,
        })
    }

    fn new_with_format(_data: Vec<u8>, _codec: VideoCodec, _format: &VideoFormat) -> Result<Self> {
        Err(Error::DecoderInit(
            "new_with_format not yet implemented".to_string(),
        ))
    }

    fn info(&self) -> DecoderInfo {
        let (chroma_subsampling, luma_bit_depth, chroma_bit_depth, profile_idc) =
            match self.sps.as_ref() {
                Some(sps) => {
                    let chroma = match sps.chroma_format_idc {
                        0 => ChromaSubsampling::Monochrome,
                        1 => ChromaSubsampling::_420,
                        2 => ChromaSubsampling::_422,
                        3 => ChromaSubsampling::_444,
                        _ => ChromaSubsampling::_420,
                    };
                    let luma = match 8 + sps.bit_depth_luma_minus8 {
                        8 => ComponentBitDepth::Bit8,
                        10 => ComponentBitDepth::Bit10,
                        12 => ComponentBitDepth::Bit12,
                        _ => ComponentBitDepth::Bit8,
                    };
                    let chroma_bd = match 8 + sps.bit_depth_chroma_minus8 {
                        8 => ComponentBitDepth::Bit8,
                        10 => ComponentBitDepth::Bit10,
                        12 => ComponentBitDepth::Bit12,
                        _ => ComponentBitDepth::Bit8,
                    };
                    (chroma, luma, chroma_bd, Some(sps.profile_idc as u32))
                }
                None => (
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit8,
                    ComponentBitDepth::Bit8,
                    None,
                ),
            };

        DecoderInfo {
            backend: "software".to_string(),
            codec: VideoCodec::DecodeH265,
            coded_size: Extent2D::new(self.coded_w, self.coded_h),
            display_size: Extent2D::new(self.display_w, self.display_h),
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            profile_idc,
            dpb_slots: self
                .dpb
                .as_ref()
                .map(|d| d.slots().len() as u32)
                .unwrap_or(0),
        }
    }

    fn submit(&mut self, data: &[u8]) -> Result<()> {
        if self.parse_offset >= self.pending_data.len() {
            self.pending_data.clear();
            self.parse_offset = 0;
        } else {
            let unconsumed = self.pending_data[self.parse_offset..].to_vec();
            self.pending_data = unconsumed;
            self.parse_offset = 0;
        }
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> Result<Option<DecodedFrame>> {
        loop {
            if let Some(frame) = self.emit_ready() {
                return Ok(Some(frame));
            }
            match self.decode_next_picture()? {
                true => continue, // decoded a picture; try emitting again
                false => return Ok(None),
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        // Drain the reorder buffer in display (POC) order.
        while let Some((&key, _)) = self.reorder.iter().next() {
            out.push(self.build_frame(&key)?);
        }
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.clear_display_pending();
        }
        Ok(out)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset_state();
        Ok(())
    }
}

impl Drop for SoftwareH265Decoder {
    fn drop(&mut self) {
        unsafe { ffi::hevcdec_destroy(self.ctx.as_ptr()) };
    }
}

fn software_threads() -> i32 {
    // VACC_SW_THREADS overrides the worker thread count (0 = sequential).
    if let Ok(v) = std::env::var("VACC_SW_THREADS")
        && let Ok(n) = v.parse::<i32>()
    {
        return n.max(0);
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(1, 8)
}
