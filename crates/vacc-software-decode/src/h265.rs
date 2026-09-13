//! Software H.265 (HEVC) decoder backend.
//!
//! Architecture:
//! - Bitstream parsing (NAL/AU splitting, SPS/PPS/slice headers), POC
//!   computation and DPB management run in Rust on the shared vacc-parser
//!   (`H265Parser`, `H265Dpb`) — the same state machines the other backends
//!   use.
//! - Pixel reconstruction (CABAC, intra/inter prediction, transform,
//!   deblocking, SAO) runs in the Rust `hevc` port (`driver::decode_picture`).
//!   The `H265Dpb` resolves reference lists by POC/RPS; decoded pictures are
//!   kept in a `PictureStore` (indexed by DPB slot) so later pictures can use
//!   them as references, and each is copied into a Rust-owned padded buffer.
//! - Output is reordered to display order: a frame is emitted once
//!   `max_num_reorder_pics` (SPS) or more pictures have been decoded after it
//!   (spec 7.4.6 bounds the decode/display delay).

use std::collections::BTreeMap;

use vacc_core::codec::VideoCodec;
use vacc_core::decoder::{Decoder, DecoderInfo};
use vacc_core::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use vacc_core::frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane};
use vacc_core::picture::{H265Pps, H265Sps};
use vacc_core::session::Extent2D;

use vacc_parser::h265::H265Parser;
use vacc_parser::h265_dpb::H265Dpb;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceEntry, SliceHeader, VideoParser};

use crate::error::{Error, Result};
use crate::hevc::driver::{self, PictureStore, RefListEntry, SliceInput};
use crate::hevc::picture::Picture;
use crate::hevc::syntax_map;
use crate::hevc::transform::ScalingListData;

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

/// Software H.265 decoder (CPU reconstruction via the Rust `hevc` port).
pub struct SoftwareH265Decoder {
    parser: H265Parser,
    dpb: Option<H265Dpb>,
    /// Decoded pictures indexed by DPB slot, for reference resolution.
    store: PictureStore,

    sps: Option<H265Sps>,
    pps: Option<H265Pps>,

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
            // Resolution/format change: wipe all decoded state.
            self.store = PictureStore::new(MAX_DPB_SLOTS);
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

    /// Decode one picture: stage it in the Rust DPB, resolve reference lists,
    /// run the Rust reconstruction core, and track the result for display-order
    /// presentation.
    fn decode_picture(
        &mut self,
        slices: &[SliceEntry],
        first_info: &vacc_parser::h265::SliceHeaderInfo,
    ) -> Result<()> {
        let sps_h = self
            .sps
            .clone()
            .ok_or_else(|| Error::InvalidState("SPS not available".to_string()))?;
        let pps_h = self
            .pps
            .clone()
            .ok_or_else(|| Error::InvalidState("PPS not available".to_string()))?;
        if !self.layout_ready {
            return Err(Error::InvalidState("picture layout not ready".to_string()));
        }

        // --- Stage in the Rust DPB (spec 8.3.2) + resolve reference lists ---
        let (slot, refs_l0, refs_l1) = {
            let dpb = self
                .dpb
                .as_mut()
                .ok_or_else(|| Error::InvalidState("DPB not initialized".to_string()))?;
            let slot = dpb.picture_start(&sps_h, first_info, first_info.is_reference);
            let lists = dpb.build_ref_lists();
            let entries = |v: &[vacc_parser::h265_dpb::H265RefPic]| -> Vec<RefListEntry> {
                v.iter()
                    .map(|r| RefListEntry {
                        slot: r.slot,
                        poc: r.poc,
                        long_term: if r.slot >= 0 {
                            dpb.slot_is_long_term(r.slot as usize)
                        } else {
                            false
                        },
                    })
                    .collect()
            };
            (slot, entries(&lists.l0), entries(&lists.l1))
        };

        // --- Map parameter sets + build per-segment slice inputs ---
        let sps = syntax_map::map_sps(&sps_h);
        let pps = syntax_map::map_pps(&pps_h, &sps);
        let has_chroma = sps.chroma_array_type != 0;
        let sps_scaling_enabled = sps_h.sps_scaling_list_data_present_flag;
        let sps_scaling = if sps_scaling_enabled {
            syntax_map::map_scaling_list(&sps_h.scaling_lists)
        } else {
            ScalingListData::default()
        };
        let pps_scaling_present = pps_h.pps_scaling_list_data_present_flag;
        let pps_scaling = if pps_scaling_present {
            syntax_map::map_scaling_list(&pps_h.scaling_lists)
        } else {
            ScalingListData::default()
        };

        // Dependent-slice inheritance from the last independent segment.
        let mut last_independent: Option<&vacc_parser::h265::SliceHeaderInfo> = None;
        let mut slice_inputs: Vec<SliceInput> = Vec::new();
        for e in slices {
            let Some(SliceHeader::H265(info)) = &e.slice_header else {
                continue;
            };
            let sh = syntax_map::map_sh(info, last_independent, &pps_h, has_chroma);
            let deblock = syntax_map::map_deblock_params(info, &pps_h);
            slice_inputs.push(SliceInput {
                nal: &e.nal_data,
                sh,
                deblock,
                header_bit_size: info.header_bit_size,
            });
            if !info.dependent_slice_segment_flag {
                last_independent = Some(info);
            }
        }
        if slice_inputs.is_empty() {
            return Err(Error::Parser("no slice segments parsed".to_string()));
        }

        // --- Run the Rust reconstruction core ---
        let pic = driver::decode_picture(
            &sps,
            &pps,
            &slice_inputs,
            &refs_l0,
            &refs_l1,
            &self.store,
            first_info.curr_pic_order_cnt_val,
            sps_scaling_enabled,
            &sps_scaling,
            pps_scaling_present,
            &pps_scaling,
        )
        .map_err(|e| Error::Core { code: -1, msg: e })?;

        // --- Commit to the Rust DPB + store for future reference resolution ---
        let planes = self.picture_to_planes(&pic);
        self.dpb.as_mut().unwrap().commit_current(slot);
        self.store.store(slot, pic);

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

    /// Convert a reconstructed `Picture` (u16 samples, tight strides) into the
    /// padded output `Planes` (bps bytes/sample, 16-aligned strides).
    fn picture_to_planes(&self, pic: &Picture) -> Planes {
        let bps = self.bps as usize;

        let mut y = vec![0u8; self.coded_h as usize * self.ystride * bps + 16];
        let w0 = pic.width[0] as usize;
        if bps == 1 {
            for yy in 0..pic.height[0] {
                let row_s = (yy * pic.stride[0]) as usize;
                let src = &pic.planes[0][row_s..row_s + w0];
                let row_d = (yy as usize) * self.ystride;
                for (d, s) in y[row_d..row_d + w0].iter_mut().zip(src) {
                    *d = *s as u8;
                }
            }
        } else {
            for yy in 0..pic.height[0] {
                let row_s = (yy * pic.stride[0]) as usize;
                let src = &pic.planes[0][row_s..row_s + w0];
                let row_d = (yy as usize) * self.ystride * bps;
                for (xx, s) in src.iter().enumerate() {
                    y[row_d + xx * bps] = *s as u8;
                    y[row_d + xx * bps + 1] = (*s >> 8) as u8;
                }
            }
        }

        let mut u = vec![0u8; self.chroma_h as usize * self.cstride * bps + 16];
        let mut v = vec![0u8; self.chroma_h as usize * self.cstride * bps + 16];
        if self.chroma_idc != 0 {
            let w1 = pic.width[1] as usize;
            if bps == 1 {
                for yy in 0..pic.height[1] {
                    let row_u = (yy * pic.stride[1]) as usize;
                    let row_v = (yy * pic.stride[2]) as usize;
                    let row_d = (yy as usize) * self.cstride;
                    for (d, s) in u[row_d..row_d + w1]
                        .iter_mut()
                        .zip(&pic.planes[1][row_u..row_u + w1])
                    {
                        *d = *s as u8;
                    }
                    for (d, s) in v[row_d..row_d + w1]
                        .iter_mut()
                        .zip(&pic.planes[2][row_v..row_v + w1])
                    {
                        *d = *s as u8;
                    }
                }
            } else {
                for yy in 0..pic.height[1] {
                    let row_u = (yy * pic.stride[1]) as usize;
                    let row_v = (yy * pic.stride[2]) as usize;
                    let row_d = (yy as usize) * self.cstride * bps;
                    for (xx, s) in pic.planes[1][row_u..row_u + w1].iter().enumerate() {
                        u[row_d + xx * bps] = *s as u8;
                        u[row_d + xx * bps + 1] = (*s >> 8) as u8;
                    }
                    for (xx, s) in pic.planes[2][row_v..row_v + w1].iter().enumerate() {
                        v[row_d + xx * bps] = *s as u8;
                        v[row_d + xx * bps + 1] = (*s >> 8) as u8;
                    }
                }
            }
        }

        Planes { y, u, v }
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
                Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                    if let Some(b) = sps
                        && let Some(s) = b.downcast_ref::<H265Sps>()
                    {
                        self.on_sps(s);
                    }
                    if let Some(b) = pps
                        && let Some(p) = b.downcast_ref::<H265Pps>()
                    {
                        self.pps = Some(p.clone());
                    }
                    // PS NALs stay in the pending region until the first slice
                    // of the next picture (bytes_consumed covers them).
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
                    // Clone slices + header info out (parser borrow ends here);
                    // the Rust driver reads each slice's NAL bytes directly.
                    let slices_owned = slices.clone();
                    let info = first_info.clone();
                    self.parse_offset = au_end;
                    self.ps_pending = false;

                    self.decode_picture(&slices_owned, &info)?;
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
        self.store = PictureStore::new(MAX_DPB_SLOTS);
        self.pps = None;
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

        let mut parser = H265Parser::new();
        parser
            .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
            .map_err(|e| Error::Parser(e.to_string()))?;

        Ok(Self {
            parser,
            dpb: None,
            store: PictureStore::new(MAX_DPB_SLOTS),
            sps: None,
            pps: None,
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

#[cfg(test)]
mod e2e_tests {
    //! End-to-end gate for the production pipeline: `SoftwareH265Decoder`
    //! (Rust parser + DPB + Rust reconstruction core + display-order reorder)
    //! vs the C++ `hevcdec_decode_picture` core as ground truth, on real
    //! streams. Frames are compared in display order, matched by POC.

    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    use vacc_core::decoder::Decoder;
    use vacc_core::picture::H265Sps;
    use vacc_parser::h265::H265Parser;
    use vacc_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

    use crate::ffi;
    use super::SoftwareH265Decoder;

    fn sample_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("VACC_SAMPLES_DIR") {
            return Some(PathBuf::from(d));
        }
        let p = PathBuf::from("/home/atkachenko/apps/vacc/assets/samples");
        p.is_dir().then_some(p)
    }

    /// Picture layout derived from the SPS (mirrors `on_sps`).
    #[derive(Clone, Copy)]
    struct Layout {
        coded_w: u32,
        coded_h: u32,
        chroma_h: u32,
        conf_left: u32, // luma samples
        conf_right: u32,
        conf_top: u32,
        conf_bottom: u32,
        sw: u32,
        sh: u32,
        bps: usize,
        ystride: usize,
        cstride: usize,
        chroma_idc: u8,
    }

    fn layout_from_sps(sps: &H265Sps) -> Layout {
        let (sw, sh) = match sps.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let w = sps.pic_width_in_luma_samples as u32;
        let h = sps.pic_height_in_luma_samples as u32;
        let bd = 8 + sps.bit_depth_luma_minus8 as u32;
        Layout {
            coded_w: w,
            coded_h: h,
            chroma_h: h.div_ceil(sh),
            conf_left: sps.conf_win_left_offset * sw,
            conf_right: sps.conf_win_right_offset * sw,
            conf_top: sps.conf_win_top_offset * sh,
            conf_bottom: sps.conf_win_bottom_offset * sh,
            sw,
            sh,
            bps: if bd > 8 { 2 } else { 1 },
            ystride: (w as usize).div_ceil(16) * 16,
            cstride: (w.div_ceil(sw) as usize).div_ceil(16) * 16,
            chroma_idc: sps.chroma_format_idc,
        }
    }

    /// Coded-size ground-truth planes for one output picture.
    struct RefFrame {
        poc: i32,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    }

    /// Decode the whole stream with the C++ core (one context across all AUs,
    /// so its internal DPB evolves exactly as in a real decode) and collect
    /// filtered planes for every AU whose `pic_output_flag` is set.
    fn cpp_ground_truth(name: &str, data: &[u8]) -> (Vec<RefFrame>, Layout) {
        let cpp_ctx = unsafe { ffi::hevcdec_create(0) };
        assert!(!cpp_ctx.is_null(), "{name}: hevcdec_create failed");

        let mut parser = H265Parser::new();
        let mut layout: Option<Layout> = None;
        let mut refs: Vec<RefFrame> = Vec::new();
        let mut parse_offset = 0usize;

        while parse_offset < data.len() {
            let mut got_au = false;
            'au: while parse_offset < data.len() {
                let remaining = &data[parse_offset..];
                let packet = BitstreamPacket::new(remaining.to_vec());
                match parser.parse(&packet) {
                    Ok(ParseResult::ParameterSet { sps, .. }) => {
                        if let Some(b) = sps
                            && let Some(s) = b.downcast_ref::<H265Sps>()
                        {
                            layout = Some(layout_from_sps(s));
                        }
                        // PS NALs stay in the pending region; re-parse for slices.
                        continue 'au;
                    }
                    Ok(ParseResult::Slice {
                        slices,
                        bytes_consumed,
                    }) => {
                        if slices.is_empty() {
                            break 'au;
                        }
                        let au_end = parse_offset + bytes_consumed;
                        assert!(au_end <= data.len(), "{name}: bytes_consumed exceeds data");
                        let au = &data[parse_offset..au_end];
                        let first_info = match &slices[0].slice_header {
                            Some(SliceHeader::H265(i)) => i,
                            _ => panic!("{name}: no H265 slice header"),
                        };
                        let layout = layout
                            .expect("SPS not ready for a picture");
                        let mut out_y = vec![0u8; layout.coded_h as usize * layout.ystride * layout.bps + 16];
                        let mut out_u = vec![0u8; layout.chroma_h as usize * layout.cstride * layout.bps + 16];
                        let mut out_v = vec![0u8; layout.chroma_h as usize * layout.cstride * layout.bps + 16];
                        let rc = unsafe {
                            ffi::hevcdec_decode_picture(
                                cpp_ctx,
                                au.as_ptr(),
                                au.len(),
                                out_y.as_mut_ptr(),
                                layout.ystride as i32,
                                out_u.as_mut_ptr(),
                                layout.cstride as i32,
                                out_v.as_mut_ptr(),
                                layout.cstride as i32,
                            )
                        };
                        assert_eq!(rc, ffi::HEVCDEC_OK, "{name}: C++ decode failed ({rc})");
                        if first_info.pic_output_flag {
                            refs.push(RefFrame {
                                poc: first_info.curr_pic_order_cnt_val,
                                y: out_y,
                                u: out_u,
                                v: out_v,
                            });
                        }
                        parse_offset = au_end;
                        got_au = true;
                        break 'au;
                    }
                    Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break 'au,
                    Err(e) => panic!("{name}: parse error: {e}"),
                }
            }
            if !got_au {
                break;
            }
        }

        unsafe { ffi::hevcdec_destroy(cpp_ctx) };
        (refs, layout.expect("no SPS in stream"))
    }

    /// Pack the coded-size C++ planes into a tight display-window buffer with
    /// the exact layout of `build_frame` (Y, then U, then V; bps bytes per
    /// sample, tight pitches).
    fn pack_display(l: &Layout, rf: &RefFrame) -> Vec<u8> {
        let cw = l.coded_w as usize - l.conf_left as usize - l.conf_right as usize;
        let ch = l.coded_h as usize - l.conf_top as usize - l.conf_bottom as usize;
        let x0 = (l.conf_left / l.sw) as usize;
        let y0 = (l.conf_top / l.sh) as usize;
        let cwidth = cw.div_ceil(l.sw as usize);
        let cheight = ch.div_ceil(l.sh as usize);
        let bps = l.bps;

        let y_len = cw * ch * bps;
        let c_len = cwidth * cheight * bps;
        let mut buf = Vec::with_capacity(y_len + 2 * c_len);

        for y in 0..ch {
            let src = &rf.y[((y0 + y) * l.ystride + x0) * bps..];
            buf.extend_from_slice(&src[..cw * bps]);
        }
        if l.chroma_idc != 0 {
            for plane in [&rf.u, &rf.v] {
                for y in 0..cheight {
                    let row = &plane[((y0 + y) * l.cstride + x0) * bps..];
                    buf.extend_from_slice(&row[..cwidth * bps]);
                }
            }
        }
        buf
    }

    fn first_buf_diff(name: &str, got: &[u8], want: &[u8]) {
        assert_eq!(got.len(), want.len(), "{name}: buffer length {} != {}", got.len(), want.len());
        let n = got.iter().zip(want).take_while(|(a, b)| a == b).count();
        assert!(
            n == got.len(),
            "{name}: first byte diff at offset {n} (got {:02x}, want {:02x})",
            got[n],
            want[n]
        );
    }

    fn run_e2e(name: &str) {
        let dir = match sample_dir() {
            Some(d) => d,
            None => {
                eprintln!("skip {name}: no samples dir (set VACC_SAMPLES_DIR)");
                return;
            }
        };
        let path = dir.join(format!("h265_{name}.h265"));
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                return;
            }
        };

        // ---- Ground truth: C++ core decodes the whole stream ----
        let (refs, layout) = cpp_ground_truth(name, &data);
        let mut truth: BTreeMap<i32, Vec<u8>> = BTreeMap::new();
        for rf in &refs {
            assert!(
                truth.insert(rf.poc, pack_display(&layout, rf)).is_none(),
                "{name}: duplicate POC {}",
                rf.poc
            );
        }

        // ---- Production decoder (same driving pattern as the app) ----
        let mut dec = SoftwareH265Decoder::new(data).expect("decoder init");
        let mut frames = Vec::new();
        while let Some(f) = dec.decode().unwrap() {
            frames.push(f);
        }
        frames.extend(dec.flush().unwrap());

        assert_eq!(
            frames.len(),
            truth.len(),
            "{name}: {} emitted frames vs {} output pictures",
            frames.len(),
            truth.len()
        );

        // Display order: POCs must be strictly increasing.
        for pair in frames.windows(2) {
            assert!(
                pair[0].poc < pair[1].poc,
                "{name}: display-order violation at frame {}: poc {} -> {}",
                pair[1].frame_index,
                pair[0].poc,
                pair[1].poc
            );
        }

        // Per-frame pixel comparison against the ground truth (by POC).
        for f in &frames {
            let pd = f.pixel_data.as_ref().expect("pixel data present");
            let want = truth
                .get(&f.poc)
                .unwrap_or_else(|| panic!("{name}: no ground truth for poc {}", f.poc));
            first_buf_diff(&format!("{name} poc {}", f.poc), &pd.buffer, want);
        }

        if !frames.is_empty() {
            let cw = layout.coded_w - layout.conf_left - layout.conf_right;
            let ch = layout.coded_h - layout.conf_top - layout.conf_bottom;
            let expect_fmt = if layout.bps > 1 { "I420-10" } else { "I420" };
            for f in &frames {
                assert_eq!((f.width, f.height), (cw, ch), "{name}: frame size mismatch");
                let pd = f.pixel_data.as_ref().unwrap();
                assert_eq!(pd.format, expect_fmt, "{name}: pixel format mismatch");
            }
        }

        println!("{name}: {} frames verified end-to-end (Rust pipeline == C++ core)", frames.len());
    }

    #[test]
    fn e2e_main() {
        run_e2e("main");
    }

    #[test]
    fn e2e_main10() {
        run_e2e("main10");
    }

    #[test]
    fn e2e_cra() {
        run_e2e("cra");
    }

    #[test]
    fn e2e_msp() {
        run_e2e("msp");
    }
}


