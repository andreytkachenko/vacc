//! Software H.264/AVC decoder.
//!
//! Control plane (common Rust implementations from `vacc-parser`):
//! - [`H264Parser`] — NAL / SPS / PPS / slice-header parsing
//! - [`H264Dpb`] — decoded picture buffer + reference marking
//! - [`PocCalculator`] — picture order count (types 0/1/2)
//! - `h264_reflist::build_ref_pic_lists` — spec 8.2.3.1+8.2.3.2 ref lists
//!
//! Data plane: the pure-Rust slice-decode core in [`crate::rust`] (macroblock
//! parse, intra/inter prediction, transform, deblocking), driven per-slice.
//! Ported bit-exactly from the edge264 C routines; every output is pinned to
//! golden hashes (`src/rust/golden_data.rs`).

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ptr::NonNull;

use vacc_core::codec::VideoCodec as CoreVideoCodec;
use vacc_core::decoder::{Decoder, DecoderInfo};
use vacc_core::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use vacc_core::frame::{DecodedFrame, PixelData, PixelPlane};
use vacc_core::picture::{H264Pps, H264Sps};
use vacc_core::session::Extent2D;
use vacc_parser::bitstream::BitstreamPacket;
use vacc_parser::h264::H264Parser;
use vacc_parser::h264_dpb::{H264Dpb, H264MmcoCommand, MARKING_LONG};
use vacc_parser::h264_poc::PocCalculator;
use vacc_parser::h264_reflist::RefPicLists;
use vacc_parser::{DetectedVideoFormat, ParseResult, SliceEntry, SliceHeader, VideoParser};

use crate::rust::bits::SliceBits;
use crate::rust::cabac::Cabac;
use crate::rust::slice::{
    PIXEL_MARGIN, RustMb, RustMbFlags, SLICEDATA_RECORD_LEN, SliceContext, UNAVAIL_MB,
};

/// 16-byte-aligned zero-initialized buffer (edge264 performs aligned SIMD
/// loads on sample planes; strides are multiples of 16 by construction).
struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBuf {
    fn new(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, 16).ok()?;
        let raw = unsafe { alloc_zeroed(layout) };
        if raw.is_null() {
            return None;
        }
        Some(Self {
            ptr: NonNull::new(raw)?,
            layout,
        })
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Error type for the software decoder.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid state: {0}")]
    InvalidState(&'static str),
    #[error("no SPS/PPS found in input data")]
    NoParameterSets,
    #[error("parser error: {0}")]
    Parser(String),
    #[error("slice decode error (errno {0})")]
    SliceDecode(i32),
    #[error("unsupported stream feature: {0}")]
    Unsupported(String),
    #[error("allocation failure")]
    Alloc,
}

/// Software H.264 decoder: vacc-parser control plane + pure-Rust data plane.
pub struct SwH264Decoder {
    /// Common parser (SPS/PPS/slice headers).
    parser: H264Parser,
    /// Common DPB.
    dpb: H264Dpb,
    /// Common POC calculator.
    poc_calc: PocCalculator,

    sps: Option<H264Sps>,
    pps: Option<H264Pps>,

    // Frame format derived from the SPS (strides chosen by the C side).
    stride_y: u32,
    stride_c: u32,
    plane_size_y: u32,
    plane_size_c: u32,
    coded_w: u32,
    coded_h: u32,
    disp_w: u32,
    disp_h: u32,
    /// Chroma subsampling factors (spec Table 8-2) used to scale crop offsets.
    sub_wc: u32,
    sub_hc: u32,

    // Per-DPB-slot storage: one contiguous block per slot holding a leading
    // zero margin (PIXEL_MARGIN) followed by the sample planes (Y + Cb/Cr)
    // plus a trailing overread margin. The inter prediction filter may read
    // up to a couple of rows past the chroma plane end; the margins absorb
    // those reads (upstream UB in the C core).
    n_slots: usize,
    slot_planes: Vec<Option<AlignedBuf>>,
    /// Per-slot macroblock arrays. A slot's flip bit toggles on every frame
    /// start; a freshly (re)allocated array has recovery_bits == 0 and its
    /// flip bit is reset, so the per-MB sentinel detects stale state.
    slot_mbs: Vec<Option<Vec<RustMb>>>,
    mbs_per_frame: usize,

    // Input buffering.
    pending_data: Vec<u8>,
    parse_offset: usize,

    // Output reordering (B frames).
    frame_count: u32,
    gop_count: i64,
    reorder_watermark: i64,
    pending_key: i64,
    pending_frames: VecDeque<(i64, DecodedFrame)>,

    /// Per-slice macroblock parse-record capture, armed via
    /// [`SwH264Decoder::arm_mb_dump`] (test-only).
    mb_dump: RefCell<MbDump>,
    /// Flip-bit sentinels per DPB slot (see `slot_mbs`).
    flip_bits: u32,
    /// Deblock progress (`next_deblock_addr`): reset per frame at frame start,
    /// updated after each slice; i32::MAX once the whole frame is deblocked.
    dec_next: i32,
    /// Total MBs set at frame start, never decremented (upstream); 0 means
    /// every MB has been decoded.
    remaining_mbs: i32,
}

/// Record-capture gate + collected per-slice parse records.
#[derive(Default)]
struct MbDump {
    /// True while capture is armed (see [`SwH264Decoder::arm_mb_dump`]).
    armed: bool,
    /// Per-slice parse records (golden-pinned).
    records: Vec<Vec<u8>>,
}

impl SwH264Decoder {
    /// Create a decoder and initialize it from the initial bitstream data.
    pub fn new(data: Vec<u8>) -> Result<Self, Error> {
        let mut parser = H264Parser::new();
        let format = DetectedVideoFormat::new(CoreVideoCodec::DecodeH264);
        parser
            .init(&format)
            .map_err(|e| Error::Parser(e.to_string()))?;

        // Parse the initial data until SPS+PPS are available.
        let mut sps: Option<H264Sps> = None;
        let mut pps: Option<H264Pps> = None;
        let offset = 0usize;
        while offset < data.len() {
            let packet = BitstreamPacket::new(data[offset..].to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps: s, pps: p, .. }) => {
                    if let Some(b) = s {
                        sps = Some(
                            b.downcast_ref::<H264Sps>()
                                .ok_or_else(|| Error::Parser("bad SPS type".into()))?
                                .clone(),
                        );
                    }
                    if let Some(b) = p {
                        pps = Some(
                            b.downcast_ref::<H264Pps>()
                                .ok_or_else(|| Error::Parser("bad PPS type".into()))?
                                .clone(),
                        );
                    }
                    continue;
                }
                Ok(ParseResult::Slice { .. }) => {
                    break; // SPS/PPS found before first slice
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
                Err(e) => return Err(Error::Parser(e.to_string())),
            }
        }
        let sps = sps.ok_or(Error::NoParameterSets)?;
        let pps = pps.ok_or(Error::NoParameterSets)?;

        // Re-anchor: the probe loop consumed NALs (and its first Slice result
        // is discarded), so reset the parser to re-parse the stream from byte 0.
        parser.reset();

        let mut dec = Self {
            parser,
            dpb: H264Dpb::new(0, 0, 0, 0),
            poc_calc: PocCalculator::new(),
            sps: Some(sps.clone()),
            pps: Some(pps.clone()),
            stride_y: 0,
            stride_c: 0,
            plane_size_y: 0,
            plane_size_c: 0,
            coded_w: 0,
            coded_h: 0,
            disp_w: 0,
            disp_h: 0,
            sub_wc: 2,
            sub_hc: 2,
            n_slots: 0,
            slot_planes: Vec::new(),
            slot_mbs: Vec::new(),
            mbs_per_frame: 0,
            pending_data: data,
            parse_offset: 0,
            frame_count: 0,
            gop_count: 0,
            reorder_watermark: i64::MIN,
            pending_key: 0,
            pending_frames: VecDeque::new(),
            mb_dump: RefCell::default(),
            flip_bits: 0,
            dec_next: 0,
            remaining_mbs: 0,
        };
        dec.init_sequence(&sps)?;
        Ok(dec)
    }

    /// (Re)initialize sequence state from an SPS: C side + DPB + slot storage.
    fn init_sequence(&mut self, sps: &H264Sps) -> Result<(), Error> {
        // This decoder's vendored C core is a single-threaded 8-bit 4:2:0 (and
        // monochrome) frame-coded subset of edge264. Reject other formats up
        // front with a clear error instead of silently mis-decoding (e.g.
        // 10-bit read as 8-bit, or field pictures decoded as frames) or
        // crashing in reconstruction paths that assume 4:2:0 chroma layout
        // (e.g. 4:4:4).
        if sps.separate_colour_plane_flag {
            return Err(Error::Unsupported("separate colour plane".into()));
        }
        match sps.chroma_format_idc {
            0 | 1 => {} // monochrome / 4:2:0
            2 => return Err(Error::Unsupported("4:2:2 chroma subsampling".into())),
            3 => return Err(Error::Unsupported("4:4:4 chroma subsampling".into())),
            _ => return Err(Error::Unsupported("unknown chroma_format_idc".into())),
        }
        if sps.bit_depth_luma_minus8 != 0 || sps.bit_depth_chroma_minus8 != 0 {
            return Err(Error::Unsupported(format!(
                "bit depth {} (luma) / {} (chroma); only 8-bit is supported",
                8 + sps.bit_depth_luma_minus8,
                8 + sps.bit_depth_chroma_minus8,
            )));
        }
        // The C core decodes progressive frame pictures only (MbaffFrameFlag is
        // forced to 0). Both field-coded streams (frame_mbs_only_flag == 0) and
        // MBAFF streams (mb_adaptive_frame_field_flag == 1) would be
        // mis-decoded, so reject anything that isn't plain progressive frames.
        if !sps.frame_mbs_only_flag {
            return Err(Error::Unsupported(
                "field/interlaced coding (frame_mbs_only_flag == 0)".into(),
            ));
        }
        if sps.mb_adaptive_frame_field_flag {
            return Err(Error::Unsupported(
                "MBAFF (mb_adaptive_frame_field_flag == 1)".into(),
            ));
        }

        // Coded size is a multiple of 16 (macroblocks). Per spec 8.3.1 the raw
        // frame_crop_*_offset values are multiplied by SubWidthC / SubHeightC
        // (Table 8-2) to yield the luma-sample crop; the chroma crop start is
        // the raw offset (the sub-resolution plane divides the factor out).
        self.coded_w = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
        self.coded_h = (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16;
        let (sub_wc, sub_hc) = match sps.chroma_format_idc {
            0 | 3 => (1, 1),
            2 => (2, 1), // 4:2:2
            _ => (2, 2), // 4:2:0 (default)
        };
        self.sub_wc = sub_wc;
        self.sub_hc = sub_hc;
        let crop_l = sps.frame_crop_left_offset;
        let crop_r = sps.frame_crop_right_offset;
        let crop_t = sps.frame_crop_top_offset;
        let crop_b = sps.frame_crop_bottom_offset;
        self.disp_w = self.coded_w.saturating_sub(sub_wc * (crop_l + crop_r));
        self.disp_h = self.coded_h.saturating_sub(sub_hc * (crop_t + crop_b));

        // Frame format (stride/plane sizes) — port of the C core's
        // sw264_set_sps. 8-bit only and 4:4:4 rejected above, so the chroma
        // stride is always `width` (+8 if a multiple of 4096).
        let mut stride_y = self.coded_w;
        if stride_y.is_multiple_of(2048) {
            stride_y += 16;
        }
        let mut stride_c = self.coded_w;
        if stride_c.is_multiple_of(4096) {
            stride_c += 8;
        }
        self.stride_y = stride_y;
        self.stride_c = stride_c;
        self.plane_size_y = stride_y * self.coded_h;
        self.plane_size_c = stride_c
            * if sps.chroma_format_idc == 1 {
                self.coded_h / 2
            } else {
                self.coded_h
            };

        // DPB sizing (same scheme as the VAAPI backend).
        let max_dpb = sps.max_num_ref_frames.clamp(1, 16) as usize;
        self.n_slots = max_dpb.max(4) + 4;
        self.dpb = H264Dpb::new(
            self.n_slots,
            self.n_slots,
            (sps.max_num_ref_frames.min(self.n_slots as u32)).max(1),
            sps.max_frame_num,
        );
        if let Some(vui) = &sps.vui {
            self.dpb
                .set_max_num_reorder_frames(vui.max_num_reorder_frames as u32);
        }

        self.mbs_per_frame = (self.coded_w / 16 + 1) as usize * ((self.coded_h / 16) as usize) - 1;
        self.slot_planes = (0..self.n_slots).map(|_| None).collect();
        self.slot_mbs = vec![None; self.n_slots];
        self.flip_bits = 0;
        self.dec_next = 0;
        self.remaining_mbs = 0;
        self.poc_calc.reset();
        Ok(())
    }

    /// Size of the sample-plane part of a slot block (planes + overread margin).
    fn plane_block_size(&self) -> usize {
        // Upstream: plane_size_Y + plane_size_C + 16. Keep a slightly larger
        // margin; the inter filter may read past the chroma plane end.
        (self.plane_size_y + self.plane_size_c + 32) as usize
    }

    /// Allocate (lazily, once) the plane + macroblock storage for a DPB slot.
    fn ensure_slot_storage(&mut self, slot: usize) -> Result<(), Error> {
        if self.slot_planes[slot].is_none() {
            // Leading zero margin so the intra kernels' edge reads (upstream UB
            // in the C core) stay in valid memory; trailing overread margin via
            // plane_block_size.
            let buf =
                AlignedBuf::new(PIXEL_MARGIN + self.plane_block_size()).ok_or(Error::Alloc)?;
            // Each row's trailing padding macroblock slot is unavail_mb;
            // column-0 MBs read it as their left neighbor. A freshly zeroed mb
            // array has recovery_bits == 0, so reset this slot's flip-bit
            // sentinel to 0 too; otherwise stale toggle history can match the
            // fresh array and the sentinel aborts the slice after 0 macroblocks
            // (e.g. after an IDR frees and re-allocates the slot).
            let w = (self.coded_w / 16) as usize;
            let mut v = vec![RustMb::default(); self.mbs_per_frame];
            for i in (0..self.mbs_per_frame).step_by(w + 1) {
                if i + w < self.mbs_per_frame {
                    v[i + w] = UNAVAIL_MB;
                }
            }
            self.flip_bits &= !(1u32 << slot);
            self.slot_mbs[slot] = Some(v);
            self.slot_planes[slot] = Some(buf);
        }
        Ok(())
    }

    /// Decode the next picture from `pending_data`, or None when no complete
    /// picture is available yet.
    fn decode_one(&mut self) -> Result<Option<DecodedFrame>, Error> {
        loop {
            if self.parse_offset >= self.pending_data.len() {
                return Ok(None);
            }
            let remaining = &self.pending_data[self.parse_offset..];
            let packet = BitstreamPacket::new(remaining.to_vec());
            match self.parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                    if let Some(b) = sps {
                        let new_sps = b
                            .downcast_ref::<H264Sps>()
                            .ok_or_else(|| Error::Parser("bad SPS type".into()))?
                            .clone();
                        let format_changed = self
                            .sps
                            .as_ref()
                            .map(|s| !same_format(s, &new_sps))
                            .unwrap_or(true);
                        self.sps = Some(new_sps.clone());
                        if format_changed {
                            self.init_sequence(&new_sps)?;
                        }
                    }
                    if let Some(b) = pps {
                        let new_pps = b
                            .downcast_ref::<H264Pps>()
                            .ok_or_else(|| Error::Parser("bad PPS type".into()))?
                            .clone();
                        self.pps = Some(new_pps);
                    }
                    continue;
                }
                Ok(ParseResult::Slice {
                    slices,
                    bytes_consumed,
                }) => {
                    if slices.is_empty() {
                        return Ok(None);
                    }
                    self.parse_offset += bytes_consumed;
                    return self.decode_h264_frame(&slices);
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                    self.parse_offset = self.pending_data.len();
                    return Ok(None);
                }
                Err(e) => return Err(Error::Parser(e.to_string())),
            }
        }
    }

    /// Decode one complete picture (all its slices) using the common DPB/POC
    /// state and the Rust slice-decode core.
    fn decode_h264_frame(&mut self, slices: &[SliceEntry]) -> Result<Option<DecodedFrame>, Error> {
        let sps = self.sps.as_ref().ok_or(Error::InvalidState("no SPS"))?;

        // Frame-level parameters from the first slice header.
        let first_slh = match &slices[0].slice_header {
            Some(SliceHeader::H264(h)) => h,
            _ => return Err(Error::InvalidState("missing slice header")),
        };
        // Redundant slices duplicate an earlier slice of the same picture.
        if first_slh.redundant_pic_cnt > 0 {
            return Ok(None);
        }

        let is_idr = first_slh.nal_unit_type == 5;
        let is_ref = first_slh.nal_ref_idc != 0;

        // POC via the common PocCalculator (ONE POC implementation across
        // backends). An IDR picture restarts the POC state (H.264 8.2.1).
        if is_idr {
            self.poc_calc.reset();
        }
        let poc = self.poc_calc.calculate(sps, first_slh, is_ref);

        // Convert dec_ref_pic_marking into common-DPB MMCO commands (8.2.5).
        let mmco_commands: Vec<H264MmcoCommand> = first_slh
            .dec_ref_pic_marking
            .iter()
            .map(|op| match op.memory_management_control_operation {
                1 => H264MmcoCommand::UnmarkShortTerm {
                    difference_of_pic_nums_minus1: op.value,
                },
                2 => H264MmcoCommand::UnmarkLongTerm {
                    long_term_frame_idx: op.value,
                },
                3 => H264MmcoCommand::AssignLongTerm {
                    difference_of_pic_nums_minus1: 0,
                    long_term_frame_idx: op.value,
                },
                4 => H264MmcoCommand::SetMaxLongTermFrameIdx {
                    max_long_term_frame_idx_plus1: op.value,
                },
                5 => H264MmcoCommand::UnmarkAll,
                6 => H264MmcoCommand::AssignLongTermToCurrent {
                    long_term_frame_idx: op.value,
                },
                _ => H264MmcoCommand::UnmarkAll,
            })
            .collect();

        // Stage the current picture in the common DPB (pre-marking state).
        self.dpb.picture_start(
            first_slh.frame_num,
            poc,
            is_ref,
            is_idr,
            first_slh.no_output_of_prior_pics_flag,
            !mmco_commands.is_empty(),
            mmco_commands,
        );

        // Build the per-slice reference lists (spec 8.2.3.1 + 8.2.3.2) from
        // the PRE-marking DPB state, before commit_current.
        let mut slice_lists: Vec<RefPicLists> = Vec::with_capacity(slices.len());
        for s in slices.iter() {
            let (st, l0m1, l1m1, mod_l0, mod_l1) = match &s.slice_header {
                Some(SliceHeader::H264(h)) => (
                    h.slice_type % 5,
                    h.num_ref_idx_l0_active_minus1,
                    h.num_ref_idx_l1_active_minus1,
                    &h.ref_pic_list_modification_l0[..],
                    &h.ref_pic_list_modification_l1[..],
                ),
                _ => (
                    first_slh.slice_type % 5,
                    first_slh.num_ref_idx_l0_active_minus1,
                    first_slh.num_ref_idx_l1_active_minus1,
                    &first_slh.ref_pic_list_modification_l0[..],
                    &first_slh.ref_pic_list_modification_l1[..],
                ),
            };
            let lists = self.dpb.build_ref_lists(st, l0m1, l1m1, mod_l0, mod_l1);
            slice_lists.push(lists);
        }

        // Determine the storage slot BEFORE decoding: prepare_current_avoiding
        // applies the marking process (which depends only on pre-decode DPB state +
        // this picture's header fields, not on decoded content) and returns the
        // first empty slot. The reference lists above were built from the
        // pre-marking state, so this is spec-equivalent to marking post-decode.
        // Pass every slot those lists still reference so we never decode the
        // current picture into a buffer it is simultaneously reading from
        // (DPB aliasing: the sliding window can free a ref the slice still uses).
        let mut used_refs: Vec<usize> = Vec::new();
        for lists in slice_lists.iter() {
            for r in lists.l0.iter().chain(lists.l1.iter()) {
                if !used_refs.contains(&r.slot) {
                    used_refs.push(r.slot);
                }
            }
        }
        let slot = self.dpb.prepare_current_avoiding(&used_refs);
        self.ensure_slot_storage(slot)?;
        // Slot 0 doubles as the dummy buffer for missing references (invalid streams).
        self.ensure_slot_storage(0)?;
        // Note: the slot's MB array is intentionally NOT re-initialized here. The
        // per-slot frame_flip_bit sentinel (toggling on every reuse) detects
        // stale MB state; zeroing recovery_bits per frame would defeat it.

        // Per-frame setup: toggle the slot's flip-bit sentinel and reset the
        // deblock progress (next_deblock_addr = 0, remaining_mbs = W*H).
        self.flip_bits ^= 1u32 << slot;
        self.dec_next = 0;
        self.remaining_mbs = (self.coded_w / 16) as i32 * (self.coded_h / 16) as i32;

        // Decode every slice of the picture.
        for (idx, s) in slices.iter().enumerate() {
            let h = match &s.slice_header {
                Some(SliceHeader::H264(h)) => h,
                _ => return Err(Error::InvalidState("missing slice header")),
            };
            let st = h.slice_type % 5;
            if st > 2 {
                return Err(Error::Unsupported("SP/SI slices".into()));
            }
            self.dec_next = self.decode_slice(h, &slices[idx], &slice_lists[idx], poc, slot)?;
        }

        // Commit the picture to the common DPB (post-decode marking + display).
        self.dpb.commit_current(slot);

        // Free storage of slots that no longer hold a picture.
        for (i, s) in self.dpb.slots.iter().enumerate() {
            if s.state == 0 {
                self.slot_planes[i] = None;
                self.slot_mbs[i] = None;
            }
        }

        // Build the output frame (cropped I420).
        let pixel_data = self.extract_frame(slot);

        // B-frame reordering key (POC resets at each IDR, so combine with the
        // GOP index to keep it monotonic across the stream).
        if is_idr && self.frame_count > 0 {
            self.gop_count += 1;
        }
        self.reorder_watermark = self.reorder_watermark.max(self.gop_count);
        self.pending_key = self.gop_count * 1_000_000 + poc as i64;

        let mut frame = DecodedFrame::new(
            self.frame_count,
            (self.frame_count as u64 * 33_333) as i64,
            self.disp_w,
            self.disp_h,
            false,
        );
        frame.pixel_data = Some(pixel_data);
        frame.poc = poc;
        frame.pts_valid = true;

        self.frame_count += 1;
        Ok(Some(frame))
    }

    /// Arm per-slice macroblock parse-record capture: one record per decoded
    /// slice (308 bytes per macroblock) is collected for golden pinning.
    /// Test-only.
    #[cfg(test)]
    pub fn arm_mb_dump(&self) {
        self.mb_dump.borrow_mut().armed = true;
    }

    /// Collect the per-slice parse records captured since [`arm_mb_dump`],
    /// one entry per decoded slice. Test-only.
    #[cfg(test)]
    pub fn take_rust_records(&self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.mb_dump.borrow_mut().records)
    }

    /// Decode one slice through the Rust core; updates the shared deblock
    /// progress (`dec_next`) and returns it. When record capture is armed
    /// (test-only), the per-MB record stream (`CurrMbAddr` u32 + 304-byte
    /// macroblock per MB) is collected as well.
    fn decode_slice(
        &self,
        h: &vacc_parser::h264::SliceHeader,
        entry: &SliceEntry,
        lists: &RefPicLists,
        poc: i32,
        slot: usize,
    ) -> Result<i32, Error> {
        let dec_next = self.dec_next;
        let sps = self.sps.as_ref().ok_or(Error::InvalidState("no SPS"))?;
        let pps = self.pps.as_ref().ok_or(Error::InvalidState("no PPS"))?;
        let st = h.slice_type % 5;

        // Same guarded payload as the C path, with >= 18 bytes of headroom
        // (SliceBits::new requirement; cabac_start's reclaim walks back to
        // cpb-4). Guard bytes are non-zero for the same EPB reason.
        let mut nal = entry.nal_data.clone();
        if nal.last() == Some(&0) {
            nal.pop();
        }
        let mut buf = vec![0xFFu8; 24 + nal.len() + 16];
        buf[24..24 + nal.len()].copy_from_slice(&nal);
        let skip_bits = 8 + h.header_bit_size as u32;

        // Prime the reader at the NAL header and skip to the slice data,
        // exactly like sw264_decode_slice (msb = 1 << 63, refill, get_u1 x N).
        let mut bits = SliceBits::new(&buf, 24, nal.len());
        for _ in 0..skip_bits {
            let _ = bits.get_u1();
        }

        let mbs = self
            .slot_mbs
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidState("no mb buffer for slot"))?;
        // B slices: collocated picture mb array (RefPicList1[0]).
        let col_slot = if st == 1 {
            lists.l1.first().map(|r| r.slot)
        } else {
            None
        };

        let qp_y = (26 + pps.pic_init_qp_minus26 + h.slice_qp_delta).clamp(0, 51) as i16;
        let cabac_init_idc = if st == 2 {
            0
        } else {
            1 + h.cabac_init_idc as i8
        };

        // Per-slot POC differences (B-slice direct mode / implicit weights).
        let mut diff_poc = [0i16; 32];
        for (i, s) in self.dpb.slots.iter().enumerate() {
            diff_poc[i] = if s.state == 0 {
                0
            } else {
                (poc - s.poc) as i16
            };
        }

        // Long-term reference mask (same DPB state as at frame start: the
        // current picture is committed only after all slices are decoded).
        let mut long_term_frames = 0u32;
        for (i, s) in self.dpb.slots.iter().enumerate() {
            if s.state != 0 && s.marking == MARKING_LONG {
                long_term_frames |= 1 << i;
            }
        }

        let mut ref_pic_list = [[0i8; 32]; 2];
        for (i, r) in lists.l0.iter().enumerate() {
            ref_pic_list[0][i] = r.slot as i8;
        }
        for (i, r) in lists.l1.iter().enumerate() {
            ref_pic_list[1][i] = r.slot as i8;
        }

        // Weighted prediction: same table materialization as the C path above
        // (L0 refs at [0..n0), L1 refs at [32..32+n1)).
        let has_pw_table =
            (st == 0 && pps.weighted_pred_flag) || (st == 1 && pps.weighted_bipred_idc == 1);
        let (mut explicit_weights, mut explicit_offsets) = ([[0i16; 64]; 3], [[0i8; 64]; 3]);
        let (mut luma_log2_weight_denom, mut chroma_log2_weight_denom) = (0i8, 0i8);
        if has_pw_table {
            luma_log2_weight_denom = h.luma_log2_weight_denom as i8;
            chroma_log2_weight_denom = h.chroma_log2_weight_denom as i8;
            let n0 = h.num_ref_idx_l0_active_minus1 as usize + 1;
            let n1 = if st == 1 {
                h.num_ref_idx_l1_active_minus1 as usize + 1
            } else {
                0
            };
            for i in 0..n0 {
                explicit_weights[0][i] = h.luma_weight_l0[i];
                explicit_offsets[0][i] = h.luma_offset_l0[i] as i8;
                explicit_weights[1][i] = h.chroma_weight_l0[i][0];
                explicit_offsets[1][i] = h.chroma_offset_l0[i][0] as i8;
                explicit_weights[2][i] = h.chroma_weight_l0[i][1];
                explicit_offsets[2][i] = h.chroma_offset_l0[i][1] as i8;
            }
            for i in 0..n1 {
                explicit_weights[0][32 + i] = h.luma_weight_l1[i];
                explicit_offsets[0][32 + i] = h.luma_offset_l1[i] as i8;
                explicit_weights[1][32 + i] = h.chroma_weight_l1[i][0];
                explicit_offsets[1][32 + i] = h.chroma_offset_l1[i][0] as i8;
                explicit_weights[2][32 + i] = h.chroma_weight_l1[i][1];
                explicit_offsets[2][32 + i] = h.chroma_offset_l1[i][1] as i8;
            }
        }

        // Reference planes for inter MC (C `t.samples_buffers`).
        let ref_plane_bases: Vec<*const u8> = self
            .slot_planes
            .iter()
            .map(|p| {
                p.as_ref().map_or(std::ptr::null(), |b| unsafe {
                    b.as_ptr().add(PIXEL_MARGIN)
                })
            })
            .collect();

        let mut ctx = SliceContext {
            bits,
            cabac: Cabac::new(),
            is_cabac: pps.entropy_coding_mode_flag,
            mb_buffer: mbs.as_ptr() as *mut RustMb,
            mb_pos: 0,
            mb_col: std::ptr::null(),
            mb_col_buffer: match col_slot {
                Some(s) => self
                    .slot_mbs
                    .get(s)
                    .and_then(Option::as_ref)
                    .map_or(std::ptr::null(), |v| v.as_ptr()),
                None => std::ptr::null(),
            },
            mb_dump: std::ptr::null_mut(),
            mb_dump_off: 0,
            mb_dump_cap: 0,
            mb_a: std::ptr::null(),
            mb_b: std::ptr::null(),
            mb_c: std::ptr::null(),
            mb_d: std::ptr::null(),
            pic_width_in_mbs: sps.pic_width_in_mbs_minus1 as i16 + 1,
            pic_height_in_mbs: (sps.pic_height_in_map_units_minus1 + 1) as i16,
            first_mb_in_slice: h.first_mb_in_slice,
            slice_type: st as i8,
            cabac_init_idc,
            frame_flip_bit: ((self.flip_bits >> slot) & 1) as i8,
            disable_deblocking_filter_idc: h.disable_deblocking_filter_idc,
            chroma_array_type: if sps.separate_colour_plane_flag {
                0
            } else {
                sps.chroma_format_idc as i8
            },
            direct_spatial_mv_pred_flag: h.direct_spatial_mv_pred_flag as i8,
            direct_8x8_inference_flag: sps.direct_8x8_inference_flag as i8,
            num_ref_idx_active: [
                (h.num_ref_idx_l0_active_minus1 + 1) as i8,
                (h.num_ref_idx_l1_active_minus1 + 1) as i8,
            ],
            pps_transform_8x8_mode_flag: pps.transform_8x8_mode_flag as i8,
            // P slices: weighted_bipred_idc follows weighted_pred_flag.
            weighted_bipred_idc: if st == 0 {
                pps.weighted_pred_flag as i8
            } else {
                pps.weighted_bipred_idc as i8
            },
            luma_log2_weight_denom,
            chroma_log2_weight_denom,
            explicit_weights,
            explicit_offsets,
            implicit_weights: [[0u8; 32]; 32],
            ref_pic_list,
            diff_poc,
            prev_long_term_frames: long_term_frames,
            qp_y,
            chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as i8,
            mbx: 0,
            mby: 0,
            curr_mb_addr: 0,
            mb_skip_run: -1,
            col_short_term: false,
            inc: RustMbFlags::default(),
            unavail4x4: [0; 48],
            nc_inc: [[0; 16]; 3],
            a4x4_int8: [0; 16],
            b4x4_int8: [0; 16],
            acbcr_int8: [0; 16],
            bcbcr_int8: [0; 16],
            refidx4x4_c: [0; 16],
            absmvd_a: [0; 16],
            absmvd_b: [0; 16],
            mvs_a: [0; 16],
            mvs_b: [0; 16],
            mvs_c: [0; 16],
            mvs_d: [0; 16],
            transform_8x8_mode_flag: 0,
            num_ref_idx_mask: 0,
            dist_scale_factor: [0; 32],
            clip_ref_idx: [0; 8],
            map_pic_to_list0: [0; 32],
            ctx_idx_offsets: [0; 4],
            coeff_abs_inc: [0; 8],
            sig_inc: [0; 64],
            last_inc: [0; 64],
            scan: [0; 64],
            qp_c: [[0; 64]; 2],
            c: [0; 64],
            mb_qp_delta_nz: 0,
            qp_s: [0; 4],
            bit_depth: [
                (8 + sps.bit_depth_luma_minus8) as u32,
                (8 + sps.bit_depth_chroma_minus8) as u32,
                (8 + sps.bit_depth_chroma_minus8) as u32,
            ],
            pixel_margin_base: self
                .slot_planes
                .get(slot)
                .and_then(Option::as_ref)
                .map_or(std::ptr::null_mut(), AlignedBuf::as_mut_ptr),
            samples_base: self
                .slot_planes
                .get(slot)
                .and_then(Option::as_ref)
                .map_or(std::ptr::null_mut(), |b| unsafe {
                    b.as_mut_ptr().add(PIXEL_MARGIN)
                }),
            stride: [self.stride_y as u16, self.stride_c as u16],
            plane_size_y: self.plane_size_y,
            plane_size_c: self.plane_size_c,
            coded_w: self.coded_w,
            coded_h: self.coded_h,
            samples_mb: [std::ptr::null_mut(); 3],
            ref_plane_bases,
            mc_y: [0u8; 441],
            mc_c: [0u8; 162],
            dblk_y: [0u8; crate::rust::deblock::DEBLOCK_LY_SIZE],
            dblk_c: [0u8; crate::rust::deblock::DEBLOCK_LC_SIZE],
            ws4: if sps.seq_scaling_matrix_present_flag {
                // C memcpys the 96 spec-ordered bytes straight into
                // weightScale4x4 (the kernel then indexes plane+inter*3).
                std::array::from_fn(|i| std::array::from_fn(|j| sps.scaling_list_4x4[i][j] as i8))
            } else {
                [[16i8; 16]; 6]
            },
            ws8: if sps.seq_scaling_matrix_present_flag {
                // C memcpys 384 bytes from a field that only holds the two
                // luma lists (latent OOB read upstream); the Cb/Cr slots are
                // UB there. We fill them with the flat list — out of Tier E
                // scope (no in-scope stream has scaling lists).
                let mut ws = [[16i8; 64]; 6];
                for (dst, src) in ws[..2].iter_mut().zip(sps.scaling_list_8x8.iter()) {
                    dst.copy_from_slice(&src.map(|v| v as i8));
                }
                ws
            } else {
                [[16i8; 64]; 6]
            },
            filter_offset_a: h.slice_alpha_c0_offset_div2 * 2,
            filter_offset_b: h.slice_beta_offset_div2 * 2,
            // C vacc_sw264.c L1132.
            next_deblock_addr: if dec_next == h.first_mb_in_slice as i32
                || h.disable_deblocking_filter_idc == 2
            {
                h.first_mb_in_slice as i32
            } else {
                i32::MIN
            },
        };
        ctx.initialize_context();

        // Per-slice parse-record capture (armed via `arm_mb_dump`); the core
        // skips writes when the target is null.
        let mut rec_buf: Option<Vec<u8>> = self
            .mb_dump
            .borrow()
            .armed
            .then(|| vec![0u8; self.mbs_per_frame * SLICEDATA_RECORD_LEN]);
        ctx.mb_dump = rec_buf
            .as_mut()
            .map_or(std::ptr::null_mut(), Vec::as_mut_ptr);
        ctx.mb_dump_cap = rec_buf.as_ref().map_or(0, Vec::len);

        if ctx.is_cabac {
            // cabac_alignment_one_bit: a good probability to catch random errors.
            if ctx.cabac.start(&mut ctx.bits) {
                return Err(Error::SliceDecode(114)); // EBADMSG, as the C core
            }
            ctx.cabac.init(qp_y as u8, cabac_init_idc as usize);
            ctx.mb_qp_delta_nz = 0;
        } else {
            ctx.mb_skip_run = -1;
        }

        ctx.parse_slice_data();

        // C vacc_sw264.c L1202-1227: deblock the rest of the MBs in this slice
        // (for a single-slice frame this is the entire last row).
        if ctx.next_deblock_addr >= 0 {
            ctx.next_deblock_addr = ctx.next_deblock_addr.max(ctx.first_mb_in_slice as i32);
            ctx.deblock_range(ctx.curr_mb_addr);
        }

        // E0 plumbing check: samples_mb must track the closed-form position of
        // (mbx, mby) — C initialises and advances it identically, so any drift
        // in the per-MB advance arithmetic shows up here on every slice.
        let base = ctx.samples_base as usize;
        let mbx = ctx.mbx as i64;
        let mby = ctx.mby as i64;
        let sy = self.stride_y as i64;
        let sc = self.stride_c as i64;
        assert_eq!(
            ctx.samples_mb[0] as usize - base,
            ((mbx + mby * sy) * 16) as usize,
            "samples_mb[0] drift at mbx={mbx} mby={mby}"
        );
        assert_eq!(
            ctx.samples_mb[1] as usize - base,
            ((mbx + mby * sc) * 8 + self.plane_size_y as i64) as usize,
            "samples_mb[1] drift at mbx={mbx} mby={mby}"
        );
        assert_eq!(
            ctx.samples_mb[2] as usize - base,
            ((mbx + mby * sc) * 8 + self.plane_size_y as i64) as usize
                + (self.stride_c / 2) as usize,
            "samples_mb[2] drift at mbx={mbx} mby={mby}"
        );

        // Update the shared deblock progress (C vacc_sw264.c L1230).
        let mut dec_next_out = if dec_next >= h.first_mb_in_slice as i32
            && !(h.disable_deblocking_filter_idc == 0 && ctx.next_deblock_addr < 0)
        {
            ctx.curr_mb_addr
        } else {
            dec_next
        };

        // C vacc_sw264.c L1235-1265: deblock the rest of the frame if all MBs
        // have been decoded (remaining_mbs = W*H, never decremented upstream),
        // then signal completion.
        let remaining_mbs = self.remaining_mbs - (ctx.curr_mb_addr - ctx.first_mb_in_slice as i32);
        if remaining_mbs == 0 {
            ctx.next_deblock_addr = dec_next_out;
            let total = (ctx.pic_width_in_mbs as i32) * (ctx.pic_height_in_mbs as i32);
            if ctx.next_deblock_addr < total {
                ctx.deblock_range(total);
            }
            dec_next_out = i32::MAX;
        }
        if let Some(mut buf) = rec_buf {
            buf.truncate(ctx.mb_dump_off);
            self.mb_dump.borrow_mut().records.push(buf);
        }
        Ok(dec_next_out)
    }

    /// Copy the cropped display area out of a DPB slot plane into I420.
    fn extract_frame(&self, slot: usize) -> PixelData {
        let plane = self.slot_planes[slot].as_ref().expect("slot plane");
        // Slot planes carry a leading zero margin before the Y plane.
        let base = unsafe { plane.as_ptr().add(PIXEL_MARGIN) };
        let sps = self.sps.as_ref().unwrap();
        let idc = sps.chroma_format_idc;
        // Luma crop start = SubWidthC/SubHeightC * raw offset (spec 8.3.1);
        // chroma crop start = raw offset (sub-resolution plane divides the
        // factor out). w/h are already the cropped display size.
        let cl = (sps.frame_crop_left_offset as usize) * self.sub_wc as usize;
        let ct = (sps.frame_crop_top_offset as usize) * self.sub_hc as usize;
        let w = self.disp_w as usize;
        let h = self.disp_h as usize;
        let cw = if idc == 3 { w } else { w / 2 };
        let ch = if idc == 1 { h / 2 } else { h };
        let clc = sps.frame_crop_left_offset as usize;
        let ctc = sps.frame_crop_top_offset as usize;

        let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
        // Y
        for row in 0..h {
            let src = unsafe { base.add((ct + row) * self.stride_y as usize + cl) };
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, w) });
        }
        // Cb / Cr. The C side stores chroma as row-pairs: at offset
        // plane_size_Y + r*stride_c, bytes [0..width_C) are the Cb samples of
        // chroma row r and [width_C..stride_C) (== +stride_C/2) are the Cr
        // samples. See edge264 get_frame / inter.c edge copy.
        let chroma_base = unsafe { base.add(self.plane_size_y as usize) };
        let half_stride_c = (self.stride_c / 2) as usize;
        for row in 0..ch {
            let rp = unsafe { chroma_base.add((ctc + row) * self.stride_c as usize + clc) };
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(rp, cw) });
        }
        for row in 0..ch {
            let src = unsafe {
                chroma_base.add((ctc + row) * self.stride_c as usize + half_stride_c + clc)
            };
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, cw) });
        }

        let y_off = 0;
        let u_off = y_off + w * h;
        let v_off = u_off + cw * ch;
        PixelData {
            format: "I420".to_string(),
            y: PixelPlane {
                data: unsafe { out.as_ptr().add(y_off) },
                pitch: w,
                width: w,
                height: h,
            },
            u: PixelPlane {
                data: unsafe { out.as_ptr().add(u_off) },
                pitch: cw,
                width: cw,
                height: ch,
            },
            v: Some(PixelPlane {
                data: unsafe { out.as_ptr().add(v_off) },
                pitch: cw,
                width: cw,
                height: ch,
            }),
            buffer: out,
        }
    }
}

/// True when two SPSes describe a different frame format (requiring DPB /
/// C-side reinitialization).
fn same_format(a: &H264Sps, b: &H264Sps) -> bool {
    a.pic_width_in_mbs_minus1 == b.pic_width_in_mbs_minus1
        && a.pic_height_in_map_units_minus1 == b.pic_height_in_map_units_minus1
        && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8
        && a.bit_depth_chroma_minus8 == b.bit_depth_chroma_minus8
        && a.chroma_format_idc == b.chroma_format_idc
}

impl Decoder for SwH264Decoder {
    type Error = Error;

    fn new(data: Vec<u8>) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        SwH264Decoder::new(data)
    }

    fn new_with_format(
        _data: Vec<u8>,
        _codec: CoreVideoCodec,
        _format: &VideoFormat,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Err(Error::InvalidState(
            "use SwH264Decoder::new with bitstream data",
        ))
    }

    fn info(&self) -> DecoderInfo {
        let sps = self.sps.as_ref();
        DecoderInfo {
            backend: "sw".to_string(),
            codec: CoreVideoCodec::DecodeH264,
            coded_size: Extent2D::new(self.coded_w, self.coded_h),
            display_size: Extent2D::new(self.disp_w, self.disp_h),
            chroma_subsampling: match sps.map(|s| s.chroma_format_idc) {
                Some(0) => ChromaSubsampling::Monochrome,
                Some(2) => ChromaSubsampling::_422,
                Some(3) => ChromaSubsampling::_444,
                _ => ChromaSubsampling::_420,
            },
            luma_bit_depth: match sps.map(|s| 8 + s.bit_depth_luma_minus8) {
                Some(10) => ComponentBitDepth::Bit10,
                Some(12) => ComponentBitDepth::Bit12,
                _ => ComponentBitDepth::Bit8,
            },
            chroma_bit_depth: match sps.map(|s| 8 + s.bit_depth_chroma_minus8) {
                Some(10) => ComponentBitDepth::Bit10,
                Some(12) => ComponentBitDepth::Bit12,
                _ => ComponentBitDepth::Bit8,
            },
            profile_idc: sps.map(|s| s.profile_idc as u32),
            dpb_slots: self.n_slots as u32,
        }
    }

    fn submit(&mut self, data: &[u8]) -> Result<(), Self::Error> {
        if self.parse_offset >= self.pending_data.len() {
            self.pending_data.clear();
        } else {
            let unconsumed = self.pending_data[self.parse_offset..].to_vec();
            self.pending_data = unconsumed;
            self.parse_offset = 0;
        }
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> Result<Option<DecodedFrame>, Self::Error> {
        loop {
            // Emit the front of the reorder buffer once it is in display order:
            // a frame's GOP is complete only after a newer GOP has been decoded.
            if let Some(&(front_key, _)) = self.pending_frames.front() {
                let exhausted = self.parse_offset >= self.pending_data.len();
                let front_gop = front_key / 1_000_000;
                if exhausted || front_gop < self.reorder_watermark {
                    return Ok(Some(self.pending_frames.pop_front().unwrap().1));
                }
            }
            if self.parse_offset >= self.pending_data.len() {
                return Ok(None);
            }
            let offset_before = self.parse_offset;
            match self.decode_one()? {
                Some(frame) => {
                    let key = self.pending_key;
                    let pos = self
                        .pending_frames
                        .iter()
                        .position(|(k, _)| *k > key)
                        .unwrap_or(self.pending_frames.len());
                    self.pending_frames.insert(pos, (key, frame));
                    continue;
                }
                None => {
                    if self.parse_offset == offset_before && self.pending_frames.is_empty() {
                        return Ok(None);
                    }
                    continue;
                }
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>, Self::Error> {
        let mut frames = Vec::new();
        while let Some((_, frame)) = self.pending_frames.pop_front() {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.dpb.invalidate_all();
        self.poc_calc.reset();
        self.parser.reset();
        self.slot_planes = (0..self.n_slots).map(|_| None).collect();
        self.slot_mbs = vec![None; self.n_slots];
        self.flip_bits = 0;
        self.dec_next = 0;
        self.remaining_mbs = 0;
        self.pending_data.clear();
        self.parse_offset = 0;
        self.frame_count = 0;
        self.gop_count = 0;
        self.reorder_watermark = i64::MIN;
        self.pending_frames.clear();
        Ok(())
    }
}

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(name: &str) -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/samples/");
        std::fs::read(format!("{path}{name}")).unwrap()
    }

    fn dump_smoke(name: &str) {
        let data = sample(name);
        let mut dec = SwH264Decoder::new(data).unwrap();
        // Coded macroblocks per frame (the `mbs_per_frame` field is the
        // padded buffer capacity, not the raster MB count).
        let coded = dec.info().coded_size;
        let mbs = (coded.width as usize / 16) * (coded.height as usize / 16);
        dec.arm_mb_dump();

        let mut frames = 0usize;
        while let Some(_frame) = dec.decode().unwrap() {
            frames += 1;
        }
        frames += dec.flush().unwrap().len();
        assert!(frames > 0, "{name}: no frames decoded");

        let records = dec.take_rust_records();
        assert!(!records.is_empty(), "{name}: no slices recorded");
        let mut total_mbs = 0usize;
        for rec in &records {
            assert_eq!(rec.len() % SLICEDATA_RECORD_LEN, 0);
            let n = rec.len() / SLICEDATA_RECORD_LEN;
            total_mbs += n;
            for i in 0..n {
                let addr = u32::from_le_bytes(rec[i * 308..i * 308 + 4].try_into().unwrap());
                if i > 0 {
                    let prev = u32::from_le_bytes(
                        rec[(i - 1) * 308..(i - 1) * 308 + 4].try_into().unwrap(),
                    );
                    assert_eq!(addr, prev + 1, "{name}: non-contiguous mb addr in slice");
                }
            }
        }
        assert_eq!(total_mbs, frames * mbs, "{name}: MB total mismatch");
    }

    /// Scope invariant for the Tier E pixel oracle: every bundled stream the
    /// 8-bit 4:2:0/mono C core accepts uses default (all-16) dequant scaling
    /// lists. Streams with `seq_scaling_matrix_present_flag == 1` are out of
    /// scope — the C core's 384-byte memcpy of a 128-byte SPS field would be
    /// an OOB read there, so matching it is neither possible nor desired.
    #[test]
    fn stream_scope_8bit_no_scaling_lists() {
        let names = [
            "h264_baseline.h264",
            "h264_constrained_baseline.h264",
            "h264_main.h264",
            "h264_high.h264",
            "h264_tC.h264",
            "h264_tD.h264",
            "h264_tN.h264",
            "h264_tW.h264",
            "h264_xallI.h264",
            "h264_xfd.h264",
        ];
        for name in names {
            let dec = SwH264Decoder::new(sample(name)).unwrap();
            let sps = dec.sps.clone().unwrap();
            assert_eq!(
                8 + sps.bit_depth_luma_minus8 as u32,
                8,
                "{name}: expected 8-bit luma"
            );
            assert!(
                (0..=1).contains(&sps.chroma_format_idc),
                "{name}: expected mono/4:2:0"
            );
            assert!(
                !sps.seq_scaling_matrix_present_flag,
                "{name}: scaling lists out of Tier E scope"
            );
        }
    }

    #[test]
    fn dump_smoke_baseline() {
        dump_smoke("h264_baseline.h264");
    }

    #[test]
    fn dump_smoke_high() {
        dump_smoke("h264_high.h264");
    }

    /// Decode a stream with the Rust path armed (the per-frame pixel oracle in
    /// `decode_h264_frame` byte-compares every frame against the C core), pin
    /// every output to its golden, and return the entries: one per frame
    /// (cropped I420 pixels) and one per slice (the Rust per-MB record stream).
    fn stream_golden(name: &str) -> Vec<(String, Vec<u8>)> {
        let data = sample(name);
        let mut dec = SwH264Decoder::new(data).unwrap();
        dec.arm_mb_dump();
        let mut out: Vec<(String, Vec<u8>)> = Vec::new();
        let mut frames = 0usize;
        while let Some(frame) = dec.decode().unwrap() {
            if let Some(pd) = &frame.pixel_data {
                let key = format!("stream::{name}::f{frames}");
                crate::rust::goldens::assert_golden(&key, &pd.buffer);
                crate::rust::goldens::record(&key, &pd.buffer);
                out.push((key, pd.buffer.clone()));
            }
            frames += 1;
        }
        frames += dec.flush().unwrap().len();
        assert!(frames > 0, "{name}: no frames decoded");
        for (si, rec) in dec.take_rust_records().iter().enumerate() {
            let key = format!("stream::{name}::s{si}");
            crate::rust::goldens::assert_golden(&key, rec);
            crate::rust::goldens::record(&key, rec);
            out.push((key, rec.clone()));
        }
        out
    }

    /// Tier E1e: full-stream pixel oracle over an all-intra stream. Every
    /// frame here is intra-only, so the Rust path (intra pred + transform +
    /// deblock) must match the C core byte-for-byte on every frame.
    #[test]
    fn pixel_oracle_xalli() {
        stream_golden("h264_xallI.h264");
    }

    /// Tier E2: full-stream pixel oracle over a P-slice (CAVLC) stream. Every
    /// frame is I/P, so inter MC (all P partition types) must match the C core
    /// byte-for-byte on every frame.
    #[test]
    fn pixel_oracle_baseline() {
        stream_golden("h264_baseline.h264");
    }

    /// Tier E3: full-stream pixel oracle over a mixed-GOP stream with B
    /// frames (direct mode, bi-pred). Every frame must match the C core
    /// byte-for-byte.
    #[test]
    fn pixel_oracle_main() {
        stream_golden("h264_main.h264");
    }

    /// Tier E4: full-stream pixel oracle + golden pinning for the High profile
    /// (8x8 transform, B frames).
    #[test]
    fn pixel_oracle_high() {
        stream_golden("h264_high.h264");
    }

    /// Re-run the stream oracles and return per-frame/per-slice golden entries
    /// (for `regenerate_goldens`).
    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for name in [
            "h264_xallI.h264",
            "h264_baseline.h264",
            "h264_main.h264",
            "h264_high.h264",
        ] {
            for (k, b) in stream_golden(name) {
                v.push((k, crate::rust::goldens::sha256_hex(&b)));
            }
        }
        v
    }
}
