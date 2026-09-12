//! Software H.264/AVC decoder.
//!
//! Control plane (common Rust implementations from `vacc-parser`):
//! - [`H264Parser`] — NAL / SPS / PPS / slice-header parsing
//! - [`H264Dpb`] — decoded picture buffer + reference marking
//! - [`PocCalculator`] — picture order count (types 0/1/2)
//! - `h264_reflist::build_ref_pic_lists` — spec 8.2.3.1+8.2.3.2 ref lists
//!
//! Data plane: the vendored edge264 C slice-decode routines (macroblock
//! parse, intra/inter prediction, transform, deblocking), statically linked
//! and driven per-slice through the FFI in [`crate::ffi`].

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::VecDeque;
use std::ffi::c_void;
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

use crate::ffi::{self, Sw264Frame, Sw264Pps, Sw264Slice, Sw264Sps};

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

/// Software H.264 decoder: vacc-parser control plane + edge264 C data plane.
pub struct SwH264Decoder {
    /// C decoder core (single translation unit, statically linked).
    cdec: Option<NonNull<c_void>>,
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

    // Per-DPB-slot storage: one contiguous block per slot holding the sample
    // planes (Y + Cb/Cr) immediately followed by the macroblock array —
    // mirroring edge264's internal_alloc. The inter prediction filter may
    // read up to a couple of rows past the chroma plane end; upstream relies
    // on the adjacent mb array absorbing those overreads, so the two must not
    // be separate allocations.
    n_slots: usize,
    slot_planes: Vec<Option<AlignedBuf>>,
    mb_size: usize,
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
            cdec: None,
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
            mb_size: unsafe { ffi::sw264_macroblock_size() },
            mbs_per_frame: 0,
            pending_data: data,
            parse_offset: 0,
            frame_count: 0,
            gop_count: 0,
            reorder_watermark: i64::MIN,
            pending_key: 0,
            pending_frames: VecDeque::new(),
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

        // Allocate the C decoder core once.
        if self.cdec.is_none() {
            let raw = unsafe { ffi::sw264_alloc() };
            if raw.is_null() {
                return Err(Error::Alloc);
            }
            self.cdec = Some(NonNull::new(raw).ok_or(Error::Alloc)?);
        }

        // Push the SPS to the C side and get the computed frame format.
        let c = self.cdec_raw()?;
        let sw_sps = Sw264Sps {
            chroma_format_idc: sps.chroma_format_idc as i8,
            chroma_array_type: if sps.separate_colour_plane_flag {
                0
            } else {
                sps.chroma_format_idc as i8
            },
            bit_depth_y: (8 + sps.bit_depth_luma_minus8) as i8,
            bit_depth_c: (8 + sps.bit_depth_chroma_minus8) as i8,
            pic_width_in_mbs: sps.pic_width_in_mbs_minus1 + 1,
            pic_height_in_mbs: (sps.pic_height_in_map_units_minus1 + 1) as i16,
            frame_crop_offsets: [
                sps.frame_crop_top_offset as i16,
                sps.frame_crop_right_offset as i16,
                sps.frame_crop_bottom_offset as i16,
                sps.frame_crop_left_offset as i16,
            ],
            log2_max_frame_num: (4 + sps.log2_max_frame_num_minus4) as i8,
            pic_order_cnt_type: sps.pic_order_cnt_type as i8,
            max_num_ref_frames: sps.max_num_ref_frames.min(16) as i8,
            direct_8x8_inference_flag: sps.direct_8x8_inference_flag as i8,
            seq_scaling_matrix_present: sps.seq_scaling_matrix_present_flag as i32,
            scaling_list_4x4: sps.scaling_list_4x4,
            scaling_list_8x8: sps.scaling_list_8x8,
        };
        let mut sy = 0i32;
        let mut sc = 0i32;
        let mut py = 0i32;
        let mut pc = 0i32;
        let rc =
            unsafe { ffi::sw264_set_sps(c.as_ptr(), &sw_sps, &mut sy, &mut sc, &mut py, &mut pc) };
        if rc != 0 {
            return Err(Error::SliceDecode(rc));
        }
        self.stride_y = sy as u32;
        self.stride_c = sc as u32;
        self.plane_size_y = py as u32;
        self.plane_size_c = pc as u32;

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
        self.poc_calc.reset();
        Ok(())
    }

    fn cdec_raw(&self) -> Result<NonNull<c_void>, Error> {
        self.cdec
            .ok_or(Error::InvalidState("C decoder not initialized"))
    }

    /// Size of the sample-plane part of a slot block (planes + overread margin).
    fn plane_block_size(&self) -> usize {
        // Upstream: plane_size_Y + plane_size_C + 16. Keep a slightly larger
        // margin; anything beyond it lands in the adjacent mb array anyway.
        (self.plane_size_y + self.plane_size_c + 32) as usize
    }

    /// Allocate (lazily, once) the contiguous plane + mb storage for a DPB slot.
    fn ensure_slot_storage(&mut self, slot: usize) -> Result<(), Error> {
        let total = self.plane_block_size() + self.mbs_per_frame * self.mb_size;
        if self.slot_planes[slot].is_none() {
            let buf = AlignedBuf::new(total).ok_or(Error::Alloc)?;
            // Upstream alloc_frame fills each row's trailing padding macroblock
            // slot with unavail_mb; column-0 MBs read it as their left neighbor.
            unsafe {
                ffi::sw264_init_mb_buffer(
                    buf.as_mut_ptr().add(self.plane_block_size()) as *mut c_void,
                    (self.coded_w / 16) as i32,
                    (self.coded_h / 16) as i32,
                );
            }
            // A freshly zeroed mb array has recovery_bits == 0, so reset this
            // slot's flip-bit sentinel to 0 too; otherwise stale toggle history
            // can match the fresh array and the sentinel aborts the slice after
            // 0 macroblocks (e.g. after an IDR frees and re-allocates the slot).
            let c = self.cdec_raw()?;
            unsafe { ffi::sw264_reset_slot_flip(c.as_ptr(), slot as i32) };
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
                        self.pps = Some(new_pps.clone());
                        // Push the PPS to the C side.
                        let c = self.cdec_raw()?;
                        let sw_pps = Sw264Pps {
                            entropy_coding_mode_flag: new_pps.entropy_coding_mode_flag as i8,
                            num_ref_idx_active: [
                                (new_pps.num_ref_idx_l0_default_active_minus1 + 1) as i8,
                                (new_pps.num_ref_idx_l1_default_active_minus1 + 1) as i8,
                            ],
                            weighted_pred_flag: new_pps.weighted_pred_flag as i8,
                            weighted_bipred_idc: new_pps.weighted_bipred_idc as i8,
                            qp_prime_y: (26 + new_pps.pic_init_qp_minus26) as i8,
                            chroma_qp_index_offset: new_pps.chroma_qp_index_offset as i8,
                            second_chroma_qp_index_offset: new_pps.second_chroma_qp_index_offset
                                as i8,
                            transform_8x8_mode_flag: new_pps.transform_8x8_mode_flag as i8,
                        };
                        unsafe { ffi::sw264_set_pps(c.as_ptr(), &sw_pps) };
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
    /// state and the C slice-decode core.
    fn decode_h264_frame(&mut self, slices: &[SliceEntry]) -> Result<Option<DecodedFrame>, Error> {
        let sps = self.sps.as_ref().ok_or(Error::InvalidState("no SPS"))?;
        let c = self.cdec_raw()?;

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
            slice_lists.push(self.dpb.build_ref_lists(st, l0m1, l1m1, mod_l0, mod_l1));
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

        // Long-term reference mask over DPB slots (for B-slice direct mode).
        let mut long_term_frames = 0u32;
        for (i, s) in self.dpb.slots.iter().enumerate() {
            if s.state != 0 && s.marking == MARKING_LONG {
                long_term_frames |= 1 << i;
            }
        }

        // Per-frame setup for the C core.
        let mut fr = Sw264Frame::new();
        fr.n_slots = self.n_slots as i32;
        fr.long_term_frames = long_term_frames;
        for (i, plane) in self.slot_planes.iter().enumerate() {
            if let Some(p) = plane {
                fr.planes[i] = p.as_ptr();
                fr.mb_arrays[i] =
                    unsafe { p.as_ptr().add(self.plane_block_size()) } as *const c_void;
            }
        }
        let curr = self.slot_planes[slot].as_ref().unwrap();
        fr.curr_plane = curr.as_mut_ptr();
        fr.curr_mb = unsafe { curr.as_mut_ptr().add(self.plane_block_size()) } as *mut c_void;
        fr.curr_slot = slot as i32;
        unsafe { ffi::sw264_frame_start(c.as_ptr(), &fr) };

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
            self.decode_slice(c.as_ptr(), h, &slices[idx], &slice_lists[idx], poc)?;
        }

        // Commit the picture to the common DPB (post-decode marking + display).
        self.dpb.commit_current(slot);

        // Free storage of slots that no longer hold a picture.
        for (i, s) in self.dpb.slots.iter().enumerate() {
            if s.state == 0 {
                self.slot_planes[i] = None;
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

    /// Decode one slice through the C core.
    fn decode_slice(
        &self,
        c: *mut c_void,
        h: &vacc_parser::h264::SliceHeader,
        entry: &SliceEntry,
        lists: &RefPicLists,
        poc: i32,
    ) -> Result<(), Error> {
        let pps = self.pps.as_ref().ok_or(Error::InvalidState("no PPS"))?;
        let sps = self.sps.as_ref().ok_or(Error::InvalidState("no SPS"))?;
        let st = h.slice_type % 5;

        // Slice data: raw NAL bytes (EPB intact — the C bitstream reader
        // removes emulation prevention bytes on the fly), with head/tail guard
        // bytes for the SIMD byte loader. The common parser extends a NAL with
        // the leading 0x00 of a following start code; a valid H.264 NAL never
        // ends in 0x00, so a trailing zero is always that extra byte.
        //
        // Guard bytes MUST be non-zero: get_bytes() loads a 16-byte chunk from
        // CPB-2 (a 2-byte lookback for on-the-fly EPB removal) and strips any
        // `00 00 n<=3` pattern. If the guard were zero, the first read at the
        // NAL header would see [00 00 <nal_hdr>] and — when nal_hdr <= 0x03 —
        // strip the NAL header as a false EPB, corrupting the bitstream.
        let mut nal = entry.nal_data.clone();
        if nal.last() == Some(&0) {
            nal.pop();
        }
        let mut buf = vec![0xFFu8; 16 + nal.len() + 16];
        buf[16..16 + nal.len()].copy_from_slice(&nal);

        // The C core primes its bit reader at the NAL header and skips to the
        // slice data at bit offset 8 + slice-header bits (EPB-stripped space,
        // matching the parser's header_bit_size — not necessarily byte-aligned).
        let skip_bits = 8 + h.header_bit_size as u32;
        if skip_bits >= nal.len() as u32 * 8 {
            return Err(Error::InvalidState("slice data out of bounds"));
        }

        let mut sl = Sw264Slice::new();
        sl.first_mb_in_slice = h.first_mb_in_slice;
        sl.slice_type = st as i8;
        sl.field_pic_flag = h.field_pic_flag as i8;
        sl.bottom_field_flag = (h.field_pic_flag && h.bottom_field) as i8;
        sl.nal_ref_idc_ne = (h.nal_ref_idc != 0) as i8;
        sl.direct_spatial_mv_pred_flag = h.direct_spatial_mv_pred_flag as i8;
        sl.disable_deblocking_filter_idc = h.disable_deblocking_filter_idc;
        sl.filter_offset_a =
            (h.slice_alpha_c0_offset_div2 * 2).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        sl.filter_offset_b =
            (h.slice_beta_offset_div2 * 2).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        // C expects the 1-based table index (I: 0, P/B: 1 + ue value).
        sl.cabac_init_idc = if st == 2 {
            0
        } else {
            1 + h.cabac_init_idc as i8
        };
        let qp_prime = 26 + pps.pic_init_qp_minus26;
        sl.qp_y = (qp_prime + h.slice_qp_delta).clamp(0, 51) as i16;
        sl.num_ref_idx_active = [
            (h.num_ref_idx_l0_active_minus1 + 1) as i8,
            (h.num_ref_idx_l1_active_minus1 + 1) as i8,
        ];
        let base = buf.as_ptr();
        sl.data = unsafe { base.add(16) }; // NAL start
        sl.skip_bits = skip_bits;
        sl.end = unsafe { base.add(16 + nal.len()) };

        // Reference picture lists: DPB slot indices in spec order.
        for (i, r) in lists.l0.iter().enumerate() {
            sl.refpic_list[0][i] = r.slot as i8;
        }
        for (i, r) in lists.l1.iter().enumerate() {
            sl.refpic_list[1][i] = r.slot as i8;
        }

        // Per-slot POC difference (B-slice direct mode / implicit weights).
        for (i, s) in self.dpb.slots.iter().enumerate() {
            sl.diff_poc[i] = if s.state == 0 {
                0
            } else {
                (poc - s.poc) as i16
            };
        }

        // Weighted prediction: the common parser already fills the per-ref
        // tables with explicit values or inferred defaults. Layout in the C
        // task: L0 refs at [0..n0), L1 refs at [32..32+n1).
        let has_pw_table =
            (st == 0 && pps.weighted_pred_flag) || (st == 1 && pps.weighted_bipred_idc == 1);
        if has_pw_table {
            sl.luma_log2_weight_denom = h.luma_log2_weight_denom as i8;
            sl.chroma_log2_weight_denom = h.chroma_log2_weight_denom as i8;
            let n0 = h.num_ref_idx_l0_active_minus1 as usize + 1;
            let n1 = if st == 1 {
                h.num_ref_idx_l1_active_minus1 as usize + 1
            } else {
                0
            };
            for i in 0..n0 {
                sl.explicit_weights[0][i] = h.luma_weight_l0[i];
                sl.explicit_offsets[0][i] = h.luma_offset_l0[i] as i8;
                sl.explicit_weights[1][i] = h.chroma_weight_l0[i][0];
                sl.explicit_offsets[1][i] = h.chroma_offset_l0[i][0] as i8;
                sl.explicit_weights[2][i] = h.chroma_weight_l0[i][1];
                sl.explicit_offsets[2][i] = h.chroma_offset_l0[i][1] as i8;
            }
            for i in 0..n1 {
                sl.explicit_weights[0][32 + i] = h.luma_weight_l1[i];
                sl.explicit_offsets[0][32 + i] = h.luma_offset_l1[i] as i8;
                sl.explicit_weights[1][32 + i] = h.chroma_weight_l1[i][0];
                sl.explicit_offsets[1][32 + i] = h.chroma_offset_l1[i][0] as i8;
                sl.explicit_weights[2][32 + i] = h.chroma_weight_l1[i][1];
                sl.explicit_offsets[2][32 + i] = h.chroma_offset_l1[i][1] as i8;
            }
        } else if st == 1 && pps.weighted_bipred_idc == 2 {
            // Implicit B weights: the C core computes them from diff_poc;
            // mirror FFmpeg's convention for the unused explicit tables.
            sl.luma_log2_weight_denom = 5;
            sl.chroma_log2_weight_denom = if sps.chroma_format_idc != 0 { 5 } else { 0 };
        }

        let rc = unsafe { ffi::sw264_decode_slice(c, &sl) };
        if rc != 0 {
            return Err(Error::SliceDecode(rc));
        }
        Ok(())
    }

    /// Copy the cropped display area out of a DPB slot plane into I420.
    fn extract_frame(&self, slot: usize) -> PixelData {
        let plane = self.slot_planes[slot].as_ref().expect("slot plane");
        let base = plane.as_ptr();
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
        self.pending_data.clear();
        self.parse_offset = 0;
        self.frame_count = 0;
        self.gop_count = 0;
        self.reorder_watermark = i64::MIN;
        self.pending_frames.clear();
        Ok(())
    }
}

impl Drop for SwH264Decoder {
    fn drop(&mut self) {
        if let Some(c) = self.cdec {
            unsafe { ffi::sw264_free(&mut c.as_ptr()) };
        }
    }
}
