//! NVDEC HEVC (H.265) decoder using vk-video-parser.
//!
//! Mirrors the H.264 decoder architecture (`decoder.rs`) but driven by the
//! [`H265Parser`]. Unlike H.264, the HEVC POC is already computed by the
//! parser (`SliceHeaderInfo::curr_pic_order_cnt_val`), so no separate POC
//! calculator is needed. Reference-picture-set (RPS) management is POC-based.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use vk_video_core::{
    codec::VideoCodec,
    decoder::{Decoder, DecoderInfo},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane},
    picture::{H265Pps, H265Sps},
    session::Extent2D,
};
use vk_video_parser::{h265::H265Parser, BitstreamPacket, ParseResult, VideoParser};

use crate::{
    device::{
        cu_ctx_set_current, cu_ctx_synchronize, cu_mem_free_host, cu_mem_host_alloc, cu_memcpy_2d,
        get_funcs, init_nvdec, CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST,
    },
    error::{NvdecError, NvdecResult},
    ffi::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
        cudaVideoSurfaceFormat, CUdeviceptr, CUvideodecoder, CUDA_SUCCESS, CUVIDDECODECREATEINFO,
        CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT,
    },
    picparams::{build_cuvid_hevc_picparams, dump_cuvid_hevc_picparams, H265DpbState},
};

/// Number of decode surfaces / DPB slots (matches the C reference).
const NUM_SURFACES: i32 = 16;

/// Maximum number of pictures held in the DPB array (matches the cuvid parser).
const MAX_DPB_ENTRIES: usize = 4;

/// A single entry in the DPB array (slot-indexed, matching the cuvid layout).
#[derive(Debug, Clone, Copy)]
struct H265DpbEntry {
    /// Surface where this picture was decoded.
    surface_idx: i32,
    /// Picture order count.
    poc: i32,
    /// Fill order, used for FIFO eviction (lower = older).
    fill_order: u32,
}

/// POC-based HEVC decoded picture buffer.
///
/// Mirrors the NVIDIA cuvid parser: a persistent 16-slot array where each
/// slot holds (surface, poc) or is empty. For a non-CRA picture the array is
/// set to the picture's references; for a CRA it carries over the previous
/// references plus the just-decoded picture, capped at [`MAX_DPB_ENTRIES`]
/// (FIFO eviction). `StCurrBefore`/`StCurrAfter` index into this array by
/// slot.
#[derive(Debug)]
struct H265Dpb {
    /// The DPB array, indexed by slot (0-15).
    slots: [Option<H265DpbEntry>; NUM_SURFACES as usize],
    /// Global fill counter for FIFO eviction.
    fill_counter: u32,
    /// POC physically on each surface (None if never used).
    surface_poc: [Option<i32>; NUM_SURFACES as usize],
    /// Whether each surface's frame has been extracted.
    surface_extracted: [bool; NUM_SURFACES as usize],
}

impl Default for H265Dpb {
    fn default() -> Self {
        Self {
            slots: [None; NUM_SURFACES as usize],
            fill_counter: 0,
            surface_poc: [None; NUM_SURFACES as usize],
            surface_extracted: [false; NUM_SURFACES as usize],
        }
    }
}

impl H265Dpb {
    fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self) {
        self.slots = [None; NUM_SURFACES as usize];
        self.fill_counter = 0;
        self.surface_poc = [None; NUM_SURFACES as usize];
        self.surface_extracted = [false; NUM_SURFACES as usize];
    }

    /// Is surface `s` held in the DPB array (a live reference)?
    fn is_live_ref(&self, s: i32) -> bool {
        self.slots
            .iter()
            .any(|e| e.as_ref().map(|e| e.surface_idx == s).unwrap_or(false))
    }

    /// Find the surface physically holding the given POC.
    fn surface_of(&self, poc: i32) -> Option<i32> {
        self.surface_poc
            .iter()
            .position(|p| *p == Some(poc))
            .map(|i| i as i32)
    }

    /// Update the DPB array to hold exactly `ref_pocs` (the current picture's
    /// references). Slots for references still present are kept (sticky);
    /// stale slots are cleared; new references are added to empty slots.
    fn set_references(&mut self, ref_pocs: &[i32]) {
        for slot in self.slots.iter_mut() {
            if let Some(e) = slot {
                if !ref_pocs.contains(&e.poc) {
                    *slot = None;
                }
            }
        }
        for &poc in ref_pocs {
            if self
                .slots
                .iter()
                .any(|s| s.as_ref().map(|e| e.poc == poc).unwrap_or(false))
            {
                continue;
            }
            let surface = match self.surface_of(poc) {
                Some(s) => s,
                None => continue,
            };
            if let Some((i, slot)) = self.slots.iter_mut().enumerate().find(|(_, s)| s.is_none()) {
                *slot = Some(H265DpbEntry {
                    surface_idx: surface,
                    poc,
                    fill_order: self.fill_counter,
                });
                self.fill_counter += 1;
            }
        }
    }

    /// Add the previous picture (a CRA's carry-over) to the DPB array, capping
    /// at [`MAX_DPB_ENTRIES`] (evicting the oldest, FIFO, when over the cap).
    fn add_prev_picture(&mut self, surface_idx: i32, poc: i32) {
        self.surface_poc[surface_idx as usize] = Some(poc);
        if let Some((_, slot)) = self.slots.iter_mut().enumerate().find(|(_, s)| s.is_none()) {
            *slot = Some(H265DpbEntry {
                surface_idx,
                poc,
                fill_order: self.fill_counter,
            });
            self.fill_counter += 1;
        }
        while self.count() > MAX_DPB_ENTRIES {
            self.evict_oldest();
        }
    }

    fn count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Evict the longest-held slot (lowest fill_order = oldest, FIFO).
    fn evict_oldest(&mut self) {
        let mut oldest: Option<(usize, u32)> = None;
        for (i, e) in self.slots.iter().enumerate() {
            if let Some(en) = e {
                if oldest.map(|(_, o)| en.fill_order < o).unwrap_or(true) {
                    oldest = Some((i, en.fill_order));
                }
            }
        }
        if let Some((i, _)) = oldest {
            self.slots[i] = None;
        }
    }

    /// Choose the surface index for the next picture (`CurrPicIdx`): the
    /// lowest-index surface that is not a live reference and has no pending
    /// (unextracted) frame.
    fn choose_surface(&self) -> i32 {
        for s in 0..NUM_SURFACES as usize {
            if self.is_live_ref(s as i32) {
                continue;
            }
            if self.surface_poc[s].is_some() && !self.surface_extracted[s] {
                continue;
            }
            return s as i32;
        }
        // Fallback: lowest-index non-live surface.
        for s in 0..NUM_SURFACES as usize {
            if !self.is_live_ref(s as i32) {
                return s as i32;
            }
        }
        0
    }

    /// Record that a picture was decoded to the given surface. The surface now
    /// holds a pending (unextracted) frame, so clear the extracted flag;
    /// otherwise a recycled surface would look free and be clobbered.
    fn note_decoded(&mut self, surface_idx: i32, poc: i32) {
        self.surface_poc[surface_idx as usize] = Some(poc);
        self.surface_extracted[surface_idx as usize] = false;
    }

    /// Mark the surface's frame as extracted (so the surface may be recycled).
    fn mark_extracted(&mut self, surface_idx: i32) {
        if self.surface_poc[surface_idx as usize].is_some() {
            self.surface_extracted[surface_idx as usize] = true;
        }
    }

    /// Find the DPB slot index holding the given POC.
    fn find_ref_slot(&self, poc: i32) -> Option<usize> {
        self.slots
            .iter()
            .position(|e| e.as_ref().map(|e| e.poc == poc).unwrap_or(false))
    }

    /// Build the CUVID HEVC DPB state.
    ///
    /// The arrays are indexed by **slot** (0-15), matching the cuvid layout:
    /// - `ref_pic_idx[slot]` = surface index of the picture in that slot
    /// - `pic_order_cnt_val[slot]` = POC of the picture in that slot
    /// - `st_curr_before[j]` / `st_curr_after[j]` = **slot index** of the
    ///   j-th L0/L1 reference
    fn build_state(
        &self,
        ref_s0: &[i32],
        ref_s1: &[i32],
        num_bits: i32,
        curr_poc: i32,
    ) -> H265DpbState {
        let mut state = H265DpbState::default();

        for (slot, e) in self.slots.iter().enumerate() {
            match e {
                Some(en) => {
                    state.ref_pic_idx[slot] = en.surface_idx;
                    state.pic_order_cnt_val[slot] = en.poc;
                    state.is_long_term[slot] = 0;
                }
                None => {
                    state.ref_pic_idx[slot] = -1;
                    state.pic_order_cnt_val[slot] = 0;
                    state.is_long_term[slot] = 0;
                }
            }
        }

        for (j, &poc) in ref_s0.iter().enumerate().take(8) {
            state.st_curr_before[j] = self.find_ref_slot(poc).unwrap_or(0) as u8;
        }
        for (j, &poc) in ref_s1.iter().enumerate().take(8) {
            state.st_curr_after[j] = self.find_ref_slot(poc).unwrap_or(0) as u8;
        }

        state.num_poc_st_curr_before = ref_s0.len() as i32;
        state.num_poc_st_curr_after = ref_s1.len() as i32;
        state.num_poc_lt_curr = 0;
        state.num_poc_total_curr = ref_s0.len() as i32 + ref_s1.len() as i32;
        state.num_bits_for_short_term_rps_in_slice = num_bits;
        state.num_delta_pocs_of_ref_rps_idx = 0;
        state.curr_pic_order_cnt_val = curr_poc;
        state
    }
}

/// Compute the number of bits used to code a short-term reference picture set
/// in the slice header. Handles both direct and predictive RPS.
///
/// `rps` is the reference picture set from the parser. `curr_poc` is the
/// current picture's POC. `ref_s0` / `ref_s1` are the reference POCs in RPS order.
fn hevc_rps_bit_count(
    curr_poc: i32,
    ref_s0: &[i32],
    ref_s1: &[i32],
    rps: Option<&vk_video_core::picture::H265ShortTermRefPicSet>,
) -> i32 {
    fn ue(v: u32) -> u32 {
        if v == 0 {
            return 1;
        }
        let n = v + 1;
        let k = (32 - n.leading_zeros()) as u32;
        2 * k - 1
    }

    let rps = match rps {
        Some(r) => r,
        None => return 0,
    };

    // Predictive RPS: different syntax
    if rps.inter_ref_pic_set_prediction_flag {
        let mut bits: u32 = 0;
        bits += ue(rps.delta_idx_minus1); // delta_idx_minus1
        bits += ue(rps.abs_delta_rps_minus1 as u32); // abs_delta_rps_minus1
        bits += 1; // delta_rps_sign
                   // For each entry in the reference RPS + 1, there's a use_delta_flag
                   // and potentially a used_by_curr_pic_flag
        let ref_rps_idx = rps.delta_idx_minus1 as usize;
        // We can't easily compute the exact reference RPS size here,
        // so use a conservative estimate based on the current RPS
        let num_entries = rps.num_negative_pics as usize + rps.num_positive_pics as usize;
        for _ in 0..=num_entries {
            bits += 1; // use_delta_flag
            bits += 1; // used_by_curr_pic_flag (when use_delta_flag == 1)
        }
        return bits as i32;
    }

    // Direct RPS
    direct_rps_bit_count(curr_poc, ref_s0, ref_s1)
}

/// Compute the direct (non-predictive) short-term RPS bit count for the given
/// reference POCs. `ref_s0` must be sorted by decreasing POC (so delta_poc_s0
/// is increasing); `ref_s1` by increasing POC (so delta_poc_s1 is increasing).
fn direct_rps_bit_count(curr_poc: i32, ref_s0: &[i32], ref_s1: &[i32]) -> i32 {
    fn ue(v: u32) -> u32 {
        if v == 0 {
            return 1;
        }
        let n = v + 1;
        let k = (32 - n.leading_zeros()) as u32;
        2 * k - 1
    }

    let d0: Vec<i32> = ref_s0.iter().map(|&r| curr_poc - r).collect();
    let d1: Vec<i32> = ref_s1.iter().map(|&r| r - curr_poc).collect();

    let mut c0: Vec<i32> = Vec::with_capacity(d0.len());
    for (i, &d) in d0.iter().enumerate() {
        c0.push(if i == 0 { d - 1 } else { d - d0[i - 1] - 1 });
    }
    let mut c1: Vec<i32> = Vec::with_capacity(d1.len());
    for (i, &d) in d1.iter().enumerate() {
        c1.push(if i == 0 { d - 1 } else { d - d1[i - 1] - 1 });
    }

    let mut bits: u32 = ue(d0.len() as u32) + ue(d1.len() as u32);
    for &c in &c0 {
        bits += ue(c as u32) + 1;
    }
    for &c in &c1 {
        bits += ue(c as u32) + 1;
    }
    bits as i32
}

/// NVDEC HEVC decoder using vk-video-parser.
///
/// Not `Send`/`Sync`; use from a single thread. The CUDA context must be set
/// current before decode methods.
pub struct NvdecH265Decoder {
    parser: H265Parser,
    decoder: Mutex<CUvideodecoder>,
    info: Mutex<DecoderInfo>,
    pending_frames: Mutex<VecDeque<DecodedFrame>>,
    frame_count: Mutex<u32>,
    display_area: Mutex<(i32, i32, i32, i32)>,
    initialized: Mutex<bool>,
    dpb: Mutex<H265Dpb>,
    /// DPB state submitted to the decoder for the previous picture. A CRA
    /// (IRAP, non-IDR) carries this forward unchanged — stale refs are
    /// retained across a CRA and only an IDR resets the DPB (matches the
    /// cuvid parser, which reports the previous picture's DPB at a CRA).
    prev_dpb_state: Mutex<H265DpbState>,
    prev_coded_size: Mutex<(u32, u32)>,
    pending_data: Vec<u8>,
    parsed_offset: usize,

    /// Reorder buffer for display-order presentation: (unwrapped_poc, seq) ->
    /// (surface_idx, seq, poc).
    reorder: BTreeMap<(i32, i32), (i32, i32, i32)>,
    presented_count: u32,
    seq_counter: i32,

    /// POC wrap period (= 2^(log2_max_pic_order_cnt_lsb_minus4 + 4)).
    poc_period: i32,
    poc_cycle: i32,
    prev_decoded_poc: Option<i32>,
    last_presented_unwrapped: Option<i32>,
    /// Minimum POC gap observed between consecutively decoded frames.
    /// Used to determine the stream's POC increment for reorder buffering.
    min_poc_gap: i32,

    /// (surface, poc) of the most recently decoded picture, used for CRA
    /// carry-over (a CRA adds the previous picture to the DPB array).
    /// (surface, poc, slice_type) of the last decoded picture.
    last_decoded: Option<(i32, i32, u8)>,

    /// Cached pinned host buffer for frame extraction.
    pinned_cache: Mutex<Option<(*mut std::ffi::c_void, usize)>>,

    /// If set (via `NVDEC_DUMP_PARAMS`), dump the exact [`CUVIDPICPARAMS`]
    /// submitted for each picture (DECODE order) to this path.
    dump_params_path: Option<std::path::PathBuf>,
    dump_params_count: u32,
    /// If set (via `NVDEC_DUMP_DECODE_ORDER`), dump each decoded picture
    /// (DECODE order) as NV12 to `{path}_{N}.yuv`.
    dump_decode_order_path: Option<std::path::PathBuf>,
    dump_decode_order_count: u32,
}

impl NvdecH265Decoder {
    /// Create a new NVDEC HEVC decoder and begin decoding the input data.
    pub fn new(data: Vec<u8>) -> NvdecResult<Self> {
        init_nvdec()?;

        let mut decoder = Self {
            parser: H265Parser::new(),
            decoder: Mutex::new(std::ptr::null_mut()),
            info: Mutex::new(DecoderInfo {
                backend: "nvdec".to_string(),
                codec: VideoCodec::DecodeH265,
                coded_size: Extent2D {
                    width: 0,
                    height: 0,
                },
                display_size: Extent2D {
                    width: 0,
                    height: 0,
                },
                chroma_subsampling: ChromaSubsampling::_420,
                luma_bit_depth: ComponentBitDepth::Bit8,
                chroma_bit_depth: ComponentBitDepth::Bit8,
                profile_idc: None,
                dpb_slots: NUM_SURFACES as u32,
            }),
            pending_frames: Mutex::new(VecDeque::new()),
            frame_count: Mutex::new(0),
            display_area: Mutex::new((0, 0, 0, 0)),
            initialized: Mutex::new(false),
            dpb: Mutex::new(H265Dpb::new()),
            prev_dpb_state: Mutex::new(H265DpbState::default()),
            prev_coded_size: Mutex::new((0, 0)),
            pending_data: data,
            parsed_offset: 0,
            reorder: BTreeMap::new(),
            presented_count: 0,
            seq_counter: 0,
            poc_period: 0,
            poc_cycle: 0,
            prev_decoded_poc: None,
            last_presented_unwrapped: None,
            min_poc_gap: 1,
            last_decoded: None,
            pinned_cache: Mutex::new(None),
            dump_params_path: std::env::var("NVDEC_DUMP_PARAMS")
                .ok()
                .map(std::path::PathBuf::from),
            dump_params_count: 0,
            dump_decode_order_path: std::env::var("NVDEC_DUMP_DECODE_ORDER")
                .ok()
                .map(std::path::PathBuf::from),
            dump_decode_order_count: 0,
        };

        decoder.init_parser_format()?;
        decoder.parse_and_decode()?;

        let initialized = *decoder.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not initialize decoder - no SPS found".into(),
            ));
        }

        Ok(decoder)
    }

    /// Initialize the parser with the HEVC format (required before parsing).
    fn init_parser_format(&mut self) -> NvdecResult<()> {
        self.parser
            .init(&vk_video_parser::DetectedVideoFormat::new(
                VideoCodec::DecodeH265,
            ))
            .map_err(|e| NvdecError::DecodeFailed(format!("parser init: {}", e)))
    }

    /// Parse pending data and decode any available frames.
    fn parse_and_decode(&mut self) -> NvdecResult<()> {
        if self.parsed_offset >= self.pending_data.len() {
            return Ok(());
        }

        let remaining = &self.pending_data[self.parsed_offset..];
        let packet = BitstreamPacket::new(remaining.to_vec());

        // Per-picture decode status logging, off unless NVDEC_DEBUG_STATUS is set.
        let debug_status = std::env::var("NVDEC_DEBUG_STATUS").is_ok();

        loop {
            match self.parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, .. }) => {
                    if let Some(sps_box) = sps {
                        if let Some(h265_sps) = sps_box.downcast_ref::<H265Sps>() {
                            let (prev_w, prev_h) = {
                                let s = self.prev_coded_size.lock().unwrap();
                                *s
                            };
                            // Compute the coded size using the EXACT same CTB
                            // rounding as create_decoder (see below). Using a
                            // 16-pixel rounding here (as before) disagreed with
                            // create_decoder's CTB rounding, so a re-emitted SPS
                            // with identical dimensions (e.g. the one x265 writes
                            // before each CRA) was misdetected as a resolution
                            // change, triggering recreate_decoder and wiping the
                            // DPB's stale refs that a CRA needs to decode.
                            let log2_ctb_size = h265_sps.log2_min_luma_coding_block_size_minus3
                                as u32
                                + h265_sps.log2_diff_max_min_luma_coding_block_size as u32
                                + 2;
                            let ctb_size = 1u32 << log2_ctb_size;
                            let coded_width =
                                ((h265_sps.pic_width_in_luma_samples as u32 + ctb_size - 1)
                                    / ctb_size)
                                    * ctb_size;
                            let coded_height =
                                ((h265_sps.pic_height_in_luma_samples as u32 + ctb_size - 1)
                                    / ctb_size)
                                    * ctb_size;
                            let resolution_changed =
                                prev_w != coded_width || prev_h != coded_height;
                            if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                                eprintln!(
                                    "[sps] new SPS {}x{} prev={}x{} resolution_changed={}",
                                    coded_width, coded_height, prev_w, prev_h, resolution_changed
                                );
                            }
                            if resolution_changed {
                                self.recreate_decoder(h265_sps)?;
                            } else {
                                let decoder_handle = {
                                    let d = self.decoder.lock().unwrap();
                                    *d
                                };
                                if decoder_handle.is_null() {
                                    self.create_decoder(h265_sps)?;
                                }
                            }
                            // POC wrap period from the SPS.
                            self.poc_period =
                                1 << (h265_sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
                        }
                    }
                }
                Ok(ParseResult::Slice { slices, .. }) => {
                    if slices.is_empty() {
                        break;
                    }
                    if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                        let dpb = self.dpb.lock().unwrap();
                        let occ: Vec<i32> = dpb
                            .slots
                            .iter()
                            .map(|s| s.as_ref().map(|e| e.poc).unwrap_or(-1))
                            .collect();
                        eprintln!("[dpb-start] slots={:?}", occ);
                    }

                    let info = if let Some(vk_video_parser::SliceHeader::H265(info)) =
                        slices[0].slice_header.as_ref()
                    {
                        info
                    } else {
                        break;
                    };

                    let sps = self
                        .parser
                        .active_sps()
                        .ok_or_else(|| NvdecError::DecodeFailed("No active SPS".into()))?;
                    let pps = self
                        .parser
                        .active_pps()
                        .ok_or_else(|| NvdecError::DecodeFailed("No active PPS".into()))?;

                    let poc = info.curr_pic_order_cnt_val;

                    // Reset DPB on IRAP (IDR) pictures.
                    if info.is_idr {
                        let mut dpb = self.dpb.lock().unwrap();
                        dpb.reset();
                        self.poc_cycle = 0;
                        self.prev_decoded_poc = None;
                        self.last_decoded = None;
                    }
                    if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                        let dpb = self.dpb.lock().unwrap();
                        let occ: Vec<i32> = dpb
                            .slots
                            .iter()
                            .map(|s| s.as_ref().map(|e| e.poc).unwrap_or(-1))
                            .collect();
                        eprintln!(
                            "[dpb-after-idr-check] is_idr={} is_rap={} slots={:?}",
                            info.is_idr, info.is_rap, occ
                        );
                    }

                    // Recover the RPS reference POCs. IDR pictures carry no RPS
                    // in the slice header, so the RPS bit count is 0.
                    let (ref_s0, ref_s1) = Self::recover_rps_pocs(sps, info);
                    // A CRA (IRAP, non-IDR) has an empty RPS in the slice header.
                    let is_cra = !info.is_idr && ref_s0.is_empty() && ref_s1.is_empty();
                    if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                        let rps_dbg = if info.short_term_ref_pic_set_sps_flag {
                            sps.short_term_ref_pic_sets
                                .get(info.short_term_ref_pic_set_idx as usize)
                        } else {
                            info.slice_strps.as_ref()
                        };
                        if let Some(r) = rps_dbg {
                            let s0: Vec<i32> = (0..r.num_negative_pics as usize)
                                .map(|i| {
                                    let st = r.delta_poc_s0_minus1[i] as i32;
                                    if st > 32767 {
                                        st - 65536
                                    } else {
                                        st
                                    }
                                })
                                .collect();
                            let s1: Vec<i32> = (0..r.num_positive_pics as usize)
                                .map(|i| {
                                    let st = r.delta_poc_s1_minus1[i] as i32;
                                    if st > 32767 {
                                        st - 65536
                                    } else {
                                        st
                                    }
                                })
                                .collect();
                            eprintln!(
                                "[rps] poc={} sps_flag={} idx={} inter={} numneg={} numpos={} s0={:?} s1={:?} used0={:b} used1={:b} -> ref_s0={:?} ref_s1={:?}",
                                poc, info.short_term_ref_pic_set_sps_flag,
                                info.short_term_ref_pic_set_idx, r.inter_ref_pic_set_prediction_flag,
                                r.num_negative_pics, r.num_positive_pics, s0, s1,
                                r.used_by_curr_pic_s0_flag, r.used_by_curr_pic_s1_flag,
                                ref_s0, ref_s1
                            );
                        }
                    }
                    // Update the DPB array to match this picture, compute the RPS
                    // bit count, and choose the decode surface (CurrPicIdx).
                    //
                    // - IDR: the DPB was already reset above (empty array);
                    //   num_bits = 0.
                    // - CRA: carry over the previous references plus the
                    //   just-decoded picture (capped at MAX_DPB_ENTRIES); the
                    //   num_bits is computed from the resulting DPB array
                    //   (matching the cuvid parser).
                    // - Other: set the array to this picture's references; the
                    //   num_bits is computed from the slice RPS.
                    let (curr_pic_idx, dpb_state) = {
                        let mut dpb = self.dpb.lock().unwrap();
                        if !info.is_idr {
                            if is_cra {
                                // Carry the previous picture into the CRA's DPB
                                // only if it is a reference (non-B slice); the
                                // cuvid parser keeps the DPB unchanged when the
                                // previous picture is a B-frame (not referenced
                                // by the following picture).
                                if let Some((prev_surf, prev_poc, prev_slice)) = self.last_decoded {
                                    if prev_slice != 2 {
                                        dpb.add_prev_picture(prev_surf, prev_poc);
                                    }
                                }
                            } else {
                                let all_refs: Vec<i32> =
                                    ref_s0.iter().chain(ref_s1.iter()).copied().collect();
                                dpb.set_references(&all_refs);
                            }
                        }
                        let num_bits = if info.is_idr {
                            0
                        } else if is_cra {
                            // The cuvid parser computes the CRA's
                            // NumBitsForShortTermRPSInSlice from the carried-over
                            // DPB array (previous references plus the
                            // just-decoded picture), not the empty intra RPS.
                            let mut s0: Vec<i32> = Vec::new();
                            let mut s1: Vec<i32> = Vec::new();
                            for slot in dpb.slots.iter() {
                                if let Some(e) = slot {
                                    if e.poc < poc {
                                        s0.push(e.poc);
                                    } else if e.poc > poc {
                                        s1.push(e.poc);
                                    }
                                }
                            }
                            s0.sort_by(|a, b| b.cmp(a));
                            s1.sort();
                            if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                                eprintln!("[dbg-cra] poc={} dpb_s0={:?} dpb_s1={:?}", poc, s0, s1);
                            }
                            direct_rps_bit_count(poc, &s0, &s1)
                        } else {
                            let rps = if info.short_term_ref_pic_set_sps_flag {
                                let idx = info.short_term_ref_pic_set_idx as usize;
                                sps.short_term_ref_pic_sets.get(idx)
                            } else {
                                info.slice_strps.as_ref()
                            };
                            hevc_rps_bit_count(poc, &ref_s0, &ref_s1, rps)
                        };
                        if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                            eprintln!(
                                "[dbg] poc={} is_idr={} is_cra={} slice_type={} rps_sps_flag={} rps_idx={} ref_s0={:?} ref_s1={:?} num_bits={}",
                                poc, info.is_idr, is_cra, info.slice_type,
                                info.short_term_ref_pic_set_sps_flag, info.short_term_ref_pic_set_idx,
                                ref_s0, ref_s1, num_bits
                            );
                        }
                        let curr_pic_idx = dpb.choose_surface();
                        let dpb_state = dpb.build_state(&ref_s0, &ref_s1, num_bits, poc);
                        (curr_pic_idx, dpb_state)
                    };
                    // Remember this picture's DPB state so a following CRA can
                    // carry its stale entries forward.
                    {
                        *self.prev_dpb_state.lock().unwrap() = dpb_state;
                    }
                    if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                        let dpb = self.dpb.lock().unwrap();
                        let occupied: Vec<i32> = dpb
                            .slots
                            .iter()
                            .map(|s| s.as_ref().map(|e| e.poc).unwrap_or(-1))
                            .collect();
                        eprintln!("[dpb] poc={} slots={:?}", poc, occupied);
                    }

                    // Build the bitstream: slice NALs only, each prefixed with a
                    // 3-byte Annex-B start code (00 00 01). HEVC does NOT prepend
                    // VPS/SPS/PPS (unlike H.264); those are conveyed via the
                    // CUVIDHEVCPICPARAMS struct fields.
                    // NAL data from the parser already includes trailing_zero_8bits,
                    // so no extra byte is needed.
                    let mut bitstream_data = Vec::with_capacity(
                        slices.iter().map(|s| s.nal_data.len() + 3).sum::<usize>(),
                    );
                    let mut slice_offsets = Vec::with_capacity(slices.len());
                    for slice_entry in &slices {
                        slice_offsets.push(bitstream_data.len() as u32);
                        bitstream_data.extend_from_slice(&[0u8, 0, 1]);
                        bitstream_data.extend_from_slice(&slice_entry.nal_data);
                    }

                    let picparams = build_cuvid_hevc_picparams(
                        sps,
                        pps,
                        info,
                        curr_pic_idx,
                        &bitstream_data,
                        &slice_offsets,
                        slices.len() as u32,
                        &dpb_state,
                    );

                    let decoder_handle = {
                        let d = self.decoder.lock().unwrap();
                        if d.is_null() {
                            break;
                        }
                        *d
                    };

                    let funcs = get_funcs()?;
                    let _ = cu_ctx_set_current();

                    let dump_idx = self.dump_params_count;
                    if let Some(dump_path) = &self.dump_params_path {
                        dump_cuvid_hevc_picparams(dump_path, dump_idx, &picparams);
                        self.dump_params_count += 1;
                    }

                    // DEBUG: dump the reference surfaces (as the decoder will see
                    // them) BEFORE decoding, when NVDEC_DUMP_REFS is set.
                    if std::env::var("NVDEC_DUMP_REFS").is_ok() {
                        for (list, arr) in [
                            ("l0", &dpb_state.st_curr_before),
                            ("l1", &dpb_state.st_curr_after),
                        ] {
                            let n = if list == "l0" {
                                dpb_state.num_poc_st_curr_before
                            } else {
                                dpb_state.num_poc_st_curr_after
                            };
                            for j in 0..n as usize {
                                let slot = arr[j] as usize;
                                let surf = dpb_state.ref_pic_idx[slot] as i32;
                                let ref_poc = dpb_state.pic_order_cnt_val[slot];
                                let path = format!(
                                    "/tmp/refdump_p{}_{}_s{}_poc{}.yuv",
                                    dump_idx, list, surf, ref_poc
                                );
                                self.dump_surface_to(&path, surf);
                            }
                        }
                    }

                    let result = unsafe {
                        (funcs.decode_picture)(decoder_handle as *mut std::ffi::c_void, &picparams)
                    };
                    if result != CUDA_SUCCESS {
                        return Err(NvdecError::DecodeFailed(format!(
                            "cuvidDecodePicture failed: {}",
                            result
                        )));
                    }

                    let _ = cu_ctx_synchronize();

                    if self.dump_decode_order_path.is_some() {
                        self.dump_decode_order_frame(curr_pic_idx, self.dump_decode_order_count);
                        self.dump_decode_order_count += 1;
                    }

                    // Poll decode status until completion.
                    let mut decode_status = crate::ffi::CUVIDGETDECODESTATUS {
                        decodeStatus: crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Invalid,
                        reserved: [0; 31],
                        pReserved: [std::ptr::null_mut(); 8],
                    };
                    let mut status_result: u32 = 0;
                    for _ in 0..100 {
                        status_result = unsafe {
                            (funcs.get_decode_status)(
                                decoder_handle as *mut std::ffi::c_void,
                                curr_pic_idx as std::os::raw::c_int,
                                &mut decode_status,
                            )
                        };
                        if decode_status.decodeStatus
                            != crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_InProgress
                        {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        let _ = cu_ctx_synchronize();
                    }
                    // DEBUG: log per-picture decode status, gated by NVDEC_DEBUG_STATUS.
                    if debug_status {
                        let st = decode_status.decodeStatus;
                        let st_name = match st {
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Invalid => "Invalid",
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_InProgress => {
                                "InProgress"
                            }
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Success => "Success",
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Error => "Error",
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Error_Concealed => {
                                "Error_Concealed"
                            }
                        };
                        eprintln!(
                            "[decode] pic={} surf={} poc={} status={} ({}) api_result={}",
                            self.dump_params_count - 1,
                            curr_pic_idx,
                            poc,
                            st as u32,
                            st_name,
                            status_result
                        );
                    }

                    // Record the decoded picture's surface (for reference
                    // lookup and CRA carry-over).
                    let seq = self.seq_counter;
                    self.seq_counter += 1;
                    {
                        let mut dpb = self.dpb.lock().unwrap();
                        dpb.note_decoded(curr_pic_idx, poc);
                    }
                    self.last_decoded = Some((curr_pic_idx, poc, info.slice_type));
                    if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
                        let dpb = self.dpb.lock().unwrap();
                        let occ: Vec<i32> = dpb
                            .slots
                            .iter()
                            .map(|s| s.as_ref().map(|e| e.poc).unwrap_or(-1))
                            .collect();
                        eprintln!("[dpb-after-add] poc={} slots={:?}", poc, occ);
                    }

                    // Track for display-order presentation.
                    let unwrapped = self.unwrapped_poc(poc);
                    self.reorder
                        .insert((unwrapped, seq), (curr_pic_idx, seq, unwrapped));

                    self.extract_ready_frames();
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
                Err(e) => return Err(NvdecError::DecodeFailed(format!("Parse error: {}", e))),
            }
        }

        self.parsed_offset = self.pending_data.len();
        Ok(())
    }

    /// Recover the RPS reference POCs (s0 = before, s1 = after) from the slice
    /// header info. Handles both slice-level and SPS-level RPS. Filters by
    /// used_by_curr_pic_*_flag to only include pictures actually used as references.
    fn recover_rps_pocs(
        sps: &H265Sps,
        info: &vk_video_parser::h265::SliceHeaderInfo,
    ) -> (Vec<i32>, Vec<i32>) {
        // Select the RPS: slice-level or SPS-level
        let rps = if info.short_term_ref_pic_set_sps_flag {
            let idx = info.short_term_ref_pic_set_idx as usize;
            sps.short_term_ref_pic_sets.get(idx)
        } else {
            info.slice_strps.as_ref()
        };
        let rps = match rps {
            Some(r) => r,
            None => return (Vec::new(), Vec::new()),
        };

        let mut ref_s0 = Vec::new();
        let mut ref_s1 = Vec::new();

        // S0: negative POC deltas (references before current picture)
        for i in 0..rps.num_negative_pics as usize {
            // Filter by used_by_curr_pic_s0_flag
            if ((rps.used_by_curr_pic_s0_flag >> i) & 1) == 0 {
                continue;
            }
            let stored = rps.delta_poc_s0_minus1[i] as i32;
            let signed = if stored > 32767 {
                stored - 65536
            } else {
                stored
            };
            ref_s0.push(info.curr_pic_order_cnt_val + signed);
        }

        // S1: positive POC deltas (references after current picture)
        for i in 0..rps.num_positive_pics as usize {
            // Filter by used_by_curr_pic_s1_flag
            if ((rps.used_by_curr_pic_s1_flag >> i) & 1) == 0 {
                continue;
            }
            let stored = rps.delta_poc_s1_minus1[i] as i32;
            let signed = if stored > 32767 {
                stored - 65536
            } else {
                stored
            };
            ref_s1.push(info.curr_pic_order_cnt_val + signed);
        }

        (ref_s0, ref_s1)
    }

    /// Create the NVDEC decoder from SPS parameters.
    fn create_decoder(&mut self, sps: &H265Sps) -> NvdecResult<()> {
        // Calculate CTB (Coding Tree Block) size for proper surface alignment.
        // Matches CUVID's alignment: surfaces are CTB-aligned, not 16-pixel aligned.
        let log2_ctb_size = sps.log2_min_luma_coding_block_size_minus3 as u32
            + sps.log2_diff_max_min_luma_coding_block_size as u32
            + 2;
        let ctb_size = 1u32 << log2_ctb_size;

        let pic_width = sps.pic_width_in_luma_samples as u32;
        let pic_height = sps.pic_height_in_luma_samples as u32;

        // Decoder surfaces must be CTB-aligned (matches CUVID).
        let coded_width = ((pic_width + ctb_size - 1) / ctb_size) * ctb_size;
        let coded_height = ((pic_height + ctb_size - 1) / ctb_size) * ctb_size;

        // Display area: the actual picture region within the CTB-padded surface.
        // The surface is CTB-aligned (coded_width x coded_height), but the real
        // picture is pic_width x pic_height. Apply the SPS conformance window
        // crop (offsets are in chroma-sample units; scale to luma samples by the
        // chroma subsampling factor).
        let (sub_w, sub_h) = match sps.chroma_format_idc {
            0 => (1, 1), // 4:0:0
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            _ => (1, 1), // 4:4:4
        };
        let crop_left = if sps.conformance_window_flag {
            sps.conf_win_left_offset * sub_w
        } else {
            0
        };
        let crop_top = if sps.conformance_window_flag {
            sps.conf_win_top_offset * sub_h
        } else {
            0
        };
        let crop_right = if sps.conformance_window_flag {
            sps.conf_win_right_offset * sub_w
        } else {
            0
        };
        let crop_bottom = if sps.conformance_window_flag {
            sps.conf_win_bottom_offset * sub_h
        } else {
            0
        };

        let display_left = crop_left as i32;
        let display_top = crop_top as i32;
        let display_right = (pic_width - crop_right) as i32;
        let display_bottom = (pic_height - crop_bottom) as i32;
        let display_width = (display_right - display_left) as u32;
        let display_height = (display_bottom - display_top) as u32;

        let output_format = if sps.bit_depth_luma_minus8 > 0 {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
        } else {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        };

        let create_info = CUVIDDECODECREATEINFO {
            ulWidth: coded_width as _,
            ulHeight: coded_height as _,
            ulNumDecodeSurfaces: NUM_SURFACES as _,
            CodecType: cudaVideoCodec::cudaVideoCodec_HEVC,
            ChromaFormat: match sps.chroma_format_idc {
                0 => cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
                1 => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
                2 => cudaVideoChromaFormat::cudaVideoChromaFormat_422,
                3 => cudaVideoChromaFormat::cudaVideoChromaFormat_444,
                _ => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            },
            ulCreationFlags: cudaVideoCreateFlags::cudaVideoCreate_Default as _,
            bitDepthMinus8: sps.bit_depth_luma_minus8 as _,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: coded_width as _,
            ulMaxHeight: coded_height as _,
            Reserved1: 0,
            // The decoder's display_area must match ulTargetWidth/Height (the
            // CTB-aligned coded size) or cuvid scales the output. The actual
            // picture crop is applied during readback via `display_area` below.
            display_area: CUVIDRECT {
                left: 0,
                top: 0,
                right: coded_width as _,
                bottom: coded_height as _,
            },
            OutputFormat: output_format,
            DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
            ulTargetWidth: coded_width as _,
            ulTargetHeight: coded_height as _,
            ulNumOutputSurfaces: NUM_SURFACES as _,
            vidLock: std::ptr::null_mut(),
            target_rect: CUVIDRECT {
                left: 0,
                top: 0,
                right: coded_width as _,
                bottom: coded_height as _,
            },
            enableHistogram: 0,
            Reserved2: [0; 4],
        };

        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();

        let mut ph_decoder: CUvideodecoder = std::ptr::null_mut();
        let result = unsafe { (funcs.create_decoder)(&mut ph_decoder, &create_info) };
        if result != CUDA_SUCCESS || ph_decoder.is_null() {
            return Err(NvdecError::DecoderCreationFailed(format!(
                "cuvidCreateDecoder failed with error {}",
                result
            )));
        }
        {
            let mut decoder = self.decoder.lock().unwrap();
            *decoder = ph_decoder;
        }

        let mut info = self.info.lock().unwrap();
        *info = DecoderInfo {
            backend: "nvdec".to_string(),
            codec: VideoCodec::DecodeH265,
            coded_size: Extent2D {
                width: coded_width,
                height: coded_height,
            },
            display_size: Extent2D {
                width: display_width,
                height: display_height,
            },
            chroma_subsampling: match sps.chroma_format_idc {
                0 => ChromaSubsampling::Monochrome,
                1 => ChromaSubsampling::_420,
                2 => ChromaSubsampling::_422,
                3 => ChromaSubsampling::_444,
                _ => ChromaSubsampling::_420,
            },
            luma_bit_depth: match sps.bit_depth_luma_minus8 {
                0 => ComponentBitDepth::Bit8,
                2 => ComponentBitDepth::Bit10,
                4 => ComponentBitDepth::Bit12,
                _ => ComponentBitDepth::Bit8,
            },
            chroma_bit_depth: match sps.bit_depth_chroma_minus8 {
                0 => ComponentBitDepth::Bit8,
                2 => ComponentBitDepth::Bit10,
                4 => ComponentBitDepth::Bit12,
                _ => ComponentBitDepth::Bit8,
            },
            profile_idc: Some(sps.profile_idc as u32),
            dpb_slots: NUM_SURFACES as u32,
        };

        {
            let mut display_area = self.display_area.lock().unwrap();
            *display_area = (display_left, display_top, display_right, display_bottom);
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (coded_width, coded_height);
        }
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = true;
        }

        Ok(())
    }

    /// Recreate the decoder due to a resolution change.
    fn recreate_decoder(&mut self, sps: &H265Sps) -> NvdecResult<()> {
        eprintln!(
            "[recreate] decoder recreated {}x{}",
            sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples
        );
        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
        }
        {
            let mut dpb = self.dpb.lock().unwrap();
            dpb.reset();
        }
        {
            let mut prev = self.prev_dpb_state.lock().unwrap();
            *prev = H265DpbState::default();
        }
        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.clear();
        }
        self.reset_presentation_state();
        {
            let mut decoder = self.decoder.lock().unwrap();
            *decoder = std::ptr::null_mut();
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (0, 0);
        }
        self.create_decoder(sps)
    }

    /// Compute the monotonic (unwrapped) presentation position for a frame.
    fn unwrapped_poc(&mut self, poc: i32) -> i32 {
        let period = self.poc_period;
        if period <= 1 {
            self.prev_decoded_poc = Some(poc);
            return poc;
        }
        if let Some(prev) = self.prev_decoded_poc {
            if poc < prev - period / 2 {
                self.poc_cycle += 1;
            } else if poc > prev + period / 2 {
                self.poc_cycle -= 1;
            }
        }
        self.prev_decoded_poc = Some(poc);
        poc + self.poc_cycle * period
    }

    fn reset_presentation_state(&mut self) {
        self.reorder.clear();
        self.presented_count = 0;
        self.prev_decoded_poc = None;
        self.last_presented_unwrapped = None;
        self.poc_cycle = 0;
    }

    /// Extract ready frames in DISPLAY (ascending PO C) order.
    ///
    /// The guard prevents premature extraction when more frames with lower POC
    /// may still arrive. We track the minimum POC gap observed between
    /// consecutively decoded frames to determine the stream's POC increment.
    /// A frame is only extracted when the gap from the last presented POC to
    /// the current minimum is less than the observed POC increment.
    fn extract_ready_frames(&mut self) {
        // TEMP DEBUG: skip extraction entirely.
        if std::env::var("NVDEC_SKIP_EXTRACT").is_ok() {
            return;
        }
        loop {
            let (&key, &(min_idx, min_seq, min_poc)) = match self.reorder.iter().next() {
                Some(x) => x,
                None => break,
            };
            let max_uw = match self.reorder.iter().next_back() {
                Some((k, _)) => k.0,
                None => break,
            };
            // Guard 1: don't extract if the latest decoded frame has POC <= min
            // (more frames with lower POC may arrive)
            if self.presented_count > 0 && max_uw <= key.0 {
                break;
            }
            // Guard 2: don't extract if there's a gap larger than the POC increment
            // between the last presented POC and the current minimum.
            if let Some(last) = self.last_presented_unwrapped {
                let gap = key.0.saturating_sub(last);
                if gap > self.min_poc_gap && gap < self.poc_period / 2 {
                    break;
                }
            }
            match self.extract_frame(min_idx, min_seq, min_poc) {
                Some(frame) => {
                    // Mark the surface extracted so it can be recycled once its
                    // picture is no longer a live reference.
                    {
                        let mut dpb = self.dpb.lock().unwrap();
                        dpb.mark_extracted(min_idx);
                    }
                    self.reorder.remove(&key);
                    self.last_presented_unwrapped = Some(key.0);
                    self.presented_count += 1;
                    self.pending_frames.lock().unwrap().push_back(frame);
                }
                None => break,
            }
        }
    }

    /// Extract a decoded frame from the NVDEC decoder by surface index.
    fn extract_frame(&self, pic_index: i32, _seq: i32, poc: i32) -> Option<DecodedFrame> {
        if pic_index < 0 {
            return None;
        }
        let decoder = {
            let d = self.decoder.lock().unwrap();
            if d.is_null() {
                return None;
            }
            *d
        };
        let info = {
            let i = self.info.lock().unwrap();
            if i.display_size.width == 0 || i.display_size.height == 0 {
                return None;
            }
            i.clone()
        };
        let display_width = info.display_size.width as usize;
        let display_height = info.display_size.height as usize;
        let funcs = match get_funcs() {
            Ok(f) => f,
            Err(_) => return None,
        };
        let _ = cu_ctx_set_current();

        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let proc_params = CUVIDPROCPARAMS {
            progressive_frame: 1,
            second_field: 0,
            top_field_first: 1,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            Reserved1: 0,
            output_stream: std::ptr::null_mut(),
            Reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            Reserved2: [std::ptr::null_mut()],
        };
        let map_result = unsafe {
            (funcs.map_video_frame64)(decoder, pic_index, &mut dev_ptr, &mut pitch, &proc_params)
        };
        if map_result != CUDA_SUCCESS {
            eprintln!("[NVDEC] cuvidMapVideoFrame64 failed: {}", map_result);
            return None;
        }

        let display_area = {
            let d = self.display_area.lock().unwrap();
            *d
        };
        let (crop_left, crop_top, _, _) = display_area;

        let y_size = display_width * display_height;
        let interleaved_uv_size = display_width * (display_height / 2);
        let total = y_size + interleaved_uv_size;
        let pinned_base = {
            let mut cache = self.pinned_cache.lock().unwrap();
            match &*cache {
                Some((p, sz)) if *sz >= total => *p,
                _ => {
                    if let Some((p, _)) = cache.take() {
                        let _ = unsafe { cu_mem_free_host(p) };
                    }
                    match cu_mem_host_alloc(total) {
                        Ok(p) => {
                            *cache = Some((p, total));
                            p
                        }
                        Err(_) => {
                            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                            return None;
                        }
                    }
                }
            }
        };
        let pinned_y = pinned_base;
        let pinned_uv = unsafe { (pinned_base as *mut u8).add(y_size) as *mut std::ffi::c_void };

        let mut copy_y = CUDA_MEMCPY2D {
            srcXInBytes: crop_left as u64,
            srcY: crop_top as u64,
            srcMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved0: 0,
            srcHost: std::ptr::null(),
            srcDevice: dev_ptr,
            srcArray: 0,
            srcPitch: pitch as u64,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CU_MEMORYTYPE_HOST,
            _reserved1: 0,
            dstHost: pinned_y,
            dstDevice: 0,
            dstArray: 0,
            dstPitch: display_width as u64,
            WidthInBytes: display_width as u64,
            Height: display_height as u64,
        };
        match unsafe { cu_memcpy_2d(&copy_y) } {
            Ok(CUDA_SUCCESS) => {}
            other => {
                eprintln!("[NVDEC] cuMemcpy2D(Y) failed: {:?}", other);
                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                return None;
            }
        }

        let coded_height = info.coded_size.height as u64;
        let mut copy_uv = CUDA_MEMCPY2D {
            srcXInBytes: crop_left as u64,
            srcY: coded_height + (crop_top as u64) / 2,
            srcMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved0: 0,
            srcHost: std::ptr::null(),
            srcDevice: dev_ptr,
            srcArray: 0,
            srcPitch: pitch as u64,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CU_MEMORYTYPE_HOST,
            _reserved1: 0,
            dstHost: pinned_uv,
            dstDevice: 0,
            dstArray: 0,
            dstPitch: display_width as u64,
            WidthInBytes: display_width as u64,
            Height: (display_height / 2) as u64,
        };
        match unsafe { cu_memcpy_2d(&copy_uv) } {
            Ok(CUDA_SUCCESS) => {}
            other => {
                eprintln!("[NVDEC] cuMemcpy2D(UV) failed: {:?}", other);
                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                return None;
            }
        }

        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

        let mut y_plane = vec![0u8; y_size];
        let mut interleaved_uv = vec![0u8; interleaved_uv_size];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_y as *const u8, y_plane.as_mut_ptr(), y_size);
            std::ptr::copy_nonoverlapping(
                pinned_uv as *const u8,
                interleaved_uv.as_mut_ptr(),
                interleaved_uv_size,
            );
        }

        // De-interleave NV12 UV to planar U and V.
        let uv_size = (display_width / 2) * (display_height / 2);
        let mut u_plane = vec![0u8; uv_size];
        let mut v_plane = vec![0u8; uv_size];
        for y in 0..(display_height / 2) {
            for x in 0..(display_width / 2) {
                let src_idx = y * display_width + x * 2;
                let dst_idx = y * (display_width / 2) + x;
                u_plane[dst_idx] = interleaved_uv[src_idx];
                v_plane[dst_idx] = interleaved_uv[src_idx + 1];
            }
        }

        let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
        buffer.extend_from_slice(&y_plane);
        buffer.extend_from_slice(&u_plane);
        buffer.extend_from_slice(&v_plane);

        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
        let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };

        let pixel_data = Some(PixelData {
            format: "I420".to_string(),
            y: PixelPlane {
                data: y_ptr,
                pitch: display_width,
                width: display_width,
                height: display_height,
            },
            u: PixelPlane {
                data: u_ptr,
                pitch: display_width / 2,
                width: display_width / 2,
                height: display_height / 2,
            },
            v: Some(PixelPlane {
                data: v_ptr,
                pitch: display_width / 2,
                width: display_width / 2,
                height: display_height / 2,
            }),
            buffer,
        });

        let frame_index = {
            let mut count = self.frame_count.lock().unwrap();
            let idx = *count;
            *count += 1;
            idx
        };

        let poc_value = poc;

        Some(DecodedFrame {
            frame_index,
            timestamp: 0,
            width: info.display_size.width,
            height: info.display_size.height,
            skipped: false,
            pts_valid: false,
            poc: poc_value,
            field_flags: FieldFlags {
                progressive_frame: true,
                field_pic: false,
                bottom_field: false,
                second_field: false,
                top_field_first: true,
                unpaired_field: false,
                sync_first_ready: false,
                sync_to_first_field: false,
                repeat_first_field: 0,
                ref_pic: false,
                apply_film_grain: false,
            },
            sync_info: vk_video_core::frame::FrameSyncInfo::default(),
            pixel_data,
        })
    }

    /// Dump a decoded picture (DECODE order) as NV12 (full coded size).
    fn dump_decode_order_frame(&self, pic_index: i32, count: u32) {
        let path = match &self.dump_decode_order_path {
            Some(p) => p.clone(),
            None => return,
        };
        let decoder = {
            let d = self.decoder.lock().unwrap();
            if d.is_null() {
                return;
            }
            *d
        };
        let info = {
            let i = self.info.lock().unwrap();
            if i.coded_size.width == 0 || i.coded_size.height == 0 {
                return;
            }
            i.clone()
        };
        let coded_w = info.coded_size.width as usize;
        let coded_h = info.coded_size.height as usize;
        let y_size = coded_w * coded_h;
        let uv_size = coded_w * (coded_h / 2);
        let total = y_size + uv_size;

        let funcs = match get_funcs() {
            Ok(f) => f,
            Err(_) => return,
        };
        let _ = cu_ctx_set_current();

        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let proc_params = CUVIDPROCPARAMS {
            progressive_frame: 1,
            second_field: 0,
            top_field_first: 1,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            Reserved1: 0,
            output_stream: std::ptr::null_mut(),
            Reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            Reserved2: [std::ptr::null_mut()],
        };
        let map_result = unsafe {
            (funcs.map_video_frame64)(decoder, pic_index, &mut dev_ptr, &mut pitch, &proc_params)
        };
        if map_result != CUDA_SUCCESS {
            return;
        }
        let host = unsafe { cu_mem_host_alloc(total) };
        match host {
            Ok(p) => {
                let mut copy_y = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CU_MEMORYTYPE_DEVICE,
                    _reserved0: 0,
                    srcHost: std::ptr::null(),
                    srcDevice: dev_ptr,
                    srcArray: 0,
                    srcPitch: pitch as u64,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CU_MEMORYTYPE_HOST,
                    _reserved1: 0,
                    dstHost: p,
                    dstDevice: 0,
                    dstArray: 0,
                    dstPitch: coded_w as u64,
                    WidthInBytes: coded_w as u64,
                    Height: coded_h as u64,
                };
                let _ = unsafe { cu_memcpy_2d(&copy_y) };
                let mut copy_uv = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: coded_h as u64,
                    srcMemoryType: CU_MEMORYTYPE_DEVICE,
                    _reserved0: 0,
                    srcHost: std::ptr::null(),
                    srcDevice: dev_ptr,
                    srcArray: 0,
                    srcPitch: pitch as u64,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CU_MEMORYTYPE_HOST,
                    _reserved1: 0,
                    dstHost: unsafe { (p as *mut u8).add(y_size) as *mut std::ffi::c_void },
                    dstDevice: 0,
                    dstArray: 0,
                    dstPitch: coded_w as u64,
                    WidthInBytes: coded_w as u64,
                    Height: (coded_h / 2) as u64,
                };
                let _ = unsafe { cu_memcpy_2d(&copy_uv) };
                let mut buf = vec![0u8; total];
                unsafe {
                    std::ptr::copy_nonoverlapping(p as *const u8, buf.as_mut_ptr(), total);
                }
                let file_path = format!("{}_{}.yuv", path.to_string_lossy(), count);
                let _ = std::fs::write(&file_path, &buf);
                let _ = unsafe { cu_mem_free_host(p) };
            }
            Err(_) => {}
        }
        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
    }

    fn get_decoded_frame(&self) -> Option<DecodedFrame> {
        let mut pending = self.pending_frames.lock().unwrap();
        pending.pop_front()
    }

    /// DEBUG: map a decode surface and write its full coded NV12 content to a file.
    fn dump_surface_to(&self, path: &str, surface_idx: i32) {
        let decoder = {
            let d = self.decoder.lock().unwrap();
            if d.is_null() {
                return;
            }
            *d
        };
        let info = {
            let i = self.info.lock().unwrap();
            if i.coded_size.width == 0 || i.coded_size.height == 0 {
                return;
            }
            i.clone()
        };
        let w = info.coded_size.width as usize;
        let h = info.coded_size.height as usize;
        let total = w * h * 3 / 2;
        let funcs = match get_funcs() {
            Ok(f) => f,
            Err(_) => return,
        };
        let _ = cu_ctx_set_current();
        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let proc_params = CUVIDPROCPARAMS {
            progressive_frame: 1,
            second_field: 0,
            top_field_first: 1,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            Reserved1: 0,
            output_stream: std::ptr::null_mut(),
            Reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            Reserved2: [std::ptr::null_mut()],
        };
        if unsafe {
            (funcs.map_video_frame64)(decoder, surface_idx, &mut dev_ptr, &mut pitch, &proc_params)
        } != CUDA_SUCCESS
        {
            return;
        }
        let host = unsafe { cu_mem_host_alloc(total) };
        if let Ok(p) = host {
            let y_size = w * h;
            let mut cy = CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: CU_MEMORYTYPE_DEVICE,
                _reserved0: 0,
                srcHost: std::ptr::null(),
                srcDevice: dev_ptr,
                srcArray: 0,
                srcPitch: pitch as u64,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: CU_MEMORYTYPE_HOST,
                _reserved1: 0,
                dstHost: p,
                dstDevice: 0,
                dstArray: 0,
                dstPitch: w as u64,
                WidthInBytes: w as u64,
                Height: h as u64,
            };
            let _ = unsafe { cu_memcpy_2d(&cy) };
            let mut cuv = CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: h as u64,
                srcMemoryType: CU_MEMORYTYPE_DEVICE,
                _reserved0: 0,
                srcHost: std::ptr::null(),
                srcDevice: dev_ptr,
                srcArray: 0,
                srcPitch: pitch as u64,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: CU_MEMORYTYPE_HOST,
                _reserved1: 0,
                dstHost: unsafe { (p as *mut u8).add(y_size) as *mut std::ffi::c_void },
                dstDevice: 0,
                dstArray: 0,
                dstPitch: w as u64,
                WidthInBytes: w as u64,
                Height: (h / 2) as u64,
            };
            let _ = unsafe { cu_memcpy_2d(&cuv) };
            let mut buf = vec![0u8; total];
            unsafe { std::ptr::copy_nonoverlapping(p as *const u8, buf.as_mut_ptr(), total) };
            let _ = std::fs::write(path, &buf);
            let _ = unsafe { cu_mem_free_host(p) };
        }
        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
    }
}

impl Decoder for NvdecH265Decoder {
    type Error = NvdecError;

    fn new(data: Vec<u8>) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        Self::new(data)
    }

    fn new_with_format(data: Vec<u8>, codec: VideoCodec, _format: &VideoFormat) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        if codec != VideoCodec::DecodeH265 {
            return Err(NvdecError::UnsupportedCodec(codec));
        }
        Self::new(data)
    }

    fn info(&self) -> DecoderInfo {
        self.info.lock().unwrap().clone()
    }

    fn submit(&mut self, data: &[u8]) -> NvdecResult<()> {
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> NvdecResult<Option<DecodedFrame>> {
        self.parse_and_decode()?;
        Ok(self.get_decoded_frame())
    }

    fn flush(&mut self) -> NvdecResult<Vec<DecodedFrame>> {
        self.parse_and_decode()?;
        let mut frames: Vec<DecodedFrame> = {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.drain(..).collect()
        };
        let remaining: Vec<((i32, i32), (i32, i32, i32))> =
            self.reorder.iter().map(|(k, v)| (*k, *v)).collect();
        for (key, (pic_index, seq, poc)) in remaining {
            if let Some(frame) = self.extract_frame(pic_index, seq, poc) {
                self.last_presented_unwrapped = Some(key.0);
                frames.push(frame);
            }
        }
        self.reorder.clear();
        Ok(frames)
    }

    fn reset(&mut self) -> NvdecResult<()> {
        let _ = cu_ctx_set_current();
        let funcs = get_funcs()?;
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            let mut d = self.decoder.lock().unwrap();
            *d = std::ptr::null_mut();
        }
        self.parser.reset();
        {
            let mut dpb = self.dpb.lock().unwrap();
            dpb.reset();
        }
        {
            let mut prev = self.prev_dpb_state.lock().unwrap();
            *prev = H265DpbState::default();
        }
        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.clear();
        }
        self.parsed_offset = 0;
        self.reset_presentation_state();
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = false;
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (0, 0);
        }
        self.init_parser_format()?;
        self.parse_and_decode()?;
        let initialized = *self.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not reinitialize decoder after reset".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for NvdecH265Decoder {
    fn drop(&mut self) {
        let _ = cu_ctx_set_current();
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            if let Ok(funcs) = get_funcs() {
                let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            }
        }
        // Free pinned host buffer to avoid leaking page-locked memory
        if let Ok(mut cache) = self.pinned_cache.lock() {
            if let Some((ptr, _)) = cache.take() {
                let _ = unsafe { crate::device::cu_mem_free_host(ptr) };
            }
        }
    }
}
