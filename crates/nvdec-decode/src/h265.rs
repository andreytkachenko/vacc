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
use vk_video_parser::h265_dpb::{resolve_refs, H265Dpb};
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

/// HEVC decoded-picture-buffer context: the common spec-compliant DPB (the
/// single source of truth shared with the other backends) plus the mapping
/// to cuvid's 16 physical decode surfaces.
struct H265DpbCtx {
    /// Common DPB (spec 8.3.2 marking / eviction / allocation).
    dpb: H265Dpb,
    /// Common-DPB slot -> physical cuvid surface index (None = unassigned).
    slot_surfaces: [Option<i32>; NUM_SURFACES as usize],
    /// Surfaces holding a decoded but not-yet-extracted frame; protected from
    /// reuse until the frame is read back.
    surface_pending: [bool; NUM_SURFACES as usize],
}

impl H265DpbCtx {
    fn new() -> Self {
        Self {
            dpb: H265Dpb::new(NUM_SURFACES as usize),
            slot_surfaces: [None; NUM_SURFACES as usize],
            surface_pending: [false; NUM_SURFACES as usize],
        }
    }

    fn reset(&mut self) {
        self.dpb.invalidate_all();
        self.slot_surfaces = [None; NUM_SURFACES as usize];
        self.surface_pending = [false; NUM_SURFACES as usize];
    }

    /// Drop surface bindings for slots invalidated by `picture_start`.
    fn drop_invalidated(&mut self) {
        for (i, s) in self.dpb.slots().iter().enumerate() {
            if !s.valid {
                self.slot_surfaces[i] = None;
            }
        }
    }

    /// Choose the decode surface (`CurrPicIdx`): the lowest-index surface that
    /// is neither a live DPB reference nor holding a pending (unextracted)
    /// frame.
    fn choose_surface(&self) -> i32 {
        for s in 0..NUM_SURFACES as usize {
            if self.slot_surfaces.iter().any(|o| *o == Some(s as i32)) {
                continue;
            }
            if self.surface_pending[s] {
                continue;
            }
            return s as i32;
        }
        // Fallback: lowest-index non-reference surface.
        for s in 0..NUM_SURFACES as usize {
            if !self.slot_surfaces.iter().any(|o| *o == Some(s as i32)) {
                return s as i32;
            }
        }
        0
    }

    /// Bind the current slot to its surface and commit the picture to the
    /// common DPB.
    fn commit(&mut self, slot: usize, surface: i32) {
        self.slot_surfaces[slot] = Some(surface);
        self.surface_pending[surface as usize] = true;
        self.dpb.commit_current(slot);
    }

    /// Mark a surface's frame as extracted (the surface may be recycled).
    fn mark_extracted(&mut self, surface: i32) {
        if (0..NUM_SURFACES as usize).contains(&(surface as usize)) {
            self.surface_pending[surface as usize] = false;
        }
    }
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
    dpb: Mutex<H265DpbCtx>,
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
    /// Range of unwrapped POCs observed so far. When min == max the stream has
    /// no reordering (e.g. an all-IDR stream where every picture has POC 0), so
    /// display order == decode order and frames must be presented immediately.
    uw_min: Option<i32>,
    uw_max: Option<i32>,

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
            dpb: Mutex::new(H265DpbCtx::new()),
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
            uw_min: None,
            uw_max: None,
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
                            // Reorder delay for the common DPB's display logic.
                            self.dpb
                                .lock()
                                .unwrap()
                                .dpb
                                .set_max_num_reorder_frames(h265_sps.max_num_reorder_pics[0] as u32);
                        }
                    }
                }
                Ok(ParseResult::Slice { slices, .. }) => {
                    if slices.is_empty() {
                        break;
                    }
                    if debug_status {
                        let ctx = self.dpb.lock().unwrap();
                        let occ: Vec<i32> = ctx
                            .dpb
                            .slots()
                            .iter()
                            .map(|s| if s.valid { s.poc } else { -1 })
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

                    // Reset presentation state on IRAP (IDR) pictures. The
                    // common DPB handles the NoRaslOutput reset itself in
                    // picture_start.
                    if info.is_idr {
                        self.poc_cycle = 0;
                        self.prev_decoded_poc = None;
                    }

                    // --- Stage the current picture in the common DPB (spec
                    // 8.3.2): NoRaslOutput reset, RPS marking (used + future-use
                    // keep-alive), eviction, slot allocation. A CRA's unused RPS
                    // entries keep the pre-CRA references alive for the
                    // following open-GOP pictures. ---
                    let slot = {
                        let mut ctx = self.dpb.lock().unwrap();
                        let slot = ctx.dpb.picture_start(sps, info, info.is_reference);
                        ctx.drop_invalidated();
                        slot
                    };

                    // --- CUVID RefPicSet* arrays: the current picture's USED
                    // short-term / long-term references in RPS order (spec 8.3.3
                    // initial lists), mapped to common-DPB slot indices. Unused
                    // RPS entries are excluded (the CUVID structs carry no used
                    // flags; a CRA has NumPocTotalCurr = 0). ---
                    let (before, after, lt) = {
                        let ctx = self.dpb.lock().unwrap();
                        let resolved = resolve_refs(sps, info);
                        let (all_b, all_a, all_lt) = ctx.dpb.match_rps_slots();
                        let b: Vec<i32> = resolved
                            .st_curr_before
                            .iter()
                            .zip(all_b.iter())
                            .filter(|(r, _)| r.used)
                            .map(|(_, s)| *s)
                            .collect();
                        let a: Vec<i32> = resolved
                            .st_curr_after
                            .iter()
                            .zip(all_a.iter())
                            .filter(|(r, _)| r.used)
                            .map(|(_, s)| *s)
                            .collect();
                        let l: Vec<i32> = resolved
                            .long_term
                            .iter()
                            .zip(all_lt.iter())
                            .filter(|(r, _)| r.used)
                            .map(|(_, s)| *s)
                            .collect();
                        (b, a, l)
                    };

                    // NumBitsForShortTermRPSInSlice: bit size of
                    // short_term_ref_pic_set() in the slice header, measured by
                    // the parser. 0 when the RPS comes from the SPS, or for IDR
                    // (no RPS block at all).
                    let num_bits = if info.short_term_ref_pic_set_sps_flag {
                        0
                    } else {
                        info.num_bits_for_strps_in_slice as i32
                    };

                    if debug_status {
                        eprintln!(
                            "[dbg] poc={} is_idr={} is_rap={} slice_type={} rps_sps_flag={} rps_idx={} before={:?} after={:?} lt={:?} num_bits={}",
                            poc, info.is_idr, info.is_rap, info.slice_type,
                            info.short_term_ref_pic_set_sps_flag, info.short_term_ref_pic_set_idx,
                            before, after, lt, num_bits
                        );
                    }

                    // --- Choose the decode surface (CurrPicIdx) and build the
                    // CUVID DPB state (16-entry array, slot-indexed). ---
                    let (curr_pic_idx, dpb_state) = {
                        let ctx = self.dpb.lock().unwrap();
                        let curr_pic_idx = ctx.choose_surface();
                        let mut state = H265DpbState::default();
                        for i in 0..NUM_SURFACES as usize {
                            if ctx.dpb.slots()[i].valid {
                                state.ref_pic_idx[i] = ctx.slot_surfaces[i].unwrap_or(-1);
                                state.pic_order_cnt_val[i] = ctx.dpb.slots()[i].poc;
                                state.is_long_term[i] = ctx.dpb.slots()[i].is_long_term as u8;
                            }
                        }
                        for (j, &s) in before.iter().enumerate().take(8) {
                            state.st_curr_before[j] = s.max(0) as u8;
                        }
                        for (j, &s) in after.iter().enumerate().take(8) {
                            state.st_curr_after[j] = s.max(0) as u8;
                        }
                        for (j, &s) in lt.iter().enumerate().take(8) {
                            state.lt_curr[j] = s.max(0) as u8;
                        }
                        state.num_poc_st_curr_before = before.len() as i32;
                        state.num_poc_st_curr_after = after.len() as i32;
                        state.num_poc_lt_curr = lt.len() as i32;
                        state.num_poc_total_curr = (before.len() + after.len() + lt.len()) as i32;
                        state.num_bits_for_short_term_rps_in_slice = num_bits;
                        state.num_delta_pocs_of_ref_rps_idx = 0;
                        state.curr_pic_order_cnt_val = poc;
                        (curr_pic_idx, state)
                    };
                    if debug_status {
                        let ctx = self.dpb.lock().unwrap();
                        let occ: Vec<i32> = ctx
                            .dpb
                            .slots()
                            .iter()
                            .map(|s| if s.valid { s.poc } else { -1 })
                            .collect();
                        eprintln!("[dpb] poc={} slots={:?}", poc, occ);
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

                    let procparams = crate::ffi::default_procparams();
                    let result = unsafe {
                        (funcs.decode_picture)(
                            decoder_handle as *mut std::ffi::c_void,
                            &picparams,
                            &procparams,
                        )
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
                            self.dump_params_count.saturating_sub(1),
                            curr_pic_idx,
                            poc,
                            st as u32,
                            st_name,
                            status_result
                        );
                    }

                    // --- Commit the picture to the common DPB and bind its
                    // surface (protected until extracted). ---
                    let seq = self.seq_counter;
                    self.seq_counter += 1;
                    {
                        let mut ctx = self.dpb.lock().unwrap();
                        ctx.commit(slot, curr_pic_idx);
                    }

                    // Track for display-order presentation.
                    let unwrapped = self.unwrapped_poc(poc);
                    self.uw_min = Some(self.uw_min.map_or(unwrapped, |m| m.min(unwrapped)));
                    self.uw_max = Some(self.uw_max.map_or(unwrapped, |m| m.max(unwrapped)));
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

        let output_format = if sps.chroma_format_idc == 0 {
            // Monochrome (4:0:0): Y-only surface.
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV400
        } else if sps.chroma_format_idc == 3 {
            // 4:4:4 planar.
            if sps.bit_depth_luma_minus8 > 0 {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444_16Bit
            } else {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444
            }
        } else if sps.bit_depth_luma_minus8 > 0 {
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
            let mut ctx = self.dpb.lock().unwrap();
            ctx.reset();
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
        self.uw_min = None;
        self.uw_max = None;
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
            // (more frames with lower POC may arrive). Exception: when POC never
            // advances at all (all-IDR stream: every picture has POC 0) no
            // reordering exists and display order == decode order — holding back
            // would stall forever while NVDEC recycles the pending surfaces.
            let poc_flat = match (self.uw_min, self.uw_max) {
                (Some(lo), Some(hi)) => lo == hi,
                _ => false,
            };
            if self.presented_count > 0 && max_uw <= key.0 && !poc_flat {
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

        let is_mono = info.chroma_subsampling == ChromaSubsampling::Monochrome;
        let is_444 = info.chroma_subsampling == ChromaSubsampling::_444;
        // P016/YUV444_16Bit: 2 bytes per sample.
        let bps = if info.luma_bit_depth == ComponentBitDepth::Bit10 { 2 } else { 1 };
        let row_bytes = display_width * bps;
        let y_size = row_bytes * display_height;
        // NV12/P016: one interleaved UV plane at half resolution.
        // YUV444/YUV444_16Bit: two planar U/V planes at full resolution.
        let uv_plane_size = if is_mono {
            0
        } else if is_444 {
            row_bytes * display_height
        } else {
            row_bytes * (display_height / 2)
        };
        let num_uv_planes = if is_mono { 0 } else if is_444 { 2 } else { 1 };
        let interleaved_uv_size = uv_plane_size * num_uv_planes;
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
            dstPitch: row_bytes as u64,
            WidthInBytes: row_bytes as u64,
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

        if !is_mono {
            let coded_height = info.coded_size.height as u64;
            for plane in 0..num_uv_planes {
                let (src_y, rows) = if is_444 {
                    (coded_height * (plane as u64 + 1) + crop_top as u64, display_height as u64)
                } else {
                    (coded_height + (crop_top as u64) / 2, (display_height / 2) as u64)
                };
                let dst = unsafe {
                    (pinned_base as *mut u8).add(y_size + plane * uv_plane_size)
                        as *mut std::ffi::c_void
                };
                let mut copy_uv = CUDA_MEMCPY2D {
                    srcXInBytes: crop_left as u64,
                    srcY: src_y,
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
                    dstHost: dst,
                    dstDevice: 0,
                    dstArray: 0,
                    dstPitch: row_bytes as u64,
                    WidthInBytes: row_bytes as u64,
                    Height: rows,
                };
                match unsafe { cu_memcpy_2d(&copy_uv) } {
                    Ok(CUDA_SUCCESS) => {}
                    other => {
                        eprintln!("[NVDEC] cuMemcpy2D(UV plane {}) failed: {:?}", plane, other);
                        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                        return None;
                    }
                }
            }
        }

        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

        let mut y_plane = vec![0u8; y_size];
        let mut interleaved_uv = vec![0u8; interleaved_uv_size];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_y as *const u8, y_plane.as_mut_ptr(), y_size);
            std::ptr::copy_nonoverlapping(
                (pinned_base as *mut u8).add(y_size),
                interleaved_uv.as_mut_ptr(),
                interleaved_uv_size,
            );
        }

        let pixel_data = if is_mono {
            // Monochrome (YUV400): Y plane only.
            let mut buffer = vec![0u8; y_size];
            buffer.copy_from_slice(&y_plane);
            let y_ptr = buffer.as_ptr();
            Some(PixelData {
                format: if bps == 2 { "GRAY_16BIT" } else { "GRAY" }.to_string(),
                y: PixelPlane {
                    data: y_ptr,
                    pitch: row_bytes,
                    width: display_width,
                    height: display_height,
                },
                u: PixelPlane {
                    data: y_ptr,
                    pitch: 0,
                    width: 0,
                    height: 0,
                },
                v: None,
                buffer,
            })
        } else if is_444 {
            // YUV444: U and V already planar at full resolution in pinned memory.
            let uv_size = display_width * display_height;
            let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
            buffer.extend_from_slice(&y_plane);
            buffer.extend_from_slice(&interleaved_uv);
            let y_ptr = buffer.as_ptr();
            let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
            let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };
            Some(PixelData {
                format: if bps == 2 { "YUV444_16BIT" } else { "YUV444" }.to_string(),
                y: PixelPlane {
                    data: y_ptr,
                    pitch: row_bytes,
                    width: display_width,
                    height: display_height,
                },
                u: PixelPlane {
                    data: u_ptr,
                    pitch: row_bytes,
                    width: display_width,
                    height: display_height,
                },
                v: Some(PixelPlane {
                    data: v_ptr,
                    pitch: row_bytes,
                    width: display_width,
                    height: display_height,
                }),
                buffer,
            })
        } else {
            // De-interleave NV12/P016 UV to planar U and V (byte-based).
            let uv_w = display_width / 2;
            let uv_h = display_height / 2;
            let uv_size = uv_w * uv_h * bps;
            let mut u_plane = vec![0u8; uv_size];
            let mut v_plane = vec![0u8; uv_size];
            for y in 0..uv_h {
                for x in 0..uv_w {
                    let src_idx = (y * display_width + x * 2) * bps;
                    let dst_idx = (y * uv_w + x) * bps;
                    u_plane[dst_idx..dst_idx + bps]
                        .copy_from_slice(&interleaved_uv[src_idx..src_idx + bps]);
                    v_plane[dst_idx..dst_idx + bps]
                        .copy_from_slice(&interleaved_uv[src_idx + bps..src_idx + 2 * bps]);
                }
            }

            let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
            buffer.extend_from_slice(&y_plane);
            buffer.extend_from_slice(&u_plane);
            buffer.extend_from_slice(&v_plane);

            let y_ptr = buffer.as_ptr();
            let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
            let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };

            Some(PixelData {
                format: if bps == 2 { "I420_16BIT" } else { "I420" }.to_string(),
                y: PixelPlane {
                    data: y_ptr,
                    pitch: row_bytes,
                    width: display_width,
                    height: display_height,
                },
                u: PixelPlane {
                    data: u_ptr,
                    pitch: row_bytes / 2,
                    width: display_width / 2,
                    height: display_height / 2,
                },
                v: Some(PixelPlane {
                    data: v_ptr,
                    pitch: row_bytes / 2,
                    width: display_width / 2,
                    height: display_height / 2,
                }),
                buffer,
            })
        };

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

    /// Dump a decoded picture (DECODE order) as raw planes (full coded size).
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
        let is_mono = info.chroma_subsampling == ChromaSubsampling::Monochrome;
        let is_444 = info.chroma_subsampling == ChromaSubsampling::_444;
        let bps = if info.luma_bit_depth == ComponentBitDepth::Bit10 { 2 } else { 1 };
        let row_bytes = coded_w * bps;
        let y_size = row_bytes * coded_h;
        let uv_plane_size = if is_mono {
            0
        } else if is_444 {
            row_bytes * coded_h
        } else {
            row_bytes * (coded_h / 2)
        };
        let num_uv_planes = if is_mono { 0 } else if is_444 { 2 } else { 1 };
        let total = y_size + uv_plane_size * num_uv_planes;

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
                    dstPitch: row_bytes as u64,
                    WidthInBytes: row_bytes as u64,
                    Height: coded_h as u64,
                };
                let _ = unsafe { cu_memcpy_2d(&copy_y) };
                if !is_mono {
                    for plane in 0..num_uv_planes {
                        let src_y = if is_444 {
                            (coded_h * (plane + 1)) as u64
                        } else {
                            coded_h as u64
                        };
                        let rows = if is_444 { coded_h as u64 } else { (coded_h / 2) as u64 };
                        let mut copy_uv = CUDA_MEMCPY2D {
                            srcXInBytes: 0,
                            srcY: src_y,
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
                            dstHost: unsafe {
                                (p as *mut u8).add(y_size + plane * uv_plane_size)
                                    as *mut std::ffi::c_void
                            },
                            dstDevice: 0,
                            dstArray: 0,
                            dstPitch: row_bytes as u64,
                            WidthInBytes: row_bytes as u64,
                            Height: rows,
                        };
                        let _ = unsafe { cu_memcpy_2d(&copy_uv) };
                    }
                }
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
        let is_mono = info.chroma_subsampling == ChromaSubsampling::Monochrome;
        let is_444 = info.chroma_subsampling == ChromaSubsampling::_444;
        let bps = if info.luma_bit_depth == ComponentBitDepth::Bit10 { 2 } else { 1 };
        let row_bytes = w * bps;
        let uv_plane_size = if is_mono {
            0
        } else if is_444 {
            row_bytes * h
        } else {
            row_bytes * (h / 2)
        };
        let num_uv_planes = if is_mono { 0 } else if is_444 { 2 } else { 1 };
        let total = row_bytes * h + uv_plane_size * num_uv_planes;
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
            let y_size = row_bytes * h;
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
                dstPitch: row_bytes as u64,
                WidthInBytes: row_bytes as u64,
                Height: h as u64,
            };
            let _ = unsafe { cu_memcpy_2d(&cy) };
            if !is_mono {
                for plane in 0..num_uv_planes {
                    let src_y = if is_444 { (h * (plane + 1)) as u64 } else { h as u64 };
                    let rows = if is_444 { h as u64 } else { (h / 2) as u64 };
                    let mut cuv = CUDA_MEMCPY2D {
                        srcXInBytes: 0,
                        srcY: src_y,
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
                        dstHost: unsafe {
                            (p as *mut u8).add(y_size + plane * uv_plane_size)
                                as *mut std::ffi::c_void
                        },
                        dstDevice: 0,
                        dstArray: 0,
                        dstPitch: row_bytes as u64,
                        WidthInBytes: row_bytes as u64,
                        Height: rows,
                    };
                    let _ = unsafe { cu_memcpy_2d(&cuv) };
                }
            }
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
            let mut ctx = self.dpb.lock().unwrap();
            ctx.reset();
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
