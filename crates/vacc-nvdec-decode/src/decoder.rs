//! NVDEC H.264 decoder implementation using vacc-parser.
//!
//! This module provides [`NvdecH264Decoder`], which uses the Rust-based
//! [`H264Parser`](vacc_parser::h264::H264Parser) for bitstream parsing,
//! SPS/PPS extraction, POC calculation, and DPB management.
//!
//! ## How It Works
//!
//! The decoder uses a pull-based decode flow:
//!
//! 1. **H264Parser** parses the bitstream to extract SPS/PPS and slice data.
//! 2. On SPS: creates or reconfigures the NVDEC decoder.
//! 3. On PPS: stores it for later use.
//! 4. On Slice:
//!    - Calculate POC using [`PocCalculator`](crate::poc::PocCalculator)
//!    - Apply MMCO using [`NvdecDpbManager`](crate::dpb::NvdecDpbManager)
//!    - Build [`CUVIDPICPARAMS`](crate::ffi::CUVIDPICPARAMS) via [`picparams`]
//!    - Call `cuvidDecodePicture`
//!    - Add frame to DPB
//!    - Extract ready frames in POC order
//!
//! ## Frame Output Format
//!
//! Decoded frames are output in **I420** (planar YUV 4:2:0) format:
//! - Y plane: full resolution
//! - U plane: half width, half height
//! - V plane: half width, half height
//!
//! The raw NV12 output from NVDEC is de-interleaved during frame extraction.
//!
//! ## Pipeline Draining
//!
//! When the bitstream ends, frames may remain in the decoder's internal
//! pipeline due to DPB reordering and B-frame delays. Call [`Decoder::flush`]
//! to extract all pending frames in POC order.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use vacc_core::{
    codec::VideoCodec,
    decoder::{Decoder, DecoderInfo},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane},
    session::Extent2D,
};
use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

use crate::device::{
    cu_ctx_set_current, cu_ctx_synchronize, cu_mem_free_host, cu_mem_host_alloc, cu_memcpy_2d,
    get_funcs, init_nvdec, CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST,
};
use crate::dpb::NvdecDpbManager;
use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{
    cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
    cudaVideoSurfaceFormat, CUdeviceptr, CUvideodecoder, CUDA_SUCCESS, CUVIDDECODECREATEINFO,
    CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT,
};
use crate::picparams::build_cuvid_picparams;
use crate::poc::PocCalculator;

/// NVDEC H.264 Decoder using vacc-parser.
///
/// This decoder uses NVIDIA's hardware NVDEC engine for H.264 video decoding.
/// It leverages the Rust-based H264Parser for bitstream parsing, which handles
/// SPS/PPS extraction, POC calculation, and DPB management.
///
/// # Thread Safety
///
/// This type is **not** `Send` or `Sync`. Use from a single thread only.
/// The internal CUDA context must be set as current before calling decode
/// methods.
///
/// # Resource Management
///
/// The decoder holds GPU resources (decoder handle, mapped surfaces).
/// These are automatically released when the decoder is dropped.
/// Call [`Decoder::flush`] before dropping to ensure all pending frames
/// are drained.
///
/// # Example
///
/// ```no_run
/// use vacc_nvdec_decode::NvdecH264Decoder;
/// use vacc_core::decoder::Decoder;
///
/// let data = std::fs::read("video.h264").unwrap();
/// let mut decoder = NvdecH264Decoder::new(data).unwrap();
///
/// println!("Codec: {:?}", decoder.info().codec);
///
/// while let Some(frame) = decoder.decode().unwrap() {
///     println!("Frame {}: {}x{}",
///         frame.frame_index, frame.width, frame.height);
/// }
///
/// // Drain remaining frames
/// let remaining = decoder.flush().unwrap();
/// ```
pub struct NvdecH264Decoder {
    /// H.264 bitstream parser.
    parser: H264Parser,

    /// NVDEC decoder handle.
    decoder: Mutex<CUvideodecoder>,

    /// Decoder info.
    info: Mutex<DecoderInfo>,

    /// Pending decoded frames queue.
    pending_frames: Mutex<VecDeque<DecodedFrame>>,

    /// Frame count for ordering.
    frame_count: Mutex<u32>,

    /// Display area (left, top, right, bottom).
    display_area: Mutex<(i32, i32, i32, i32)>,

    /// Whether decoder is initialized.
    initialized: Mutex<bool>,

    /// POC calculator.
    poc_calculator: Mutex<PocCalculator>,

    /// DPB manager.
    dpb_manager: Mutex<NvdecDpbManager>,

    /// Profile IDC.
    profile_idc: Mutex<Option<u32>>,

    /// Previous coded dimensions (for detecting resolution changes).
    prev_coded_size: Mutex<(u32, u32)>,

    /// Pending bitstream data not yet parsed.
    pending_data: Vec<u8>,

    /// Offset in `pending_data` that has already been parsed.
    parsed_offset: usize,

    /// Reorder buffer for display-order presentation.
    ///
    /// Maps (unwrapped_poc, seq) -> (pic_index, seq) for every decoded frame
    /// that has not yet been presented. `unwrapped_poc` is a monotonic
    /// presentation position (see [`unwrapped_poc`](Self::unwrapped_poc));
    /// `seq` breaks ties. A frame is presentable once a higher-POC frame
    /// has been decoded (so no lower-POC frame can still arrive), or it is
    /// the first frame.
    reorder: BTreeMap<(i32, i32), (i32, i32)>,

    /// Number of frames presented so far (display-order gating).
    presented_count: u32,

    /// Observed POC range (for inferring the POC wrap period).
    poc_min: i32,
    poc_max: i32,

    /// GCD of consecutive decode-order POC differences (POC step).
    poc_gcd: i32,

    /// POC of the previously decoded frame (for `poc_gcd` tracking).
    prev_decoded_poc: Option<i32>,

    /// Unwrapped POC of the last presented frame.
    last_presented_unwrapped: Option<i32>,

    /// POC wrap period (= `max_pic_order_cnt_lsb` = 2^(log2_max_pic_order_cnt_lsb_minus4 + 4)).
    /// Set from the SPS. 0 when POC type != 0 (no lsb wrap) or unknown.
    poc_period: i32,

    /// Current POC wrap cycle, tracked in DECODE order. The unwrapped
    /// presentation position is `raw_poc + poc_cycle * poc_period`.
    poc_cycle: i32,

    /// Raw SPS NAL bytes (to feed to the decoder's internal parser).
    sps_nal_data: Mutex<Option<Vec<u8>>>,

    /// Raw PPS NAL bytes (to feed to the decoder's internal parser).
    pps_nal_data: Mutex<Option<Vec<u8>>>,

    /// Whether the SPS/PPS NALs have been fed to the decoder yet.
    sps_pps_fed: Mutex<bool>,

    /// Cached pinned host buffer for frame extraction. Reused across frames
    /// to avoid per-frame `cuMemHostAlloc`/`cuMemFreeHost` overhead. The
    /// buffer is grown (and reallocated) when the display resolution changes.
    pinned_cache: Mutex<Option<(*mut std::ffi::c_void, usize)>>,

    /// If set (via `NVDEC_DUMP_PARAMS` env var), dump the exact
    /// [`CUVIDPICPARAMS`] submitted for each picture to this path, in the
    /// NVIDIA C reference (cuvid_ref.c) text format, for diffing.
    dump_params_path: Option<std::path::PathBuf>,

    /// Per-instance picture counter for the params dump (DECODE order, starts at 0).
    dump_params_count: u32,

    /// If set (via `NVDEC_DUMP_DECODE_ORDER` env var), dump each decoded
    /// picture (in DECODE order, mapped from its CurrPicIdx surface) to
    /// `{path}_{N}.yuv` as NV12, for direct comparison with the C reference.
    dump_decode_order_path: Option<std::path::PathBuf>,

    /// Per-instance decode-order picture counter (starts at 0).
    dump_decode_order_count: u32,
}

impl NvdecH264Decoder {
    /// Create a new NVDEC H.264 decoder.
    ///
    /// Parses the input data to extract SPS/PPS, creates the NVDEC decoder,
    /// and begins decoding immediately. All frames in the input data are
    /// decoded and queued.
    ///
    /// # Arguments
    ///
    /// * `data` — Complete H.264 bitstream data (Annex-B format with start
    ///   codes). Must contain at least one SPS and PPS NAL unit.
    ///
    /// # Errors
    ///
    /// * [`NvdecError::CudaError`] — CUDA initialization failed
    /// * [`NvdecError::LibLoadError`] — libcuda.so or libnvcuvid.so not found
    /// * [`NvdecError::DecoderCreationFailed`] — Decoder creation failed,
    ///   or no SPS/PPS found in input data
    pub fn new(data: Vec<u8>) -> NvdecResult<Self> {
        init_nvdec()?;

        let mut decoder = Self {
            parser: H264Parser::new(),
            decoder: Mutex::new(std::ptr::null_mut()),
            info: Mutex::new(DecoderInfo {
                backend: "nvdec".to_string(),
                codec: VideoCodec::DecodeH264,
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
                dpb_slots: 0,
            }),
            pending_frames: Mutex::new(VecDeque::new()),
            frame_count: Mutex::new(0),
            display_area: Mutex::new((0, 0, 0, 0)),
            initialized: Mutex::new(false),
            poc_calculator: Mutex::new(PocCalculator::new()),
            dpb_manager: Mutex::new(NvdecDpbManager::new(16)), // default; updated from SPS
            profile_idc: Mutex::new(None),
            prev_coded_size: Mutex::new((0, 0)),
            pending_data: data,
            parsed_offset: 0,
            reorder: BTreeMap::new(),
            presented_count: 0,
            poc_min: i32::MAX,
            poc_max: i32::MIN,
            poc_gcd: 0,
            prev_decoded_poc: None,
            last_presented_unwrapped: None,
            poc_period: 0,
            poc_cycle: 0,
            sps_nal_data: Mutex::new(None),
            pps_nal_data: Mutex::new(None),
            sps_pps_fed: Mutex::new(false),
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

        // Parse all initial data
        decoder.parse_and_decode()?;

        let initialized = *decoder.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not initialize decoder - no SPS/PPS found".into(),
            ));
        }

        Ok(decoder)
    }

    /// Parse pending data and decode any available frames.
    fn parse_and_decode(&mut self) -> NvdecResult<()> {
        if self.parsed_offset >= self.pending_data.len() {
            return Ok(());
        }

        let remaining = &self.pending_data[self.parsed_offset..];
        let packet = BitstreamPacket::new(remaining.to_vec());

        loop {
            match self.parser.parse(&packet) {
                Ok(ParseResult::ParameterSet {
                    sps,
                    pps,
                    sps_nal,
                    pps_nal,
                    ..
                }) => {
                    // Handle SPS — create or recreate decoder
                    if let Some(sps_box) = sps {
                        if let Some(h264_sps) =
                            sps_box.downcast_ref::<vacc_core::picture::H264Sps>()
                        {
                            let (prev_w, prev_h) = {
                                let s = self.prev_coded_size.lock().unwrap();
                                *s
                            };

                            let coded_width = (h264_sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
                            let coded_height = if h264_sps.frame_mbs_only_flag {
                                (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                            } else {
                                (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                            };

                            let resolution_changed =
                                prev_w != coded_width || prev_h != coded_height;
                            if resolution_changed {
                                self.recreate_decoder(h264_sps)?;
                            } else {
                                // Check if decoder needs to be created for the first time
                                let decoder_handle = {
                                    let d = self.decoder.lock().unwrap();
                                    *d
                                };
                                if decoder_handle.is_null() {
                                    self.create_decoder(h264_sps)?;
                                }
                            }

                            // Store profile_idc
                            {
                                let mut p = self.profile_idc.lock().unwrap();
                                *p = Some(h264_sps.profile_idc as u32);
                            }

                            // Update DPB manager from SPS
                            {
                                let mut dpb = self.dpb_manager.lock().unwrap();
                                dpb.set_max_frame_num(h264_sps.max_frame_num);
                                dpb.set_max_dpb_size(h264_sps.max_num_ref_frames as usize);
                            }

                            // Set the POC wrap period from the SPS (POC type 0).
                            if h264_sps.pic_order_cnt_type == 0 {
                                self.poc_period = h264_sps.max_pic_order_cnt_lsb as i32;
                            } else {
                                self.poc_period = 0;
                            }
                        }
                    }

                    // Handle PPS — parser already cached it; nothing extra needed
                    if pps.is_some() {
                        // PPS stored in parser's active_pps
                    }

                    // Store raw SPS/PPS NAL bytes so they can be fed to the
                    // decoder's internal parser (it needs the actual NAL bitstream,
                    // not just the parsed structs, to decode slices).
                    if let Some(nal) = sps_nal {
                        *self.sps_nal_data.lock().unwrap() = Some(nal);
                        *self.sps_pps_fed.lock().unwrap() = false;
                    }
                    if let Some(nal) = pps_nal {
                        *self.pps_nal_data.lock().unwrap() = Some(nal);
                    }
                }
                Ok(ParseResult::Slice { slices, .. }) => {
                    if slices.is_empty() {
                        break;
                    }

                    // Get the first slice header for this frame
                    let slh = if let Some(vacc_parser::SliceHeader::H264(h264_slh)) =
                        slices[0].slice_header.as_ref()
                    {
                        h264_slh
                    } else {
                        break;
                    };

                    // Get active SPS/PPS from parser. Cloned so that the
                    // immutable borrow of `self.parser` does not conflict with
                    // the `&mut self` calls below (drain_reorder, create_decoder).
                    let sps = self
                        .parser
                        .active_sps()
                        .cloned()
                        .ok_or_else(|| NvdecError::DecodeFailed("No active SPS".into()))?;
                    let pps = self
                        .parser
                        .active_pps()
                        .cloned()
                        .ok_or_else(|| NvdecError::DecodeFailed("No active PPS".into()))?;

                    // Determine if this is an IDR picture
                    let is_idr = slh.nal_unit_type == 5;

                    // Reset POC calculator for IDR pictures
                    if is_idr {
                        // A new GOP starts here: POC restarts from 0, so the
                        // wrap tracker must restart with it, and everything
                        // still held for reordering belongs to an earlier GOP
                        // and has to be presented first (H.264 8.2.5.2 outputs
                        // all needed-for-output pictures at an IDR unless
                        // no_output_of_prior_pics_flag is set).
                        let drained = self.drain_reorder();
                        if !drained.is_empty() {
                            self.pending_frames.lock().unwrap().extend(drained);
                        }
                        self.poc_cycle = 0;
                        self.prev_decoded_poc = None;
                        let mut poc_calc = self.poc_calculator.lock().unwrap();
                        poc_calc.reset();
                    }

                    // Calculate POC
                    let poc = {
                        let mut poc_calc = self.poc_calculator.lock().unwrap();
                        let is_reference = slh.nal_ref_idc > 0;
                        poc_calc.calculate(&sps, slh, is_reference)
                    };

                    // Apply the IDR reset BEFORE the picture is added, so old
                    // references are cleared first. Non-IDR MMCO operations are
                    // applied after add_frame (see below) so they affect the
                    // DPB state seen by subsequent pictures (H.264 spec 8.2.5).
                    if is_idr {
                        let mut dpb = self.dpb_manager.lock().unwrap();
                        dpb.apply_idr_reset(slh.long_term_reference_flag);
                    }

                    // Get current picture index from DPB manager (before adding current frame)
                    let curr_pic_idx = {
                        let dpb = self.dpb_manager.lock().unwrap();
                        dpb.get_next_pic_index()
                    };

                    // Build CUVID picture parameters with DPB entries (references only, not current frame)
                    let dpb_entries = {
                        let dpb = self.dpb_manager.lock().unwrap();
                        dpb.to_cuvid_dpb_entries()
                    };

                    // Build bitstream buffer: each slice NAL is prefixed with a
                    // 3-byte Annex-B start code (00 00 01), matching the layout the
                    // NVIDIA cuvid parser produces. Slice offsets point to the START
                    // of each start code (not the NAL header).
                    // cuvid requires the SPS/PPS NALs in the bitstream before the
                    // first slice that references them (the C reference gets them
                    // via cuvidParser internally). Feed the stored SPS/PPS once,
                    // prepended with start codes, on the first picture after they
                    // are (re)received.
                    let sps_nal = self.sps_nal_data.lock().unwrap().clone();
                    let pps_nal = self.pps_nal_data.lock().unwrap().clone();
                    let need_sps_pps = !*self.sps_pps_fed.lock().unwrap();
                    let sps_pps_len = sps_nal.as_ref().map(|n| n.len() + 3).unwrap_or(0)
                        + pps_nal.as_ref().map(|n| n.len() + 3).unwrap_or(0);

                    let mut bitstream_data = Vec::with_capacity(
                        slices.iter().map(|s| s.nal_data.len() + 3).sum::<usize>()
                            + if need_sps_pps { sps_pps_len } else { 0 },
                    );
                    if need_sps_pps {
                        if let Some(nal) = &sps_nal {
                            bitstream_data.extend_from_slice(&[0u8, 0, 1]);
                            bitstream_data.extend_from_slice(nal);
                        }
                        if let Some(nal) = &pps_nal {
                            bitstream_data.extend_from_slice(&[0u8, 0, 1]);
                            bitstream_data.extend_from_slice(nal);
                        }
                        *self.sps_pps_fed.lock().unwrap() = true;
                    }
                    let mut slice_offsets = Vec::with_capacity(slices.len() + 1);
                    for slice_entry in &slices {
                        slice_offsets.push(bitstream_data.len() as u32);
                        bitstream_data.extend_from_slice(&[0u8, 0, 1]);
                        bitstream_data.extend_from_slice(&slice_entry.nal_data);
                    }
                    // The NVDEC front end may read nNumSlices+1 entries (the
                    // last one as a terminating offset) — provide it.
                    slice_offsets.push(bitstream_data.len() as u32);

                    let picparams = build_cuvid_picparams(
                        &sps,
                        &pps,
                        slh,
                        slh.frame_num,
                        poc,
                        slh.nal_ref_idc > 0,
                        curr_pic_idx,
                        &bitstream_data,
                        &slice_offsets,
                        slices.len() as u32,
                        &dpb_entries,
                    );

                    // Decode the picture
                    let decoder_handle = {
                        let d = self.decoder.lock().unwrap();
                        if d.is_null() {
                            break;
                        }
                        *d
                    };

                    let funcs = get_funcs()?;
                    let _ = cu_ctx_set_current();

                    // Dump the exact CUVIDPICPARAMS about to be submitted
                    // (gated by NVDEC_DUMP_PARAMS), in NVIDIA C reference format.
                    if let Some(dump_path) = &self.dump_params_path {
                        dump_cuvid_picparams(dump_path, self.dump_params_count, &picparams);
                        self.dump_params_count += 1;
                    }

                    // Keep bitstream_data and slice_offsets alive during decode
                    let procparams = crate::ffi::default_procparams();
                    let result =
                        unsafe { (funcs.decode_picture)(decoder_handle, &picparams, &procparams) };
                    if result != CUDA_SUCCESS {
                        return Err(NvdecError::DecodeFailed(format!(
                            "cuvidDecodePicture failed: {}",
                            result
                        )));
                    }

                    let _ = cu_ctx_synchronize();

                    // Dump the decoded picture in DECODE order (gated by
                    // NVDEC_DUMP_DECODE_ORDER) for surface-content comparison.
                    if self.dump_decode_order_path.is_some() {
                        self.dump_decode_order_frame(curr_pic_idx, self.dump_decode_order_count);
                        self.dump_decode_order_count += 1;
                    }

                    // Poll decode status until completion
                    let mut decode_status = crate::ffi::CUVIDGETDECODESTATUS {
                        decodeStatus: crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Invalid,
                        reserved: [0; 31],
                        pReserved: [std::ptr::null_mut(); 8],
                    };
                    for _ in 0..100 {
                        let _ = unsafe {
                            (funcs.get_decode_status)(
                                decoder_handle,
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

                    // Add current frame to DPB after decode
                    let is_reference = slh.nal_ref_idc > 0;
                    let (pic_index, seq) = {
                        let mut dpb = self.dpb_manager.lock().unwrap();
                        let pic_index = dpb.add_frame(slh.frame_num, poc, is_reference);
                        (pic_index, dpb.last_seq())
                    };

                    // Apply non-IDR MMCO operations AFTER the current picture is
                    // added, so they update the DPB state seen by subsequent
                    // pictures (H.264 spec 8.2.5).
                    if !is_idr {
                        let mut dpb = self.dpb_manager.lock().unwrap();
                        dpb.apply_mmco_ops(slh.frame_num, slh);
                    }

                    // Track for display-order presentation (unwrapped POC,
                    // since raw POCs may wrap/cycle within the stream)
                    let unwrapped = self.unwrapped_poc(poc);
                    self.reorder.insert((unwrapped, seq), (pic_index, seq));

                    if std::env::var("NVDEC_DEBUG_SURFACES").is_ok() {
                        eprintln!(
                            "[DECODE] seq={} fn={} poc={} uw={} pre_surf={} post_surf={} ref={}",
                            seq,
                            slh.frame_num,
                            poc,
                            unwrapped,
                            curr_pic_idx,
                            pic_index,
                            is_reference
                        );
                    }

                    // Extract ready frames in display (POC) order
                    self.extract_ready_frames(unwrapped);
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                    break;
                }
                Err(e) => {
                    return Err(NvdecError::DecodeFailed(format!("Parse error: {}", e)));
                }
            }
        }

        self.parsed_offset = self.pending_data.len();
        Ok(())
    }

    /// Create the NVDEC decoder from SPS parameters.
    fn create_decoder(&mut self, sps: &vacc_core::picture::H264Sps) -> NvdecResult<()> {
        let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
        let coded_height = if sps.frame_mbs_only_flag {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
        } else {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
        };

        // Calculate display area from frame cropping
        let crop_left = if sps.frame_cropping_flag {
            sps.frame_crop_left_offset * (if sps.chroma_format_idc == 1 { 2 } else { 1 })
        } else {
            0
        };
        let crop_top = if sps.frame_cropping_flag {
            if sps.frame_mbs_only_flag {
                sps.frame_crop_top_offset * 2
            } else {
                sps.frame_crop_top_offset * 4
            }
        } else {
            0
        };
        let crop_right = if sps.frame_cropping_flag {
            sps.frame_crop_right_offset * (if sps.chroma_format_idc == 1 { 2 } else { 1 })
        } else {
            0
        };
        let crop_bottom = if sps.frame_cropping_flag {
            if sps.frame_mbs_only_flag {
                sps.frame_crop_bottom_offset * 2
            } else {
                sps.frame_crop_bottom_offset * 4
            }
        } else {
            0
        };

        let display_left = crop_left as i32;
        let display_top = crop_top as i32;
        let display_right = (coded_width - crop_right) as i32;
        let display_bottom = (coded_height - crop_bottom) as i32;

        let display_width = (display_right - display_left) as u32;
        let display_height = (display_bottom - display_top) as u32;

        // Pre-check driver capabilities so unsupported streams (e.g. H.264
        // High 10-bit or 4:2:2/4:4:4 on GPUs whose NVDEC only does 8-bit
        // 4:2:0) fail with a clear message instead of an opaque cuvid 801.
        let chroma_fmt = match sps.chroma_format_idc {
            0 => cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
            2 => cudaVideoChromaFormat::cudaVideoChromaFormat_422,
            3 => cudaVideoChromaFormat::cudaVideoChromaFormat_444,
            _ => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        };
        let caps = crate::device::query_decoder_caps(
            cudaVideoCodec::cudaVideoCodec_H264,
            chroma_fmt,
            sps.bit_depth_luma_minus8 as u32,
        )?;
        if caps.bIsSupported == 0 {
            let chroma_name = match chroma_fmt {
                cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome => "4:0:0",
                cudaVideoChromaFormat::cudaVideoChromaFormat_422 => "4:2:2",
                cudaVideoChromaFormat::cudaVideoChromaFormat_444 => "4:4:4",
                _ => "4:2:0",
            };
            return Err(NvdecError::DecoderCreationFailed(format!(
                "HW does not support H.264 {}-bit {} decode on this NVDEC device (cuvidGetDecoderCaps reports unsupported)",
                8 + sps.bit_depth_luma_minus8, chroma_name,
            )));
        }

        let output_format = if sps.bit_depth_luma_minus8 > 0 {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
        } else {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        };

        // Number of decode/output surfaces: enough to hold all DPB references
        // plus frames held back by B-frame reordering, with headroom. Clamped
        // to a sane range. The DPB manager wraps CurrPicIdx at this value.
        let num_surfaces = sps.max_num_ref_frames.saturating_add(4).clamp(5u32, 32u32);

        let create_info = CUVIDDECODECREATEINFO {
            ulWidth: coded_width as _,
            ulHeight: coded_height as _,
            ulNumDecodeSurfaces: num_surfaces as _,
            CodecType: cudaVideoCodec::cudaVideoCodec_H264,
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
            // coded size) or cuvid scales the output. The actual picture crop
            // is applied during readback via `self.display_area` (set below).
            display_area: CUVIDRECT {
                left: 0,
                top: 0,
                right: coded_width as _,
                bottom: coded_height as _,
            },
            OutputFormat: output_format,
            DeinterlaceMode: if sps.frame_mbs_only_flag {
                cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave
            } else {
                cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive
            },
            ulTargetWidth: coded_width as _,
            ulTargetHeight: coded_height as _,
            ulNumOutputSurfaces: num_surfaces as _,
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

        // The DPB manager wraps CurrPicIdx at max_decode_surfaces, which MUST
        // equal ulNumDecodeSurfaces or cuvidDecodePicture rejects the index.
        {
            let mut dpb = self.dpb_manager.lock().unwrap();
            dpb.set_max_decode_surfaces(create_info.ulNumDecodeSurfaces as i32);
        }

        // Update decoder info
        let profile_idc = {
            let p = self.profile_idc.lock().unwrap();
            *p
        };

        let mut info = self.info.lock().unwrap();
        *info = DecoderInfo {
            backend: "nvdec".to_string(),
            codec: VideoCodec::DecodeH264,
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
            profile_idc,
            dpb_slots: num_surfaces,
        };

        // Store display area
        {
            let mut display_area = self.display_area.lock().unwrap();
            *display_area = (display_left, display_top, display_right, display_bottom);
        }

        // Update previous coded size
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (coded_width, coded_height);
        }

        // Mark as initialized
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = true;
        }

        Ok(())
    }

    /// Recreate the decoder due to resolution change.
    fn recreate_decoder(&mut self, sps: &vacc_core::picture::H264Sps) -> NvdecResult<()> {
        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();

        // Destroy old decoder
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
        }

        // Clear DPB state
        {
            let mut dpb = self.dpb_manager.lock().unwrap();
            dpb.reset();
        }

        // Clear pending frames
        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.clear();
        }

        // Clear reorder buffer and presentation state
        self.reset_presentation_state();

        // Reset POC calculator
        {
            let mut poc_calc = self.poc_calculator.lock().unwrap();
            poc_calc.reset();
        }

        // Reset decoder handle
        {
            let mut decoder = self.decoder.lock().unwrap();
            *decoder = std::ptr::null_mut();
        }

        // Reset prev coded size so create_decoder proceeds
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (0, 0);
        }

        // Create new decoder
        self.create_decoder(sps)?;

        Ok(())
    }

    /// Compute the monotonic (unwrapped) presentation position for a frame
    /// with the given (possibly wrapping) POC.
    ///
    /// Raw POCs (`pic_order_cnt_lsb`) wrap at `poc_period`, so they are not
    /// unique across the stream. The wrap cycle is tracked in DECODE order:
    /// when the raw POC jumps backward by more than half a period it wrapped
    /// forward (cycle++), and forward by more than half a period it wrapped
    /// backward (cycle--). The unwrapped position is
    /// `raw_poc + poc_cycle * poc_period`, which is unique per frame and
    /// preserves display (POC) order within and across cycles.
    fn unwrapped_poc(&mut self, poc: i32) -> i32 {
        let period = self.poc_period;
        if period <= 1 {
            // No wrap (or unknown period): raw POC is already the position.
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

    /// Clear the reorder buffer and all presentation-order state.
    fn reset_presentation_state(&mut self) {
        self.reorder.clear();
        self.presented_count = 0;
        self.poc_min = i32::MAX;
        self.poc_max = i32::MIN;
        self.poc_gcd = 0;
        self.prev_decoded_poc = None;
        self.last_presented_unwrapped = None;
        self.poc_cycle = 0;
    }

    /// Extract ready frames in DISPLAY (ascending POC) order.
    ///
    /// `current_uw_poc` is the unwrapped POC of the picture that was just
    /// decoded (it is already in `self.reorder`).
    ///
    /// Output rule (H.264 8.2.5.2, same shape as the common DPB's display
    /// bumping): present the lowest pending POC only once the *current*
    /// picture has a strictly greater POC. The current picture is the only
    /// evidence available — a higher-POC picture merely sitting in the reorder
    /// buffer proves nothing, because with hierarchical B-frames (b-pyramid)
    /// pictures with a POC *below* the current one are still to be decoded
    /// (e.g. decode order ... POC 14, POC 10, POC 8: after POC 10 the pending
    /// set {10, 14} would release POC 10 even though POC 8 has not been
    /// decoded yet, swapping two adjacent display frames).
    ///
    /// Pictures still held back when the stream ends are drained by
    /// [`flush`](Self::flush) in ascending POC order.
    fn extract_ready_frames(&mut self, current_uw_poc: i32) {
        while let Some((&key, &(min_idx, min_seq))) = self.reorder.iter().next() {
            // Hold back until the just-decoded picture overtakes the oldest
            // pending one; until then a lower-POC picture may still arrive.
            if key.0 >= current_uw_poc {
                break;
            }

            if std::env::var("NVDEC_DEBUG_SURFACES").is_ok() {
                eprintln!(
                    "[PRESENT] uw_poc={} surf={} seq={}",
                    key.0, min_idx, min_seq
                );
            }
            match self.extract_frame(min_idx, min_seq) {
                Some(frame) => {
                    // Mark extracted by seq so the surface can be recycled.
                    {
                        let mut dpb = self.dpb_manager.lock().unwrap();
                        dpb.mark_extracted(min_seq);
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

    /// Extract a decoded frame from the NVDEC decoder by picture index.
    ///
    /// Maps the GPU surface, copies Y and UV planes with cropping,
    /// de-interleaves NV12 to I420, and returns a [`DecodedFrame`].
    /// `seq` identifies the specific decoded frame for metadata lookup
    /// (pic_index alone is ambiguous once surfaces wrap).
    fn extract_frame(&self, pic_index: i32, seq: i32) -> Option<DecodedFrame> {
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

        // 10-bit content decodes into P016 surfaces (2 bytes per sample,
        // left-justified values); 8-bit uses NV12.
        let bps = if info.luma_bit_depth == ComponentBitDepth::Bit10 {
            2
        } else {
            1
        };

        let funcs = match get_funcs() {
            Ok(f) => f,
            Err(_) => return None,
        };
        let _ = cu_ctx_set_current();

        // Map the decoded frame
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

        // Get display area (cropping offsets)
        let display_area = {
            let d = self.display_area.lock().unwrap();
            *d
        };
        let (crop_left, crop_top, _crop_right, _crop_bottom) = display_area;

        // Get (or grow) the cached pinned host buffer. One contiguous block
        // holds both the Y plane and the interleaved UV plane.
        let y_size = display_width * display_height * bps;
        let interleaved_uv_size = display_width * (display_height / 2) * bps;
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

        // Copy the Y plane with a single 2D memcpy (accounts for cropping).
        let copy_y = CUDA_MEMCPY2D {
            srcXInBytes: (crop_left as usize * bps) as u64,
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
            dstPitch: (display_width * bps) as u64,
            WidthInBytes: (display_width * bps) as u64,
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

        // Copy the UV plane (NV12: interleaved UV rows follow the Y rows) with
        // a single 2D memcpy. UV rows start after `coded_height` Y rows.
        let coded_height = info.coded_size.height as u64;
        let copy_uv = CUDA_MEMCPY2D {
            srcXInBytes: (crop_left as usize * bps) as u64,
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
            dstPitch: (display_width * bps) as u64,
            WidthInBytes: (display_width * bps) as u64,
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

        // Unmap the frame
        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

        // Copy from pinned memory to owned buffers (the pinned buffer is
        // cached and reused, so it is not freed here).
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

        // De-interleave NV12/P016 UV to planar U and V (byte-based).
        let uv_size = (display_width / 2) * (display_height / 2) * bps;
        let mut u_plane = vec![0u8; uv_size];
        let mut v_plane = vec![0u8; uv_size];
        for y in 0..(display_height / 2) {
            for x in 0..(display_width / 2) {
                let src_idx = (y * display_width + x * 2) * bps;
                let dst_idx = (y * (display_width / 2) + x) * bps;
                u_plane[dst_idx..dst_idx + bps]
                    .copy_from_slice(&interleaved_uv[src_idx..src_idx + bps]);
                v_plane[dst_idx..dst_idx + bps]
                    .copy_from_slice(&interleaved_uv[src_idx + bps..src_idx + 2 * bps]);
            }
        }

        // Build output buffer
        let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
        buffer.extend_from_slice(&y_plane);
        buffer.extend_from_slice(&u_plane);
        buffer.extend_from_slice(&v_plane);

        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
        let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };

        let pixel_data = Some(PixelData {
            // P016 samples stay left-justified (top-justified 10-bit in u16);
            // the consumer normalizes to bottom-justified for bps=2 formats.
            format: if bps == 2 {
                "I420_16BIT".to_string()
            } else {
                "I420".to_string()
            },
            y: PixelPlane {
                data: y_ptr,
                pitch: display_width * bps,
                width: display_width,
                height: display_height,
            },
            u: PixelPlane {
                data: u_ptr,
                pitch: display_width / 2 * bps,
                width: display_width / 2,
                height: display_height / 2,
            },
            v: Some(PixelPlane {
                data: v_ptr,
                pitch: display_width / 2 * bps,
                width: display_width / 2,
                height: display_height / 2,
            }),
            buffer,
        });

        // Get frame index
        let frame_index = {
            let mut count = self.frame_count.lock().unwrap();
            let idx = *count;
            *count += 1;
            idx
        };

        // Get POC and is_reference from DPB entry (by unique seq)
        let (poc, is_reference) = {
            let dpb = self.dpb_manager.lock().unwrap();
            if let Some(entry) = dpb.get_entry_by_seq(seq) {
                (entry.pic_order_cnt, entry.is_reference)
            } else {
                (0, false)
            }
        };

        // Create decoded frame
        Some(DecodedFrame {
            frame_index,
            timestamp: 0,
            width: info.display_size.width,
            height: info.display_size.height,
            skipped: false,
            pts_valid: false,
            poc,
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
                ref_pic: is_reference,
                apply_film_grain: false,
            },
            sync_info: vacc_core::frame::FrameSyncInfo::default(),
            pixel_data,
        })
    }

    /// Dump a decoded picture (in DECODE order) to `{dump_decode_order_path}_{count}.yuv`
    /// as NV12 (Y plane + interleaved UV, full coded size, no cropping), matching the
    /// C reference (cuvid_ref.c) format for direct surface-content comparison.
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

        let host = cu_mem_host_alloc(total);
        if let Ok(p) = host {
            // Copy Y plane (full coded size, no cropping, matching C-ref).
            let copy_y = CUDA_MEMCPY2D {
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
            // Copy interleaved UV plane (NV12: UV rows follow Y rows).
            let copy_uv = CUDA_MEMCPY2D {
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
        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
    }

    /// Present every picture still held in the reorder buffer, in ascending
    /// (unwrapped) POC order, and leave the buffer empty.
    ///
    /// Used at end of stream and at IDR boundaries, where no later picture can
    /// still reorder in front of them.
    fn drain_reorder(&mut self) -> Vec<DecodedFrame> {
        let remaining: Vec<((i32, i32), (i32, i32))> =
            self.reorder.iter().map(|(k, v)| (*k, *v)).collect();
        let mut frames = Vec::with_capacity(remaining.len());
        for (key, (pic_index, seq)) in remaining {
            if let Some(frame) = self.extract_frame(pic_index, seq) {
                let mut dpb = self.dpb_manager.lock().unwrap();
                dpb.mark_extracted(seq);
                self.last_presented_unwrapped = Some(key.0);
                self.presented_count += 1;
                frames.push(frame);
            }
        }
        self.reorder.clear();
        frames
    }

    /// Get the next decoded frame if available.
    fn get_decoded_frame(&self) -> Option<DecodedFrame> {
        let frame = {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.pop_front()
        };
        frame
    }
}

impl Decoder for NvdecH264Decoder {
    type Error = NvdecError;

    /// Create a new decoder from raw bitstream data.
    fn new(data: Vec<u8>) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        Self::new(data)
    }

    /// Create a new decoder with a specific codec format.
    fn new_with_format(data: Vec<u8>, codec: VideoCodec, _format: &VideoFormat) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        if codec != VideoCodec::DecodeH264 {
            return Err(NvdecError::UnsupportedCodec(codec));
        }
        Self::new(data)
    }

    /// Get decoder information (codec, resolution, format, etc.).
    fn info(&self) -> DecoderInfo {
        self.info.lock().unwrap().clone()
    }

    /// Submit additional bitstream data for decoding.
    fn submit(&mut self, data: &[u8]) -> NvdecResult<()> {
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    /// Decode and return the next available frame.
    fn decode(&mut self) -> NvdecResult<Option<DecodedFrame>> {
        // Parse any pending data
        self.parse_and_decode()?;

        // Get decoded frame if available
        Ok(self.get_decoded_frame())
    }

    /// Flush the decoder pipeline and return all remaining frames.
    ///
    /// Extracts all remaining frames from the DPB in POC order.
    fn flush(&mut self) -> NvdecResult<Vec<DecodedFrame>> {
        // Process any remaining pending data first
        self.parse_and_decode()?;

        // Collect frames already in the pending queue
        let mut frames: Vec<DecodedFrame> = {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.drain(..).collect()
        };

        // Extract remaining frames from the reorder buffer in ascending
        // POC (display) order. BTreeMap iteration is already sorted by
        // (unwrapped_poc, seq).
        frames.append(&mut self.drain_reorder());

        Ok(frames)
    }

    /// Reset the decoder for re-use with new bitstream data.
    fn reset(&mut self) -> NvdecResult<()> {
        let _ = cu_ctx_set_current();
        let funcs = get_funcs()?;

        // Destroy old decoder
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            let mut d = self.decoder.lock().unwrap();
            *d = std::ptr::null_mut();
        }

        // Reset parser
        self.parser.reset();

        // Clear DPB state
        {
            let mut dpb = self.dpb_manager.lock().unwrap();
            dpb.reset();
        }

        // Clear pending frames
        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.clear();
        }

        // Reset parsing state
        self.parsed_offset = 0;

        // Reset reorder buffer and presentation state
        self.reset_presentation_state();

        // Reset POC calculator
        {
            let mut poc_calc = self.poc_calculator.lock().unwrap();
            poc_calc.reset();
        }

        // Reset initialized flag
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = false;
        }

        // Reset prev coded size
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (0, 0);
        }

        // Re-parse all data
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

impl Drop for NvdecH264Decoder {
    fn drop(&mut self) {
        // Ensure CUDA context is current for cleanup operations
        let _ = cu_ctx_set_current();

        // Destroy decoder
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            if let Ok(funcs) = get_funcs() {
                let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            }
        }
    }
}

/// Dump the exact [`CUVIDPICPARAMS`] being submitted to `cuvidDecodePicture`
/// in the same text format as the NVIDIA C reference (cuvid_ref.c), so the
/// output can be diffed character-for-character.
///
/// The file is created/truncated on the first picture of the run
/// (`pic_num == 0`) and appended for subsequent pictures.
fn dump_cuvid_picparams(path: &std::path::Path, pic_num: u32, p: &CUVIDPICPARAMS) {
    use std::io::Write;

    let file = match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(pic_num == 0)
        .append(pic_num != 0)
        .open(path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[NVDEC-DUMP] cannot open {}: {}", path.display(), e);
            return;
        }
    };
    let mut out = std::io::BufWriter::new(file);

    let h = unsafe { &p.CodecSpecific.h264 };
    let bs = unsafe { std::slice::from_raw_parts(p.pBitstreamData, p.nBitstreamDataLen as usize) };
    let offsets = unsafe { std::slice::from_raw_parts(p.pSliceDataOffsets, p.nNumSlices as usize) };

    let mut s = String::new();
    s.push_str(&format!("=== PIC {} ===\n", pic_num));
    s.push_str(&format!(
        "PicWidthInMbs={} FrameHeightInMbs={} CurrPicIdx={} field_pic_flag={} bottom_field_flag={} second_field={}\n",
        p.PicWidthInMbs,
        p.FrameHeightInMbs,
        p.CurrPicIdx,
        p.field_pic_flag,
        p.bottom_field_flag,
        p.second_field
    ));
    s.push_str(&format!(
        "nBitstreamDataLen={} nNumSlices={} ref_pic_flag={} intra_pic_flag={}\n",
        p.nBitstreamDataLen, p.nNumSlices, p.ref_pic_flag, p.intra_pic_flag
    ));

    // slice_offsets=<u32> ... (one per slice, space-separated, trailing space)
    s.push_str("slice_offsets=");
    for &off in offsets {
        s.push_str(&format!("{} ", off));
    }
    s.push('\n');

    // bs_first16=<hex> ... (first min(16, len) bytes of pBitstreamData)
    s.push_str("bs_first16=");
    for b in &bs[..bs.len().min(16)] {
        s.push_str(&format!("{:02x} ", b));
    }
    s.push('\n');

    // bs_at_slice<i>(off=<u32>)=<hex> ... (8 bytes at each slice offset)
    for (i, &off) in offsets.iter().enumerate() {
        let start = off as usize;
        let count = 8.min(bs.len().saturating_sub(start));
        s.push_str(&format!("bs_at_slice{}(off={})=", i, off));
        for b in &bs[start..start + count] {
            s.push_str(&format!("{:02x} ", b));
        }
        s.push('\n');
    }

    // SPS/PPS/PIC/DPB lines from CodecSpecific.h264
    s.push_str(&format!(
        "SPS: log2_max_frame_num_minus4={} pic_order_cnt_type={} log2_max_pic_order_cnt_lsb_minus4={} delta_pic_order_always_zero_flag={} frame_mbs_only_flag={} direct_8x8_inference_flag={} num_ref_frames={} residual_colour_transform_flag={} bit_depth_luma_minus8={} bit_depth_chroma_minus8={} qpprime_y_zero_transform_bypass_flag={}\n",
        h.log2_max_frame_num_minus4,
        h.pic_order_cnt_type,
        h.log2_max_pic_order_cnt_lsb_minus4,
        h.delta_pic_order_always_zero_flag,
        h.frame_mbs_only_flag,
        h.direct_8x8_inference_flag,
        h.num_ref_frames,
        h.residual_colour_transform_flag,
        h.bit_depth_luma_minus8,
        h.bit_depth_chroma_minus8,
        h.qpprime_y_zero_transform_bypass_flag
    ));
    s.push_str(&format!(
        "PPS: entropy_coding_mode_flag={} pic_order_present_flag={} num_ref_idx_l0_active_minus1={} num_ref_idx_l1_active_minus1={} weighted_pred_flag={} weighted_bipred_idc={} pic_init_qp_minus26={} deblocking_filter_control_present_flag={} redundant_pic_cnt_present_flag={} transform_8x8_mode_flag={} MbaffFrameFlag={} constrained_intra_pred_flag={} chroma_qp_index_offset={} second_chroma_qp_index_offset={}\n",
        h.entropy_coding_mode_flag,
        h.pic_order_present_flag,
        h.num_ref_idx_l0_active_minus1,
        h.num_ref_idx_l1_active_minus1,
        h.weighted_pred_flag,
        h.weighted_bipred_idc,
        h.pic_init_qp_minus26,
        h.deblocking_filter_control_present_flag,
        h.redundant_pic_cnt_present_flag,
        h.transform_8x8_mode_flag,
        h.MbaffFrameFlag,
        h.constrained_intra_pred_flag,
        h.chroma_qp_index_offset,
        h.second_chroma_qp_index_offset
    ));
    s.push_str(&format!(
        "PIC: ref_pic_flag={} frame_num={} CurrFieldOrderCnt=[{},{}]\n",
        h.ref_pic_flag, h.frame_num, h.CurrFieldOrderCnt[0], h.CurrFieldOrderCnt[1]
    ));
    for (i, e) in h.dpb.iter().enumerate() {
        s.push_str(&format!(
            "dpb[{}]: PicIdx={} FrameIdx={} is_long_term={} not_existing={} used_for_reference={} FOC=[{},{}]\n",
            i,
            e.PicIdx,
            e.FrameIdx,
            e.is_long_term,
            e.not_existing,
            e.used_for_reference,
            e.FieldOrderCnt[0],
            e.FieldOrderCnt[1]
        ));
    }

    if let Err(e) = out.write_all(s.as_bytes()) {
        eprintln!("[NVDEC-DUMP] write failed: {}", e);
    }
}
