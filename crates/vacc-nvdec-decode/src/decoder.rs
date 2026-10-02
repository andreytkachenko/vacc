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
use vacc_parser::{BitstreamPacket, PacketFlags, ParseResult, VideoParser, h264::H264Parser};

use crate::device::{
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CUDA_MEMCPY2D, NvdecFuncs, cu_ctx_set_current,
    cu_event_create, cu_event_destroy, cu_event_record,
    cu_event_synchronize,
    cu_mem_free_host,
    cu_mem_host_alloc, cu_memcpy_2d, cu_memcpy_2d_async, cu_stream_create, cu_stream_destroy,
    cu_stream_synchronize, get_funcs, init_nvdec,
};
use crate::dpb::NvdecDpbManager;
use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{
    CUDA_SUCCESS, CUVIDDECODECREATEINFO, CUVIDPROCPARAMS, CUVIDRECT, CUdeviceptr,
    CUvideodecoder, cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags,
    cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat,
};
use crate::picparams::build_cuvid_picparams;
use crate::poc::PocCalculator;
use crate::{window_end, MAX_PICTURES_PER_PASS, PassOutcome, WINDOW_LIMIT};

/// Depth of the per-decoder copy-done event pool. Each extraction records the
/// next event in round-robin; a consumer that lags more than one full pool
/// behind simply over-waits on a reused event (still correct: the reused
/// record is ordered after the original copy on the same stream). 16 matches
/// the typical PREFEED depth of packet-ahead decode pipelines.
const COPY_EVENT_POOL: usize = 16;

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

    /// Previous coded dimensions (for detecting resolution changes).
    prev_coded_size: Mutex<(u32, u32)>,

    /// Pending bitstream data not yet parsed.
    pending_data: Vec<u8>,

    /// Offset in `pending_data` that has already been parsed.
    parsed_offset: usize,

    /// End-of-stream latch set by [`Decoder::flush`]: the window's tail NAL is
    /// then provably complete, so it is no longer held back. Cleared by
    /// [`Decoder::reset`].
    eos: bool,

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

    /// Zero-copy mode: frames are copied device-to-device into owned buffers
    /// and emitted as [`GpuFrame`](vacc_core::gpu::GpuFrame) instead of host
    /// `PixelData`.
    gpu_mode: bool,

    /// Optional device buffers for direct frame extraction (NV12 at display
    /// size). While set, [`extract_frame`] copies decoded frames into them
    /// with device-to-device 2D memcpys and returns metadata-only frames —
    /// no host staging. See
    /// [`set_device_output_ring`](Self::set_device_output_ring).
    /// `(buffers, next_index)`: frames are extracted round-robin over the
    /// ring, so a single `decode()` call can present several frames (B-frame
    /// bursts) without clobbering a frame that is still in flight; each
    /// returned frame carries its buffer via `DecodedFrame::device_ptr`.
    device_output_ring: Mutex<(Vec<CUdeviceptr>, usize)>,

    /// Per-ring-slot wait events (same length as the ring, or empty). Before
    /// copying into slot `i`, the private copy stream waits on every event in
    /// `slot_waits[i]`. Callers use them to tell the decoder "this buffer is
    /// still being read by my kernel": record the event on your own stream
    /// after the reading work is enqueued, and the next copy into that buffer
    /// is ordered after it. Events must be created with
    /// `CU_EVENT_DISABLE_TIMING` (they are waited on from this decoder's
    /// private context). See
    /// [`set_device_output_ring_with_slot_waits`](Self::set_device_output_ring_with_slot_waits).
    slot_wait_events: Mutex<Vec<Vec<*const std::ffi::c_void>>>,

    /// Private stream for the device-output D2D copies. Created lazily on
    /// first use. Copies are asynchronous: each extraction records a
    /// copy-done event (see `copy_events`) that the returned frame carries,
    /// and the mapped source surface is unmapped later, once its copy has
    /// completed (see `pending_unmaps`).
    copy_stream: Mutex<Option<*mut std::ffi::c_void>>,

    /// Round-robin pool of copy-done events. One event is recorded per
    /// extraction; the frame carries it via
    /// [`DecodedFrame::device_copy_event`] so consumers can host-sync it
    /// (`cuEventSynchronize`) before reading `device_ptr`. Created lazily on
    /// first use.
    copy_events: Mutex<(Vec<*mut std::ffi::c_void>, usize)>,

    /// Mapped surfaces awaiting unmap: `(mapped dev ptr, frame seq,
    /// copy-done event)`. Drained (blocking on each copy-done event, in the
    /// decoder's own context) at the top of every extraction — by then the
    /// copy is long finished, so this never stalls in steady state; flushed
    /// on `reset`/drop.
    pending_unmaps: Mutex<Vec<(u64, i32, *mut std::ffi::c_void)>>,

    /// Set when a decode error was recovered by resetting the decoder (see
    /// `decode`). All buffered input was dropped and the NVDEC handle will be
    /// re-created at the next SPS (i.e., the next keyframe). Cleared by
    /// [`take_recovered`](Self::take_recovered).
    recovered: Mutex<bool>,
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
        Self::new_impl(data, false)
    }

    fn new_impl(data: Vec<u8>, gpu_mode: bool) -> NvdecResult<Self> {
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
            prev_coded_size: Mutex::new((0, 0)),
            pending_data: data,
            parsed_offset: 0,
            eos: false,
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
            gpu_mode,
            device_output_ring: Mutex::new((Vec::new(), 0)),
            slot_wait_events: Mutex::new(Vec::new()),
            copy_stream: Mutex::new(None),
            copy_events: Mutex::new((Vec::new(), 0)),
            pending_unmaps: Mutex::new(Vec::new()),
            recovered: Mutex::new(false),
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

    /// Create a decoder in zero-copy GPU mode: decoded frames are copied
    /// device-to-device into owned CUDA buffers and emitted via
    /// [`DecodedFrame::gpu`](vacc_core::frame::DecodedFrame) (no host pixel
    /// data). The image pipeline and inference engines can then operate on
    /// the device buffer directly. Only 4:2:0 content is supported.
    pub fn new_gpu(data: Vec<u8>) -> NvdecResult<Self> {
        Self::new_impl(data, true)
    }

    /// Returns and clears the flag indicating that the most recent
    /// `decode()` call hit an error and recovered by resetting the decoder.
    ///
    /// When true, all previously submitted but not yet decoded input was
    /// dropped: callers tracking outstanding input (e.g., packet counters)
    /// must reset it, or they will underfeed the pipeline. The decoder
    /// resynchronizes at the next keyframe (the first picture carrying SPS).
    pub fn take_recovered(&self) -> bool {
        let mut r = self.recovered.lock().unwrap();
        std::mem::take(&mut *r)
    }

    /// Recover from a decode error: log it, reset the decoder (destroys the
    /// NVDEC handle, clears DPB and all buffered input), and set the flag
    /// reported by [`take_recovered`](Self::take_recovered). Callers resume
    /// feeding fresh data; the decoder resynchronizes at the next keyframe
    /// (the first picture carrying SPS).
    fn recover_from_error(&mut self, e: &NvdecError) -> NvdecResult<()> {
        eprintln!("[NVDEC] decode error, resetting decoder (resyncs at next keyframe): {e}");
        self.reset()?;
        *self.recovered.lock().unwrap() = true;
        Ok(())
    }

    /// Set the device buffer that decoded frames are extracted into.
    ///
    /// The buffer must hold `display_width * display_height * 3 / 2` bytes:
    /// an NV12 image at display size (Y plane followed by the interleaved UV
    /// plane). While set, [`Decoder::decode`] and [`Decoder::flush`] copy each
    /// frame's Y and UV planes (cropped to the display area) into `dst` with
    /// device-to-device 2D memcpys and return metadata-only frames
    /// (`pixel_data: None`). The copies run on a private decoder-owned stream
    /// and are synchronized before `decode()` returns, so the pixels are
    /// guaranteed to have landed in `dst` by the time the frame is handed out
    /// — no cross-stream ordering is required from the caller.
    ///
    /// Only 8-bit content is supported in this mode; 10/12-bit streams (P010/
    /// P012) fall back to host extraction. Use
    /// [`clear_device_output`](Self::clear_device_output) to switch back.
    pub fn set_device_output(&self, dst: CUdeviceptr) {
        *self.device_output_ring.lock().unwrap() = (vec![dst], 0);
    }

    /// Set a ring of device buffers that decoded frames are extracted into,
    /// round-robin. Prefer this over [`set_device_output`](Self::set_device_output):
    /// one `decode()` call can present several frames (B-frame bursts), and a
    /// single buffer would be clobbered by the later copies of that burst.
    /// Each returned frame carries the buffer it was copied into via
    /// [`DecodedFrame::device_ptr`] and the copy-done event via
    /// [`DecodedFrame::device_copy_event`]. Extraction is asynchronous:
    /// `decode()` returns as soon as the copies are ENQUEUED, so order your
    /// reads of `device_ptr` after `device_copy_event` (e.g.
    /// `cuStreamWaitEvent`).
    ///
    /// Buffers are reused every `dsts.len()` frames. Without slot-wait
    /// events (this setter) the caller must keep the ring deep enough that a
    /// buffer's contents have been consumed before its reuse; with
    /// [`set_device_output_ring_with_slot_waits`](Self::set_device_output_ring_with_slot_waits)
    /// any depth is safe.
    pub fn set_device_output_ring(&self, dsts: &[CUdeviceptr]) {
        *self.device_output_ring.lock().unwrap() = (dsts.to_vec(), 0);
        self.slot_wait_events.lock().unwrap().clear();
    }

    /// Like [`set_device_output_ring`](Self::set_device_output_ring), but
    /// additionally arms per-slot reader barriers. `slot_waits[i]` is the
    /// list of CUDA events (raw `CUevent` pointers, created with
    /// `CU_EVENT_DISABLE_TIMING`) that must be signaled before the decoder
    /// may copy into `dsts[i]` again: before each copy into slot `i`, the
    /// private copy stream waits on every event in `slot_waits[i]`.
    ///
    /// Usage: after enqueuing the work that READS `dsts[i]` (e.g. a
    /// preprocess kernel) on your own stream, record `slot_waits[i][j]` on
    /// that same stream. The next decode copy into slot `i` is then ordered
    /// after your reader — with this mechanism any ring depth (even 1) is
    /// safe no matter how far the consumer lags. Each event must be in a
    /// signaled state before the first copy into its slot (record it once on
    /// your stream at setup), since waiting on a never-recorded event blocks
    /// forever.
    pub fn set_device_output_ring_with_slot_waits(
        &self,
        dsts: &[CUdeviceptr],
        slot_waits: &[Vec<*const std::ffi::c_void>],
    ) {
        assert_eq!(slot_waits.len(), dsts.len(), "one wait list per ring buffer");
        *self.device_output_ring.lock().unwrap() = (dsts.to_vec(), 0);
        *self.slot_wait_events.lock().unwrap() = slot_waits.iter().map(|w| w.to_vec()).collect();
    }

    /// Clear the device output buffers; frames are extracted to host memory
    /// again (the default behavior).
    pub fn clear_device_output(&self) {
        *self.device_output_ring.lock().unwrap() = (Vec::new(), 0);
    }

    /// The SPS parsed from the bitstream, if any — e.g. to read VUI flags
    /// such as `video_full_range_flag` or timing info.
    pub fn active_sps(&self) -> Option<&vacc_core::picture::H264Sps> {
        self.parser.active_sps()
    }

    /// Parse pending data and decode up to [`MAX_PICTURES_PER_PASS`] pictures.
    ///
    /// Each pass works on a bounded window of the bitstream (cut at a NAL
    /// boundary, see [`window_end`]) and decodes at most
    /// [`MAX_PICTURES_PER_PASS`] pictures, so constructing a decoder from a
    /// long stream neither re-copies the whole bitstream per `decode` call nor
    /// queues every frame up front. Returns [`PassOutcome::More`] while input
    /// remains, [`PassOutcome::Exhausted`] once it is fully consumed. The
    /// parser's SPS/PPS and POC state persists across passes; its NAL scan
    /// cache is invalidated per window.
    fn parse_and_decode(&mut self) -> NvdecResult<PassOutcome> {
        let window_len = self.pending_data.len() - self.parsed_offset;
        if window_len == 0 {
            return Ok(PassOutcome::Exhausted);
        }

        let off = self.parsed_offset;
        let cut = window_end(&self.pending_data, off, WINDOW_LIMIT);
        // Only a window that reaches the end of the buffered data can have its
        // last NAL split across submits. A window cut at an internal start
        // code contains only complete NALs — flagging it would make the parser
        // hold back the (complete) tail NAL, which this pass would then
        // consume without decoding.
        let tail_hold = !self.eos && off + cut == self.pending_data.len();
        let window = self.pending_data[off..off + cut].to_vec();

        self.parser.invalidate_nal_cache();
        let mut packet = BitstreamPacket::new(window);
        packet.flags.set(PacketFlags::TRUNCATED_TAIL, tail_hold);

        let mut pictures = 0u32;
        let mut capped = false;
        let mut last_consumed = 0usize;
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
                    if let Some(sps_box) = sps
                        && let Some(h264_sps) =
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

                        let resolution_changed = prev_w != coded_width || prev_h != coded_height;
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
                Ok(ParseResult::Slice {
                    slices,
                    bytes_consumed,
                    ..
                }) => {
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
                    // Clone the cached NAL bytes only when they are actually
                    // prepended (first picture after (re)receipt) — not per
                    // picture on the hot path.
                    let need_sps_pps = !*self.sps_pps_fed.lock().unwrap();
                    let sps_nal = need_sps_pps
                        .then(|| self.sps_nal_data.lock().unwrap().clone())
                        .flatten();
                    let pps_nal = need_sps_pps
                        .then(|| self.pps_nal_data.lock().unwrap().clone())
                        .flatten();
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

                    // Wait for THIS picture's decode to complete before its
                    // surface is read (the extraction D2D copy / host staging
                    // both read it). cuvidGetDecodeStatus polls just this
                    // picture, so — unlike a full context sync — it does not
                    // wait on unrelated work (other decoders, consumer-stream
                    // kernels, inference), keeping the pipeline asynchronous.
                    let mut status = crate::ffi::CUVIDGETDECODESTATUS {
                        decodeStatus: crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Invalid,
                        reserved: [0; 31],
                        pReserved: [std::ptr::null_mut(); 8],
                    };
                    loop {
                        let rc = unsafe {
                            (funcs.get_decode_status)(decoder_handle, curr_pic_idx, &mut status)
                        };
                        if rc != CUDA_SUCCESS {
                            return Err(NvdecError::DecodeFailed(format!(
                                "cuvidGetDecodeStatus failed: {}",
                                rc
                            )));
                        }
                        match status.decodeStatus {
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Success
                            | crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Error_Concealed => {
                                break;
                            }
                            crate::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Error => {
                                return Err(NvdecError::DecodeFailed(
                                    "decode reported an error".into(),
                                ));
                            }
                            _ => {
                                // In progress: yield briefly.
                                std::thread::sleep(std::time::Duration::from_micros(50));
                            }
                        }
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

                    // Extract ready frames in display (POC) order
                    self.extract_ready_frames(unwrapped);

                    // `bytes_consumed` is measured from the window start and is
                    // start-code aligned: a safe resume point for the next pass.
                    last_consumed = bytes_consumed;
                    pictures += 1;
                    if pictures >= MAX_PICTURES_PER_PASS {
                        capped = true;
                        break;
                    }
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                    break;
                }
                Err(e) => {
                    return Err(NvdecError::DecodeFailed(format!("Parse error: {}", e)));
                }
            }
        }

        // A pass stopped at the picture budget resumes right after its last
        // slice. A tail-hold pass (window reached the end of the buffered data
        // and its last NAL may be incomplete) resumes right after the last
        // complete NAL and reports Stalled: nothing more can be produced until
        // new data is submitted or flush() runs. Any other exit consumed the
        // whole window.
        self.parsed_offset += if capped || tail_hold { last_consumed } else { cut };
        let more = self.parsed_offset < self.pending_data.len();
        self.compact();
        Ok(if more {
            if tail_hold {
                PassOutcome::Stalled
            } else {
                PassOutcome::More
            }
        } else {
            PassOutcome::Exhausted
        })
    }

    /// Free consumed input bytes so host memory stays flat on long streams.
    /// Cut points are always start-code aligned (see [`window_end`]), so the
    /// remaining data re-anchors cleanly at the new offset 0.
    fn compact(&mut self) {
        if self.parsed_offset > 0 {
            self.pending_data.drain(..self.parsed_offset);
            self.parsed_offset = 0;
        }
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
                8 + sps.bit_depth_luma_minus8,
                chroma_name,
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

        // GPU mode emits tight NV12/P016 device buffers and has no 4:2:2,
        // 4:4:4 or monochrome layout; reject such content up front.
        if self.gpu_mode && sps.chroma_format_idc != 1 {
            return Err(NvdecError::DecoderCreationFailed(format!(
                "GPU mode only supports 4:2:0 content (chroma_format_idc={})",
                sps.chroma_format_idc
            )));
        }

        // Update decoder info

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
            profile_idc: Some(sps.profile_idc as u32),
            dpb_slots: sps.max_num_ref_frames + 1,
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

        // Unmap surfaces whose device-output copies have completed. This
        // runs BEFORE the map below so NVDEC can never be asked to reuse a
        // surface we still hold mapped (a second mapping of the same
        // surface fails with CUDA_ERROR_MAP_FAILED). Blocking on each
        // copy-done event is a no-op in steady state: the copy is a small
        // D2D that finished long before the next frame was decoded.
        self.unmap_completed(decoder, &funcs);

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
            eprintln!(
                "[NVDEC] cuvidMapVideoFrame64 failed: {} (seq={} pic_index={})",
                map_result, seq, pic_index
            );
            return None;
        }

        // Get display area (cropping offsets)
        let display_area = {
            let d = self.display_area.lock().unwrap();
            *d
        };
        let (crop_left, crop_top, _crop_right, _crop_bottom) = display_area;

        // Zero-copy mode: copy the surface into an owned device buffer and
        // skip all host staging. The helper unmaps the surface.
        if self.gpu_mode {
            let gpu_frame = crate::gpu::extract_gpu_surface(
                decoder,
                funcs,
                dev_ptr,
                pitch,
                display_width,
                display_height,
                bps,
                crop_left,
                crop_top,
                info.coded_size.height as i32,
                true,
            )?;

            let frame_index = {
                let mut count = self.frame_count.lock().unwrap();
                let idx = *count;
                *count += 1;
                idx
            };
            let (poc, is_reference) = {
                let dpb = self.dpb_manager.lock().unwrap();
                if let Some(entry) = dpb.get_entry_by_seq(seq) {
                    (entry.pic_order_cnt, entry.is_reference)
                } else {
                    (0, false)
                }
            };

            return Some(DecodedFrame {
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
                pixel_data: None,
                rgb_pixels: None,
                gpu: Some(gpu_frame),
            });
        }

        // Device output mode: copy Y + interleaved UV straight into the
        // caller's device buffer (NV12 layout, cropped to the display size)
        // with two D2D 2D memcpys and return a metadata-only frame — no host
        // staging. 8-bit content only. The copies are asynchronous: the
        // frame carries a copy-done event, and the source surface stays
        // mapped until the copy completes (see `pending_unmaps`).
        // Next ring buffer + slot index for this extraction (round-robin;
        // None when no device output is configured — host staging below).
        let dst: Option<(CUdeviceptr, usize)> = {
            let mut ring = self.device_output_ring.lock().unwrap();
            if ring.0.is_empty() {
                None
            } else {
                let idx = ring.1 % ring.0.len();
                ring.1 += 1;
                Some((ring.0[idx], idx))
            }
        };
        if bps == 1 {
            if let Some((dst, slot_idx)) = dst {
                // Private stream for the D2D copies.
                let copy_stream = {
                    let mut s = self.copy_stream.lock().unwrap();
                    match *s {
                        Some(st) => st,
                        None => match cu_stream_create() {
                            Ok(st) => {
                                *s = Some(st);
                                st
                            }
                            Err(e) => {
                                eprintln!("[NVDEC] copy-stream create failed: {e}");
                                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                                return None;
                            }
                        },
                    }
                };
                // Reader barrier: the caller's work that still reads this ring
                // slot must be finished before we overwrite it. Host-sync the
                // slot event instead of enqueuing a stream wait: an in-flight
                // cuStreamWaitEvent node can execute before the caller's NEXT
                // re-record of the same event is issued, in which case it only
                // waits on the stale record and the copy races the new readers.
                // Blocking here is a no-op in steady state (the reader work is
                // long done by the time the ring wraps around).
                {
                    let waits = self.slot_wait_events.lock().unwrap();
                    if let Some(list) = waits.get(slot_idx) {
                        for ev in list {
                            if let Err(e) = unsafe { cu_event_synchronize(*ev as *mut _) } {
                                eprintln!("[NVDEC] device-output slot sync failed: {e}");
                                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                                return None;
                            }
                        }
                    }
                }
                let copy_y = CUDA_MEMCPY2D {
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
                    dstMemoryType: CU_MEMORYTYPE_DEVICE,
                    _reserved1: 0,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: dst,
                    dstArray: 0,
                    dstPitch: display_width as u64,
                    WidthInBytes: display_width as u64,
                    Height: display_height as u64,
                };
                let uv_dst =
                    unsafe { (dst as *mut u8).add(display_width * display_height) as CUdeviceptr };
                let coded_height = info.coded_size.height as u64;
                let copy_uv = CUDA_MEMCPY2D {
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
                    dstMemoryType: CU_MEMORYTYPE_DEVICE,
                    _reserved1: 0,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: uv_dst,
                    dstArray: 0,
                    dstPitch: display_width as u64,
                    WidthInBytes: display_width as u64,
                    Height: (display_height / 2) as u64,
                };
                match (
                    unsafe { cu_memcpy_2d_async(&copy_y, copy_stream) },
                    unsafe { cu_memcpy_2d_async(&copy_uv, copy_stream) },
                ) {
                    (Ok(CUDA_SUCCESS), Ok(CUDA_SUCCESS)) => {}
                    _ => {
                        eprintln!("[NVDEC] device-output cuMemcpy2DAsync failed");
                        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                        return None;
                    }
                }
                // Record the copy-done event (round-robin pool) and defer the
                // unmap until the copies have actually landed: the mapped
                // surface must stay valid while they read it, and the caller
                // reads `dst` from its own stream with no ordering against
                // this one — hence the event carried by the frame.
                let done_event = match self.record_copy_event(copy_stream) {
                    Some(ev) => ev,
                    None => {
                        eprintln!("[NVDEC] device-output event record failed");
                        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                        return None;
                    }
                };
                self.pending_unmaps
                    .lock()
                    .unwrap()
                    .push((dev_ptr, seq, done_event));
                return Some(self.build_frame(
                    seq,
                    &info,
                    None,
                    Some(dst as *const u8),
                    Some(done_event as *const std::ffi::c_void),
                ));
            }
        }

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

        // Create decoded frame
        Some(self.build_frame(seq, &info, pixel_data, None, None))
    }

    /// Assemble the metadata of a [`DecodedFrame`] (frame index, POC and
    /// reference flag from the DPB) shared by the host-extraction and
    /// device-output paths.
    /// Record the next copy-done event (round-robin pool, created lazily) on
    /// `copy_stream`. Returns the recorded event handle.
    fn record_copy_event(&self, copy_stream: *mut std::ffi::c_void) -> Option<*mut std::ffi::c_void> {
        let mut pool = self.copy_events.lock().unwrap();
        if pool.0.is_empty() {
            let mut events = Vec::with_capacity(COPY_EVENT_POOL);
            for _ in 0..COPY_EVENT_POOL {
                match cu_event_create() {
                    Ok(ev) => events.push(ev),
                    Err(e) => {
                        eprintln!("[NVDEC] copy-event create failed: {e}");
                        break;
                    }
                }
            }
            if events.is_empty() {
                return None;
            }
            pool.0 = events;
        }
        let ev = pool.0[pool.1 % pool.0.len()];
        pool.1 += 1;
        unsafe { cu_event_record(ev, copy_stream) }.ok()?;
        Some(ev)
    }

    /// Wait for every pending device-output copy to land and unmap its
    /// source surface. Called at the top of each extraction (in the
    /// decoder's own context, where the events were recorded): a blocking
    /// `cuEventSynchronize` on an already-signaled event returns
    /// immediately, so this never stalls in steady state. Unmapping before
    /// the new frame's map guarantees NVDEC can recycle any surface.
    fn unmap_completed(&self, decoder: *mut std::ffi::c_void, funcs: &NvdecFuncs) {
        let pending: Vec<(u64, i32, *mut std::ffi::c_void)> =
            self.pending_unmaps.lock().unwrap().drain(..).collect();
        for (ptr, seq, ev) in pending {
            if let Err(e) = unsafe { cu_event_synchronize(ev) } {
                eprintln!("[NVDEC] copy-done event sync failed: {e}");
            }
            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, ptr) };
            let mut dpb = self.dpb_manager.lock().unwrap();
            dpb.mark_unmapped(seq);
        }
    }

    /// Unmap every pending device-output surface, blocking until all copies
    /// have completed. Used on `reset`/drop when no more extractions will
    /// drain the queue.
    fn flush_unmaps(&self) {
        let decoder = *self.decoder.lock().unwrap();
        if decoder.is_null() {
            return;
        }
        let Ok(funcs) = get_funcs() else {
            return;
        };
        let copy_stream = *self.copy_stream.lock().unwrap();
        if let Some(st) = copy_stream {
            // Block until every enqueued copy has landed, so the unmaps
            // below cannot run while a copy still reads its surface.
            let _ = unsafe { cu_stream_synchronize(st) };
        }
        let pending: Vec<(u64, i32, *mut std::ffi::c_void)> =
            self.pending_unmaps.lock().unwrap().drain(..).collect();
        for (ptr, seq, _ev) in pending {
            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, ptr) };
            let mut dpb = self.dpb_manager.lock().unwrap();
            dpb.mark_unmapped(seq);
        }
    }

    fn build_frame(
        &self,
        seq: i32,
        info: &DecoderInfo,
        pixel_data: Option<PixelData>,
        device_ptr: Option<*const u8>,
        device_copy_event: Option<*const std::ffi::c_void>,
    ) -> DecodedFrame {
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

        DecodedFrame {
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
            device_ptr,
            device_copy_event,
            device_seq: device_copy_event.map(|_| seq),
            rgb_pixels: None,
            gpu: None,
        })
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
        self.pending_frames.lock().unwrap().pop_front()
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
    ///
    /// Resumes bounded parse/decode passes until a display-order frame is
    /// ready or the input window is exhausted.
    fn decode(&mut self) -> NvdecResult<Option<DecodedFrame>> {
        loop {
            if let Some(frame) = self.get_decoded_frame() {
                return Ok(Some(frame));
            }
            match self.parse_and_decode() {
                Ok(PassOutcome::More) => continue,
                // The window tail NAL may be incomplete: wait for the next
                // submit() (or flush() at end of stream).
                Ok(PassOutcome::Stalled) => return Ok(None),
                Ok(PassOutcome::Exhausted) => return Ok(None),
                // Recover from a transient decode/parse error instead of
                // failing the whole stream (see `recover_from_error`).
                Err(e) => {
                    self.recover_from_error(&e)?;
                    return Ok(None);
                }
            }
        }
    }

    /// Flush the decoder pipeline and return all remaining frames.
    ///
    /// Extracts all remaining frames from the DPB in POC order.
    fn flush(&mut self) -> NvdecResult<Vec<DecodedFrame>> {
        // End of stream: the window tail NAL is provably complete, so release
        // it and process any remaining pending data first.
        self.eos = true;
        while !matches!(self.parse_and_decode()?, PassOutcome::Exhausted) {}

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

        // Release any still-mapped device-output surfaces before the decoder
        // (and its surface pool) goes away; blocks until copies finish.
        self.flush_unmaps();

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

        // Reset parsing state and drop all buffered input: the decoder is now
        // ready for a new stream fed via submit() (same contract as the sw
        // and vaapi backends). Keeps host memory flat after long streams.
        self.pending_data.clear();
        self.parsed_offset = 0;
        self.eos = false;

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

        // Drop the previous stream's parameter-set NALs: feeding them to a
        // decoder re-initialized from a new stream would corrupt it.
        {
            *self.sps_nal_data.lock().unwrap() = None;
            *self.pps_nal_data.lock().unwrap() = None;
            *self.sps_pps_fed.lock().unwrap() = false;
        }

        Ok(())
    }
}

impl Drop for NvdecH264Decoder {
    fn drop(&mut self) {
        // Ensure CUDA context is current for cleanup operations
        let _ = cu_ctx_set_current();

        // Unmap all pending device-output surfaces (syncs the copy stream as
        // part of it), then destroy the copy stream: destroying a stream with
        // work in flight is undefined behavior.
        self.flush_unmaps();
        if let Some(stream) = self.copy_stream.lock().unwrap().take() {
            let _ = unsafe { cu_stream_synchronize(stream) };
            let _ = unsafe { cu_stream_destroy(stream) };
        }

        // Destroy the copy-done event pool. Callers must have synchronized any
        // stream that waited on these events first (a pending wait on a
        // destroyed event is undefined behavior).
        let (events, _) = {
            let mut pool = self.copy_events.lock().unwrap();
            (std::mem::take(&mut pool.0), pool.1)
        };
        for ev in events {
            let _ = unsafe { cu_event_destroy(ev) };
        }

        // Destroy decoder
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null()
            && let Ok(funcs) = get_funcs()
        {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
        }
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    fn load_born_trailer() -> Vec<u8> {
        let path = format!(
            "{}/assets/born_trailer.h264",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../..")
        );
        std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read test file: {}", path))
    }

    /// Scan Annex-B start codes in `data`, returning
    /// `(start incl. start code, end excl., NAL type)` per NAL. A NAL cut off
    /// at the end of `data` is reported with `end == data.len()`.
    fn nal_ranges(data: &[u8]) -> Vec<(usize, usize, u8)> {
        let mut ranges = Vec::new();
        let mut i = 0;
        while i + 3 <= data.len() {
            let sc_len = if data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && i + 4 <= data.len()
                && data[i + 3] == 1
            {
                4
            } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };
            let nal_start = i + sc_len;
            if nal_start >= data.len() {
                break;
            }
            let mut j = nal_start;
            while j + 2 < data.len() && !(data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 1) {
                j += 1;
            }
            let end = if j + 2 < data.len() { j } else { data.len() };
            ranges.push((i, end, data[nal_start] & 0x1F));
            i = nal_start + 1;
        }
        ranges
    }

    /// Verify that decode() recovers from a decode error instead of
    /// propagating it: the decoder resets (dropping all buffered input),
    /// `take_recovered()` reports the recovery exactly once, and decoding
    /// resumes once fresh SPS/PPS + keyframe data is submitted.
    ///
    /// The recovery is forced with a synthetic [`NvdecError`] via
    /// `recover_from_error` — the same handler `decode()` invokes on any
    /// parse/decode failure (e.g., a failed `cuvidDecodePicture` from corrupt
    /// RTSP input). Real hardware errors are not reproducible deterministically
    /// (NVDEC conceals most bitstream corruption), but from the recovery path's
    /// point of view every error looks the same.
    #[test]
    fn test_decode_error_recovery_resyncs_at_next_keyframe() {
        let data = load_born_trailer();

        // Locate the first IDR and the non-IDR slices that follow it.
        let r = nal_ranges(&data[..1_000_000]);
        let idr_i = r
            .iter()
            .position(|&(_, _, t)| t == 5)
            .expect("born_trailer should contain an IDR");
        let mut non_idr = Vec::new();
        for (i, &(_, _, t)) in r.iter().enumerate() {
            if i > idr_i && (1..=4).contains(&t) {
                non_idr.push(i);
                if non_idr.len() == 2 {
                    break;
                }
            }
        }
        assert_eq!(non_idr.len(), 2, "need two non-IDR slices after the IDR");
        // Warm-up input: SPS/PPS + IDR + two slices (the last one is held by
        // the tail-hold rule until more data arrives).
        let warm_end = r[non_idr[1]].1;

        // Sanity: the decoder works — new() decodes the IDR + first slice and
        // presents the IDR (overtaken by the second picture's POC).
        let mut decoder = NvdecH264Decoder::new(data[..warm_end].to_vec())
            .expect("Failed to create decoder");
        let mut warm = 0;
        loop {
            match decoder.decode() {
                Ok(Some(_)) => warm += 1,
                Ok(None) => break,
                Err(e) => panic!("Unexpected error before recovery: {:?}", e),
            }
        }
        assert!(warm > 0, "expected a frame from the warm-up");

        // Force the recovery path (what decode() does on any parse/decode
        // error): reset + flag.
        assert!(!decoder.take_recovered(), "no recovery before the error");
        decoder
            .recover_from_error(&NvdecError::DecodeFailed("synthetic test error".into()))
            .expect("recovery reset should succeed");
        assert!(decoder.take_recovered(), "take_recovered must report the recovered error");
        assert!(!decoder.take_recovered(), "take_recovered must clear the flag");

        // Resync: re-submit the stream head (SPS/PPS + keyframe) and verify
        // decoding resumes on a freshly created decoder.
        decoder.submit(&data[..warm_end]).expect("resync submit");
        let mut resumed = 0;
        for _ in 0..10 {
            match decoder.decode() {
                Ok(Some(_)) => resumed += 1,
                Ok(None) => break,
                Err(e) => panic!("Unexpected error after resync: {:?}", e),
            }
        }
        assert!(resumed > 0, "Decoder must resume decoding after resync");
    }
}


