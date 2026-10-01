//! # vacc-nvdec-decode
//!
//! Hardware-accelerated video decoding using NVIDIA's NVDEC (Video Decode) engine
//! via the Video Codec SDK (`cuviddec.h`).
//!
//! ## Overview
//!
//! This crate provides a Rust wrapper around NVIDIA's NVDEC hardware decoder,
//! implementing the [`vacc_core::decoder::Decoder`] trait for seamless
//! integration with the vacc ecosystem. It supports H.264 decoding with
//! automatic SPS/PPS parsing, DPB (Decoded Picture Buffer) management, and
//! frame reordering.
//!
//! ## Architecture
//!
//! The decoder uses a two-component architecture:
//!
//! 1. **vacc-parser** (`H264Parser`): Rust-based H.264 bitstream parser that
//!    extracts SPS/PPS NAL units, calculates POC values, and identifies slices.
//!
//! 2. **Custom Decoder** (`cuvidDecodePicture` + frame extraction): Uses the
//!    NVIDIA decoder engine for hardware-accelerated decode, then maps and
//!    copies decoded frames to host memory in I420 (planar YUV 4:2:0) format.
//!
//! ```text
//! Bitstream ──► H264Parser ──► SPS/PPS  ──► cuvidCreateDecoder (create/reconfig)
//!                    │
//!                    ├──► Slice ──► build CUVIDPICPARAMS ──► cuvidDecodePicture (HW decode)
//!                    │
//!                    └──► extract_frame ──► map/copy/unmap ──► DecodedFrame (I420)
//! ```
//!
//! The parser is pull-based: the decoder calls `parser.parse()` to advance
//! through the bitstream, processing SPS/PPS and slice data as they appear.
//!
//! ## Thread Safety
//!
//! - The CUDA context is created once and shared across threads via
//!   [`cu_ctx_set_current()`](device::cu_ctx_set_current).
//! - Decoder state uses `Mutex` guards for all shared fields.
//! - Individual `NvdecH264Decoder` instances are **not** `Send`/`Sync` and
//!   should be used from a single thread.
//! - The library-level function pointers (`NvdecFuncs`, `CudaFuncs`) are
//!   stored in `OnceLock` and are `Send` + `Sync`.
//!
//! ## Platform Requirements
//!
//! - **OS**: Linux (x86_64)
//! - **GPU**: NVIDIA GPU with NVDEC hardware support (Kepler or newer)
//! - **Drivers**: NVIDIA proprietary driver with CUDA support
//! - **Libraries**: `libcuda.so` (CUDA Driver API) and `libnvcuvid.so`
//!   (Video Codec SDK runtime)
//!
//! Use [`is_available()`](device::is_available) to check runtime availability.
//!
//! ## Examples
//!
//! ### Basic Decode
//!
//! Decode an entire H.264 bitstream at once:
//!
//! ```no_run
//! use vacc_nvdec_decode::NvdecDecoder;
//! use vacc_core::decoder::Decoder;
//!
//! let data = std::fs::read("video.h264").unwrap();
//! let mut decoder = NvdecDecoder::new(data).unwrap();
//!
//! println!("Codec: {:?}", decoder.info().codec);
//! println!("Resolution: {}x{}",
//!     decoder.info().display_size.width,
//!     decoder.info().display_size.height);
//!
//! while let Some(frame) = decoder.decode().unwrap() {
//!     println!("Frame {}: {}x{}",
//!         frame.frame_index, frame.width, frame.height);
//! }
//! ```
//!
//! ### Streaming (Submit + Decode)
//!
//! Feed data incrementally for streaming scenarios:
//!
//! ```no_run
//! use vacc_nvdec_decode::NvdecDecoder;
//! use vacc_core::decoder::Decoder;
//!
//! // Initialize with SPS/PPS data (first access unit)
//! let header = std::fs::read("header.h264").unwrap();
//! let mut decoder = NvdecDecoder::new(header).unwrap();
//!
//! // Submit additional data in chunks
//! let chunk1 = std::fs::read("chunk1.h264").unwrap();
//! let chunk2 = std::fs::read("chunk2.h264").unwrap();
//! decoder.submit(&chunk1).unwrap();
//! decoder.submit(&chunk2).unwrap();
//!
//! // Decode frames as they become available
//! loop {
//!     match decoder.decode() {
//!         Ok(Some(frame)) => { /* process frame */ }
//!         Ok(None) => { /* no frame yet, submit more data */ }
//!         Err(e) => { eprintln!("Decode error: {}", e); break; }
//!     }
//! }
//!
//! // Flush remaining frames from DPB
//! let remaining = decoder.flush().unwrap();
//! for frame in remaining {
//!     println!("Flushed frame {}", frame.frame_index);
//! }
//! ```
//!
//! ### Error Handling
//!
//! ```no_run
//! use vacc_nvdec_decode::{NvdecDecoder, is_available, NvdecError};
//! use vacc_core::decoder::Decoder;
//!
//! // Check availability before decoding
//! if !is_available() {
//!     eprintln!("NVDEC not available on this system");
//!     return;
//! }
//!
//! let data = std::fs::read("video.h264").unwrap();
//! match NvdecDecoder::new(data) {
//!     Ok(mut decoder) => {
//!         while let Some(frame) = decoder.decode().unwrap() {
//!             // process frame
//!         }
//!     }
//!     Err(NvdecError::LibLoadError(msg)) => {
//!         eprintln!("Library not found: {}", msg);
//!     }
//!     Err(NvdecError::DecoderCreationFailed(msg)) => {
//!         eprintln!("Decoder creation failed: {}", msg);
//!     }
//!     Err(e) => {
//!         eprintln!("Unexpected error: {}", e);
//!     }
//! }
//! ```
//!
//! ## Modules
//!
//! - [`decoder`] — H.264 decoder implementation using vacc-parser
//! - [`device`] — CUDA/NVDEC device management and initialization
//! - [`dpb`] — Decoded Picture Buffer management with MMCO support
//! - [`error`] — Error types and result aliases
//! - [`ffi`] — Raw FFI bindings for `cuviddec.h` types and functions
//! - [`picparams`] — CUVIDPICPARAMS construction from parser output
//! - [`poc`] — H.264 Picture Order Count calculation
//! - [`vp9`] — VP9 decoder (`NvdecVp9Decoder`), DPB state, and
//!   `CUVIDPICPARAMS` construction

pub mod av1;
pub mod decoder;
pub mod device;
pub mod dpb;
pub mod error;
mod gpu;
pub mod ffi;
pub mod h265;
pub mod picparams;
pub mod poc;
pub mod vp9;

pub use av1::{NvdecAv1Decoder, build_cuvid_av1_picparams};
pub use decoder::NvdecH264Decoder;
pub use device::{
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CUDA_MEMCPY2D, cu_memcpy_2d, init_nvdec,
    is_available, is_codec_supported, query_decoder_caps,
};
pub use error::{NvdecError, NvdecResult};
pub use h265::NvdecH265Decoder;
pub use vp9::{NvdecVp9Decoder, Vp9DpbState, build_cuvid_vp9_picparams};

/// Outcome of one bounded internal parse/decode pass.
///
/// Every decoder caps each pass at [`MAX_PICTURES_PER_PASS`] pictures so that
/// constructing a decoder from a long bitstream does not queue (and hold in
/// host memory) every frame of the stream up front; `decode`/`flush` resume
/// where the previous pass stopped.
pub const MAX_PICTURES_PER_PASS: u32 = 16;

pub enum PassOutcome {
    /// The per-pass picture budget was hit while input remained in the
    /// current window. Call `decode` (or `flush`) again to continue.
    More,
    /// The window was processed as far as possible, but its tail NAL may be
    /// incomplete (the window reached the end of the buffered data). No
    /// further pictures can be produced until new data is submitted — or
    /// `flush()` runs at end of stream, which releases the held tail.
    Stalled,
    /// The current input window is fully consumed; no further pictures will be
    /// produced from it until new data is submitted.
    Exhausted,
}

/// Per-pass parse window size. Large enough to hold several pictures of any
/// realistic resolution; a single NAL longer than this (or with no start code
/// within it) simply extends the window to its true terminator.
pub const WINDOW_LIMIT: usize = 4 * 1024 * 1024;

/// Length of a parse window starting at `off` whose NAL units are all complete
/// under the parsers' extraction convention (a NAL followed by a 4-byte start
/// code includes that code's first zero byte in its RBSP). Cuts at the last
/// start code within `limit`; if the window holds no start code at all, extends
/// to the covering NAL's true terminator (or the stream end). The cut point is
/// always a position from which the next window re-anchors on a clean start
/// code, so per-window extraction matches whole-stream extraction byte for
/// byte.
pub(crate) fn window_end(data: &[u8], off: usize, limit: usize) -> usize {
    let total = data.len();
    let end = (off + limit).min(total);

    // Last start code in (off, end].
    let mut last: Option<(usize, usize)> = None;
    let mut pos = off;
    while pos < end {
        match vacc_parser::nal::find_next_start_code(data, pos) {
            Some((p, cl)) => {
                if p > off && p <= end {
                    last = Some((p, cl));
                }
                pos = p + cl;
            }
            None => break,
        }
    }

    match last {
        Some((p, cl)) => (p - off) + if cl == 4 { 1 } else { 0 },
        // No start code in (off, end]: the NAL covering [off..] runs past the
        // limit; extend to its terminator (or the stream end).
        None => match vacc_parser::nal::find_next_start_code(data, end) {
            Some((q, cl)) if q < total => (q - off) + if cl == 4 { 1 } else { 0 },
            _ => total - off,
        },
    }
}

/// Convenience type alias for the H.264 decoder.
///
/// Shorthand for [`NvdecH264Decoder`]. Use this when you only need H.264
/// decoding (the currently supported codec).
///
/// # Example
///
/// ```no_run
/// use vacc_nvdec_decode::NvdecDecoder;
/// use vacc_core::decoder::Decoder;
///
/// let data = std::fs::read("video.h264").unwrap();
/// let mut decoder = NvdecDecoder::new(data).unwrap();
/// while let Some(frame) = decoder.decode().unwrap() {
///     println!("Frame {}", frame.frame_index);
/// }
/// ```
pub type NvdecDecoder = NvdecH264Decoder;
