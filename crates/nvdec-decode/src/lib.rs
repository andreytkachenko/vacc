//! # nvdec-decode
//!
//! Hardware-accelerated video decoding using NVIDIA's NVDEC (Video Decode) engine
//! via the Video Codec SDK (`cuviddec.h`).
//!
//! ## Overview
//!
//! This crate provides a Rust wrapper around NVIDIA's NVDEC hardware decoder,
//! implementing the [`vk_video_core::decoder::Decoder`] trait for seamless
//! integration with the vk-video ecosystem. It supports H.264 decoding with
//! automatic SPS/PPS parsing, DPB (Decoded Picture Buffer) management, and
//! frame reordering.
//!
//! ## Architecture
//!
//! The decoder uses a two-component architecture:
//!
//! 1. **vk-video-parser** (`H264Parser`): Rust-based H.264 bitstream parser that
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
//! use nvdec_decode::NvdecDecoder;
//! use vk_video_core::decoder::Decoder;
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
//! use nvdec_decode::NvdecDecoder;
//! use vk_video_core::decoder::Decoder;
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
//! use nvdec_decode::{NvdecDecoder, is_available, NvdecError};
//! use vk_video_core::decoder::Decoder;
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
//! - [`decoder`] — H.264 decoder implementation using vk-video-parser
//! - [`device`] — CUDA/NVDEC device management and initialization
//! - [`dpb`] — Decoded Picture Buffer management with MMCO support
//! - [`error`] — Error types and result aliases
//! - [`ffi`] — Raw FFI bindings for `cuviddec.h` types and functions
//! - [`picparams`] — CUVIDPICPARAMS construction from parser output
//! - [`poc`] — H.264 Picture Order Count calculation
//! - [`vp9`] — VP9 decoder (`NvdecVp9Decoder`), DPB state, and
//!   `CUVIDPICPARAMS` construction

pub mod decoder;
pub mod device;
pub mod dpb;
pub mod error;
pub mod ffi;
pub mod h265;
pub mod picparams;
pub mod poc;
pub mod vp9;

pub use decoder::NvdecH264Decoder;
pub use device::{
    cu_memcpy_2d, init_nvdec, is_available, is_codec_supported, query_decoder_caps, CUDA_MEMCPY2D,
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST,
};
pub use error::{NvdecError, NvdecResult};
pub use h265::NvdecH265Decoder;
pub use vp9::{build_cuvid_vp9_picparams, NvdecVp9Decoder, Vp9DpbState};

/// Convenience type alias for the H.264 decoder.
///
/// Shorthand for [`NvdecH264Decoder`]. Use this when you only need H.264
/// decoding (the currently supported codec).
///
/// # Example
///
/// ```no_run
/// use nvdec_decode::NvdecDecoder;
/// use vk_video_core::decoder::Decoder;
///
/// let data = std::fs::read("video.h264").unwrap();
/// let mut decoder = NvdecDecoder::new(data).unwrap();
/// while let Some(frame) = decoder.decode().unwrap() {
///     println!("Frame {}", frame.frame_index);
/// }
/// ```
pub type NvdecDecoder = NvdecH264Decoder;
