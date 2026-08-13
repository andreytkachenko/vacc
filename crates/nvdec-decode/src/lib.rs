//! # nvdec-decode
//!
//! NVDEC (NVIDIA Video Decode) backend for vk-video.
//!
//! Provides hardware-accelerated video decoding using NVIDIA's NVDEC via
//! the Video Codec SDK (cuviddec.h).
//!
//! ## Features
//!
//! - H.264 hardware decoding
//! - Uses existing bitstream parsers from vk-video-parser
//! - Implements the Decoder trait from vk-video-core
//!
//! ## Example
//!
//! ```no_run
//! use nvdec_decode::NvdecDecoder;
//! use vk_video_core::decoder::Decoder;
//!
//! let data = std::fs::read("video.h264").unwrap();
//! let mut decoder = NvdecDecoder::new(data).unwrap();
//!
//! while let Some(frame) = decoder.decode().unwrap() {
//!     println!("Decoded frame {}", frame.frame_index);
//! }
//! ```

pub mod ffi;
pub mod error;
pub mod device;
pub mod decoder;

pub use error::{NvdecError, NvdecResult};
pub use device::{CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, cu_memcpy_2d, init_nvdec, is_available, is_codec_supported, query_decoder_caps};
pub use decoder::NvdecH264Decoder;

/// Convenience type alias for the H.264 decoder.
pub type NvdecDecoder = NvdecH264Decoder;
