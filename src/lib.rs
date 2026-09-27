//! # vacc
//!
//! One streaming decoder for every backend: [`VaccDecoder`] accepts a
//! [`DecoderConfig`] with a preferred backend order (Vulkan, NVDEC, VAAPI,
//! software) and falls back to the next backend in the list whenever the
//! previous one is unavailable or cannot decode the stream.
//!
//! The API is streaming-only: construct from an initial chunk of the
//! bitstream, feed the rest with [`vacc_core::decoder::Decoder::submit`],
//! and pull frames one at a time with [`vacc_core::decoder::Decoder::decode`].
//! Nothing in the public API takes or returns the whole stream, so memory
//! stays flat on long-running streams (with the exception of the Vulkan
//! backend, whose inner decoder works on the complete bitstream — see
//! [`VaccDecoder`]).
//!
//! ## Quick start
//!
//! ```no_run
//! use std::io::Read;
//! use vacc_core::decoder::Decoder;
//! use vacc::{Backend, DecoderConfig, VaccDecoder};
//!
//! // Seed the decoder with the stream head; it is consumed as the first
//! // input, and the rest of the stream follows via submit().
//! let mut file = std::fs::File::open("video.h264").unwrap();
//! let mut probe = [0u8; 64 * 1024];
//! let n = file.read(&mut probe).unwrap();
//!
//! // Default chain: vulkan -> nvdec -> vaapi -> software.
//! let mut decoder = VaccDecoder::new_auto(&probe[..n]).unwrap();
//! println!("using backend: {}", decoder.backend());
//!
//! let mut buf = vec![0u8; 1024 * 1024];
//! loop {
//!     let n = file.read(&mut buf).unwrap();
//!     if n == 0 {
//!         break;
//!     }
//!     decoder.submit(&buf[..n]).unwrap();
//!     while let Some(frame) = decoder.decode().unwrap() {
//!         println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
//!     }
//! }
//! for frame in decoder.flush().unwrap() {
//!     println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
//! }
//! ```
//!
//! ## Custom fallback order
//!
//! ```no_run
//! use vacc::{Backend, DecoderConfig, VaccDecoder};
//!
//! let probe: &[u8] = b"\x00\x00\x00\x01..."; // initial bitstream chunk
//! // Prefer NVDEC, fall back to software only.
//! let config = DecoderConfig::new([Backend::Nvdec, Backend::Software]);
//! // or: DecoderConfig::only(Backend::Vaapi) / DecoderConfig::default_order()
//! let decoder = VaccDecoder::new(probe, &config).unwrap();
//! ```
//!
//! If every configured backend fails, the returned error lists each
//! per-backend failure so the caller can report *why* nothing worked.
//!
//! ## Backends and codecs
//!
//! | Backend    | H.264 | H.265 | VP9 | AV1 |
//! |------------|-------|-------|-----|-----|
//! | `vulkan`   | yes   | yes   | yes | yes |
//! | `nvdec`    | yes   | yes   | yes | yes |
//! | `vaapi`    | yes   | yes   | yes | yes |
//! | `software` | yes   | yes   | -   | -   |
//!
//! Each backend is an optional crate feature (`vulkan`, `nvdec`, `vaapi`,
//! `sw`, all on by default), so you can build a distribution without any
//! given GPU dependency.

pub mod backend;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod error;

pub use backend::Backend;
pub use codec::detect_codec;
pub use config::DecoderConfig;
pub use decoder::VaccDecoder;
pub use error::{BackendFailure, UnifiedError, UnifiedResult};

// Re-export the core API so a user only needs this crate for the common path.
pub use vacc_core::{decoder::Decoder, frame::DecodedFrame, DecoderInfo, VideoCodec};
