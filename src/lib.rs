//! # vacc
//!
//! One decoder for every backend: [`VaccDecoder`] accepts a
//! [`DecoderConfig`] with a preferred backend order (Vulkan, NVDEC, VAAPI,
//! software) and falls back to the next backend in the list whenever the
//! previous one is unavailable or cannot decode the stream.
//!
//! ## Quick start
//!
//! ```no_run
//! use vacc_core::decoder::Decoder;
//! use vacc::{Backend, DecoderConfig, VaccDecoder};
//!
//! let data = std::fs::read("video.h264").unwrap();
//!
//! // Default chain: vulkan -> nvdec -> vaapi -> software.
//! let mut decoder = VaccDecoder::new_auto(data).unwrap();
//! println!("using backend: {}", decoder.backend());
//! for frame in decoder.decode_all(usize::MAX).unwrap() {
//!     println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
//! }
//! ```
//!
//! ## Custom fallback order
//!
//! ```no_run
//! use vacc::{Backend, DecoderConfig, VaccDecoder};
//!
//! let data = std::fs::read("video.h264").unwrap();
//! // Prefer NVDEC, fall back to software only.
//! let config = DecoderConfig::new([Backend::Nvdec, Backend::Software]);
//! // or: DecoderConfig::only(Backend::Vaapi) / DecoderConfig::default_order()
//! let decoder = VaccDecoder::new(data, &config).unwrap();
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
pub use decoder::{decode_all, VaccDecoder};
pub use error::{BackendFailure, UnifiedError, UnifiedResult};

// Re-export the core API so a user only needs this crate for the common path.
pub use vacc_core::{decoder::Decoder, frame::DecodedFrame, DecoderInfo, VideoCodec};
