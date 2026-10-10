//! # vacc
//!
//! One video codec library for every backend: [`VaccDecoder`] decodes and
//! [`VaccEncoder`] encodes, each accepting a config with a preferred backend
//! order (Vulkan, NVDEC/NVENC, VAAPI, software) and falling back to the next
//! backend in the list whenever the previous one is unavailable or cannot
//! handle the stream.
//!
//! ## Decoding
//!
//! The decode API is streaming-only: construct from an initial chunk of the
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
//! Decode support:
//!
//! | Backend    | H.264 | H.265 | VP9 | AV1 |
//! |------------|-------|-------|-----|-----|
//! | `vulkan`   | yes   | yes   | yes | yes |
//! | `nvdec`    | yes   | yes   | yes | yes |
//! | `vaapi`    | yes   | yes   | yes | yes |
//! | `software` | yes   | yes   | -   | -   |
//!
//! Encode support (H.264 CQP, NV12 in / Annex B out):
//!
//! | Backend  | H.264   |
//! |----------|---------|
//! | `vulkan` | planned |
//! | `nvdec`  | planned |
//! | `vaapi`  | yes     |
//!
//! Each backend is an optional crate feature (`vulkan`, `nvdec`, `vaapi`,
//! `sw`, all on by default); a backend feature enables both its decode and
//! encode crates, so you can build a distribution without any given GPU
//! dependency.
//!
//! ## Encoding
//!
//! [`VaccEncoder`] takes an [`EncoderConfig`] (core parameters plus a
//! preferred backend order) and encodes NV12 frames one at a time:
//!
//! ```no_run
//! use std::io::Write;
//! use vacc_core::encoder::{Encoder, EncoderConfig as CoreCfg, InputFrame};
//! use vacc::{EncoderConfig, VaccEncoder};
//!
//! let params = CoreCfg { width: 320, height: 240, ..Default::default() };
//! let config = EncoderConfig::new(params); // vulkan -> nvdec -> vaapi
//! let mut enc = VaccEncoder::new(&config).unwrap();
//! println!("using backend: {}", enc.backend());
//!
//! let mut out = Vec::new();
//! for i in 0..30u32 {
//!     let y = vec![i as u8; 320 * 240];
//!     let uv = vec![128u8; 320 * 120];
//!     let frame = InputFrame::packed(320, 240, &y, &uv);
//!     let au = enc.encode(&frame, (i as i64) * 33_333_333).unwrap();
//!     out.extend_from_slice(&au.data);
//! }
//! for au in enc.flush().unwrap() {
//!     out.extend_from_slice(&au.data);
//! }
//! let mut f = std::fs::File::create("out.h264").unwrap();
//! f.write_all(&out).unwrap();
//! ```
//!
//! ## Image pipeline
//!
//! A [`DecoderConfig`] may additionally carry an [`ImageConfig`]: after
//! decoding, every frame is optionally down-cast (10/12 -> 8 bit), resized
//! and/or converted to packed RGB(R)(A). On NVIDIA hosts the pipeline runs
//! on the GPU through NPP; on any other host with a Vulkan compute device it
//! runs on the GPU through `vacc-vkimage`; everywhere else (and whenever the
//! GPU path fails) the SIMD software pipeline in `vacc-image` is used. The
//! result lands in the frame itself: scaled 4:2:0 output replaces
//! `pixel_data`, RGB output lands in `rgb_pixels`.
//!
//! ## GPU decode track
//!
//! [`DecoderConfig::with_gpu`] requests the zero-host-copy path: NVDEC writes
//! decoded frames straight into device buffers and the image pipeline runs on
//! the GPU through NPP, so frames come out with [`DecodedFrame::gpu`] set and
//! `pixel_data`/`rgb_pixels` empty. The device pointer is stable while the
//! frame handle is alive, which makes the buffer suitable for downstream
//! inference (e.g. TensorRT). Only the NVDEC backend supports the track;
//! unsupported combinations (e.g. 10-bit sources) transparently fall back to
//! a readback + host processing.

pub mod backend;
pub mod codec;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod error;
mod transform;

pub use backend::Backend;
pub use codec::detect_codec;
pub use config::{DecoderConfig, EncoderConfig};
pub use decoder::VaccDecoder;
pub use encoder::VaccEncoder;
pub use error::{BackendFailure, UnifiedError, UnifiedResult};

// Re-export the core API so a user only needs this crate for the common path.
pub use vacc_core::{
    decoder::Decoder,
    encoder::{EncodedFrame, Encoder, EncoderInfo, InputFrame},
    frame::DecodedFrame,
    DecoderInfo, VideoCodec,
};

// GPU decode track types (device-resident frames).
pub use vacc_core::gpu::{GpuDevice, GpuFrame, GpuPixelFormat};

// Image-pipeline configuration types.
pub use vacc_image::{Affine, ColorSpec, ImageConfig, Interpolation, RgbChannels, Scale, Warp};
