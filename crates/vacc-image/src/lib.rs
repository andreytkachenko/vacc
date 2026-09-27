//! SIMD-optimized image conversion and scaling for `vacc`.
//!
//! The crate provides portable, runtime-dispatched (`AVX2` / `SSE4.1` /
//! scalar) Y'CbCr(4:2:0) -> RGB routines used as the reference and software
//! fallback by every backend:
//!
//! - [`yuv_to_rgb`]: planar (I420) and semi-planar (NV12) 8-bit sources, and
//!   packed 10/12-bit (P010/P012) sources, into packed RGB24 or RGBA32.
//! - [`yuv_high_to_i420`]: exact top-justified `u16` -> `u8` down-cast into a
//!   tight 8-bit I420 buffer.
//! - [`process`]: the full [`ImageConfig`] pipeline (downcast -> scale -> rgb).
//!
//! Backends with a fast GPU primitive (NVIDIA: `cuda-npp`) may implement the
//! same operations out-of-band; this crate is the fallback and the ground
//! truth used by the verification tooling.

mod coeff;
pub mod conv;
mod error;
mod pixel;
mod pipeline;
mod resize;
mod spec;
pub mod warp;

pub use crate::coeff::{Conv8, RND, table};
pub use crate::error::{ImageError, ImageResult};
pub use crate::pixel::{YuvImage, YuvLayout, RgbImage, RgbImageMut, bps_bytes};
pub use crate::pipeline::{process, ProcessedFrame, RgbOutput, YuvOutput};
pub use crate::resize::{resize_rgb, resize_yuv};
pub use crate::warp::Affine;
pub use crate::spec::{ColorRange, ColorSpec, Filter, ImageConfig, MatrixCoefficients, RgbChannels, Scale};
pub use crate::conv::{
    conv_px, convert_rows, i420_size, parallel_rows, scratch_view, simd_features, yuv_high_to_i420,
    yuv_to_rgb, Kernel,
};
