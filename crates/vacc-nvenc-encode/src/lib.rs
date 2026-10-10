//! # vacc-nvenc-encode
//!
//! NVIDIA NVENC H.264 encode implementation using `libnvidia-encode.so`
//! (loaded at runtime via libloading, vtable populated by
//! `NvEncodeAPICreateInstance`). Implements `vacc_core::Encoder` for
//! `VideoCodec::EncodeH264`.
//!
//! The encoder runs in synchronous CQP mode with the driver's internal motion
//! estimation: the app supplies NV12 frames and receives Annex B access units.
//! SPS/PPS are generated and emitted by the driver (re-emitted per IDR, as
//! legal Annex B).

pub mod cuda;
pub mod encoder;
pub mod nvenc;
pub mod nvenc_api;

pub use encoder::NvencEncoder;

/// Result type for NVENC encode operations.
pub type Result<T> = std::result::Result<T, Error>;

/// NVENC encode error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("NVENC library not found: {0}")]
    LibraryNotFound(String),

    #[error("CUDA driver error: {0}")]
    Cuda(String),

    #[error("NVENC API error: {0}")]
    Api(String),

    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    #[error("Encoder initialization failed: {0}")]
    EncoderInit(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
