//! Error types for NVDEC decoder operations.
//!
//! This module defines [`NvdecError`], which covers all error conditions
//! that can occur during NVDEC initialization, decoding, and cleanup.
//! Use [`NvdecResult<T>`] as the return type for all public API functions.

use thiserror::Error;

/// NVDEC decoder error types.
///
/// Covers errors from library loading, CUDA initialization, decoder creation,
/// frame decoding, and memory operations.
///
/// # Common Error Scenarios
///
/// | Error | Cause |
/// |-------|-------|
/// | `LibLoadError` | Missing `libcuda.so` or `libnvcuvid.so` |
/// | `CudaError` | CUDA driver API call failed |
/// | `DecoderCreationFailed` | No SPS/PPS in input, or GPU doesn't support codec |
/// | `DecodeFailed` | Bitstream parse error or HW decode failure |
/// | `UnsupportedCodec` | Non-H.264 codec passed to decoder |
/// | `InvalidState` | Method called before initialization or after reset |
#[derive(Debug, Error)]
pub enum NvdecError {
    /// CUDA driver API returned an error.
    ///
    /// Contains the error message with the CUDA error code.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Failed to load a required shared library.
    ///
    /// Typically indicates missing `libcuda.so` (CUDA Driver API) or
    /// `libnvcuvid.so` (Video Codec SDK runtime).
    #[error("Library load error: {0}")]
    LibLoadError(String),

    /// Decoder or parser creation failed.
    ///
    /// Common causes: no SPS/PPS in input data, unsupported codec/profile,
    /// or insufficient GPU resources.
    #[error("Decoder creation failed: {0}")]
    DecoderCreationFailed(String),

    /// Frame decoding failed.
    ///
    /// Can occur due to corrupted bitstream data, unsupported features,
    /// or hardware decode errors.
    #[error("Decode failed: {0}")]
    DecodeFailed(String),

    /// Failed to map a decoded video frame from GPU memory.
    #[error("Map video frame failed: {0}")]
    MapVideoFrameFailed(String),

    /// Failed to unmap a previously mapped video frame.
    #[error("Unmap video frame failed: {0}")]
    UnmapVideoFrameFailed(String),

    /// GPU device error (e.g., device not found, driver issue).
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Codec not supported by this decoder.
    ///
    /// Only H.264 is currently supported.
    #[error("Unsupported codec: {0:?}")]
    UnsupportedCodec(vk_video_core::VideoCodec),

    /// Pixel format not supported.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Error from the bitstream parser.
    #[error("Parser error: {0}")]
    ParserError(#[from] vk_video_parser::ParserError),

    /// I/O error (e.g., file read failure).
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid decoder state for the requested operation.
    ///
    /// E.g., calling `decode()` before initialization or after `reset()`.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// No frames currently available for output.
    ///
    /// Submit more data or wait for the next decode cycle.
    #[error("No frames available")]
    NoFramesAvailable,

    /// End of stream reached.
    #[error("End of stream")]
    EndOfStream,
}

/// Result type for NVDEC operations.
///
/// Alias for `Result<T, NvdecError>`. Use as the return type for all
/// NVDEC-related functions.
pub type NvdecResult<T> = std::result::Result<T, NvdecError>;
