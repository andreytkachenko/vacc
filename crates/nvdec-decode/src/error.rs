//! Error types for NVDEC decoder.

use thiserror::Error;

/// NVDEC decoder error types.
#[derive(Debug, Error)]
pub enum NvdecError {
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Library load error: {0}")]
    LibLoadError(String),

    #[error("Decoder creation failed: {0}")]
    DecoderCreationFailed(String),

    #[error("Decode failed: {0}")]
    DecodeFailed(String),

    #[error("Map video frame failed: {0}")]
    MapVideoFrameFailed(String),

    #[error("Unmap video frame failed: {0}")]
    UnmapVideoFrameFailed(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Unsupported codec: {0:?}")]
    UnsupportedCodec(vk_video_core::VideoCodec),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Parser error: {0}")]
    ParserError(#[from] vk_video_parser::ParserError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("No frames available")]
    NoFramesAvailable,

    #[error("End of stream")]
    EndOfStream,
}

/// Result type for NVDEC operations.
pub type NvdecResult<T> = std::result::Result<T, NvdecError>;
