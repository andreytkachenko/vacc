//! Error types for the Vulkan Video decoder.

use thiserror::Error;

/// Result type for Vulkan Video operations.
pub type VideoResult<T> = std::result::Result<T, VideoError>;

/// Error types for Vulkan Video operations.
#[derive(Debug, Error)]
pub enum VideoError {
    /// Vulkan driver does not support video decode.
    #[error("Video decode not supported by driver: {0}")]
    VideoNotSupported(String),

    /// Vulkan feature not available.
    #[error("Vulkan feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Invalid codec or profile.
    #[error("Invalid codec or profile: {0}")]
    InvalidCodec(String),

    /// Bitstream parsing error.
    #[error("Bitstream parse error: {0}")]
    BitstreamParse(String),

    /// Out of host memory.
    #[error("Out of host memory")]
    OutOfHostMemory,

    /// Out of device memory.
    #[error("Out of device memory")]
    OutOfDeviceMemory,

    /// Initialization failed.
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),

    /// Device lost.
    #[error("Device lost")]
    DeviceLost,

    /// Format not supported.
    #[error("Format not supported: {0}")]
    FormatNotSupported(String),

    /// Generic Vulkan error.
    #[error("Vulkan error: {0}")]
    VulkanError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid input data.
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    /// Frame not ready.
    #[error("Frame not ready")]
    FrameNotReady,

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Codec-specific error.
    #[error("Codec error: {0}")]
    CodecError(String),
}

impl VideoError {
    /// Check if this is a recoverable error.
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::FrameNotReady | Self::BitstreamParse(_) | Self::CodecError(_)
        )
    }
}

/// Video queue result types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoQueueResult {
    /// Successfully got a frame.
    GotFrame,
    /// No frame available yet.
    NoFrame,
    /// End of stream.
    EndOfStream,
    /// Error occurred.
    Error,
}

impl VideoQueueResult {
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::GotFrame)
    }

    pub const fn is_eof(&self) -> bool {
        matches!(self, Self::EndOfStream)
    }
}
