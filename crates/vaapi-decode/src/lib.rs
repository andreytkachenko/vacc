//! # vaapi-decode
//!
//! VAAPI Video Decode implementation using cros-libva.
//! Implements `vk_video_core::Decoder` and `vk_video_core::DecoderDevice` traits.

pub mod device;
pub mod decoder;
mod vp9_qlookup;

pub use device::VaapiDecoderDevice;
pub use decoder::VaapiDecoder;

/// Result type for VAAPI operations.
pub type Result<T> = std::result::Result<T, Error>;

/// VAAPI error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("VA-API error: {0}")]
    VaApi(String),

    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    #[error("Decoder initialization failed: {0}")]
    DecoderInit(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parser error: {0}")]
    Parser(String),
}

impl From<libva::VaError> for Error {
    fn from(e: libva::VaError) -> Self {
        Error::VaApi(e.to_string())
    }
}
