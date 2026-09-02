//! # vaapi-decode
//!
//! VAAPI Video Decode implementation using cros-libva.
//! Implements `vacc_core::Decoder` and `vacc_core::DecoderDevice` traits.

pub mod decoder;
pub mod device;
mod vp9_qlookup;

pub use decoder::VaapiDecoder;
pub use device::VaapiDecoderDevice;

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
