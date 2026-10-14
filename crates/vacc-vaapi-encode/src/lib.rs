//! # vacc-vaapi-encode
//!
//! VAAPI H.264 encode implementation using cros-libva.
//! Implements `vacc_core::Encoder` for `VideoCodec::EncodeH264`.
//!
//! Uses the driver's internal motion estimation (no per-macroblock buffer):
//! the app supplies packed SPS/PPS/slice headers, picture and slice
//! parameters, and NV12 input surfaces.

pub mod device;
pub mod encoder;
pub mod h264;
pub mod h265;

pub use device::VaapiEncoderDevice;
pub use encoder::VaapiEncoder;

/// Result type for VAAPI encode operations.
pub type Result<T> = std::result::Result<T, Error>;

/// VAAPI encode error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("VA-API error: {0}")]
    VaApi(String),

    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    #[error("Encoder initialization failed: {0}")]
    EncoderInit(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Codec error: {0}")]
    CodecError(String),
}

impl From<libva::VaError> for Error {
    fn from(e: libva::VaError) -> Self {
        Error::VaApi(e.to_string())
    }
}
