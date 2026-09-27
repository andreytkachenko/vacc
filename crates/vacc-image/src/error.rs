//! Error types for image operations.

use thiserror::Error;

/// Result alias for image operations.
pub type ImageResult<T> = Result<T, ImageError>;

/// Errors from image conversion / scaling routines.
#[derive(Debug, Error)]
pub enum ImageError {
    /// The requested conversion combination is not supported.
    #[error("unsupported conversion: {0}")]
    Unsupported(String),

    /// Dimensions or buffer lengths do not match.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// The output buffer has the wrong length.
    #[error("output buffer too small: need {need} bytes, have {have}")]
    OutputTooSmall { need: usize, have: usize },

    /// A backend-specific pipeline failure (GPU init, allocation, submission).
    #[error("image pipeline error: {0}")]
    Pipeline(String),
}
