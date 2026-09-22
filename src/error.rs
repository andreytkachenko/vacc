//! Error types for the unified decoder.

use thiserror::Error;

use crate::backend::Backend;

/// A single backend's failure during [`VaccDecoder`](crate::VaccDecoder)
/// initialization.
#[derive(Debug, Clone)]
pub struct BackendFailure {
    pub backend: Backend,
    pub message: String,
}

impl std::fmt::Display for BackendFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.backend, self.message)
    }
}

fn format_failures(failures: &[BackendFailure]) -> String {
    failures
        .iter()
        .map(|f| format!("  - {}", f))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Errors from the unified decoder.
#[derive(Debug, Error)]
pub enum UnifiedError {
    /// Every backend in the configured order failed to initialize the stream.
    #[error("all configured backends failed to initialize the stream:\n{details}", details = format_failures(&failures))]
    AllBackendsFailed { failures: Vec<BackendFailure> },

    /// The configured backend list is empty.
    #[error("decoder config contains no backends")]
    EmptyBackendOrder,

    /// The input codec could not be detected (needed by backends that have
    /// per-codec decoder types).
    #[error("could not detect the video codec of the input data")]
    CodecNotDetected,

    /// A backend does not support the requested operation or codec.
    #[error("{message}")]
    Unsupported { message: String },

    /// A runtime error from the active backend (decode, flush, ...).
    #[error("decoder backend error: {source}")]
    Backend { source: Box<dyn std::error::Error + Send + Sync> },
}

impl UnifiedError {
    /// The per-backend failures, if this is an [`AllBackendsFailed`](Self::AllBackendsFailed).
    pub fn failures(&self) -> Option<&[BackendFailure]> {
        match self {
            Self::AllBackendsFailed { failures } => Some(failures),
            _ => None,
        }
    }
}

pub type UnifiedResult<T> = std::result::Result<T, UnifiedError>;
