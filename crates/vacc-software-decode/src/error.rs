//! Error type for the software decode backend.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("decoder init failed: {0}")]
    DecoderInit(String),

    #[error("parser error: {0}")]
    Parser(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("C++ core error {code}: {msg}")]
    Core { code: i32, msg: String },

    #[error("decode failed: {0}")]
    DecodeFailed(String),
}

pub type Result<T> = std::result::Result<T, Error>;
