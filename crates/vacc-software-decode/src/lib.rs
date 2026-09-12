//! Software (CPU) video decode backends for vacc.
//!
//! H.265/HEVC reconstruction runs in the hevc.js core (MIT-licensed, see
//! `hevc/LICENSE`); bitstream parsing, POC computation and DPB management run
//! in Rust on the shared `vacc-parser`.

mod error;
mod ffi;
pub mod h265;

pub use error::{Error, Result};
pub use h265::SoftwareH265Decoder;
