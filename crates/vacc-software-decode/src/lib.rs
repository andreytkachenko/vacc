//! Software (CPU) video decode backends for vacc.
//!
//! H.265/HEVC is a pure-Rust port of the hevc.js core (MIT-licensed, see
//! `HEVC_LICENSE`): bitstream parsing, POC computation, DPB management and
//! pixel reconstruction all run on the shared `vacc-parser` / Rust kernels.

mod error;
pub mod h265;
pub mod hevc;

pub use error::{Error, Result};
pub use h265::SoftwareH265Decoder;
