//! Software (CPU) video decode backends for vacc.
//!
//! H.265/HEVC reconstruction runs in the hevc.js core (MIT-licensed, see
//! `hevc/LICENSE`); bitstream parsing, POC computation and DPB management run
//! in Rust on the shared `vacc-parser`.

mod error;
// The C++ hevc.js core is retained only as the differential-test oracle
// (tier_e / tier_f); production reconstruction runs in the Rust `hevc` port.
#[cfg(test)]
mod ffi;
#[cfg(test)]
mod ffi_test;
pub mod h265;
pub mod hevc;

pub use error::{Error, Result};
pub use h265::SoftwareH265Decoder;
