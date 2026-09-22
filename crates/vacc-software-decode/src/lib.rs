//! `vacc-software-decode` — software (CPU) video decode backends for vacc.
//!
//! Both decoders use the common Rust bitstream parsers from [`vacc_parser`]
//! as their control plane (NAL/SPS/PPS/slice parsing, DPB, POC, ref lists);
//! the data planes are pure-Rust ports pinned to golden hashes:
//!
//! - [`SwH264Decoder`] — H.264/AVC, ported bit-exactly from edge264's C
//!   routines ([`rust`] module).
//! - [`SoftwareH265Decoder`] — H.265/HEVC, a pure-Rust port of the hevc.js
//!   core (MIT-licensed, see `HEVC_LICENSE`), with the decode kernels in the
//!   [`hevc`] module.

pub mod decoder;
pub mod error;
pub mod h265;
pub mod hevc;
pub mod rust;

pub use decoder::SwH264Decoder;
pub use error::{Error, Result};
pub use h265::SoftwareH265Decoder;
