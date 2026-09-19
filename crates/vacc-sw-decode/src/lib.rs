//! `vacc-sw-decode` — software H.264/AVC decoder.
//!
//! Control plane uses the common Rust H.264 implementations from
//! [`vacc_parser`] (NAL/SPS/PPS/slice parsing, DPB, POC, ref lists); the data
//! plane is the pure-Rust slice-decode core in [`rust`], ported bit-exactly
//! from edge264's C routines and pinned to golden hashes.

pub mod decoder;
pub mod rust;

pub use decoder::SwH264Decoder;
