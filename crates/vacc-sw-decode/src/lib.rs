//! `vacc-sw-decode` — software H.264/AVC decoder.
//!
//! Control plane uses the common Rust H.264 implementations from
//! [`vacc_parser`] (NAL/SPS/PPS/slice parsing, DPB, POC, ref lists); the data
//! plane is a small statically-linked subset of [edge264]'s C slice-decode
//! routines, driven per-slice through FFI.

pub mod decoder;
pub mod ffi;

pub use decoder::SwH264Decoder;
