//! Pure-Rust ports of hevc.js decoder kernels (incremental rewrite).
//!
//! Sprint 1: shared types, CABAC tables, bitstream reader, transform +
//! dequantization. Sprint 2: picture buffer, interpolation (motion
//! compensation) and intra prediction kernels. These modules are NOT yet
//! wired into the decode pipeline — the C++ core (`hevc/`) remains the
//! reference implementation and serves as the differential-test oracle via
//! the `hevcdec_test_*` exports.

pub mod bitreader;
pub mod cabac;
pub mod cabac_tables;
pub mod interpolation;
pub mod intra_prediction;
pub mod picture;
pub mod syntax_elements;
pub mod transform;
pub mod types;
