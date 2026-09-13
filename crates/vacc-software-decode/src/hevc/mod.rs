//! Pure-Rust port of the hevc.js decoder kernels (MIT, see `HEVC_LICENSE`).
//!
//! Sprint 1: shared types, CABAC tables, bitstream reader, transform +
//! dequantization. Sprint 2: picture buffer, interpolation (motion
//! compensation) and intra prediction kernels. Sprint 3: CABAC engine +
//! syntax elements. Sprint 4: loop filters (SAO §8.7.3, deblocking §8.7.2).
//! The full pipeline is wired into the `h265` decoder; outputs are pinned by
//! the SHA-256 goldens in `goldens` (originally verified byte-exact against
//! the C++ hevc.js oracle, since removed).

pub mod bitreader;
pub mod cabac;
pub mod cabac_tables;
pub mod coding_tree;
pub mod deblocking;
pub mod driver;
#[cfg(test)]
pub mod goldens;
pub mod interpolation;
pub mod inter_prediction;
pub mod intra_prediction;
pub mod picture;
pub mod residual_coding;
pub mod sao;
pub mod syntax_elements;
pub mod syntax_map;
#[cfg(test)]
pub mod tier_e;
#[cfg(test)]
pub mod tier_f;
pub mod transform;
pub mod types;
