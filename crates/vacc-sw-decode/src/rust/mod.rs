//! Pure-Rust reimplementation of the edge264 slice-decode core.
//!
//! Production data plane for `vacc-sw-decode`. Ported bit-exactly from the
//! edge264 C routines (each module's header names its C source); every output
//! is pinned to golden hashes in `golden_data.rs` (`goldens.rs` provides the
//! collector, regeneration via `SW264_REGEN_GOLDENS=1`).

pub mod bits;
pub mod cabac;
pub mod deblock;
pub mod inter;
pub mod intra;
pub mod mvpred;
pub mod residual;
pub mod slice;
#[rustfmt::skip]
mod tables;

#[cfg(test)]
pub mod goldens;
#[cfg(test)]
mod tests;
