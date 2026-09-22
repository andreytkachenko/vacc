//! Code shared by the H.264 (`rust/`) and H.265 (`hevc/`) software decoders
//! in `vacc-software-decode`.
//!
//! Scope (established by the DRY audit of the two decoders):
//!
//! - [`goldens`] — the golden-hash test infrastructure (SHA-256, keyed
//!   assertions, record/collect, `golden_data.rs` writer) previously
//!   duplicated line-by-line in `rust/goldens.rs` and `hevc/goldens.rs`.
//! - [`rng`] — the deterministic test PRNGs (splitmix64, xorshift64*,
//!   Numerical Recipes LCG) previously copy-pasted into up to ten test
//!   modules.
//! - [`clip`] — the bit-depth-agnostic saturation helpers.
//!
//! Deliberately NOT shared: the entropy layer (`rust/bits` + `rust/cabac`
//! vs `hevc/bitreader` + `hevc/cabac`) and the reconstruction kernels
//! (interpolation, intra, residual/transform, mvpred, deblocking). Those are
//! ports of *different* reference implementations (edge264's C algorithms vs
//! the hevc.js C++/spec algorithms) with structurally distinct arithmetic
//! (128-bit CAVLC/CABAC cache vs 64-bit cached reader; folded C context
//! tables vs spec P/MPS-LPS tables; quarter-pel u8 6-tap vs 1/16-pel u16
//! 8-tap filters) and are pinned bit-exactly by golden hashes; unifying them
//! would risk drift that cannot be exercised against a reference. See the
//! module docs of each data-plane module for its port source.

pub mod clip;
pub mod goldens;
pub mod rng;
