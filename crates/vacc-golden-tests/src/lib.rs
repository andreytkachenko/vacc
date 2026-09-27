//! `vacc-golden-tests` — backend-generic test framework for vacc decoders.
//!
//! Every decoder backend (software H.264/H.265, NVDEC, VAAPI, Vulkan) is
//! exercised against the same golden data through the uniform
//! [`vacc_core::decoder::Decoder`] trait:
//!
//! - [`golden_stream`] — decode a sample and pin every display frame's
//!   canonical pixels (plus the POC sequence and frame size) to SHA-256
//!   goldens. Canonical form matches `verify-all.py` / the `decode` example.
//! - [`whole_file`] / [`incremental_per_access_unit`] — common behavioral
//!   tests: whole-file decode, and one-access-unit-per-`submit()` feeding
//!   producing an identical display-order stream.
//! - [`goldens`] — the shared golden machinery (digest, keyed assertions,
//!   record/collect capture, table writer).
//!
//! Backend crates keep their own golden tables (`tests/data/*_golden_data.rs`)
//! and their backend-specific tests; this crate only provides the generic
//! test implementations and the machinery they pin against.
//!
//! # Usage (from a backend crate's integration tests)
//!
//! ```rust,ignore
//! include!("data/h264_golden_data.rs"); // pub(crate) const GOLDENS: ...
//!
//! #[test]
//! fn h264_main() {
//!     let data = match vacc_golden_tests::load_sample("h264_main.h264") {
//!         Some(d) => d,
//!         None => return, // samples not installed
//!     };
//!     let dec = MyH264Decoder::new(data).expect("backend available");
//!     vacc_golden_tests::golden_stream(dec, "h264_main.h264", GOLDENS);
//! }
//! ```

pub mod bitstream;
pub mod canonical;
pub mod common;
pub mod goldens;
pub mod samples;
pub mod stream;

pub use canonical::canonical_pixels;
pub use common::{common_stream_tests, incremental_per_access_unit, whole_file};
pub use samples::{load_sample, sample_dir};
pub use stream::{drain, golden_stream, FrameSink};
