//! Sample-stream resolution shared by all backend test suites.
//!
//! Samples live in `assets/samples` at the repository root; override the
//! location with `VACC_SAMPLES_DIR`. Tests skip (rather than fail) when a
//! sample is not installed.

use std::path::{Path, PathBuf};

/// Locate the sample directory: `$VACC_SAMPLES_DIR`, else `assets/samples`
/// relative to this crate's repository root. Returns `None` if absent.
pub fn sample_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("VACC_SAMPLES_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    // crates/vacc-golden-tests -> repo root
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/samples");
    p.is_dir().then_some(p)
}

/// Load sample `name` (e.g. `"h264_main.h264"`). `None` if the samples are
/// not installed or the file is missing — callers skip in that case.
pub fn load_sample(name: &str) -> Option<Vec<u8>> {
    let dir = sample_dir()?;
    std::fs::read(dir.join(name)).ok()
}
