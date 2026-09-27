//! Golden stream pinning for the VAAPI backend.
//!
//! Uses the backend-generic framework in `vacc-golden-tests`: each sample's
//! full decode pins, per display frame, the SHA-256 of the canonical pixels
//! (see `vacc_golden_tests::canonical`), plus the POC sequence and the first
//! frame's display size, to `tests/data/vaapi_golden_data.rs`. The common
//! streaming-contract tests (whole-file vs incremental per-access-unit
//! feeding) run against the decoder as well.
//!
//! Tests skip (rather than fail) when a sample is not installed or the
//! backend is unavailable on the current machine.
//!
//! Regeneration (only after an intentional, re-verified behavior change):
//! ```sh
//! VAAPI_REGEN_GOLDENS=1 cargo test -p vacc-vaapi-decode --test golden_stream_tests regenerate_goldens
//! ```

include!("data/vaapi_golden_data.rs");

use vacc_golden_tests::goldens;
use vacc_golden_tests::samples::load_sample;
use vacc_golden_tests::stream::golden_stream;
use vacc_vaapi_decode::VaapiDecoder;

/// Pin one sample's full decode against the goldens. Returns `None` when
/// skipped (sample missing or backend unavailable); hard-fails in that case
/// during a regeneration run.
fn pin(sample: &str) -> Option<usize> {
    let Some(data) = load_sample(sample) else {
        if goldens::collecting() {
            panic!("{sample}: sample not installed during regeneration");
        }
        eprintln!("{sample}: sample not installed, skipping");
        return None;
    };
    let d = match VaapiDecoder::new(data) {
        Ok(d) => d,
        Err(e) if goldens::collecting() => {
            panic!("{sample}: backend unavailable during regeneration: {e}")
        }
        Err(e) => {
            eprintln!("{sample}: backend unavailable ({e}), skipping");
            return None;
        }
    };
    Some(golden_stream(d, sample, GOLDENS))
}

fn run_all() {
    pin("h264_main.h264");
    pin("vp9_profile0.ivf");
    pin("av1_main.ivf");
}

#[test]
fn h264_main_golden() {
    assert!(pin("h264_main.h264").is_some());
}

#[test]
fn vp9_profile0_golden() {
    assert!(pin("vp9_profile0.ivf").is_some());
}

#[test]
fn av1_main_golden() {
    assert!(pin("av1_main.ivf").is_some());
}

#[test]
fn h264_common_stream_tests() {
    vacc_golden_tests::common_stream_tests::<VaapiDecoder>("h264_main.h264");
}

#[test]
fn regenerate_goldens() {
    if std::env::var_os("VAAPI_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set VAAPI_REGEN_GOLDENS=1 to rewrite tests/data/vaapi_golden_data.rs");
        return;
    }
    let entries = goldens::collect(run_all);
    assert!(!entries.is_empty(), "no samples produced golden entries");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vaapi_golden_data.rs");
    let n = goldens::write_golden_file(
        entries,
        &path,
        "VAAPI_REGEN_GOLDENS=1 cargo test -p vacc-vaapi-decode --test golden_stream_tests regenerate_goldens",
    );
    eprintln!("wrote {n} golden entries to {}", path.display());
}
