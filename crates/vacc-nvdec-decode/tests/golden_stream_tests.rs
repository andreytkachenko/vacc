//! Golden stream pinning for the NVDEC (cuvid) backend.
//!
//! Uses the backend-generic framework in `vacc-golden-tests`: each sample's
//! full decode pins, per display frame, the SHA-256 of the canonical pixels
//! (see `vacc_golden_tests::canonical`), plus the POC sequence and the first
//! frame's display size, to `tests/data/nvdec_golden_data.rs`. The common
//! streaming-contract tests (whole-file vs incremental per-access-unit
//! feeding) run against each decoder as well.
//!
//! Tests skip (rather than fail) when a sample is not installed or the
//! backend is unavailable on the current machine.
//!
//! Regeneration (only after an intentional, re-verified behavior change):
//! ```sh
//! NVDEC_REGEN_GOLDENS=1 cargo test -p vacc-nvdec-decode --test golden_stream_tests regenerate_goldens
//! ```

include!("data/nvdec_golden_data.rs");

use std::error::Error;

use vacc_core::decoder::Decoder;
use vacc_golden_tests::goldens;
use vacc_golden_tests::samples::load_sample;
use vacc_golden_tests::stream::golden_stream;
use vacc_nvdec_decode::{NvdecAv1Decoder, NvdecH264Decoder, NvdecH265Decoder, NvdecVp9Decoder};

/// Pin one sample's full decode on decoder `D` against the goldens. Returns
/// `None` when skipped (sample missing or backend unavailable); hard-fails
/// in that case during a regeneration run.
fn pin<D: Decoder>(sample: &str) -> Option<usize>
where
    D::Error: Error + 'static,
{
    let Some(data) = load_sample(sample) else {
        if goldens::collecting() {
            panic!("{sample}: sample not installed during regeneration");
        }
        eprintln!("{sample}: sample not installed, skipping");
        return None;
    };
    let d = match D::new(data) {
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
    pin::<NvdecH264Decoder>("h264_main.h264");
    pin::<NvdecH265Decoder>("h265_main.h265");
    pin::<NvdecAv1Decoder>("av1_main.ivf");
    pin::<NvdecVp9Decoder>("vp9_profile0.ivf");
}

#[test]
fn h264_main_golden() {
    assert!(pin::<NvdecH264Decoder>("h264_main.h264").is_some());
}

#[test]
fn h265_main_golden() {
    assert!(pin::<NvdecH265Decoder>("h265_main.h265").is_some());
}

#[test]
fn av1_main_golden() {
    assert!(pin::<NvdecAv1Decoder>("av1_main.ivf").is_some());
}

#[test]
fn vp9_profile0_golden() {
    assert!(pin::<NvdecVp9Decoder>("vp9_profile0.ivf").is_some());
}

#[test]
fn h264_common_stream_tests() {
    vacc_golden_tests::common_stream_tests::<NvdecH264Decoder>("h264_main.h264");
}

#[test]
fn h265_common_stream_tests() {
    vacc_golden_tests::common_stream_tests::<NvdecH265Decoder>("h265_main.h265");
}

#[test]
fn regenerate_goldens() {
    if std::env::var_os("NVDEC_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set NVDEC_REGEN_GOLDENS=1 to rewrite tests/data/nvdec_golden_data.rs");
        return;
    }
    let entries = goldens::collect(run_all);
    assert!(!entries.is_empty(), "no samples produced golden entries");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/nvdec_golden_data.rs");
    let n = goldens::write_golden_file(
        entries,
        &path,
        "NVDEC_REGEN_GOLDENS=1 cargo test -p vacc-nvdec-decode --test golden_stream_tests regenerate_goldens",
    );
    eprintln!("wrote {n} golden entries to {}", path.display());
}
