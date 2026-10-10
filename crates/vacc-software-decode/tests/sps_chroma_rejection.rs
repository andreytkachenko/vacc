//! The software HEVC decoder's reconstruction pipeline is monochrome / 4:2:0
//! only. An SPS with wider chroma (4:2:2, 4:4:4) or separate colour planes
//! must be rejected with a clear `Unsupported` error instead of being
//! silently misdecoded as 4:2:0 (the pre-fix behaviour).

use std::env;
use std::path::PathBuf;

use vacc_core::decoder::Decoder;
use vacc_software_decode::{Error, SoftwareH265Decoder};

fn sample_dir() -> Option<PathBuf> {
    if let Ok(d) = env::var("VACC_SAMPLES_DIR") {
        return Some(PathBuf::from(d));
    }
    let p = PathBuf::from("/home/atkachenko/apps/vacc/assets/samples");
    p.is_dir().then_some(p)
}

fn sample(name: &str) -> Vec<u8> {
    let dir = sample_dir().unwrap_or_else(|| panic!("sample dir not found ({name})"));
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

/// `new()` only detects the bitstream (VPS/SPS scan), so construction
/// succeeds; the SPS chroma check fires on the first `decode()`.
fn assert_rejected(name: &str, expect: &str) {
    let mut dec = SoftwareH265Decoder::new(sample(name))
        .unwrap_or_else(|e| panic!("{name}: construction should succeed (got {e})"));
    let err = dec
        .decode()
        .expect_err(&format!("{name}: wide-chroma SPS must be rejected, not decoded"));
    match &err {
        Error::Unsupported(msg) => assert!(
            msg.contains(expect),
            "{name}: expected rejection mentioning {expect:?}, got {msg:?}"
        ),
        other => panic!("{name}: expected Unsupported error, got {other:?}"),
    }
}

#[test]
fn sw_hevc_rejects_422_sps() {
    assert_rejected("h265_rext_422.h265", "4:2:2");
}

#[test]
fn sw_hevc_rejects_444_sps() {
    assert_rejected("h265_rext_444.h265", "4:4:4");
}
