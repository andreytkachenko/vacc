//! Golden stream pinning for the Vulkan backend.
//!
//! Uses the backend-generic framework in `vacc-golden-tests`. The Vulkan
//! decoder's `Decoder`-trait path is not implemented (it decodes through
//! `decode_all`/`decode_stream`), so this suite drives
//! [`FrameSink`] directly with canonical pixels computed from
//! [`vacc_vulkan::DecodedFrame`]: per display frame the SHA-256 of the
//! canonical planar Y+U+V bytes (cropped to the display rect, bottom-justified
//! 16-bit samples — see `vacc_golden_tests::canonical`), plus the POC
//! sequence and the first frame's display size, pinned to
//! `tests/data/vulkan_golden_data.rs`.
//!
//! Tests skip (rather than fail) when a sample is not installed or the
//! backend is unavailable on the current machine.
//!
//! Regeneration (only after an intentional, re-verified behavior change):
//! ```sh
//! VULKAN_REGEN_GOLDENS=1 cargo test -p vacc-vulkan-decode --test golden_stream_tests regenerate_goldens
//! ```

include!("data/vulkan_golden_data.rs");

use vacc_golden_tests::goldens;
use vacc_golden_tests::samples::load_sample;
use vacc_golden_tests::stream::FrameSink;
use vacc_vulkan_decode::VulkanDecoder;

/// Canonical planar Y+U+V bytes for one 4:2:0 Vulkan frame, cropped to the
/// display rect. Readback already stores bottom-justified samples, so the
/// planes are copied verbatim (packed rows).
fn canonical(f: &vacc_vulkan::DecodedFrame) -> Vec<u8> {
    let px = &f.pixels;
    let ss = px.sample_size as usize;
    assert_eq!(
        px.chroma_width as usize,
        (f.coded_width as usize).div_ceil(2),
        "expected 4:2:0 readback"
    );
    let luma_pitch = f.coded_width as usize * ss;
    let chroma_pitch = px.chroma_width as usize * ss;
    let cw = f.display_width as usize;
    let ch = f.display_height as usize;
    let uchw = cw.div_ceil(2);
    let uchh = ch.div_ceil(2);
    let ul = (f.crop_left as usize).div_ceil(2);
    let ut = (f.crop_top as usize).div_ceil(2);

    let mut out = Vec::with_capacity(px.y_plane.len() + px.u_plane.len() + px.v_plane.len());
    for r in 0..ch {
        let off = (f.crop_top as usize + r) * luma_pitch + f.crop_left as usize * ss;
        out.extend_from_slice(&px.y_plane[off..off + cw * ss]);
    }
    for r in 0..uchh {
        let off = (ut + r) * chroma_pitch + ul * ss;
        out.extend_from_slice(&px.u_plane[off..off + uchw * ss]);
    }
    for r in 0..uchh {
        let off = (ut + r) * chroma_pitch + ul * ss;
        out.extend_from_slice(&px.v_plane[off..off + uchw * ss]);
    }
    out
}

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
    let mut d = match VulkanDecoder::new(data) {
        Ok(d) => d,
        Err(e) if goldens::collecting() => {
            panic!("{sample}: backend unavailable during regeneration: {e}")
        }
        Err(e) => {
            eprintln!("{sample}: backend unavailable ({e}), skipping");
            return None;
        }
    };
    let frames = match d.decode_all(1_000_000) {
        Ok(f) => f,
        Err(e) if goldens::collecting() => {
            panic!("{sample}: decode failed during regeneration: {e}")
        }
        Err(e) => {
            eprintln!("{sample}: decode failed ({e}), skipping");
            return None;
        }
    };
    let mut sink = FrameSink::new(sample, GOLDENS);
    for f in &frames {
        sink.on_frame(f.poc, f.display_width, f.display_height, Some(&canonical(f)));
    }
    Some(sink.finish())
}

fn run_all() {
    pin("h264_main.h264");
}

#[test]
fn h264_main_golden() {
    assert!(pin("h264_main.h264").is_some());
}

#[test]
fn regenerate_goldens() {
    if std::env::var_os("VULKAN_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set VULKAN_REGEN_GOLDENS=1 to rewrite tests/data/vulkan_golden_data.rs");
        return;
    }
    let entries = goldens::collect(run_all);
    assert!(!entries.is_empty(), "no samples produced golden entries");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vulkan_golden_data.rs");
    let n = goldens::write_golden_file(
        entries,
        &path,
        "VULKAN_REGEN_GOLDENS=1 cargo test -p vacc-vulkan-decode --test golden_stream_tests regenerate_goldens",
    );
    eprintln!("wrote {n} golden entries to {}", path.display());
}
