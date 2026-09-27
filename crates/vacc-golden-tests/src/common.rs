//! Common behavioral tests, generic over any `Decoder` backend.
//!
//! These exercise the streaming contract that every backend must uphold:
//! whole-file decode produces a display-order stream with sane POCs, and
//! feeding one access unit per `submit()` (RTSP-style incremental feeding)
//! produces exactly the same stream as whole-file decode.

use std::error::Error;

use vacc_core::decoder::Decoder;
use vacc_core::frame::DecodedFrame;

use crate::bitstream::{is_hevc_stream, is_slice_ty, split};
use crate::samples::load_sample;
use crate::stream::drain;

/// Decode `sample` whole-file with `D`; returns the display frames.
/// No-ops (returns an empty vec) when the sample is not installed.
pub fn whole_file<D: Decoder>(sample: &str) -> Vec<DecodedFrame>
where
    D::Error: Error + 'static,
{
    let Some(data) = load_sample(sample) else {
        eprintln!("{sample}: sample not installed, skipping");
        return Vec::new();
    };
    let mut d = match D::new(data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{sample}: backend unavailable ({e}), skipping");
            return Vec::new();
        }
    };
    let frames = drain(&mut d);
    // Display order: POC strictly increases within a GOP and restarts at
    // each IDR.
    let mut last_poc = i32::MIN;
    for f in &frames {
        if f.poc <= last_poc {
            last_poc = i32::MIN;
        }
        assert!(
            f.poc > last_poc,
            "{sample}: POC went backwards within a GOP: {} -> {}",
            last_poc,
            f.poc
        );
        last_poc = f.poc;
    }
    eprintln!("{sample}: whole-file frames: {}", frames.len());
    frames
}

/// Regression: the decoder must decode the stream the same when fed one
/// access unit per `submit()` as when the whole file is up front. Catches
/// parse-state not being reset after full consumption in `submit()`, and
/// stale NAL caches being reused for two same-length chunks (parsers key
/// their cache on length).
///
/// Streaming contract: a picture may be held back until its display-order
/// successor has decoded, so an individual submit can emit zero frames. What
/// must never happen is an out-of-order emission: at every point the frames
/// emitted so far are an exact prefix of the final display-order sequence.
/// Held-back frames are released by later submits or by flush() at end of
/// stream.
///
/// No-ops when the sample is not installed or the backend is unavailable.
pub fn incremental_per_access_unit<D: Decoder>(sample: &str)
where
    D::Error: Error + 'static,
{
    let Some(data) = load_sample(sample) else {
        eprintln!("{sample}: sample not installed, skipping");
        return;
    };
    let hevc = is_hevc_stream(&data);
    let units = split(&data);

    let mut whole = match D::new(data.clone()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{sample}: backend unavailable ({e}), skipping");
            return;
        }
    };
    let whole_frames = drain(&mut whole);
    let whole_n = whole_frames.len();
    assert!(whole_n > 0, "{sample}: no frames decoded");

    // Find bootstrap prefix: all preamble units up to the first slice NAL.
    let sidx = units
        .iter()
        .position(|(_, t, _)| is_slice_ty(*t, hevc))
        .expect("no slice unit found in {sample}?!");
    let mut bootstrap = Vec::new();
    for (_, _, b) in units.iter().take(sidx) {
        bootstrap.extend_from_slice(b);
    }
    let mut d = D::new(bootstrap).expect("backend available (whole-file decode worked)");
    let mut emitted: Vec<DecodedFrame> = Vec::new();
    for (idx, (_, t, chunk)) in units.iter().enumerate().skip(sidx) {
        if !is_slice_ty(*t, hevc) {
            continue; // parameter-set units produce no frames
        }
        d.submit(chunk).expect("submit failed");
        while let Some(f) = d.decode().expect("decode failed") {
            // Mid-stream emissions must be an ordered prefix of the final
            // display order: the next frame is exactly the next whole-file
            // frame, pixel for pixel.
            assert_eq!(
                f.pixel_data.as_ref().map(|p| &p.buffer),
                whole_frames[emitted.len()].pixel_data.as_ref().map(|p| &p.buffer),
                "{sample}: out-of-order emission at unit {idx} (frame {})",
                emitted.len()
            );
            emitted.push(f);
        }
    }
    for f in d.flush().expect("flush failed") {
        assert_eq!(
            f.pixel_data.as_ref().map(|p| &p.buffer),
            whole_frames[emitted.len()].pixel_data.as_ref().map(|p| &p.buffer),
            "{sample}: out-of-order frame in flush (frame {})",
            emitted.len()
        );
        emitted.push(f);
    }
    eprintln!(
        "{sample}: incremental frames={} whole={whole_n} units={}",
        emitted.len(),
        units.len()
    );
    assert_eq!(
        emitted.len(),
        whole_n,
        "{sample}: incremental decode produced {} frames, whole-file produced {whole_n}",
        emitted.len()
    );
}

/// Run the full common suite for one (backend, sample): whole-file decode
/// plus incremental per-access-unit feeding. Use from a backend crate's
/// integration tests, one call per applicable sample.
pub fn common_stream_tests<D: Decoder>(sample: &str)
where
    D::Error: Error + 'static,
{
    let frames = whole_file::<D>(sample);
    if frames.is_empty() {
        return; // sample missing or backend unavailable — already reported
    }
    incremental_per_access_unit::<D>(sample);
}
