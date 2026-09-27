use std::fs;
use vacc_core::decoder::Decoder;
use vacc_software_decode::SwH264Decoder;

const SAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/samples/h264_main.h264");

fn drain(d: &mut impl Decoder) -> Vec<vacc_core::frame::DecodedFrame> {
    let mut frames = Vec::new();
    while let Some(f) = d.decode().expect("decode failed") {
        frames.push(f);
    }
    frames.extend(d.flush().expect("flush failed"));
    frames
}

fn is_slice_ty(t: u8) -> bool {
    t != 0 && !(6..=14).contains(&t)
}

/// split annex-b into (start_code_offset, nal_unit_type, bytes) units
fn split(data: &[u8]) -> Vec<(usize, u8, Vec<u8>)> {
    let mut heads: Vec<(usize, u8)> = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        let sc = if data[i..i + 4] == [0, 0, 0, 1] {
            4
        } else if data[i..i + 3] == [0, 0, 1] {
            3
        } else {
            i += 1;
            continue;
        };
        if i + sc < data.len() {
            heads.push((i, data[i + sc] & 0x1F));
        }
        i += sc;
    }
    heads
        .windows(2)
        .map(|w| (w[0].0, w[0].1, data[w[0].0..w[1].0].to_vec()))
        .chain(std::iter::once({
            let (o, t) = heads.last().unwrap();
            (*o, *t, data[*o..].to_vec())
        }))
        .collect()
}

#[test]
fn whole_file() {
    let data = fs::read(SAMPLE).unwrap();
    let mut d = SwH264Decoder::new(data).unwrap();
    let frames = drain(&mut d);
    println!("whole-file frames: {}", frames.len());
    assert!(frames.len() > 10);
}

/// Regression: the software H.264 decoder must decode the stream the same
/// when fed one access unit per `submit()` (RTSP-style incremental feeding)
/// as when the whole file is up front. Catches the `parse_offset` not being
/// reset after full consumption in `submit()`, and the stale NAL cache being
/// reused for two same-length chunks (parse() keys its cache on length).
///
/// Streaming contract: a picture may be held back until its display-order
/// successor has decoded, so an individual submit can emit zero frames. What
/// must never happen is an out-of-order emission: at every point the frames
/// emitted so far are an exact prefix of the final display-order sequence.
/// Held-back frames are released by later submits or by flush() at end of
/// stream.
#[test]
fn incremental_per_access_unit() {
    let data = fs::read(SAMPLE).unwrap();
    let units = split(&data);

    let mut whole = SwH264Decoder::new(data.clone()).unwrap();
    let whole_frames = drain(&mut whole);
    // Whole-file decode emits pictures in display order: POC is strictly
    // increasing within a GOP and restarts at each IDR.
    let mut last_poc = i32::MIN;
    for f in &whole_frames {
        if f.poc <= last_poc {
            last_poc = i32::MIN;
        }
        assert!(f.poc > last_poc, "POC went backwards within a GOP: {} -> {}", last_poc, f.poc);
        last_poc = f.poc;
    }
    let whole_n = whole_frames.len();
    println!("whole_file frames: {whole_n}");

    // Find bootstrap prefix: all preamble units up to the first slice NAL.
    let sidx = units
        .iter()
        .position(|(_, t, _)| is_slice_ty(*t))
        .expect("no unit found?!");
    let mut bootstrap = Vec::new();
    for (_, _, b) in units.iter().take(sidx) {
        bootstrap.extend_from_slice(b);
    }
    let mut d = SwH264Decoder::new(bootstrap).unwrap();
    let mut emitted: Vec<vacc_core::frame::DecodedFrame> = Vec::new();
    for (idx, (_, t, chunk)) in units.iter().enumerate().skip(sidx) {
        if !is_slice_ty(*t) {
            continue; // parameter-set units produce no frames
        }
        d.submit(chunk).unwrap();
        while let Some(f) = d.decode().unwrap() {
            // Mid-stream emissions must be an ordered prefix of the final
            // display order: the next frame is exactly the next whole-file
            // frame, pixel for pixel.
            assert_eq!(
                f.pixel_data.as_ref().unwrap().buffer,
                whole_frames[emitted.len()].pixel_data.as_ref().unwrap().buffer,
                "out-of-order emission at unit {idx} (frame {})",
                emitted.len()
            );
            emitted.push(f);
        }
    }
    for f in d.flush().unwrap() {
        assert_eq!(
            f.pixel_data.as_ref().unwrap().buffer,
            whole_frames[emitted.len()].pixel_data.as_ref().unwrap().buffer,
            "out-of-order frame in flush (frame {})",
            emitted.len()
        );
        emitted.push(f);
    }
    println!("incremental: frames={} whole={whole_n} units={}", emitted.len(), units.len());
    assert_eq!(
        emitted.len(),
        whole_n,
        "incremental decode produced {} frames, whole-file produced {whole_n}",
        emitted.len()
    );
}
