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
#[test]
fn incremental_per_access_unit() {
    let data = fs::read(SAMPLE).unwrap();
    let units = split(&data);

    let mut whole = SwH264Decoder::new(data.clone()).unwrap();
    let whole_frames = drain(&mut whole);
    // Whole-file decode emits pictures in display order: POC is strictly
    // increasing within a GOP and restarts at each IDR.
    let mut last_poc = i32::MIN;
    let mut restarts = 0;
    for f in &whole_frames {
        if f.poc <= last_poc {
            restarts += 1;
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
    let mut frames = 0usize;
    let mut missing_frames_at: Vec<(usize, u8)> = Vec::new();
    for (idx, (_, t, chunk)) in units.iter().enumerate().skip(sidx) {
        if !is_slice_ty(*t) {
            continue; // parameter-set units produce no frames
        }
        let before = frames;
        d.submit(chunk).unwrap();
        while d.decode().unwrap().is_some() {
            frames += 1;
        }
        if frames == before {
            missing_frames_at.push((idx, *t));
        }
    }
    frames += d.flush().unwrap().len();
    println!(
        "incremental: frames={frames} whole={whole_n} units={} missing_at={missing_frames_at:?}",
        units.len()
    );
    assert!(
        missing_frames_at.is_empty(),
        "access units that produced no frame: {missing_frames_at:?}",
    );
    assert_eq!(
        frames, whole_n,
        "incremental decode produced {frames} frames, whole-file produced {whole_n}"
    );
}
