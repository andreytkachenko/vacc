//! End-to-end HEVC encode test: 10 synthetic NV12 frames -> Annex B stream.
//!
//! Requires a VA display with HEVC Main encode support; skipped otherwise.

use std::io::Write;
use vacc_core::encoder::{EncodedFrame, Encoder, EncoderConfig, InputFrame};
use vacc_core::VideoCodec;
use vacc_vaapi_encode::VaapiEncoder;

const W: u32 = 320;
const H: u32 = 240;
const NFRAMES: u32 = 10;

/// Synthetic NV12 frame (same pattern as the H.264 probe).
fn make_frame(frame_idx: u32) -> (Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; (W * H) as usize];
    for py in 0..H {
        for px in 0..W {
            y[(py * W + px) as usize] = ((px * 7 + py * 3 + frame_idx * 16) % 256) as u8;
        }
    }
    let mut uv = vec![0u8; (W * H / 2) as usize];
    for row in 0..(H / 2) {
        for col in 0..(W / 2) {
            let o = (row * W + col * 2) as usize;
            uv[o] = (128 + frame_idx * 7) as u8;
            uv[o + 1] = (128 - frame_idx * 5) as u8;
        }
    }
    (y, uv)
}

/// All NAL unit types in an Annex B access unit (3- and 4-byte start codes).
fn nal_types(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i..i + 4] == [0, 0, 0, 1] {
            if i + 4 < data.len() {
                out.push((data[i + 4] >> 1) & 0x3F);
            }
            i += 4;
        } else if data[i..i + 3] == [0, 0, 1] && (i == 0 || data[i - 1] != 0) {
            if i + 3 < data.len() {
                out.push((data[i + 3] >> 1) & 0x3F);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn encode_h265_10_frames() {
    let config = EncoderConfig {
        codec: VideoCodec::EncodeH265,
        width: W,
        height: H,
        qp: 26,
        ..Default::default()
    };
    let mut enc = match VaapiEncoder::new(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("VA HEVC encode unavailable (skipping): {e}");
            return;
        }
    };

    let info = enc.info();
    assert_eq!(info.codec, VideoCodec::EncodeH265);
    assert_eq!(info.coded_size.width, W);
    assert_eq!(info.coded_size.height, H);

    // B-frame reordering: a mini-GOP's anchor must be encoded *before* the B
    // frames that reference it, so `encode()` returns AUs in decode order and
    // may emit several (a completed mini-GOP) or none (frame still buffered)
    // per call. Accumulate every AU across all calls plus flush().
    let mut all: Vec<EncodedFrame> = Vec::new();
    for i in 0..NFRAMES {
        let (y, uv) = make_frame(i);
        let frame = InputFrame::packed(W, H, &y, &uv);
        let aus = enc.encode(&frame, (i as i64) * 33_333_333).unwrap();
        for au in &aus {
            assert!(!au.data.is_empty(), "frame {} produced no data", au.frame_index);
            // Annex B: must start with a start code.
            let sc = au.data
                .windows(4)
                .position(|w| w == [0, 0, 0, 1])
                .unwrap_or_else(|| panic!("frame {}: no start code in output", au.frame_index));
            assert!(sc <= 4, "start code not at beginning (offset {sc})");
            let nals = nal_types(&au.data);
            assert!(!nals.is_empty(), "frame {}: no NAL units found", au.frame_index);
            // DTS contract: never ahead of the presentation timestamp.
            assert!(
                au.dts <= au.timestamp,
                "frame {}: dts {} > timestamp {}",
                au.frame_index,
                au.dts,
                au.timestamp
            );
        }
        all.extend(aus);
    }
    all.extend(enc.flush().unwrap());

    // Every input frame is emitted exactly once (decode order differs).
    assert_eq!(all.len() as u32, NFRAMES, "expected {NFRAMES} AUs, got {}", all.len());
    let mut idxs: Vec<u32> = all.iter().map(|f| f.frame_index).collect();
    idxs.sort_unstable();
    assert_eq!(idxs, (0..NFRAMES).collect::<Vec<_>>(), "each frame emitted exactly once");

    // First AU is the IDR and carries VPS(32)/SPS(33)/PPS(34) + IDR_W_RADL(19).
    let first = &all[0];
    assert_eq!(first.frame_index, 0);
    assert!(first.is_idr, "first AU must be IDR");
    let nals0 = nal_types(&first.data);
    for t in [32u8, 33, 34, 19] {
        assert!(nals0.contains(&t), "IDR AU missing NAL type {t}: {nals0:?}");
    }

    // Decode order for 10 frames with bf=2: IDR then mini-GOPs [anchor,B,B].
    let order: Vec<u32> = all.iter().map(|f| f.frame_index).collect();
    assert_eq!(
        order,
        vec![0, 3, 1, 2, 6, 4, 5, 9, 7, 8],
        "decode order mismatch: {order:?}"
    );

    // Write for external validation (ffmpeg / project oracle).
    let mut out = Vec::new();
    for f in &all {
        out.extend_from_slice(&f.data);
    }
    let path = std::env::temp_dir().join("vacc_vaenc_test.hevc");
    let mut fh = std::fs::File::create(&path).unwrap();
    fh.write_all(&out).unwrap();
    eprintln!("wrote {} bytes to {}", out.len(), path.display());
}
