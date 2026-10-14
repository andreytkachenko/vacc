//! End-to-end HEVC encode test: 10 synthetic NV12 frames -> Annex B stream.
//!
//! Requires a VA display with HEVC Main encode support; skipped otherwise.

use std::io::Write;
use vacc_core::encoder::{Encoder, EncoderConfig, InputFrame};
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
        if &data[i..i + 4] == [0, 0, 0, 1] {
            if i + 4 < data.len() {
                out.push((data[i + 4] >> 1) & 0x3F);
            }
            i += 4;
        } else if &data[i..i + 3] == [0, 0, 1] && (i == 0 || data[i - 1] != 0) {
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

    let mut out = Vec::new();
    for i in 0..NFRAMES {
        let (y, uv) = make_frame(i);
        let frame = InputFrame::packed(W, H, &y, &uv);
        let aus = enc.encode(&frame, (i as i64) * 33_333_333).unwrap();
        assert_eq!(aus.len(), 1, "frame {i}: GPU backends emit exactly one AU");
        let encoded = &aus[0];
        assert!(!encoded.data.is_empty(), "frame {i} produced no data");
        assert_eq!(encoded.frame_index, i);
        assert_eq!(encoded.is_idr, i == 0);
        // Annex B: must start with a start code.
        let sc = encoded
            .data
            .windows(4)
            .position(|w| w == [0, 0, 0, 1])
            .unwrap_or_else(|| panic!("frame {i}: no start code in output"));
        assert!(sc <= 4, "start code not at beginning (offset {sc})");
        // HEVC NAL types: 19 = IDR_W_RADL, 1 = TRAIL_N; the first access unit
        // also carries VPS(32)/SPS(33)/PPS(34).
        let nals = nal_types(&encoded.data);
        assert!(!nals.is_empty(), "frame {i}: no NAL units found");
        if i == 0 {
            assert!(
                nals.contains(&19),
                "first frame must contain IDR_W_RADL, got {nals:?}"
            );
        } else {
            assert!(
                nals.iter().any(|&t| t <= 21 && t != 0),
                "frame {i}: no VCL NAL, got {nals:?}"
            );
        }
        out.extend_from_slice(&encoded.data);
    }
    enc.flush().unwrap();

    // Write for external validation (ffmpeg / project oracle).
    let path = std::env::temp_dir().join("vacc_vaenc_test.hevc");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&out).unwrap();
    eprintln!("wrote {} bytes to {}", out.len(), path.display());
}
