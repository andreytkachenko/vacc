//! End-to-end encode test: 10 synthetic NV12 frames -> Annex B HEVC stream.
//!
//! Requires a Vulkan device with VK_KHR_video_encode_h265 (on this machine:
//! the NVIDIA GPU — the Intel iGPU exposes no video extensions); skipped
//! otherwise.

use std::io::Write;
use vacc_core::codec::VideoCodec;
use vacc_core::encoder::{Encoder, EncoderConfig, InputFrame};
use vacc_vulkan_encode::VulkanEncoder;

const W: u32 = 320;
const H: u32 = 240;
const NFRAMES: u32 = 10;

/// Synthetic NV12 frame matching the VAAPI/NVENC test pattern.
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

#[test]
fn encode_h265_10_frames() {
    let config = EncoderConfig {
        width: W,
        height: H,
        qp: 26,
        codec: VideoCodec::EncodeH265,
        ..Default::default()
    };
    let mut enc = match VulkanEncoder::new(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Vulkan H.265 encode unavailable (skipping): {e}");
            return;
        }
    };

    let info = enc.info();
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
            .windows(3)
            .position(|w| w == [0, 0, 1])
            .unwrap_or_else(|| panic!("frame {i}: no start code in output"));
        assert!(sc <= 4, "start code not at beginning (offset {sc})");
        out.extend_from_slice(&encoded.data);
    }
    enc.flush().unwrap();

    // Write for external validation (ffmpeg / project oracle).
    let path = std::env::temp_dir().join("vacc_vkenc_test.h265");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&out).unwrap();
    eprintln!("wrote {} bytes to {}", out.len(), path.display());
}
