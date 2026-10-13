//! Round-trip oracle without ffmpeg: encode with [`H264Encoder`], decode with the
//! project's own CPU decoder ([`SwH264Decoder`], verified byte-exact against FFmpeg on
//! the decode matrix), and require the decoded pixels to be **bit-identical** to the
//! encoder's own reconstruction. Any spec deviation in prediction, transform, quantisation
//! or deblocking breaks the match.

mod common;

use common::Yuv;
use vacc_core::decoder::Decoder;
use vacc_software_decode::SwH264Decoder;
use vacc_software_encode::{EncoderConfig, H264Encoder, Preset, Profile, RateControl, ReconFrame};

/// Decode an Annex-B stream with the project SW decoder into planar I420 frames.
/// Drains exactly like the golden harness: `decode()` loop, then `flush()` at end of
/// stream (the last NAL is held back as possibly-truncated until flush releases it).
fn decode_stream(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut d = SwH264Decoder::new(stream.to_vec()).expect("SW decoder must accept our stream");
    let mut out = Vec::new();
    while let Some(f) = d.decode().expect("SW decode failed") {
        out.push(frame_bytes(&f));
    }
    for f in d.flush().expect("SW flush failed") {
        out.push(frame_bytes(&f));
    }
    out
}

fn frame_bytes(f: &vacc_core::DecodedFrame) -> Vec<u8> {
    let pd = f.pixel_data.as_ref().expect("pixel data present");
    assert_eq!(pd.format, "I420", "unexpected output format {}", pd.format);
    let mut buf = Vec::with_capacity(f.width as usize * f.height as usize * 3 / 2);
    for plane in [&pd.y, &pd.u, pd.v.as_ref().expect("I420 has a V plane")] {
        for r in 0..plane.height {
            let row = pd.plane_row(plane, r, plane.width);
            buf.extend_from_slice(row);
        }
    }
    buf
}

struct Run {
    stream: Vec<u8>,
    recon: Vec<ReconFrame>,
}

fn encode(cfg: EncoderConfig, frames: &[Yuv]) -> Run {
    let mut enc = H264Encoder::new(cfg).unwrap();
    enc.set_recon_capture(true);
    let mut stream = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        for p in enc.encode(&f.frame(), i as i64).unwrap() {
            stream.extend_from_slice(&p.data);
        }
    }
    for p in enc.flush() {
        stream.extend_from_slice(&p.data);
    }
    let mut recon = enc.take_recon();
    recon.sort_by_key(|r| r.pts);
    Run { stream, recon }
}

/// Decoded frames must be bit-identical to the encoder's reconstruction (display order).
fn verify(name: &str, run: &Run, w: usize, h: usize, n: usize) {
    let dec = decode_stream(&run.stream);
    assert_eq!(dec.len(), n, "{name}: decoded frame count");
    assert_eq!(run.recon.len(), n, "{name}: reconstruction count");
    for (i, r) in run.recon.iter().enumerate() {
        let d = &dec[i];
        assert!(d[..w * h] == r.y[..], "{name}: frame {i} luma differs from reconstruction");
        assert!(d[w * h..w * h * 5 / 4] == r.u[..], "{name}: frame {i} Cb differs");
        assert!(d[w * h * 5 / 4..] == r.v[..], "{name}: frame {i} Cr differs");
    }
}

fn cfg(w: usize, h: usize, profile: Profile, bframes: u8, rate: RateControl, slices: usize) -> EncoderConfig {
    let mut c = EncoderConfig::new(w as u32, h as u32, 30, 1);
    c.profile = profile;
    c.preset = Preset::Balanced;
    c.bframes = bframes;
    c.rate = rate;
    c.slices = slices;
    c.threads = 4;
    c
}

#[test]
fn roundtrip_baseline_cavlc() {
    let (w, h, n) = (176, 144, 12);
    let frames: Vec<Yuv> = (0..n).map(|t| common::synth(w, h, t)).collect();
    let run = encode(cfg(w, h, Profile::Baseline, 0, RateControl::Qp(26), 1), &frames);
    verify("baseline", &run, w, h, n);
}

#[test]
fn roundtrip_main_b2() {
    let (w, h, n) = (320, 240, 12);
    let frames: Vec<Yuv> = (0..n).map(|t| common::synth(w, h, t)).collect();
    let run = encode(cfg(w, h, Profile::Main, 2, RateControl::Qp(30), 1), &frames);
    verify("main_b2", &run, w, h, n);
}

#[test]
fn roundtrip_high_b3_slices() {
    let (w, h, n) = (320, 240, 12);
    let frames: Vec<Yuv> = (0..n).map(|t| common::synth(w, h, t)).collect();
    let run = encode(cfg(w, h, Profile::High, 3, RateControl::Qp(22), 3), &frames);
    verify("high_b3_slices", &run, w, h, n);
}

#[test]
fn roundtrip_crf_b2() {
    let (w, h, n) = (320, 240, 12);
    let frames: Vec<Yuv> = (0..n).map(|t| common::synth(w, h, t)).collect();
    let run = encode(cfg(w, h, Profile::High, 2, RateControl::Crf(23.0), 1), &frames);
    verify("high_crf_b2", &run, w, h, n);
}

#[test]
fn roundtrip_odd_size_crop() {
    let (w, h, n) = (330, 250, 8);
    let frames: Vec<Yuv> = (0..n).map(|t| common::synth(w, h, t)).collect();
    let run = encode(cfg(w, h, Profile::High, 1, RateControl::Qp(34), 1), &frames);
    verify("odd_size", &run, w, h, n);
}
