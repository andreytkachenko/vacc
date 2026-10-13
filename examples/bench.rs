//! Performance baseline for the unsafe-elimination campaign (see
//! UNSAFE_AUDIT.md). Run in release; record the output before and after any
//! SIMD/hot-loop change — a regression blocks the change.
//!
//! ```text
//! cargo run -p vacc --release --example bench -- decode <file> [max_frames]
//! cargo run -p vacc --release --example bench -- image
//! ```

use std::time::Instant;

use vacc_core::decoder::Decoder;
use vacc_image::{Affine, ImageConfig, Interpolation, Kernel, RgbChannels, Scale, Warp};
use vacc::{Backend, DecoderConfig, VaccDecoder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("decode") => {
            let path = args.get(1).expect("usage: bench decode <file> [max_frames]");
            let max_frames = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(300);
            bench_decode(path, max_frames)
        }
        Some("image") => bench_image(),
        other => {
            eprintln!("unknown subcommand: {other:?}");
            std::process::exit(2)
        }
    }
}

/// Software-decode `max_frames` frames of `path`; report best-of-3 fps.
fn bench_decode(path: &str, max_frames: usize) {
    let data = std::fs::read(path).expect("read input");
    let cfg = DecoderConfig::new([Backend::Software]);
    let probe_len = data.len().min(256 * 1024);

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let mut dec = VaccDecoder::new(&data[..probe_len], &cfg).expect("init decoder");
        dec.submit(&data[probe_len..]).expect("submit");
        let t = Instant::now();
        let mut n = 0usize;
        while n < max_frames {
            if dec.decode().expect("decode").is_none() {
                break;
            }
            n += 1;
        }
        let secs = t.elapsed().as_secs_f64();
        best = best.min(secs);
        eprintln!("  run: {n} frames in {secs:.3}s");
    }
    println!(
        "decode {path}: {:.1} fps ({} frames, best of 3)",
        (max_frames as f64) / best,
        max_frames
    );
}

/// Synthetic 1080p sources through every kernel: conversion, resize, warp.
/// Reports ms/op; compare before/after SIMD rewrites.
fn bench_image() {
    const W: usize = 1920;
    const H: usize = 1080;
    let cw = W / 2;
    let ch = H / 2;

    // Deterministic gradients (not all-zero: exercises every SIMD lane).
    let mut i420 = vec![0u8; W * H + 2 * cw * ch];
    for r in 0..H {
        for c in 0..W {
            i420[r * W + c] = ((c * 255 / W) ^ (r * 17 % 256)) as u8;
        }
    }
    let uv_off = W * H;
    for i in 0..2 * cw * ch {
        i420[uv_off + i] = 128 + (i * 7 % 64) as i32 as u8 - 32;
    }
    let y_slice = &i420[..W * H];
    let cb_slice = &i420[W * H..W * H + cw * ch];
    let cr_slice = &i420[W * H + cw * ch..];
    let planar = vacc_image::YuvImage::planar(y_slice, W, cb_slice, cw, cr_slice, cw, W, H, 8);
    let semi = vacc_image::YuvImage::semi(&i420[..W * H], W, &i420[W * H..], W, W, H, 8);

    let kernels = [Kernel::Scalar, Kernel::Sse, Kernel::Avx2];
    let mut sink: Vec<u8> = Vec::new();

    // RGB conversion via convert_rows directly (yuv_to_rgb forces Auto).
    for k in kernels {
        for ch_out in [RgbChannels::Rgb24, RgbChannels::Rgba32] {
            time_conv(&format!("conv {k:?} {:?} 1080p", ch_out), k, &planar, ch_out, &mut sink);
        }
    }
    // Resize to 720p (bilinear + bicubic).
    for k in kernels {
        for filter in [Interpolation::Bilinear, Interpolation::Bicubic] {
            let cfg = ImageConfig {
                scale: Some(Scale::new(1280, 720, filter)),
                ..Default::default()
            };
            time_op(&format!("resize {k:?} {:?} 1080p->720p", filter), k, &semi, &cfg, &mut sink);
        }
        // Warp (translate + rgb).
        let cfg = ImageConfig {
            affine: Some(Warp::bilinear(Affine::translate(4.0, 2.0))),
            rgb: Some(RgbChannels::Rgba32),
            ..Default::default()
        };
        time_op(&format!("warp {k:?} translate+rgb 1080p"), k, &planar, &cfg, &mut sink);
    }
    // P010 downcast path (u16 source).
    let mut p010 = vec![0u16; W * H + 2 * cw * ch];
    for i in 0..W * H {
        p010[i] = ((i as u32 % 1024) << 6) as u16;
    }
    for i in W * H..W * H + 2 * cw * ch {
        p010[i] = (512u16) << 6;
    }
    let bytes: Vec<u8> = p010.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p010_img = vacc_image::YuvImage::planar(
        &bytes[..W * H * 2],
        W * 2,
        &bytes[W * H * 2..W * H * 2 + cw * ch * 2],
        cw * 2,
        &bytes[W * H * 2 + cw * ch * 2..],
        cw * 2,
        W,
        H,
        10,
    );
    let cfg = ImageConfig { rgb: Some(RgbChannels::Rgb24), ..Default::default() };
    for k in [Kernel::Sse, Kernel::Avx2] {
        time_op(&format!("p010-conv+downcast {k:?} 1080p"), k, &p010_img, &cfg, &mut sink);
    }
    eprintln!("sink bytes: {}", sink.len());
}

/// One RGB conversion with an explicit kernel (convert_rows directly).
fn time_conv(
    label: &str,
    kernel: Kernel,
    src: &vacc_image::YuvImage,
    ch_out: RgbChannels,
    sink: &mut Vec<u8>,
) {
    let chb = ch_out.bytes() as usize;
    let mut out = vec![0u8; src.width * src.height * chb];
    vacc_image::convert_rows(src, &mut out, src.width * chb, chb, &vacc_image::table(vacc_image::ColorSpec::default()), kernel);
    let best = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            vacc_image::convert_rows(src, &mut out, src.width * chb, chb, &vacc_image::table(vacc_image::ColorSpec::default()), kernel);
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    sink.clear();
    sink.extend_from_slice(&out);
    println!("{label:44} {:.4} ms/op", best * 1000.0);
}

fn time_op(
    label: &str,
    kernel: Kernel,
    src: &vacc_image::YuvImage,
    cfg: &ImageConfig,
    sink: &mut Vec<u8>,
) {
    // Warm-up (page faults, allocator growth).
    vacc_image::process(src, cfg, kernel).expect("process");
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let out = vacc_image::process(src, cfg, kernel).expect("process");
        best = best.min(t.elapsed().as_secs_f64());
        sink.clear();
        if let vacc_image::ProcessedFrame::Rgb(r) = out {
            sink.extend_from_slice(&r.data);
        }
    }
    println!("{label:44} {:.4} ms/op", best * 1000.0);
}
