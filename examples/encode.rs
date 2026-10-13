//! Encode synthetic NV12 frames with the unified encoder and write Annex B.
//!
//! ```text
//! cargo run -p vacc --example encode -- [options]
//!
//!   -o, --order    <csv>     backend order, e.g. "nvdec,vaapi"
//!                             (default: vulkan,nvdec,vaapi)
//!   -n, --frames   <num>     number of frames to encode (default: 30)
//!   -w, --width    <px>      frame width, multiple of 16 (default: 320)
//!   -H, --height   <px>      frame height, multiple of 16 (default: 240)
//!   -q, --qp       <n>       constant QP 0-51 (default: 26)
//!   -I, --intra    <n>       IDR period in frames (default: 30)
//!   -O, --out      <file>    output bitstream (default: /tmp/vacc_encode.h264)
//! ```

use std::io::Write;
use std::time::Instant;

use vacc_core::encoder::{Encoder, EncoderConfig as CoreCfg, FrameKind, InputFrame};
use vacc::{Backend, EncoderConfig, VaccEncoder};

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

struct Args {
    order: Vec<Backend>,
    frames: u32,
    width: u32,
    height: u32,
    qp: u8,
    intra: u32,
    out: String,
}

fn parse_args() -> Args {
    let mut args = Args {
        order: vec![],
        frames: 30,
        width: 320,
        height: 240,
        qp: 26,
        intra: 30,
        out: "/tmp/vacc_encode.h264".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--order" => {
                let s = it.next().unwrap_or_else(|| die("missing order"));
                args.order = s
                    .split(',')
                    .map(|b| b.trim().parse::<Backend>().unwrap_or_else(|e| die(&e)))
                    .collect();
            }
            "-n" | "--frames" => args.frames = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| die("bad frame count")),
            "-w" | "--width" => args.width = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| die("bad width")),
            "-H" | "--height" => args.height = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| die("bad height")),
            "-q" | "--qp" => args.qp = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| die("bad qp")),
            "-I" | "--intra" => args.intra = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| die("bad intra period")),
            "-O" | "--out" => args.out = it.next().unwrap_or_else(|| die("missing output path")),
            other => die(&format!("unknown argument {other}")),
        }
    }
    if !args.width.is_multiple_of(16) || !args.height.is_multiple_of(16) {
        die("width/height must be multiples of 16");
    }
    args
}

/// Synthetic NV12 frame (moving gradient luma, drifting chroma).
fn make_frame(frame_idx: u32, w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; (w * h) as usize];
    for py in 0..h {
        for px in 0..w {
            y[(py * w + px) as usize] = ((px * 7 + py * 3 + frame_idx * 16) % 256) as u8;
        }
    }
    let mut uv = vec![0u8; (w * h / 2) as usize];
    for row in 0..(h / 2) {
        for col in 0..(w / 2) {
            let o = (row * w + col * 2) as usize;
            uv[o] = (128u32.wrapping_add(frame_idx * 7)) as u8;
            uv[o + 1] = (128u32.wrapping_sub(frame_idx * 5)) as u8;
        }
    }
    (y, uv)
}

fn main() {
    let args = parse_args();
    let params = CoreCfg {
        width: args.width,
        height: args.height,
        qp: args.qp,
        intra_period: args.intra,
        ..Default::default()
    };
    let config = if args.order.is_empty() {
        EncoderConfig::new(params)
    } else {
        EncoderConfig::new(params).with_backends(args.order)
    };

    let mut enc = VaccEncoder::new(&config).unwrap_or_else(|e| die(&e.to_string()));
    println!("backend: {}  config: {}", enc.backend(), config);

    let t0 = Instant::now();
    let mut out = Vec::new();
    for i in 0..args.frames {
        let (y, uv) = make_frame(i, args.width, args.height);
        let frame = InputFrame::packed(args.width, args.height, &y, &uv);
        let aus = enc.encode(&frame, (i as i64) * 33_333_333).unwrap_or_else(|e| die(&e.to_string()));
        for au in &aus {
            println!(
                "frame {:>3}: {} bytes {} poc={}",
                au.frame_index,
                au.size(),
                if au.is_idr {
                    "IDR"
                } else {
                    match au.kind {
                        FrameKind::I => "I",
                        FrameKind::P => "P",
                        FrameKind::B => "B",
                    }
                },
                au.poc
            );
        }
        for au in &aus {
            out.extend_from_slice(&au.data);
        }
    }
    for au in enc.flush().unwrap_or_else(|e| die(&e.to_string())) {
        out.extend_from_slice(&au.data);
    }

    let mut f = std::fs::File::create(&args.out).unwrap_or_else(|e| die(&e.to_string()));
    f.write_all(&out).unwrap_or_else(|e| die(&e.to_string()));
    println!(
        "wrote {} bytes to {} in {:.1} ms ({:.1} fps)",
        out.len(),
        args.out,
        t0.elapsed().as_secs_f64() * 1000.0,
        args.frames as f64 / t0.elapsed().as_secs_f64()
    );
}
