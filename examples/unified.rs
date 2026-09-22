//! Decode a file with the unified decoder and print per-frame info.
//!
//! ```text
//! cargo run -p vacc --example unified -- -i <file> [-o order] [-n max_frames]
//!
//!   -i, --input    <file>   bitstream (h264/h265 annex-b, vp9/av1 ivf) — required
//!   -o, --order    <csv>    backend order, e.g. "nvdec,software"
//!                           (default: vulkan,nvdec,vaapi,software)
//!   -n, --max      <num>    stop after this many frames (default: all)
//! ```

use std::time::Instant;

use vacc_core::decoder::Decoder;
use vacc::{Backend, DecoderConfig, VaccDecoder};

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// FNV-1a 64 over the frame's pixel buffer (smoke hash, no deps).
fn fnv1a(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Args {
    input: String,
    order: String,
    max_frames: usize,
}

fn parse_args() -> Args {
    let mut input = None;
    let mut order = "vulkan,nvdec,vaapi,software".to_string();
    let mut max_frames = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-i" | "--input" => input = Some(args.next().unwrap_or_else(|| die("-i needs a value"))),
            "-o" | "--order" => order = args.next().unwrap_or_else(|| die("-o needs a value")),
            "-n" | "--max" => max_frames = args
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| die("-n needs a number")),
            other => die(&format!("unknown argument '{other}'")),
        }
    }
    Args { input: input.unwrap_or_else(|| die("usage: unified -i <file> [-o order] [-n max]")), order, max_frames }
}

fn main() {
    let args = parse_args();
    let data = std::fs::read(&args.input).unwrap_or_else(|e| die(&format!("cannot read {}: {}", args.input, e)));

    let order: Vec<Backend> = args
        .order
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|e: String| die(&e)))
        .collect();
    let config = DecoderConfig::new(order);

    let start = Instant::now();
    let mut decoder =
        VaccDecoder::new(data, &config).unwrap_or_else(|e| die(&format!("init: {}", e)));
    println!(
        "backend={} codec={:?} order={} size={}x{} profile={:?}",
        decoder.backend(),
        decoder.info().codec,
        config,
        decoder.info().display_size.width,
        decoder.info().display_size.height,
        decoder.info().profile_idc
    );

    let frames = decoder.decode_all(args.max_frames).unwrap_or_else(|e| die(&format!("decode: {}", e)));
    if frames.is_empty() {
        die("no frames decoded");
    }
    for (i, frame) in frames.iter().enumerate() {
        let hash = frame.pixel_data.as_ref().map(|p| fnv1a(&p.buffer)).unwrap_or(0);
        println!(
            "frame {}: ts={} size={}x{} hash={:016x}",
            i, frame.timestamp, frame.width, frame.height, hash
        );
    }
    let elapsed = start.elapsed();
    println!(
        "total_frames={} elapsed={:?} fps={:.1}",
        frames.len(),
        elapsed,
        frames.len() as f64 / elapsed.as_secs_f64()
    );
}
