//! Decode a file with the unified decoder and print per-frame info.
//!
//! ```text
//! cargo run -p vacc --example unified -- -i <file> [options]
//!
//!   -i, --input    <file>   bitstream (h264/h265 annex-b, vp9/av1 ivf) — required
//!   -o, --order    <csv>    backend order, e.g. "nvdec,software"
//!                           (default: vulkan,nvdec,vaapi,software)
//!   -n, --max      <num>    stop after this many frames (default: all)
//!   -w, --width    <px>     resize output width (with -H)
//!   -H, --height   <px>     resize output height (with -w)
//!   -f, --filter   <name>   resampling interpolation:
//!                           nearest | box | bilinear | bicubic (default: bilinear)
//!       --rgb24          convert frames to packed RGB24
//!       --rgb32          convert frames to packed RGBA32
//!       --gpu            GPU decode track (NVDEC only): frames stay in
//!                        device memory; the image pipeline runs on the GPU
//!   -O, --out      <file>   write the first decoded frame as PPM (needs --rgb24/--rgb32)
//! ```

use std::io::Read;
use std::time::Instant;

use vacc_core::decoder::Decoder;
use vacc_core::frame::RgbFrame;
use vacc::{
    Backend, DecodedFrame, DecoderConfig, GpuPixelFormat, ImageConfig, Interpolation, RgbChannels,
    Scale, VaccDecoder,
};

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// Print one decoded frame and bump the counter.
fn print_frame(frame: &DecodedFrame, total: &mut usize) {
    // Hash whatever the frame carries after the image pipeline ran.
    // Hash whatever the frame carries after the image pipeline ran. A
    // device-resident frame (GPU track) reports its pointer instead.
    let (hash, kind) = if let Some(rgb) = &frame.rgb_pixels {
        (fnv1a(&rgb.data), "rgb".to_string())
    } else if let Some(p) = &frame.pixel_data {
        (fnv1a(&p.buffer), "yuv".to_string())
    } else if let Some(g) = &frame.gpu {
        (g.ptr as u64, format!("gpu:{:?}", g.format))
    } else {
        (0, "none".to_string())
    };
    println!(
        "frame {}: ts={} size={}x{} {kind} hash={:016x}",
        *total, frame.timestamp, frame.width, frame.height, hash
    );
    *total += 1;
}

/// Write the first RGB frame as PPM (once), when `-O` was given. In GPU mode
/// the device RGB buffer is read back once for the file.
fn maybe_write_ppm(args: &Args, frame: &DecodedFrame, wrote: &mut bool) {
    if *wrote {
        return;
    }
    let Some(path) = &args.out else { return };
    if let Some(rgb) = &frame.rgb_pixels {
        write_ppm(path, rgb);
        println!("wrote {path} ({}x{})", rgb.width, rgb.height);
        *wrote = true;
        return;
    }
    if let Some(g) = frame.gpu.as_ref().filter(|g| g.format == GpuPixelFormat::Rgb24) {
        let buf = vacc_npp::readback(g).unwrap_or_else(|e| die(&format!("gpu readback: {e}")));
        let rgb = RgbFrame { data: buf, width: g.width, height: g.height, channels: 3 };
        write_ppm(path, &rgb);
        println!("wrote {path} ({}x{})", g.width, g.height);
        *wrote = true;
    }
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
    width: Option<u32>,
    height: Option<u32>,
    filter: Interpolation,
    rgb: Option<RgbChannels>,
    gpu: bool,
    out: Option<String>,
}

fn parse_args() -> Args {
    let mut input = None;
    let mut order = "vulkan,nvdec,vaapi,software".to_string();
    let mut max_frames = usize::MAX;
    let mut width = None;
    let mut height = None;
    let mut filter = Interpolation::Bilinear;
    let mut rgb = None;
    let mut gpu = false;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-i" | "--input" => input = Some(args.next().unwrap_or_else(|| die("-i needs a value"))),
            "-o" | "--order" => order = args.next().unwrap_or_else(|| die("-o needs a value")),
            "-n" | "--max" => max_frames = args
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| die("-n needs a number")),
            "-w" | "--width" => width = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| die("-w needs a number"))),
            "-H" | "--height" => height = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| die("-H needs a number"))),
            "-f" | "--filter" => {
                let name = args.next().unwrap_or_else(|| die("-f needs a value"));
                filter = match name.as_str() {
                    "nearest" => Interpolation::Nearest,
                    "box" => Interpolation::Box,
                    "bilinear" => Interpolation::Bilinear,
                    "bicubic" => Interpolation::Bicubic,
                    other => die(&format!("unknown filter '{other}' (nearest|box|bilinear|bicubic)")),
                }
            }
            "--rgb24" => rgb = Some(RgbChannels::Rgb24),
            "--rgb32" => rgb = Some(RgbChannels::Rgba32),
            "--gpu" => gpu = true,
            "-O" | "--out" => out = Some(args.next().unwrap_or_else(|| die("-O needs a value"))),
            other => die(&format!("unknown argument '{other}'")),
        }
    }
    if (width.is_none() && height.is_some()) || (width.is_some() && height.is_none()) {
        die("-w and -H must be given together");
    }
    if out.is_some() && rgb.is_none() {
        die("--out writes RGB frames; pass --rgb24 or --rgb32");
    }
    let usage = "usage: unified -i <file> [-o order] [-n max] [-w px -H px] [-f filter] [--rgb24|--rgb32] [--gpu] [-O out.ppm]";
    Args { input: input.unwrap_or_else(|| die(usage)), order, max_frames, width, height, filter, rgb, gpu, out }
}

/// Write the first frame as a binary PPM (alpha is dropped for RGBA32).
fn write_ppm(path: &str, frame: &RgbFrame) {
    let mut out = Vec::with_capacity(64 + frame.data.len());
    out.extend_from_slice(format!("P6\n{} {}\n255\n", frame.width, frame.height).as_bytes());
    if frame.channels == 3 {
        out.extend_from_slice(&frame.data);
    } else {
        for px in frame.data.chunks_exact(4) {
            out.extend_from_slice(&px[..3]);
        }
    }
    std::fs::write(path, out).unwrap_or_else(|e| die(&format!("cannot write {path}: {e}")));
}

fn main() {
    let args = parse_args();

    let order: Vec<Backend> = args
        .order
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|e: String| die(&e)))
        .collect();
    let mut config = DecoderConfig::new(order);
    if args.width.is_some() || args.rgb.is_some() {
        let image = ImageConfig {
            scale: args.width.zip(args.height).map(|(w, h)| Scale::new(w, h, args.filter)),
            rgb: args.rgb,
            ..Default::default()
        };
        config = config.with_image(image);
    }
    if args.gpu {
        config = config.with_gpu();
    }

    // Stream the file in chunks: seed the decoder with the head, then feed
    // the rest via submit() and pull frames one at a time.
    let mut file =
        std::fs::File::open(&args.input).unwrap_or_else(|e| die(&format!("cannot read {}: {}", args.input, e)));
    let mut probe = [0u8; 64 * 1024];
    let n = file
        .read(&mut probe)
        .unwrap_or_else(|e| die(&format!("read: {}", e)));
    if n == 0 {
        die("empty input");
    }

    let start = Instant::now();
    let mut decoder =
        VaccDecoder::new(&probe[..n], &config).unwrap_or_else(|e| die(&format!("init: {}", e)));
    println!(
        "backend={} codec={:?} config={} size={}x{} profile={:?}",
        decoder.backend(),
        decoder.info().codec,
        config,
        decoder.info().display_size.width,
        decoder.info().display_size.height,
        decoder.info().profile_idc
    );

    let mut total = 0usize;
    let mut wrote_out = false;

    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).unwrap_or_else(|e| die(&format!("read: {}", e)));
        if n == 0 {
            break;
        }
        decoder.submit(&buf[..n]).unwrap_or_else(|e| die(&format!("decode: {}", e)));
        while total < args.max_frames {
            match decoder.decode().unwrap_or_else(|e| die(&format!("decode: {}", e))) {
                Some(frame) => {
                    maybe_write_ppm(&args, &frame, &mut wrote_out);
                    print_frame(&frame, &mut total);
                }
                None => break,
            }
        }
    }
    for frame in decoder.flush().unwrap_or_else(|e| die(&format!("decode: {}", e))) {
        if total >= args.max_frames {
            break;
        }
        maybe_write_ppm(&args, &frame, &mut wrote_out);
        print_frame(&frame, &mut total);
    }

    if total == 0 {
        die("no frames decoded");
    }
    if args.out.is_some() && !wrote_out {
        die("--out requested but no RGB frame was produced (pass --rgb24/--rgb32)");
    }
    let elapsed = start.elapsed();
    println!(
        "total_frames={} elapsed={:?} fps={:.1}",
        total,
        elapsed,
        total as f64 / elapsed.as_secs_f64()
    );
}
