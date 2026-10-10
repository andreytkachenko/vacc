//! Parallel NVDEC + image-pipeline stress test: N threads each decode the same
//! file with scale+RGB on the GPU, reporting per-thread fps over a window.
//!
//! cargo run -p vacc --release --example parallel_stress -- -i <file> -t <threads> [-w 640 -h 480]

use std::sync::Arc;
use std::time::{Duration, Instant};

use vacc::{Backend, Decoder as VaccTrait, DecoderConfig, ImageConfig, Interpolation, RgbChannels, Scale, VaccDecoder};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input: Option<String> = None;
    let mut threads = 20usize;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut gpu = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "-i" | "--input" => input = Some(args.next().unwrap()),
            "-t" | "--threads" => threads = args.next().unwrap().parse().unwrap(),
            "-w" => width = Some(args.next().unwrap().parse().unwrap()),
            "-h" => height = Some(args.next().unwrap().parse().unwrap()),
            "--gpu" => gpu = true,
            _ => {}
        }
    }
    let input = input.expect("need -i <file>");
    let data = Arc::new(std::fs::read(&input).expect("read input"));

    let cfg = DecoderConfig::new([Backend::Nvdec])
        .with_image(ImageConfig {
            scale: width.zip(height).map(|(w, h)| Scale::new(w, h, Interpolation::Bilinear)),
            rgb: Some(RgbChannels::Rgb24),
            ..Default::default()
        })
        .with_gpu();
    println!("config: {cfg} gpu_track={gpu}");

    let window = Duration::from_secs(10);
    let mut handles = Vec::new();

    for i in 0..threads {
        let data = Arc::clone(&data);
        let cfg = cfg.clone();
        handles.push(std::thread::spawn(move || {
            let mut dec = VaccDecoder::new(&data, &cfg).expect("bootstrap");
            println!("thread {i}: backend={}", dec.backend());

            dec.submit(data.as_ref()).expect("submit");

            let start = Instant::now();
            let mut frames = 0u64;
            loop {
                match dec.decode() {
                    Ok(Some(f)) => {
                        frames += 1;
                        if f.rgb_pixels.is_none() && f.pixel_data.is_none() && f.gpu.is_none() {
                            eprintln!("thread {i}: frame without any pixels");
                        }
                        if gpu && f.gpu.is_none() {
                            eprintln!("thread {i}: gpu track fell back to host (frame {})", frames);
                        }
                        let el = start.elapsed();
                        if el >= Duration::from_secs(5) && el < window + Duration::from_secs(1) {
                            println!(
                                "thread {i}: t+{}s frames={} fps={:.1}",
                                el.as_secs(),
                                frames,
                                frames as f64 / el.as_secs_f64()
                            );
                        }
                        if start.elapsed() > window + Duration::from_secs(5) {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("thread {i}: decode error: {e}");
                        break;
                    }
                }
            }
            let el = start.elapsed();
            frames as f64 / el.as_secs_f64()
        }));
    }

    let mut total = 0.0f64;
    for (i, h) in handles.into_iter().enumerate() {
        let fps = h.join().expect("join");
        println!("thread {i}: final fps={fps:.1}");
        total += fps;
    }
    println!("TOTAL: {total:.1} fps over {threads} threads");
}
