//! HEVC software-decode throughput benchmark (Rust pipeline only).
//!
//! Usage: cargo run -p vacc-software-decode --release --example hevc_bench
//!         <file.h265> [iters]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use vacc_core::decoder::Decoder;
use vacc_software_decode::SoftwareH265Decoder;

fn run_once(data: Vec<u8>) -> u32 {
    let mut dec = SoftwareH265Decoder::new(data).expect("init");
    let mut n = 0u32;
    while let Some(_f) = dec.decode().expect("decode") {
        n += 1;
    }
    n += dec.flush().expect("flush").len() as u32;
    n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| panic!("usage: hevc_bench <file.h265> [iters]")),
    );
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let data = fs::read(&path).expect("read stream");

    let n = run_once(data.clone()); // warmup (page cache, icache)
    println!("{}: {} frames", path.display(), n);

    let mut best = f64::MAX;
    for _ in 0..iters {
        let payload = data.clone(); // copy outside the timed region
        let t0 = Instant::now();
        run_once(payload);
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        println!("  {:.3}s -> {:.1} fps", dt, n as f64 / dt);
    }
    println!("best: {:.3}s ({:.1} fps)", best, n as f64 / best);
}
