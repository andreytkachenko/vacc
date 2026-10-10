//! Dump the generated SPS+PPS prefix NALUs (Annex B) for field-level
//! comparison against reference encoders / the project oracle.
//!
//! Usage: cargo run -p vacc-vulkan-encode --example dump_prefix [--width W] [--height H]

use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let mut width = 320u32;
    let mut height = 240u32;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--width" => width = args.next().unwrap().parse().unwrap(),
            "--height" => height = args.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    // Main profile, level 3.1, 1 reference frame (matches the unit test).
    let params = vacc_vulkan_encode::H264Params::new(width / 16, height / 16, 77, 31, 1);
    let nalus = params.prefix_nalus();
    std::fs::write("/tmp/vkenc_prefix.h264", &nalus).unwrap();
    let hex: Vec<String> = nalus.iter().map(|b| format!("{b:02x}")).collect();
    println!("{} bytes written to /tmp/vkenc_prefix.h264", nalus.len());
    println!("{}", hex.join(" "));
}
