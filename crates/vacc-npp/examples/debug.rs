//! Smoke test: run the NPP pipeline on a small frame and print samples.
use vacc_image::{i420_size, ColorSpec, Interpolation, RgbChannels, Scale, YuvImage};
use vacc_npp::Npp;

fn main() {
    let npp = match Npp::load() {
        Ok(n) => n,
        Err(e) => {
            println!("load failed: {e}");
            return;
        }
    };
    println!("loaded ok");
    println!("{}", npp.debug_probe());

    // 8x8 NV12: Y = x*32 (0..224 by column), UV = 128/128 (neutral).
    let w = 8usize;
    let h = 8usize;
    let cw = w / 2;
    let chh = h / 2;
    let y: Vec<u8> = (0..h * w).map(|i| ((i % w) * 32) as u8).collect();
    let uv = vec![128u8; cw * 2 * chh];
    let mut buf = Vec::new();
    buf.extend_from_slice(&y);
    buf.extend_from_slice(&uv);
    let img = YuvImage::semi(&buf[..w * h], w, &buf[w * h..], cw * 2, w, h, 8);

    // YUV -> RGB (bt.709 limited, neutral UV => gray ramp).
    let mut rgb = vec![0u8; w * h * 3];
    npp.yuv_to_rgb(&img, ColorSpec::default(), RgbChannels::Rgb24, &mut rgb)
        .expect("yuv_to_rgb");
    println!("rgb (bt709 limited, neutral uv):");
    for py in 0..h {
        let row: Vec<u8> = rgb[py * w * 3..py * w * 3 + w * 3].to_vec();
        println!("row{py}: {row:?}");
    }

    // Resize 8x8 -> 4x4 (I420 out).
    let mut small = vec![0u8; i420_size(4, 4)];
    npp.resize_yuv(&img, Scale::new(4, 4, Interpolation::Bilinear), &mut small)
        .expect("resize_yuv");
    println!("resize 8x8 -> 4x4 luma:");
    for py in 0..4 {
        println!("row{py}: {:?}", &small[py * 4..py * 4 + 4]);
    }
}
