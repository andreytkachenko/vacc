// Bilinear NV12 (byte-packed) -> NV12 resize for the zero-copy device path.
//
// Input offsets/pitches are in bytes; the output is tight NV12 (Y plane
// first, then interleaved CbCr). Dimensions must be even, so the Y/UV
// boundary is word-aligned and every output word is written by exactly one
// invocation — no byte read-modify-write races.
//
// Each plane is resampled independently with a centered two-tap bilinear:
//
//   center = (dst + 0.5) * (src / dst) - 0.5, taps clamped to [0, src-1]
//
// matching the reference software pipeline bit-for-bit: the h pass rounds to
// u8 (the software scratch is u8), then the v pass blends the rounded rows —
// at integer scale ratios every intermediate is exact, so the result is
// identical to the SIMD host kernels.

struct Params {
    src_w: u32,
    src_h: u32,
    y_off: u32,
    y_pitch: u32,
    uv_off: u32,
    uv_pitch: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn sbyte(off: u32) -> f32 {
    let w = src[off / 4u];
    return f32((w >> (8u * (off % 4u))) & 0xFFu);
}

// One horizontal bilinear tap pair over a byte plane, rounded to u8 exactly
// like the software h pass (its scratch is u8, so the v pass sees rounded
// values). `stride` is the byte distance between horizontal samples (1 for
// luma, 2 for interleaved CbCr).
fn h_tap(off: u32, x0: u32, x1: u32, wx: f32, stride: u32) -> u32 {
    let a = sbyte(off + x0 * stride);
    let b = sbyte(off + x1 * stride);
    return round_clamp(f32(a) * (1.0 - wx) + f32(b) * wx);
}

// Centered two-tap bilinear over a byte plane: horizontal taps rounded to
// u8, then a vertical blend of the rounded rows (two passes, like the
// reference software pipeline).
fn bilinear(off: u32, pitch: u32, fx: f32, fy: f32, w: u32, h: u32, stride: u32) -> u32 {
    let cx = clamp(fx, 0.0, f32(w - 1u));
    let cy = clamp(fy, 0.0, f32(h - 1u));
    let x0 = u32(cx);
    let y0 = u32(cy);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let wx = cx - f32(x0);
    let wy = cy - f32(y0);
    let top = h_tap(off + y0 * pitch, x0, x1, wx, stride);
    let bot = h_tap(off + y1 * pitch, x0, x1, wx, stride);
    return round_clamp(f32(top) * (1.0 - wy) + f32(bot) * wy);
}

fn round_clamp(v: f32) -> u32 {
    return u32(clamp(floor(v + 0.5), 0.0, 255.0));
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let y_bytes = p.dst_w * p.dst_h;
    let y_words = (y_bytes + 3u) / 4u;
    if (g < y_words) {
        // Luma word: output bytes [4g, 4g+4) of the Y plane.
        var out = 0u;
        for (var k = 0u; k < 4u; k++) {
            let b = g * 4u + k;
            if (b >= y_bytes) {
                break;
            }
            let x = b % p.dst_w;
            let y = b / p.dst_w;
            let fx = (f32(x) + 0.5) * f32(p.src_w) / f32(p.dst_w) - 0.5;
            let fy = (f32(y) + 0.5) * f32(p.src_h) / f32(p.dst_h) - 0.5;
            out = out | (bilinear(p.y_off, p.y_pitch, fx, fy, p.src_w, p.src_h, 1u) << (8u * k));
        }
        dst[p.dst_off / 4u + g] = out;
        return;
    }

    // Chroma word: output bytes [4h, 4h+4) of the CbCr plane. The region
    // starts right after the Y plane (word-aligned for even dimensions).
    let h = g - y_words;
    let dwc = (p.dst_w + 1u) / 2u;
    let dhc = (p.dst_h + 1u) / 2u;
    let swc = (p.src_w + 1u) / 2u;
    let shc = (p.src_h + 1u) / 2u;
    let uv_bytes = p.dst_w * dhc;
    if (h >= (uv_bytes + 3u) / 4u) {
        return;
    }
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        let b = h * 4u + k;
        if (b >= uv_bytes) {
            break;
        }
        let row = b / p.dst_w;
        let col = b % p.dst_w;
        let cxo = col / 2u;
        let fxc = (f32(cxo) + 0.5) * f32(swc) / f32(dwc) - 0.5;
        let fyc = (f32(row) + 0.5) * f32(shc) / f32(dhc) - 0.5;
        let base = p.uv_off + select(0u, 1u, (col % 2u) == 1u);
        out = out | (bilinear(base, p.uv_pitch, fxc, fyc, swc, shc, 2u) << (8u * k));
    }
    dst[p.dst_off / 4u + y_words + h] = out;
}
