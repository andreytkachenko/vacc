// Bilinear Y'CbCr 4:2:0 resize (8-bit samples, one u32 word per sample).
//
// Each plane is resampled independently with a centered two-tap bilinear:
//
//   center = (dst + 0.5) * (src / dst) - 0.5, taps clamped to [0, src-1]
//
// matching the reference software pipeline bit-for-bit: the h pass rounds to
// u8 (the software scratch is u8), then the v pass blends the rounded rows —
// at integer scale ratios every intermediate is exact, so the result is
// identical to the SIMD host kernels.
//
// The output scratch is always stored planar-style: [Y][Cb][Cr], one word
// per sample; the CPU repacks it into tight I420 or NV12 bytes on readback.
// For interleaved (semi) input chroma, `chroma_stride` is 2.

struct Params {
    src_w: u32,
    src_h: u32,
    y_off: u32,
    y_pitch: u32,
    cb_off: u32,
    cb_pitch: u32,
    cr_off: u32,
    cr_pitch: u32,
    chroma_stride: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

// One horizontal bilinear tap pair over a plane of samples, rounded to u8
// exactly like the software h pass (its scratch is u8, so the v pass sees
// rounded values). `stride` is the word distance between horizontal samples.
fn h_tap(off: u32, x0: u32, x1: u32, wx: f32, stride: u32) -> u32 {
    let a = f32(src[off + x0 * stride]);
    let b = f32(src[off + x1 * stride]);
    return round_clamp(a * (1.0 - wx) + b * wx);
}

// Centered two-tap bilinear over a plane of `w x h` samples: horizontal taps
// rounded to u8, then a vertical blend of the rounded rows (two passes, like
// the reference software pipeline).
fn bilinear(off: u32, pitch: u32, stride: u32, fx: f32, fy: f32, w: u32, h: u32) -> u32 {
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;

    // Luma.
    let fx = (f32(x) + 0.5) * f32(p.src_w) / f32(p.dst_w) - 0.5;
    let fy = (f32(y) + 0.5) * f32(p.src_h) / f32(p.dst_h) - 0.5;
    let yv = bilinear(p.y_off, p.y_pitch, 1u, fx, fy, p.src_w, p.src_h);
    dst[p.dst_off + y * p.dst_w + x] = yv;

    // Chroma: output plane is (dw+1)/2 x (dh+1)/2.
    let dwc = (p.dst_w + 1u) / 2u;
    let dhc = (p.dst_h + 1u) / 2u;
    let cxo = x / 2u;
    let cyo = y / 2u;
    if (cxo >= dwc || cyo >= dhc) {
        return;
    }
    let swc = (p.src_w + 1u) / 2u;
    let shc = (p.src_h + 1u) / 2u;
    let fxc = (f32(cxo) + 0.5) * f32(swc) / f32(dwc) - 0.5;
    let fyc = (f32(cyo) + 0.5) * f32(shc) / f32(dhc) - 0.5;
    let cb = bilinear(p.cb_off, p.cb_pitch, p.chroma_stride, fxc, fyc, swc, shc);
    let cr = bilinear(p.cr_off, p.cr_pitch, p.chroma_stride, fxc, fyc, swc, shc);

    // Planar-style scratch: [Y][Cb][Cr].
    let uv_base = p.dst_off + p.dst_w * p.dst_h;
    dst[uv_base + cyo * dwc + cxo] = cb;
    dst[uv_base + dwc * dhc + cyo * dwc + cxo] = cr;
}
