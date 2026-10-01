// NV12 (byte-packed) -> RGBA32 conversion for the zero-copy device path.
//
// `src` is a byte buffer holding a Y plane followed by an interleaved CbCr
// plane (all offsets/pitches in bytes); `dst` is an array of u32 words, one
// RGBA per pixel. The Q14 fixed-point conversion matches the reference
// software pipeline (vacc-image `conv_px`) bit-for-bit, like yuv2rgb.wgsl.

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
    ky: i32,
    r_cr: i32,
    r_off: i32,
    g_cb: i32,
    g_cr: i32,
    g_off: i32,
    b_cb: i32,
    b_off: i32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn byte_at(off: u32) -> f32 {
    let w = src[off / 4u];
    return f32((w >> (8u * (off % 4u))) & 0xFFu);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let yv = byte_at(p.y_off + y * p.y_pitch + x);
    let uv_base = p.uv_off + (y / 2u) * p.uv_pitch + (x / 2u) * 2u;
    let cb = byte_at(uv_base);
    let cr = byte_at(uv_base + 1u);

    // Q14 fixed-point, matching the software tables bit-for-bit (all
    // intermediates stay below 2**24, so f32 is exact).
    let r = (f32(p.ky) * yv + f32(p.r_cr) * cr + f32(p.r_off) + 8192.0) / 16384.0;
    let g = (f32(p.ky) * yv + f32(p.g_cb) * cb + f32(p.g_cr) * cr + f32(p.g_off) + 8192.0) / 16384.0;
    let b = (f32(p.ky) * yv + f32(p.b_cb) * cb + f32(p.b_off) + 8192.0) / 16384.0;

    let ro = u32(clamp(floor(r), 0.0, 255.0));
    let go = u32(clamp(floor(g), 0.0, 255.0));
    let bo = u32(clamp(floor(b), 0.0, 255.0));
    dst[p.dst_off + y * p.dst_w + x] = ro | (go << 8u) | (bo << 16u) | (255u << 24u);
}
