// Y'CbCr 4:2:0 -> RGBA32 conversion (alpha = 0xFF; RGB24 is produced by
// dropping alpha on the CPU side).
//
// Reproduces the reference software pipeline's Q14 fixed-point conversion
// (vacc-image `conv_px`) exactly: with samples y/cb/cr in 0..=255,
//
//   c = clamp0_255( floor( (ky*y + cb*cb_k + cr*cr_k + off + 8192) / 16384 ) )
//
// All planes live in one storage buffer of u32 words, one word per sample
// (8-bit values, or top-justified 10-bit values down-cast to 8 on the CPU).
// Offsets/pitches are in words. `semi`: 0 = separate Cb/Cr planes, 1 =
// interleaved CbCr plane (Cr sits one word after Cb).

struct Params {
    src_w: u32,
    src_h: u32,
    y_off: u32,
    y_pitch: u32,
    cb_off: u32,
    cb_pitch: u32,
    cr_off: u32,
    cr_pitch: u32,
    semi: u32,
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

fn load_plane(off: u32, pitch: u32, x: u32, y: u32) -> f32 {
    return f32(src[off + y * pitch + x]);
}

// 4:2:0 chroma sample at luma coordinates (cx, cy).
fn load_chroma(is_cr: bool, cx: u32, cy: u32) -> f32 {
    if (p.semi == 1u) {
        let k = cx * 2u + select(0u, 1u, is_cr);
        return f32(src[p.cb_off + cy * p.cb_pitch + k]);
    }
    if (!is_cr) {
        return load_plane(p.cb_off, p.cb_pitch, cx, cy);
    }
    return load_plane(p.cr_off, p.cr_pitch, cx, cy);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let yv = load_plane(p.y_off, p.y_pitch, x, y);
    let cb = load_chroma(false, x / 2u, y / 2u);
    let cr = load_chroma(true, x / 2u, y / 2u);

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
