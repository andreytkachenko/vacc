// Affine warp of RGBA32 pixels via inverse mapping (one u32 word per pixel:
// R | G<<8 | B<<16 | A<<24).
//
// The uniform carries the *inverse* 2x3 transform, so every output pixel maps
// to a source position (no holes):
//
//   sx = m00*x + m01*y + m02
//   sy = m10*x + m11*y + m12
//
// Samples outside the source contribute zero, so edges fade to black. Output
// alpha is always 0xFF, matching the reference software pipeline (its warp
// input always carries alpha 0xFF and it forces alpha on output).
//
// interp: 0 = nearest (floor of the source position),
//         1 = bilinear (2x2 taps),
//         2 = bicubic (Mitchell B=0.5 C=0.5, separable 4x4 taps).

struct Params {
    src_w: u32,
    src_h: u32,
    src_off: u32,
    src_pitch: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    interp: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

// One source pixel as RGBA floats; out-of-bounds reads are zero.
fn sample(x: i32, y: i32) -> vec4<f32> {
    if (x < 0 || y < 0 || x >= i32(p.src_w) || y >= i32(p.src_h)) {
        return vec4(0.0);
    }
    let w = src[p.src_off + u32(y) * p.src_pitch + u32(x)];
    return vec4(
        f32(w & 0xFFu),
        f32((w >> 8u) & 0xFFu),
        f32((w >> 16u) & 0xFFu),
        f32((w >> 24u) & 0xFFu),
    );
}

// Mitchell-Netravali cubic (B = C = 0.5) at distance d >= 0 (mirrors
// vacc-image's warp mitchell, f32).
fn mitchell(d: f32) -> f32 {
    if (d < 1.0) {
        return (4.5 * d * d * d - 9.0 * d * d + 5.0) / 6.0;
    } else if (d < 2.0) {
        return (-3.5 * d * d * d + 18.0 * d * d - 30.0 * d + 16.0) / 6.0;
    }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) {
        return;
    }
    let x = f32(gid.x);
    let y = f32(gid.y);
    let sx = p.m00 * x + p.m01 * y + p.m02;
    let sy = p.m10 * x + p.m11 * y + p.m12;

    var rgb: vec3<f32>;
    if (p.interp == 0u) {
        // Nearest: the floor of the source position.
        rgb = sample(i32(floor(sx)), i32(floor(sy))).xyz;
    } else if (p.interp == 1u) {
        let x0 = i32(floor(sx));
        let y0 = i32(floor(sy));
        let fx = sx - f32(x0);
        let fy = sy - f32(y0);
        let p00 = sample(x0, y0);
        let p10 = sample(x0 + 1, y0);
        let p01 = sample(x0, y0 + 1);
        let p11 = sample(x0 + 1, y0 + 1);
        // Same per-component order as the software reference:
        // top = p00*(1-fx) + p10*fx; out = top*(1-fy) + bot*fy.
        let top = p00 * (1.0 - fx) + p10 * fx;
        let bot = p01 * (1.0 - fx) + p11 * fx;
        rgb = (top * (1.0 - fy) + bot * fy).xyz;
    } else {
        // Bicubic: separable 4x4 Mitchell taps around floor(s).
        let x0 = i32(floor(sx));
        let y0 = i32(floor(sy));
        var acc = vec3(0.0);
        for (var kx = -1; kx <= 2; kx += 1) {
            let wx = mitchell(abs(sx - f32(x0 + kx)));
            if (wx == 0.0) {
                continue;
            }
            for (var ky = -1; ky <= 2; ky += 1) {
                let wy = mitchell(abs(sy - f32(y0 + ky)));
                if (wy == 0.0) {
                    continue;
                }
                acc += sample(x0 + kx, y0 + ky).xyz * (wx * wy);
            }
        }
        rgb = acc;
    }

    let r = u32(clamp(floor(rgb.x + 0.5), 0.0, 255.0));
    let g = u32(clamp(floor(rgb.y + 0.5), 0.0, 255.0));
    let b = u32(clamp(floor(rgb.z + 0.5), 0.0, 255.0));
    dst[p.dst_off + gid.y * p.dst_w + gid.x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
