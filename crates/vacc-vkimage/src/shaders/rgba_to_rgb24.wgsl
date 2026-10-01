// RGBA32 words -> packed RGB24 bytes for the zero-copy device path.
//
// One invocation per output word (4 bytes). Byte k of output word w belongs
// to pixel (w*4+k)/3, channel (w*4+k)%3, so every output word is written by
// exactly one invocation and odd pixel counts need no tail handling.

struct Params {
    count: u32,
    src_off: u32,
    dst_off: u32,
    total_bytes: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= p.count) {
        return;
    }
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        let rel = w * 4u + k;
        if (rel >= p.total_bytes) {
            break;
        }
        let px = rel / 3u;
        let ch = rel % 3u;
        let sw = src[p.src_off + px];
        var v = sw & 0xFFu;
        if (ch == 1u) {
            v = (sw >> 8u) & 0xFFu;
        } else if (ch == 2u) {
            v = (sw >> 16u) & 0xFFu;
        }
        out = out | (v << (8u * k));
    }
    dst[(p.dst_off + w * 4u) / 4u] = out;
}
