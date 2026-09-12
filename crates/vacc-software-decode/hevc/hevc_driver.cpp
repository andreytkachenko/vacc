/*
 * HEVC driver (vacc software decoder): C API over the hevc.js core.
 *
 * The hevc.js core is MIT-licensed (see hevc/LICENSE); this file is the only
 * glue between it and the Rust side. The Rust side owns bitstream parsing,
 * POC computation and display-order reordering (vacc-parser). This driver
 * feeds whole access units through the native decode pipeline; the hevc.js
 * DPB keeps decoded pictures (pixels + motion metadata) for reference
 * resolution. After each access unit the final filtered planes are copied
 * into caller-provided buffers; the picture's output flag is then cleared so
 * the DPB can evict it once it is no longer a reference (eviction happens in
 * DPB::alloc_picture, which runs before any picture that might still use it
 * as a reference).
 */

#include "hevc_driver.h"

#include <memory>

#include "common/picture.h"
#include "decoding/decoder.h"

// Complete definition of the opaque C type declared in hevc_driver.h. The C++
// tag `hevcdec_context` is the same entity as the C typedef, so no casts are
// needed at the FFI boundary.
struct hevcdec_context {
    std::unique_ptr<hevc::Decoder> dec;
    int nthreads = 0;
};

namespace {

// Copy one plane (uint16_t samples, stride in samples) into the caller's
// 8-bit or 16-bit destination buffer (stride in samples).
void copy_plane(const hevc::Picture& pic, int c, uint8_t* dst, int dst_stride, int bps) {
    const auto& src = pic.planes[c];
    const int w = pic.width[c];
    const int h = pic.height[c];
    const int ss = pic.stride[c];
    if (bps == 1) {
        for (int y = 0; y < h; y++) {
            const uint16_t* s = &src[static_cast<size_t>(y) * ss];
            uint8_t* d = dst + static_cast<size_t>(y) * dst_stride;
            for (int x = 0; x < w; x++) {
                d[x] = static_cast<uint8_t>(s[x]);
            }
        }
    } else {
        for (int y = 0; y < h; y++) {
            const uint16_t* s = &src[static_cast<size_t>(y) * ss];
            uint16_t* d = reinterpret_cast<uint16_t*>(dst + static_cast<size_t>(y) * dst_stride * 2);
            for (int x = 0; x < w; x++) {
                d[x] = s[x];
            }
        }
    }
}

} // namespace

extern "C" {

hevcdec_context* hevcdec_create(int nthreads) {
    auto* ctx = new hevcdec_context();
    ctx->nthreads = nthreads;
    ctx->dec = std::make_unique<hevc::Decoder>(nthreads);
    return ctx;
}

void hevcdec_destroy(hevcdec_context* ctx) {
    delete ctx;
}

void hevcdec_reset(hevcdec_context* ctx) {
    // Decoder owns a non-movable thread pool: rebuild the whole object.
    ctx->dec = std::make_unique<hevc::Decoder>(ctx->nthreads);
}

int hevcdec_decode_picture(hevcdec_context* ctx,
                           const uint8_t* data, size_t len,
                           uint8_t* out_y, int out_ystride,
                           uint8_t* out_u, int out_ustride,
                           uint8_t* out_v, int out_vstride) {
    if (!ctx || !ctx->dec || !data || len == 0) {
        return HEVCDEC_ERR_GENERIC;
    }

    // A decoded picture replaces current_pic; use that to detect "no picture
    // in AU" (DPB size alone is unreliable: eviction can shrink it).
    const hevc::Picture* pic_before = ctx->dec->dpb().current_pic();

    const hevc::DecodeStatus status = ctx->dec->decode(data, len);
    if (status == hevc::DecodeStatus::ERROR) {
        return HEVCDEC_ERR_DECODE;
    }

    const hevc::Picture* pic = ctx->dec->dpb().current_pic();
    if (!pic || pic == pic_before) {
        return HEVCDEC_ERR_NO_PICTURE;
    }

    copy_plane(*pic, 0, out_y, out_ystride, pic->bit_depth_luma > 8 ? 2 : 1);
    if (pic->chroma_format != hevc::ChromaFormat::MONOCHROME) {
        const int cbps = pic->bit_depth_chroma > 8 ? 2 : 1;
        copy_plane(*pic, 1, out_u, out_ustride, cbps);
        copy_plane(*pic, 2, out_v, out_vstride, cbps);
    }

    // The Rust side owns a copy of the pixels: allow the DPB to evict this
    // picture once it stops being a reference.
    const_cast<hevc::Picture*>(pic)->needed_for_output = false;

    return HEVCDEC_OK;
}

int hevcdec_last_pic_poc(hevcdec_context* ctx) {
    if (!ctx || !ctx->dec) {
        return -1;
    }
    const hevc::Picture* pic = ctx->dec->dpb().current_pic();
    return pic ? pic->poc : -1;
}

} // extern "C"
