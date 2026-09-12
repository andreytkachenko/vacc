/*
 * HEVC driver (vacc software decoder): C API over the hevc.js core.
 *
 * The Rust side owns bitstream parsing, POC computation and display-order
 * reordering (vacc-parser). This driver feeds whole access units through the
 * native decode pipeline; the hevc.js DPB keeps decoded pictures (pixels +
 * motion metadata) for reference resolution. After each access unit the final
 * filtered planes are copied into caller-provided buffers and the picture is
 * released from the DPB as soon as it is no longer a reference.
 *
 * All functions are thread-confined to a single hevcdec_context (one context
 * per decode thread).
 */
#ifndef HEVC_DRIVER_H
#define HEVC_DRIVER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hevcdec_context hevcdec_context;

/* Error codes returned by hevcdec_decode_picture (negative values). */
#define HEVCDEC_OK               0
#define HEVCDEC_ERR_GENERIC     -1
#define HEVCDEC_ERR_NO_PICTURE  -2 /* AU contained no decodable picture */
#define HEVCDEC_ERR_DECODE      -3 /* slice/transform decode failure */

/* Create a decoder context. nthreads: worker threads for WPP parallelism
 * (0 = hardware concurrency). */
hevcdec_context* hevcdec_create(int nthreads);

void hevcdec_destroy(hevcdec_context* ctx);

/* Reset all internal state (parameter sets, POC state, DPB). */
void hevcdec_reset(hevcdec_context* ctx);

/* Decode one access unit.
 *
 * data/len: Annex B bytes of the AU (start codes included). May contain VPS/SPS/
 *           PPS NALs followed by all slice segments of the picture.
 * out_*: destination planes (8/16-bit samples per bit depth). Must have at
 *        least height*stride samples per plane; strides are in samples
 *        (not bytes).
 *
 * Returns HEVCDEC_OK or a negative HEVCDEC_ERR_*. */
int hevcdec_decode_picture(hevcdec_context* ctx,
                           const uint8_t* data, size_t len,
                           uint8_t* out_y, int out_ystride,
                           uint8_t* out_u, int out_ustride,
                           uint8_t* out_v, int out_vstride);

/* POC of the last decoded picture as computed by the C++ core (for cross-check
 * against the Rust-side POC computation). Returns -1 if none. */
int hevcdec_last_pic_poc(hevcdec_context* ctx);

#ifdef __cplusplus
}
#endif

#endif /* HEVC_DRIVER_H */
