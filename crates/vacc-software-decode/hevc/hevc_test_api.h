/*
 * Test-only C API: differential-testing oracles for the Rust ports of the
 * hevc.js kernels. These exports let Rust unit tests invoke the original C++
 * implementation on synthetic inputs and compare outputs byte-for-byte.
 *
 * They are NOT part of the decode pipeline (hevc_driver.h) and must never be
 * called from production code.
 */
#ifndef HEVC_TEST_API_H
#define HEVC_TEST_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Inverse transform (§8.6.4) — direct call into perform_transform_inverse.
 * scaled/residual hold (1<<log2TrafoSize)^2 int16 samples each.
 * Returns 0 on success, -1 on invalid arguments. */
int hevcdec_test_transform_inverse(int log2TrafoSize, int cIdx, int is_intra,
                                   int transform_skip, int bit_depth,
                                   const int16_t* scaled, int16_t* residual);

/* Dequantization (§8.6.3) with a minimal DecodingContext.
 * sl_data/sl_dc: SPS scaling list (4*6*64 / 2*6 bytes); NULL = spec defaults.
 * pps_sl_data/pps_sl_dc: PPS scaling list; NULL = same as SPS.
 * Returns 0 on success, -1 on invalid arguments. */
int hevcdec_test_dequant(int bit_depth_luma, int bit_depth_chroma,
                         int cIdx, int log2TrafoSize, int qp,
                         int cu_is_intra,
                         int use_scaling_list, int pps_scaling_list_present,
                         const uint8_t* sl_data, const uint8_t* sl_dc,
                         const uint8_t* pps_sl_data, const uint8_t* pps_sl_dc,
                         const int16_t* coefficients, int16_t* scaled);

/* Luma 8-tap interpolation (§8.5.3.3.3). plane: reference luma samples
 * (picW x picH, row-major with given stride, uint16). xFrac/yFrac in {0..3}.
 * pred: output nPbW*nPbH int16 samples in extended precision. */
int hevcdec_test_interpolate_luma(const uint16_t* plane, int picW, int picH, int stride,
                                 int xInt, int yInt, int xFrac, int yFrac,
                                 int nPbW, int nPbH, int bitDepth, int16_t* pred);

/* Chroma 4-tap interpolation (§8.5.3.3.3). plane: reference chroma samples for
 * component cIdx (picW x picH row-major with stride). xFrac/yFrac in {0..7}. */
int hevcdec_test_interpolate_chroma(const uint16_t* plane, int cIdx, int picW, int picH,
                                   int stride, int xInt, int yInt, int xFrac, int yFrac,
                                   int nPbWC, int nPbHC, int bitDepth, int16_t* pred);

/* Default weighted sample prediction (§8.5.3.3.4.2). */
int hevcdec_test_weighted_pred_default(const int16_t* predL0, const int16_t* predL1,
                                      int flagL0, int flagL1, int nSamples, int bitDepth,
                                      int16_t* output);

/* Explicit weighted sample prediction (§8.5.3.3.4.3).
 * Weight tables as flat arrays:
 *   w_luma/o_luma:  [2 lists][16 refs]
 *   w_chroma/o_chroma: [2 lists][2 Cb/Cr][16 refs] */
int hevcdec_test_weighted_pred_explicit(const int16_t* predL0, const int16_t* predL1,
                                       int flagL0, int flagL1, int refIdxL0, int refIdxL1,
                                       int cIdx, int nSamples, int bitDepth,
                                       uint32_t luma_log2_weight_denom,
                                       int32_t delta_chroma_log2_weight_denom,
                                       const int16_t* w_luma, const int16_t* o_luma,
                                       const int16_t* w_chroma, const int16_t* o_chroma,
                                       int16_t* output);

/* Full intra prediction path (§8.4.4.2) with a minimal DecodingContext:
 * reference sample construction (availability + substitution), optional
 * smoothing, and mode dispatch. Single slice, no tiles.
 * plane: reconstructed samples of component cIdx (component dims derived from
 * picW/picH and subW/subH), row-major with given stride.
 * pred: output nTbS*nTbS int16 samples. */
int hevcdec_test_intra_predict(int picW, int picH, int ctbSizeY, int minTbSizeY,
                              int bitDepth, int chromaArrayType, int subW, int subH,
                              int intra_smoothing_disabled, int strong_smoothing_enabled,
                              const uint16_t* plane, int stride,
                              int x0, int y0, int log2PredSize, int cIdx, int intra_mode,
                              int16_t* pred);

/* CABAC engine + syntax element differential oracle.
 * Initializes the engine (init_decoder from bit 0 of `data`, then
 * init_contexts with sliceType/qp/cabac_init_flag) and executes `n_ops`
 * operations in order:
 *   ops[i]  = op code, args[i] = per-op argument
 *     0  decision        arg = ctxIdx (0..154)          out: 1 bin
 *     1  bypass                                          out: 1 bin
 *     2  terminate                                       out: 1 bin
 *     3  bypass_bits   arg = numBins (1..16)             out: packed value
 *     4  align_bypass                                      out: none
 *     5  sao_type_idx                                      out: 0..2
 *     6  split_cu_flag   arg = ctxInc                      out: 1 bin
 *     7  cu_skip_flag    arg = ctxInc                      out: 1 bin
 *     8  part_mode       arg = predMode | log2CbSize<<2 |
 *                                 log2MinCbSize<<5 | amp<<8  out: PartMode int
 *     9  intra_chroma_pred_mode                            out: 0..4
 *     10 merge_idx       arg = maxNumMergeCand             out: idx
 *     11 inter_pred_idc  arg = nPbW | nPbH<<7 | ctDepth<<14  out: 0..2
 *     12 ref_idx         arg = numRefIdxActive             out: idx
 *     13 mvd                                             out: mv.x, mv.y (2)
 *     14 split_transform_flag arg = log2TrafoSize          out: 1 bin
 *     15 cbf_luma        arg = trafoDepth                  out: 1 bin
 *     16 cbf_chroma      arg = trafoDepth                  out: 1 bin
 *     17 cu_qp_delta                                       out: signed delta
 *     18 transform_skip_flag arg = cIdx                    out: 1 bin
 *     19 last_sig_coeff_prefix arg = ctxOffset | cIdx<<8 |
 *                                 log2TrafoSize<<9          out: prefix
 *     20 last_sig_coeff_suffix arg = prefix                out: suffix
 *     21 coded_sub_block_flag arg = ctxInc                 out: 1 bin
 *     22 sig_coeff_flag  arg = ctxInc                      out: 1 bin
 *     23 coeff_abs_level_greater1_flag arg = ctxSet |
 *                                 greater1Ctx<<1 | cIdx<<5  out: 1 bin
 *     24 coeff_abs_level_greater2_flag arg = ctxSet | cIdx<<1 out: 1 bin
 *     25 coeff_abs_level_remaining arg = cRiceParam        out: level
 * Returns the number of values written to `out` (must fit in max_out).
 * final_range/final_offset/final_ctx (NUM_CABAC_CONTEXTS*2 bytes,
 * interleaved pStateIdx/valMps) capture the terminal engine state. */
int hevcdec_test_cabac_run(const uint8_t* data, int len_bytes,
                           int sliceType, int qp, int cabac_init_flag,
                           const uint8_t* ops, const int32_t* args, int n_ops,
                           int32_t* out, int max_out,
                           uint16_t* final_range, uint16_t* final_offset,
                           uint8_t* final_ctx);

#ifdef __cplusplus
}
#endif

#endif /* HEVC_TEST_API_H */
