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

/* SAO (§8.7.3) differential oracle — full apply_sao over a synthetic picture.
 * picW/picH: luma dimensions; ctbLog2SizeY in 5..7; subW/subH chroma
 * subsampling (1 or 2); chromaArrayType 0 = monochrome, 1 = 4:2:0.
 * planes: three u16 buffers of stride_c * compH samples each (compW/compH =
 * picW/picH divided by subW/subH for c > 0); for chromaArrayType == 0 only
 * plane_y is read and out_cb/out_cr may be NULL.
 * slice_idx: per-CTB slice index, PicSizeInCtbsY bytes; NULL = single slice.
 * slice_across_slices: num_slices u8 flags (slice_loop_filter_across_slices).
 * sao_params: flat int32 array, 24 ints per CTU in raster order:
 *   [type_idx x3, eo_class x3, band_position x3, offset_val x3x5]
 * cu_pcm / cu_bypass: per-min-CB (picW/4 * picH/4) u8 flags; NULL = all zero.
 * tile_id / ctb_addr_rs_to_ts: PicSizeInCtbsY entries each; NULL = no tiles.
 * Writes the filtered planes to out_*; returns 0 on success, -1 on bad args. */
int hevcdec_test_sao_run(
    int picW, int picH, int ctbLog2SizeY, int subW, int subH,
    int chromaArrayType, int bitDepthY, int bitDepthC,
    int sao_enabled, int pcm_filter_disabled,
    int transquant_bypass_enabled, int loop_filter_across_tiles,
    const uint8_t* tile_id, const int32_t* ctb_addr_rs_to_ts,
    const uint16_t* plane_y, const uint16_t* plane_cb, const uint16_t* plane_cr,
    int stride_y, int stride_cb, int stride_cr,
    const uint8_t* slice_idx, int num_slices,
    const uint8_t* slice_across_slices,
    const int32_t* sao_params,
    const uint8_t* cu_pcm, const uint8_t* cu_bypass,
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr);

/* Deblocking (§8.7.2) differential oracle — full apply_deblocking over a
 * synthetic picture. Plane/slice/tile conventions as in hevcdec_test_sao_run
 * (no sao_enabled; pcm_filter_disabled from the SPS).
 * sh_params: flat int32, 4 per slice:
 *   [deblocking_disabled, across_slices_enabled, beta_offset_div2, tc_offset_div2]
 * cu_fields: flat int32, 4 per min-CB (picW/4 * picH/4):
 *   [pred_mode (0 inter / 1 intra), qp_y, is_pcm, transquant_bypass]
 * motion: flat int32, 8 per 4x4 block (picW/4 * picH/4):
 *   [mvx_l0, mvy_l0, mvx_l1, mvy_l1, ref_idx_l0, ref_idx_l1, pred_flag_l0, pred_flag_l1]
 * cbf_luma / log2_tu_size / edge_v / edge_h: u8 per 4x4 block.
 * poc_l0 / poc_l1: reference picture POCs (n_ref_l0 / n_ref_l1 entries);
 * NULL = empty list. Writes the filtered planes to out_*; returns 0 on
 * success, -1 on bad args. */
int hevcdec_test_deblock_run(
    int picW, int picH, int ctbLog2SizeY, int subW, int subH,
    int chromaArrayType, int bitDepthY, int bitDepthC,
    int pcm_filter_disabled,
    int loop_filter_across_tiles,
    const uint8_t* tile_id, const int32_t* ctb_addr_rs_to_ts,
    int pps_cb_qp_offset, int pps_cr_qp_offset,
    const uint16_t* plane_y, const uint16_t* plane_cb, const uint16_t* plane_cr,
    int stride_y, int stride_cb, int stride_cr,
    const uint8_t* slice_idx, int num_slices,
    const int32_t* sh_params,
    const int32_t* cu_fields,
    const int32_t* motion,
    const uint8_t* cbf_luma, const uint8_t* log2_tu_size,
    const uint8_t* edge_v, const uint8_t* edge_h,
    const int32_t* poc_l0, int n_ref_l0,
    const int32_t* poc_l1, int n_ref_l1,
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr);

/* ---- Tier E: slice-segment replay oracle ------------------------------
 *
 * Differential oracle for the Rust coding_tree / residual_coding /
 * inter_prediction ports. Replays one picture's slice segments through the
 * C++ decode_slice_segment_data with fully synthetic reference pictures
 * (caller-provided, identical on both sides), so outputs are comparable
 * byte-for-byte without decoding prior frames.
 *
 * Flat layouts (int32 arrays):
 *   SPS_FLAT (30): picW, picH, bitDepthY, bitDepthC, chromaArrayType,
 *     ctbSizeY, minTbSizeY, picWidthInCtbsY, subW, subH,
 *     intraSmoothingDisabled, strongIntraSmoothing, minCbLog2SizeY,
 *     ctbLog2SizeY, minCbSizeY, picHeightInCtbsY, picSizeInCtbsY,
 *     minTbLog2SizeY, maxTbLog2SizeY, qpBdOffsetY, qpBdOffsetC,
 *     ampEnabled, pcmEnabled, pcmBitDepthLumaMinus1, pcmBitDepthChromaMinus1,
 *     log2MinIpcmCbSizeY, log2MaxIpcmCbSizeY, maxTrHIERarchDepthInter,
 *     maxTransformHierarchyDepthIntra, cabacBypassAlignment
 *   PPS_FLAT (48): signDataHiding, transformSkip, cuQpDelta,
 *     diffCuQpDeltaDepth, ppsCbQpOffset, ppsCrQpOffset, weightedPred,
 *     weightedBipred, transquantBypass, tilesEnabled, entropyCodingSync,
 *     loopFilterAcrossSlices, log2ParallelMergeLevelMinus2,
 *     numTileColumnsMinus1, numTileRowsMinus1, uniformSpacing,
 *     columnWidthMinus1[16] (0-padded), rowHeightMinus1[16] (0-padded)
 *   SH_FLAT (343): sliceSegmentAddress, dependentSliceSegmentFlag,
 *     sliceType (0=B,1=P,2=I), picOutputFlag, sliceTemporalMvpEnabled,
 *     sliceSaoLumaFlag, sliceSaoChromaFlag, numRefIdxL0ActiveMinus1,
 *     numRefIdxL1ActiveMinus1, mvdL1Zero, cabacInitFlag,
 *     collocatedFromL0Flag, collocatedRefIdx, fiveMinusMaxNumMergeCand,
 *     sliceQpDelta, sliceCbQpOffset, sliceCrQpOffset, sliceQpY (derived),
 *     maxNumMergeCand (derived), numEntryPointOffsets,
 *     sliceHeaderCodedSize (coded NAL offset of slice data start),
 *     lumaLog2WeightDenom, deltaChromaLog2WeightDenom,
 *     then per list(0,1) x ref(0..15): lumaWeightFlag, chromaWeightFlag,
 *     lumaWeight, lumaOffset, cbWeight, cbOffset, crWeight, crOffset
 *
 * Reference pool (n_refs entries), all content caller-generated:
 *   ref_poc[r], ref_st_ref[r] (0/1), ref_lt_ref[r] (0/1)
 *   ref_y  [r][picW * picH]            (stride = picW)
 *   ref_cb [r][compW * compH]          (stride = compW; compW = picW/subW)
 *   ref_cr [r][compW * compH]
 *   ref_motion [r][modeGridW*modeGridH*8]: per 4x4 block
 *     mvx0, mvy0, refIdx0, predFlag0, mvx1, mvy1, refIdx1, predFlag1
 *   ref_refpoc [r][2][16] (0-padded)
 * list0_idx/list1_idx: pool indices (-1 = no reference), length n_list0/n_list1.
 * col_pic_idx: pool index of the collocated picture or -1.
 *
 * Returns 0 on success; -1 bad arguments / parse failure; -2 unsupported
 * feature (scaling lists, tiles, dependent slice segments, separate colour
 * planes); -3 slice header decode failure; -4 slice data decode failure. */

#define HEVCDEC_TEST_SPS_FLAT 30
#define HEVCDEC_TEST_PPS_FLAT 48
#define HEVCDEC_TEST_SH_FLAT 343
#define HEVCDEC_TEST_MAX_SEGS 16

int hevcdec_test_parse_sps_pps(const uint8_t* sps_nal, int sps_len,
                               const uint8_t* pps_nal, int pps_len,
                               int32_t* out_sps, int32_t* out_pps);

int hevcdec_test_frame_new(
    const uint8_t* sps_nal, int sps_len,
    const uint8_t* pps_nal, int pps_len,
    int cur_poc,
    int n_refs,
    const int32_t* ref_poc, const int32_t* ref_st_ref, const int32_t* ref_lt_ref,
    const uint16_t* ref_y, const uint16_t* ref_cb, const uint16_t* ref_cr,
    const int32_t* ref_motion, const int32_t* ref_refpoc,
    const int32_t* list0_idx, int n_list0,
    const int32_t* list1_idx, int n_list1,
    int col_pic_idx, int no_backward_pred,
    int32_t* out_sps, int32_t* out_pps,
    void** out_state);

/* Decode all slice segments of one picture (concatenated raw VCL NAL bytes)
 * in the frame state created by hevcdec_test_frame_new. Grids/plane outputs
 * are the full-picture state after the last segment:
 *   out_y/out_cb/out_cr:  plane samples, stride = width
 *   out_cu_info:  [minCbsW*minCbsH*8] per min-CB: predMode, partMode,
 *                 log2CbSize, intraModeLuma, qpY, isPcm, transquantBypass,
 *                 mergeFlag
 *   out_intra_luma / out_intra_chroma: [modeGridW*modeGridH]
 *   out_motion:   [modeGridW*modeGridH*8] (same layout as ref_motion)
 *   out_cbf_luma / out_log2_tu / out_edge_v / out_edge_h: [modeGridW*modeGridH] u8
 *   out_sao_params: [ctbCount*24] per CTU: typeIdx[3], eoClass[3],
 *                   bandPosition[3], offsetVal[3][5]
 *   out_slice_idx: [ctbCount] u8
 * Per segment s (0..out_n_segments-1): out_sh[s*SH_FLAT..] and
 * out_final_bit_pos[s] = RBSP bit position after the segment's slice data. */
int hevcdec_test_frame_decode(
    void* state,
    const uint8_t* vcl_nals, int vcl_len,
    int* out_n_segments,
    int32_t* out_sh, uint32_t* out_final_bit_pos,
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr,
    int32_t* out_cu_info,
    int32_t* out_intra_luma, int32_t* out_intra_chroma,
    int32_t* out_motion,
    uint8_t* out_cbf_luma, uint8_t* out_log2_tu,
    uint8_t* out_edge_v, uint8_t* out_edge_h,
    int32_t* out_sao_params, uint8_t* out_slice_idx);

void hevcdec_test_frame_free(void* state);

#ifdef __cplusplus
}
#endif

#endif /* HEVC_TEST_API_H */
