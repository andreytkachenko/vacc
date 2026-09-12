/*
 * Test-only C API: differential-testing oracles for the Rust ports of the
 * hevc.js kernels (see hevc_test_api.h). Not part of the decode pipeline.
 */

#include "hevc_test_api.h"

#include <cstring>

#include "bitstream/bitstream_reader.h"
#include "common/types.h"
#include "decoding/cabac.h"
#include "decoding/coding_tree.h"
#include "decoding/interpolation.h"
#include "decoding/intra_prediction.h"
#include "decoding/syntax_elements.h"
#include "decoding/transform.h"
#include "syntax/pps.h"
#include "syntax/slice_header.h"
#include "syntax/sps.h"

namespace {

hevc::ScalingListData make_scaling_list(const uint8_t* sl_data, const uint8_t* sl_dc) {
    hevc::ScalingListData sl;
    if (sl_data) {
        std::memcpy(sl.scaling_list[0][0].data(), sl_data, sizeof(sl.scaling_list));
    } else {
        sl.set_defaults();
    }
    if (sl_dc) {
        std::memcpy(sl.scaling_list_dc[0].data(), sl_dc, sizeof(sl.scaling_list_dc));
    } else {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 6; j++)
                sl.scaling_list_dc[i][j] = 16;
    }
    return sl;
}

// Picture holding only component `cIdx`, filled from a row-major plane.
hevc::Picture make_test_picture(int cIdx, int w, int h, int stride, const uint16_t* plane) {
    hevc::Picture pic;
    pic.width[cIdx] = w;
    pic.height[cIdx] = h;
    pic.stride[cIdx] = stride;
    pic.planes[cIdx].assign(plane, plane + static_cast<size_t>(stride) * h);
    return pic;
}

} // namespace

extern "C" {

int hevcdec_test_transform_inverse(int log2TrafoSize, int cIdx, int is_intra,
                                   int transform_skip, int bit_depth,
                                   const int16_t* scaled, int16_t* residual) {
    if (!scaled || !residual || log2TrafoSize < 2 || log2TrafoSize > 5) return -1;
    hevc::perform_transform_inverse(log2TrafoSize, cIdx, is_intra != 0,
                                    transform_skip != 0, bit_depth, scaled, residual);
    return 0;
}

int hevcdec_test_dequant(int bit_depth_luma, int bit_depth_chroma,
                         int cIdx, int log2TrafoSize, int qp,
                         int cu_is_intra,
                         int use_scaling_list, int pps_scaling_list_present,
                         const uint8_t* sl_data, const uint8_t* sl_dc,
                         const uint8_t* pps_sl_data, const uint8_t* pps_sl_dc,
                         const int16_t* coefficients, int16_t* scaled) {
    if (!coefficients || !scaled || log2TrafoSize < 2 || log2TrafoSize > 5) return -1;

    hevc::SPS sps;
    sps.BitDepthY = bit_depth_luma;
    sps.BitDepthC = bit_depth_chroma;
    sps.MinCbLog2SizeY = 3; // cu_at(x0, y0) with (x0, y0) = (0, 0) -> grid index 0
    sps.scaling_list_enabled_flag = use_scaling_list != 0;
    sps.scaling_list_data = make_scaling_list(sl_data, sl_dc);

    hevc::PPS pps;
    pps.pps_scaling_list_data_present_flag = pps_scaling_list_present != 0;
    pps.scaling_list_data = make_scaling_list(pps_sl_data ? pps_sl_data : sl_data,
                                              pps_sl_dc ? pps_sl_dc : sl_dc);

    hevc::CUInfo cu;
    cu.pred_mode = cu_is_intra ? hevc::PredMode::MODE_INTRA : hevc::PredMode::MODE_INTER;

    hevc::DecodingContext ctx;
    ctx.sps = &sps;
    ctx.pps = &pps;
    ctx.cu_info = &cu;
    ctx.cu_info_stride = 1;

    hevc::perform_dequant(ctx, 0, 0, log2TrafoSize, cIdx, qp, coefficients, scaled);
    return 0;
}

int hevcdec_test_interpolate_luma(const uint16_t* plane, int picW, int picH, int stride,
                                  int xInt, int yInt, int xFrac, int yFrac,
                                  int nPbW, int nPbH, int bitDepth, int16_t* pred) {
    if (!plane || !pred) return -1;
    hevc::Picture pic = make_test_picture(0, picW, picH, stride, plane);
    hevc::interpolate_luma(pic, xInt, yInt, xFrac, yFrac, nPbW, nPbH, bitDepth, pred);
    return 0;
}

int hevcdec_test_interpolate_chroma(const uint16_t* plane, int cIdx, int picW, int picH,
                                    int stride, int xInt, int yInt, int xFrac, int yFrac,
                                    int nPbWC, int nPbHC, int bitDepth, int16_t* pred) {
    if (!plane || !pred || (cIdx != 1 && cIdx != 2)) return -1;
    hevc::Picture pic = make_test_picture(cIdx, picW, picH, stride, plane);
    hevc::interpolate_chroma(pic, cIdx, xInt, yInt, xFrac, yFrac, nPbWC, nPbHC, bitDepth, pred);
    return 0;
}

int hevcdec_test_weighted_pred_default(const int16_t* predL0, const int16_t* predL1,
                                       int flagL0, int flagL1, int nSamples, int bitDepth,
                                       int16_t* output) {
    if (!predL0 || !predL1 || !output) return -1;
    hevc::weighted_pred_default(predL0, predL1, flagL0 != 0, flagL1 != 0,
                                nSamples, bitDepth, output);
    return 0;
}

int hevcdec_test_weighted_pred_explicit(const int16_t* predL0, const int16_t* predL1,
                                        int flagL0, int flagL1, int refIdxL0, int refIdxL1,
                                        int cIdx, int nSamples, int bitDepth,
                                        uint32_t luma_log2_weight_denom,
                                        int32_t delta_chroma_log2_weight_denom,
                                        const int16_t* w_luma, const int16_t* o_luma,
                                        const int16_t* w_chroma, const int16_t* o_chroma,
                                        int16_t* output) {
    if (!predL0 || !predL1 || !output) return -1;
    hevc::PredWeightTable pwt;
    pwt.luma_log2_weight_denom = luma_log2_weight_denom;
    pwt.delta_chroma_log2_weight_denom = delta_chroma_log2_weight_denom;
    for (int list = 0; list < 2; list++) {
        auto& dst = (list == 0) ? pwt.l0 : pwt.l1;
        for (int r = 0; r < 16; r++) {
            dst[r].luma_weight = w_luma[list * 16 + r];
            dst[r].luma_offset = o_luma[list * 16 + r];
            for (int c = 0; c < 2; c++) {
                dst[r].chroma_weight[c] = w_chroma[(list * 2 + c) * 16 + r];
                dst[r].chroma_offset[c] = o_chroma[(list * 2 + c) * 16 + r];
            }
        }
    }
    hevc::weighted_pred_explicit(predL0, predL1, flagL0 != 0, flagL1 != 0,
                                 refIdxL0, refIdxL1, cIdx, nSamples, bitDepth, pwt, output);
    return 0;
}

int hevcdec_test_intra_predict(int picW, int picH, int ctbSizeY, int minTbSizeY,
                               int bitDepth, int chromaArrayType, int subW, int subH,
                               int intra_smoothing_disabled, int strong_smoothing_enabled,
                               const uint16_t* plane, int stride,
                               int x0, int y0, int log2PredSize, int cIdx, int intra_mode,
                               int16_t* pred) {
    if (!plane || !pred || cIdx < 0 || cIdx > 2) return -1;

    int compW = (cIdx > 0) ? picW / subW : picW;
    int compH = (cIdx > 0) ? picH / subH : picH;

    hevc::SPS sps;
    sps.pic_width_in_luma_samples = static_cast<uint32_t>(picW);
    sps.pic_height_in_luma_samples = static_cast<uint32_t>(picH);
    sps.CtbSizeY = ctbSizeY;
    sps.MinTbSizeY = minTbSizeY;
    sps.PicWidthInCtbsY = (picW + ctbSizeY - 1) / ctbSizeY;
    sps.BitDepthY = bitDepth;
    sps.BitDepthC = bitDepth;
    sps.ChromaArrayType = chromaArrayType;
    sps.SubWidthC = subW;
    sps.SubHeightC = subH;
    sps.intra_smoothing_disabled_flag = intra_smoothing_disabled != 0;
    sps.strong_intra_smoothing_enabled_flag = strong_smoothing_enabled != 0;

    hevc::PPS pps;  // TileId / CtbAddrRsToTs empty -> no tile checks

    hevc::Picture pic = make_test_picture(cIdx, compW, compH, stride, plane);

    hevc::DecodingContext ctx;
    ctx.sps = &sps;
    ctx.pps = &pps;
    ctx.pic = &pic;

    hevc::perform_intra_prediction(ctx, x0, y0, log2PredSize, cIdx, intra_mode, pred);
    return 0;
}

int hevcdec_test_cabac_run(const uint8_t* data, int len_bytes,
                           int sliceType, int qp, int cabac_init_flag,
                           const uint8_t* ops, const int32_t* args, int n_ops,
                           int32_t* out, int max_out,
                           uint16_t* final_range, uint16_t* final_offset,
                           uint8_t* final_ctx) {
    if (!data || !ops || !args || !out || n_ops <= 0 || len_bytes <= 0) return -1;

    hevc::BitstreamReader bs(data, static_cast<size_t>(len_bytes));
    hevc::CabacEngine cabac;
    cabac.init_decoder(bs);
    cabac.init_contexts(sliceType, qp, cabac_init_flag != 0);

    int n_out = 0;
    for (int i = 0; i < n_ops; i++) {
        int a = args[i];
        switch (ops[i]) {
            case 0: out[n_out++] = cabac.decode_decision(a & 127); break;
            case 1: out[n_out++] = cabac.decode_bypass(); break;
            case 2: out[n_out++] = cabac.decode_terminate(); break;
            case 3: out[n_out++] = cabac.decode_bypass_bins(a & 15); break;
            case 4: cabac.align_bypass(); break;
            case 5: out[n_out++] = hevc::decode_sao_type_idx(cabac); break;
            case 6: out[n_out++] = hevc::decode_split_cu_flag(cabac, a & 3); break;
            case 7: out[n_out++] = hevc::decode_cu_skip_flag(cabac, a & 3); break;
            case 8: {
                int predMode = a & 1;
                int log2CbSize = (a >> 2) & 7;
                int log2MinCbSize = (a >> 5) & 7;
                bool amp = (a >> 8) & 1;
                out[n_out++] = hevc::decode_part_mode(cabac,
                    static_cast<hevc::PredMode>(predMode), log2CbSize, log2MinCbSize, amp);
                break;
            }
            case 9: out[n_out++] = hevc::decode_intra_chroma_pred_mode(cabac); break;
            case 10: out[n_out++] = hevc::decode_merge_idx(cabac, a & 7); break;
            case 11: {
                int nPbW = a & 127;
                int nPbH = (a >> 7) & 127;
                int ctDepth = (a >> 14) & 15;
                out[n_out++] = hevc::decode_inter_pred_idc(cabac, nPbW, nPbH, ctDepth);
                break;
            }
            case 12: out[n_out++] = hevc::decode_ref_idx(cabac, a & 31); break;
            case 13: {
                hevc::MV mv = hevc::decode_mvd(cabac);
                out[n_out++] = mv.x;
                out[n_out++] = mv.y;
                break;
            }
            case 14: out[n_out++] = hevc::decode_split_transform_flag(cabac, a & 7); break;
            case 15: out[n_out++] = hevc::decode_cbf_luma(cabac, a & 3); break;
            case 16: out[n_out++] = hevc::decode_cbf_chroma(cabac, a & 7); break;
            case 17: out[n_out++] = hevc::decode_cu_qp_delta(cabac); break;
            case 18: out[n_out++] = hevc::decode_transform_skip_flag(cabac, a & 1); break;
            case 19: {
                // ctxOffset must be a valid base (CTX_LAST_SIG_COEFF_X=42 / _Y=60)
                // and log2TrafoSize a valid last_sig size (2..5) to stay in bounds.
                int ctxOffset = (a & 1) ? 60 : 42;
                int cIdx = (a >> 1) & 1;
                int log2TrafoSize = 2 + ((a >> 2) & 3);
                out[n_out++] = hevc::decode_last_sig_coeff_prefix(cabac, ctxOffset,
                                                                  cIdx, log2TrafoSize);
                break;
            }
            case 20: out[n_out++] = hevc::decode_last_sig_coeff_suffix(cabac, a & 15); break;
            case 21: out[n_out++] = hevc::decode_coded_sub_block_flag(cabac, a & 3); break;
            case 22: out[n_out++] = hevc::decode_sig_coeff_flag(cabac, a & 41); break;
            case 23: {
                int ctxSet = a & 1;
                int greater1Ctx = (a >> 1) & 3; // 0..3 keeps ctxIdx < NUM_CABAC_CONTEXTS
                int cIdx = (a >> 5) & 1;
                out[n_out++] = hevc::decode_coeff_abs_level_greater1_flag(cabac,
                                                                          ctxSet, greater1Ctx, cIdx);
                break;
            }
            case 24: {
                int ctxSet = a & 1;
                int cIdx = (a >> 1) & 1;
                out[n_out++] = hevc::decode_coeff_abs_level_greater2_flag(cabac, ctxSet, cIdx);
                break;
            }
            case 25: out[n_out++] = hevc::decode_coeff_abs_level_remaining(cabac, a & 7); break;
            default: return -1;
        }
        if (n_out > max_out) return -1;
    }

    *final_range = cabac.dbg_range();
    *final_offset = cabac.dbg_offset();
    for (int i = 0; i < hevc::NUM_CABAC_CONTEXTS; i++) {
        final_ctx[2 * i] = cabac.context(i).pStateIdx;
        final_ctx[2 * i + 1] = cabac.context(i).valMps;
    }
    return n_out;
}

} // extern "C"
