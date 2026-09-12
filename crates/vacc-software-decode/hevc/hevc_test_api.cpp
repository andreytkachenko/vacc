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
#include "decoding/dpb.h"
#include "decoding/inter_prediction.h"
#include "decoding/interpolation.h"
#include "decoding/intra_prediction.h"
#include "decoding/syntax_elements.h"
#include "decoding/transform.h"
#include "filters/deblocking.h"
#include "filters/sao.h"
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

// SPS shared by the SAO and deblocking oracles. Min CB/TB are fixed at 4x4.
hevc::SPS make_filter_sps(int picW, int picH, int ctbLog2SizeY, int subW, int subH,
                          int chromaArrayType, int bitDepthY, int bitDepthC,
                          int pcm_filter_disabled) {
    hevc::SPS sps;
    int ctbSize = 1 << ctbLog2SizeY;
    sps.pic_width_in_luma_samples = static_cast<uint32_t>(picW);
    sps.pic_height_in_luma_samples = static_cast<uint32_t>(picH);
    sps.CtbSizeY = ctbSize;
    sps.CtbLog2SizeY = ctbLog2SizeY;
    sps.MinCbLog2SizeY = 2;
    sps.MinCbSizeY = 4;
    sps.MinTbLog2SizeY = 2;
    sps.MinTbSizeY = 4;
    sps.SubWidthC = subW;
    sps.SubHeightC = subH;
    sps.ChromaArrayType = chromaArrayType;
    sps.BitDepthY = bitDepthY;
    sps.BitDepthC = bitDepthC;
    sps.pcm_loop_filter_disabled_flag = pcm_filter_disabled != 0;
    sps.PicWidthInCtbsY = (picW + ctbSize - 1) / ctbSize;
    sps.PicHeightInCtbsY = (picH + ctbSize - 1) / ctbSize;
    // apply_sao's anySao quick check iterates PicSizeInCtbsY; the real parser
    // derives it in SPS::parse, but the test SPS skips parsing.
    sps.PicSizeInCtbsY = sps.PicWidthInCtbsY * sps.PicHeightInCtbsY;
    return sps;
}

// PPS shared by the SAO and deblocking oracles.
hevc::PPS make_filter_pps(int loop_filter_across_tiles, int pps_cb_qp_offset,
                          int pps_cr_qp_offset, const uint8_t* tile_id,
                          const int32_t* ctb_addr_rs_to_ts, int numCtbs) {
    hevc::PPS pps;
    pps.loop_filter_across_tiles_enabled_flag = loop_filter_across_tiles != 0;
    pps.pps_cb_qp_offset = pps_cb_qp_offset;
    pps.pps_cr_qp_offset = pps_cr_qp_offset;
    if (tile_id && ctb_addr_rs_to_ts) {
        for (int i = 0; i < numCtbs; i++) {
            pps.TileId.push_back(tile_id[i]);
            pps.CtbAddrRsToTs.push_back(ctb_addr_rs_to_ts[i]);
        }
    }
    return pps;
}

// Picture with up to three planes filled from row-major buffers.
hevc::Picture make_filter_picture(int picW, int picH, int subW, int subH,
                                  int chromaArrayType,
                                  const uint16_t* plane_y, int stride_y,
                                  const uint16_t* plane_cb, int stride_cb,
                                  const uint16_t* plane_cr, int stride_cr) {
    hevc::Picture pic;
    auto fill = [&](int c, const uint16_t* plane, int stride) {
        int w = (c == 0) ? picW : picW / subW;
        int h = (c == 0) ? picH : picH / subH;
        pic.width[c] = w;
        pic.height[c] = h;
        pic.stride[c] = stride;
        if (plane && w > 0 && h > 0)
            pic.planes[c].assign(plane, plane + static_cast<size_t>(stride) * h);
    };
    fill(0, plane_y, stride_y);
    if (chromaArrayType != 0) {
        fill(1, plane_cb, stride_cb);
        fill(2, plane_cr, stride_cr);
    }
    return pic;
}

// Per-min-CB (4x4) CU grid from flat [pred_mode, qp_y, is_pcm, bypass] quads.
std::vector<hevc::CUInfo> make_cu_grid(int picW, int picH, const int32_t* cu_fields) {
    int n = (picW / 4) * (picH / 4);
    std::vector<hevc::CUInfo> cus(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        const int32_t* f = cu_fields + 4 * i;
        cus[i].pred_mode = (f[0] == 1) ? hevc::PredMode::MODE_INTRA
                                       : hevc::PredMode::MODE_INTER;
        cus[i].qp_y = f[1];
        cus[i].is_pcm = f[2] != 0;
        cus[i].cu_transquant_bypass = f[3] != 0;
    }
    return cus;
}

// Copy (possibly filtered) planes back out to the caller's buffers.
void copy_planes_out(const hevc::Picture& pic, int chromaArrayType,
                     int stride_y, int stride_cb, int stride_cr,
                     int picW, int picH, int subW, int subH,
                     uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr) {
    std::memcpy(out_y, pic.planes[0].data(),
                static_cast<size_t>(stride_y) * picH * sizeof(uint16_t));
    if (chromaArrayType != 0) {
        int ch = picH / subH;
        std::memcpy(out_cb, pic.planes[1].data(),
                    static_cast<size_t>(stride_cb) * ch * sizeof(uint16_t));
        std::memcpy(out_cr, pic.planes[2].data(),
                    static_cast<size_t>(stride_cr) * ch * sizeof(uint16_t));
    }
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
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr) {
    if (picW <= 0 || picH <= 0 || ctbLog2SizeY < 5 || ctbLog2SizeY > 7 ||
        !plane_y || !out_y || !sao_params)
        return -1;

    hevc::SPS sps = make_filter_sps(picW, picH, ctbLog2SizeY, subW, subH,
                                    chromaArrayType, bitDepthY, bitDepthC,
                                    pcm_filter_disabled);
    sps.sample_adaptive_offset_enabled_flag = sao_enabled != 0;

    int numCtbs = sps.PicWidthInCtbsY * sps.PicHeightInCtbsY;
    hevc::PPS pps = make_filter_pps(loop_filter_across_tiles, 0, 0, tile_id,
                                    ctb_addr_rs_to_ts, numCtbs);
    pps.transquant_bypass_enabled_flag = transquant_bypass_enabled != 0;

    hevc::Picture pic = make_filter_picture(picW, picH, subW, subH, chromaArrayType,
                                            plane_y, stride_y, plane_cb, stride_cb,
                                            plane_cr, stride_cr);

    // Per-min-CB PCM/bypass flags (NULL input = all zero).
    int minCbsW = picW / 4;
    std::vector<hevc::CUInfo> cus(static_cast<size_t>(minCbsW) * (picH / 4));
    for (int i = 0; i < static_cast<int>(cus.size()); i++) {
        if (cu_pcm) cus[i].is_pcm = cu_pcm[i] != 0;
        if (cu_bypass) cus[i].cu_transquant_bypass = cu_bypass[i] != 0;
    }

    // Per-slice headers (only the across-slices flag matters for SAO).
    std::vector<hevc::SliceHeader> shs(num_slices);
    for (int s = 0; s < num_slices; s++)
        shs[s].slice_loop_filter_across_slices_enabled_flag = slice_across_slices[s] != 0;

    std::vector<hevc::DecodingContext::SaoParams> saoParams(numCtbs);
    for (int i = 0; i < numCtbs; i++) {
        const int32_t* p = sao_params + 24 * i;
        auto& sp = saoParams[i];
        for (int c = 0; c < 3; c++) {
            sp.sao_type_idx[c] = p[c];
            sp.sao_eo_class[c] = p[3 + c];
            sp.sao_band_position[c] = p[6 + c];
            for (int k = 0; k < 5; k++)
                sp.sao_offset_val[c][k] = p[9 + c * 5 + k];
        }
    }

    std::vector<hevc::SliceHeader*> shPtrs(num_slices);
    for (int s = 0; s < num_slices; s++) shPtrs[s] = &shs[s];

    std::vector<uint16_t> backup[3];
    hevc::SliceHeader fallbackSh;  // sh_at_ctb fallback when slice_idx is NULL
    // DecodingContext::slice_idx is non-const (decoder-owned); copy it.
    auto sliceIdxCopy = std::vector<uint8_t>();
    if (slice_idx) sliceIdxCopy.assign(slice_idx, slice_idx + numCtbs);

    hevc::DecodingContext ctx;
    ctx.sps = &sps;
    ctx.pps = &pps;
    ctx.pic = &pic;
    ctx.sh = &fallbackSh;
    ctx.cu_info = cus.data();
    ctx.cu_info_stride = minCbsW;
    ctx.sao_params = saoParams.data();
    ctx.sao_params_stride = sps.PicWidthInCtbsY;
    ctx.sao_backup = backup;
    if (slice_idx) {
        ctx.slice_idx = sliceIdxCopy.data();
        ctx.slice_headers = shPtrs.data();
        ctx.num_slices = num_slices;
    }

    hevc::apply_sao(ctx);

    copy_planes_out(pic, chromaArrayType, stride_y, stride_cb, stride_cr,
                    picW, picH, subW, subH, out_y, out_cb, out_cr);
    return 0;
}

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
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr) {
    if (picW <= 0 || picH <= 0 || ctbLog2SizeY < 5 || ctbLog2SizeY > 7 ||
        !plane_y || !out_y || !sh_params || !cu_fields || !motion ||
        !cbf_luma || !log2_tu_size || !edge_v || !edge_h)
        return -1;

    hevc::SPS sps = make_filter_sps(picW, picH, ctbLog2SizeY, subW, subH,
                                    chromaArrayType, bitDepthY, bitDepthC,
                                    pcm_filter_disabled);

    int numCtbs = sps.PicWidthInCtbsY * sps.PicHeightInCtbsY;
    hevc::PPS pps = make_filter_pps(loop_filter_across_tiles, pps_cb_qp_offset,
                                    pps_cr_qp_offset, tile_id, ctb_addr_rs_to_ts, numCtbs);

    hevc::Picture pic = make_filter_picture(picW, picH, subW, subH, chromaArrayType,
                                            plane_y, stride_y, plane_cb, stride_cb,
                                            plane_cr, stride_cr);

    auto cus = make_cu_grid(picW, picH, cu_fields);

    // DecodingContext grid fields are non-const (the decoder writes them);
    // copy the const test inputs.
    int minCbsW = picW / 4;
    int nTb = minCbsW * (picH / 4);
    auto copyGrid = [nTb](const uint8_t* src) {
        return std::vector<uint8_t>(src, src + static_cast<size_t>(nTb));
    };
    auto cbfGrid = copyGrid(cbf_luma);
    auto tuGrid = copyGrid(log2_tu_size);
    auto edgeV = copyGrid(edge_v);
    auto edgeH = copyGrid(edge_h);
    auto sliceIdxCopy = std::vector<uint8_t>();
    if (slice_idx) sliceIdxCopy.assign(slice_idx, slice_idx + numCtbs);
    std::vector<hevc::PUMotionInfo> mi(static_cast<size_t>(nTb));
    for (int i = 0; i < nTb; i++) {
        const int32_t* m = motion + 8 * i;
        mi[i].mv[0] = {static_cast<int16_t>(m[0]), static_cast<int16_t>(m[1])};
        mi[i].mv[1] = {static_cast<int16_t>(m[2]), static_cast<int16_t>(m[3])};
        mi[i].ref_idx[0] = static_cast<int8_t>(m[4]);
        mi[i].ref_idx[1] = static_cast<int8_t>(m[5]);
        mi[i].pred_flag[0] = m[6] != 0;
        mi[i].pred_flag[1] = m[7] != 0;
    }

    std::vector<hevc::SliceHeader> shs(num_slices);
    for (int s = 0; s < num_slices; s++) {
        const int32_t* f = sh_params + 4 * s;
        shs[s].slice_deblocking_filter_disabled_flag = f[0] != 0;
        shs[s].slice_loop_filter_across_slices_enabled_flag = f[1] != 0;
        shs[s].slice_beta_offset_div2 = f[2];
        shs[s].slice_tc_offset_div2 = f[3];
    }

    std::vector<hevc::SliceHeader*> shPtrs(num_slices);
    for (int s = 0; s < num_slices; s++) shPtrs[s] = &shs[s];

    hevc::DPB dpb;
    {
        std::vector<int32_t> pocs0;
        if (n_ref_l0 > 0 && poc_l0) pocs0.assign(poc_l0, poc_l0 + n_ref_l0);
        std::vector<int32_t> pocs1;
        if (n_ref_l1 > 0 && poc_l1) pocs1.assign(poc_l1, poc_l1 + n_ref_l1);
        dpb.test_set_ref_pic_lists(pocs0, pocs1);
    }

    // sh_at_ctb fallback (single-slice pictures) mirrors slice 0's parameters.
    hevc::SliceHeader fallbackSh;
    fallbackSh.slice_deblocking_filter_disabled_flag = sh_params[0] != 0;
    fallbackSh.slice_loop_filter_across_slices_enabled_flag = sh_params[1] != 0;
    fallbackSh.slice_beta_offset_div2 = sh_params[2];
    fallbackSh.slice_tc_offset_div2 = sh_params[3];

    hevc::DecodingContext ctx;
    ctx.sps = &sps;
    ctx.pps = &pps;
    ctx.pic = &pic;
    ctx.sh = &fallbackSh;
    ctx.dpb = &dpb;
    ctx.cu_info = cus.data();
    ctx.cu_info_stride = minCbsW;
    ctx.motion_info = mi.data();
    ctx.motion_info_stride = minCbsW;
    ctx.cbf_luma_grid = cbfGrid.data();
    ctx.log2_tu_size_grid = tuGrid.data();
    ctx.edge_flags_v = edgeV.data();
    ctx.edge_flags_h = edgeH.data();
    ctx.filter_grid_stride = minCbsW;
    if (slice_idx) {
        ctx.slice_idx = sliceIdxCopy.data();
        ctx.slice_headers = shPtrs.data();
        ctx.num_slices = num_slices;
    }

    hevc::apply_deblocking(ctx);

    copy_planes_out(pic, chromaArrayType, stride_y, stride_cb, stride_cr,
                    picW, picH, subW, subH, out_y, out_cb, out_cr);
    return 0;
}

} // extern "C"
