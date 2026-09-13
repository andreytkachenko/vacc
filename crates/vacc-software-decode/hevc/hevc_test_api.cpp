/*
 * Test-only C API: differential-testing oracles for the Rust ports of the
 * hevc.js kernels (see hevc_test_api.h). Not part of the decode pipeline.
 */

#include "hevc_test_api.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "bitstream/bitstream_reader.h"
#include "bitstream/nal_unit.h"
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

// ============================================================
// Tier E: slice-segment replay oracle
// ============================================================

namespace {

// Frame-level state for the slice-segment replay oracle.
struct TestFrameState {
    hevc::SPS sps;
    hevc::PPS pps;
    hevc::DPB dpb;
    std::vector<std::shared_ptr<hevc::Picture>> ref_pics;

    hevc::DecodingContext ctx;
    hevc::CabacEngine cabac;
    std::vector<hevc::CUInfo> cu_info_buf;
    std::vector<int> intra_mode_buf;
    std::vector<int> chroma_mode_buf;
    std::vector<hevc::PUMotionInfo> motion_info_buf;
    std::vector<uint8_t> cbf_luma_buf;
    std::vector<uint8_t> log2_tu_buf;
    std::vector<uint8_t> edge_v_buf;
    std::vector<uint8_t> edge_h_buf;
    std::vector<hevc::DecodingContext::SaoParams> sao_params_buf;
    std::vector<uint8_t> slice_idx_buf;
    std::vector<hevc::SliceHeader> slice_headers;
    std::vector<const hevc::SliceHeader*> slice_header_ptrs;

    int picW = 0, picH = 0;
    int compW = 0, compH = 0;
    int modeGridW = 0, modeGridH = 0;
    int minCbsW = 0, minCbsH = 0;
    int ctbCount = 0;
};

// Parse SPS/PPS NALs and flatten them. Returns 0 on success, -1 parse
// failure, -2 unsupported feature (the Rust port has no path for these yet).
int parse_and_flatten_ps(const uint8_t* sps_nal, int sps_len,
                         const uint8_t* pps_nal, int pps_len,
                         hevc::SPS& sps_out, hevc::PPS& pps_out,
                         int32_t* out_sps, int32_t* out_pps) {
    if (!sps_nal || !pps_nal || sps_len <= 0 || pps_len <= 0) return -1;
    hevc::NalParser parser;
    auto sps_nals = parser.parse(sps_nal, static_cast<size_t>(sps_len));
    auto pps_nals = parser.parse(pps_nal, static_cast<size_t>(pps_len));
    if (sps_nals.size() != 1 || pps_nals.size() != 1) return -1;

    {
        hevc::BitstreamReader bs(sps_nals[0].rbsp.data(), sps_nals[0].rbsp.size());
        if (!sps_out.parse(bs)) return -1;
    }
    {
        hevc::BitstreamReader bs(pps_nals[0].rbsp.data(), pps_nals[0].rbsp.size());
        if (!pps_out.parse(bs, sps_out)) return -1;
    }

    if (sps_out.scaling_list_enabled_flag || pps_out.pps_scaling_list_data_present_flag ||
        pps_out.tiles_enabled_flag || pps_out.dependent_slice_segments_enabled_flag ||
        sps_out.separate_colour_plane_flag) return -2;

    // Flatten SPS (HEVCDEC_TEST_SPS_FLAT)
    int32_t* o = out_sps;
    o[0] = static_cast<int32_t>(sps_out.pic_width_in_luma_samples);
    o[1] = static_cast<int32_t>(sps_out.pic_height_in_luma_samples);
    o[2] = sps_out.BitDepthY;
    o[3] = sps_out.BitDepthC;
    o[4] = sps_out.ChromaArrayType;
    o[5] = sps_out.CtbSizeY;
    o[6] = sps_out.MinTbSizeY;
    o[7] = sps_out.PicWidthInCtbsY;
    o[8] = sps_out.SubWidthC;
    o[9] = sps_out.SubHeightC;
    o[10] = sps_out.intra_smoothing_disabled_flag;
    o[11] = sps_out.strong_intra_smoothing_enabled_flag;
    o[12] = sps_out.MinCbLog2SizeY;
    o[13] = sps_out.CtbLog2SizeY;
    o[14] = sps_out.MinCbSizeY;
    o[15] = sps_out.PicHeightInCtbsY;
    o[16] = sps_out.PicSizeInCtbsY;
    o[17] = sps_out.MinTbLog2SizeY;
    o[18] = sps_out.MaxTbLog2SizeY;
    o[19] = sps_out.QpBdOffsetY;
    o[20] = sps_out.QpBdOffsetC;
    o[21] = sps_out.amp_enabled_flag;
    o[22] = sps_out.pcm_enabled_flag;
    o[23] = sps_out.pcm_sample_bit_depth_luma_minus1;
    o[24] = sps_out.pcm_sample_bit_depth_chroma_minus1;
    o[25] = sps_out.Log2MinIpcmCbSizeY;
    o[26] = sps_out.Log2MaxIpcmCbSizeY;
    o[27] = static_cast<int32_t>(sps_out.max_transform_hierarchy_depth_inter);
    o[28] = static_cast<int32_t>(sps_out.max_transform_hierarchy_depth_intra);
    o[29] = sps_out.cabac_bypass_alignment_enabled_flag;

    // Flatten PPS (HEVCDEC_TEST_PPS_FLAT)
    o = out_pps;
    o[0] = pps_out.sign_data_hiding_enabled_flag;
    o[1] = pps_out.transform_skip_enabled_flag;
    o[2] = pps_out.cu_qp_delta_enabled_flag;
    o[3] = static_cast<int32_t>(pps_out.diff_cu_qp_delta_depth);
    o[4] = pps_out.pps_cb_qp_offset;
    o[5] = pps_out.pps_cr_qp_offset;
    o[6] = pps_out.weighted_pred_flag;
    o[7] = pps_out.weighted_bipred_flag;
    o[8] = pps_out.transquant_bypass_enabled_flag;
    o[9] = pps_out.tiles_enabled_flag;
    o[10] = pps_out.entropy_coding_sync_enabled_flag;
    o[11] = pps_out.pps_loop_filter_across_slices_enabled_flag;
    o[12] = static_cast<int32_t>(pps_out.log2_parallel_merge_level_minus2);
    o[13] = static_cast<int32_t>(pps_out.num_tile_columns_minus1);
    o[14] = static_cast<int32_t>(pps_out.num_tile_rows_minus1);
    o[15] = pps_out.uniform_spacing_flag;
    for (int i = 0; i < 16; i++) {
        o[16 + i] = i < static_cast<int>(pps_out.column_width_minus1.size())
                        ? static_cast<int32_t>(pps_out.column_width_minus1[i]) : 0;
        o[32 + i] = i < static_cast<int>(pps_out.row_height_minus1.size())
                        ? static_cast<int32_t>(pps_out.row_height_minus1[i]) : 0;
    }
    return 0;
}

} // namespace

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
    void** out_state) {
    if (!sps_nal || !pps_nal || sps_len <= 0 || pps_len <= 0 || n_refs <= 0 ||
        n_refs > 32 || n_list0 < 0 || n_list0 > 16 || n_list1 < 0 || n_list1 > 16 ||
        col_pic_idx < -1 || !out_sps || !out_pps || !out_state) return -1;

    hevc::SPS sps;
    hevc::PPS pps;
    int rc = parse_and_flatten_ps(sps_nal, sps_len, pps_nal, pps_len, sps, pps,
                                  out_sps, out_pps);
    if (rc != 0) return rc;

    auto* st = new TestFrameState();
    st->sps = std::move(sps);
    st->pps = std::move(pps);

    const int picW = static_cast<int>(st->sps.pic_width_in_luma_samples);
    const int picH = static_cast<int>(st->sps.pic_height_in_luma_samples);
    st->picW = picW;
    st->picH = picH;
    st->compW = picW / st->sps.SubWidthC;
    st->compH = picH / st->sps.SubHeightC;
    st->modeGridW = picW / st->sps.MinTbSizeY;
    st->modeGridH = picH / st->sps.MinTbSizeY;
    st->minCbsW = st->sps.PicWidthInMinCbsY;
    st->minCbsH = st->sps.PicHeightInMinCbsY;
    st->ctbCount = st->sps.PicSizeInCtbsY;

    // Synthetic reference picture pool
    hevc::ChromaFormat fmt = static_cast<hevc::ChromaFormat>(st->sps.chroma_format_idc);
    const size_t modeGrid = static_cast<size_t>(st->modeGridW) * st->modeGridH;
    for (int r = 0; r < n_refs; r++) {
        auto pic = std::make_shared<hevc::Picture>();
        pic->allocate(picW, picH, fmt, st->sps.BitDepthY, st->sps.BitDepthC);
        const uint16_t* src[3] = {
            ref_y + static_cast<size_t>(r) * picW * picH,
            ref_cb + static_cast<size_t>(r) * st->compW * st->compH,
            ref_cr + static_cast<size_t>(r) * st->compW * st->compH,
        };
        for (int c = 0; c < 3; c++) {
            if (pic->width[c] > 0) {
                std::memcpy(pic->planes[c].data(), src[c],
                            static_cast<size_t>(pic->stride[c]) * pic->height[c] * sizeof(uint16_t));
            }
        }
        pic->poc = ref_poc[r];
        pic->used_for_short_term_ref = ref_st_ref[r] != 0;
        pic->used_for_long_term_ref = ref_lt_ref[r] != 0;
        pic->motion_info_buf.resize(modeGrid);
        const int32_t* rm = ref_motion + static_cast<size_t>(r) * modeGrid * 8;
        for (size_t b = 0; b < modeGrid; b++) {
            auto& mi = pic->motion_info_buf[b];
            mi.mv_x[0] = static_cast<int16_t>(rm[b * 8 + 0]);
            mi.mv_y[0] = static_cast<int16_t>(rm[b * 8 + 1]);
            mi.ref_idx[0] = static_cast<int8_t>(rm[b * 8 + 2]);
            mi.pred_flag[0] = rm[b * 8 + 3] != 0;
            mi.mv_x[1] = static_cast<int16_t>(rm[b * 8 + 4]);
            mi.mv_y[1] = static_cast<int16_t>(rm[b * 8 + 5]);
            mi.ref_idx[1] = static_cast<int8_t>(rm[b * 8 + 6]);
            mi.pred_flag[1] = rm[b * 8 + 7] != 0;
        }
        pic->motion_info_stride = st->modeGridW;
        for (int l = 0; l < 2; l++) {
            pic->ref_poc[l].clear();
            for (int k = 0; k < 16; k++)
                pic->ref_poc[l].push_back(ref_refpoc[r * 32 + l * 16 + k]);
        }
        st->ref_pics.push_back(std::move(pic));
    }

    std::vector<int> l0, l1;
    for (int i = 0; i < n_list0; i++) l0.push_back(list0_idx[i]);
    for (int i = 0; i < n_list1; i++) l1.push_back(list1_idx[i]);
    st->dpb.test_install_ref_pics(st->ref_pics, l0, l1, col_pic_idx, no_backward_pred != 0);

    // Current picture (planes allocated by the DPB)
    hevc::Picture* pic = st->dpb.alloc_picture(picW, picH, fmt, st->sps.BitDepthY,
                                               st->sps.BitDepthC);
    pic->poc = cur_poc;
    pic->motion_info_buf.assign(modeGrid, hevc::Picture::PUMotionInfoCompact{});
    pic->motion_info_stride = st->modeGridW;

    // Per-picture grids (mirrors Decoder::decode_picture)
    st->cu_info_buf.assign(static_cast<size_t>(st->minCbsW) * st->minCbsH, hevc::CUInfo{});
    st->intra_mode_buf.assign(modeGrid, 1);   // DC default
    st->chroma_mode_buf.assign(modeGrid, 0);  // planar default
    st->motion_info_buf.assign(modeGrid, hevc::PUMotionInfo{});
    st->cbf_luma_buf.assign(modeGrid, 0);
    st->log2_tu_buf.assign(modeGrid, static_cast<uint8_t>(st->sps.CtbLog2SizeY));
    st->edge_v_buf.assign(modeGrid, 0);
    st->edge_h_buf.assign(modeGrid, 0);
    st->sao_params_buf.assign(st->ctbCount, hevc::DecodingContext::SaoParams{});
    st->slice_idx_buf.assign(st->ctbCount, 0);
    st->slice_headers.resize(HEVCDEC_TEST_MAX_SEGS);

    // Decoding context. No thread pool: always the serial path — the Rust
    // side compares both its serial and WPP paths against this.
    hevc::DecodingContext& ctx = st->ctx;
    ctx.sps = &st->sps;
    ctx.pps = &st->pps;
    ctx.cabac = &st->cabac;
    ctx.pic = pic;
    ctx.dpb = &st->dpb;
    ctx.cu_info = st->cu_info_buf.data();
    ctx.cu_info_stride = st->minCbsW;
    ctx.intra_pred_mode_y = st->intra_mode_buf.data();
    ctx.intra_pred_mode_c = st->chroma_mode_buf.data();
    ctx.intra_pred_mode_stride = st->modeGridW;
    ctx.motion_info = st->motion_info_buf.data();
    ctx.motion_info_stride = st->modeGridW;
    ctx.cbf_luma_grid = st->cbf_luma_buf.data();
    ctx.log2_tu_size_grid = st->log2_tu_buf.data();
    ctx.edge_flags_v = st->edge_v_buf.data();
    ctx.edge_flags_h = st->edge_h_buf.data();
    ctx.filter_grid_stride = st->modeGridW;
    ctx.sao_params = st->sao_params_buf.data();
    ctx.sao_params_stride = st->sps.PicWidthInCtbsY;
    ctx.slice_idx = st->slice_idx_buf.data();
    ctx.wpp_contexts_available = false;
    ctx.thread_pool = nullptr;

    *out_state = st;
    return 0;
}

int hevcdec_test_frame_decode(
    void* vstate,
    const uint8_t* vcl_nals, int vcl_len,
    int* out_n_segments,
    int32_t* out_sh, uint32_t* out_final_bit_pos,
    uint16_t* out_y, uint16_t* out_cb, uint16_t* out_cr,
    int32_t* out_cu_info,
    int32_t* out_intra_luma, int32_t* out_intra_chroma,
    int32_t* out_motion,
    uint8_t* out_cbf_luma, uint8_t* out_log2_tu,
    uint8_t* out_edge_v, uint8_t* out_edge_h,
    int32_t* out_sao_params, uint8_t* out_slice_idx) {
    auto* st = static_cast<TestFrameState*>(vstate);
    if (!st || !vcl_nals || vcl_len <= 0 || !out_n_segments || !out_sh ||
        !out_final_bit_pos || !out_y || !out_cu_info || !out_intra_luma ||
        !out_intra_chroma || !out_motion || !out_cbf_luma || !out_log2_tu ||
        !out_edge_v || !out_edge_h || !out_sao_params || !out_slice_idx) return -1;

    hevc::NalParser parser;
    auto nals = parser.parse(vcl_nals, static_cast<size_t>(vcl_len));
    if (nals.empty() || nals.size() > HEVCDEC_TEST_MAX_SEGS) return -1;

    hevc::SPS& sps = st->sps;
    hevc::PPS& pps = st->pps;
    hevc::DecodingContext& ctx = st->ctx;

    // Parse all slice headers first; dependent segments inherit from the last
    // independent one (§7.4.7.1, mirrors Decoder::decode_picture).
    int last_independent = 0;
    for (size_t s = 0; s < nals.size(); s++) {
        hevc::SliceHeader sh;
        hevc::BitstreamReader bs(nals[s].rbsp.data(), nals[s].rbsp.size());
        if (!sh.parse(bs, sps, pps, nals[s].header.nal_unit_type,
                      nals[s].header.TemporalId())) return -3;
        if (sh.dependent_slice_segment_flag) {
            uint32_t saved_address = sh.slice_segment_address;
            bool saved_dependent = sh.dependent_slice_segment_flag;
            bool saved_first = sh.first_slice_segment_in_pic_flag;
            st->slice_headers[s] = st->slice_headers[last_independent];
            st->slice_headers[s].slice_segment_address = saved_address;
            st->slice_headers[s].dependent_slice_segment_flag = saved_dependent;
            st->slice_headers[s].first_slice_segment_in_pic_flag = saved_first;
        } else {
            st->slice_headers[s] = std::move(sh);
            last_independent = static_cast<int>(s);
        }
    }
    st->slice_header_ptrs.resize(nals.size());
    for (size_t s = 0; s < nals.size(); s++) st->slice_header_ptrs[s] = &st->slice_headers[s];
    ctx.slice_headers = st->slice_header_ptrs.data();
    ctx.num_slices = static_cast<int>(nals.size());

    for (size_t s = 0; s < nals.size(); s++) {
        const auto& nal = nals[s];
        ctx.sh = &st->slice_headers[s];
        ctx.current_slice_idx = static_cast<int>(s);

        // Re-parse the slice header to advance the bitstream position.
        hevc::BitstreamReader bs(nal.rbsp.data(), nal.rbsp.size());
        {
            hevc::SliceHeader sh_skip;
            (void)sh_skip.parse(bs, sps, pps, nal.header.nal_unit_type,
                                nal.header.TemporalId());
        }
        if (!bs.byte_aligned()) bs.byte_alignment();
        size_t slice_header_coded_size = bs.byte_position();
        for (size_t ep : nal.epb_positions)
            if (ep < slice_header_coded_size + 2) slice_header_coded_size++;

        fprintf(stderr, "[TEST] seg %zu rbsp_pos=%zu sh_coded=%zu n_epb=%zu eps=%u\n", s,
                bs.byte_position(), slice_header_coded_size, nal.epb_positions.size(),
                st->slice_headers[s].num_entry_point_offsets);

        if (!hevc::decode_slice_segment_data(ctx, bs, nal.epb_positions,
                                             slice_header_coded_size))
            return -4;

        // Flatten the slice header for the Rust side.
        const auto& sh = st->slice_headers[s];
        int32_t* o = out_sh + s * HEVCDEC_TEST_SH_FLAT;
        o[0] = static_cast<int32_t>(sh.slice_segment_address);
        o[1] = sh.dependent_slice_segment_flag;
        o[2] = static_cast<int32_t>(sh.slice_type);
        o[3] = sh.pic_output_flag;
        o[4] = sh.slice_temporal_mvp_enabled_flag;
        o[5] = sh.slice_sao_luma_flag;
        o[6] = sh.slice_sao_chroma_flag;
        o[7] = static_cast<int32_t>(sh.num_ref_idx_l0_active_minus1);
        o[8] = static_cast<int32_t>(sh.num_ref_idx_l1_active_minus1);
        o[9] = sh.mvd_l1_zero_flag;
        o[10] = sh.cabac_init_flag;
        o[11] = sh.collocated_from_l0_flag;
        o[12] = static_cast<int32_t>(sh.collocated_ref_idx);
        o[13] = static_cast<int32_t>(sh.five_minus_max_num_merge_cand);
        o[14] = sh.slice_qp_delta;
        o[15] = sh.slice_cb_qp_offset;
        o[16] = sh.slice_cr_qp_offset;
        o[17] = sh.SliceQpY;
        o[18] = sh.MaxNumMergeCand;
        o[19] = static_cast<int32_t>(sh.num_entry_point_offsets);
        o[20] = static_cast<int32_t>(slice_header_coded_size);
        o[21] = static_cast<int32_t>(sh.pred_weight_table.luma_log2_weight_denom);
        o[22] = sh.pred_weight_table.delta_chroma_log2_weight_denom;
        int32_t* w = o + 23;
        for (int list = 0; list < 2; list++) {
            const auto& arr = (list == 0) ? sh.pred_weight_table.l0 : sh.pred_weight_table.l1;
            for (int r = 0; r < 16; r++, w += 8) {
                w[0] = arr[r].luma_weight_flag;
                w[1] = arr[r].chroma_weight_flag;
                w[2] = arr[r].luma_weight;
                w[3] = arr[r].luma_offset;
                w[4] = arr[r].chroma_weight[0];
                w[5] = arr[r].chroma_offset[0];
                w[6] = arr[r].chroma_weight[1];
                w[7] = arr[r].chroma_offset[1];
            }
        }
        for (int i = 0; i < 64; i++) {
            o[279 + i] = i < static_cast<int>(sh.num_entry_point_offsets)
                             ? static_cast<int32_t>(sh.entry_point_offset_minus1[i]) : 0;
        }
        out_final_bit_pos[s] = static_cast<uint32_t>(bs.bits_read());
    }

    // Copy out the full picture state.
    hevc::Picture& pic = *ctx.pic;
    std::memcpy(out_y, pic.planes[0].data(),
                static_cast<size_t>(st->picW) * st->picH * sizeof(uint16_t));
    if (st->sps.ChromaArrayType > 0) {
        std::memcpy(out_cb, pic.planes[1].data(),
                    static_cast<size_t>(st->compW) * st->compH * sizeof(uint16_t));
        std::memcpy(out_cr, pic.planes[2].data(),
                    static_cast<size_t>(st->compW) * st->compH * sizeof(uint16_t));
    }

    const size_t minCbs = static_cast<size_t>(st->minCbsW) * st->minCbsH;
    for (size_t i = 0; i < minCbs; i++) {
        const auto& cu = st->cu_info_buf[i];
        out_cu_info[i * 8 + 0] = static_cast<int32_t>(cu.pred_mode);
        out_cu_info[i * 8 + 1] = static_cast<int32_t>(cu.part_mode);
        out_cu_info[i * 8 + 2] = cu.log2CbSize;
        out_cu_info[i * 8 + 3] = cu.intra_mode_luma;
        out_cu_info[i * 8 + 4] = cu.qp_y;
        out_cu_info[i * 8 + 5] = cu.is_pcm;
        out_cu_info[i * 8 + 6] = cu.cu_transquant_bypass;
        out_cu_info[i * 8 + 7] = cu.merge_flag;
    }

    const size_t modeGrid = static_cast<size_t>(st->modeGridW) * st->modeGridH;
    std::memcpy(out_intra_luma, st->intra_mode_buf.data(), modeGrid * sizeof(int32_t));
    std::memcpy(out_intra_chroma, st->chroma_mode_buf.data(), modeGrid * sizeof(int32_t));
    for (size_t b = 0; b < modeGrid; b++) {
        const auto& mi = st->motion_info_buf[b];
        out_motion[b * 8 + 0] = mi.mv[0].x;
        out_motion[b * 8 + 1] = mi.mv[0].y;
        out_motion[b * 8 + 2] = mi.ref_idx[0];
        out_motion[b * 8 + 3] = mi.pred_flag[0];
        out_motion[b * 8 + 4] = mi.mv[1].x;
        out_motion[b * 8 + 5] = mi.mv[1].y;
        out_motion[b * 8 + 6] = mi.ref_idx[1];
        out_motion[b * 8 + 7] = mi.pred_flag[1];
    }
    std::memcpy(out_cbf_luma, st->cbf_luma_buf.data(), modeGrid);
    std::memcpy(out_log2_tu, st->log2_tu_buf.data(), modeGrid);
    std::memcpy(out_edge_v, st->edge_v_buf.data(), modeGrid);
    std::memcpy(out_edge_h, st->edge_h_buf.data(), modeGrid);

    for (int c = 0; c < st->ctbCount; c++) {
        const auto& sp = st->sao_params_buf[c];
        int32_t* o = out_sao_params + c * 24;
        for (int k = 0; k < 3; k++) o[k] = sp.sao_type_idx[k];
        for (int k = 0; k < 3; k++) o[3 + k] = sp.sao_eo_class[k];
        for (int k = 0; k < 3; k++) o[6 + k] = sp.sao_band_position[k];
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 5; j++) o[9 + k * 5 + j] = sp.sao_offset_val[k][j];
    }
    std::memcpy(out_slice_idx, st->slice_idx_buf.data(), st->ctbCount);

    *out_n_segments = static_cast<int>(nals.size());
    return 0;
}

int hevcdec_test_parse_sps_pps(const uint8_t* sps_nal, int sps_len,
                               const uint8_t* pps_nal, int pps_len,
                               int32_t* out_sps, int32_t* out_pps) {
    if (!out_sps || !out_pps) return -1;
    hevc::SPS sps;
    hevc::PPS pps;
    return parse_and_flatten_ps(sps_nal, sps_len, pps_nal, pps_len, sps, pps,
                                out_sps, out_pps);
}

void hevcdec_test_frame_free(void* vstate) { delete static_cast<TestFrameState*>(vstate); }

} // extern "C"
