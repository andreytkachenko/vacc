#pragma once

// Intra Prediction — 35 modes (Planar, DC, Angular 2-34)
// Spec §8.4.4.2

#include <cstdint>

namespace hevc {

struct DecodingContext;

// Perform intra prediction for a block
// pred_samples: output array of size (1<<log2PredSize)^2
void perform_intra_prediction(DecodingContext& ctx, int x0, int y0,
                              int log2PredSize, int cIdx, int intra_mode,
                              int16_t* pred_samples);

// ============================================================
// Kernels exposed for differential testing (hevc_test_api.h)
// ============================================================

// Reference sample construction (§8.4.4.2.2). refTop/refLeft: 2*nTbS+1 samples each.
void build_reference_samples(const DecodingContext& ctx, int x0, int y0,
                             int nTbS, int cIdx,
                             int16_t* refTop, int16_t* refLeft);

// §8.4.4.2.3: is reference filtering required for this mode/size?
bool needs_filtering(int intra_mode, int log2BlkSize);

// §8.4.4.2.3: in-place [1,2,1]/4 (or bilinear) filtering of 2*nTbS+1 samples.
void filter_reference_samples(int16_t* ref, int nTbS, bool biIntFlag, int bitDepth);

// Planar prediction (mode 0) — §8.4.4.2.4.
void predict_planar(const int16_t* refTop, const int16_t* refLeft, int nTbS, int16_t* pred);

// DC prediction (mode 1) — §8.4.4.2.5.
void predict_dc(const int16_t* refTop, const int16_t* refLeft, int nTbS, int log2BlkSize,
                int cIdx, int16_t* pred);

// Angular prediction (modes 2-34) — §8.4.4.2.6.
void predict_angular(const int16_t* refTop, const int16_t* refLeft, int nTbS, int intra_mode,
                     int cIdx, int bitDepth, int16_t* pred);

} // namespace hevc
