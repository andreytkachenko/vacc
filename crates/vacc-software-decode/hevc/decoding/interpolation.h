#pragma once

// Inter sample interpolation — spec §8.5.3.3
// Luma 8-tap (§8.5.3.3.3), Chroma 4-tap (§8.5.3.3.3)
// Weighted prediction (§8.5.3.3.4)

#include <cstdint>
#include "common/picture.h"
#include "common/types.h"

namespace hevc {

struct DecodingContext;

// ============================================================
// Motion compensation for one PU — §8.5.3.3
// ============================================================

// Perform motion compensation and produce final prediction samples.
// Handles uni-pred and bi-pred, luma and chroma interpolation,
// and default weighted prediction averaging.
// pred_samples: output array, nPbW * nPbH (or nPbW/2 * nPbH/2 for chroma)
void perform_inter_prediction(DecodingContext& ctx,
                               int xPb, int yPb, int nPbW, int nPbH,
                               int cIdx,
                               const MV& mvL0, const MV& mvL1,
                               int refIdxL0, int refIdxL1,
                               bool predFlagL0, bool predFlagL1,
                               int16_t* pred_samples);

// ============================================================
// Kernels exposed for differential testing (hevc_test_api.h)
// ============================================================

struct PredWeightTable;

// Luma 8-tap interpolation (§8.5.3.3.3). Output in extended precision.
void interpolate_luma(const Picture& refPic, int xInt, int yInt, int xFrac, int yFrac,
                      int nPbW, int nPbH, int bitDepth, int16_t* pred);

// Chroma 4-tap interpolation (§8.5.3.3.3). Output in extended precision.
void interpolate_chroma(const Picture& refPic, int cIdx, int xInt, int yInt, int xFrac, int yFrac,
                        int nPbWC, int nPbHC, int bitDepth, int16_t* pred);

// Default weighted sample prediction (§8.5.3.3.4.2).
void weighted_pred_default(const int16_t* predL0, const int16_t* predL1, bool flagL0,
                           bool flagL1, int nSamples, int bitDepth, int16_t* output);

// Explicit weighted sample prediction (§8.5.3.3.4.3).
void weighted_pred_explicit(const int16_t* predL0, const int16_t* predL1, bool flagL0,
                            bool flagL1, int refIdxL0, int refIdxL1, int cIdx, int nSamples,
                            int bitDepth, const PredWeightTable& pwt, int16_t* output);

} // namespace hevc
