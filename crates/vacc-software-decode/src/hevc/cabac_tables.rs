//! Port of `hevc/decoding/cabac_tables.h` — CABAC arithmetic decoder tables.
//!
//! Spec: ITU-T H.265, Section 9.3 (Tables 9-48 to 9-50), context
//! initialization values (Tables 9-5 to 9-31), scan orders (§6.5.3) and
//! dequantization helpers (Tables 8-10, §8.6.3).

/// Table 9-48 — `rangeTabLps[64][4]`.
/// Index: `[pStateIdx][qRangeIdx]` where `qRangeIdx = (ivlCurrRange >> 6) & 3`.
pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// Table 9-49 — `transIdxMps[64]`.
pub const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// Table 9-50 — `transIdxLps[64]`.
pub const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

// Context index offsets (global ctxIdx assignment).
pub const CTX_SAO_MERGE_FLAG: usize = 0; // 1 context
pub const CTX_SAO_TYPE_IDX: usize = 1; // 1 context
pub const CTX_SPLIT_CU_FLAG: usize = 2; // 3 contexts
pub const CTX_CU_TRANSQUANT_BYPASS: usize = 5; // 1 context
pub const CTX_CU_SKIP_FLAG: usize = 6; // 3 contexts
pub const CTX_PRED_MODE_FLAG: usize = 9; // 1 context
pub const CTX_PART_MODE: usize = 10; // 4 contexts
pub const CTX_PREV_INTRA_LUMA_PRED: usize = 14; // 1 context
pub const CTX_INTRA_CHROMA_PRED_MODE: usize = 15; // 1 context
pub const CTX_MERGE_FLAG: usize = 16; // 1 context
pub const CTX_MERGE_IDX: usize = 17; // 1 context
pub const CTX_INTER_PRED_IDC: usize = 18; // 5 contexts
pub const CTX_REF_IDX: usize = 23; // 2 contexts
pub const CTX_MVP_FLAG: usize = 25; // 1 context
pub const CTX_SPLIT_TRANSFORM_FLAG: usize = 26; // 3 contexts
pub const CTX_CBF_LUMA: usize = 29; // 2 contexts
pub const CTX_CBF_CHROMA: usize = 31; // 5 contexts (cb+cr share) — Table 9-22
pub const CTX_ABS_MVD_GREATER0: usize = 36; // 1 context
pub const CTX_ABS_MVD_GREATER1: usize = 37; // 1 context
pub const CTX_CU_QP_DELTA_ABS: usize = 38; // 2 contexts
pub const CTX_TRANSFORM_SKIP_FLAG: usize = 40; // 2 contexts
pub const CTX_LAST_SIG_COEFF_X: usize = 42; // 18 contexts
pub const CTX_LAST_SIG_COEFF_Y: usize = 60; // 18 contexts
pub const CTX_CODED_SUB_BLOCK_FLAG: usize = 78; // 4 contexts
pub const CTX_SIG_COEFF_FLAG: usize = 82; // 42 contexts (27 luma + 15 chroma, spec Table 9-29)
pub const CTX_COEFF_ABS_LEVEL_GREATER1: usize = 124; // 24 contexts
pub const CTX_COEFF_ABS_LEVEL_GREATER2: usize = 148; // 6 contexts
pub const CTX_RQT_ROOT_CBF: usize = 154; // 1 context — Table 9-14
pub const NUM_CABAC_CONTEXTS: usize = 155;

/// Context initialization values (Tables 9-5 to 9-31).
/// `CABAC_INIT_VALUES[ctxIdx][sliceType]` where sliceType: 0=I, 1=P, 2=B.
pub const CABAC_INIT_VALUES: [[u8; 3]; NUM_CABAC_CONTEXTS] = [
    // CTX_SAO_MERGE_FLAG (0) — Table 9-5
    [153, 153, 153],
    // CTX_SAO_TYPE_IDX (1) — Table 9-6
    [200, 185, 160],
    // CTX_SPLIT_CU_FLAG (2-4) — Table 9-7
    [139, 107, 107],
    [141, 139, 139],
    [157, 126, 126],
    // CTX_CU_TRANSQUANT_BYPASS (5) — Table 9-8
    [154, 154, 154],
    // CTX_CU_SKIP_FLAG (6-8) — Table 9-9
    [0, 197, 197],
    [0, 185, 185],
    [0, 201, 201],
    // CTX_PRED_MODE_FLAG (9) — Table 9-10
    [0, 149, 134],
    // CTX_PART_MODE (10-13) — Table 9-11
    [184, 154, 154],
    [0, 139, 139],
    [0, 154, 154],
    [0, 154, 154],
    // CTX_PREV_INTRA_LUMA_PRED (14) — Table 9-12
    [184, 154, 183],
    // CTX_INTRA_CHROMA_PRED_MODE (15) — Table 9-13
    [63, 152, 152],
    // CTX_MERGE_FLAG (16) — Table 9-14
    [0, 110, 154],
    // CTX_MERGE_IDX (17) — Table 9-15
    [0, 122, 137],
    // CTX_INTER_PRED_IDC (18-22) — Table 9-16
    [0, 95, 95],
    [0, 79, 79],
    [0, 63, 63],
    [0, 31, 31],
    [0, 31, 31],
    // CTX_REF_IDX (23-24) — Table 9-17
    [0, 153, 153],
    [0, 153, 153],
    // CTX_MVP_FLAG (25) — Table 9-18
    [0, 168, 168],
    // CTX_SPLIT_TRANSFORM_FLAG (26-28) — Table 9-20
    [153, 124, 224],
    [138, 138, 167],
    [138, 94, 122],
    // CTX_CBF_LUMA (29-30) — Table 9-21
    [111, 153, 153],
    [141, 111, 111],
    // CTX_CBF_CHROMA (31-35) — Table 9-22 (5 contexts, not 4)
    [94, 149, 149],
    [138, 107, 92],
    [182, 167, 167],
    [154, 154, 154],
    [154, 154, 154],
    // CTX_ABS_MVD_GREATER0 (36) — Table 9-23
    [0, 140, 169],
    // CTX_ABS_MVD_GREATER1 (37) — Table 9-23
    [0, 198, 198],
    // CTX_CU_QP_DELTA_ABS (38-39) — Table 9-24
    [154, 154, 154],
    [154, 154, 154],
    // CTX_TRANSFORM_SKIP_FLAG (40-41) — Table 9-25
    [139, 139, 139],
    [139, 139, 139],
    // CTX_LAST_SIG_COEFF_X (42-59) — Table 9-26
    // I: ctxIdx 0-17, P: ctxIdx 18-35, B: ctxIdx 36-53
    [110, 125, 125],
    [110, 110, 110],
    [124, 94, 124],
    [125, 110, 110],
    [140, 95, 95],
    [153, 79, 94],
    [125, 125, 125],
    [127, 111, 111],
    [140, 110, 111],
    [109, 78, 79],
    [111, 110, 125],
    [143, 111, 126],
    [127, 111, 111],
    [111, 95, 111],
    [79, 94, 79],
    [108, 108, 108],
    [123, 123, 123],
    [63, 108, 93],
    // CTX_LAST_SIG_COEFF_Y (60-77) — Table 9-27
    // I: ctxIdx 0-17, P: ctxIdx 18-35, B: ctxIdx 36-53
    [110, 125, 125],
    [110, 110, 110],
    [124, 94, 124],
    [125, 110, 110],
    [140, 95, 95],
    [153, 79, 94],
    [125, 125, 125],
    [127, 111, 111],
    [140, 110, 111],
    [109, 78, 79],
    [111, 110, 125],
    [143, 111, 126],
    [127, 111, 111],
    [111, 95, 111],
    [79, 94, 79],
    [108, 108, 108],
    [123, 123, 123],
    [63, 108, 93],
    // CTX_CODED_SUB_BLOCK_FLAG (78-81) — Table 9-28
    [91, 121, 121],
    [171, 140, 140],
    [134, 61, 61],
    [141, 154, 154],
    // CTX_SIG_COEFF_FLAG (82-123) — 27 luma + 15 chroma = 42 contexts (spec Table 9-29)
    // Luma (27 contexts: 4x4[9], 8x8_diag[6], 8x8_nondiag[6], NxN[6])
    [111, 155, 170],
    [111, 154, 154],
    [125, 139, 139],
    [110, 153, 153],
    [110, 139, 139],
    [94, 123, 123],
    [124, 123, 123],
    [108, 63, 63],
    [124, 153, 124],
    [107, 166, 166],
    [125, 183, 183],
    [141, 140, 140],
    [179, 136, 136],
    [153, 153, 153],
    [125, 154, 154],
    [107, 166, 166],
    [125, 183, 183],
    [141, 140, 140],
    [179, 136, 136],
    [153, 153, 153],
    [125, 154, 154],
    [107, 166, 166],
    [125, 183, 183],
    [141, 140, 140],
    [179, 136, 136],
    [153, 153, 153],
    [125, 154, 154],
    // Chroma (15 contexts)
    [140, 170, 170],
    [139, 153, 153],
    [182, 123, 138],
    [182, 123, 138],
    [152, 107, 122],
    [136, 121, 121],
    [152, 107, 122],
    [136, 121, 121],
    [153, 167, 167],
    [136, 151, 151],
    [139, 183, 183],
    [111, 140, 140],
    [136, 151, 151],
    [139, 183, 183],
    [111, 140, 140],
    // CTX_COEFF_ABS_LEVEL_GREATER1 (124-147) — Table 9-30
    [140, 154, 154],
    [92, 196, 196],
    [137, 196, 167],
    [138, 167, 167],
    [140, 154, 154],
    [152, 152, 152],
    [138, 167, 167],
    [139, 182, 182],
    [153, 182, 182],
    [74, 134, 134],
    [149, 149, 149],
    [92, 136, 136],
    [139, 153, 153],
    [107, 121, 121],
    [122, 136, 136],
    [152, 137, 122],
    [140, 169, 169],
    [179, 194, 208],
    [166, 166, 166],
    [182, 167, 167],
    [140, 154, 154],
    [227, 167, 152],
    [122, 137, 167],
    [197, 182, 182],
    // CTX_COEFF_ABS_LEVEL_GREATER2 (148-153) — Table 9-31
    [138, 107, 107],
    [153, 167, 167],
    [136, 91, 91],
    [167, 122, 107],
    [152, 107, 107],
    [152, 167, 167],
    // CTX_RQT_ROOT_CBF (154) — Table 9-14
    [79, 79, 79],
];

// Scan order tables — spec §6.5.3.

/// Diagonal scan for 4x4.
pub const DIAG_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1), (3, 0), (1, 3),
    (2, 2), (3, 1), (2, 3), (3, 2), (3, 3),
];

/// Horizontal scan for 4x4 (used with transform skip).
pub const HORIZ_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2),
    (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
];

/// Vertical scan for 4x4.
pub const VERT_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2),
    (2, 3), (3, 0), (3, 1), (3, 2), (3, 3),
];

/// Sub-block scan for 8x8 TUs (2x2 sub-blocks of 4x4).
pub const DIAG_SCAN_2X2: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// Sub-block scan for 32x32 TUs (8x8 sub-blocks).
pub const DIAG_SCAN_8X8: [(u8, u8); 64] = [
    (0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1), (3, 0), (0, 4),
    (1, 3), (2, 2), (3, 1), (4, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (0, 6),
    (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3),
    (5, 2), (6, 1), (7, 0), (1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1), (2, 7),
    (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (4, 7),
    (5, 6), (6, 5), (7, 4), (5, 7), (6, 6), (7, 5), (6, 7), (7, 6), (7, 7),
];

/// Table 8-10 — QP chroma mapping (spec §8.6.1).
pub const QP_CHROMA_TABLE: [i8; 58] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51,
];

/// Level scale table for dequantization (spec §8.6.3).
pub const LEVEL_SCALE: [i16; 6] = [40, 45, 51, 57, 64, 72];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_offsets_are_consistent() {
        // Offsets must match the per-group sizes documented in the spec.
        assert_eq!(CTX_SAO_TYPE_IDX - CTX_SAO_MERGE_FLAG, 1);
        assert_eq!(CTX_CU_TRANSQUANT_BYPASS - CTX_SPLIT_CU_FLAG, 3);
        assert_eq!(CTX_CU_SKIP_FLAG - CTX_CU_TRANSQUANT_BYPASS, 1);
        assert_eq!(CTX_PRED_MODE_FLAG - CTX_CU_SKIP_FLAG, 3);
        assert_eq!(CTX_PART_MODE - CTX_PRED_MODE_FLAG, 1);
        assert_eq!(CTX_PREV_INTRA_LUMA_PRED - CTX_PART_MODE, 4);
        assert_eq!(CTX_MERGE_FLAG - CTX_INTRA_CHROMA_PRED_MODE, 1);
        assert_eq!(CTX_INTER_PRED_IDC - CTX_MERGE_IDX, 1);
        assert_eq!(CTX_REF_IDX - CTX_INTER_PRED_IDC, 5);
        assert_eq!(CTX_MVP_FLAG - CTX_REF_IDX, 2);
        assert_eq!(CTX_CBF_LUMA - CTX_SPLIT_TRANSFORM_FLAG, 3);
        assert_eq!(CTX_CBF_CHROMA - CTX_CBF_LUMA, 2);
        assert_eq!(CTX_ABS_MVD_GREATER0 - CTX_CBF_CHROMA, 5);
        assert_eq!(CTX_CU_QP_DELTA_ABS - CTX_ABS_MVD_GREATER1, 1);
        assert_eq!(CTX_TRANSFORM_SKIP_FLAG - CTX_CU_QP_DELTA_ABS, 2);
        assert_eq!(CTX_LAST_SIG_COEFF_Y - CTX_LAST_SIG_COEFF_X, 18);
        assert_eq!(CTX_CODED_SUB_BLOCK_FLAG - CTX_LAST_SIG_COEFF_Y, 18);
        assert_eq!(CTX_SIG_COEFF_FLAG - CTX_CODED_SUB_BLOCK_FLAG, 4);
        assert_eq!(CTX_COEFF_ABS_LEVEL_GREATER1 - CTX_SIG_COEFF_FLAG, 42);
        assert_eq!(CTX_COEFF_ABS_LEVEL_GREATER2 - CTX_COEFF_ABS_LEVEL_GREATER1, 24);
        assert_eq!(CTX_RQT_ROOT_CBF - CTX_COEFF_ABS_LEVEL_GREATER2, 6);
        assert_eq!(NUM_CABAC_CONTEXTS - CTX_RQT_ROOT_CBF, 1);
    }

    #[test]
    fn table_shapes() {
        assert_eq!(RANGE_TAB_LPS.len(), 64);
        assert_eq!(TRANS_IDX_MPS.len(), 64);
        assert_eq!(TRANS_IDX_LPS.len(), 64);
        assert_eq!(CABAC_INIT_VALUES.len(), NUM_CABAC_CONTEXTS);
        assert_eq!(QP_CHROMA_TABLE.len(), 58);
        // Range tab sanity: monotonically non-increasing per column.
        for row in 1..RANGE_TAB_LPS.len() {
            let prev = &RANGE_TAB_LPS[row - 1];
            let cur = &RANGE_TAB_LPS[row];
            for (a, b) in prev.iter().zip(cur.iter()) {
                assert!(b <= a);
            }
        }
    }
}
