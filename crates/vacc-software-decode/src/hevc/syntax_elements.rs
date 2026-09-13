//! Port of `hevc/decoding/syntax_elements.{h,cpp}` — CABAC syntax element
//! decoding, spec §9.3.2, §9.3.3. Each function decodes one syntax element
//! using the appropriate binarization and context model(s).

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

use crate::hevc::cabac::CabacEngine;
use crate::hevc::cabac_tables::{
    CTX_ABS_MVD_GREATER0, CTX_ABS_MVD_GREATER1, CTX_CBF_CHROMA, CTX_CBF_LUMA,
    CTX_CODED_SUB_BLOCK_FLAG, CTX_COEFF_ABS_LEVEL_GREATER1, CTX_COEFF_ABS_LEVEL_GREATER2,
    CTX_CU_QP_DELTA_ABS, CTX_CU_SKIP_FLAG, CTX_CU_TRANSQUANT_BYPASS,
    CTX_INTRA_CHROMA_PRED_MODE, CTX_INTER_PRED_IDC, CTX_MERGE_FLAG, CTX_MERGE_IDX, CTX_MVP_FLAG,
    CTX_PART_MODE,
    CTX_PRED_MODE_FLAG, CTX_PREV_INTRA_LUMA_PRED, CTX_REF_IDX, CTX_RQT_ROOT_CBF, CTX_SAO_MERGE_FLAG,
    CTX_SAO_TYPE_IDX, CTX_SIG_COEFF_FLAG, CTX_SPLIT_CU_FLAG, CTX_SPLIT_TRANSFORM_FLAG,
    CTX_TRANSFORM_SKIP_FLAG,
};
use crate::hevc::types::{Mv, PartMode, PredMode};

/// §9.3.3.4 — end_of_slice_segment_flag (decoded via terminate).
#[inline]
pub fn decode_end_of_slice_segment_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_terminate()
}

/// §7.3.8.5 — sao_merge_left_flag, sao_merge_up_flag
#[inline]
pub fn decode_sao_merge_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_SAO_MERGE_FLAG)
}

/// §7.3.8.5 — sao_type_idx_luma, sao_type_idx_chroma
pub fn decode_sao_type_idx(cabac: &mut CabacEngine) -> i32 {
    let bin0 = cabac.decode_decision(CTX_SAO_TYPE_IDX);
    if bin0 == 0 {
        return 0;
    }
    let bin1 = cabac.decode_bypass();
    if bin1 != 0 { 2 } else { 1 }
}

/// §7.3.8.6 — split_cu_flag: 1 bin, context depends on neighbours.
pub fn decode_split_cu_flag(cabac: &mut CabacEngine, ctx_inc: i32) -> i32 {
    cabac.decode_decision(CTX_SPLIT_CU_FLAG + ctx_inc as usize)
}

/// §7.3.8.6 — cu_transquant_bypass_flag
#[inline]
pub fn decode_cu_transquant_bypass_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_CU_TRANSQUANT_BYPASS)
}

/// §7.3.8.6 — cu_skip_flag: 1 bin, context depends on neighbours.
pub fn decode_cu_skip_flag(cabac: &mut CabacEngine, ctx_inc: i32) -> i32 {
    cabac.decode_decision(CTX_CU_SKIP_FLAG + ctx_inc as usize)
}

/// §7.3.8.7 — pred_mode_flag
#[inline]
pub fn decode_pred_mode_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_PRED_MODE_FLAG)
}

/// §7.3.8.7 — part_mode (depends on pred_mode, log2CbSize, amp_enabled).
pub fn decode_part_mode(
    cabac: &mut CabacEngine,
    pred_mode: PredMode,
    log2_cb_size: i32,
    log2_min_cb_size: i32,
    amp_enabled: bool,
) -> i32 {
    if pred_mode == PredMode::Intra {
        // Intra: only 2Nx2N (bin=1) or NxN (bin=0), NxN only at min CB size
        if log2_cb_size == log2_min_cb_size {
            let bin = cabac.decode_decision(CTX_PART_MODE);
            return if bin != 0 {
                PartMode::Part2Nx2N as i32
            } else {
                PartMode::PartNxN as i32
            };
        }
        return PartMode::Part2Nx2N as i32;
    }

    // Inter modes
    let bin0 = cabac.decode_decision(CTX_PART_MODE);
    if bin0 != 0 {
        return PartMode::Part2Nx2N as i32;
    }

    if log2_cb_size == log2_min_cb_size {
        // Table 9-45: binarization depends on log2CbSize
        if log2_cb_size > 3 {
            // log2CbSize > 3: NxN available — "01"=2NxN, "001"=Nx2N, "000"=NxN
            let bin1 = cabac.decode_decision(CTX_PART_MODE + 1);
            if bin1 != 0 {
                return PartMode::Part2NxN as i32;
            }
            let bin2 = cabac.decode_decision(CTX_PART_MODE + 2);
            if bin2 != 0 {
                return PartMode::PartNx2N as i32;
            }
            return PartMode::PartNxN as i32;
        } else {
            // log2CbSize == 3: no NxN for inter — "01"=2NxN, "00"=Nx2N
            let bin1 = cabac.decode_decision(CTX_PART_MODE + 1);
            if bin1 != 0 {
                return PartMode::Part2NxN as i32;
            }
            return PartMode::PartNx2N as i32;
        }
    }

    if log2_cb_size > 3 {
        let bin1 = cabac.decode_decision(CTX_PART_MODE + 1);
        if bin1 != 0 {
            if !amp_enabled {
                return PartMode::Part2NxN as i32;
            }
            let bin2 = cabac.decode_decision(CTX_PART_MODE + 3);
            if bin2 != 0 {
                return PartMode::Part2NxN as i32;
            }
            let bin3 = cabac.decode_bypass();
            return if bin3 != 0 {
                PartMode::Part2NxnD as i32
            } else {
                PartMode::Part2NxnU as i32
            };
        } else {
            if !amp_enabled {
                return PartMode::PartNx2N as i32;
            }
            let bin2 = cabac.decode_decision(CTX_PART_MODE + 3);
            if bin2 != 0 {
                return PartMode::PartNx2N as i32;
            }
            let bin3 = cabac.decode_bypass();
            return if bin3 != 0 {
                PartMode::PartNRx2N as i32
            } else {
                PartMode::PartNlx2N as i32
            };
        }
    }

    // log2CbSize == 3, no AMP
    let bin1 = cabac.decode_decision(CTX_PART_MODE + 1);
    if bin1 != 0 {
        return PartMode::Part2NxN as i32;
    }
    PartMode::PartNx2N as i32
}

/// §7.3.8.8 — prev_intra_luma_pred_flag
#[inline]
pub fn decode_prev_intra_luma_pred_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_PREV_INTRA_LUMA_PRED)
}

/// §7.3.8.8 — mpm_idx (FL, 2 bins bypass)
#[inline]
pub fn decode_mpm_idx(cabac: &mut CabacEngine) -> i32 {
    // TR binarization cMax=2: bin0 bypass, if 1 then bin1 bypass
    let bin0 = cabac.decode_bypass();
    if bin0 == 0 {
        return 0;
    }
    let bin1 = cabac.decode_bypass();
    if bin1 != 0 { 2 } else { 1 }
}

/// §7.3.8.8 — rem_intra_luma_pred_mode (FL, 5 bins bypass)
#[inline]
pub fn decode_rem_intra_luma_pred_mode(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_bypass_bins(5)
}

/// §7.3.8.8 — intra_chroma_pred_mode: 1 context + 2 bypass
pub fn decode_intra_chroma_pred_mode(cabac: &mut CabacEngine) -> i32 {
    let bin0 = cabac.decode_decision(CTX_INTRA_CHROMA_PRED_MODE);
    if bin0 == 0 {
        return 4; // DM mode (derived from luma)
    }
    cabac.decode_bypass_bins(2) // 0=Planar, 1=V, 2=H, 3=DC
}

/// §7.3.8.9 — merge_flag
#[inline]
pub fn decode_merge_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_MERGE_FLAG)
}

/// §7.3.8.9 — merge_idx (TU, cMax = MaxNumMergeCand - 1)
pub fn decode_merge_idx(cabac: &mut CabacEngine, max_num_merge_cand: i32) -> i32 {
    if max_num_merge_cand <= 1 {
        return 0;
    }
    let bin0 = cabac.decode_decision(CTX_MERGE_IDX);
    if bin0 == 0 {
        return 0;
    }
    // Remaining bins are bypass (TU)
    let mut idx = 1;
    for _ in 1..max_num_merge_cand - 1 {
        if cabac.decode_bypass() == 0 {
            break;
        }
        idx += 1;
    }
    idx
}

/// §7.3.8.11 — split_transform_flag
pub fn decode_split_transform_flag(cabac: &mut CabacEngine, log2_trafo_size: i32) -> i32 {
    // Match C++: index computed in i32 (5 - log2TrafoSize can be negative).
    let ctx_idx = (CTX_SPLIT_TRANSFORM_FLAG as i32 + (5 - log2_trafo_size)) as usize;
    cabac.decode_decision(ctx_idx)
}

/// §7.3.8.11 — cbf_luma
pub fn decode_cbf_luma(cabac: &mut CabacEngine, trafo_depth: i32) -> i32 {
    let ctx_idx = CTX_CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 };
    cabac.decode_decision(ctx_idx)
}

/// §7.3.8.11 — cbf_cb, cbf_cr (shared contexts) — Table 9-48: ctxInc = trafoDepth
pub fn decode_cbf_chroma(cabac: &mut CabacEngine, trafo_depth: i32) -> i32 {
    let ctx_idx = CTX_CBF_CHROMA + (trafo_depth.min(4)) as usize;
    cabac.decode_decision(ctx_idx)
}

/// §7.3.8.11 — cu_qp_delta_abs: TU+EGk
pub fn decode_cu_qp_delta(cabac: &mut CabacEngine) -> i32 {
    // Prefix: TU cMax=5, context 0 for first bin, context 1 for rest
    let bin = cabac.decode_decision(CTX_CU_QP_DELTA_ABS);
    if bin == 0 {
        return 0;
    }
    let mut prefix = 1;
    for _ in 1..5 {
        let bin = cabac.decode_decision(CTX_CU_QP_DELTA_ABS + 1);
        if bin == 0 {
            break;
        }
        prefix += 1;
    }

    let mut val = prefix;
    if prefix >= 5 {
        // Suffix: EG0 bypass
        let mut k = 0;
        while cabac.decode_bypass() != 0 {
            k += 1;
        }
        let suffix = (1 << k) - 1 + cabac.decode_bypass_bins(k);
        val = prefix + suffix;
    }

    if val == 0 {
        return 0;
    }
    // Sign
    let sign = cabac.decode_bypass();
    if sign != 0 { -val } else { val }
}

/// §7.3.8.11 — transform_skip_flag: 1 bin
pub fn decode_transform_skip_flag(cabac: &mut CabacEngine, c_idx: i32) -> i32 {
    let ctx_idx = CTX_TRANSFORM_SKIP_FLAG + if c_idx > 0 { 1 } else { 0 };
    cabac.decode_decision(ctx_idx)
}

/// §9.3.3.5 — last_sig_coeff prefix (X or Y).
/// `ctx_offset` = CTX_LAST_SIG_COEFF_X or CTX_LAST_SIG_COEFF_Y.
pub fn decode_last_sig_coeff_prefix(
    cabac: &mut CabacEngine,
    ctx_offset: i32,
    c_idx: i32,
    log2_trafo_size: i32,
) -> i32 {
    // Context index offset and shift depend on component and size
    let (ctx_off, ctx_shift) = if c_idx == 0 {
        // Luma
        (
            3 * (log2_trafo_size - 2) + ((log2_trafo_size - 1) >> 2),
            (log2_trafo_size + 1) >> 2,
        )
    } else {
        // Chroma
        (15, log2_trafo_size - 2)
    };

    let max_bins = (log2_trafo_size << 1) - 1;
    let mut prefix = 0;
    for i in 0..max_bins {
        let ctx_idx = ctx_offset + ctx_off + (i >> ctx_shift);
        let bin = cabac.decode_decision(ctx_idx as usize);
        if bin == 0 {
            break;
        }
        prefix += 1;
    }
    prefix
}

/// §9.3.3.5 — last_sig_coeff suffix (bypass, EG0-like)
pub fn decode_last_sig_coeff_suffix(cabac: &mut CabacEngine, prefix: i32) -> i32 {
    if prefix < 4 {
        return 0; // no suffix needed
    }
    let num_bins = (prefix >> 1) - 1;
    cabac.decode_bypass_bins(num_bins)
}

/// §7.3.8.12 — coded_sub_block_flag
pub fn decode_coded_sub_block_flag(cabac: &mut CabacEngine, ctx_inc: i32) -> i32 {
    cabac.decode_decision(CTX_CODED_SUB_BLOCK_FLAG + ctx_inc as usize)
}

/// §7.3.8.12 — sig_coeff_flag
pub fn decode_sig_coeff_flag(cabac: &mut CabacEngine, ctx_inc: i32) -> i32 {
    cabac.decode_decision(CTX_SIG_COEFF_FLAG + ctx_inc as usize)
}

/// §7.3.8.12 — coeff_abs_level_greater1_flag
pub fn decode_coeff_abs_level_greater1_flag(
    cabac: &mut CabacEngine,
    ctx_set: i32,
    greater1_ctx: i32,
    c_idx: i32,
) -> i32 {
    let ctx_idx = if c_idx == 0 {
        CTX_COEFF_ABS_LEVEL_GREATER1 + (ctx_set * 4 + greater1_ctx) as usize
    } else {
        CTX_COEFF_ABS_LEVEL_GREATER1 + 16 + (ctx_set * 4 + greater1_ctx) as usize
    };
    cabac.decode_decision(ctx_idx)
}

/// §7.3.8.12 — coeff_abs_level_greater2_flag
pub fn decode_coeff_abs_level_greater2_flag(cabac: &mut CabacEngine, ctx_set: i32, c_idx: i32) -> i32 {
    let ctx_idx = if c_idx == 0 {
        CTX_COEFF_ABS_LEVEL_GREATER2 + ctx_set as usize
    } else {
        CTX_COEFF_ABS_LEVEL_GREATER2 + 4 + (ctx_set & 1) as usize
    };
    cabac.decode_decision(ctx_idx)
}

/// §9.3.3.11 — coeff_abs_level_remaining (Rice + EGk bypass)
pub fn decode_coeff_abs_level_remaining(cabac: &mut CabacEngine, c_rice_param: i32) -> i32 {
    // Prefix: unary in bypass
    let mut prefix = 0;
    while prefix < 4 && cabac.decode_bypass() != 0 {
        prefix += 1;
    }

    if prefix < 4 {
        // Suffix: FL with cRiceParam bins
        let suffix = if c_rice_param > 0 {
            cabac.decode_bypass_bins(c_rice_param)
        } else {
            0
        };
        (prefix << c_rice_param) + suffix
    } else {
        // EGk suffix with k = cRiceParam + 1
        // §9.3.3.3: read 1's incrementing k, then 0, then k bits
        let mut k = c_rice_param + 1;
        while cabac.decode_bypass() != 0 {
            k += 1;
        }
        let mut suffix = cabac.decode_bypass_bins(k);
        // Reconstruct: sum of (1<<k_j) for each '1' read + final k bits
        // = (1 << k) - (1 << (cRiceParam + 1)) + suffix
        suffix += (1 << k) - (1 << (c_rice_param + 1));
        (4 << c_rice_param) + suffix
    }
}

/// §7.3.8.12 — coeff_sign_flag (bypass)
#[inline]
pub fn decode_coeff_sign_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_bypass()
}

/// §7.3.8.5 — rqt_root_cbf (for inter CU)
#[inline]
pub fn decode_rqt_root_cbf(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_RQT_ROOT_CBF)
}

/// §7.3.8.9 — mvp_l0_flag / mvp_l1_flag
#[inline]
pub fn decode_mvp_flag(cabac: &mut CabacEngine) -> i32 {
    cabac.decode_decision(CTX_MVP_FLAG)
}

/// §9.3.3.9 — inter_pred_idc: spec Table 9-47.
/// 0=PRED_L0, 1=PRED_L1, 2=PRED_BI
pub fn decode_inter_pred_idc(cabac: &mut CabacEngine, n_pb_w: i32, n_pb_h: i32, ct_depth: i32) -> i32 {
    // §9.3.4.2.3 Table 9-48: binIdx 0 ctxInc = (nPbW+nPbH != 12) ? CtDepth : 4
    let ctx_inc = if n_pb_w + n_pb_h != 12 { ct_depth } else { 4 };
    let bin0 = cabac.decode_decision(CTX_INTER_PRED_IDC + ctx_inc as usize);
    // Table 9-47: (nPbW+nPbH)==12: "0"→PRED_L0, "1"→PRED_L1 (no PRED_BI)
    if n_pb_w + n_pb_h == 12 {
        return if bin0 != 0 { 1 } else { 0 }; // PRED_L1 or PRED_L0
    }
    // Table 9-47: (nPbW+nPbH)!=12: "1"→PRED_BI, "00"→PRED_L0, "01"→PRED_L1
    if bin0 == 1 {
        return 2; // PRED_BI
    }
    let bin1 = cabac.decode_decision(CTX_INTER_PRED_IDC + 4);
    if bin1 != 0 { 1 } else { 0 } // PRED_L1 or PRED_L0
}

/// §7.3.8.9 — ref_idx_l0 / ref_idx_l1: TU binarization, max=numRefIdxActive
pub fn decode_ref_idx(cabac: &mut CabacEngine, num_ref_idx_active: i32) -> i32 {
    if num_ref_idx_active == 0 {
        return 0;
    }
    let bin0 = cabac.decode_decision(CTX_REF_IDX);
    if bin0 == 0 {
        return 0;
    }
    let mut idx = 1;
    if num_ref_idx_active > 1 {
        let bin1 = cabac.decode_decision(CTX_REF_IDX + 1);
        if bin1 == 0 {
            return 1;
        }
        idx = 2;
        // Remaining bins: bypass (TU)
        for _ in 2..num_ref_idx_active {
            if cabac.decode_bypass() == 0 {
                break;
            }
            idx += 1;
        }
    }
    idx
}

/// §7.3.8.10 — mvd_coding: abs_mvd_greater0, abs_mvd_greater1, abs_mvd_minus2, sign
pub fn decode_mvd(cabac: &mut CabacEngine) -> Mv {
    let mut mvd = Mv::default();

    // §9.3.3.3 eq 9-13: k-th order Exp-Golomb with k=1 (Table 9-43)
    let decode_eg1 = |cabac: &mut CabacEngine| -> i32 {
        let mut k = 1;
        let mut abs_v = 0;
        while cabac.decode_bypass() != 0 {
            abs_v += 1 << k;
            k += 1;
        }
        if k > 0 {
            abs_v += cabac.decode_bypass_bins(k);
        }
        abs_v
    };

    // abs_mvd_greater0_flag[0], abs_mvd_greater0_flag[1]
    let g0_h = cabac.decode_decision(CTX_ABS_MVD_GREATER0);
    let g0_v = cabac.decode_decision(CTX_ABS_MVD_GREATER0);

    // abs_mvd_greater1_flag[0], abs_mvd_greater1_flag[1]
    let (g1_h, g1_v) = (
        if g0_h != 0 { cabac.decode_decision(CTX_ABS_MVD_GREATER1) } else { 0 },
        if g0_v != 0 { cabac.decode_decision(CTX_ABS_MVD_GREATER1) } else { 0 },
    );

    // §7.3.8.9: H component (abs_mvd_minus2[0] + mvd_sign_flag[0])
    if g0_h != 0 {
        let mut abs_h = g1_h + 1;
        if g1_h != 0 {
            abs_h += decode_eg1(cabac);
        }
        let sign = cabac.decode_bypass(); // mvd_sign_flag[0]
        mvd.x = if sign != 0 { -abs_h } else { abs_h } as i16;
    }

    // §7.3.8.9: V component (abs_mvd_minus2[1] + mvd_sign_flag[1])
    if g0_v != 0 {
        let mut abs_v = g1_v + 1;
        if g1_v != 0 {
            abs_v += decode_eg1(cabac);
        }
        let sign = cabac.decode_bypass(); // mvd_sign_flag[1]
        mvd.y = if sign != 0 { -abs_v } else { abs_v } as i16;
    }

    mvd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::bitreader::BitstreamReader;
    use crate::hevc::goldens;

    use crate::hevc::cabac::CabacEngine;
    use crate::hevc::cabac_tables::NUM_CABAC_CONTEXTS;
    use crate::hevc::types::PredMode;

    /// Deterministic xorshift64* RNG (same scheme as other hevc tests).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Op dispatch mirrors the C++ hevc.js test oracle (`hevc_test_api.cpp`,
    /// since removed); op codes keep that historical numbering.
    fn run_ops_rust(
        data: &[u8],
        slice_type: i32,
        qp: i32,
        cabac_init_flag: bool,
        ops: &[u8],
        args: &[i32],
    ) -> (Vec<i32>, u16, u16, Vec<u8>) {
        let mut bs = BitstreamReader::new(data);
        let mut cabac = CabacEngine::new(&mut bs);
        cabac.init_decoder();
        cabac.init_contexts(slice_type, qp, cabac_init_flag);

        let mut out: Vec<i32> = Vec::new();
        for (&op, &a) in ops.iter().zip(args.iter()) {
            match op {
                0 => out.push(cabac.decode_decision((a & 127) as usize)),
                1 => out.push(cabac.decode_bypass()),
                2 => out.push(cabac.decode_terminate()),
                3 => out.push(cabac.decode_bypass_bins(a & 15)),
                4 => cabac.align_bypass(),
                5 => out.push(decode_sao_type_idx(&mut cabac)),
                6 => out.push(decode_split_cu_flag(&mut cabac, a & 3)),
                7 => out.push(decode_cu_skip_flag(&mut cabac, a & 3)),
                8 => {
                    let pred_mode = if (a & 1) == 0 { PredMode::Inter } else { PredMode::Intra };
                    let log2_cb_size = (a >> 2) & 7;
                    let log2_min_cb_size = (a >> 5) & 7;
                    let amp = (a >> 8) & 1 != 0;
                    out.push(decode_part_mode(
                        &mut cabac,
                        pred_mode,
                        log2_cb_size,
                        log2_min_cb_size,
                        amp,
                    ));
                }
                9 => out.push(decode_intra_chroma_pred_mode(&mut cabac)),
                10 => out.push(decode_merge_idx(&mut cabac, a & 7)),
                11 => {
                    let n_pb_w = a & 127;
                    let n_pb_h = (a >> 7) & 127;
                    let ct_depth = (a >> 14) & 15;
                    out.push(decode_inter_pred_idc(&mut cabac, n_pb_w, n_pb_h, ct_depth));
                }
                12 => out.push(decode_ref_idx(&mut cabac, a & 31)),
                13 => {
                    let mv = decode_mvd(&mut cabac);
                    out.push(mv.x as i32);
                    out.push(mv.y as i32);
                }
                14 => out.push(decode_split_transform_flag(&mut cabac, a & 7)),
                15 => out.push(decode_cbf_luma(&mut cabac, a & 3)),
                16 => out.push(decode_cbf_chroma(&mut cabac, a & 7)),
                17 => out.push(decode_cu_qp_delta(&mut cabac)),
                18 => out.push(decode_transform_skip_flag(&mut cabac, a & 1)),
                19 => {
                    let ctx_offset = if (a & 1) != 0 { 60 } else { 42 };
                    let c_idx = (a >> 1) & 1;
                    let log2_trafo_size = 2 + ((a >> 2) & 3);
                    out.push(decode_last_sig_coeff_prefix(
                        &mut cabac,
                        ctx_offset,
                        c_idx,
                        log2_trafo_size,
                    ));
                }
                20 => out.push(decode_last_sig_coeff_suffix(&mut cabac, a & 15)),
                21 => out.push(decode_coded_sub_block_flag(&mut cabac, a & 3)),
                22 => out.push(decode_sig_coeff_flag(&mut cabac, a & 41)),
                23 => {
                    let ctx_set = a & 1;
                    let greater1_ctx = (a >> 1) & 3;
                    let c_idx = (a >> 5) & 1;
                    out.push(decode_coeff_abs_level_greater1_flag(
                        &mut cabac,
                        ctx_set,
                        greater1_ctx,
                        c_idx,
                    ));
                }
                24 => {
                    let ctx_set = a & 1;
                    let c_idx = (a >> 1) & 1;
                    out.push(decode_coeff_abs_level_greater2_flag(&mut cabac, ctx_set, c_idx));
                }
                25 => out.push(decode_coeff_abs_level_remaining(&mut cabac, a & 7)),
                _ => unreachable!("oracle only emits ops 0..=25"),
            }
        }

        let final_range = cabac.dbg_range();
        let final_offset = cabac.dbg_offset();
        let mut final_ctx = vec![0u8; NUM_CABAC_CONTEXTS * 2];
        for i in 0..NUM_CABAC_CONTEXTS {
            let c = cabac.context(i);
            final_ctx[2 * i] = c.p_state_idx;
            final_ctx[2 * i + 1] = c.val_mps;
        }
        (out, final_range, final_offset, final_ctx)
    }

    /// Serialize one op run's full state (outputs + final range/offset/contexts).
    fn push_run_state(out: &mut Vec<u8>, res: &(Vec<i32>, u16, u16, Vec<u8>)) {
        for v in &res.0 {
            goldens::push_i32(out, *v);
        }
        goldens::push_u16(out, res.1);
        goldens::push_u16(out, res.2);
        out.extend_from_slice(&res.3);
    }

    fn compute_cabac_syntax() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x000C_ABAC_0001);
        const ITERS: u32 = 400;
        let mut buf = Vec::new();
        for _it in 0..ITERS {
            // Random bitstream (4 KiB). The first byte is kept <= 0xFE so the CABAC
            // init offset (first 9 bits) is < 510 = initial range: a valid init state.
            // (offset >= range at init is a latent quirk that corrupts the segment and
            // is never reached by valid slice segments' first context-coded bin.)
            let mut data: Vec<u8> = (0..4096u32).map(|_| rng.below(256) as u8).collect();
            data[0] = rng.below(255) as u8; // 0..254 => init offset <= 509 < 510
            let slice_type = rng.below(3) as i32; // 0=B, 1=P, 2=I
            let qp = rng.below(52) as i32; // 0..51
            let cabac_init_flag = rng.below(2) != 0;

            // Realistic CABAC op sequence (1..30 ops). The first op is a context-coded
            // decision (as in a real slice segment), establishing offset < range. We
            // exclude decode_terminate (op 2, segment-ending) and align_bypass (op 4) so
            // the `offset < range` invariant holds before every bypass — valid CABAC
            // never has a bare bypass while offset > range. Args cover bits 0..23.
            const ALLOWED_OPS: [u8; 24] = [
                0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25,
            ];
            let n_ops = (rng.below(30) + 1) as usize;
            let mut ops = Vec::with_capacity(n_ops);
            let mut args = Vec::with_capacity(n_ops);
            for i in 0..n_ops {
                let op = if i == 0 { 0u8 } else { ALLOWED_OPS[rng.below(24) as usize] };
                ops.push(op);
                args.push(rng.below(1u64 << 24) as i32);
            }

            let res = run_ops_rust(&data, slice_type, qp, cabac_init_flag, &ops, &args);
            push_run_state(&mut buf, &res);
        }
        vec![("cabac::syntax".to_string(), buf)]
    }

    #[test]
    fn cabac_syntax_matches_golden() {
        for (key, data) in compute_cabac_syntax() {
            goldens::assert_golden(&key, &data);
        }
    }

    /// Long pure-bypass sequence: exercises decode_bypass over hundreds of
    /// consecutive bypass bins (offset/range trajectories).
    fn compute_long_bypass() -> Vec<(String, Vec<u8>)> {
        // Deterministic bitstream (0xA5 = 10100101 repeating).
        let data = vec![0xA5u8; 256];
        const N: usize = 600;
        let ops = vec![1u8; N]; // op 1 = decode_bypass
        let args = vec![0i32; N];

        let res = run_ops_rust(&data, 2, 26, false, &ops, &args);

        // A pure bypass sequence with range in [256,511] and a valid init (offset <
        // range) cannot produce runs longer than ~log2(range); guard against the
        // offset>=range invariant-corruption that would make bypass return 1 forever.
        let max_run = |v: &[i32]| -> usize {
            let mut best = 0;
            let mut cur = 0;
            for &b in v {
                cur = if b != 0 { cur + 1 } else { 0 };
                best = best.max(cur);
            }
            best
        };
        assert!(max_run(&res.0) < 32, "anomalous bypass run (invariant corruption)");

        let mut buf = Vec::new();
        push_run_state(&mut buf, &res);
        vec![("cabac::long_bypass".to_string(), buf)]
    }

    #[test]
    fn long_bypass_sequence_matches_golden() {
        for (key, data) in compute_long_bypass() {
            goldens::assert_golden(&key, &data);
        }
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (k, b) in compute_cabac_syntax() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_long_bypass() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        v
    }
}
