//! Port of `hevc/decoding/residual_coding.cpp` — residual_coding
//! (§7.3.8.11) with scan-order generation and sig_coeff_flag context
//! derivation (§9.3.4.2.8).

use crate::hevc::cabac_tables::{DIAG_SCAN_2X2, DIAG_SCAN_4X4, DIAG_SCAN_8X8, HORIZ_SCAN_4X4, VERT_SCAN_4X4};
use crate::hevc::coding_tree::DecodingContext;
use crate::hevc::syntax_elements::*;
use crate::hevc::types::PredMode;

// ============================================================
// Scan order generation
// ============================================================

// Pre-computed horizontal/vertical sub-block scan tables
// (diagonal scans are in cabac_tables: DIAG_SCAN_2X2, 4X4, 8X8).
const HORIZ_SB_2X2: [(u8, u8); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
const HORIZ_SB_4X4: [(u8, u8); 16] = [
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2),
    (0, 3), (1, 3), (2, 3), (3, 3),
];
const HORIZ_SB_8X8: [(u8, u8); 64] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
    (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6),
    (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
];
const VERT_SB_2X2: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
const VERT_SB_4X4: [(u8, u8); 16] = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3),
];
const VERT_SB_8X8: [(u8, u8); 64] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
];
const DIAG_SCAN_1X1: [(u8, u8); 1] = [(0, 0)];

/// Return pointer to sub-block scan table (no copy).
fn get_sub_block_scan(log2_trafo_size: i32, scan_idx: i32) -> &'static [(u8, u8)] {
    let sb_size = (1 << log2_trafo_size) >> 2;
    let sb_size = if sb_size <= 0 { 1 } else { sb_size };

    match scan_idx {
        1 => {
            if sb_size <= 1 {
                &DIAG_SCAN_1X1
            } else if sb_size == 2 {
                &HORIZ_SB_2X2
            } else if sb_size == 4 {
                &HORIZ_SB_4X4
            } else {
                &HORIZ_SB_8X8
            }
        }
        2 => {
            if sb_size <= 1 {
                &DIAG_SCAN_1X1
            } else if sb_size == 2 {
                &VERT_SB_2X2
            } else if sb_size == 4 {
                &VERT_SB_4X4
            } else {
                &VERT_SB_8X8
            }
        }
        _ => {
            if sb_size <= 1 {
                &DIAG_SCAN_1X1
            } else if sb_size == 2 {
                &DIAG_SCAN_2X2
            } else if sb_size == 4 {
                &DIAG_SCAN_4X4
            } else {
                &DIAG_SCAN_8X8
            }
        }
    }
}

/// Return pointer to coefficient scan table (no copy).
fn get_coeff_scan(scan_idx: i32) -> &'static [(u8, u8)] {
    match scan_idx {
        1 => &HORIZ_SCAN_4X4,
        2 => &VERT_SCAN_4X4,
        _ => &DIAG_SCAN_4X4,
    }
}

/// Derive scan index from intra mode and TU size (§8.4.4.2.1 / §7.4.9.11).
/// scanIdx: 0=up-right diagonal, 1=horizontal, 2=vertical.
fn derive_scan_idx(pred_mode_intra: i32, mode_dependent: bool) -> i32 {
    if mode_dependent {
        if (6..=14).contains(&pred_mode_intra) {
            return 2; // vertical scan
        }
        if (22..=30).contains(&pred_mode_intra) {
            return 1; // horizontal scan
        }
    }
    0 // diagonal
}

// ============================================================
// sig_coeff_flag context derivation (§9.3.4.2.8)
// The most complex context derivation in HEVC CABAC
// ============================================================

/// §9.3.4.2.5 — derivation of ctxInc for sig_coeff_flag (eq 9-40..9-55).
#[allow(clippy::too_many_arguments)]
fn derive_sig_coeff_flag_ctx(
    c_idx: i32,
    log2_trafo_size: i32,
    x_c: i32,
    y_c: i32,
    scan_idx: i32,
    coded_sub_block_flag: &[i32],
    num_sb_per_side: i32,
    transform_skip_or_bypass: bool,
) -> i32 {
    // Table 9-50 — spec has only 15 entries (i=0..14), position 15 is never
    // accessed (it's always lastScanPos which is implicitly significant).
    // 99 = sentinel.
    const CTX_IDX_MAP: [i32; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 99];

    let mut sig_ctx: i32;

    // eq 9-40: transform_skip_context_enabled + (transform_skip || bypass)
    if transform_skip_or_bypass {
        sig_ctx = if c_idx == 0 { 42 } else { 16 }; // eq 9-40
    }
    // eq 9-41: 4x4 TU
    else if log2_trafo_size == 2 {
        sig_ctx = CTX_IDX_MAP[((y_c << 2) + x_c) as usize]; // eq 9-41
    }
    // eq 9-42: DC position
    else if x_c + y_c == 0 {
        sig_ctx = 0; // eq 9-42
    }
    // eq 9-43 to 9-53: non-4x4, non-DC
    else {
        let x_s = x_c >> 2; // sub-block location
        let y_s = y_c >> 2;

        let mut prev_csbf = 0i32; // eq 9-43, 9-44
        if x_s < num_sb_per_side - 1 {
            prev_csbf += coded_sub_block_flag[(y_s * num_sb_per_side + (x_s + 1)) as usize];
        }
        if y_s < num_sb_per_side - 1 {
            prev_csbf += coded_sub_block_flag[((y_s + 1) * num_sb_per_side + x_s) as usize] << 1;
        }

        let x_p = x_c & 3; // inner sub-block location
        let y_p = y_c & 3;

        match prev_csbf {
            // eq 9-45 to 9-48
            0 => sig_ctx = if x_p + y_p == 0 { 2 } else if x_p + y_p < 3 { 1 } else { 0 },
            1 => sig_ctx = if y_p == 0 { 2 } else if y_p == 1 { 1 } else { 0 },
            2 => sig_ctx = if x_p == 0 { 2 } else if x_p == 1 { 1 } else { 0 },
            _ => sig_ctx = 2,
        }

        if c_idx == 0 {
            if (x_s + y_s) > 0 {
                sig_ctx += 3; // eq 9-49
            }
            if log2_trafo_size == 3 {
                sig_ctx += if scan_idx == 0 { 9 } else { 15 }; // eq 9-50
            } else {
                sig_ctx += 21; // eq 9-51
            }
        } else {
            if log2_trafo_size == 3 {
                sig_ctx += 9; // eq 9-52
            } else {
                sig_ctx += 12; // eq 9-53
            }
        }
    }

    // eq 9-54, 9-55: ctxInc derivation (27 = chroma offset per spec eq 9-55)
    if c_idx == 0 {
        sig_ctx // eq 9-54
    } else {
        27 + sig_ctx // eq 9-55
    }
}

// ============================================================
// residual_coding (§7.3.8.11)
// ============================================================

pub fn decode_residual_coding(
    ctx: &mut DecodingContext,
    x0: i32,
    y0: i32,
    log2_trafo_size: i32,
    c_idx: i32,
    coefficients: &mut [i16],
) {
    let tr_size = 1 << log2_trafo_size;
    coefficients.fill(0);

    // §7.4.9.11 — scanIdx derivation
    let cu_pred_mode = ctx.cu_at(x0, y0).pred_mode;
    let mut scan_idx = 0i32;
    if cu_pred_mode == PredMode::Intra
        && (log2_trafo_size == 2
            || (log2_trafo_size == 3 && c_idx == 0)
            || (log2_trafo_size == 3 && ctx.sps.chroma_array_type == 3))
    {
        let pred_mode_intra = if c_idx == 0 {
            ctx.intra_mode_at(x0, y0)
        } else {
            ctx.chroma_mode_at(x0, y0)
        };
        scan_idx = derive_scan_idx(pred_mode_intra, true);
    }

    // Last significant coefficient position
    // §7.3.8.11: For vertical scan, swap width/height for context derivation,
    // then swap the decoded X/Y coordinates back
    let mut log2_w = log2_trafo_size;
    let mut log2_h = log2_trafo_size;
    if scan_idx == 2 {
        std::mem::swap(&mut log2_w, &mut log2_h); // vertical scan: swap dimensions
    }

    let last_sig_coeff_x_prefix = decode_last_sig_coeff_prefix(
        ctx.cabac,
        crate::hevc::cabac_tables::CTX_LAST_SIG_COEFF_X as i32,
        c_idx,
        log2_w,
    );
    let last_sig_coeff_y_prefix = decode_last_sig_coeff_prefix(
        ctx.cabac,
        crate::hevc::cabac_tables::CTX_LAST_SIG_COEFF_Y as i32,
        c_idx,
        log2_h,
    );

    let mut last_significant_coeff_x = last_sig_coeff_x_prefix;
    let mut last_significant_coeff_y = last_sig_coeff_y_prefix;

    if last_sig_coeff_x_prefix > 3 {
        let suffix = decode_last_sig_coeff_suffix(ctx.cabac, last_sig_coeff_x_prefix);
        let base = (last_sig_coeff_x_prefix >> 1) - 1;
        last_significant_coeff_x =
            (1 << base) * ((last_sig_coeff_x_prefix & 1) + 2) + suffix;
    }
    if last_sig_coeff_y_prefix > 3 {
        let suffix = decode_last_sig_coeff_suffix(ctx.cabac, last_sig_coeff_y_prefix);
        let base = (last_sig_coeff_y_prefix >> 1) - 1;
        last_significant_coeff_y =
            (1 << base) * ((last_sig_coeff_y_prefix & 1) + 2) + suffix;
    }

    if scan_idx == 2 {
        std::mem::swap(&mut last_significant_coeff_x, &mut last_significant_coeff_y);
    }

    if crate::hevc::coding_tree::hevc_trace() {
        eprintln!(
            "RUST residual ({},{}) log2={} cIdx={} scanIdx={} lastSig=({}, {})",
            x0, y0, log2_trafo_size, c_idx, scan_idx, last_significant_coeff_x,
            last_significant_coeff_y
        );
    }

    // Scan tables
    let mut num_sb_per_side = tr_size >> 2;
    if num_sb_per_side < 1 {
        num_sb_per_side = 1;
    }
    let num_sub_blocks = num_sb_per_side * num_sb_per_side;

    let sb_scan = get_sub_block_scan(log2_trafo_size, scan_idx);
    let coeff_scan = get_coeff_scan(scan_idx);

    // Find last sub-block and last scan position
    let mut last_sub_block = num_sub_blocks - 1;
    let mut last_scan_pos = 16i32;

    // Search for last position
    loop {
        if last_scan_pos == 0 {
            last_scan_pos = 16;
            last_sub_block -= 1;
        }
        last_scan_pos -= 1;
        if last_sub_block < 0 {
            // Invalid bitstream (lastSig outside the TU) — C++ would read OOB.
            // Defensive exit: leave coefficients zero.
            return;
        }
        let (x_s, y_s) = sb_scan[last_sub_block as usize];
        let x_s = x_s as i32;
        let y_s = y_s as i32;
        let (x_c, y_c) = coeff_scan[last_scan_pos as usize];
        let x_c = (x_s << 2) + x_c as i32;
        let y_c = (y_s << 2) + y_c as i32;
        if x_c == last_significant_coeff_x && y_c == last_significant_coeff_y {
            break;
        }
    }

    // coded_sub_block_flag array
    let mut coded_sub_block_flag = [0i32; 64];
    // Set the sub-block containing the last sig coeff
    {
        let (x_s, y_s) = sb_scan[last_sub_block as usize];
        let x_s = x_s as i32;
        let y_s = y_s as i32;
        coded_sub_block_flag[(y_s * num_sb_per_side + x_s) as usize] = 1;
    }
    // DC sub-block is always implicitly 1 if there's any coefficient
    coded_sub_block_flag[0] = 1;

    // Cross-sub-block state for ctxSet derivation (§9.3.4.2.6)
    // prev_greater1_ctx tracks greater1Ctx from the last sub-block where
    // coeff_abs_level_greater1_flag was decoded (skipping empty sub-blocks).
    // ctxSet++ when prev_greater1_ctx == 0 (previous sub-block had a coeff > 1).
    let mut prev_greater1_ctx = 0i32; // 0 = no previous sub-block yet
    let mut has_greater1_history = false;

    // Process sub-blocks from last to first
    for i in (0..=last_sub_block).rev() {
        let (x_s, y_s) = sb_scan[i as usize];
        let x_s = x_s as i32;
        let y_s = y_s as i32;
        let sb_idx = (y_s * num_sb_per_side + x_s) as usize;

        let mut infer_sb_dc_sig_coeff_flag = false;

        // Read coded_sub_block_flag for non-first, non-last sub-blocks
        if i < last_sub_block && i > 0 {
            // Context: depends on right and below neighbour csbf
            let (csbf_right, csbf_below) = if x_s + 1 < num_sb_per_side {
                (
                    coded_sub_block_flag[(y_s * num_sb_per_side + (x_s + 1)) as usize],
                    if y_s + 1 < num_sb_per_side {
                        coded_sub_block_flag[((y_s + 1) * num_sb_per_side + x_s) as usize]
                    } else {
                        0
                    },
                )
            } else {
                (
                    0,
                    if y_s + 1 < num_sb_per_side {
                        coded_sub_block_flag[((y_s + 1) * num_sb_per_side + x_s) as usize]
                    } else {
                        0
                    },
                )
            };

            let mut ctx_inc = if c_idx > 0 { 2 } else { 0 };
            ctx_inc += if csbf_right != 0 || csbf_below != 0 { 1 } else { 0 };

            coded_sub_block_flag[sb_idx] = decode_coded_sub_block_flag(ctx.cabac, ctx_inc);
            infer_sb_dc_sig_coeff_flag = true;
        }

        // sig_coeff_flag array for this sub-block (scan order)
        let mut sig_coeff_flag = [0i32; 16];

        let first_n = if i == last_sub_block { last_scan_pos - 1 } else { 15 };

        for n in (0..=first_n).rev() {
            let (cx, cy) = coeff_scan[n as usize];
            let x_c = (x_s << 2) + cx as i32;
            let y_c = (y_s << 2) + cy as i32;

            if coded_sub_block_flag[sb_idx] != 0 && (n > 0 || !infer_sb_dc_sig_coeff_flag) {
                let sig_ctx = derive_sig_coeff_flag_ctx(
                    c_idx,
                    log2_trafo_size,
                    x_c,
                    y_c,
                    scan_idx,
                    &coded_sub_block_flag,
                    num_sb_per_side,
                    false,
                );
                sig_coeff_flag[n as usize] = decode_sig_coeff_flag(ctx.cabac, sig_ctx);
                if sig_coeff_flag[n as usize] != 0 {
                    infer_sb_dc_sig_coeff_flag = false;
                }
            } else if coded_sub_block_flag[sb_idx] != 0 && n == 0 && infer_sb_dc_sig_coeff_flag {
                // Infer DC coefficient as significant
                sig_coeff_flag[0] = 1;
            }
        }

        // For the last sub-block, the last scan position is always significant
        if i == last_sub_block {
            sig_coeff_flag[last_scan_pos as usize] = 1;
        }

        // Decode coefficient levels
        let mut first_sig_scan_pos = 16i32;
        let mut last_sig_scan_pos = -1i32;
        let mut num_greater1_flag = 0i32;
        let mut last_greater1_scan_pos = -1i32;

        let mut coeff_abs_greater1 = [0i32; 16];

        // Context set selection (§9.3.4.2.6)
        let mut ctx_set = if i > 0 && c_idx == 0 { 2 } else { 0 };

        // §9.3.4.2.6: increment ctxSet when the previous sub-block
        // (where greater1 was decoded) had greater1Ctx == 0
        if has_greater1_history && prev_greater1_ctx == 0 {
            ctx_set += 1;
        }

        let mut greater1_ctx = 1i32;

        for n in (0..16).rev() {
            if sig_coeff_flag[n as usize] != 0 {
                if num_greater1_flag < 8 {
                    coeff_abs_greater1[n as usize] = decode_coeff_abs_level_greater1_flag(
                        ctx.cabac,
                        ctx_set,
                        greater1_ctx,
                        c_idx,
                    );
                    num_greater1_flag += 1;

                    if coeff_abs_greater1[n as usize] != 0 {
                        if last_greater1_scan_pos == -1 {
                            last_greater1_scan_pos = n;
                        }
                        greater1_ctx = 0;
                    } else if (1..3).contains(&greater1_ctx) {
                        greater1_ctx += 1;
                    }
                }

                if last_sig_scan_pos == -1 {
                    last_sig_scan_pos = n;
                }
                first_sig_scan_pos = n;
            }
        }

        // Sign hidden condition (§7.3.8.11)
        let cu_bypass = ctx.cu_at(x0, y0).cu_transquant_bypass;
        let sign_hidden = !cu_bypass
            && ctx.pps.sign_data_hiding_enabled_flag
            && (last_sig_scan_pos - first_sig_scan_pos > 3);

        // coeff_abs_greater2_flag
        let mut coeff_abs_greater2_flag = [0i32; 16];
        if last_greater1_scan_pos != -1 {
            coeff_abs_greater2_flag[last_greater1_scan_pos as usize] =
                decode_coeff_abs_level_greater2_flag(ctx.cabac, ctx_set, c_idx);
        }

        // §7.3.8.11 + SPS RExt §7.4.3.2.2: alignment only when
        // cabac_bypass_alignment_enabled_flag is 1 (RExt profiles).
        if ctx.sps.cabac_bypass_alignment_enabled_flag {
            ctx.cabac.align_bypass();
        }

        // Sign flags — read for all sig coeffs (except hidden one)
        let mut coeff_sign_flag = [0i32; 16];
        for n in (0..16).rev() {
            if sig_coeff_flag[n as usize] != 0 && (!sign_hidden || n != first_sig_scan_pos) {
                coeff_sign_flag[n as usize] = decode_coeff_sign_flag(ctx.cabac);
            }
        }

        // coeff_abs_level_remaining + reconstruction (spec §7.3.8.11 final loop)
        let mut num_sig_coeff = 0i32;
        let mut sum_abs_level = 0i32;
        let mut c_rice_param = 0i32;

        for n in (0..16).rev() {
            if sig_coeff_flag[n as usize] != 0 {
                let (cx, cy) = coeff_scan[n as usize];
                let x_c = (x_s << 2) + cx as i32;
                let y_c = (y_s << 2) + cy as i32;

                let base_level = 1 + coeff_abs_greater1[n as usize] + coeff_abs_greater2_flag[n as usize];

                // Spec: read remaining when baseLevel equals max possible
                // max = 3 if numSigCoeff<8 and n==lastGreater1ScanPos
                // max = 2 if numSigCoeff<8 and n!=lastGreater1ScanPos
                // max = 1 if numSigCoeff>=8
                let max_base = if num_sig_coeff < 8 {
                    if n == last_greater1_scan_pos { 3 } else { 2 }
                } else {
                    1
                };

                let mut coeff_remaining = 0i32;
                if base_level == max_base {
                    coeff_remaining = decode_coeff_abs_level_remaining(ctx.cabac, c_rice_param);

                    // Update cRiceParam (§9.3.3.11)
                    let abs_level = base_level + coeff_remaining;
                    if abs_level > 3 * (1 << c_rice_param) && c_rice_param < 4 {
                        c_rice_param += 1;
                    }
                }

                let abs_level = base_level + coeff_remaining;

                // Sign
                let sign = if sign_hidden && n == first_sig_scan_pos {
                    // §7.4.9.11: parity of sum INCLUDING current level
                    if (sum_abs_level + abs_level) & 1 != 0 {
                        -1
                    } else {
                        1
                    }
                } else {
                    if coeff_sign_flag[n as usize] != 0 {
                        -1
                    } else {
                        1
                    }
                };

                coefficients[(y_c * tr_size + x_c) as usize] = (sign * abs_level) as i16;

                sum_abs_level += abs_level;
                num_sig_coeff += 1;
            }
        }

        // Save cross-sub-block state for next iteration (§9.3.4.2.6)
        // Only update when greater1 flags were actually decoded
        if num_greater1_flag > 0 {
            prev_greater1_ctx = greater1_ctx;
            has_greater1_history = true;
        }
    }
}
