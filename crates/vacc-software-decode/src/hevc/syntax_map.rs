//! Mapping from vacc-parser H.265 syntax types to hevc kernel types.
//!
//! The parser (`vacc_parser::h265`) stores raw spec fields; the kernels
//! consume C++-equivalent derived values (`Sps`/`Pps`/`SliceHeader`). These
//! mappers reproduce the C++ `sps.cpp`/`pps.cpp`/`slice_header.cpp`
//! derivation logic exactly so the Rust decode path can be driven from
//! parser output instead of re-parsing the bitstream.

use vacc_core::picture::{H265Pps, H265ScalingLists, H265Sps};
use vacc_parser::h265::SliceHeaderInfo;

use crate::hevc::deblocking::DeblockSliceParams;
use crate::hevc::interpolation::PredWeightTable;
use crate::hevc::transform::ScalingListData;
use crate::hevc::types::{Pps, Sps, SliceHeader, SliceType};

/// C++ `SPS` derivation (sps.cpp) from the parsed fields.
pub fn map_sps(h: &H265Sps) -> Sps {
    let (sub_w, sub_h) = match h.chroma_format_idc {
        0 => (1, 1),
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let w = h.pic_width_in_luma_samples as i32;
    let hgt = h.pic_height_in_luma_samples as i32;
    let bd_y = 8 + h.bit_depth_luma_minus8 as i32;
    let bd_c = 8 + h.bit_depth_chroma_minus8 as i32;

    let min_cb_log2 = 3 + h.log2_min_luma_coding_block_size_minus3 as i32;
    let ctb_log2 = min_cb_log2 + h.log2_diff_max_min_luma_coding_block_size as i32;
    let ctb_size = 1 << ctb_log2;
    let w_ctb = (w + ctb_size - 1) / ctb_size;
    let h_ctb = (hgt + ctb_size - 1) / ctb_size;

    let min_tb_log2 = 2 + h.log2_min_luma_transform_block_size_minus2 as i32;
    let max_tb_log2 = min_tb_log2 + h.log2_diff_max_min_luma_transform_block_size as i32;

    let (log2_min_ipcm, log2_max_ipcm) = if h.pcm_enabled_flag {
        let mn = 3 + h.log2_min_pcm_luma_coding_block_size_minus3 as i32;
        (mn, mn + h.log2_diff_max_min_pcm_luma_coding_block_size as i32)
    } else {
        (0, 0)
    };

    Sps {
        pic_width_in_luma_samples: w,
        pic_height_in_luma_samples: hgt,
        bit_depth_y: bd_y,
        bit_depth_c: bd_c,
        chroma_array_type: h.chroma_format_idc as i32,
        ctb_size_y: ctb_size,
        min_tb_size_y: 1 << min_tb_log2,
        pic_width_in_ctbs_y: w_ctb,
        sub_width_c: sub_w,
        sub_height_c: sub_h,
        intra_smoothing_disabled_flag: h.intra_smoothing_disabled_flag,
        strong_intra_smoothing_enabled_flag: h.strong_intra_smoothing_enabled_flag,
        min_cb_log2_size_y: min_cb_log2,
        ctb_log2_size_y: ctb_log2,
        min_cb_size_y: 1 << min_cb_log2,
        pic_height_in_ctbs_y: h_ctb,
        pic_size_in_ctbs_y: w_ctb * h_ctb,
        min_tb_log2_size_y: min_tb_log2,
        max_tb_log2_size_y: max_tb_log2,
        qp_bd_offset_y: 6 * (bd_y - 8),
        qp_bd_offset_c: 6 * (bd_c - 8),
        amp_enabled_flag: h.amp_enabled_flag,
        pcm_enabled_flag: h.pcm_enabled_flag,
        pcm_sample_bit_depth_luma_minus1: h.pcm_sample_bit_depth_luma_minus1 as i32,
        pcm_sample_bit_depth_chroma_minus1: h.pcm_sample_bit_depth_chroma_minus1 as i32,
        log2_min_ipcm_cb_size_y: log2_min_ipcm,
        log2_max_ipcm_cb_size_y: log2_max_ipcm,
        max_transform_hierarchy_depth_inter: h.max_transform_hierarchy_depth_inter as i32,
        max_transform_hierarchy_depth_intra: h.max_transform_hierarchy_depth_intra as i32,
        cabac_bypass_alignment_enabled_flag: h.cabac_bypass_alignment_enabled_flag,
        sample_adaptive_offset_enabled_flag: h.sample_adaptive_offset_enabled_flag,
        pcm_loop_filter_disabled_flag: h.pcm_loop_filter_disabled_flag,
    }
}

/// C++ `PPS` derivation (pps.cpp) from the parsed fields, including the
/// tile-scan tables.
pub fn map_pps(h: &H265Pps, sps: &Sps) -> Pps {
    let ntc = (h.num_tile_columns_minus1 + 1) as usize;
    let ntr = (h.num_tile_rows_minus1 + 1) as usize;
    let mut p = Pps {
        sign_data_hiding_enabled_flag: h.sign_data_hiding_enabled_flag,
        transform_skip_enabled_flag: h.transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag: h.cu_qp_delta_enabled_flag,
        // §7.4.3.2.1: CtbLog2SizeY - diff_cu_qp_delta_depth.
        log2_min_cu_qp_delta_size: sps.ctb_log2_size_y - h.diff_cu_qp_delta_depth as i32,
        pps_cb_qp_offset: h.pps_cb_qp_offset as i32,
        pps_cr_qp_offset: h.pps_cr_qp_offset as i32,
        weighted_pred_flag: h.weighted_pred_flag,
        weighted_bipred_flag: h.weighted_bipred_flag,
        transquant_bypass_enabled_flag: h.transquant_bypass_enabled_flag,
        tiles_enabled_flag: h.tiles_enabled_flag,
        entropy_coding_sync_enabled_flag: h.entropy_coding_sync_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag: h.pps_loop_filter_across_slices_enabled_flag,
        loop_filter_across_tiles_enabled_flag: h.loop_filter_across_tiles_enabled_flag,
        log2_parallel_merge_level_minus2: h.log2_parallel_merge_level_minus2 as i32,
        num_tile_columns_minus1: h.num_tile_columns_minus1 as i32,
        num_tile_rows_minus1: h.num_tile_rows_minus1 as i32,
        uniform_spacing_flag: h.uniform_spacing_flag,
        column_width_minus1: h.column_width_minus1[..ntc].iter().map(|&v| v as u32).collect(),
        row_height_minus1: h.row_height_minus1[..ntr].iter().map(|&v| v as u32).collect(),
        // Filled by `derive_tile_scan` below.
        ctb_addr_rs_to_ts: Vec::new(),
        ctb_addr_ts_to_rs: Vec::new(),
        tile_id: Vec::new(),
    };
    p.derive_tile_scan(sps);
    p
}

/// Map the parser's derived scaling lists to the kernel layout. Only called
/// when the corresponding `*_scaling_list_data_present_flag` is set; callers
/// pass `ScalingListData::default()` otherwise (flat-16 path in dequant).
pub fn map_scaling_list(h: &H265ScalingLists) -> ScalingListData {
    let mut out = ScalingListData::default();
    for m in 0..6 {
        out.scaling_list[m][..16].copy_from_slice(&h.scaling_list_4x4[m]);
        out.scaling_list[6 + m].copy_from_slice(&h.scaling_list_8x8[m]);
        out.scaling_list[12 + m].copy_from_slice(&h.scaling_list_16x16[m]);
        out.scaling_list_dc[m] = h.scaling_list_dc_coef_16x16[m][0] as u8;
    }
    // 32x32: only matrixId 0 (luma) and 3 (chroma) exist.
    out.scaling_list[18].copy_from_slice(&h.scaling_list_32x32[0]);
    out.scaling_list[21].copy_from_slice(&h.scaling_list_32x32[1]);
    out.scaling_list_dc[6] = h.scaling_list_dc_coef_32x32[0][0] as u8;
    out.scaling_list_dc[9] = h.scaling_list_dc_coef_32x32[1][0] as u8;
    out
}

/// C++ `SliceHeader::parse` field mapping from one parsed segment.
///
/// `pps_init_qp` is the PPS `pps_init_qp_minus26` (C++ derives
/// `SliceQpY = 26 + PpsInitQpMinus26 + SliceQpDelta`).
fn map_sh_fields(h: &SliceHeaderInfo, pps_h: &H265Pps, has_chroma: bool) -> SliceHeader {
    let slice_type = match h.slice_type {
        0 => SliceType::I,
        1 => SliceType::P,
        _ => SliceType::B,
    };
    SliceHeader {
        slice_segment_address: h.slice_segment_address as i32,
        dependent_slice_segment_flag: h.dependent_slice_segment_flag,
        slice_type,
        pic_output_flag: h.pic_output_flag,
        slice_temporal_mvp_enabled_flag: h.slice_temporal_mvp_enabled_flag,
        slice_sao_luma_flag: h.slice_sao_luma_flag,
        slice_sao_chroma_flag: h.slice_sao_chroma_flag,
        num_ref_idx_l0_active_minus1: h.num_ref_idx_l0_active_minus1 as i32,
        num_ref_idx_l1_active_minus1: h.num_ref_idx_l1_active_minus1 as i32,
        mvd_l1_zero_flag: h.mvd_l1_zero_flag,
        cabac_init_flag: h.cabac_init_flag,
        collocated_from_l0_flag: h.collocated_from_l0_flag,
        // C++ default is 0; the field is only meaningful with TMVP on inter.
        collocated_ref_idx: if h.slice_temporal_mvp_enabled_flag && slice_type != SliceType::I {
            h.collocated_ref_idx as i32
        } else {
            0
        },
        five_minus_max_num_merge_cand: h.five_minus_max_num_merge_cand as i32,
        slice_qp_delta: h.slice_qp_delta,
        slice_cb_qp_offset: h.slice_cb_qp_offset,
        slice_cr_qp_offset: h.slice_cr_qp_offset,
        slice_qp_y: 26 + pps_h.pps_init_qp_minus26 + h.slice_qp_delta,
        max_num_merge_cand: 5 - h.five_minus_max_num_merge_cand as i32,
        pred_weight_table: map_weight_table(h, slice_type, pps_h, has_chroma),
        num_entry_point_offsets: h.num_entry_point_offsets as i32,
        entry_point_offset_minus1: h.entry_point_offsets.clone(),
    }
}

/// Map a parsed segment to the kernel `SliceHeader`, applying the C++
/// dependent-slice inheritance: a dependent segment takes every field from
/// the last independent segment (C++ `decoder.cpp` copies the whole header)
/// and keeps only its own `slice_segment_address` + flag.
pub fn map_sh(
    own: &SliceHeaderInfo,
    last_independent: Option<&SliceHeaderInfo>,
    pps_h: &H265Pps,
    has_chroma: bool,
) -> SliceHeader {
    let src = if own.dependent_slice_segment_flag {
        match last_independent {
            Some(li) => li,
            None => own,
        }
    } else {
        own
    };
    let mut sh = map_sh_fields(src, pps_h, has_chroma);
    sh.slice_segment_address = own.slice_segment_address as i32;
    sh.dependent_slice_segment_flag = own.dependent_slice_segment_flag;
    sh
}

/// Effective per-slice deblocking parameters (C++ `slice_header.cpp`): PPS
/// values by default, slice-coded values when the override flag was applied
/// and the filter stays enabled.
pub fn map_deblock_params(h: &SliceHeaderInfo, pps_h: &H265Pps) -> DeblockSliceParams {
    let (beta, tc) = if h.deblocking_filter_override_flag && !h.slice_deblocking_filter_disabled_flag
    {
        (h.slice_beta_offset_div2, h.slice_tc_offset_div2)
    } else {
        (pps_h.pps_beta_offset_div2 as i32, pps_h.pps_tc_offset_div2 as i32)
    };
    DeblockSliceParams {
        deblocking_disabled: h.slice_deblocking_filter_disabled_flag,
        across_slices_enabled: h.slice_loop_filter_across_slices_enabled_flag,
        beta_offset_div2: beta,
        tc_offset_div2: tc,
    }
}

/// C++ `parse_pred_weight_table` materialization (§7.3.6.3 / §7.4.7.3).
///
/// The parser stores the raw coded values (zeroed for unflagged references),
/// so the derivation applies uniformly to all entries: unflagged entries
/// (delta=0, offset=0) yield exactly the C++ defaults
/// (`weight = 1 << log2_weight_denom`, `offset = 0`).
fn map_weight_table(
    h: &SliceHeaderInfo,
    slice_type: SliceType,
    pps_h: &H265Pps,
    has_chroma: bool,
) -> PredWeightTable {
    let has_table = (pps_h.weighted_pred_flag && slice_type == SliceType::P)
        || (pps_h.weighted_bipred_flag && slice_type == SliceType::B);
    if !has_table {
        return PredWeightTable::default();
    }

    let luma_log2_weight_denom = h.luma_log2_weight_denom;
    let mut pwt = PredWeightTable {
        luma_log2_weight_denom: luma_log2_weight_denom as u32,
        delta_chroma_log2_weight_denom: h.delta_chroma_log2_weight_denom as i32,
        ..Default::default()
    };

    let luma_base = 1i32 << luma_log2_weight_denom;
    let chroma_log2_weight_denom =
        (luma_log2_weight_denom as i32) + h.delta_chroma_log2_weight_denom as i32;
    let chroma_base = 1i32 << chroma_log2_weight_denom.max(0);

    for i in 0..15 {
        let luma_weight = (luma_base + h.delta_luma_weight_l0[i] as i32) as i16;
        pwt.l0[i].luma_weight = luma_weight;
        pwt.l0[i].luma_offset = h.luma_offset_l0[i];
        if has_chroma {
            for j in 0..2 {
                let w = (chroma_base + h.delta_chroma_weight_l0[i][j] as i32) as i16;
                pwt.l0[i].chroma_weight[j] = w;
                // §7.4.7.3 chroma offset derivation (C++ slice_header.cpp:50).
                let o = h.chroma_offset_l0[i][j] as i32
                    - ((128 * w as i32) >> chroma_log2_weight_denom.max(0))
                    + 128;
                pwt.l0[i].chroma_offset[j] = o.clamp(-128, 127) as i16;
            }
        }
    }

    if slice_type == SliceType::B {
        for i in 0..15 {
            let luma_weight = (luma_base + h.delta_luma_weight_l1[i] as i32) as i16;
            pwt.l1[i].luma_weight = luma_weight;
            pwt.l1[i].luma_offset = h.luma_offset_l1[i];
            if has_chroma {
                for j in 0..2 {
                    let w = (chroma_base + h.delta_chroma_weight_l1[i][j] as i32) as i16;
                    pwt.l1[i].chroma_weight[j] = w;
                    let o = h.chroma_offset_l1[i][j] as i32
                        - ((128 * w as i32) >> chroma_log2_weight_denom.max(0))
                        + 128;
                    pwt.l1[i].chroma_offset[j] = o.clamp(-128, 127) as i16;
                }
            }
        }
    }

    pwt
}
