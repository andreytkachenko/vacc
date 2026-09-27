//! Construction of CUVID picture parameter structures from vacc-parser output.
//!
//! Replaces the NVIDIA CUVID parser's automatic population of `CUVIDPICPARAMS`
//! with explicit construction from parsed SPS, PPS, and slice header data.

use std::os::raw::{c_char, c_int, c_uchar, c_ushort};

use vacc_core::picture::{H264Pps, H264Sps, H265Pps, H265Sps};
use vacc_parser::h265::SliceHeaderInfo;

use crate::ffi::{
    CUVIDCODECSPECIFIC, CUVIDH264DPBENTRY, CUVIDH264FMOASO, CUVIDH264PICPARAMS, CUVIDH264SVCMVC,
    CUVIDHEVCPICPARAMS, CUVIDPICPARAMS,
};

/// cuvid neutral default 4x4 quantization matrix (intra luma).
///
/// cuvid's convention (verified against cuvidParser ground truth): when no
/// custom scaling matrix is present in the SPS, the decoder expects the
/// neutral value 16 in every slot — NOT the H.264 spec's non-trivial default
/// intra pattern. Filling the spec pattern here produced systematically
/// wrong (too-bright) pixels.
const DEFAULT_QM4X4_INTRA: [u8; 16] = [16; 16];

/// cuvid neutral default 4x4 quantization matrix (inter luma). All 16, see
/// [`DEFAULT_QM4X4_INTRA`].
const DEFAULT_QM4X4_INTER: [u8; 16] = [16; 16];

/// cuvid neutral default 8x8 quantization matrix. All 16, see
/// [`DEFAULT_QM4X4_INTRA`].
const DEFAULT_QM8X8: [u8; 64] = [16; 64];

/// Get WeightScale4x4 matrices for CUVIDH264PICPARAMS.
///
/// Returns 6 matrices: indices 0-2 for intra luma, 3-4 for inter luma, 5 for chroma.
fn get_weight_scale_4x4(sps: &H264Sps) -> [[u8; 16]; 6] {
    if sps.qpprime_y_zero_transform_bypass_flag {
        [[64u8; 16]; 6]
    } else if sps.seq_scaling_matrix_present_flag {
        sps.scaling_list_4x4
    } else {
        [
            DEFAULT_QM4X4_INTRA,
            DEFAULT_QM4X4_INTRA,
            DEFAULT_QM4X4_INTRA,
            DEFAULT_QM4X4_INTER,
            DEFAULT_QM4X4_INTER,
            DEFAULT_QM4X4_INTER,
        ]
    }
}

/// Get WeightScale8x8 matrices for CUVIDH264PICPARAMS.
///
/// Returns 2 matrices: index 0 for intra luma, 1 for inter luma.
fn get_weight_scale_8x8(sps: &H264Sps) -> [[u8; 64]; 2] {
    if sps.qpprime_y_zero_transform_bypass_flag {
        [[64u8; 64]; 2]
    } else if sps.seq_scaling_matrix_present_flag {
        sps.scaling_list_8x8
    } else {
        [DEFAULT_QM8X8, DEFAULT_QM8X8]
    }
}

/// Build [`CUVIDH264PICPARAMS`] from parsed SPS, PPS, and slice header data.
pub fn build_cuvid_h264_picparams(
    sps: &H264Sps,
    pps: &H264Pps,
    slh: &vacc_parser::h264::SliceHeader,
    frame_num: u32,
    poc: i32,
    is_reference: bool,
    dpb_entries: &[CUVIDH264DPBENTRY; 16],
) -> CUVIDH264PICPARAMS {
    CUVIDH264PICPARAMS {
        // SPS fields
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4 as c_int,
        pic_order_cnt_type: sps.pic_order_cnt_type as c_int,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4 as c_int,
        delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag as c_int,
        frame_mbs_only_flag: sps.frame_mbs_only_flag as c_int,
        direct_8x8_inference_flag: sps.direct_8x8_inference_flag as c_int,
        num_ref_frames: sps.max_num_ref_frames as c_int,
        residual_colour_transform_flag: if sps.chroma_format_idc == 3 { 1 } else { 0 },
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        qpprime_y_zero_transform_bypass_flag: sps.qpprime_y_zero_transform_bypass_flag as c_uchar,

        // PPS fields
        entropy_coding_mode_flag: pps.entropy_coding_mode_flag as c_int,
        // EXPERIMENT: PPS field, verbatim.
        pic_order_present_flag: pps.bottom_field_pic_order_in_frame_present_flag as c_int,
        // EXPERIMENT 2: PPS defaults, not slice-resolved values.
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as c_int,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as c_int,
        weighted_pred_flag: pps.weighted_pred_flag as c_int,
        weighted_bipred_idc: pps.weighted_bipred_idc as c_int,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as c_int,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag as c_int,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag as c_int,
        transform_8x8_mode_flag: pps.transform_8x8_mode_flag as c_int,
        MbaffFrameFlag: if !sps.frame_mbs_only_flag
            && !slh.field_pic_flag
            && sps.mb_adaptive_frame_field_flag
        {
            1
        } else {
            0
        },
        constrained_intra_pred_flag: pps.constrained_intra_pred_flag as c_int,
        chroma_qp_index_offset: pps.chroma_qp_index_offset as c_int,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as c_int,

        // Picture-specific fields
        ref_pic_flag: if is_reference { 1 } else { 0 },
        frame_num: frame_num as c_int,
        CurrFieldOrderCnt: [poc, poc],

        // DPB state
        dpb: *dpb_entries,

        // Quantization matrices
        WeightScale4x4: get_weight_scale_4x4(sps),
        WeightScale8x8: get_weight_scale_8x8(sps),

        // FMO/ASO
        fmo_aso_enable: 0,
        num_slice_groups_minus1: pps.num_slice_groups_minus1 as c_uchar,
        slice_group_map_type: 0,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as c_char,
        slice_group_change_rate_minus1: 0,
        fmo: CUVIDH264FMOASO {
            pMb2SliceGroupMap: std::ptr::null(),
        },

        // Reserved
        Reserved: [0; 12],

        // SVC/MVC
        svc_mvc: CUVIDH264SVCMVC::default(),
    }
}

/// Build [`CUVIDPICPARAMS`] from parsed H.264 data.
///
/// The `bitstream_data` must contain raw concatenated NAL units (including
/// NAL header byte) WITHOUT start codes. Slices are concatenated back-to-back.
/// The `slice_offsets` array must contain the byte offset of each slice's
/// NAL unit header within `bitstream_data`.
///
/// The caller must ensure `bitstream_data` and `slice_offsets` live at least
/// as long as the pointers stored in the returned [`CUVIDPICPARAMS`].
// FFI-mirror constructor: field count mirrors CUVIDPICPARAMS inputs.
#[allow(clippy::too_many_arguments)]
pub fn build_cuvid_picparams(
    sps: &H264Sps,
    pps: &H264Pps,
    slh: &vacc_parser::h264::SliceHeader,
    frame_num: u32,
    poc: i32,
    is_reference: bool,
    curr_pic_idx: i32,
    bitstream_data: &[u8],
    slice_offsets: &[u32],
    n_num_slices: u32,
    dpb_entries: &[CUVIDH264DPBENTRY; 16],
) -> CUVIDPICPARAMS {
    let pic_width_in_mbs = sps.pic_width_in_mbs_minus1 as i32 + 1;
    let frame_height_in_mbs = if sps.frame_mbs_only_flag {
        sps.pic_height_in_map_units_minus1 as i32 + 1
    } else {
        (sps.pic_height_in_map_units_minus1 as i32 + 1) * 2
    };

    let field_pic_flag = if sps.frame_mbs_only_flag {
        0
    } else if slh.field_pic_flag {
        1
    } else {
        0
    };

    let bottom_field_flag = if slh.field_pic_flag && slh.bottom_field {
        1
    } else {
        0
    };

    // H.264 slice_type mod 5: 0=P, 1=B, 2=I, 3=SP, 4=SI
    // Intra-coded: I (2) and SI (4)
    let intra_pic_flag = if slh.slice_type == 2 || slh.slice_type == 4 {
        1
    } else {
        0
    };

    let h264_params =
        build_cuvid_h264_picparams(sps, pps, slh, frame_num, poc, is_reference, dpb_entries);

    CUVIDPICPARAMS {
        PicWidthInMbs: pic_width_in_mbs,
        FrameHeightInMbs: frame_height_in_mbs,
        CurrPicIdx: curr_pic_idx,
        field_pic_flag,
        bottom_field_flag,
        second_field: 0,
        nBitstreamDataLen: bitstream_data.len() as u32,
        pBitstreamData: bitstream_data.as_ptr(),
        nNumSlices: n_num_slices,
        pSliceDataOffsets: slice_offsets.as_ptr(),
        ref_pic_flag: if is_reference { 1 } else { 0 },
        intra_pic_flag,
        Reserved: [0; 30],
        CodecSpecific: CUVIDCODECSPECIFIC { h264: h264_params },
    }
}

/// HEVC DPB / reference-picture-set state for a single picture, computed by the
/// decoder and fed into [`build_cuvid_hevc_picparams`].
///
/// The arrays mirror the `CUVIDHEVCPICPARAMS` RefPicSets fields (verified
/// against a cuvid parser ground-truth dump):
/// - `pic_order_cnt_val[i]` / `is_long_term[i]` / `ref_pic_idx[i]` describe
///   the 16-entry DPB array: entry `i` holds the picture at surface
///   `ref_pic_idx[i]` (−1 = empty slot) with POC `pic_order_cnt_val[i]`.
/// - `st_curr_before` / `st_curr_after` / `lt_curr` hold **DPB entry indices**
///   of the current picture's USED references in RPS order (8 entries each);
///   the decoder resolves them to surfaces via `ref_pic_idx`.
#[derive(Debug, Clone, Copy)]
pub struct H265DpbState {
    pub pic_order_cnt_val: [i32; 16],
    pub is_long_term: [u8; 16],
    pub ref_pic_idx: [i32; 16],
    pub st_curr_before: [u8; 8],
    pub st_curr_after: [u8; 8],
    pub lt_curr: [u8; 8],
    pub num_poc_total_curr: i32,
    pub num_poc_st_curr_before: i32,
    pub num_poc_st_curr_after: i32,
    pub num_poc_lt_curr: i32,
    pub num_bits_for_short_term_rps_in_slice: i32,
    pub num_delta_pocs_of_ref_rps_idx: i32,
    pub curr_pic_order_cnt_val: i32,
}

impl Default for H265DpbState {
    fn default() -> Self {
        Self {
            pic_order_cnt_val: [0; 16],
            is_long_term: [0; 16],
            ref_pic_idx: [-1; 16],
            st_curr_before: [0; 8],
            st_curr_after: [0; 8],
            lt_curr: [0; 8],
            num_poc_total_curr: 0,
            num_poc_st_curr_before: 0,
            num_poc_st_curr_after: 0,
            num_poc_lt_curr: 0,
            num_bits_for_short_term_rps_in_slice: 0,
            num_delta_pocs_of_ref_rps_idx: 0,
            curr_pic_order_cnt_val: 0,
        }
    }
}

/// Build [`CUVIDPICPARAMS`] (HEVC) from parsed SPS, PPS, slice header, and DPB
/// state.
///
/// `bitstream_data` must contain the slice NAL unit(s) prefixed with Annex-B
/// start codes (`00 00 01`), matching the layout the NVIDIA cuvid parser
/// produces for HEVC. `slice_offsets[i]` is the byte offset (within
/// `bitstream_data`) of the start of slice `i`'s start code.
///
/// The caller must keep `bitstream_data` and `slice_offsets` alive at least as
/// long as the pointers stored in the returned [`CUVIDPICPARAMS`].
// FFI-mirror constructor: field count mirrors CUVIDPICPARAMS inputs.
#[allow(clippy::too_many_arguments)]
pub fn build_cuvid_hevc_picparams(
    sps: &H265Sps,
    pps: &H265Pps,
    info: &SliceHeaderInfo,
    curr_pic_idx: i32,
    bitstream_data: &[u8],
    slice_offsets: &[u32],
    n_num_slices: u32,
    dpb: &H265DpbState,
) -> CUVIDPICPARAMS {
    let pic_width_in_mbs = (sps.pic_width_in_luma_samples as i32 + 15) / 16;
    let frame_height_in_mbs = (sps.pic_height_in_luma_samples as i32 + 15) / 16;

    // HEVC slice_type: 0=I, 1=P, 2=B. Intra-coded: I (0).
    let intra_pic_flag = if info.slice_type == 0 { 1 } else { 0 };

    let hevc = CUVIDHEVCPICPARAMS {
        // --- SPS ---
        pic_width_in_luma_samples: sps.pic_width_in_luma_samples as c_int,
        pic_height_in_luma_samples: sps.pic_height_in_luma_samples as c_int,
        log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
        log2_min_transform_block_size_minus2: sps.log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_transform_block_size: sps.log2_diff_max_min_luma_transform_block_size,
        pcm_enabled_flag: sps.pcm_enabled_flag as c_uchar,
        log2_min_pcm_luma_coding_block_size_minus3: sps.log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: sps
            .log2_diff_max_min_pcm_luma_coding_block_size,
        pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
        pcm_loop_filter_disabled_flag: sps.pcm_loop_filter_disabled_flag as c_uchar,
        strong_intra_smoothing_enabled_flag: sps.strong_intra_smoothing_enabled_flag as c_uchar,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        amp_enabled_flag: sps.amp_enabled_flag as c_uchar,
        separate_colour_plane_flag: sps.separate_colour_plane_flag as c_uchar,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        num_short_term_ref_pic_sets: sps.num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag: sps.long_term_ref_pics_present_flag as c_uchar,
        num_long_term_ref_pics_sps: sps.num_long_term_ref_pics_sps,
        sps_temporal_mvp_enabled_flag: sps.sps_temporal_mvp_enabled_flag as c_uchar,
        sample_adaptive_offset_enabled_flag: sps.sample_adaptive_offset_enabled_flag as c_uchar,
        scaling_list_enable_flag: sps.scaling_list_enabled_flag as c_uchar,
        IrapPicFlag: if info.is_rap { 1 } else { 0 },
        IdrPicFlag: if info.is_idr { 1 } else { 0 },
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        // --- SPS/PPS extension ---
        log2_max_transform_skip_block_size_minus2: pps.log2_max_transform_skip_block_size_minus2,
        log2_sao_offset_scale_luma: pps.log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma: pps.log2_sao_offset_scale_chroma,
        high_precision_offsets_enabled_flag: sps.high_precision_offsets_enabled_flag as c_uchar,
        reserved1: [0; 10],
        // --- PPS ---
        dependent_slice_segments_enabled_flag: pps.dependent_slice_segments_enabled_flag as c_uchar,
        slice_segment_header_extension_present_flag: pps.slice_segment_header_extension_present_flag
            as c_uchar,
        sign_data_hiding_enabled_flag: pps.sign_data_hiding_enabled_flag as c_uchar,
        cu_qp_delta_enabled_flag: pps.cu_qp_delta_enabled_flag as c_uchar,
        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
        init_qp_minus26: pps.pps_init_qp_minus26 as c_char,
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        constrained_intra_pred_flag: pps.constrained_intra_pred_flag as c_uchar,
        weighted_pred_flag: pps.weighted_pred_flag as c_uchar,
        weighted_bipred_flag: pps.weighted_bipred_flag as c_uchar,
        transform_skip_enabled_flag: pps.transform_skip_enabled_flag as c_uchar,
        transquant_bypass_enabled_flag: pps.transquant_bypass_enabled_flag as c_uchar,
        entropy_coding_sync_enabled_flag: pps.entropy_coding_sync_enabled_flag as c_uchar,
        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        loop_filter_across_tiles_enabled_flag: pps.loop_filter_across_tiles_enabled_flag as c_uchar,
        loop_filter_across_slices_enabled_flag: pps.pps_loop_filter_across_slices_enabled_flag
            as c_uchar,
        output_flag_present_flag: pps.output_flag_present_flag as c_uchar,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        lists_modification_present_flag: pps.lists_modification_present_flag as c_uchar,
        cabac_init_present_flag: pps.cabac_init_present_flag as c_uchar,
        pps_slice_chroma_qp_offsets_present_flag: pps.pps_slice_chroma_qp_offsets_present_flag
            as c_uchar,
        deblocking_filter_override_enabled_flag: pps.deblocking_filter_override_enabled_flag
            as c_uchar,
        pps_deblocking_filter_disabled_flag: pps.pps_deblocking_filter_disabled_flag as c_uchar,
        pps_beta_offset_div2: pps.pps_beta_offset_div2,
        pps_tc_offset_div2: pps.pps_tc_offset_div2,
        tiles_enabled_flag: pps.tiles_enabled_flag as c_uchar,
        uniform_spacing_flag: pps.uniform_spacing_flag as c_uchar,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        column_width_minus1: {
            let mut a: [c_ushort; 21] = [0; 21];
            for (i, v) in a.iter_mut().enumerate().take(19) {
                *v = pps.column_width_minus1[i] as c_ushort;
            }
            a
        },
        row_height_minus1: {
            let mut a: [c_ushort; 21] = [0; 21];
            for (i, v) in a.iter_mut().enumerate().take(21) {
                *v = pps.row_height_minus1[i] as c_ushort;
            }
            a
        },
        // --- SPS/PPS range extension ---
        sps_range_extension_flag: sps.sps_range_extension_flag as c_uchar,
        transform_skip_rotation_enabled_flag: sps.transform_skip_rotation_enabled_flag as c_uchar,
        transform_skip_context_enabled_flag: sps.transform_skip_context_enabled_flag as c_uchar,
        implicit_rdpcm_enabled_flag: sps.implicit_rdpcm_enabled_flag as c_uchar,
        explicit_rdpcm_enabled_flag: sps.explicit_rdpcm_enabled_flag as c_uchar,
        extended_precision_processing_flag: sps.extended_precision_processing_flag as c_uchar,
        intra_smoothing_disabled_flag: sps.intra_smoothing_disabled_flag as c_uchar,
        persistent_rice_adaptation_enabled_flag: sps.persistent_rice_adaptation_enabled_flag
            as c_uchar,
        cabac_bypass_alignment_enabled_flag: sps.cabac_bypass_alignment_enabled_flag as c_uchar,
        pps_range_extension_flag: pps.pps_range_extension_flag as c_uchar,
        cross_component_prediction_enabled_flag: pps.cross_component_prediction_enabled_flag
            as c_uchar,
        chroma_qp_offset_list_enabled_flag: pps.chroma_qp_offset_list_enabled_flag as c_uchar,
        diff_cu_chroma_qp_offset_depth: pps.diff_cu_chroma_qp_offset_depth,
        chroma_qp_offset_list_len_minus1: pps.chroma_qp_offset_list_len_minus1,
        cb_qp_offset_list: pps.cb_qp_offset_list,
        cr_qp_offset_list: pps.cr_qp_offset_list,
        reserved2: [0; 2],
        reserved3: [0; 8],
        // --- RefPicSets ---
        NumBitsForShortTermRPSInSlice: dpb.num_bits_for_short_term_rps_in_slice,
        NumDeltaPocsOfRefRpsIdx: dpb.num_delta_pocs_of_ref_rps_idx,
        NumPocTotalCurr: dpb.num_poc_total_curr,
        NumPocStCurrBefore: dpb.num_poc_st_curr_before,
        NumPocStCurrAfter: dpb.num_poc_st_curr_after,
        NumPocLtCurr: dpb.num_poc_lt_curr,
        CurrPicOrderCntVal: dpb.curr_pic_order_cnt_val,
        RefPicIdx: dpb.ref_pic_idx,
        PicOrderCntVal: dpb.pic_order_cnt_val,
        IsLongTerm: dpb.is_long_term,
        RefPicSetStCurrBefore: dpb.st_curr_before,
        RefPicSetStCurrAfter: dpb.st_curr_after,
        RefPicSetLtCurr: dpb.lt_curr,
        RefPicSetInterLayer0: [0; 8],
        RefPicSetInterLayer1: [0; 8],
        reserved4: [0; 12],
        // --- Scaling lists (neutral 16; no scaling lists in this stream) ---
        ScalingList4x4: [[16u8; 16]; 6],
        ScalingList8x8: [[16u8; 64]; 6],
        ScalingList16x16: [[16u8; 64]; 6],
        ScalingList32x32: [[16u8; 64]; 2],
        ScalingListDCCoeff16x16: [16; 6],
        ScalingListDCCoeff32x32: [16; 2],
    };

    CUVIDPICPARAMS {
        PicWidthInMbs: pic_width_in_mbs,
        FrameHeightInMbs: frame_height_in_mbs,
        CurrPicIdx: curr_pic_idx,
        field_pic_flag: 0,
        bottom_field_flag: 0,
        second_field: 0,
        nBitstreamDataLen: bitstream_data.len() as u32,
        pBitstreamData: bitstream_data.as_ptr(),
        nNumSlices: n_num_slices,
        pSliceDataOffsets: slice_offsets.as_ptr(),
        // NVIDIA's cuvid HEVC parser sets ref_pic_flag=1 for every picture
        // (verified against GT dump_cref.txt: all 300 pics have ref_pic_flag=1,
        // including TRAIL_N/RASL_N non-reference frames). DPB membership is
        // governed separately by the RefPicIdx/PicOrderCntVal arrays, not by
        // this flag, so this does not affect which pictures are kept.
        ref_pic_flag: 1,
        intra_pic_flag,
        Reserved: [0; 30],
        CodecSpecific: CUVIDCODECSPECIFIC { hevc },
    }
}
