//! Construction of CUVID picture parameter structures from vk-video-parser output.
//!
//! Replaces the NVIDIA CUVID parser's automatic population of `CUVIDPICPARAMS`
//! with explicit construction from parsed SPS, PPS, and slice header data.

use std::os::raw::{c_char, c_int, c_uchar};

use vk_video_core::picture::{H264Pps, H264Sps};

use crate::ffi::{
    CUVIDH264DPBENTRY, CUVIDH264FMOASO, CUVIDH264PICPARAMS, CUVIDH264SVCMVC, CUVIDPICPARAMS,
    CUVIDCODECSPECIFIC,
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
    slh: &vk_video_parser::h264::SliceHeader,
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
        pic_order_present_flag: 0, // EXPERIMENT: match GT (was: poc_type != 2 ? 1 : 0)
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as c_int,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as c_int,
        weighted_pred_flag: pps.weighted_pred_flag as c_int,
        weighted_bipred_idc: pps.weighted_bipred_idc as c_int,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as c_int,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag as c_int,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag as c_int,
        transform_8x8_mode_flag: pps.transform_8x8_mode_flag as c_int,
        MbaffFrameFlag: if sps.frame_mbs_only_flag { 0 }
                        else if slh.field_pic_flag { 0 }
                        else if sps.mb_adaptive_frame_field_flag { 1 }
                        else { 0 },
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
pub fn build_cuvid_picparams(
    sps: &H264Sps,
    pps: &H264Pps,
    slh: &vk_video_parser::h264::SliceHeader,
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
    let intra_pic_flag = if slh.slice_type == 2 || slh.slice_type == 4 { 1 } else { 0 };

    let h264_params = build_cuvid_h264_picparams(sps, pps, slh, frame_num, poc, is_reference, dpb_entries);

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
