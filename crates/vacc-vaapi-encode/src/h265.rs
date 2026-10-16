//! H.265/HEVC bitstream generation: VPS, SPS, PPS and slice NAL units.
//!
//! Every byte layout here is verified byte-for-byte against FFmpeg 8.0's
//! `hevc_vaapi` output on iHD (ground truth `/tmp/ff10.hevc`, decoded at
//! ~47 dB on the gradient test pattern). iHD has a fixed internal codec/CABAC
//! configuration but emits user-supplied packed NAL units verbatim, so our
//! param sets and slice headers must match that known-good reference exactly
//! — mismatched values corrupt the stream.
//!
//! Configuration (matching the ground truth):
//! - Main profile (compat `0x60000000`), level 6.0, 4:2:0, 8-bit, progressive
//! - DPB: max_dec_pic_buffering=3, max_num_reorder_pics=2, latency+1=1
//! - SPS: log2_max_poc_lsb_minus4=8 (u(12) poc), CTU 64 / TU 32, no SPS
//!   short-term RPS sets, SAO + AMP enabled, temporal MVP on, VUI with
//!   vui_video_signal_type_present and bitstream restriction
//! - PPS: transform_skip_enabled=1, entropy_coding_sync=0,
//!   loop_filter_across_slices=1, init_qp_minus26=0
//!
//! GOP structure (FF's hevc_vaapi "P emitted as B" pattern): IDR followed by
//! mini-GOPs of [anchor + 2×B] where the anchor is a *B-slice* with a single
//! L0 reference (the previous anchor/IDR); each B references the previous
//! anchor (L0) and the current anchor (L1). Slice QP: IDR/anchor = qp,
//! B = round(1.2·qp + 0.5), both expressed as `slice_qp_delta` against
//! pic_init_qp = 26.

use crate::h264::BitWriter;

/// Main profile compatibility flags (bits 30 and 31 set).
const MAIN_COMPAT: u32 = 0x60_00_00_00;

/// `profile_tier_level` as used by VPS/SPS: Main (idc 1), level 6.0,
/// progressive / non-packed / frame-only, no constraint sets, inbld off.
fn write_ptl(w: &mut BitWriter) {
    // general_profile_space(2)=0 | general_tier_flag(1)=0 | general_profile_idc(5)=1
    w.put_u(1, 8);
    w.put_u(MAIN_COMPAT, 32); // general_profile_compatibility_flag[0]
    w.put_bit(true); // progressive_source_flag
    w.put_bit(false); // interlaced_source_flag
    w.put_bit(true); // non_packed_constraint_flag
    w.put_bit(true); // frame_only_constraint_flag
    // constraint_set0_present(pc(2)) == 0:
    w.put_u(0, 7); // general_constraint_flag[6:0]
    w.put_bit(false); // general_constraint_flag[7] (one_reactive_sync_pic)
    w.put_u(0, 24); // general_constraint_flag[31:8]
    w.put_u(0, 11); // general_reserved_zero_bit[10:0]
    w.put_bit(false); // general_inbld_flag
    w.put_u(60, 8); // general_level_idc (level 6.0)
}

/// Sub-layer DPB ordering with sl_ordering_present=0 (single set):
/// max_dec_pic_buffering_minus1=2, max_num_reorder_pics=1,
/// max_latency_increase_plus1=0.
fn write_dpb(w: &mut BitWriter) {
    w.put_ue(2);
    w.put_ue(1);
    w.put_ue(0);
}

/// H.265 VPS NAL unit (type 32). Returns the NAL without start code.
pub fn make_vps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_u(0, 4); // vps_video_parameter_set_id
    w.put_bit(true); // vps_base_layer_internal_flag
    w.put_bit(true); // vps_base_layer_available_flag
    w.put_u(0, 6); // vps_max_layers_minus1
    w.put_u(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(true); // vps_temporal_id_nesting_flag
    w.put_u(0xFFFF, 16); // vps_reserved_0xffff_23bits
    write_ptl(&mut w);
    w.put_bit(false); // vps_sub_layer_ordering_info_present_flag
    write_dpb(&mut w);
    w.put_u(0, 6); // vps_max_layer_id
    w.put_ue(0); // vps_num_layer_sets_minus1
    w.put_bit(true); // vps_timing_info_present_flag
    w.put_u(1, 32); // vps_num_units_in_tick
    w.put_u(30, 32); // vps_time_scale
    w.put_bit(true); // vps_poc_proportional_to_timing_flag
    w.put_ue(0); // vps_ticks_poc_diff_one_minus1
    w.put_ue(0); // vps_num_hrd_parameters
    w.put_bit(false); // vps_extension_flag
    let rbsp = w.finish();
    let mut nal = Vec::with_capacity(2 + rbsp.len());
    nal.extend_from_slice(&[0x40, 0x01]); // NAL header: type 32 (VPS_NUT)
    nal.extend_from_slice(&rbsp);
    nal
}

/// H.265 SPS NAL unit (type 33) for `width`×`height`, 4:2:0, 8-bit.
pub fn make_sps(width: u32, height: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_u(0, 4); // sps_video_parameter_set_id
    w.put_u(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(true); // sps_temporal_id_nesting_flag
    write_ptl(&mut w);
    w.put_ue(0); // sps_seq_parameter_set_id
    w.put_ue(1); // chroma_format_idc (4:2:0)
    w.put_ue(width); // pic_width_in_luma_samples
    w.put_ue(height); // pic_height_in_luma_samples
    w.put_bit(false); // conformance_window_flag
    w.put_ue(0); // bit_depth_luma_minus8
    w.put_ue(0); // bit_depth_chroma_minus8
    w.put_ue(8); // log2_max_pic_order_cnt_lsb_minus4 (u(12) poc_lsb)
    w.put_bit(false); // sps_sub_layer_ordering_info_present_flag
    write_dpb(&mut w);
    w.put_ue(0); // log2_min_luma_coding_block_size_minus3 (8)
    w.put_ue(3); // log2_diff_max_min_luma_coding_block_size (CTU 64)
    w.put_ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.put_ue(3); // log2_diff_max_min_luma_transform_block_size (TU 32)
    w.put_ue(2); // max_transform_hierarchy_depth_inter
    w.put_ue(2); // max_transform_hierarchy_depth_intra
    w.put_bit(false); // scaling_list_enabled_flag
    w.put_bit(true); // amp_enabled_flag
    w.put_bit(true); // sample_adaptive_offset_enabled_flag
    w.put_bit(false); // pcm_enabled_flag
    w.put_ue(0); // num_short_term_ref_pic_sets (RPS is inline in slice headers)
    w.put_bit(false); // long_term_ref_pics_present_flag
    w.put_bit(true); // sps_temporal_mvp_enabled_flag
    w.put_bit(false); // strong_intra_smoothing_enabled_flag
    w.put_bit(true); // vui_parameters_present_flag
    // VUI
    w.put_bit(false); // aspect_ratio_info_present_flag
    w.put_bit(false); // overscan_info_present_flag
    w.put_bit(true); // video_signal_type_present_flag
    w.put_u(5, 3); // video_format (unspecified)
    w.put_bit(false); // video_full_range_flag
    w.put_bit(false); // colour_description_present_flag
    w.put_bit(true); // chroma_loc_info_present_flag
    w.put_ue(0); // chroma_sample_loc_type_top_field
    w.put_ue(0); // chroma_sample_loc_type_bottom_field
    w.put_bit(false); // neutral_chroma_indication_flag
    w.put_bit(false); // field_seq_flag
    w.put_bit(false); // frame_field_flag
    w.put_bit(false); // default_display_window_flag
    w.put_bit(true); // vui_timing_info_present_flag
    w.put_u(1, 32); // vui_num_units_in_tick
    w.put_u(30, 32); // vui_time_scale
    w.put_bit(true); // vui_poc_proportional_to_timing_flag
    w.put_ue(0); // vui_ticks_poc_diff_one_minus1
    w.put_bit(false); // vui_hrd_parameters_present_flag
    w.put_bit(true); // vui_bitstream_restriction_flag
    w.put_bit(false); // tiles_fixed_structure_flag
    w.put_bit(true); // motion_vectors_over_pic_boundaries_flag
    w.put_bit(true); // restricted_ref_pic_lists_flag
    w.put_ue(0); // min_spatial_segmentation_idc_minus1
    w.put_ue(0); // max_bytes_per_pic_denom
    w.put_ue(0); // max_bits_per_min_cu_denom
    w.put_ue(15); // log2_max_mv_length_horizontal
    w.put_ue(15); // log2_max_mv_length_vertical
    w.put_bit(false); // sps_extension_present_flag
    let rbsp = w.finish();
    let mut nal = Vec::with_capacity(2 + rbsp.len());
    nal.extend_from_slice(&[0x42, 0x01]); // NAL header: type 33 (SPS_NUT)
    nal.extend_from_slice(&rbsp);
    nal
}

/// H.265 PPS NAL unit (type 34).
pub fn make_pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_ue(0); // pps_pic_parameter_set_id
    w.put_ue(0); // pps_seq_parameter_set_id
    w.put_bit(false); // dependent_slice_segments_enabled_flag
    w.put_bit(false); // output_flag_present_flag
    w.put_u(0, 3); // num_extra_slice_header_bits
    w.put_bit(false); // sign_data_hiding_enabled_flag
    w.put_bit(false); // cabac_init_present_flag
    w.put_ue(0); // num_ref_idx_l0_default_active_minus1
    w.put_ue(0); // num_ref_idx_l1_default_active_minus1
    w.put_se(0); // pps_init_qp_minus26
    w.put_bit(false); // constrained_intra_pred_flag
    w.put_bit(true); // transform_skip_enabled_flag
    w.put_bit(false); // cu_qp_delta_enabled_flag
    w.put_se(0); // pps_cb_qp_offset
    w.put_se(0); // pps_cr_qp_offset
    w.put_bit(false); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(false); // weighted_pred_flag
    w.put_bit(false); // weighted_bipred_flag
    w.put_bit(false); // transquant_bypass_enabled_flag
    w.put_bit(false); // tiles_enabled_flag
    w.put_bit(false); // entropy_coding_sync_enabled_flag
    w.put_bit(true); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(false); // deblocking_filter_control_present_flag
    w.put_bit(false); // pps_scaling_list_data_present_flag
    w.put_bit(false); // lists_modification_present_flag
    w.put_ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(false); // slice_segment_header_extension_present_flag
    w.put_bit(false); // pps_extension_present_flag
    let rbsp = w.finish();
    let mut nal = Vec::with_capacity(2 + rbsp.len());
    nal.extend_from_slice(&[0x44, 0x01]); // NAL header: type 34 (PPS_NUT)
    nal.extend_from_slice(&rbsp);
    nal
}

/// An inter slice. All inter frames in this configuration are B-slices with
/// at most one negative and one positive reference (both used).
pub struct InterSlice {
    /// NAL unit type: 0 = TRAIL_R (B frame), 1 = TRAIL_N (anchor).
    pub nal_type: u8,
    /// pic_order_cnt_lsb, u(12) (SPS log2_max_poc_lsb_minus4 = 8).
    pub poc_lsb: u32,
    /// delta_idx_minus1 of the single negative reference (always used).
    pub neg_delta_minus1: u32,
    /// delta_idx_minus1 of the positive reference; `None` for anchors
    /// (empty L1).
    pub pos_delta_minus1: Option<u32>,
    /// slice_qp_delta relative to pic_init_qp = 26.
    pub qp_delta: i32,
}

/// A slice NAL unit.
pub enum SliceKind {
    /// IDR_W_RADL (type 19), I-slice.
    Idr,
    Inter(InterSlice),
}

/// Build a slice NAL unit: `[00 00 00 01][NAL header][EPB'd RBSP]`.
///
/// Returns `(nal_bytes, bit_length)` where `bit_length` is the packed data
/// length in bits (for `VAEncPackedHeaderParameterBuffer`).
pub fn make_slice_hdr(kind: &SliceKind) -> (Vec<u8>, u32) {
    let mut w = BitWriter::new();
    match kind {
        SliceKind::Idr => {
            w.put_bit(true); // first_slice_segment_in_pic_flag
            w.put_bit(false); // no_output_of_prior_pics_flag (IRAP)
            w.put_ue(0); // pic_parameter_set_id
            w.put_ue(2); // slice_type (2 = I)
            w.put_bit(true); // slice_sao_luma_flag
            w.put_bit(true); // slice_sao_chroma_flag
            w.put_se(0); // slice_qp_delta
            w.put_bit(false); // slice_loop_filter_across_slices_enabled_flag
        }
        SliceKind::Inter(s) => {
            w.put_bit(true); // first_slice_segment_in_pic_flag
            w.put_ue(0); // pic_parameter_set_id
            w.put_ue(0); // slice_type (0 = B)
            w.put_u(s.poc_lsb & 0xFFF, 12); // pic_order_cnt_lsb u(12)
            w.put_bit(false); // short_term_ref_pic_set_sps_flag (inline RPS)
            let npos = if s.pos_delta_minus1.is_some() { 1 } else { 0 };
            w.put_ue(1); // NumNegPics
            w.put_ue(npos); // NumPosPics
            w.put_ue(s.neg_delta_minus1); // delta_idx_minus1[0] (negative)
            w.put_bit(true); // used_by_curr_pic_s0_flag[0]
            if let Some(pdm) = s.pos_delta_minus1 {
                w.put_ue(pdm); // delta_idx_minus1[0] (positive)
                w.put_bit(true); // used_by_curr_pic_s1_flag[0]
            }
            w.put_bit(true); // slice_temporal_mvp_enabled_flag
            w.put_bit(true); // slice_sao_luma_flag
            w.put_bit(true); // slice_sao_chroma_flag
            w.put_bit(false); // num_ref_idx_active_override_flag
            w.put_bit(false); // mvd_l1_zero_flag
            w.put_bit(true); // collocated_from_l0_flag
            w.put_ue(0); // five_minus_max_num_merge_cand (5 candidates)
            w.put_se(s.qp_delta); // slice_qp_delta
            w.put_bit(false); // slice_loop_filter_across_slices_enabled_flag
        }
    }
    let rbsp = w.finish();
    let nal_type = match kind {
        SliceKind::Idr => 19u8, // IDR_W_RADL
        SliceKind::Inter(s) => s.nal_type,
    };
    let mut out = Vec::with_capacity(4 + 2 + rbsp.len());
    out.extend_from_slice(&[0, 0, 0, 1]); // 4-byte start code
    // NAL header: forbidden(0) | nal_unit_type(6) | layer_id high2(0).
    out.push(nal_type << 1);
    out.push(0x01); // layer_id low4(0) | temporal_id_plus1(1)
    out.extend_from_slice(&rbsp);
    let bits = (out.len() as u32) * 8;
    (out, bits)
}

/// Packed Sequence buffer payload: VPS + SPS + PPS NALs, each prefixed with a
/// 4-byte start code.
pub fn seq_packed_data(width: u32, height: u32) -> Vec<u8> {
    let vps = make_vps();
    let sps = make_sps(width, height);
    let pps = make_pps();
    let mut out = Vec::with_capacity(12 + vps.len() + sps.len() + pps.len());
    for nal in [&vps, &sps, &pps] {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground-truth NAL bytes from FFmpeg 8.0 hevc_vaapi output
    /// (`/tmp/ff10.hevc`, 320x240@30, qp 26).
    const VPS: &[u8] = &[
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x35, 0x03, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00,
        0x03, 0x00, 0x1e, 0xe8,
    ];
    const SPS_320X240: &[u8] = &[
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x3c, 0xa0, 0x0a, 0x08, 0x0f, 0x16, 0x24, 0xd6, 0x49, 0x1b, 0x6a, 0x9a, 0x70,
        0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x0f, 0x6b, 0xe1, 0x00, 0x82,
    ];
    const PPS: &[u8] = &[0x44, 0x01, 0xc0, 0x75, 0x81, 0x12];
    /// IDR slice header (RBSP `af a0`).
    const SLICE_IDR: &[u8] = &[0x26, 0x01, 0xaf, 0xa0];
    /// Anchor at poc 3: B-slice, single L0 ref 3 POCs back, qp_delta 0.
    const SLICE_ANCHOR_3: &[u8] = &[0x02, 0x01, 0xe0, 0x06, 0x57, 0xe7, 0x40];
    /// B at poc 1: L0 = poc 0 (dm 0), L1 = poc 3 (dm 1), qp_delta +5.
    const SLICE_B_1: &[u8] = &[0x00, 0x01, 0xe0, 0x02, 0x4b, 0x5e, 0x62, 0x90];
    /// B at poc 2: L0 = poc 0 (dm 1), L1 = poc 3 (dm 0), qp_delta +5.
    const SLICE_B_2: &[u8] = &[0x00, 0x01, 0xe0, 0x04, 0x49, 0x7e, 0x62, 0x90];

    #[test]
    fn vps_matches_ff_bytes() {
        assert_eq!(make_vps(), VPS);
    }

    #[test]
    fn sps_matches_ff_bytes_320x240() {
        assert_eq!(make_sps(320, 240), SPS_320X240);
    }

    #[test]
    fn pps_matches_ff_bytes() {
        assert_eq!(make_pps(), PPS);
    }

    #[test]
    fn slice_idr_matches_ff_bytes() {
        let (nal, bits) = make_slice_hdr(&SliceKind::Idr);
        assert_eq!(&nal[4..], SLICE_IDR);
        assert_eq!(bits, (nal.len() as u32) * 8);
    }

    #[test]
    fn slice_anchor_matches_ff_bytes() {
        let (nal, _) = make_slice_hdr(&SliceKind::Inter(InterSlice {
            nal_type: 1,
            poc_lsb: 3,
            neg_delta_minus1: 2,
            pos_delta_minus1: None,
            qp_delta: 0,
        }));
        assert_eq!(&nal[4..], SLICE_ANCHOR_3);
    }

    #[test]
    fn slice_b_matches_ff_bytes() {
        let (b1, _) = make_slice_hdr(&SliceKind::Inter(InterSlice {
            nal_type: 0,
            poc_lsb: 1,
            neg_delta_minus1: 0,
            pos_delta_minus1: Some(1),
            qp_delta: 5,
        }));
        assert_eq!(&b1[4..], SLICE_B_1);
        let (b2, _) = make_slice_hdr(&SliceKind::Inter(InterSlice {
            nal_type: 0,
            poc_lsb: 2,
            neg_delta_minus1: 1,
            pos_delta_minus1: Some(0),
            qp_delta: 5,
        }));
        assert_eq!(&b2[4..], SLICE_B_2);
    }

    #[test]
    fn seq_packed_is_sc_prefixed_nals() {
        let data = seq_packed_data(320, 240);
        let mut expected = Vec::new();
        for nal in [VPS, SPS_320X240, PPS] {
            expected.extend_from_slice(&[0, 0, 0, 1]);
            expected.extend_from_slice(nal);
        }
        assert_eq!(data, expected);
    }
}
