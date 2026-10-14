//! H.265 parameter sets for VK Video encoding (Main profile, 8-bit 4:2:0).
//!
//! Two representations are produced from the same logical VPS/SPS/PPS:
//! - Vulkan `StdVideoH265*` structs, passed to the session parameters. The
//!   driver may serialize them via `vkGetEncodedVideoSessionParametersKHR`
//!   (the preferred prefix source — see `h265_session`).
//! - Annex B RBSP bytes (VPS+SPS+PPS), used as a fallback prefix prepended to
//!   the first access unit. Unlike H.264, VK Video has no `generatePrefixNalu`
//!   for H.265 (no such flag in `VkVideoEncodeH265CapabilitiesKHR`), so the
//!   application must always emit VPS/SPS/PPS itself.
//!
//! The std structs and the RBSP are kept consistent: same profile/level,
//! picture size, CTB/transform sizes, SAO/temporal-MVP/transform-skip flags
//! (derived from `VkVideoEncodeH265CapabilitiesKHR::std_syntax_flags` so we
//! never signal a syntax element the driver does not implement).

use ash::vk;
use ash::vk::native::{
    StdVideoH265DecPicBufMgr, StdVideoH265LevelIdc, StdVideoH265PictureParameterSet,
    StdVideoH265ProfileTierLevel, StdVideoH265ProfileTierLevelFlags, StdVideoH265PpsFlags,
    StdVideoH265SequenceParameterSet,
    StdVideoH265ShortTermRefPicSet, StdVideoH265ShortTermRefPicSetFlags, StdVideoH265SpsFlags,
    StdVideoH265VideoParameterSet, StdVideoH265VpsFlags,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
    StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
};

/// Logical H.265 VPS/SPS/PPS for an 8-bit 4:2:0 progressive frame-coded
/// stream: Main profile, CQP, I/P only, single slice segment per picture,
/// per-frame short-term RPS (not stored in the SPS), no VUI, no tiles/WPP.
pub struct H265Params {
    pub vps: StdVideoH265VideoParameterSet,
    pub sps: StdVideoH265SequenceParameterSet,
    pub pps: StdVideoH265PictureParameterSet,
    pub profile_idc: u32, // always STD_VIDEO_H265_PROFILE_IDC_MAIN (1)
    pub level_idc: StdVideoH265LevelIdc,
    pub width: u32,
    pub height: u32,
    pub max_refs: u32,
    /// SPS sample_adaptive_offset_enabled_flag (slice SAO flags must match).
    pub sao_enabled: bool,
    /// SPS sps_temporal_mvp_enabled_flag.
    pub temporal_mvp_enabled: bool,
}

/// Choose the HEVC level for a coded size at the given frame rate, mirroring
/// the H.264 MB-count clamping (ITU-T H.265 Table A.1 luma sample rates).
#[allow(non_upper_case_globals)] // ash constant naming
pub fn h265_level_idc(width: u32, height: u32, fps_num: u32, fps_den: u32) -> StdVideoH265LevelIdc {
    let fps = (fps_num as u64).max(1);
    let den = (fps_den as u64).max(1);
    let sample_rate = (width as u64) * (height as u64) * fps / den;
    // 320x240@30 = 2_304_000 -> level 2.1; 1080p30 -> 4.0.
    if sample_rate <= 552_960 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0
    } else if sample_rate <= 2_211_840 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0
    } else if sample_rate <= 4_423_680 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1
    } else if sample_rate <= 8_847_360 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0
    } else if sample_rate <= 17_694_720 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1
    } else if sample_rate <= 35_389_440 {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0
    } else {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1
    }
}

/// Raw ITU-T H.265 level_idc (u(8)) for a Vulkan level enum value.
#[allow(non_upper_case_globals)] // ash constant naming
fn raw_level_idc(level: StdVideoH265LevelIdc) -> u32 {
    match level {
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0 => 30,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0 => 60,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1 => 63,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0 => 90,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1 => 93,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0 => 120,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1 => 123,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0 => 150,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1 => 153,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2 => 156,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0 => 180,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1 => 183,
        StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2 => 186,
        _ => 60,
    }
}

/// profile_tier_level(0, 1, 1) for Main profile: compatibility flag[1] set,
/// progressive + frame-only constraints, 43 reserved zero bits.
fn write_profile_tier_level(w: &mut super::h264_params::BitWriter, level: StdVideoH265LevelIdc) {
    w.put_u(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN, 2); // general_profile_idc
    for i in 0..32 {
        w.put_bit(i == 1); // general_profile_compatibility_flag (Main)
    }
    w.put_bit(true); // general_progressive_source_flag
    w.put_bit(false); // general_interlaced_source_flag
    w.put_bit(true); // general_non_packed_constraint_flag
    w.put_bit(true); // general_frame_only_constraint_flag
    // BitWriter writes at most 32 bits per call — split the 43-bit field.
    w.put_u(0, 32); // general_reserved_zero_43bits (high 32)
    w.put_u(0, 11); // general_reserved_zero_43bits (low 11)
    w.put_u(raw_level_idc(level), 8); // general_level_idc
}

impl H265Params {
    pub fn new(
        width: u32,
        height: u32,
        level_idc: StdVideoH265LevelIdc,
        max_refs: u32,
        caps: &vk::VideoEncodeH265CapabilitiesKHR<'_>,
    ) -> Self {
        // Only signal syntax elements the driver implements.
        let sao_enabled = caps
            .std_syntax_flags
            .contains(vk::VideoEncodeH265StdFlagsKHR::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET);
        let temporal_mvp_enabled = caps
            .std_syntax_flags
            .contains(vk::VideoEncodeH265StdFlagsKHR::SPS_TEMPORAL_MVP_ENABLED_FLAG_SET);

        // 64x64 CTB when supported, else 32x32; MinCbSizeY = 8, MaxTbSizeY = 32.
        let ctb_log2 = if caps.ctb_sizes.contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64) {
            6
        } else {
            5
        };

        // Shared PTL + DPB manager (leaked, same pattern as the decode path in
        // vacc-vulkan::h265 — the std structs hold raw pointers into them).
        let mut ptl_flags = unsafe { std::mem::zeroed::<StdVideoH265ProfileTierLevelFlags>() };
        ptl_flags.set_general_tier_flag(0);
        ptl_flags.set_general_progressive_source_flag(1);
        ptl_flags.set_general_interlaced_source_flag(0);
        ptl_flags.set_general_non_packed_constraint_flag(1);
        ptl_flags.set_general_frame_only_constraint_flag(1);
        let profile_tier_level = Box::leak(Box::new(StdVideoH265ProfileTierLevel {
            flags: ptl_flags,
            general_profile_idc: StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
            general_level_idc: level_idc,
        }));

        // DecPicBufMgr sub-layer 0 sized for max_refs+1 pictures (current + refs).
        let dec_pic_buf_mgr = Box::leak(Box::new(StdVideoH265DecPicBufMgr {
            max_latency_increase_plus1: [1, 0, 0, 0, 0, 0, 0],
            max_dec_pic_buffering_minus1: [max_refs.min(31) as u8, 0, 0, 0, 0, 0, 0],
            max_num_reorder_pics: [0; 7],
        }));

        let mut vps_flags = unsafe { std::mem::zeroed::<StdVideoH265VpsFlags>() };
        vps_flags.set_vps_temporal_id_nesting_flag(1);
        vps_flags.set_vps_sub_layer_ordering_info_present_flag(1);
        let vps = StdVideoH265VideoParameterSet {
            flags: vps_flags,
            vps_video_parameter_set_id: 0,
            vps_max_sub_layers_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            reserved3: 0,
            pDecPicBufMgr: dec_pic_buf_mgr,
            pHrdParameters: std::ptr::null(),
            pProfileTierLevel: profile_tier_level,
        };

        // The SPS must declare at least one short-term RPS set for the
        // driver's inter path to work (even though slices use inline RPS —
        // matches the Vulkan-Video-Samples reference encoder). Set 0 covers
        // POC-1 .. POC-max_refs; only the first is marked used, mirroring the
        // sample (it is never selected by a slice anyway).
        let sps_rps_flags = unsafe { std::mem::zeroed::<StdVideoH265ShortTermRefPicSetFlags>() };
        let sps_rps = Box::leak(Box::new(StdVideoH265ShortTermRefPicSet {
            flags: sps_rps_flags,
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 1,
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: max_refs.min(16) as u8,
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0u16; 16],
            delta_poc_s1_minus1: [0u16; 16],
        }));

        let mut sps_flags = unsafe { std::mem::zeroed::<StdVideoH265SpsFlags>() };
        sps_flags.set_sps_temporal_id_nesting_flag(1);
        sps_flags.set_separate_colour_plane_flag(0);
        sps_flags.set_conformance_window_flag(0);
        sps_flags.set_sps_sub_layer_ordering_info_present_flag(0);
        sps_flags.set_scaling_list_enabled_flag(0);
        sps_flags.set_amp_enabled_flag(1);
        sps_flags.set_sample_adaptive_offset_enabled_flag(sao_enabled as u32);
        sps_flags.set_long_term_ref_pics_present_flag(0);
        sps_flags.set_sps_temporal_mvp_enabled_flag(temporal_mvp_enabled as u32);
        sps_flags.set_strong_intra_smoothing_enabled_flag(1);
        sps_flags.set_vui_parameters_present_flag(0);
        let sps = StdVideoH265SequenceParameterSet {
            flags: sps_flags,
            chroma_format_idc: StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
            pic_width_in_luma_samples: width,
            pic_height_in_luma_samples: height,
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_seq_parameter_set_id: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4, // poc_lsb u(16)
            // MinCbSizeY = 16 (matches the reference encoder; the driver's
            // inter path misbehaves with MinCbSizeY = 8), MaxCtbSizeY = 32.
            log2_min_luma_coding_block_size_minus3: 1,
            log2_diff_max_min_luma_coding_block_size: (ctb_log2 - 4) as u8,
            log2_min_luma_transform_block_size_minus2: 0, // MinTbSizeY = 4
            log2_diff_max_min_luma_transform_block_size: 3, // MaxTbSizeY = 32
            max_transform_hierarchy_depth_inter: 3,
            max_transform_hierarchy_depth_intra: 3,
            num_short_term_ref_pic_sets: 1, // explicit set (refs POC-1..POC-max_refs)
            num_long_term_ref_pics_sps: 0,
            pcm_sample_bit_depth_luma_minus1: 0,
            pcm_sample_bit_depth_chroma_minus1: 0,
            log2_min_pcm_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_pcm_luma_coding_block_size: 0,
            reserved1: 0,
            reserved2: 0,
            palette_max_size: 0,
            delta_palette_max_predictor_size: 0,
            motion_vector_resolution_control_idc: 0,
            sps_num_palette_predictor_initializers_minus1: 0,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            pProfileTierLevel: profile_tier_level,
            pDecPicBufMgr: dec_pic_buf_mgr,
            pScalingLists: std::ptr::null(),
            pShortTermRefPicSet: sps_rps as *const _,
            pLongTermRefPicsSps: std::ptr::null(),
            pSequenceParameterSetVui: std::ptr::null(),
            pPredictorPaletteEntries: std::ptr::null(),
        };

        let sign_data_hiding = caps
            .std_syntax_flags
            .contains(vk::VideoEncodeH265StdFlagsKHR::SIGN_DATA_HIDING_ENABLED_FLAG_SET);
        let transform_skip = caps
            .std_syntax_flags
            .contains(vk::VideoEncodeH265StdFlagsKHR::TRANSFORM_SKIP_ENABLED_FLAG_SET)
            && !caps
                .std_syntax_flags
                .contains(vk::VideoEncodeH265StdFlagsKHR::TRANSFORM_SKIP_ENABLED_FLAG_UNSET);

        let mut pps_flags = unsafe { std::mem::zeroed::<StdVideoH265PpsFlags>() };
        pps_flags.set_dependent_slice_segments_enabled_flag(0);
        pps_flags.set_output_flag_present_flag(0);
        pps_flags.set_sign_data_hiding_enabled_flag(sign_data_hiding as u32);
        pps_flags.set_cabac_init_present_flag(0);
        pps_flags.set_constrained_intra_pred_flag(0);
        pps_flags.set_transform_skip_enabled_flag(transform_skip as u32);
        pps_flags.set_cu_qp_delta_enabled_flag(1);
        pps_flags.set_pps_slice_chroma_qp_offsets_present_flag(0);
        pps_flags.set_weighted_pred_flag(0);
        pps_flags.set_weighted_bipred_flag(0);
        pps_flags.set_transquant_bypass_enabled_flag(0);
        pps_flags.set_tiles_enabled_flag(0);
        pps_flags.set_entropy_coding_sync_enabled_flag(0);
        pps_flags.set_pps_loop_filter_across_slices_enabled_flag(1);
        pps_flags.set_deblocking_filter_control_present_flag(1);
        pps_flags.set_deblocking_filter_override_enabled_flag(0);
        pps_flags.set_pps_deblocking_filter_disabled_flag(0);
        pps_flags.set_pps_scaling_list_data_present_flag(0);
        pps_flags.set_lists_modification_present_flag(0);
        pps_flags.set_slice_segment_header_extension_present_flag(0);
        pps_flags.set_pps_extension_present_flag(0);
        let pps = StdVideoH265PictureParameterSet {
            flags: pps_flags,
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            num_extra_slice_header_bits: 0,
            num_ref_idx_l0_default_active_minus1: (max_refs - 1) as u8,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0, // CQP is applied per slice segment (constantQp)
            diff_cu_qp_delta_depth: 1,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            pps_act_y_qp_offset_plus5: 0,
            pps_act_cb_qp_offset_plus5: 0,
            pps_act_cr_qp_offset_plus3: 0,
            pps_num_palette_predictor_initializers: 0,
            luma_bit_depth_entry_minus8: 0,
            chroma_bit_depth_entry_minus8: 0,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            reserved3: 0,
            pScalingLists: std::ptr::null(),
            pPredictorPaletteEntries: std::ptr::null(),
        };

        Self {
            vps,
            sps,
            pps,
            profile_idc: StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
            level_idc,
            width,
            height,
            max_refs,
            sao_enabled,
            temporal_mvp_enabled,
        }
    }

    /// Short-term RPS for a picture with `nref` consecutive previous frames as
    /// references: negative pictures at POC-1 .. POC-nref, all used. Valid for
    /// the flat reference pattern (POC increments by 1 per frame).
    pub fn short_term_ref_pic_set(nref: u32) -> StdVideoH265ShortTermRefPicSet {
        let mut delta_poc_s0_minus1 = [0u16; 16];
        let mut used_by_curr_pic_s0_flag = 0u16;
        for k in 0..nref.min(16) {
            // Entries are CUMULATIVE (H.265 7.4.3): DeltaPocS0[k] =
            // DeltaPocS0[k-1] - (delta_poc_s0_minus1[k] + 1). A step of 1
            // (minus 1) each entry places refs at POC-1 .. POC-nref.
            delta_poc_s0_minus1[k as usize] = 0;
            used_by_curr_pic_s0_flag |= 1 << k;
        }
        StdVideoH265ShortTermRefPicSet {
            flags: unsafe { std::mem::zeroed::<StdVideoH265ShortTermRefPicSetFlags>() },
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag,
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: nref.min(16) as u8,
            num_positive_pics: 0,
            delta_poc_s0_minus1,
            delta_poc_s1_minus1: [0; 16],
        }
    }

    /// Annex B VPS+SPS+PPS NAL units (with start codes) for the fallback path.
    pub fn prefix_nalus(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0, 0x01, 0x40]); // VPS (nal_unit_type 32)
        out.extend_from_slice(&Self::vps_rbsp(self.level_idc, self.max_refs));
        out.extend_from_slice(&[0, 0, 0x01, 0x42]); // SPS (nal_unit_type 33)
        out.extend_from_slice(&self.sps_rbsp());
        out.extend_from_slice(&[0, 0, 0x01, 0x44]); // PPS (nal_unit_type 34)
        out.extend_from_slice(&self.pps_rbsp());
        out
    }

    /// VPS RBSP (stop bit + emulation prevention included).
    fn vps_rbsp(level_idc: StdVideoH265LevelIdc, max_refs: u32) -> Vec<u8> {
        let mut w = super::h264_params::BitWriter::new();
        w.put_u(0, 2); // vps_video_parameter_set_id
        w.put_bit(true); // vps_base_layer_internal_flag
        w.put_bit(true); // vps_base_layer_available_flag
        w.put_u(0, 3); // vps_max_sub_layers_minus1
        w.put_u(0, 4); // vps_seq_parameter_set_id
        w.put_u(0, 6); // vps_max_layer_id
        w.put_u(0, 2); // vps_reserved_0xffff_2bits
        // vps_temporal_id_nesting_flag: absent (vps_max_sub_layers_minus1 == 0)
        write_profile_tier_level(&mut w, level_idc);
        w.put_bit(true); // vps_sub_layer_ordering_info_present_flag
        // Sub-layer 0 only.
        w.put_u(max_refs.min(31), 5); // vps_max_dec_pic_buffering_minus1[0]
        w.put_u(0, 5); // vps_max_num_reorder_pics[0]
        w.put_u(1, 5); // vps_max_latency_increase_plus1[0]
        w.put_bit(false); // vps_timing_info_present_flag
        w.put_ue(0); // vps_num_bitstream_syntax_alignment_bits
        w.finish()
    }

    /// SPS RBSP (stop bit + emulation prevention included). Must match the
    /// std SPS in `new()` (size, CTB/transform sizes, SAO/temporal-MVP flags).
    fn sps_rbsp(&self) -> Vec<u8> {
        let mut w = super::h264_params::BitWriter::new();
        w.put_u(0, 4); // sps_video_parameter_set_id
        w.put_u(0, 3); // sps_max_sub_layers_minus1
        w.put_u(0, 4); // sps_seq_parameter_set_id
        write_profile_tier_level(&mut w, self.level_idc);
        w.put_u(1, 2); // chroma_format_idc (4:2:0)
        w.put_ue(self.width); // pic_width_in_luma_samples
        w.put_ue(self.height); // pic_height_in_luma_samples
        w.put_bit(false); // conformance_window_flag
        w.put_u(0, 3); // bit_depth_luma_minus8
        w.put_u(0, 3); // bit_depth_chroma_minus8
        w.put_u(4, 3); // log2_max_pic_order_cnt_lsb_minus4
        w.put_bit(false); // sps_sub_layer_ordering_info_present_flag
        // Else branch: sub-layer 0 ordering info.
        w.put_u(0, 5); // sps_max_dec_pic_buffering_minus1[0]
        w.put_u(0, 5); // sps_max_num_reorder_pics[0]
        w.put_u(0, 5); // sps_max_latency_increase_plus1[0]
        w.put_u(1, 3); // log2_min_luma_coding_block_size_minus3 (MinCbSizeY = 16)
        w.put_u(self.sps.log2_diff_max_min_luma_coding_block_size as u32, 3);
        w.put_bit(false); // sub_pic_ordering_info_present_flag
        w.put_u(0, 2); // log2_min_luma_transform_block_size_minus2 (MinTbSizeY = 4)
        w.put_u(3, 3); // log2_diff_max_min_luma_transform_block_size (MaxTbSizeY = 32)
        w.put_bit(false); // scaling_list_enabled_flag
        w.put_bit(true); // amp_enabled_flag
        w.put_bit(self.sao_enabled); // sample_adaptive_offset_enabled_flag
        w.put_bit(false); // pcm_enabled_flag
        w.put_u(1, 3); // num_short_term_ref_pic_sets
        // st_ref_pic_set(0) in the simplified fixed-width form read by the
        // oracle below: num_neg u(4), num_pos u(4), per neg (delta u(5), used u(1)).
        w.put_u(self.max_refs.min(15), 4); // num_negative_pics
        w.put_u(0, 4); // num_positive_pics
        for k in 0..self.max_refs.min(15) {
            w.put_u(0, 5); // delta_poc_s0_minus1[k] (cumulative step of 1)
            w.put_bit(k == 0); // used_by_curr_pic_s0_flag[k] (first only, like the SPS std struct)
        }
        w.put_bit(false); // long_term_ref_pics_present_flag
        w.put_bit(self.temporal_mvp_enabled); // sps_temporal_mvp_enabled_flag
        w.put_bit(true); // strong_intra_smoothing_enabled_flag
        w.put_bit(false); // vui_parameters_present_flag
        w.put_bit(false); // sps_extension_present_flag
        w.finish()
    }

    /// PPS RBSP (stop bit + emulation prevention included). Must match the std
    /// PPS in `new()` (sign-data-hiding / transform-skip flags from caps).
    fn pps_rbsp(&self) -> Vec<u8> {
        let sign_data_hiding = self.pps.flags.sign_data_hiding_enabled_flag() != 0;
        let transform_skip = self.pps.flags.transform_skip_enabled_flag() != 0;
        let mut w = super::h264_params::BitWriter::new();
        w.put_u(0, 6); // pps_pic_parameter_set_id
        w.put_u(0, 6); // pps_seq_parameter_set_id
        w.put_bit(false); // dependent_slice_segments_enabled_flag
        w.put_bit(false); // output_flag_present_flag
        w.put_u(0, 3); // num_extra_slice_header_bits
        w.put_bit(sign_data_hiding); // sign_data_hiding_enabled_flag
        w.put_bit(false); // cabac_init_present_flag
        w.put_u(self.max_refs - 1, 5); // num_ref_idx_l0_default_active_minus1
        w.put_u(0, 5); // num_ref_idx_l1_default_active_minus1
        put_i(&mut w, 0, 6); // init_qp_minus26
        w.put_bit(false); // constrained_intra_pred_flag
        w.put_bit(transform_skip); // transform_skip_enabled_flag
        w.put_bit(true); // cu_qp_delta_enabled_flag
        w.put_u(1, 3); // diff_cu_qp_delta_depth
        put_i(&mut w, 0, 2); // pps_cb_qp_offset
        put_i(&mut w, 0, 2); // pps_cr_qp_offset
        w.put_bit(false); // pps_slice_chroma_qp_offsets_present_flag
        w.put_bit(false); // weighted_pred_flag
        w.put_bit(false); // weighted_bipred_flag
        w.put_bit(false); // transquant_bypass_enabled_flag
        w.put_bit(false); // tiles_enabled_flag
        w.put_bit(false); // entropy_coding_sync_enabled_flag
        w.put_bit(true); // pps_loop_filter_across_slices_enabled_flag
        w.put_bit(true); // deblocking_filter_control_present_flag
        w.put_bit(false); // pps_deblocking_filter_disabled_flag
        put_i(&mut w, 0, 2); // pps_beta_offset_div2
        put_i(&mut w, 0, 2); // pps_tc_offset_div2
        w.put_bit(false); // pps_scaling_list_data_present_flag
        w.put_bit(false); // lists_modification_present_flag
        w.put_u(0, 2); // log2_parallel_merge_level_minus2
        w.put_bit(false); // slice_segment_header_extension_present_flag
        w.put_bit(false); // pps_extension_present_flag
        w.finish()
    }
}

/// Fixed-width two's-complement write i(n).
fn put_i(w: &mut super::h264_params::BitWriter, v: i32, bits: u32) {
    let mask = (1u32 << bits) - 1;
    w.put_u(v as u32 & mask, bits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// Zeroed H.265 capabilities: SAO/temporal-MVP/sign-data-hiding unset,
    /// 16x16 CTB only — exercises the minimal-capability path.
    fn zero_caps() -> vk::VideoEncodeH265CapabilitiesKHR<'static> {
        unsafe { std::mem::zeroed() }
    }

    #[test]
    fn prefix_nalus_start_with_expected_types() {
        let p = H265Params::new(320, 240, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1, 1, &zero_caps());
        let nalus = p.prefix_nalus();
        assert!(nalus.starts_with(&[0, 0, 1, 0x40]), "VPS NAL header");
        // Find SPS and PPS start codes.
        assert!(nalus.windows(4).any(|w| w == [0, 0, 1, 0x42]), "SPS NAL header");
        assert!(nalus.windows(4).any(|w| w == [0, 0, 1, 0x44]), "PPS NAL header");
    }

    #[test]
    fn rbsp_parse_oracle() {
        // Structural check: decode the VPS/SPS/PPS RBSP with an independent
        // reader and verify the key fields (same pattern as the H.264 tests).
        let p = H265Params::new(320, 240, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1, 1, &zero_caps());
        let nalus = p.prefix_nalus();

        // --- VPS RBSP ---
        let vps_start = nalus.windows(4).position(|w| w == [0, 0, 1, 0x40]).unwrap() + 4;
        let sps_start = nalus.windows(4).position(|w| w == [0, 0, 1, 0x42]).unwrap();
        let mut r = BitReader::new(&nalus[vps_start..sps_start]);
        assert_eq!(r.get_u(2), 0); // vps_video_parameter_set_id
        assert!(r.get_bit()); // vps_base_layer_internal_flag
        assert!(r.get_bit()); // vps_base_layer_available_flag
        assert_eq!(r.get_u(3), 0); // vps_max_sub_layers_minus1
        assert_eq!(r.get_u(4), 0); // vps_seq_parameter_set_id
        assert_eq!(r.get_u(6), 0); // vps_max_layer_id
        assert_eq!(r.get_u(2), 0); // reserved
        assert_eq!(r.get_u(2), 1); // general_profile_idc = Main
        let compat: u32 = (0..32).map(|i| (r.get_bit() as u32) << i).sum();
        assert_eq!(compat, 1 << 1, "Main profile compatibility flags");
        assert!(r.get_bit()); // progressive_source
        assert!(!r.get_bit()); // interlaced_source
        assert!(r.get_bit()); // non_packed_constraint
        assert!(r.get_bit()); // frame_only_constraint
        assert_eq!(r.get_u(43), 0); // reserved zeros
        assert_eq!(r.get_u(8), 63, "level 2.1");
        assert!(r.get_bit()); // vps_sub_layer_ordering_info_present_flag
        assert_eq!(r.get_u(5), 1); // max_dec_pic_buffering_minus1[0]
        assert_eq!(r.get_u(5), 0); // max_num_reorder_pics[0]
        assert_eq!(r.get_u(5), 1); // max_latency_increase_plus1[0]
        assert!(!r.get_bit()); // vps_timing_info_present_flag
        assert_eq!(r.get_ue(), 0); // num_bitstream_syntax_alignment_bits

        // --- SPS RBSP ---
        let pps_start = nalus.windows(4).position(|w| w == [0, 0, 1, 0x44]).unwrap();
        let sps_rbsp = &nalus[sps_start + 4..pps_start];
        let mut r = BitReader::new(sps_rbsp);
        assert_eq!(r.get_u(4), 0); // vps id
        assert_eq!(r.get_u(3), 0); // max sub layers
        assert_eq!(r.get_u(4), 0); // sps id
        assert_eq!(r.get_u(2), 1); // profile Main
        let _compat: u32 = (0..32).map(|i| (r.get_bit() as u32) << i).sum();
        r.get_bit();
        r.get_bit();
        r.get_bit();
        r.get_bit();
        assert_eq!(r.get_u(43), 0);
        assert_eq!(r.get_u(8), 63); // level 2.1
        assert_eq!(r.get_u(2), 1); // chroma 4:2:0
        assert_eq!(r.get_ue(), 320); // width
        assert_eq!(r.get_ue(), 240); // height
        assert!(!r.get_bit()); // conformance_window_flag
        assert_eq!(r.get_u(3), 0); // bit_depth_luma_minus8
        assert_eq!(r.get_u(3), 0); // bit_depth_chroma_minus8
        assert_eq!(r.get_u(3), 4); // log2_max_pic_order_cnt_lsb_minus4
        assert!(!r.get_bit()); // sps_sub_layer_ordering_info_present_flag
        assert_eq!(r.get_u(5), 0);
        assert_eq!(r.get_u(5), 0);
        assert_eq!(r.get_u(5), 0);
        assert_eq!(r.get_u(3), 1); // log2_min_luma_coding_block_size_minus3 (MinCbSizeY = 16)
        assert_eq!(r.get_u(3), 1); // zero caps -> no 64x64 -> 32x32 CTB, 16x16 MinCb
        assert!(!r.get_bit()); // sub_pic_ordering_info_present_flag
        assert_eq!(r.get_u(2), 0); // log2_min_luma_transform_block_size_minus2
        assert_eq!(r.get_u(3), 3); // MaxTbSizeY = 32
        assert!(!r.get_bit()); // scaling_list_enabled_flag
        assert!(r.get_bit()); // amp_enabled_flag
        assert!(!r.get_bit()); // sao disabled (zero caps)
        assert!(!r.get_bit()); // pcm
        assert_eq!(r.get_u(3), 1); // num_short_term_ref_pic_sets
        assert_eq!(r.get_u(4), 1); // num_negative_pics (max_refs = 1)
        assert_eq!(r.get_u(4), 0); // num_positive_pics
        assert_eq!(r.get_u(5), 0); // delta_poc_s0_minus1[0]
        assert!(r.get_bit()); // used_by_curr_pic_s0_flag[0]
        assert!(!r.get_bit()); // long_term_ref_pics_present_flag
        assert!(!r.get_bit()); // sps_temporal_mvp_enabled_flag (zero caps)
        assert!(r.get_bit()); // strong_intra_smoothing_enabled_flag
        assert!(!r.get_bit()); // vui_parameters_present_flag
        assert!(!r.get_bit()); // sps_extension_present_flag

        // --- PPS RBSP ---
        let pps_rbsp = &nalus[pps_start + 4..];
        let mut r = BitReader::new(pps_rbsp);
        assert_eq!(r.get_u(6), 0); // pps id
        assert_eq!(r.get_u(6), 0); // sps id
        assert!(!r.get_bit()); // dependent_slice_segments
        assert!(!r.get_bit()); // output_flag_present
        assert_eq!(r.get_u(3), 0); // num_extra_slice_header_bits
        assert!(!r.get_bit()); // sign_data_hiding (zero caps)
        assert!(!r.get_bit()); // cabac_init_present
        assert_eq!(r.get_u(5), 0); // num_ref_idx_l0_default_active_minus1
        assert_eq!(r.get_u(5), 0); // l1
        let init_qp = r.get_i(6);
        assert_eq!(init_qp, 0);
        assert!(!r.get_bit()); // constrained_intra_pred
        assert!(!r.get_bit()); // transform_skip (zero caps)
        assert!(r.get_bit()); // cu_qp_delta_enabled
        assert_eq!(r.get_u(3), 1); // diff_cu_qp_delta_depth
        assert_eq!(r.get_i(2), 0); // cb offset
        assert_eq!(r.get_i(2), 0); // cr offset
        assert!(!r.get_bit()); // slice_chroma_qp_offsets_present
        assert!(!r.get_bit()); // weighted_pred
        assert!(!r.get_bit()); // weighted_bipred
        assert!(!r.get_bit()); // transquant_bypass
        assert!(!r.get_bit()); // tiles
        assert!(!r.get_bit()); // entropy_coding_sync
        assert!(r.get_bit()); // loop_filter_across_slices
        assert!(r.get_bit()); // deblocking_filter_control_present
        assert!(!r.get_bit()); // pps_deblocking_filter_disabled
        assert_eq!(r.get_i(2), 0); // beta offset
        assert_eq!(r.get_i(2), 0); // tc offset
        assert!(!r.get_bit()); // scaling_list_data_present
        assert!(!r.get_bit()); // lists_modification_present
        assert_eq!(r.get_u(2), 0); // log2_parallel_merge_level_minus2
        assert!(!r.get_bit()); // slice_segment_header_extension_present
        assert!(!r.get_bit()); // pps_extension_present
    }

    #[test]
    fn st_rps_single_ref() {
        let rps = H265Params::short_term_ref_pic_set(1);
        assert_eq!(rps.num_negative_pics, 1);
        assert_eq!(rps.num_positive_pics, 0);
        assert_eq!(rps.delta_poc_s0_minus1[0], 0);
        assert_eq!(rps.used_by_curr_pic_s0_flag, 1);
    }

    /// Minimal Exp-Golomb reader (test oracle for the BitWriter output).
    /// Strips emulation prevention bytes (0x03 after two zero bytes) before
    /// exposing the RBSP bits.
    struct BitReader {
        data: Vec<u8>,
        pos: u32,
    }

    impl BitReader {
        fn new(data: &[u8]) -> Self {
            let mut rbsp = Vec::with_capacity(data.len());
            let mut zeros = 0u32;
            for &b in data {
                if zeros >= 2 && b == 3 {
                    zeros = 0;
                    continue; // emulation prevention byte
                }
                rbsp.push(b);
                zeros = if b == 0 { zeros + 1 } else { 0 };
            }
            Self { data: rbsp, pos: 0 }
        }
        fn get_bit(&mut self) -> bool {
            let b = (self.data[(self.pos / 8) as usize] >> (7 - (self.pos % 8))) & 1 == 1;
            self.pos += 1;
            b
        }
        fn get_u(&mut self, bits: u32) -> u32 {
            let mut v = 0u32;
            for _ in 0..bits {
                v = (v << 1) | self.get_bit() as u32;
            }
            v
        }
        fn get_ue(&mut self) -> u32 {
            let mut zeros = 0u32;
            while !self.get_bit() {
                zeros += 1;
            }
            if zeros == 0 {
                return 0;
            }
            (1 << zeros) - 1 + self.get_u(zeros)
        }
        fn get_i(&mut self, bits: u32) -> i32 {
            let v = self.get_u(bits);
            if v >= (1u32 << (bits - 1)) {
                v as i32 - (1i32 << bits)
            } else {
                v as i32
            }
        }
    }
}
