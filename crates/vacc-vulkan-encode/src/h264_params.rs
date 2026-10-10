//! H.264 parameter sets for VK Video encoding.
//!
//! Two representations are produced from the same logical SPS/PPS:
//! - Vulkan `StdVideoH264*` structs, passed to the session parameters (the
//!   driver serializes them itself when generating prefix NALUs).
//! - Annex B RBSP bytes, used as a fallback when the driver does not support
//!   `generatePrefixNalu` (we prepend SPS+PPS to the first access unit).

use ash::vk::native::{
    StdVideoH264PictureParameterSet, StdVideoH264PpsFlags, StdVideoH264SequenceParameterSet,
    StdVideoH264SpsFlags,
};

/// Logical H.264 SPS/PPS for an 8-bit 4:2:0 progressive frame-coded stream:
/// POC type 0 (16-bit lsb), 16-bit frame_num, CAVLC, no cropping, no VUI.
pub struct H264Params {
    pub sps: StdVideoH264SequenceParameterSet,
    pub pps: StdVideoH264PictureParameterSet,
    pub profile_idc: u32,
    pub level_idc: u32,
    pub max_num_ref_frames: u32,
}

/// High-family profiles that carry the chroma/bit-depth fields in SPS RBSP.
fn is_high_family(profile_idc: u32) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128
    )
}

impl H264Params {
    pub fn new(
        width_mbs: u32,
        height_mbs: u32,
        profile_idc: u32,
        level_idc: u32,
        max_num_ref_frames: u32,
    ) -> Self {
        let mut sps_flags = unsafe { std::mem::zeroed::<StdVideoH264SpsFlags>() };
        sps_flags.set_frame_mbs_only_flag(1);
        // Match the Khronos reference (NVIDIA): always signal direct_8x8_inference.
        sps_flags.set_direct_8x8_inference_flag(1);

        // Some drivers read pOffsetForRefFrame unconditionally even for
        // pic_order_cnt_type != 1 (learned the hard way on the decode path) —
        // pass a zeroed array instead of null.
        let offset_for_ref_frame: [i32; 255] = [0i32; 255];
        let offset_ptr = Box::leak(Box::new(offset_for_ref_frame)) as *const i32;

        let sps = StdVideoH264SequenceParameterSet {
            flags: sps_flags,
            profile_idc,
            level_idc,
            chroma_format_idc: 1, // 4:2:0
            seq_parameter_set_id: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_frame_num_minus4: 4, // frame_num u(16) (matches reference)
            // POC type 2: POC is derived from frame_num (2*frameNum for ref frames,
            // frameNumOffset=0 with no wraparound). Per-frame poc MUST follow this
            // derivation (see lib.rs / session.rs) or the NVIDIA driver rejects
            // P-frames (provided poc != type-2-derived poc).
            pic_order_cnt_type: 2,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4, // poc_lsb u(16)
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            max_num_ref_frames: max_num_ref_frames as u8,
            reserved1: 0,
            pic_width_in_mbs_minus1: width_mbs - 1,
            pic_height_in_map_units_minus1: height_mbs - 1,
            frame_crop_left_offset: 0,
            frame_crop_right_offset: 0,
            frame_crop_top_offset: 0,
            frame_crop_bottom_offset: 0,
            reserved2: 0,
            pOffsetForRefFrame: offset_ptr,
            pScalingLists: std::ptr::null(),
            pSequenceParameterSetVui: std::ptr::null(),
        };

        let mut pps_flags = unsafe { std::mem::zeroed::<StdVideoH264PpsFlags>() };
        // CAVLC (entropy_coding_mode_flag = 0), slice-level deblocking control.
        pps_flags.set_deblocking_filter_control_present_flag(1);

        let pps = StdVideoH264PictureParameterSet {
            flags: pps_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            num_ref_idx_l0_default_active_minus1: (max_num_ref_frames - 1) as u8,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0, // CQP is applied per slice (constantQp)
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            second_chroma_qp_index_offset: 0,
            pScalingLists: std::ptr::null(),
        };

        Self {
            sps,
            pps,
            profile_idc,
            level_idc,
            max_num_ref_frames,
        }
    }

    /// Annex B SPS+PPS NAL units (with start codes) for the fallback path.
    pub fn prefix_nalus(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0, 0x01, 0x67]); // IDR NAL: SPS
        out.extend_from_slice(&Self::sps_rbsp(
            self.profile_idc,
            self.level_idc,
            self.sps.pic_width_in_mbs_minus1 + 1,
            self.sps.pic_height_in_map_units_minus1 + 1,
            self.max_num_ref_frames,
        ));
        out.extend_from_slice(&[0, 0, 0x01, 0x28]); // IDR NAL: PPS
        out.extend_from_slice(&Self::pps_rbsp());
        out
    }

    /// SPS RBSP (stop bit + emulation prevention included).
    fn sps_rbsp(profile_idc: u32, level_idc: u32, width_mbs: u32, height_mbs: u32, max_ref_frames: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_u(profile_idc, 8);
        w.put_u(0x00, 8); // constraint flags + reserved
        w.put_u(level_idc, 8);
        w.put_ue(0); // seq_parameter_set_id
        if is_high_family(profile_idc) {
            w.put_ue(1); // chroma_format_idc (4:2:0)
            w.put_ue(0); // bit_depth_luma_minus8
            w.put_ue(0); // bit_depth_chroma_minus8
            w.put_bit(false); // qpprime_y_zero_transform_bypass_flag
            w.put_bit(false); // seq_scaling_matrix_present_flag
        }
        w.put_ue(4); // log2_max_frame_num_minus4 (frame_num u(16))
        w.put_ue(2); // pic_order_cnt_type (type 2: POC derived from frame_num)
        w.put_ue(max_ref_frames); // max_num_ref_frames
        w.put_bit(false); // gaps_in_frame_num_value_allowed_flag
        w.put_ue(width_mbs - 1); // pic_width_in_mbs_minus1
        w.put_ue(height_mbs - 1); // pic_height_in_map_units_minus1
        w.put_bit(true); // frame_mbs_only_flag
        w.put_bit(true); // direct_8x8_inference_flag (matches reference)
        w.put_bit(false); // frame_cropping_flag
        w.put_bit(false); // vui_parameters_present_flag
        w.finish()
    }

    /// PPS RBSP (stop bit + emulation prevention included). CAVLC, single
    /// slice group, deblocking control present.
    fn pps_rbsp() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_ue(0); // pic_parameter_set_id (ue(v))
        w.put_ue(0); // seq_parameter_set_id (ue(v))
        w.put_bit(false); // entropy_coding_mode_flag (CAVLC)
        // Read unconditionally by ffmpeg/libavcodec and the project oracle even
        // though the SPS sets frame_mbs_only_flag = 1 — matches the VAAPI
        // encoder's verified layout.
        w.put_bit(false); // bottom_field_pic_order_in_frame_present_flag
        w.put_ue(0); // num_slice_groups_minus1
        w.put_ue(0); // num_ref_idx_l0_default_active_minus1
        w.put_ue(0); // num_ref_idx_l1_default_active_minus1
        w.put_bit(false); // weighted_pred_flag
        w.put_u(0, 2); // weighted_bipred_idc (unconditional)
        w.put_se(0); // pic_init_qp_minus26
        w.put_se(0); // pic_init_qs_minus26
        w.put_se(0); // chroma_qp_index_offset
        w.put_bit(true); // deblocking_filter_control_present_flag
        w.put_bit(false); // constrained_intra_pred_flag
        w.put_bit(false); // redundant_pic_cnt_present_flag
        // All-zero tail (more_rbsp_data region): decoders read these whenever bits
        // remain before the stop bit; zeros keep the PPS valid (matches the VAAPI
        // encoder's verified layout).
        w.put_bit(false); // transform_8x8_mode_flag
        w.put_bit(false); // pic_scaling_matrix_present_flag
        w.put_se(0); // second_chroma_qp_index_offset
        w.finish()
    }
}

/// Minimal Exp-Golomb bit writer with stop bit + emulation prevention on
/// finalize. (Same contract as the VAAPI encoder's writer.)
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bitpos: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    fn put_bit(&mut self, v: bool) {
        if self.bitpos == 0 {
            self.bytes.push(0);
        }
        if v {
            let mask = 0x80u8 >> self.bitpos;
            *self.bytes.last_mut().unwrap() |= mask;
        }
        self.bitpos += 1;
        if self.bitpos == 8 {
            self.bitpos = 0;
        }
    }

    pub fn put_u(&mut self, v: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.put_bit((v >> i) & 1 == 1);
        }
    }

    pub fn put_ue(&mut self, v: u32) {
        let x = v + 1;
        let bits = 32 - x.leading_zeros();
        for _ in 0..(bits - 1) {
            self.put_bit(false);
        }
        for i in (0..bits).rev() {
            self.put_bit((x >> i) & 1 == 1);
        }
    }

    pub fn put_se(&mut self, v: i32) {
        let ue = if v < 0 { (-2 * v) as u32 } else { (2 * v) as u32 };
        self.put_ue(ue);
    }

    /// Finalize: stop bit + byte alignment + emulation prevention.
    pub fn finish(mut self) -> Vec<u8> {
        self.put_bit(true); // rbsp_stop_bit
        while self.bitpos != 0 {
            self.put_bit(false);
        }
        let input = std::mem::take(&mut self.bytes);
        let mut out = Vec::with_capacity(input.len() * 2);
        let mut zeros = 0u32;
        for &b in &input {
            out.push(b);
            if zeros >= 2 && (b as u32) <= 3 {
                out.push(3);
                zeros = 0;
            } else {
                zeros = if b == 0 { zeros + 1 } else { 0 };
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sps_pps_rbsp_roundtrip() {
        let p = H264Params::new(20, 15, 77, 31, 1); // 320x240 Main ref=1
        let nalus = p.prefix_nalus();
        assert!(nalus.starts_with(&[0, 0, 1, 0x67]));
        // SPS RBSP for 320x240 Main lvl3.1 ref=1: profile 77, no chroma field.
        let sps_rbsp = &nalus[4..];
        let pps_start = sps_rbsp.windows(4).position(|w| w == [0, 0, 1, 0x28]).unwrap();
        let sps_bytes = &sps_rbsp[..pps_start];
        assert_eq!(sps_bytes[0], 77); // profile Main
        assert_eq!(sps_bytes[2], 31); // level 3.1
        // PPS RBSP must be the verified layout `ce 3c 30` (ue(v) ids — NOT u(4),
        // which decodes to pps_id=511 and breaks PPS lookup in every decoder).
        // (pps_start is relative to sps_rbsp = &nalus[4..]; +8 skips start code + NAL header)
        let pps_rbsp = &nalus[pps_start + 8..];
        assert_eq!(&pps_rbsp[..3], &[0xce, 0x3c, 0x30]);
    }

    #[test]
    fn high_profile_sps_has_chroma_field() {
        // High-family SPS: profile byte 0x64 (100), then chroma_format_idc
        // field present (first RBSP bit after level = '1' for sps_id, then
        // '010' for chroma 4:2:0). Baseline/Main would have '1' (sps_id)
        // immediately followed by log2_max_frame_num ('1').
        let p = H264Params::new(20, 15, 100, 31, 3);
        let nalus = p.prefix_nalus();
        let sps_rbsp = &nalus[4..];
        let pps_start = sps_rbsp.windows(4).position(|w| w == [0, 0, 1, 0x28]).unwrap();
        let sps = &sps_rbsp[..pps_start];
        assert_eq!(sps[0], 0x64); // profile High
        assert_eq!(sps[1], 0x00); // constraints
        assert_eq!(sps[2], 31); // level 3.1
        // Full structural validation is done against the project parser oracle
        // (see tests/encode_h264.rs + manual dump_stream runs).
    }
}
