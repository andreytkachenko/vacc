//! H.264 bitstream generation: SPS, PPS, and packed slice headers.
//!
//! All byte layouts are verified against FFmpeg n8.0, x264, and the project
//! parser oracle (see memory `h264_encode_probe.md`). Do not reorder fields —
//! a single shifted bit flips `constrained_intra_pred_flag` and breaks
//! decoder-side intra availability checks.

/// Minimal Exp-Golomb bit writer with emulation prevention on finalize.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bitpos: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put_bit(&mut self, v: bool) {
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

    /// Write `bits` LSB-first... no: MSB-first fixed-width value.
    pub fn put_u(&mut self, v: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.put_bit((v >> i) & 1 == 1);
        }
    }

    /// Write an unsigned Exp-Golomb value.
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

    /// Write a signed Exp-Golomb value.
    pub fn put_se(&mut self, v: i32) {
        let ue = if v < 0 { (-2 * v) as u32 } else { (2 * v) as u32 };
        self.put_ue(ue);
    }

    /// Number of valid bits written so far (before stop bit).
    pub fn valid_bits(&self) -> usize {
        if self.bitpos == 0 {
            self.bytes.len() * 8
        } else {
            (self.bytes.len() - 1) * 8 + self.bitpos as usize
        }
    }

    /// Finalize: stop bit + byte alignment + emulation prevention.
    /// Returns the complete RBSP bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.put_bit(true); // rbsp_stop_bit
        while self.bitpos != 0 {
            self.put_bit(false);
        }
        // Emulation prevention: after two consecutive zero bytes, insert a
        // 0x03 BEFORE any byte <= 0x03 (so the stream never contains
        // `00 00 00`/`00 00 01`/`00 00 02`).
        let input = std::mem::take(&mut self.bytes);
        let mut out = Vec::with_capacity(input.len() * 2);
        let mut zeros = 0u32;
        for &b in &input {
            if zeros >= 2 && (b as u32) <= 3 {
                out.push(3);
                zeros = 0;
            }
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
        out
    }
}

/// H.264 SPS for the supported encoder configuration:
/// Main profile, 4:2:0, 8-bit, poc_type=0, frame coding only.
///
/// Verified layout (320x240, mref=1): `64 00 1f ac ca 81 41 f9`.
pub fn make_sps(width_mbs: u32, height_mbs: u32, level_idc: u32, max_ref_frames: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_u(100, 8); // profile_idc (Main)
    w.put_u(0x00, 8); // constraint_set flags + reserved
    w.put_u(level_idc, 8);
    w.put_ue(0); // seq_parameter_set_id
    w.put_ue(1); // chroma_format_idc (4:2:0)
    w.put_ue(0); // bit_depth_luma_minus8 (ue(v) in High-family profiles)
    w.put_ue(0); // bit_depth_chroma_minus8 (ue(v))
    w.put_bit(false); // qpprime_y_zero_transform_bypass_flag
    w.put_bit(false); // seq_scaling_matrix_present_flag
    w.put_ue(0); // log2_max_frame_num_minus4 (frame_num u(4))
    w.put_ue(0); // pic_order_cnt_type (0)
    w.put_ue(4); // log2_max_pic_order_cnt_lsb_minus4 (poc_lsb u(8))
    w.put_ue(max_ref_frames); // max_num_ref_frames
    w.put_bit(false); // gaps_in_frame_num_value_allowed_flag
    w.put_ue(width_mbs - 1); // pic_width_in_mbs_minus1
    w.put_ue(height_mbs - 1); // pic_height_in_map_units_minus1
    w.put_bit(true); // frame_mbs_only_flag
    w.put_bit(true); // direct_8x8_inference_flag
    w.put_bit(false); // frame_cropping_flag
    w.put_bit(false); // vui_parameters_present_flag
    w.finish()
}

/// H.264 PPS: CAVLC, single reference, deblocking control on.
///
/// The all-zero tail (transform_8x8_mode_flag, pic_scaling_matrix_present_flag,
/// second_chroma_qp_index_offset) is required: FFmpeg reads it whenever bits
/// remain before the stop bit, and the project oracle peeks one bit — zeros
/// satisfy both. Verified layout: `ce 3c 30`.
pub fn make_pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_ue(0); // pic_parameter_set_id
    w.put_ue(0); // seq_parameter_set_id
    w.put_bit(false); // entropy_coding_mode_flag (CAVLC)
    w.put_bit(false); // bottom_field_pic_order_in_frame_present_flag
    w.put_ue(0); // num_slice_groups_minus1
    w.put_ue(0); // num_ref_idx_l0_default_active_minus1
    w.put_ue(0); // num_ref_idx_l1_default_active_minus1
    w.put_bit(false); // weighted_pred_flag
    w.put_u(0, 2); // weighted_bipred_idc (UNCONDITIONAL)
    w.put_se(0); // pic_init_qp_minus26
    w.put_se(0); // pic_init_qs_minus26
    w.put_se(0); // chroma_qp_index_offset
    w.put_bit(true); // deblocking_filter_control_present_flag
    w.put_bit(false); // constrained_intra_pred_flag
    w.put_bit(false); // redundant_pic_cnt_present_flag
    // Tail (more_rbsp_data):
    w.put_bit(false); // transform_8x8_mode_flag
    w.put_bit(false); // pic_scaling_matrix_present_flag
    w.put_se(0); // second_chroma_qp_index_offset (se(v), not a present flag)
    w.finish()
}

/// Packed slice header: `[00 00 01][NAL][slice_header RBSP]`.
///
/// The RBSP is raw bits without stop bit or emulation prevention — the
/// driver inserts EPB (from the NAL byte) and appends
/// rbsp_slice_trailing_bits. Returns `(bytes, valid_bit_count)` where
/// `valid_bit_count` = 32 + RBSP bits (for `VAEncPackedHeaderParameterBuffer.bit_length`).
pub fn make_slice_hdr(is_idr: bool, frame_num: u32, poc_lsb: u32, idr_pic_id: u32) -> (Vec<u8>, u32) {
    let mut w = BitWriter::new();
    w.put_ue(0); // first_mb_in_slice
    w.put_ue(if is_idr { 7 } else { 5 }); // slice_type (I=7, P=5; raw H.264 numbering)
    w.put_ue(0); // pic_parameter_set_id
    w.put_u(frame_num & 0xF, 4); // frame_num u(log2_max_frame_num_minus4+4)
    if is_idr {
        w.put_ue(idr_pic_id); // idr_pic_id
    }
    w.put_u(poc_lsb & 0xFF, 8); // pic_order_cnt_lsb u(log2_max_poc_lsb_minus4+4)
    if !is_idr {
        // Both flags are UNCONDITIONAL for non-I slices (no num_ref_idx>1 condition).
        w.put_bit(false); // num_ref_idx_active_override_flag
        w.put_bit(false); // ref_pic_list_modification_flag_l0
    }
    // dec_ref_pic_marking: present because nal_ref_idc != 0.
    if is_idr {
        w.put_bit(false); // no_output_of_prior_pics_flag
        w.put_bit(false); // long_term_reference_flag
    } else {
        w.put_bit(false); // adaptive_ref_pic_marking_mode_flag (no MMCO ops)
    }
    w.put_se(0); // slice_qp_delta
    w.put_ue(0); // slice_disable_deblocking_filter_idc — ue(v), NOT u(2)
    w.put_se(0); // pic_init... alpha_c0_offset
    w.put_se(0); // beta_offset

    let rbsp_bits = w.valid_bits() as u32;
    let mut out = Vec::with_capacity(4 + 8);
    out.extend_from_slice(&[0, 0, 1]);
    // NAL header: IDR = ref_idc 3, type 5 (0x65); P = ref_idc 2, type 1 (0x41).
    out.push(if is_idr { 0x65 } else { 0x41 });
    // Raw zero-padded RBSP bits — NO emulation prevention here: with
    // has_emulation_bytes=0 the driver inserts EPB from the NAL byte onward
    // and appends rbsp_slice_trailing_bits. Pre-inserting 0x03 bytes would
    // shift the payload bits and corrupt the header.
    out.extend_from_slice(&w.bytes);
    (out, 32 + rbsp_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sps_matches_verified_bytes() {
        // 320x240 (20x15 MB), level 31, mref=1.
        let sps = make_sps(20, 15, 31, 1);
        assert_eq!(sps, vec![0x64, 0x00, 0x1f, 0xac, 0xca, 0x81, 0x41, 0xf9]);
    }

    #[test]
    fn pps_matches_verified_bytes() {
        let pps = make_pps();
        assert_eq!(pps, vec![0xce, 0x3c, 0x30]);
    }

    #[test]
    fn slice_hdr_idr_layout() {
        let (bytes, bits) = make_slice_hdr(true, 0, 0, 1);
        assert_eq!(&bytes[0..4], &[0, 0, 1, 0x65]);
        // RBSP: first_mb ue(0)=1, st ue(7)=7, pps ue(0)=1, fn u(4)=4,
        // idr ue(1)=3, poc u(8)=8, no_out=1, ltrf=1, qp se(0)=1,
        // db ue(0)=1, alpha se(0)=1, beta se(0)=1 => 30 bits.
        assert_eq!(bits, 32 + 30);
    }

    #[test]
    fn slice_hdr_p_layout() {
        let (bytes, bits) = make_slice_hdr(false, 1, 1, 0);
        assert_eq!(&bytes[0..4], &[0, 0, 1, 0x41]);
        // RBSP: first_mb ue(0)=1, st ue(5)=5, pps ue(0)=1, fn u(4)=4, poc u(8)=8,
        // override=1, rplm=1, mmco=1, qp se(0)=1, db ue(0)=1, alpha=1, beta=1
        // => 26 bits.
        assert_eq!(bits, 32 + 26);
    }

    #[test]
    fn exp_golomb_lengths() {
        let mut w = BitWriter::new();
        w.put_ue(0);
        w.put_ue(1);
        w.put_ue(7);
        w.put_se(-1);
        w.put_se(3);
        let bits = w.valid_bits();
        // ue(0)=1, ue(1)=3('010'), ue(7)=7, se(-1)=ue(1)=3, se(3)=ue(6)=5 => 19 bits
        assert_eq!(bits, 19);
    }

    #[test]
    fn exp_golomb_bit_patterns() {
        // Verified against the C probe stream parsed by FFmpeg + project oracle.
        let mut w = BitWriter::new();
        w.put_ue(0);
        w.put_ue(1);
        // '1' then '010' => bits 1,0,1,0 => 0xA0
        assert_eq!(&w.bytes, &[0xA0]);
        let mut w = BitWriter::new();
        w.put_ue(5); // slice_type P: '00' + '110'
        assert_eq!(w.valid_bits(), 5);
        let mut w = BitWriter::new();
        w.put_se(-2); // ue(4) = '00' + '101' => bits 0,0,1,0,1 => 0x28
        assert_eq!(&w.bytes, &[0x28]);
    }
}
