//! Raw FFI declarations for the vendored edge264 slice-decode routines.
//!
//! The C side (`c/src/vacc_sw264.c`) exposes a single-threaded decoder core:
//! given externally parsed SPS/PPS/slice-header state and DPB plane pointers,
//! it decodes one slice (macroblocks + deblocking) into the current frame's
//! planes. The control plane (NAL parsing, DPB, POC, ref lists) lives in Rust.

use std::ffi::c_void;

/// Opaque C decoder handle (`Sw264Decoder`).
pub type Sw264Decoder = c_void;

/// SPS fields consumed by the C slice decoder (scan-order scaling lists).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Sw264Sps {
    pub chroma_format_idc: i8,
    pub chroma_array_type: i8,
    pub bit_depth_y: i8,
    pub bit_depth_c: i8,
    pub pic_width_in_mbs: u16,
    pub pic_height_in_mbs: i16,
    pub frame_crop_offsets: [i16; 4], // {top,right,bottom,left}
    pub log2_max_frame_num: i8,
    pub pic_order_cnt_type: i8,
    pub max_num_ref_frames: i8,
    pub direct_8x8_inference_flag: i8,
    pub seq_scaling_matrix_present: i32,
    pub scaling_list_4x4: [[u8; 16]; 6],
    pub scaling_list_8x8: [[u8; 64]; 2],
}

/// PPS fields consumed by the C slice decoder.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Sw264Pps {
    pub entropy_coding_mode_flag: i8,
    pub num_ref_idx_active: [i8; 2], // defaults (minus1 + 1)
    pub weighted_pred_flag: i8,
    pub weighted_bipred_idc: i8,
    pub qp_prime_y: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub transform_8x8_mode_flag: i8,
}

/// Per-frame setup: DPB slot planes + current frame buffers.
#[repr(C)]
pub struct Sw264Frame {
    pub planes: [*const u8; 32],
    pub mb_arrays: [*const c_void; 32],
    pub n_slots: i32,
    pub long_term_frames: u32,
    pub curr_plane: *mut u8,
    pub curr_mb: *mut c_void,
    /// Real DPB slot index of the current picture (flip-bit tracking).
    pub curr_slot: i32,
}

impl Sw264Frame {
    pub fn new() -> Self {
        Self {
            planes: [std::ptr::null(); 32],
            mb_arrays: [std::ptr::null(); 32],
            n_slots: 0,
            long_term_frames: 0,
            curr_plane: std::ptr::null_mut(),
            curr_mb: std::ptr::null_mut(),
            curr_slot: -1,
        }
    }
}

impl Default for Sw264Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-slice parameters (slice header already parsed by the Rust control plane).
#[repr(C)]
pub struct Sw264Slice {
    pub first_mb_in_slice: u32,
    pub slice_type: i8, // 0=P, 1=B, 2=I
    pub field_pic_flag: i8,
    pub bottom_field_flag: i8,
    pub nal_ref_idc_ne: i8,
    pub direct_spatial_mv_pred_flag: i8,
    pub disable_deblocking_filter_idc: i8,
    pub filter_offset_a: i8,
    pub filter_offset_b: i8,
    pub cabac_init_idc: i8, // 1-based (I: 0, P/B: 1 + ue value)
    pub qp_y: i16,
    pub num_ref_idx_active: [i8; 2],
    /// NAL start incl. header byte (>= 16 bytes headroom before).
    pub data: *const u8,
    /// Bits to skip from `data` to reach slice data (EPB-stripped space).
    pub skip_bits: u32,
    /// One past NAL end (>= 16 bytes tail guard recommended).
    pub end: *const u8,
    pub refpic_list: [[i8; 32]; 2],
    pub diff_poc: [i16; 32],
    pub luma_log2_weight_denom: i8,
    pub chroma_log2_weight_denom: i8,
    pub explicit_weights: [[i16; 64]; 3],
    pub explicit_offsets: [[i8; 64]; 3],
}

impl Sw264Slice {
    pub fn new() -> Self {
        Self {
            first_mb_in_slice: 0,
            slice_type: 2,
            field_pic_flag: 0,
            bottom_field_flag: 0,
            nal_ref_idc_ne: 1,
            direct_spatial_mv_pred_flag: 1,
            disable_deblocking_filter_idc: 0,
            filter_offset_a: 0,
            filter_offset_b: 0,
            cabac_init_idc: 0,
            qp_y: 26,
            num_ref_idx_active: [1, 1],
            data: std::ptr::null(),
            skip_bits: 0,
            end: std::ptr::null(),
            refpic_list: [[0; 32]; 2], // unused entries point at slot 0 (dummy buffer)
            diff_poc: [0; 32],
            luma_log2_weight_denom: 0,
            chroma_log2_weight_denom: 0,
            explicit_weights: [[0; 64]; 3],
            explicit_offsets: [[0; 64]; 3],
        }
    }
}

impl Default for Sw264Slice {
    fn default() -> Self {
        Self::new()
    }
}

unsafe extern "C" {
    pub fn sw264_alloc() -> *mut Sw264Decoder;
    pub fn sw264_free(psw: *mut *mut Sw264Decoder);
    pub fn sw264_macroblock_size() -> usize;
    /// Fill a slot's mb array padding slots with unavail_mb (upstream alloc_frame).
    pub fn sw264_init_mb_buffer(mb: *mut c_void, width: i32, height: i32);
    /// Reset a slot's recovery_bits sentinel to 0 (call on (re)allocation of its mb array).
    pub fn sw264_reset_slot_flip(sw: *mut Sw264Decoder, slot: i32);
    pub fn sw264_set_sps(
        sw: *mut Sw264Decoder,
        sps: *const Sw264Sps,
        stride_y: *mut i32,
        stride_c: *mut i32,
        plane_size_y: *mut i32,
        plane_size_c: *mut i32,
    ) -> i32;
    pub fn sw264_set_pps(sw: *mut Sw264Decoder, pps: *const Sw264Pps);
    pub fn sw264_frame_start(sw: *mut Sw264Decoder, fr: *const Sw264Frame);
    /// Returns 0 or a negative errno.
    pub fn sw264_decode_slice(sw: *mut Sw264Decoder, sl: *const Sw264Slice) -> i32;
}
