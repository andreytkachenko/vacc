//! Tier D: slice data parser — faithful port of `c/src/edge264_slice.c`.
//!
//! Parses the coupled I/P/B macroblock loop (`parse_slice_data`): CAVLC and
//! CABAC entropy, residuals, QP update, and neighbour propagation. The C file
//! is compiled twice (once with `CABAC=0`, once with `CABAC=1`); this port
//! uses a runtime `is_cabac` branch instead. Each fully-parsed macroblock is
//! emitted as a 304-byte record (a byte-exact `#[repr(C)]` copy of
//! `Edge264Macroblock`) and pinned to golden hashes in the stream tests.
//!
//! Motion-vector prediction for the large partitions reuses the verified Tier C
//! core (`mvpred.rs`); the 8x8 sub-partition prediction is transcribed inline
//! here from `parse_P_sub_mb` / `parse_B_sub_mb` using the same neighbour
//! offset-array gather pattern.

#![allow(non_snake_case)]

use super::bits::{SliceBits, shld};
use super::cabac::Cabac;
use super::deblock::{DEBLOCK_LC_SIZE, DEBLOCK_LY_SIZE, DeblockRect, deblock_mb_inplace};
use super::inter;
use super::intra::{
    I4X4_DC, I4X4_DC_A, I4X4_DC_AB, I4X4_DC_B, I4X4_DDL, I4X4_DDL_C, I4X4_DDR, I4X4_H, I4X4_HD,
    I4X4_HU, I4X4_V, I4X4_VL, I4X4_VL_C, I4X4_VR, I8X8_DC, I8X8_DC_A, I8X8_DC_AB, I8X8_DC_AC,
    I8X8_DC_ACD, I8X8_DC_AD, I8X8_DC_B, I8X8_DC_BD, I8X8_DC_C, I8X8_DC_CD, I8X8_DC_D, I8X8_DDL,
    I8X8_DDL_C, I8X8_DDL_CD, I8X8_DDL_D, I8X8_DDR, I8X8_DDR_C, I8X8_H, I8X8_H_D, I8X8_HD, I8X8_HU,
    I8X8_HU_D, I8X8_V, I8X8_V_C, I8X8_V_CD, I8X8_V_D, I8X8_VL, I8X8_VL_C, I8X8_VL_CD, I8X8_VL_D,
    I8X8_VR, I8X8_VR_C, I16X16_DC, I16X16_DC_A, I16X16_DC_AB, I16X16_DC_B, I16X16_H, I16X16_P,
    I16X16_V, IC8X8_DC, IC8X8_DC_A, IC8X8_DC_AB, IC8X8_DC_B, IC8X8_H, IC8X8_P, IC8X8_V,
    intra_chroma, intra4x4, intra8x8, intra16x16,
};
use super::residual::{add_dc4x4, add_idct4x4, add_idct8x8, transform_dc2x2, transform_dc4x4};

/// Leading zero-filled margin before the Y plane of `rust_planes` buffers.
/// The C core's intra kernels can read up to a few bytes before the luma
/// plane start in corner configurations (upstream UB); the margin makes those
/// reads valid (zero) memory on the Rust side instead of panics.
pub const PIXEL_MARGIN: usize = 32;

/// Per-macroblock parse record: `CurrMbAddr` u32 + 304-byte macroblock.
pub const SLICEDATA_RECORD_LEN: usize = 4 + 304;

/// Byte-exact copy of `Edge264MbFlags` (16 bytes). Field order and sizes match
/// the C union so that `#[repr(C)]` reproduces the oracle layout: 8 flag bytes,
/// then the `coded_block_flags_16x16` union (4 bytes) and `inter_eqs` union
/// (4 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct RustMbFlags {
    pub mb_field_decoding_flag: i8,
    pub mb_skip_flag: i8,
    pub mb_type_I_NxN: i8,
    pub mb_type_B_Direct: i8,
    pub transform_size_8x8_flag: i8,
    pub intra_chroma_pred_mode_non_zero: i8,
    pub CodedBlockPatternChromaDC: i8,
    pub CodedBlockPatternChromaAC: i8,
    /// C `union { int8_t coded_block_flags_16x16[3]; int32_t ...; }` (4 bytes).
    pub coded_block_flags_16x16: [u8; 4],
    /// C `union { uint8_t inter_eqs[4]; uint32_t inter_eqs_s; }` (4 bytes).
    pub inter_eqs: [u8; 4],
}

impl RustMbFlags {
    /// C `flags_twice`: the per-byte "twice" mask used to compute `inc`.
    pub const fn twice() -> Self {
        RustMbFlags {
            mb_field_decoding_flag: 0,
            mb_skip_flag: 0,
            mb_type_I_NxN: 0,
            mb_type_B_Direct: 0,
            transform_size_8x8_flag: 0,
            intra_chroma_pred_mode_non_zero: 0,
            CodedBlockPatternChromaDC: 1,
            CodedBlockPatternChromaAC: 1,
            coded_block_flags_16x16: [1, 1, 1, 0],
            inter_eqs: [0; 4],
        }
    }
}

/// Byte-exact copy of `Edge264Macroblock` (304 bytes, 16-aligned in C).
///
/// Offsets (verified against the C build via `offsetof`): err@0 rec@1
/// mbIsInterFlag@2 filter_edges@3 QP@4 bits@8 Intra4x4PredMode@16 nC@32
/// absMvd@80 f@144 refIdx@160 refPic@168 mvs@176. The plain array fields keep
/// the C union sizes (QP=4, bits=8) so `#[repr(C)]` needs no padding beyond the
/// natural alignment of `bits` (u32x2 @8) and `mvs` (i16 @176).
#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct RustMb {
    pub error_probability: i8,
    pub recovery_bits: i8,
    pub mbIsInterFlag: i8,
    pub filter_edges: i8,
    /// C `union { uint8_t QP[3]; i8x4 QP_s; }` (4 bytes).
    pub QP: [u8; 4],
    /// C `union { uint32_t bits[2]; uint64_t bits_l; }` (8 bytes).
    pub bits: [u32; 2],
    pub Intra4x4PredMode: [i8; 16],
    pub nC: [i8; 48],
    pub absMvd: [u8; 64],
    pub f: RustMbFlags,
    pub refIdx: [i8; 8],
    pub refPic: [i8; 8],
    /// C `union { int16_t mvs[64]; int32_t mvs_s[32]; ...; }` (128 bytes).
    pub mvs: [i16; 64],
}

impl Default for RustMb {
    fn default() -> Self {
        RustMb {
            error_probability: 0,
            recovery_bits: 0,
            mbIsInterFlag: 0,
            filter_edges: 0,
            QP: [0; 4],
            bits: [0; 2],
            Intra4x4PredMode: [0; 16],
            nC: [0; 48],
            absMvd: [0; 64],
            f: RustMbFlags::default(),
            refIdx: [0; 8],
            refPic: [0; 8],
            mvs: [0; 64],
        }
    }
}

impl RustMb {
    /// Serialize the deblock-relevant state as a 202-byte record (layout:
    /// `deblock::parse_mb`). Used by the in-slice row-lag / slice-tail
    /// deblocking.
    pub fn serialize_deblock(&self) -> [u8; 202] {
        let mut s = [0u8; 202];
        s[0] = self.QP[0];
        s[1] = self.QP[1];
        s[2] = self.QP[2];
        s[3] = self.mbIsInterFlag as u8;
        s[4] = self.filter_edges as u8;
        s[5..9].copy_from_slice(&self.f.inter_eqs);
        s[9] = self.f.transform_size_8x8_flag as u8;
        for (i, v) in s[10..58].iter_mut().enumerate() {
            *v = self.nC[i] as u8;
        }
        for (i, v) in s[58..66].iter_mut().enumerate() {
            *v = self.refIdx[i] as u8;
        }
        for (i, v) in s[66..74].iter_mut().enumerate() {
            *v = self.refPic[i] as u8;
        }
        for i in 0..64 {
            s[74 + 2 * i..76 + 2 * i].copy_from_slice(&self.mvs[i].to_le_bytes());
        }
        s
    }

    /// Read the `mvs` field reinterpreted as `int32_t mvs_s[32]` (packed
    /// i16 pairs), mirroring C `mb->mvs_s[idx]`. `idx` may be negative (a
    /// neighbour offset); this is raw pointer arithmetic into the contiguous
    /// macroblock buffer, exactly as the C code does.
    ///
    /// # Safety
    /// `idx` must stay within the allocated macroblock buffer (neighbour
    /// offsets only), as with every C `mvs_s[idx]` access.
    #[inline]
    pub unsafe fn mvs_s(&self, idx: isize) -> i32 {
        let base = unsafe { (self as *const RustMb).cast::<u8>().add(176) };
        unsafe { *(base.cast::<i32>().offset(idx)) }
    }
}

/// Per-slice parse state. Mirrors the fields of `Edge264Task` +
/// `Edge264Context` used by `edge264_slice.c`. The macroblock buffer is held as
/// a raw pointer (the owning `Vec<RustMb>` must outlive this context and not be
/// aliased); neighbour access uses the same negative-offset pointer arithmetic
/// as the C code.
pub struct SliceContext<'a> {
    // entropy coding state (CAVLC caches or CABAC offset/range, per `is_cabac`)
    pub bits: SliceBits<'a>,
    pub cabac: Cabac,
    pub is_cabac: bool,

    // macroblock storage (stride = pic_width_in_mbs + 1)
    pub mb_buffer: *mut RustMb,
    pub mb_pos: usize,
    /// Collocated macroblock for B-slice direct mode (nullptr unless B).
    pub mb_col: *const RustMb,
    /// Collocated frame's mb array (non-null only for B slices; C
    /// `t.mbCol_buffer`).
    pub mb_col_buffer: *const RustMb,
    /// Per-MB parse-state dump target (C `t.test_mb_dump`); null = disabled.
    pub mb_dump: *mut u8,
    pub mb_dump_off: usize,
    pub mb_dump_cap: usize,
    /// Spatial neighbours (set per-MB in `parse_slice_data`); point at
    /// `UNAVAIL_MB` when unavailable.
    pub mb_a: *const RustMb,
    pub mb_b: *const RustMb,
    pub mb_c: *const RustMb,
    pub mb_d: *const RustMb,

    // picture geometry / task fields
    pub pic_width_in_mbs: i16,
    pub pic_height_in_mbs: i16,
    pub first_mb_in_slice: u32,
    pub slice_type: i8,
    pub cabac_init_idc: i8,
    pub frame_flip_bit: i8,
    pub disable_deblocking_filter_idc: i8,
    pub chroma_array_type: i8,
    pub direct_spatial_mv_pred_flag: i8,
    pub direct_8x8_inference_flag: i8,
    pub num_ref_idx_active: [i8; 2],
    pub pps_transform_8x8_mode_flag: i8,
    pub weighted_bipred_idc: i8,
    /// C `t.luma_log2_weight_denom` / `t.chroma_log2_weight_denom`.
    pub luma_log2_weight_denom: i8,
    pub chroma_log2_weight_denom: i8,
    /// C `t.explicit_weights[iYCbCr][slot]` / `t.explicit_offsets`: L0 refs at
    /// [0..n0), L1 refs at [32..32+n1).
    pub explicit_weights: [[i16; 64]; 3],
    pub explicit_offsets: [[i8; 64]; 3],
    /// C `ctx->implicit_weights[refIdxL0][refIdxL1]`, stored with the +64
    /// offset (B slices only).
    pub implicit_weights: [[u8; 32]; 32],
    pub ref_pic_list: [[i8; 32]; 2],
    pub diff_poc: [i16; 32],
    pub prev_long_term_frames: u32,
    pub qp_y: i16,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,

    // per-macroblock loop state
    pub mbx: i16,
    pub mby: i16,
    pub curr_mb_addr: i32,
    pub mb_skip_run: i32,
    pub col_short_term: bool,

    // CABAC context increments + neighbour unavailability
    pub inc: RustMbFlags,
    pub unavail4x4: [i8; 48],
    pub nc_inc: [[i8; 16]; 3],

    // neighbour offset arrays (relative to the current macroblock)
    pub a4x4_int8: [i16; 16],
    pub b4x4_int8: [i32; 16],
    pub acbcr_int8: [i16; 16],
    pub bcbcr_int8: [i32; 16],
    pub refidx4x4_c: [i8; 16],
    pub absmvd_a: [i16; 16],
    pub absmvd_b: [i32; 16],
    pub mvs_a: [i16; 16],
    pub mvs_b: [i32; 16],
    pub mvs_c: [i32; 16],
    pub mvs_d: [i32; 16],

    // inter context (updated during parsing)
    pub transform_8x8_mode_flag: i8,
    pub num_ref_idx_mask: u8,
    pub dist_scale_factor: [i16; 32],
    pub clip_ref_idx: [i8; 8],
    pub map_pic_to_list0: [i8; 32],

    // residual context
    pub ctx_idx_offsets: [i16; 4],
    pub coeff_abs_inc: [i8; 8],
    pub sig_inc: [i8; 64],
    pub last_inc: [i8; 64],
    pub scan: [i8; 64],
    pub qp_c: [[i8; 64]; 2],
    /// Non-scaled residual coefficients (C `ctx->c[64]`).
    pub c: [i32; 64],

    // I-path / QP state (C `ctx->mb_qp_delta_nz`, `ctx->t.QP_s`)
    pub mb_qp_delta_nz: i8,
    pub qp_s: [u8; 4],
    /// Per-component bit depth (C `ctz(ctx->t.samples_clip[i][0] + 1)`), used
    /// only by I_PCM sample consumption.
    pub bit_depth: [u32; 3],

    // pixel plane access (C `t.samples_buffers[currPic]`, `t.stride[iYCbCr]`,
    // `t.plane_size_Y`)
    /// Start of the Rust pixel buffer's leading margin (`PIXEL_MARGIN` bytes
    /// before `samples_base`; see `plane_slice`).
    pub pixel_margin_base: *mut u8,
    pub samples_base: *mut u8,
    /// Row strides in sample units: luma (`t.stride[0]`) and chroma
    /// (`t.stride[1]`; the Cr row sits at +stride/2 within each chroma row).
    pub stride: [u16; 2],
    pub plane_size_y: u32,
    /// Chroma plane size in bytes (C `t.plane_size_C`).
    pub plane_size_c: u32,
    /// Coded frame extent in luma samples (Rust-only: bounds for the deblock
    /// window copy; C has no bounds checks).
    pub coded_w: u32,
    pub coded_h: u32,
    /// Current macroblock pixel positions (C `ctx->samples_mb[0..3]`): luma,
    /// Cb, Cr — top-left of the current MB's 16x16 / 8x8 chroma blocks.
    pub samples_mb: [*mut u8; 3],
    /// Rust reference planes (C `t.samples_buffers`): Y-plane base (after
    /// `PIXEL_MARGIN`) of each DPB slot, indexed by `mb->refPic`. Null for
    /// empty slots. All buffers share the same stride/plane layout.
    pub ref_plane_bases: Vec<*const u8>,
    /// Inter MC neighborhood scratch (replaces per-block heap Vecs): max luma
    /// neighborhood (16+5)*(16+5) = 441 bytes, max chroma 9*2*9 = 162 bytes.
    pub mc_y: [u8; 441],
    pub mc_c: [u8; 162],
    /// Reusable deblock window buffers (MB origin at row 48 col 32 / chroma
    /// buffer row 48 col 8): avoids per-MB zeroing and Vec traffic. The
    /// kernel never reads outside the filled 20x20 luma / 20x10 chroma window.
    pub dblk_y: [u8; DEBLOCK_LY_SIZE],
    pub dblk_c: [u8; DEBLOCK_LC_SIZE],

    // scaling / deblock parameters (C `t.pps.weightScale*`, `t.FilterOffset*`,
    // `t.next_deblock_addr`)
    /// C `pps.weightScale4x4_v`: slot = iYCbCr + inter*3. All 16 when
    /// seq_scaling_matrix_present_flag == 0.
    pub ws4: [[i8; 16]; 6],
    /// C `pps.weightScale8x8_v`: slot = iYCbCr*2 + inter.
    pub ws8: [[i8; 64]; 6],
    /// C `t.FilterOffsetA/B` = slice alpha/beta offsets * 2.
    pub filter_offset_a: i32,
    pub filter_offset_b: i32,
    /// C `t.next_deblock_addr` (INT_MIN when deblocking is not tracked for
    /// this slice).
    pub next_deblock_addr: i32,
}

impl<'a> SliceContext<'a> {
    /// Address of the current macroblock in the buffer.
    #[inline]
    pub fn mb(&self) -> *mut RustMb {
        unsafe { self.mb_buffer.add(self.mb_pos) }
    }
    /// C `mb->mvs_s[k]` (packed i16 pairs reinterpreted as i32); `k` may be a
    /// negative neighbour offset.
    ///
    /// # Safety
    /// `k` must stay within the allocated macroblock buffer (neighbour
    /// offsets only).
    #[inline]
    pub unsafe fn read_mvs_s(&self, k: isize) -> i32 {
        let base = unsafe { (self.mb() as *const u8).add(176) };
        unsafe { *(base.cast::<i32>().offset(k)) }
    }
    /// C `*(mb + byte_off)` — read a raw i8 at a neighbour byte offset.
    ///
    /// # Safety
    /// `byte_off` must stay within the allocated macroblock buffer (neighbour
    /// offsets only).
    #[inline]
    pub unsafe fn read_i8_at(&self, byte_off: isize) -> i8 {
        unsafe { *((self.mb() as *const i8).offset(byte_off)) }
    }

    /// The whole Rust pixel-plane region (leading margin + Y plane + C plane)
    /// as one mutable slice. Intra kernels index it with offsets relative to
    /// the Y plane start (see `plane_off`), mirroring C's raw pointer reads.
    fn plane_slice(&mut self) -> &mut [u8] {
        let len = PIXEL_MARGIN + self.plane_size_y as usize + self.plane_size_c as usize;
        unsafe { std::slice::from_raw_parts_mut(self.pixel_margin_base, len) }
    }

    /// Offset within `plane_slice` for a plane-relative pointer.
    #[inline]
    fn plane_off(&self, p: *const u8) -> usize {
        (p as usize - self.samples_base as usize) + PIXEL_MARGIN
    }
}

/// Copy `n` samples from source column `x0` of the plane row `src_row`
/// (width `swide`) into `dst`, clamping at the picture edges (columns 0 /
/// swide-1) — the per-row form of the neighborhood gather's per-pixel clamp.
#[inline]
fn copy_clamped_row(dst: &mut [u8], src_row: *const u8, x0: i32, swide: i32, n: usize) {
    let mut c = 0usize;
    while c < n && (x0 + c as i32) < 0 {
        dst[c] = unsafe { *src_row };
        c += 1;
    }
    let mut end = n;
    while end > c && (x0 + end as i32 - 1) >= swide {
        end -= 1;
    }
    if end > c {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_row.add((x0 + c as i32) as usize),
                dst.as_mut_ptr().add(c),
                end - c,
            );
        };
        c = end;
    }
    while c < n {
        dst[c] = unsafe { *src_row.add((swide - 1) as usize) };
        c += 1;
    }
}

/// wod weighting for one MC sample, matching `inter_luma` / `inter_chroma`
/// bit-for-bit: luma shifts arithmetically by `sh`; chroma uses
/// `sra_machine` semantics. Identity when (wq, wp, oy, sh) = (0, 1, 0, 0).
#[inline]
fn mc_weight(p: u8, q: u8, wq: i32, wp: i32, oy: i32, sh: u32, chroma: bool) -> u8 {
    let s = (q as i32 * wq + p as i32 * wp).clamp(i16::MIN as i32, i16::MAX as i32);
    let v = (s + oy).clamp(i16::MIN as i32, i16::MAX as i32);
    let v = if chroma {
        inter::sra_machine(v, sh)
    } else {
        v >> sh
    };
    v.clamp(0, 255) as u8
}

/// One integer-pel MC row: clamped copy + wod weighting (see `mc_weight`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn mc_int_row(
    dst_row: *mut u8,
    src_row: *const u8,
    x0: i32,
    swide: i32,
    n: usize,
    wq: i32,
    wp: i32,
    oy: i32,
    sh: u32,
    chroma: bool,
) {
    let weighted = !(wq == 0 && wp == 1 && oy == 0 && sh == 0);
    let mut c = 0usize;
    while c < n && (x0 + c as i32) < 0 {
        let p = unsafe { *src_row };
        unsafe { *dst_row.add(c) = mc_weight(p, *dst_row.add(c), wq, wp, oy, sh, chroma) };
        c += 1;
    }
    let mut end = n;
    while end > c && (x0 + end as i32 - 1) >= swide {
        end -= 1;
    }
    if end > c {
        if !weighted {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_row.add((x0 + c as i32) as usize),
                    dst_row.add(c),
                    end - c,
                );
            };
        } else {
            while c < end {
                let p = unsafe { *src_row.add((x0 + c as i32) as usize) };
                unsafe { *dst_row.add(c) = mc_weight(p, *dst_row.add(c), wq, wp, oy, sh, chroma) };
                c += 1;
            }
        }
        c = end;
    }
    while c < n {
        let p = unsafe { *src_row.add((swide - 1) as usize) };
        unsafe { *dst_row.add(c) = mc_weight(p, *dst_row.add(c), wq, wp, oy, sh, chroma) };
        c += 1;
    }
}

impl SliceContext<'_> {
    /// C `decode_inter` (edge264_inter.c:1108): motion-compensate one inter
    /// block into `samples_mb`. `i` indexes the MV pair `mvs[2*i]/[2*i+1]` and
    /// the 8x8-region reference `refIdx[i>>2]`/`refPic[i>>2]`; the 4x4-block
    /// origin within the MB is `i & 15`. Bi-pred pairs call this twice (L0 at
    /// `i`, L1 at `i + 16`); the second call combines over the first's output
    /// through the wod q-term, so no dst prefill is ever needed.
    fn decode_inter(&mut self, i: usize, w: u32, h: u32) {
        let m = unsafe { &*self.mb() };
        let x = m.mvs[i * 2] as i32;
        let y = m.mvs[i * 2 + 1] as i32;
        let i8x8 = i >> 2;
        let i4x4 = i & 15;
        let ref_base = self.ref_plane_bases[m.refPic[i8x8] as usize];
        debug_assert!(!ref_base.is_null(), "inter MC into an empty DPB slot");
        let x_int_y = self.mbx as i32 * 16 + X444[i4x4] as i32 + (x >> 2);
        let x_int_c = self.mbx as i32 * 8 + (X444[i4x4] as i32 >> 1) + (x >> 3);
        let y_int_y = self.mby as i32 * 16 + Y444[i4x4] as i32 + (y >> 2);
        let y_int_c = self.mby as i32 * 8 + (Y444[i4x4] as i32 >> 1) + (y >> 3);
        let stride_y = self.stride[0] as usize;
        let stride_c = self.stride[1] as usize;
        let width_y = self.pic_width_in_mbs as i32 * 16;
        let height_y = self.pic_height_in_mbs as i32 * 16;

        // prediction coeffs {wY, oY, logWD_Y, logWD_C, wCb, wCr, oCb, oCr}
        let pack_w = |w0: i32, w1: i32| -> i16 { ((w1 << 8) | (w0 & 255)) as i16 };
        let mut wod: [i16; 8] = inter::WOD_NO_WEIGHT; // no_weight
        let ref_idx = m.refIdx[i8x8] as i32;
        let ref_idx_x = m.refIdx[i8x8 ^ 4] as i32;
        if self.weighted_bipred_idc != 1 {
            if (((i8x8 as i32) - 4) | ref_idx_x) >= 0 {
                if self.weighted_bipred_idc == 0 {
                    // default2
                    wod = [257, 1, 1, 1, 257, 257, 1, 1];
                } else {
                    // implicit2
                    let w1 =
                        self.implicit_weights[ref_idx_x as usize][ref_idx as usize] as i32 - 64;
                    let p = pack_w(64 - w1, w1);
                    wod = [p, 32, 6, 6, p, p, 32, 32];
                    // w0 or w1 overflow if w1 is 128 or -64 (untested in
                    // conformance bitstreams)
                    if (w1 + 63) as u32 >= 191 {
                        let p = pack_w(2 - (w1 >> 5), w1 >> 5);
                        wod = [p, 1, 1, 1, p, p, 1, 1];
                    }
                }
            }
        } else if ref_idx_x < 0 {
            // explicit1
            let ri = (ref_idx + (i8x8 as i32 & 4) * 8) as usize;
            if self.explicit_weights[0][ri] < 128 {
                wod[0] = pack_w(0, self.explicit_weights[0][ri] as i32);
                wod[1] = (((self.explicit_offsets[0][ri] as i32 * 2 + 1)
                    << self.luma_log2_weight_denom as i32)
                    >> 1) as i16;
                wod[2] = self.luma_log2_weight_denom as i16;
            }
            if self.explicit_weights[1][ri] < 128 {
                wod[4] = pack_w(0, self.explicit_weights[1][ri] as i32);
                wod[5] = pack_w(0, self.explicit_weights[2][ri] as i32);
                wod[6] = (((self.explicit_offsets[1][ri] as i32 * 2 + 1)
                    << self.chroma_log2_weight_denom as i32)
                    >> 1) as i16;
                wod[7] = (((self.explicit_offsets[2][ri] as i32 * 2 + 1)
                    << self.chroma_log2_weight_denom as i32)
                    >> 1) as i16;
                wod[3] = self.chroma_log2_weight_denom as i16;
            }
        } else if i8x8 >= 4 {
            // explicit2
            let ri = (ref_idx + 32) as usize;
            let rix = ref_idx_x as usize;
            if (self.explicit_weights[0][rix] & self.explicit_weights[0][ri]) != 128 {
                wod[0] = pack_w(
                    self.explicit_weights[0][rix] as i32,
                    self.explicit_weights[0][ri] as i32,
                );
                wod[1] = (((self.explicit_offsets[0][rix] as i32
                    + self.explicit_offsets[0][ri] as i32
                    + 1)
                    | 1)
                    << self.luma_log2_weight_denom as i32) as i16;
                wod[2] = self.luma_log2_weight_denom as i16 + 1;
            } else {
                wod[0] = pack_w(
                    (self.explicit_weights[0][rix] >> 1) as i32,
                    (self.explicit_weights[0][ri] >> 1) as i32,
                );
                wod[1] = (((self.explicit_offsets[0][rix] as i32
                    + self.explicit_offsets[0][ri] as i32
                    + 1)
                    | 1)
                    << self.luma_log2_weight_denom as i32
                    >> 1) as i16;
                wod[2] = self.luma_log2_weight_denom as i16;
            }
            if (self.explicit_weights[1][rix] & self.explicit_weights[1][ri]) != 128 {
                wod[4] = pack_w(
                    self.explicit_weights[1][rix] as i32,
                    self.explicit_weights[1][ri] as i32,
                );
                wod[5] = pack_w(
                    self.explicit_weights[2][rix] as i32,
                    self.explicit_weights[2][ri] as i32,
                );
                wod[6] = (((self.explicit_offsets[1][rix] as i32
                    + self.explicit_offsets[1][ri] as i32
                    + 1)
                    | 1)
                    << self.chroma_log2_weight_denom as i32) as i16;
                wod[7] = (((self.explicit_offsets[2][rix] as i32
                    + self.explicit_offsets[2][ri] as i32
                    + 1)
                    | 1)
                    << self.chroma_log2_weight_denom as i32) as i16;
                wod[3] = self.chroma_log2_weight_denom as i16 + 1;
            } else {
                wod[4] = pack_w(
                    (self.explicit_weights[1][rix] >> 1) as i32,
                    (self.explicit_weights[1][ri] >> 1) as i32,
                );
                wod[5] = pack_w(
                    (self.explicit_weights[2][rix] >> 1) as i32,
                    (self.explicit_weights[2][ri] >> 1) as i32,
                );
                wod[6] = (((self.explicit_offsets[1][rix] as i32
                    + self.explicit_offsets[1][ri] as i32
                    + 1)
                    | 1)
                    << self.chroma_log2_weight_denom as i32
                    >> 1) as i16;
                wod[7] = (((self.explicit_offsets[2][rix] as i32
                    + self.explicit_offsets[2][ri] as i32
                    + 1)
                    | 1)
                    << self.chroma_log2_weight_denom as i32
                    >> 1) as i16;
                wod[3] = self.chroma_log2_weight_denom as i16;
            }
        }

        let wu = w as usize;
        let hu = h as usize;
        let cw = wu / 2;
        // chroma prediction first (C order; the dst regions don't overlap)
        let off_c = self.plane_off(self.samples_mb[1])
            + (Y444[i4x4] as usize >> 1) * stride_c
            + (X444[i4x4] as usize >> 1);
        let len_c = (hu - 1) * (stride_c / 2) + cw;
        // luma prediction
        let off_y = self.plane_off(self.samples_mb[0])
            + Y444[i4x4] as usize * stride_y
            + X444[i4x4] as usize;
        let len_y = (hu - 1) * stride_y + wu;
        // Raw pointers for the plane region and the MC neighborhood scratch:
        // the writes below would otherwise fight the borrow checker over
        // `plane_slice()`'s `&mut self`.
        let base = self.pixel_margin_base;
        let src_y = self.mc_y.as_mut_ptr();
        let src_c = self.mc_c.as_mut_ptr();

        // Integer-pel fast path: a clamped byte copy — C's SIMD row-copy
        // path. Edge clamping is per-row, not per-pixel; the no-weight case is
        // a plain copy of the interior run. Luma is integer at x&3 == 0 &&
        // y&3 == 0, but chroma needs x&7 == 0 && y&7 == 0: an integer luma pel
        // with x&7 == 4 (e.g. MV x = 4) sits at a half chroma pel, where the
        // bilinear is a true average, not a direct sample.
        let luma_int = (x & 3) == 0 && (y & 3) == 0;
        let chroma_int = (x & 7) == 0 && (y & 7) == 0;

        // Chroma prediction first (C order; the dst regions don't overlap).
        if chroma_int {
            for k in 0..(hu / 2) {
                for (pi, wi, oi) in [(0usize, 4usize, 6usize), (1usize, 5usize, 7usize)] {
                    let (wq_c, wp_c) = (
                        (wod[wi] & 0xFF) as i8 as i32,
                        ((wod[wi] >> 8) & 0xFF) as i8 as i32,
                    );
                    let ry = (y_int_c + k as i32).clamp(0, height_y / 2 - 1);
                    let soff = (ry * stride_c as i32) as usize + self.plane_size_y as usize;
                    mc_int_row(
                        unsafe { base.add(off_c + (2 * k + pi) * (stride_c / 2)) },
                        unsafe { ref_base.add(soff + pi * (stride_c / 2)) },
                        x_int_c,
                        width_y / 2,
                        cw,
                        wq_c,
                        wp_c,
                        wod[oi] as i32,
                        wod[3] as u32,
                        true,
                    );
                }
            }
        } else {
            // Half chroma pel: gather the clamped neighborhood (C edge_buf_c
            // semantics), then run the bilinear.
            for k in 0..hu / 2 + 1 {
                let ry = (y_int_c + k as i32).clamp(0, height_y / 2 - 1);
                let roff = (ry * stride_c as i32) as usize + self.plane_size_y as usize;
                copy_clamped_row(
                    unsafe {
                        std::slice::from_raw_parts_mut(src_c.add((2 * k) * (cw + 1)), cw + 1)
                    },
                    unsafe { ref_base.add(roff) },
                    x_int_c,
                    width_y / 2,
                    cw + 1,
                );
                copy_clamped_row(
                    unsafe {
                        std::slice::from_raw_parts_mut(src_c.add((2 * k + 1) * (cw + 1)), cw + 1)
                    },
                    unsafe { ref_base.add(roff + stride_c / 2) },
                    x_int_c,
                    width_y / 2,
                    cw + 1,
                );
            }
            inter::inter_chroma(
                unsafe { std::slice::from_raw_parts(src_c, (hu / 2 + 1) * 2 * (cw + 1)) },
                unsafe { std::slice::from_raw_parts_mut(base.add(off_c), len_c) },
                wu,
                hu,
                (x & 7) as u32,
                (y & 7) as u32,
                cw + 1,
                stride_c / 2,
                &wod,
            );
        }

        // Luma prediction.
        if luma_int {
            let (wq, wp) = (
                (wod[0] & 0xFF) as i8 as i32,
                ((wod[0] >> 8) & 0xFF) as i8 as i32,
            );
            for r in 0..hu {
                let ry = (y_int_y + r as i32).clamp(0, height_y - 1);
                mc_int_row(
                    unsafe { base.add(off_y + r * stride_y) },
                    unsafe { ref_base.add((ry * stride_y as i32) as usize) },
                    x_int_y,
                    width_y,
                    wu,
                    wq,
                    wp,
                    wod[1] as i32,
                    wod[2] as u32,
                    false,
                );
            }
        } else {
            // Fractional luma pel. Interior blocks read the neighborhood
            // directly from the reference plane (C's non-edge path — no
            // clamping needed); edge blocks gather the clamped neighborhood
            // into scratch (C edge_buf_l semantics). Then run the 6-tap filter.
            // The 6-tap filter reads relative rows/cols -2..h+2 / -2..w+2, so
            // absolute extents reach (x_int_y + wu + 2, y_int_y + hu + 2);
            // both must stay inside the frame for a direct read.
            let interior = x_int_y >= 2
                && x_int_y + wu as i32 + 2 < width_y
                && y_int_y >= 2
                && y_int_y + hu as i32 + 2 < height_y;
            if !interior {
                for r in 0..hu + 5 {
                    let ry = (y_int_y - 2 + r as i32).clamp(0, height_y - 1);
                    copy_clamped_row(
                        unsafe { std::slice::from_raw_parts_mut(src_y.add(r * (wu + 5)), wu + 5) },
                        unsafe { ref_base.add((ry * stride_y as i32) as usize) },
                        x_int_y - 2,
                        width_y,
                        wu + 5,
                    );
                }
            }
            let src_luma = if interior {
                // Full read extent: rows yInt-2..yInt+h+1, cols xInt-2..xInt+w+2
                // at the real stride — inside the ref plane by `interior`.
                unsafe {
                    std::slice::from_raw_parts(
                        ref_base.add(((y_int_y - 2) * stride_y as i32 + (x_int_y - 2)) as usize),
                        (hu + 4) * stride_y + wu + 5,
                    )
                }
            } else {
                unsafe { std::slice::from_raw_parts(src_y, (hu + 5) * (wu + 5)) }
            };
            inter::inter_luma(
                src_luma,
                unsafe { std::slice::from_raw_parts_mut(base.add(off_y), len_y) },
                wu,
                hu,
                ((y & 3) as u32) * 4 + (x & 3) as u32,
                if interior { stride_y } else { wu + 5 },
                stride_y,
                &wod,
            );
        }
    }

    /// C `deblock_mb(ctx)` for the macroblock at buffer position `pos` (row-lag
    /// and slice-tail call sites). Neighbour states follow C
    /// edge264_deblock.c L942-943: an unavailable neighbour contributes the
    /// current MB's own state. The pixel window is copied out of the real
    /// planes, filtered by `deblock_mb_inplace`, and copied back.
    fn deblock_mb_at(&mut self, pos: usize) {
        let m = unsafe { *self.mb_buffer.add(pos) };
        let fe = m.filter_edges as i32;
        if fe == 0 {
            return; // deblock_mb returns early
        }
        let width = self.pic_width_in_mbs as usize;
        let left_pos = if fe & 1 != 0 { pos - 1 } else { pos };
        let top_pos = if fe & 2 != 0 { pos - (width + 1) } else { pos };
        let mut mb_state = [0u8; 606];
        mb_state[0..202]
            .copy_from_slice(&unsafe { *self.mb_buffer.add(top_pos) }.serialize_deblock());
        mb_state[202..404]
            .copy_from_slice(&unsafe { *self.mb_buffer.add(left_pos) }.serialize_deblock());
        mb_state[404..606].copy_from_slice(&m.serialize_deblock());

        let mby = pos / (width + 1);
        let mbx = pos % (width + 1);
        let ly = DeblockRect {
            stride: self.stride[0] as usize,
            x: mbx * 16,
            y: mby * 16,
            w: self.coded_w as usize,
            h: self.coded_h as usize,
        };
        // Cb row 0 in chroma filter-row units (stride[1]>>1): Cb row r / Cr row
        // r sit at rows 2r / 2r+1, so MB row mby starts at 16*mby. `None` for
        // 4:0:0 (no chroma plane).
        let lc = (self.chroma_array_type == 1).then(|| DeblockRect {
            stride: self.stride[1] as usize / 2,
            x: mbx * 8,
            y: mby * 16,
            w: (self.coded_w / 2) as usize,
            h: self.coded_h as usize,
        });
        let entropy = self.is_cabac as i32;
        let off_a = self.filter_offset_a;
        let off_b = self.filter_offset_b;
        let plane_size_y = self.plane_size_y as usize;

        // plane_slice via the raw base pointer (same as the helper): keeps
        // these borrows disjoint from `dblk_y`/`dblk_c` below.
        let region = unsafe {
            std::slice::from_raw_parts_mut(
                self.pixel_margin_base,
                PIXEL_MARGIN + self.plane_size_y as usize + self.plane_size_c as usize,
            )
        };
        let (y_plane, c_plane) = {
            let (_, rest) = region.split_at_mut(PIXEL_MARGIN);
            rest.split_at_mut(plane_size_y)
        };
        deblock_mb_inplace(
            &mb_state,
            fe,
            entropy,
            off_a,
            off_b,
            &mut self.dblk_y,
            &mut self.dblk_c,
            y_plane,
            &ly,
            c_plane,
            lc.as_ref(),
        );
    }

    /// C vacc_sw264.c L1211-1226 / L1247-1262: deblock the MBs in
    /// `[next_deblock_addr, end)` in raster order. (C recomputes and advances
    /// its `_mb`/`samples_mb` pointers per step; here the buffer position is
    /// derived directly from the address: `pos = addr + addr / width`.)
    pub(crate) fn deblock_range(&mut self, end: i32) {
        let width = self.pic_width_in_mbs as usize;
        while self.next_deblock_addr < end {
            let addr = self.next_deblock_addr as usize;
            self.deblock_mb_at(addr + addr / width);
            self.next_deblock_addr += 1;
        }
    }

    /// C `initialize_context` (vacc_sw264.c): per-slice context setup. Must be
    /// called after the task fields, `mb_buffer`, `mb_col_buffer` and (for CABAC)
    /// nothing else are filled; it positions `mb_pos`/`mb_col` at the slice's
    /// first macroblock and initialises the offset/QP/residual tables.
    pub fn initialize_context(&mut self) {
        let w = self.pic_width_in_mbs;
        self.curr_mb_addr = self.first_mb_in_slice as i32;
        self.mby = (self.first_mb_in_slice / w as u32) as i16;
        self.mbx = (self.first_mb_in_slice % w as u32) as i16;
        let mb_offset = self.mbx as usize + self.mby as usize * (w as usize + 1);
        self.mb_pos = mb_offset;
        unsafe {
            self.mb_col = if self.mb_col_buffer.is_null() {
                self.mb_buffer.add(mb_offset) as *const RustMb
            } else {
                self.mb_col_buffer.add(mb_offset)
            };
        }
        // C `initialize_context` L760-762: first MB's pixel positions.
        let mbx = self.mbx as i64;
        let mby = self.mby as i64;
        let off0 = ((mbx + mby * self.stride[0] as i64) * 16) as usize;
        let off1 = ((mbx + mby * self.stride[1] as i64) * 8 + self.plane_size_y as i64) as usize;
        unsafe {
            self.samples_mb[0] = self.samples_base.add(off0);
            self.samples_mb[1] = self.samples_base.add(off1);
            self.samples_mb[2] = self.samples_mb[1].add((self.stride[1] >> 1) as usize);
        }
        self.a4x4_int8 = [0, 0, 2, 2, 1, 4, 3, 6, 8, 8, 10, 10, 9, 12, 11, 14];
        self.b4x4_int8 = [0, 1, 0, 1, 4, 5, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13];
        if self.chroma_array_type == 1 {
            self.acbcr_int8[..8].copy_from_slice(&[0, 0, 2, 2, 4, 4, 6, 6]);
            self.bcbcr_int8[..8].copy_from_slice(&[0, 1, 0, 1, 4, 5, 4, 5]);
        }

        // QP_Y2C loads (C `loadu128(QP_Y2C + 12 + offset)`); offsets are in
        // -12..12 so the 64-byte window never wraps.
        let o0 = (12 + self.chroma_qp_index_offset as i32) as usize;
        let o1 = (12 + self.second_chroma_qp_index_offset as i32) as usize;
        for i in 0..64 {
            self.qp_c[0][i] = QP_Y2C[o0 + i];
            self.qp_c[1][i] = QP_Y2C[o1 + i];
        }
        let qy = self.qp_y as usize;
        self.qp_s = [
            self.qp_y as u8,
            self.qp_c[0][qy] as u8,
            self.qp_c[1][qy] as u8,
            0,
        ];
        self.sig_inc[16..].copy_from_slice(&SIG_INC_8X8[0][16..]);
        self.last_inc[16..].copy_from_slice(&LAST_INC_8X8[16..]);
        self.scan[16..].copy_from_slice(&SCAN_8X8_CABAC[0][16..]);
        self.c = [0; 64];

        // P/B slices
        if self.slice_type < 2 {
            self.refidx4x4_c = [2, 3, 12, -1, 3, 6, 13, -1, 12, 13, 14, -1, 13, -1, 15, -1];
            self.absmvd_a = [0, 0, 4, 4, 2, 8, 6, 12, 16, 16, 20, 20, 18, 24, 22, 28];
            self.absmvd_b = [0, 2, 0, 2, 8, 10, 8, 10, 4, 6, 16, 18, 12, 14, 24, 26];
            self.mvs_a = [0, 0, 2, 2, 1, 4, 3, 6, 8, 8, 10, 10, 9, 12, 11, 14];
            self.mvs_b = [0, 1, 0, 1, 4, 5, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13];
            self.mvs_c = [0, 1, 1, -1, 4, 5, 5, -1, 3, 6, 9, -1, 7, -1, 13, -1];
            self.mvs_d = [0, 1, 2, 0, 4, 5, 1, 4, 8, 2, 10, 8, 3, 6, 9, 12];
            self.num_ref_idx_mask = (self.num_ref_idx_active[0] > 1) as u8 * 0x0f
                + (self.num_ref_idx_active[1] > 1) as u8 * 0xf0;
            self.transform_8x8_mode_flag = self.pps_transform_8x8_mode_flag;
            // C `initialize_context`: clip is the max *index* (active minus 1).
            let max0 = self.num_ref_idx_active[0] - 1;
            let max1 = if self.slice_type == 0 {
                -1
            } else {
                self.num_ref_idx_active[1] - 1
            };
            self.clip_ref_idx = [max0, max0, max0, max0, max1, max1, max1, max1];

            // B slices: temporal prediction and implicit weights
            if self.slice_type == 1 {
                let pic1 = self.ref_pic_list[1][0] as u32;
                self.col_short_term = (self.prev_long_term_frames >> pic1) & 1 == 0;

                // C `(rangeL1 = 1, !direct_spatial_mv_pred_flag)`: the
                // assignment happens whenever the second operand is evaluated.
                let mut range_l1 = self.num_ref_idx_active[1] as i32;
                let do_weights = self.weighted_bipred_idc == 2 || {
                    range_l1 = 1;
                    self.direct_spatial_mv_pred_flag == 0
                };
                if do_weights {
                    // tb.q[i] = sat8(diff_poc[i])
                    let mut tb = [0i8; 32];
                    for (t, d) in tb.iter_mut().zip(self.diff_poc.iter().copied()) {
                        *t = d.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
                    }
                    self.map_pic_to_list0 = [0; 32];
                    let n0 = self.num_ref_idx_active[0] as usize;
                    for ref_idx_l0 in (0..n0).rev() {
                        let p0 = self.ref_pic_list[0][ref_idx_l0] as usize;
                        self.map_pic_to_list0[p0] = ref_idx_l0 as i8;
                        // td.q[i] = sat8(diff_poc[p0] - diff_poc[i])
                        let mut dsf: i32 = 0;
                        for ref_idx_l1 in 0..range_l1 {
                            let p1 = self.ref_pic_list[1][ref_idx_l1 as usize] as usize;
                            let td = (self.diff_poc[p0] - self.diff_poc[p1])
                                .clamp(i8::MIN as i16, i8::MAX as i16)
                                as i32;
                            let iw = if td != 0 && self.prev_long_term_frames & (1 << p0) == 0 {
                                let tx = (16384 + (td / 2).unsigned_abs() as i32) / td;
                                dsf = (((tb[p0] as i32) * tx + 32) >> 6).clamp(-1024, 1023);
                                if self.prev_long_term_frames & (1 << p1) == 0
                                    && (-256..=515).contains(&dsf)
                                {
                                    dsf >> 2
                                } else {
                                    32
                                }
                            } else {
                                dsf = 256;
                                32
                            };
                            // implicit_weights[refIdxL0][refIdxL1] = iw + 64 —
                            // pixel-only (inter MC), not part of the record.
                            self.implicit_weights[ref_idx_l0][ref_idx_l1 as usize] =
                                (iw + 64) as u8;
                        }
                        self.dist_scale_factor[ref_idx_l0] = dsf as i16;
                    }
                }
            }
        }
    }
}

// ---- auto-generated residual / scan tables (verbatim from edge264 C) ----

const CODES_TOTAL_ZEROS: [[u8; 36]; 27] = [
    [
        16, 16, 16, 16, 33, 33, 33, 33, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
        51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
    ],
    [
        16, 16, 16, 16, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
        34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    ],
    [
        16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        16, 16, 16, 16, 49, 49, 50, 50, 67, 67, 68, 68, 69, 69, 69, 69, 86, 86, 86, 86, 87, 87, 87,
        87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87,
    ],
    [
        51, 52, 53, 54, 33, 33, 33, 33, 50, 50, 50, 50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
        48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    ],
    [
        35, 35, 52, 53, 34, 34, 34, 34, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
        48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    ],
    [
        35, 35, 48, 52, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    ],
    [
        34, 34, 35, 35, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    ],
    [
        18, 18, 18, 18, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    ],
    [
        17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        16, 16, 16, 16, 50, 50, 49, 49, 68, 68, 67, 67, 86, 86, 85, 85, 104, 104, 103, 103, 122,
        122, 121, 121, 140, 140, 139, 139, 158, 158, 157, 157, 159, 159, 159, 159,
    ],
    [
        51, 50, 49, 48, 70, 69, 52, 52, 72, 72, 71, 71, 90, 90, 89, 89, 108, 108, 107, 107, 109,
        109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
    ],
    [
        54, 51, 50, 49, 68, 64, 55, 55, 72, 72, 69, 69, 90, 90, 89, 89, 92, 92, 92, 92, 107, 107,
        107, 107, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
    ],
    [
        54, 53, 52, 49, 67, 66, 56, 56, 73, 73, 71, 71, 90, 90, 80, 80, 91, 91, 91, 91, 92, 92, 92,
        92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
    ],
    [
        54, 53, 52, 51, 65, 64, 55, 55, 72, 72, 66, 66, 74, 74, 74, 74, 89, 89, 89, 89, 91, 91, 91,
        91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91,
    ],
    [
        53, 52, 51, 50, 55, 55, 54, 54, 57, 57, 57, 57, 72, 72, 72, 72, 81, 81, 81, 81, 96, 96, 96,
        96, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106,
    ],
    [
        51, 50, 37, 37, 54, 54, 52, 52, 56, 56, 56, 56, 71, 71, 71, 71, 81, 81, 81, 81, 96, 96, 96,
        96, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
    ],
    [
        37, 37, 36, 36, 54, 54, 51, 51, 55, 55, 55, 55, 65, 65, 65, 65, 82, 82, 82, 82, 96, 96, 96,
        96, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104,
    ],
    [
        36, 36, 35, 35, 38, 38, 38, 38, 53, 53, 53, 53, 66, 66, 66, 66, 87, 87, 87, 87, 96, 96, 96,
        96, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97,
    ],
    [
        36, 36, 35, 35, 37, 37, 37, 37, 50, 50, 50, 50, 70, 70, 70, 70, 80, 80, 80, 80, 81, 81, 81,
        81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
    ],
    [
        20, 20, 20, 20, 51, 51, 53, 53, 50, 50, 50, 50, 65, 65, 65, 65, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        19, 19, 19, 19, 34, 34, 34, 34, 52, 52, 52, 52, 65, 65, 65, 65, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        18, 18, 18, 18, 35, 35, 35, 35, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
        48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    ],
    [
        18, 18, 18, 18, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    ],
    [
        17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
];

const RUN_BEFORE_CODES: [[i8; 8]; 7] = [
    [9, 9, 9, 9, 8, 8, 8, 8],
    [18, 18, 17, 17, 8, 8, 8, 8],
    [19, 19, 18, 18, 17, 17, 16, 16],
    [28, 27, 18, 18, 17, 17, 16, 16],
    [29, 28, 27, 26, 17, 17, 16, 16],
    [25, 26, 28, 27, 30, 29, 16, 16],
    [0, 30, 29, 28, 27, 26, 25, 24],
];

const TOKENS_4X4: [i16; 304] = [
    543, 539, 535, 531, 527, 522, 517, 512, 661, 662, 657, 658, 653, 675, 654, 649, 780, 798, 797,
    776, 807, 794, 793, 772, 924, 920, 934, 916, 939, 930, 929, 912, 1075, 1070, 1065, 1060, 1071,
    1066, 1061, 1056, 1200, 1206, 1201, 1196, 1207, 1202, 1197, 1192, 1341, 1336, 1339, 1338, 1337,
    1332, 1205, 1205, 1345, 1345, 1340, 1340, 1343, 1343, 1342, 1342, 1347, 1347, 1347, 1347, 1346,
    1346, 1346, 1346, 1344, 1344, 1344, 1344, 1344, 1344, 1344, 1344, 261, 261, 261, 261, 256, 256,
    256, 256, 531, 531, 527, 527, 394, 394, 394, 394, 795, 782, 781, 772, 663, 663, 649, 649, 799,
    799, 786, 786, 785, 785, 776, 776, 931, 931, 918, 918, 917, 917, 908, 908, 1044, 1044, 1050,
    1050, 1049, 1049, 1040, 1040, 1191, 1191, 1182, 1182, 1181, 1181, 1176, 1176, 1455, 1446, 1445,
    1440, 1451, 1442, 1441, 1436, 1580, 1582, 1581, 1576, 1587, 1578, 1577, 1572, 1723, 1718, 1717,
    1716, 1719, 1714, 1713, 1712, 1853, 1852, 1854, 1849, 1722, 1722, 1720, 1720, 1859, 1859, 1858,
    1858, 1857, 1857, 1856, 1856, 1727, 1727, 1727, 1727, 1727, 1727, 1727, 1727, 128, 128, 128,
    128, 128, 128, 128, 128, 261, 261, 261, 261, 261, 261, 261, 261, 394, 394, 394, 394, 394, 394,
    394, 394, 777, 777, 772, 772, 655, 655, 655, 655, 919, 919, 910, 910, 787, 787, 787, 787, 1051,
    1051, 1042, 1042, 1037, 1037, 1032, 1032, 1183, 1183, 1174, 1174, 1169, 1169, 1164, 1164, 1315,
    1315, 1306, 1306, 1301, 1301, 1296, 1296, 1447, 1447, 1438, 1438, 1433, 1433, 1428, 1428, 1696,
    1702, 1697, 1692, 1707, 1698, 1693, 1688, 1843, 1838, 1833, 1832, 1839, 1834, 1829, 1828, 1979,
    1974, 1969, 1968, 1975, 1970, 1965, 1964, 2115, 2110, 2109, 2104, 2111, 2106, 2105, 2100, 2112,
    2112, 2114, 2114, 2113, 2113, 2108, 2108, 1973, 1973, 1973, 1973, 1973, 1973, 1973, 1973,
];

const NC_OFFSET_4X4: [u8; 8] = [184, 184, 80, 80, 0, 0, 0, 0];

const TOKENS_2X2: [i16; 32] = [
    133, 133, 133, 133, 256, 256, 256, 256, 394, 394, 394, 394, 776, 783, 777, 772, 784, 784, 780,
    780, 910, 910, 909, 909, 1042, 1042, 1041, 1041, 915, 915, 915, 915,
];

const SCAN_4X4: [[i8; 16]; 2] = [
    [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15],
    [0, 1, 4, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];
const SCAN_8X8_CAVLC: [[i8; 64]; 2] = [
    [
        0, 9, 10, 18, 33, 5, 27, 56, 28, 15, 43, 51, 23, 52, 46, 61, 8, 16, 3, 25, 26, 6, 34, 49,
        21, 22, 50, 44, 31, 59, 39, 62, 1, 24, 4, 32, 19, 13, 41, 42, 14, 29, 57, 37, 38, 60, 47,
        55, 2, 17, 11, 40, 12, 20, 48, 35, 7, 36, 58, 30, 45, 53, 54, 63,
    ],
    [
        0, 9, 16, 7, 18, 19, 20, 27, 28, 35, 36, 43, 45, 56, 54, 60, 1, 3, 11, 12, 13, 25, 21, 33,
        29, 41, 37, 49, 46, 57, 55, 61, 2, 4, 5, 17, 14, 32, 22, 40, 30, 48, 38, 50, 47, 52, 58,
        62, 8, 10, 6, 24, 15, 26, 23, 34, 31, 42, 39, 44, 51, 53, 59, 63,
    ],
];
const SCAN_8X8_CABAC: [[i8; 64]; 2] = [
    [
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27,
        34, 41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
        23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
    ],
    [
        0, 1, 2, 8, 9, 3, 4, 10, 16, 11, 5, 6, 7, 12, 17, 24, 18, 13, 14, 15, 19, 25, 32, 26, 20,
        21, 22, 23, 27, 33, 40, 34, 28, 29, 30, 31, 35, 41, 48, 42, 36, 37, 38, 39, 43, 49, 50, 44,
        45, 46, 47, 51, 56, 57, 52, 53, 54, 55, 58, 59, 60, 61, 62, 63,
    ],
];
const INC_8X8: [i8; 12] = [0, 5, 4, 2, 8, 13, 12, 10, 16, 21, 20, 18];
const BIT_8X8: [i8; 12] = [5, 3, 2, 7, 13, 11, 10, 15, 21, 19, 18, 23];

/// C `x444`/`y444` (edge264_internal.h:553-554): 4x4 block origins.
const X444: [i8; 16] = [0, 4, 0, 4, 8, 12, 8, 12, 0, 4, 0, 4, 8, 12, 8, 12];
const Y444: [i8; 16] = [0, 0, 4, 4, 0, 0, 4, 4, 8, 8, 12, 12, 8, 8, 12, 12];
/// C `x420`/`y420` (edge264_internal.h:555-556): chroma 4x4 block origins.
const X420: [i8; 8] = [0, 4, 0, 4, 0, 4, 0, 4];
const Y420: [i8; 8] = [0, 0, 4, 4, 0, 0, 4, 4];

/// C `Intra4x4Modes[predMode][unavail4x4]` (edge264_slice.c:561) mapped to
/// `intra` mode constants.
const INTRA4X4_MODES: [[u32; 16]; 9] = [
    [
        I4X4_V, I4X4_V, I4X4_DC_AB, I4X4_DC_AB, I4X4_V, I4X4_V, I4X4_DC_AB, I4X4_DC_AB, I4X4_V,
        I4X4_V, I4X4_DC_AB, I4X4_DC_AB, I4X4_V, I4X4_V, I4X4_DC_AB, I4X4_DC_AB,
    ],
    [
        I4X4_H, I4X4_DC_AB, I4X4_H, I4X4_DC_AB, I4X4_H, I4X4_DC_AB, I4X4_H, I4X4_DC_AB, I4X4_H,
        I4X4_DC_AB, I4X4_H, I4X4_DC_AB, I4X4_H, I4X4_DC_AB, I4X4_H, I4X4_DC_AB,
    ],
    [
        I4X4_DC, I4X4_DC_A, I4X4_DC_B, I4X4_DC_AB, I4X4_DC, I4X4_DC_A, I4X4_DC_B, I4X4_DC_AB,
        I4X4_DC, I4X4_DC_A, I4X4_DC_B, I4X4_DC_AB, I4X4_DC, I4X4_DC_A, I4X4_DC_B, I4X4_DC_AB,
    ],
    [
        I4X4_DDL, I4X4_DDL, I4X4_DC_AB, I4X4_DC_AB, I4X4_DDL_C, I4X4_DDL_C, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DDL, I4X4_DDL, I4X4_DC_AB, I4X4_DC_AB, I4X4_DDL_C, I4X4_DDL_C, I4X4_DC_AB, I4X4_DC_AB,
    ],
    [
        I4X4_DDR, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DDR, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB,
    ],
    [
        I4X4_VR, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_VR, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB,
    ],
    [
        I4X4_HD, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_HD, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_DC_AB,
    ],
    [
        I4X4_VL, I4X4_VL, I4X4_DC_AB, I4X4_DC_AB, I4X4_VL_C, I4X4_VL_C, I4X4_DC_AB, I4X4_DC_AB,
        I4X4_VL, I4X4_VL, I4X4_DC_AB, I4X4_DC_AB, I4X4_VL_C, I4X4_VL_C, I4X4_DC_AB, I4X4_DC_AB,
    ],
    [
        I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB,
        I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB, I4X4_HU, I4X4_DC_AB,
    ],
];

/// C `Intra8x8Modes[predMode][unavail4x4]` (edge264_slice.c:572).
const INTRA8X8_MODES: [[u32; 16]; 9] = [
    [
        I8X8_V, I8X8_V, I8X8_DC_AB, I8X8_DC_AB, I8X8_V_C, I8X8_V_C, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_V_D, I8X8_V_D, I8X8_DC_AB, I8X8_DC_AB, I8X8_V_CD, I8X8_V_CD, I8X8_DC_AB, I8X8_DC_AB,
    ],
    [
        I8X8_H, I8X8_DC_AB, I8X8_H, I8X8_DC_AB, I8X8_H, I8X8_DC_AB, I8X8_H, I8X8_DC_AB, I8X8_H_D,
        I8X8_DC_AB, I8X8_H_D, I8X8_DC_AB, I8X8_H_D, I8X8_DC_AB, I8X8_H_D, I8X8_DC_AB,
    ],
    [
        I8X8_DC,
        I8X8_DC_A,
        I8X8_DC_B,
        I8X8_DC_AB,
        I8X8_DC_C,
        I8X8_DC_AC,
        I8X8_DC_B,
        I8X8_DC_AB,
        I8X8_DC_D,
        I8X8_DC_AD,
        I8X8_DC_BD,
        I8X8_DC_AB,
        I8X8_DC_CD,
        I8X8_DC_ACD,
        I8X8_DC_BD,
        I8X8_DC_AB,
    ],
    [
        I8X8_DDL,
        I8X8_DDL,
        I8X8_DC_AB,
        I8X8_DC_AB,
        I8X8_DDL_C,
        I8X8_DDL_C,
        I8X8_DC_AB,
        I8X8_DC_AB,
        I8X8_DDL_D,
        I8X8_DDL_D,
        I8X8_DC_AB,
        I8X8_DC_AB,
        I8X8_DDL_CD,
        I8X8_DDL_CD,
        I8X8_DC_AB,
        I8X8_DC_AB,
    ],
    [
        I8X8_DDR, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DDR_C, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB, I8X8_DC_AB,
    ],
    [
        I8X8_VR, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_VR_C, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB,
    ],
    [
        I8X8_HD, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_HD, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_DC_AB,
    ],
    [
        I8X8_VL, I8X8_VL, I8X8_DC_AB, I8X8_DC_AB, I8X8_VL_C, I8X8_VL_C, I8X8_DC_AB, I8X8_DC_AB,
        I8X8_VL_D, I8X8_VL_D, I8X8_DC_AB, I8X8_DC_AB, I8X8_VL_CD, I8X8_VL_CD, I8X8_DC_AB,
        I8X8_DC_AB,
    ],
    [
        I8X8_HU, I8X8_DC_AB, I8X8_HU, I8X8_DC_AB, I8X8_HU, I8X8_DC_AB, I8X8_HU, I8X8_DC_AB,
        I8X8_HU_D, I8X8_DC_AB, I8X8_HU_D, I8X8_DC_AB, I8X8_HU_D, I8X8_DC_AB, I8X8_HU_D, I8X8_DC_AB,
    ],
];

/// C `IntraChromaModes[predMode][unavail4x4 & 3]` (edge264_slice.c:704).
const INTRA_CHROMA_MODES: [[u32; 4]; 4] = [
    [IC8X8_DC, IC8X8_DC_A, IC8X8_DC_B, IC8X8_DC_AB],
    [IC8X8_H, IC8X8_DC_A, IC8X8_H, IC8X8_DC_AB],
    [IC8X8_V, IC8X8_V, IC8X8_DC_B, IC8X8_DC_AB],
    [IC8X8_P, IC8X8_DC_A, IC8X8_DC_B, IC8X8_DC_AB],
];

/// C `Intra16x16Modes[mode][unavail4x4 & 3]` (edge264_slice.c:854).
const INTRA16X16_MODES: [[u32; 4]; 4] = [
    [I16X16_V, I16X16_V, I16X16_DC_B, I16X16_DC_AB],
    [I16X16_H, I16X16_DC_A, I16X16_H, I16X16_DC_AB],
    [I16X16_DC, I16X16_DC_A, I16X16_DC_B, I16X16_DC_AB],
    [I16X16_P, I16X16_DC_A, I16X16_DC_B, I16X16_DC_AB],
];

const SIG_INC_8X8: [[i8; 64]; 2] = [
    [
        0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9,
        8, 7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9,
        11, 12, 13, 11, 14, 10, 12, 0,
    ],
    [
        0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5, 6, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10,
        8, 11, 12, 11, 9, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8, 13, 13, 9, 9, 10, 10, 8, 13,
        13, 9, 9, 10, 10, 14, 14, 14, 14, 14, 0,
    ],
];
const LAST_INC_8X8: [i8; 64] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
];

const CTX_IDX_16X16_DC: [[[i16; 4]; 2]; 3] = [
    [[85, 105, 166, 227], [85, 277, 338, 227]],
    [[460, 484, 572, 952], [460, 776, 864, 952]],
    [[472, 528, 616, 982], [472, 820, 908, 982]],
];
const CTX_IDX_16X16_AC: [[[i16; 4]; 2]; 3] = [
    [[89, 119, 180, 237], [89, 291, 352, 237]],
    [[464, 498, 586, 962], [464, 790, 878, 962]],
    [[476, 542, 630, 992], [476, 834, 922, 992]],
];
const CTX_IDX_CHROMA_DC: [[i16; 4]; 2] = [[97, 149, 210, 257], [97, 321, 382, 257]];
const CTX_IDX_CHROMA_AC: [[i16; 4]; 2] = [[101, 151, 212, 266], [101, 323, 384, 266]];
const CTX_IDX_4X4: [[[i16; 4]; 2]; 3] = [
    [[93, 134, 195, 247], [93, 306, 367, 247]],
    [[468, 528, 616, 972], [468, 805, 893, 972]],
    [[480, 557, 645, 1002], [480, 849, 937, 1002]],
];
const CTX_IDX_8X8: [[[i16; 4]; 2]; 3] = [
    [[1012, 402, 417, 426], [1012, 436, 451, 426]],
    [[1016, 660, 690, 708], [1016, 675, 699, 708]],
    [[1020, 718, 748, 766], [1020, 733, 757, 766]],
];

const QP_Y2C: [i8; 88] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36,
    36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
];

// ====================================================================
// Residual block parsing (9.2 CAVLC / 9.3 CABAC)
//
// Port of the `#if !CABAC` / `#if CABAC` halves of edge264_slice.c. The MB
// parser sets `scan` and zeroes `c` before each block call; the bit-cache
// mutation must stay in lockstep with C (msb/lsb shld + delayed refills).
// ====================================================================

impl<'a> SliceContext<'a> {
    /// C `parse_total_zeros(ctx, endIdx, TotalCoeff)` — total_zeros code.
    #[inline]
    fn parse_total_zeros(&mut self, end_idx: i32, total_coeff: i32) -> i32 {
        let msb = self.bits.msb;
        let leading_zero_bits = (msb | (1u64 << 55)).leading_zeros() as i32; // SIZE_BIT-9
        let suffix = ((msb >> ((leading_zero_bits + 2) ^ 63)) & 3) as i32;
        let code = CODES_TOTAL_ZEROS[(end_idx + total_coeff - 4) as usize]
            [(leading_zero_bits * 4 + suffix) as usize];
        let v = (code >> 4) as u32;
        self.bits.msb = shld(self.bits.lsb, msb, v);
        self.bits.lsb <<= v;
        (code & 15) as i32
    }

    /// C `parse_residual_coeffs_cavlc(ctx, startIdx, endIdx, TrailingOnes,
    /// TotalCoeff)`.
    fn parse_residual_coeffs_cavlc(
        &mut self,
        start_idx: i32,
        end_idx: i32,
        trailing_ones: i32,
        total_coeff: i32,
    ) {
        // parse all level values from end to start
        let mut level = [0i32; 16];
        let signs = !self.bits.msb;
        level[0] = ((signs >> 62) & 2) as i32 - 1;
        level[1] = ((signs >> 61) & 2) as i32 - 1;
        level[2] = ((signs >> 60) & 2) as i32 - 1;
        self.bits.msb = shld(self.bits.lsb, self.bits.msb, trailing_ones as u32);
        self.bits.lsb <<= trailing_ones as u32;
        let mut suffix_length = 1i32;
        for i in trailing_ones..total_coeff {
            let level_prefix = (self.bits.msb | (1u64 << 38)).leading_zeros() as i32; // limit given in 9.2.2.1
            let (v, offset) = if level_prefix >= 15 {
                (level_prefix * 2 - 2, (15 << suffix_length) - 4096)
            } else if i > trailing_ones || (total_coeff > 10 && trailing_ones < 3) {
                (
                    level_prefix + suffix_length + 1,
                    (level_prefix - 1) << suffix_length,
                )
            } else if level_prefix < 14 {
                (level_prefix + 1, level_prefix - 1)
            } else {
                (19, -2)
            };
            // C: int levelCode = get_uv(gb, v) + offset — 32-bit wrap on both
            // sides (get_uv returns C `unsigned`), so truncate + wrapping add.
            let mut level_code = self.bits.get_uv(v as usize) as i32;
            level_code = level_code.wrapping_add(offset);
            level_code += (i == trailing_ones && trailing_ones < 3) as i32 * 2;
            level[i as usize] = if level_code & 1 != 0 {
                (-level_code - 1) / 2
            } else {
                (level_code + 2) / 2
            };
            suffix_length = (suffix_length + (level_code >= (3 << suffix_length)) as i32).min(6);
        }

        // store level values at proper positions in memory
        let mut zeros_left = 0i32;
        if total_coeff <= end_idx - start_idx {
            zeros_left = self.parse_total_zeros(end_idx, total_coeff);
        }
        let mut scan = start_idx + zeros_left + total_coeff - 1;
        self.c[self.scan[scan as usize] as usize] = level[0];
        for i in 1..total_coeff {
            scan -= 1;
            if zeros_left > 0 {
                let three_bits = (self.bits.msb >> 61) as u32;
                let (v, run_before) = if zeros_left <= 6 || three_bits > 0 {
                    let code =
                        RUN_BEFORE_CODES[(zeros_left.min(7) - 1) as usize][three_bits as usize];
                    ((code >> 3) as u32, (code & 7) as i32)
                } else {
                    // C `clz(msb) + 1`: clz(0) = 64 on x86 (verified vs oracle
                    // compiler), so v can be 65 when the cache is exhausted.
                    let v = self.bits.msb.leading_zeros() as i32 + 1;
                    (v as u32, (v + 3).min(zeros_left))
                };
                scan -= run_before;
                zeros_left -= run_before;
                self.bits.msb = shld(self.bits.lsb, self.bits.msb, v);
                self.bits.lsb <<= v & 63; // x86 masks shift counts to 6 bits
            }
            self.c[self.scan[scan as usize] as usize] = level[i as usize];
        }

        // trailing_ones_sign_flags+total_zeros+run_before consumed at most 31
        // bits, so we can delay refill here
        if self.bits.lsb == 0 {
            self.bits.refill();
        }
    }

    /// C `parse_residual_block_4x4_cavlc(ctx, startIdx, i4x4, nA, nB)`.
    /// Returns true if the block has coded coefficients (C returns 1/0).
    fn parse_residual_block_4x4_cavlc(
        &mut self,
        start_idx: i32,
        i4x4: i32,
        n_a: i32,
        n_b: i32,
    ) -> bool {
        let sum = n_a + n_b;
        let n_c = if self.unavail4x4[i4x4 as usize] & 3 == 0 {
            (sum + 1) >> 1
        } else {
            sum
        };
        let (coeff_token, v) = if n_c < 8 {
            let msb = self.bits.msb;
            let leading_zero_bits = (msb | (1u64 << 49)).leading_zeros() as i32; // SIZE_BIT-15
            let suffix = ((msb >> (60 - leading_zero_bits)) & 7) as i32; // SIZE_BIT-4-lz
            let token = TOKENS_4X4
                [NC_OFFSET_4X4[n_c as usize] as usize + (leading_zero_bits * 8 + suffix) as usize];
            ((token & 127) as i32, (token >> 7) as u32)
        } else {
            let mut coeff_token = (self.bits.msb >> 58) as i32 + 4; // SIZE_BIT-6
            if coeff_token == 7 {
                coeff_token = 0;
            }
            (coeff_token, 6)
        };
        self.bits.msb = shld(self.bits.lsb, self.bits.msb, v);
        self.bits.lsb <<= v;
        if self.bits.lsb == 0 {
            self.bits.refill();
        }
        if coeff_token != 0 {
            let mb = self.mb();
            unsafe {
                (*mb).nC[i4x4 as usize] = (coeff_token >> 2) as i8;
            }
            self.parse_residual_coeffs_cavlc(start_idx, 15, coeff_token & 3, coeff_token >> 2);
            true
        } else {
            false
        }
    }

    /// C `parse_residual_block_2x2_cavlc(ctx)` — 4:2:0 chroma.
    fn parse_residual_block_2x2_cavlc(&mut self) {
        let msb = self.bits.msb;
        let leading_zero_bits = (msb | (1u64 << 56)).leading_zeros() as i32; // SIZE_BIT-8
        let suffix = ((msb >> (61 - leading_zero_bits)) & 3) as i32; // SIZE_BIT-3-lz
        let token = TOKENS_2X2[(leading_zero_bits * 4 + suffix) as usize];
        let coeff_token = (token & 127) as i32;
        let v = (token >> 7) as u32;
        self.bits.msb = shld(self.bits.lsb, msb, v);
        self.bits.lsb <<= v;
        if self.bits.lsb == 0 {
            self.bits.refill();
        }
        if coeff_token != 0 {
            self.parse_residual_coeffs_cavlc(0, 3, coeff_token & 3, coeff_token >> 2);
        }
    }

    /// C `parse_residual_coeffs_cabac(ctx, significant_coeff_flags)`.
    fn parse_residual_coeffs_cabac(&mut self, mut significant_coeff_flags: u64) {
        let off = self.ctx_idx_offsets[3] as usize;
        let mut ctx_idx0 = off + 1;
        let mut ctx_idx1 = off + 5;
        while significant_coeff_flags != 0 {
            let coeff_level;
            if self.cabac.get_ae(&mut self.bits, ctx_idx0) == 0 {
                const TRANS: [i8; 5] = [0, 2, 3, 4, 4];
                ctx_idx0 = off + TRANS[ctx_idx0 - off] as usize;
                coeff_level = if self.cabac.get_bypass(&mut self.bits) != 0 {
                    -1
                } else {
                    1
                };
            } else {
                let mut cl: i32 = 2;
                loop {
                    if self.cabac.get_ae(&mut self.bits, ctx_idx1) == 0 {
                        break;
                    }
                    cl += 1;
                    if cl == 15 {
                        break;
                    }
                }
                coeff_level = if cl != 15 {
                    if self.cabac.get_bypass(&mut self.bits) != 0 {
                        -cl
                    } else {
                        cl
                    }
                } else {
                    // we need at least 51 bits in offset to get 42 bits with a
                    // division by 9 bits
                    let zeros = self.bits.lsb.leading_zeros();
                    if zeros > 64 - 51 {
                        self.cabac.renorm_bits(&mut self.bits, zeros as i32);
                    }
                    let range = self.bits.lsb >> 42;
                    let quo = self.bits.msb / range; // contains 42 bypass bits in lsb
                    let rem = self.bits.msb % range;
                    let k = ((!quo) << 22 | (1u64 << 43)).leading_zeros() as i32;
                    let unused = 42 - k * 2 - 2;
                    let mut lv =
                        14 + (1 << k) + (((quo >> unused >> 1) as u32) & ((1u32 << k) - 1)) as i32;
                    if quo & (1u64 << unused) != 0 {
                        lv = -lv;
                    }
                    self.bits.msb = (quo & ((1u64 << unused) - 1)) * range + rem;
                    self.bits.lsb = range << unused;
                    lv
                };
                ctx_idx0 = off;
                ctx_idx1 = ctx_idx0 + self.coeff_abs_inc[ctx_idx1 - off - 5] as usize;
            }

            // store in transposed scan position
            let i = 63 - significant_coeff_flags.leading_zeros() as i32;
            self.c[self.scan[i as usize] as usize] = coeff_level;
            significant_coeff_flags &= !((1u64) << i);
        }
    }

    /// C `parse_residual_block_8x8_cabac(ctx, startIdx, endIdx)`.
    fn parse_residual_block_8x8_cabac(&mut self, start_idx: i32, end_idx: i32) {
        let mut significant_coeff_flags: u64 = 0;
        let mut i = start_idx;
        'block: loop {
            if self.cabac.get_ae(
                &mut self.bits,
                (self.ctx_idx_offsets[1] + self.sig_inc[i as usize] as i16) as usize,
            ) != 0
            {
                significant_coeff_flags |= 1u64 << i as u32;
                if self.cabac.get_ae(
                    &mut self.bits,
                    (self.ctx_idx_offsets[2] + self.last_inc[i as usize] as i16) as usize,
                ) != 0
                {
                    break 'block;
                }
            }
            i += 1;
            if i >= end_idx {
                break 'block;
            }
        }
        significant_coeff_flags |= 1u64 << i as u32;
        self.parse_residual_coeffs_cabac(significant_coeff_flags);
    }

    /// C `parse_residual_block_cabac(ctx, startIdx, endIdx)` — 4x4 / 2x2.
    fn parse_residual_block_cabac(&mut self, start_idx: i32, end_idx: i32) {
        let mut significant_coeff_flags: u32 = 0;
        let mut i = start_idx;
        'block: loop {
            if self.cabac.get_ae(
                &mut self.bits,
                (self.ctx_idx_offsets[1] + i as i16) as usize,
            ) != 0
            {
                significant_coeff_flags |= 1u32 << i as u32;
                if self.cabac.get_ae(
                    &mut self.bits,
                    (self.ctx_idx_offsets[2] + i as i16) as usize,
                ) != 0
                {
                    break 'block;
                }
            }
            i += 1;
            if i >= end_idx {
                break 'block;
            }
        }
        significant_coeff_flags |= 1u32 << i as u32;
        self.parse_residual_coeffs_cabac(significant_coeff_flags as u64);
    }
}

// ---- mvpred bridging + P-slice path (Tier D) --------------------------------

use super::mvpred;

/// Zeroed neighbour used when a spatial neighbour is unavailable (C `unavail_mb`).
pub(crate) const UNAVAIL_MB: RustMb = RustMb {
    error_probability: 0,
    recovery_bits: 0,
    mbIsInterFlag: 0,
    filter_edges: 0,
    QP: [0; 4],
    bits: [0xac, 0],
    Intra4x4PredMode: [-2; 16],
    nC: [0; 48],
    absMvd: [0; 64],
    f: RustMbFlags {
        mb_field_decoding_flag: 0,
        mb_skip_flag: 1,
        mb_type_I_NxN: 1,
        mb_type_B_Direct: 1,
        transform_size_8x8_flag: 0,
        intra_chroma_pred_mode_non_zero: 0,
        CodedBlockPatternChromaDC: 0,
        CodedBlockPatternChromaAC: 0,
        coded_block_flags_16x16: [0; 4],
        inter_eqs: [0; 4],
    },
    refIdx: [-1; 8],
    refPic: [0; 8],
    mvs: [0; 64],
};

// -- scalar replications of the SSE primitives used by parse_P_sub_mb ---------

#[inline]
fn pack_mv(x: i16, y: i16) -> i32 {
    (x as u16 as i32) | ((y as u16 as i32) << 16)
}
#[inline]
fn pair_x(p: i32) -> i16 {
    p as i16
}
#[inline]
fn pair_y(p: i32) -> i16 {
    (p >> 16) as i16
}
/// `mvp + mvd` with per-component i16 wrapping (C `i16x8 mv = mvp + mvd`).
#[inline]
fn add_pair_i32(a: i32, b: i32) -> i32 {
    pack_mv(
        (pair_x(a) as i32 + pair_x(b) as i32) as i16,
        (pair_y(a) as i32 + pair_y(b) as i32) as i16,
    )
}
/// component-wise median of three packed pairs (C `median16`).
#[inline]
fn median_pair_i32(a: i32, b: i32, c: i32) -> i32 {
    let med = |x: i16, y: i16, z: i16| x.max(y).min(z).max(x.min(y));
    pack_mv(
        med(pair_x(a), pair_x(b), pair_x(c)),
        med(pair_y(a), pair_y(b), pair_y(c)),
    )
}
/// `shufflen` on one byte (verified SSE semantics): m<0 -> 0xFF else a[m&15].
#[inline]
fn shufflen_byte(a: &[i8; 16], m: i8) -> i8 {
    if m < 0 { -1 } else { a[m as usize & 15] }
}
/// `_mm_shuffle_epi8(a, m)`: r[i] = m[i]<0 ? 0 : a[m[i]&15].
fn shuffle_i8x16(a: &[i8; 16], m: &[i8; 16]) -> [i8; 16] {
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = if m[i] < 0 { 0 } else { a[m[i] as usize & 15] };
    }
    r
}
/// `unziplo32(a,b)` (SSE `_mm_shuffle_ps`, NEON `vuzp1q_s32`, WASM shuffle 0,2,4,6):
/// i32-lane interleave [a0, a2, b0, b2] = [a[0..4], a[8..12], b[0..4], b[8..12]].
fn unziplo32_bytes(a: &[i8; 16], b: &[i8; 16]) -> [i8; 16] {
    let mut r = [0i8; 16];
    r[0..4].copy_from_slice(&a[0..4]);
    r[4..8].copy_from_slice(&a[8..12]);
    r[8..12].copy_from_slice(&b[0..4]);
    r[12..16].copy_from_slice(&b[8..12]);
    r
}
/// `unziphi32(a,b)` (NEON `vuzp2q_s32`, WASM shuffle 1,3,5,7):
/// [a1, a3, b1, b3] = [a[4..8], a[12..16], b[4..8], b[12..16]].
fn unziphi32_bytes(a: &[i8; 16], b: &[i8; 16]) -> [i8; 16] {
    let mut r = [0i8; 16];
    r[0..4].copy_from_slice(&a[4..8]);
    r[4..8].copy_from_slice(&a[12..16]);
    r[8..12].copy_from_slice(&b[4..8]);
    r[12..16].copy_from_slice(&b[12..16]);
    r
}
/// `_pext_u32(f, 0x11111111)`: pack f's bit 0 of each byte into bits 0..7.
#[inline]
/// C `mvd_flags2ref_idx` = `_pext_u32(f, 0x11111111)`: extract the first-slot
/// bit of each 4x4 group (bits 0,4,8,...,28) and pack them into bits 0..7.
/// Bit i selects refIdx slot i ([LX][i8x8]: L0 TL/TR/BL/BR then L1).
fn mvd_flags2ref_idx(f: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        if f >> (4 * i) & 1 != 0 {
            r |= 1 << i;
        }
    }
    r
}
/// `_mm_unpacklo_epi16(x, x)`: [x0,x0,x1,x1,x2,x2,x3,x3].
#[inline]
fn ziplo16_self(x: &[i16; 8]) -> [i16; 8] {
    [x[0], x[0], x[1], x[1], x[2], x[2], x[3], x[3]]
}
/// `pack_absMvd(mvd)` as i8x16: [sat8abs(x), sat8abs(y)] repeated 8 times.
fn pack_absmvd_bytes(mvd: i32) -> [i8; 16] {
    let ax = pair_x(mvd).clamp(-128, 127).unsigned_abs() as i8;
    let ay = pair_y(mvd).clamp(-128, 127).unsigned_abs() as i8;
    let mut r = [0i8; 16];
    for j in 0..8 {
        r[2 * j] = ax;
        r[2 * j + 1] = ay;
    }
    r
}

/// C `ctx->inc.v = mbA->f.v + mbB->f.v + (mbB->f.v & flags_twice.v)` —
/// per-byte i8 wrap-add (SIMD lane semantics).
#[inline]
fn flags_inc(a: RustMbFlags, b: RustMbFlags) -> RustMbFlags {
    let t = RustMbFlags::twice();
    RustMbFlags {
        mb_field_decoding_flag: a
            .mb_field_decoding_flag
            .wrapping_add(b.mb_field_decoding_flag),
        mb_skip_flag: a.mb_skip_flag.wrapping_add(b.mb_skip_flag),
        mb_type_I_NxN: a.mb_type_I_NxN.wrapping_add(b.mb_type_I_NxN),
        mb_type_B_Direct: a.mb_type_B_Direct.wrapping_add(b.mb_type_B_Direct),
        transform_size_8x8_flag: a
            .transform_size_8x8_flag
            .wrapping_add(b.transform_size_8x8_flag),
        intra_chroma_pred_mode_non_zero: a
            .intra_chroma_pred_mode_non_zero
            .wrapping_add(b.intra_chroma_pred_mode_non_zero),
        CodedBlockPatternChromaDC: a
            .CodedBlockPatternChromaDC
            .wrapping_add(b.CodedBlockPatternChromaDC)
            .wrapping_add(b.CodedBlockPatternChromaDC & t.CodedBlockPatternChromaDC),
        CodedBlockPatternChromaAC: a
            .CodedBlockPatternChromaAC
            .wrapping_add(b.CodedBlockPatternChromaAC)
            .wrapping_add(b.CodedBlockPatternChromaAC & t.CodedBlockPatternChromaAC),
        coded_block_flags_16x16: [
            (a.coded_block_flags_16x16[0] as i8)
                .wrapping_add(b.coded_block_flags_16x16[0] as i8)
                .wrapping_add(
                    (b.coded_block_flags_16x16[0] as i8) & t.coded_block_flags_16x16[0] as i8,
                ) as u8,
            (a.coded_block_flags_16x16[1] as i8)
                .wrapping_add(b.coded_block_flags_16x16[1] as i8)
                .wrapping_add(
                    (b.coded_block_flags_16x16[1] as i8) & t.coded_block_flags_16x16[1] as i8,
                ) as u8,
            (a.coded_block_flags_16x16[2] as i8)
                .wrapping_add(b.coded_block_flags_16x16[2] as i8)
                .wrapping_add(
                    (b.coded_block_flags_16x16[2] as i8) & t.coded_block_flags_16x16[2] as i8,
                ) as u8,
            a.coded_block_flags_16x16[3].wrapping_add(b.coded_block_flags_16x16[3]),
        ],
        inter_eqs: [
            a.inter_eqs[0].wrapping_add(b.inter_eqs[0]),
            a.inter_eqs[1].wrapping_add(b.inter_eqs[1]),
            a.inter_eqs[2].wrapping_add(b.inter_eqs[2]),
            a.inter_eqs[3].wrapping_add(b.inter_eqs[3]),
        ],
    }
}

#[inline]
fn nb_from(m: &RustMb) -> mvpred::Nb {
    let mut mvs_s = [0i32; 32];
    for (o, p) in mvs_s.iter_mut().zip(m.mvs.chunks_exact(2)) {
        *o = pack_mv(p[0], p[1]);
    }
    mvpred::Nb {
        ref_idx: m.refIdx,
        mvs_s,
    }
}
#[inline]
fn mb_from(m: &RustMb) -> mvpred::Mb {
    mvpred::Mb {
        ref_idx: m.refIdx,
        ref_pic: m.refPic,
        mvs: m.mvs,
        abs_mvd: m.absMvd,
        inter_eqs: u32::from_le_bytes(m.f.inter_eqs),
    }
}
/// Write back the macroblock fields a mvpred op is responsible for (the op starts
/// from a copy of the current MB and only mutates these five).
#[inline]
fn mb_apply(m: &mut RustMb, out: &mvpred::Mb) {
    m.refIdx = out.ref_idx;
    m.refPic = out.ref_pic;
    m.mvs = out.mvs;
    m.absMvd = out.abs_mvd;
    m.f.inter_eqs = out.inter_eqs.to_le_bytes();
}

impl<'a> SliceContext<'a> {
    /// Build the mvpred input state from the live parse context (neighbours are
    /// set per-MB in `parse_slice_data`).
    fn mvpred_state(&self) -> mvpred::State {
        let mb = unsafe { &*self.mb() };
        // C leaves edge-adjacent neighbour pointers as raw (before the buffer
        // start) when the decoded counter marks them "available" but unavail4x4
        // marks them unavailable; its mvpred ops guard on unavail4x4 and never
        // read them. We dereference all four eagerly, so substitute UNAVAIL_MB
        // (zeroed data) for any pointer that falls before the buffer — the only
        // OOB direction, since every neighbour sits at a lower index than the
        // current MB. This matches C's effective behaviour (unavailable == 0).
        let base = self.mb_buffer as usize;
        let nb_ref = |p: *const RustMb| -> &RustMb {
            if (p as usize) >= base {
                unsafe { &*p }
            } else {
                &UNAVAIL_MB
            }
        };
        let a = nb_ref(self.mb_a);
        let b = nb_ref(self.mb_b);
        let c = nb_ref(self.mb_c);
        let d = nb_ref(self.mb_d);
        let col = if self.mb_col.is_null() {
            mvpred::Col {
                ref_idx_s: [0; 2],
                ref_pic_s: [0; 2],
                mvs: [0; 64],
                inter_eqs: 0,
            }
        } else {
            let p = unsafe { &*self.mb_col };
            // C reads mbCol->refIdx_s/refPic_s as packed little-endian byte
            // quads ([LX][i8x8]); replicate the exact 4-byte packing. Bytes
            // must be zero-extended (a plain `i8 as u32` sign-extends and
            // poisons the higher bytes after the shift).
            let pack4 = |v: &[i8]| {
                (((v[0] as u8) as u32)
                    | (((v[1] as u8) as u32) << 8)
                    | (((v[2] as u8) as u32) << 16)
                    | (((v[3] as u8) as u32) << 24)) as i32
            };
            mvpred::Col {
                ref_idx_s: [pack4(&p.refIdx[0..4]), pack4(&p.refIdx[4..8])],
                ref_pic_s: [pack4(&p.refPic[0..4]), pack4(&p.refPic[4..8])],
                mvs: p.mvs,
                inter_eqs: u32::from_le_bytes(p.f.inter_eqs),
            }
        };
        mvpred::State {
            mb: mb_from(mb),
            mb_a: nb_from(a),
            mb_b: nb_from(b),
            mb_c: nb_from(c),
            mb_d: nb_from(d),
            mb_col: col,
            unavail4x4: self.unavail4x4,
            ref_pic_list0: self.ref_pic_list[0],
            ref_pic_list1: self.ref_pic_list[1],
            map_pic_to_list0: self.map_pic_to_list0,
            dist_scale_factor: self.dist_scale_factor,
            col_short_term: self.col_short_term,
            direct_8x8_inference_flag: self.direct_8x8_inference_flag != 0,
        }
    }

    /// C `parse_mvd_pair` — returns the packed (x | y<<16) MVD. The caller (a
    /// mvpred op or `parse_*_sub_mb`) writes absMvd via pack_absMvd; this only
    /// consumes entropy. `absmvd_off` is the byte offset of `absMvd_lx` relative
    /// to the MB's absMvd start (0 for L0, 32 for L1 in B slices).
    fn parse_mvd_pair(&mut self, i4x4: usize, absmvd_off: isize) -> i32 {
        if !self.is_cabac {
            let x = self.bits.get_se16(-32768, 32767);
            let y = self.bits.get_se16(-32768, 32767);
            pack_mv(x as i16, y as i16)
        } else {
            let base = unsafe { (self.mb() as *const u8).offset(80 + absmvd_off) }; // absMvd @ offset 80
            let mut first: i32 = 0;
            let mut ctx_base: i32 = 40;
            for i in 0..2 {
                let a = unsafe { *base.offset(self.absmvd_a[i4x4] as isize + i as isize) } as i32;
                let b = unsafe { *base.offset(self.absmvd_b[i4x4] as isize + i as isize) } as i32;
                let sum = a + b;
                let mut ctx_idx = (ctx_base + (sum >= 3) as i32 + (sum > 32) as i32) as usize;
                let mut mvd: i32 = 0;
                ctx_base += 3;
                while mvd != 9 {
                    if self.cabac.get_ae(&mut self.bits, ctx_idx) == 0 {
                        break;
                    }
                    ctx_idx = (ctx_base + mvd.min(3)) as usize;
                    mvd += 1;
                }
                if mvd == 9 {
                    let zeros = self.bits.lsb.leading_zeros() as i32;
                    if zeros > 64 - 36 {
                        // C reassigns zeros but never reads it (renorm side effect only)
                        let _ = self.cabac.renorm_bits(&mut self.bits, zeros);
                    }
                    let range = self.bits.lsb >> 27;
                    let quo = self.bits.msb / range;
                    let rem = self.bits.msb % range;
                    let k = 3 + ((!quo << 37) | (1u64 << 52)).leading_zeros() as i32;
                    let unused = 27 - k * 2 + 1;
                    mvd = 1 + (1 << k) + (((quo >> unused) >> 1) & ((1u64 << k) - 1)) as i32;
                    if (quo & (1u64 << unused)) != 0 {
                        mvd = -mvd;
                    }
                    self.bits.msb = (quo & ((1u64 << unused) - 1)) * range + rem;
                    self.bits.lsb = range << unused;
                } else if mvd > 0 && self.cabac.get_bypass(&mut self.bits) != 0 {
                    mvd = -mvd;
                }
                if i == 1 {
                    return pack_mv(first as i16, mvd as i16);
                }
                first = mvd;
                ctx_base = 47;
            }
            unreachable!()
        }
    }

    /// C `parse_ref_idx` (9.3.3.1.1.6): parse the ref_idx symbols flagged by `f`,
    /// clip, broadcast to the inferred block shapes and compute refPic.
    fn parse_ref_idx(&mut self, f: u32) {
        let m = unsafe { &mut *self.mb() };
        let inc8x8 = [0i8, 5, 4, 2, 8, 13, 12, 10, 16, 21, 20, 18];
        let bit8x8 = [5i8, 3, 2, 7, 13, 11, 10, 15, 21, 19, 18, 23];
        // "set to 0 if parsed"
        let mut ri: [i8; 8] = m.refIdx;
        for (i, r) in ri.iter_mut().enumerate() {
            if f & (1 << i) != 0 {
                *r = 0;
            }
        }
        let mut u = f & self.num_ref_idx_mask as u32;
        while u != 0 {
            let i = u.trailing_zeros() as usize;
            let ref_idx: i8 = if !self.is_cabac {
                if self.clip_ref_idx[i] == 1 {
                    (self.bits.get_u1() ^ 1) as i8
                } else {
                    self.bits.get_ue16(self.clip_ref_idx[i] as u32) as i8
                }
            } else {
                let mut ref_idx: i8 = 0;
                if self.cabac.get_ae(
                    &mut self.bits,
                    54 + ((m.bits[0] >> inc8x8[4 + i]) & 3) as usize,
                ) != 0
                {
                    ref_idx = 1;
                    m.bits[0] |= 1 << bit8x8[4 + i];
                    let mut ctx_idx = 58usize;
                    while (ref_idx as u32) < 32 && self.cabac.get_ae(&mut self.bits, ctx_idx) != 0 {
                        ref_idx += 1;
                        ctx_idx = 59;
                    }
                }
                ref_idx
            };
            ri[i] = ref_idx;
            u &= u - 1;
        }
        // clip (CABAC only)
        if self.is_cabac {
            for (r, c) in ri.iter_mut().zip(self.clip_ref_idx.iter()) {
                *r = (*r).min(*c);
            }
        }
        // broadcast to block shapes
        if f & 0x122 == 0 {
            let idx = [0i8, 0, 2, 2, 4, 4, 6, 6];
            for k in 0..8 {
                ri[k] = ri[idx[k] as usize];
            }
            if self.is_cabac {
                m.bits[0] |= (m.bits[0] >> 2 & 0x080800) | (m.bits[0] << 5 & 0x808000);
            }
        }
        if f & 0x144 == 0 {
            let idx = [0i8, 1, 0, 1, 4, 5, 4, 5];
            for k in 0..8 {
                ri[k] = ri[idx[k] as usize];
            }
            if self.is_cabac {
                m.bits[0] |= (m.bits[0] >> 3 & 0x040400) | (m.bits[0] << 4 & 0x808000);
            }
        }
        m.refIdx = ri;
        // compute reference picture numbers
        let rpl0: [i8; 16] = self.ref_pic_list[0][..16].try_into().unwrap();
        let rpl1: [i8; 16] = self.ref_pic_list[1][..16].try_into().unwrap();
        for (p, r) in m.refPic[..4].iter_mut().zip(ri[..4].iter()) {
            *p = shufflen_byte(&rpl0, *r);
        }
        for (p, r) in m.refPic[4..].iter_mut().zip(ri[4..].iter()) {
            *p = shufflen_byte(&rpl1, *r);
        }
    }

    /// C `parse_P_mb` (mb_skip_flag + mb_type + large-block decode).
    fn parse_P_mb(&mut self) {
        let m = unsafe { &mut *self.mb() };
        // Inter initializations
        m.mbIsInterFlag = 1;
        for i in 0..16 {
            m.Intra4x4PredMode[i] = 2;
        }
        m.refIdx = [0, 0, 0, 0, -1, -1, -1, -1];
        if self.is_cabac {
            // C clears only the L0 half (`absMvd_v[0] = absMvd_v[1] = {}`);
            // the L1 half keeps stale bytes from the previous frame that used
            // this DPB slot. Mirror exactly — the record must be bit-exact.
            for i in 0..32 {
                m.absMvd[i] = 0;
            }
        }

        // parse mb_skip_run / flag
        let mb_skip_flag: u32 = if !self.is_cabac {
            if self.mb_skip_run < 0 {
                self.mb_skip_run = self.bits.get_ue16(139264) as i32;
            }
            // C: `mb_skip_run-- > 0` — post-decrement, test the OLD value.
            let flag = self.mb_skip_run > 0;
            self.mb_skip_run -= 1;
            flag as u32
        } else {
            self.cabac
                .get_ae(&mut self.bits, 13 - self.inc.mb_skip_flag as usize)
        };

        // earliest handling for P_Skip
        if mb_skip_flag != 0 {
            if self.is_cabac {
                m.f.mb_skip_flag = 1;
                self.mb_qp_delta_nz = 0;
            }
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::p_skip(&state));
            // C decode_P_skip ends with decode_inter(ctx, 0, 16, 16).
            self.decode_inter(0, 16, 16);
            return;
        }
        self.transform_8x8_mode_flag = self.pps_transform_8x8_mode_flag;

        // parse mb_type
        let mb_type: i32 = if !self.is_cabac {
            let mt = self.bits.get_ue16(30) as i32;
            if mt > 4 {
                self.parse_I_mb(mt - 5);
                return;
            }
            mt
        } else {
            if self.cabac.get_ae(&mut self.bits, 14) != 0 {
                self.parse_I_mb(17);
                return;
            }
            let mut mt = self.cabac.get_ae(&mut self.bits, 15) as i32;
            mt = mt + mt + self.cabac.get_ae(&mut self.bits, (16 + mt) as usize) as i32;
            mt
        };

        // mvs_v[4..8] = {} (L1 unused in P)
        for i in 32..64 {
            m.mvs[i] = 0;
        }

        if !self.is_cabac {
            if mb_type > 2 {
                self.parse_P_sub_mb(((mb_type + 12) & 15) as u32);
                return;
            }
            self.parse_ref_idx((0x351 >> (mb_type << 2)) & 15);
        } else {
            if mb_type == 1 {
                self.parse_P_sub_mb(15);
                return;
            }
            self.parse_ref_idx(((mb_type + 1) | 1) as u32);
        }

        // decoding large blocks (each C decode_inter_*X* ends with its own
        // decode_inter call, interleaved with the next MVD parse)
        if mb_type == 0 {
            m.f.inter_eqs = 0x1b5fbbffu32.to_le_bytes();
            let mvd = self.parse_mvd_pair(0, 0);
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::inter_16x16(&state, mvd, 0));
            self.decode_inter(0, 16, 16);
        } else if mb_type == 2 {
            m.f.inter_eqs = 0x1b1bbbbb_u32.to_le_bytes();
            let mvd0 = self.parse_mvd_pair(0, 0);
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::inter_8x16_left(&state, mvd0, 0));
            self.decode_inter(0, 8, 16);
            let mvd1 = self.parse_mvd_pair(4, 0);
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::inter_8x16_right(&state, mvd1, 0));
            self.decode_inter(4, 8, 16);
        } else {
            m.f.inter_eqs = 0x1b5f1b5f_u32.to_le_bytes();
            let mvd0 = self.parse_mvd_pair(0, 0);
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::inter_16x8_top(&state, mvd0, 0));
            self.decode_inter(0, 16, 8);
            let mvd1 = self.parse_mvd_pair(8, 0);
            let state = self.mvpred_state();
            mb_apply(m, &mvpred::inter_16x8_bottom(&state, mvd1, 0));
            self.decode_inter(8, 16, 8);
        }
        self.parse_inter_residual();
    }

    /// C `parse_P_sub_mb` (sub_mb_type + refIdx shuffle + per-block MVD/MVP).
    /// Faithful scalar transcription of the SSE mask/shuffle logic. NOTE: the
    /// test samples contain no sub-MB partitions, so this path is not exercised
    /// by the differential oracle; it needs synthetic-vector validation.
    fn parse_P_sub_mb(&mut self, ref_idx_flags: u32) {
        let m = unsafe { &mut *self.mb() };
        for i in 0..32 {
            m.mvs[i] = 0; // mvs_v[0..3] = {}
        }
        m.f.inter_eqs = [0; 4];
        let mut mvd_flags: u32 = 0;
        for i8x8 in 0..4 {
            let i4x4 = i8x8 * 4;
            let mut flags: u32 = 1;
            let mut eqs: u8 = 0x1b;
            let sub_mb_type = if !self.is_cabac {
                self.bits.get_ue16(3)
            } else {
                0
            };
            let cond = if self.is_cabac {
                if self.cabac.get_ae(&mut self.bits, 21) != 0 {
                    true
                } else {
                    self.transform_8x8_mode_flag = 0;
                    flags = 5;
                    eqs = 0x11;
                    self.cabac.get_ae(&mut self.bits, 22) == 0
                }
            } else if sub_mb_type == 0 {
                true
            } else {
                self.transform_8x8_mode_flag = 0;
                flags = 5;
                eqs = 0x11;
                sub_mb_type == 1
            };
            if cond {
                // 8x4
                self.unavail4x4[i4x4] =
                    (self.unavail4x4[i4x4] & 11) | (self.unavail4x4[i4x4 + 1] & 4);
                self.unavail4x4[i4x4 + 2] |= 4;
                self.refidx4x4_c[i4x4] = (0x0d63 >> i4x4) as i8 & 15;
                self.mvs_c[i4x4] = self.mvs_c[i4x4 + 1];
            } else {
                // 4xN
                self.refidx4x4_c[i4x4] = (0xdc32 >> i4x4) as i8 & 15;
                self.mvs_c[i4x4] = self.mvs_b[i4x4 + 1];
                let c2 = if self.is_cabac {
                    self.cabac.get_ae(&mut self.bits, 23) != 0
                } else {
                    sub_mb_type == 2
                };
                if c2 {
                    flags = 3;
                    eqs = 0x0a;
                } else {
                    flags = 15;
                    eqs = 0;
                }
            }
            mvd_flags |= flags << i4x4;
            m.f.inter_eqs[i8x8] = eqs;
        }
        self.parse_ref_idx(ref_idx_flags);

        // load neighbouring refIdx values and shuffle them into A/B/C/D
        let b = unsafe { &*self.mb_b };
        let c = unsafe { &*self.mb_c };
        let a = unsafe { &*self.mb_a };
        let d = unsafe { &*self.mb_d };
        let mut bc = [0i8; 16];
        bc[..8].copy_from_slice(&b.refIdx);
        bc[8..].copy_from_slice(&c.refIdx);
        let mut ar = [0i8; 16];
        ar[..8].copy_from_slice(&a.refIdx);
        ar[8..].copy_from_slice(&m.refIdx);
        let bcar0 = unziplo32_bytes(&bc, &ar);
        let r0 = shuffle_i8x16(
            &bcar0,
            &[
                12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
            ],
        );
        let a0 = shuffle_i8x16(
            &bcar0,
            &[9, 12, 9, 12, 12, 13, 12, 13, 11, 14, 11, 14, 14, 15, 14, 15],
        );
        let b0 = shuffle_i8x16(
            &bcar0,
            &[2, 2, 12, 12, 3, 3, 13, 13, 12, 12, 14, 14, 13, 13, 15, 15],
        );
        let c0 = shuffle_i8x16(&bcar0, &self.refidx4x4_c);
        let mut d0 = shuffle_i8x16(
            &bcar0,
            &[-1, 2, 9, 12, 2, 3, 12, 13, 9, 12, 11, 14, 12, 13, 14, 15],
        );
        d0[0] = d.refIdx[3];

        // combine into a per-4x4 4-bit equality mask:
        // bit0 = (r0==A0 | u==14), bit1 = r0==B0, bit2 = eqC (C->D if u&4), bit3 = u&4
        let mut eqs = [0i8; 16];
        for i in 0..16 {
            let u = self.unavail4x4[i];
            let u_c = (u & 4) != 0;
            let eq_c = if u_c { r0[i] == d0[i] } else { r0[i] == c0[i] };
            let eq_b = r0[i] == b0[i];
            let eq_a = r0[i] == a0[i] || u == 14;
            eqs[i] =
                (eq_a as i32 | (eq_b as i32) << 1 | (eq_c as i32) << 2 | (u_c as i32) << 3) as i8;
        }

        // loop on mvs
        const MASKS: [i8; 16] = [0, 15, 10, 5, 12, 3, 0, 0, 8, 0, 0, 0, 4, 0, 2, 1];
        let mut mf = mvd_flags;
        while mf != 0 {
            let i = mf.trailing_zeros() as usize;
            let mvd = self.parse_mvd_pair(i, 0);
            let eq = eqs[i] as u32;
            let mvs_dc = if eq & 8 != 0 {
                self.mvs_d[i]
            } else {
                self.mvs_c[i]
            };
            let mvp = if (0xe9e9u16 >> eq) & 1 != 0 {
                unsafe {
                    let mv_a = self.read_mvs_s(self.mvs_a[i] as isize);
                    let mv_b = self.read_mvs_s(self.mvs_b[i] as isize);
                    let mv_dc = self.read_mvs_s(mvs_dc as isize);
                    median_pair_i32(mv_a, mv_b, mv_dc)
                }
            } else {
                let mvs_ab = if eq & 1 != 0 {
                    self.mvs_a[i] as i32
                } else {
                    self.mvs_b[i]
                };
                let off = if eq & 4 != 0 { mvs_dc } else { mvs_ab };
                unsafe { self.read_mvs_s(off as isize) }
            };

            let type_ = ((mf >> (i & !3)) & 15) as usize;
            let msk = MASKS[type_] as i16;
            let i8x8 = i >> 2;
            // absMvd_mask: bits={1,2,4,8,0,0,0,0}; lane j set iff (m & bits[j])==bits[j]
            let bits16: [i16; 8] = [1, 2, 4, 8, 0, 0, 0, 0];
            let abs_mask: [i16; 8] = (0..8)
                .map(|j| {
                    if (msk & bits16[j]) == bits16[j] {
                        -1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            // selectively write absMvd (low 4 i16 pairs of the 8x8)
            let p = pack_absmvd_bytes(mvd);
            let base = i8x8 * 8;
            for (lane, &am) in abs_mask[..4].iter().enumerate() {
                if am == -1 {
                    m.absMvd[base + 2 * lane] = p[0] as u8;
                    m.absMvd[base + 2 * lane + 1] = p[1] as u8;
                }
            }
            // selectively write mvs (mvs_mask = ziplo16(abs_mask, abs_mask))
            let mv = add_pair_i32(mvp, mvd);
            let mvx = pair_x(mv);
            let mvy = pair_y(mv);
            let mvs_mask = ziplo16_self(&abs_mask);
            for (lane, &mm) in mvs_mask.iter().enumerate() {
                if mm == -1 {
                    m.mvs[base + lane] = if lane & 1 == 0 { mvx } else { mvy };
                }
            }
            // C masks/widths/heights tables (edge264_slice.c parse_P_sub_mb)
            const WIDTHS: [i8; 16] = [0, 8, 4, 4, 8, 8, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4];
            const HEIGHTS: [i8; 16] = [0, 8, 8, 8, 4, 4, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4];
            self.decode_inter(i, WIDTHS[type_] as u32, HEIGHTS[type_] as u32);
            mf &= mf - 1;
        }
        self.parse_inter_residual();
    }

    /// C `parse_B_mb` (mb_skip_flag + mb_type + direct + large-block decode).
    fn parse_B_mb(&mut self) {
        let m = unsafe { &mut *self.mb() };
        // Inter initializations
        m.mbIsInterFlag = 1;
        for i in 0..16 {
            m.Intra4x4PredMode[i] = 2;
        }
        if self.is_cabac {
            for i in 0..64 {
                m.absMvd[i] = 0;
            }
        }
        for i in 0..64 {
            m.mvs[i] = 0;
        }

        // parse mb_skip_run / flag
        let mb_skip_flag: u32 = if !self.is_cabac {
            if self.mb_skip_run < 0 {
                self.mb_skip_run = self.bits.get_ue16(139264) as i32;
            }
            // C: `mb_skip_run-- > 0` — post-decrement, test the OLD value.
            let flag = self.mb_skip_run > 0;
            self.mb_skip_run -= 1;
            flag as u32
        } else {
            self.cabac
                .get_ae(&mut self.bits, 26 - self.inc.mb_skip_flag as usize)
        };

        // B_Skip
        if mb_skip_flag != 0 {
            if self.is_cabac {
                m.f.mb_skip_flag = 1;
                m.f.mb_type_B_Direct = 1;
                self.mb_qp_delta_nz = 0;
            }
            m.f.inter_eqs = [0; 4];
            let state = self.mvpred_state();
            let mut mc = mvpred::McOps::new();
            let out = mvpred::direct(
                &state,
                self.direct_spatial_mv_pred_flag != 0,
                0xffffffff,
                &mut mc,
            );
            mb_apply(m, &out);
            for (i, w, h) in mc.iter() {
                self.decode_inter(*i, *w, *h);
            }
            return;
        }

        // B_Direct_16x16
        let mb_type_cavlc = if !self.is_cabac {
            self.bits.get_ue16(48) as i32
        } else {
            0
        };
        let is_direct = if self.is_cabac {
            self.cabac
                .get_ae(&mut self.bits, 29 - self.inc.mb_type_B_Direct as usize)
                == 0
        } else {
            mb_type_cavlc == 0
        };
        if is_direct {
            if self.is_cabac {
                m.f.mb_type_B_Direct = 1;
            }
            self.transform_8x8_mode_flag =
                self.pps_transform_8x8_mode_flag & self.direct_8x8_inference_flag;
            m.f.inter_eqs = [0; 4];
            let state = self.mvpred_state();
            let mut mc = mvpred::McOps::new();
            mb_apply(
                m,
                &mvpred::direct(
                    &state,
                    self.direct_spatial_mv_pred_flag != 0,
                    0xffffffff,
                    &mut mc,
                ),
            );
            for (i, w, h) in mc.iter() {
                self.decode_inter(*i, *w, *h);
            }
            self.parse_inter_residual();
            return;
        }
        self.transform_8x8_mode_flag = self.pps_transform_8x8_mode_flag;

        // parse mb_type -> flags8x8 (or jump to I / sub_mb)
        let flags8x8: u32 = if !self.is_cabac {
            if mb_type_cavlc > 22 {
                self.parse_I_mb(mb_type_cavlc - 23);
                return;
            }
            m.refIdx = [-1; 8];
            for i in 0..64 {
                m.mvs[i] = 0;
            }
            if mb_type_cavlc == 22 {
                self.parse_B_sub_mb();
                return;
            }
            const MB_TYPE2FLAGS: [u8; 22] = [
                0, 1, 0x10, 0x11, 5, 3, 0x50, 0x30, 0x41, 0x21, 0x14, 0x12, 0x45, 0x23, 0x54, 0x32,
                0x15, 0x13, 0x51, 0x31, 0x55, 0x33,
            ];
            MB_TYPE2FLAGS[mb_type_cavlc as usize] as u32
        } else {
            let mut str_ = 4i32;
            if self.cabac.get_ae(&mut self.bits, 30) == 0 {
                str_ = str_ + str_ + self.cabac.get_ae(&mut self.bits, 32) as i32;
            } else {
                str_ = self.cabac.get_ae(&mut self.bits, 31) as i32;
                str_ = str_ + str_ + self.cabac.get_ae(&mut self.bits, 32) as i32;
                str_ = str_ + str_ + self.cabac.get_ae(&mut self.bits, 32) as i32;
                str_ = str_ + str_ + self.cabac.get_ae(&mut self.bits, 32) as i32;
                if str_ - 8 >= 0 && str_ - 8 < 5 {
                    str_ = str_ + str_ + self.cabac.get_ae(&mut self.bits, 32) as i32;
                }
            }
            if str_ == 13 {
                self.parse_I_mb(32);
                return;
            }
            m.refIdx = [-1; 8];
            for i in 0..64 {
                m.mvs[i] = 0;
            }
            if str_ == 15 {
                self.parse_B_sub_mb();
                return;
            }
            const STR2FLAGS: [u8; 26] = [
                0x11, 0x05, 0x03, 0x50, 0x30, 0x41, 0x21, 0x14, 0x01, 0x10, 0, 0, 0, 0, 0x12, 0,
                0x45, 0x23, 0x54, 0x32, 0x15, 0x13, 0x51, 0x31, 0x55, 0x33,
            ];
            STR2FLAGS[str_ as usize] as u32
        };

        // decoding large blocks
        self.parse_ref_idx(flags8x8);
        if flags8x8 & 0xee == 0 {
            // 16x16
            m.f.inter_eqs = 0x1b5fbbff_u32.to_le_bytes();
            if flags8x8 & 0x01 != 0 {
                let mvd = self.parse_mvd_pair(0, 0);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x16(&s, mvd, 0));
                self.decode_inter(0, 16, 16);
            }
            if flags8x8 & 0x10 != 0 {
                let mvd = self.parse_mvd_pair(0, 32);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x16(&s, mvd, 1));
                self.decode_inter(16, 16, 16);
            }
        } else if flags8x8 & 0xcc == 0 {
            // 8x16
            m.f.inter_eqs = 0x1b1bbbbb_u32.to_le_bytes();
            if flags8x8 & 0x01 != 0 {
                let mvd = self.parse_mvd_pair(0, 0);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_8x16_left(&s, mvd, 0));
                self.decode_inter(0, 8, 16);
            }
            if flags8x8 & 0x02 != 0 {
                let mvd = self.parse_mvd_pair(4, 0);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_8x16_right(&s, mvd, 0));
                self.decode_inter(4, 8, 16);
            }
            if flags8x8 & 0x10 != 0 {
                let mvd = self.parse_mvd_pair(0, 32);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_8x16_left(&s, mvd, 1));
                self.decode_inter(16, 8, 16);
            }
            if flags8x8 & 0x20 != 0 {
                let mvd = self.parse_mvd_pair(4, 32);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_8x16_right(&s, mvd, 1));
                self.decode_inter(20, 8, 16);
            }
        } else {
            // 16x8
            m.f.inter_eqs = 0x1b5f1b5f_u32.to_le_bytes();
            if flags8x8 & 0x01 != 0 {
                let mvd = self.parse_mvd_pair(0, 0);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x8_top(&s, mvd, 0));
                self.decode_inter(0, 16, 8);
            }
            if flags8x8 & 0x04 != 0 {
                let mvd = self.parse_mvd_pair(8, 0);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x8_bottom(&s, mvd, 0));
                self.decode_inter(8, 16, 8);
            }
            if flags8x8 & 0x10 != 0 {
                let mvd = self.parse_mvd_pair(0, 32);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x8_top(&s, mvd, 1));
                self.decode_inter(16, 16, 8);
            }
            if flags8x8 & 0x40 != 0 {
                let mvd = self.parse_mvd_pair(8, 32);
                let s = self.mvpred_state();
                mb_apply(m, &mvpred::inter_16x8_bottom(&s, mvd, 1));
                self.decode_inter(24, 16, 8);
            }
        }
        self.parse_inter_residual();
    }

    /// C `parse_B_sub_mb` (sub_mb_type + L0/L1 refIdx shuffle + per-block MVD/MVP).
    /// Faithful scalar transcription; not exercised by the current test samples.
    fn parse_B_sub_mb(&mut self) {
        let m = unsafe { &mut *self.mb() };
        let mut mvd_flags: u32 = 0;
        m.f.inter_eqs = [0; 4];
        for i8x8 in 0..4 {
            let i4x4 = i8x8 * 4;
            let sub_mb_type = if !self.is_cabac {
                self.bits.get_ue16(12)
            } else {
                0
            };
            let is_direct = if self.is_cabac {
                self.cabac.get_ae(&mut self.bits, 36) == 0
            } else {
                sub_mb_type == 0
            };
            if is_direct {
                // B_Direct_8x8
                m.f.inter_eqs[i8x8] = 0x1b;
                if self.direct_8x8_inference_flag == 0 {
                    self.transform_8x8_mode_flag = 0;
                }
            } else {
                let sub: u32;
                if !self.is_cabac {
                    const SUB_MB_TYPE2FLAGS: [u32; 13] = [
                        0, 0x00001, 0x10000, 0x10001, 0x00005, 0x00003, 0x50000, 0x30000, 0x50005,
                        0x30003, 0x0000f, 0xf0000, 0xf000f,
                    ];
                    const SUB_MB_TYPE2EQS: [u8; 13] = [
                        0, 0x1b, 0x1b, 0x1b, 0x11, 0x0a, 0x11, 0x0a, 0x11, 0x0a, 0, 0, 0,
                    ];
                    sub = sub_mb_type;
                    mvd_flags |= SUB_MB_TYPE2FLAGS[sub as usize] << i4x4;
                    m.f.inter_eqs[i8x8] = SUB_MB_TYPE2EQS[sub as usize];
                } else {
                    let mut sub_ = 2i32;
                    if self.cabac.get_ae(&mut self.bits, 37) == 0 {
                        sub_ = sub_ + sub_ + self.cabac.get_ae(&mut self.bits, 39) as i32;
                    } else {
                        sub_ = self.cabac.get_ae(&mut self.bits, 38) as i32;
                        sub_ = sub_ + sub_ + self.cabac.get_ae(&mut self.bits, 39) as i32;
                        sub_ = sub_ + sub_ + self.cabac.get_ae(&mut self.bits, 39) as i32;
                        if sub_ - 4 >= 0 && sub_ - 4 < 2 {
                            sub_ = sub_ + sub_ + self.cabac.get_ae(&mut self.bits, 39) as i32;
                        }
                    }
                    sub = sub_ as u32;
                    const SUB2FLAGS: [u32; 12] = [
                        0x10001, 0x00005, 0x00003, 0x50000, 0x00001, 0x10000, 0xf0000, 0xf000f,
                        0x30000, 0x50005, 0x30003, 0x0000f,
                    ];
                    const SUB2EQS: [u8; 12] = [
                        0x1b, 0x11, 0x0a, 0x11, 0x1b, 0x1b, 0, 0, 0x0a, 0x11, 0x0a, 0,
                    ];
                    mvd_flags |= SUB2FLAGS[sub as usize] << i4x4;
                    m.f.inter_eqs[i8x8] = SUB2EQS[sub as usize];
                }
                let is_8xn = if self.is_cabac {
                    (0x23b & (1 << sub)) != 0
                } else {
                    (0x015f & (1 << sub_mb_type)) != 0
                };
                if is_8xn {
                    // 8xN
                    self.unavail4x4[i4x4] =
                        (self.unavail4x4[i4x4] & 11) | (self.unavail4x4[i4x4 + 1] & 4);
                    self.unavail4x4[i4x4 + 2] |= 4;
                    self.refidx4x4_c[i4x4] = (0x0d63 >> i4x4) as i8 & 15;
                    self.mvs_c[i4x4] = self.mvs_c[i4x4 + 1];
                    let is_8x8_like = if self.is_cabac {
                        (0xfce & (1 << sub)) != 0
                    } else {
                        (0x1ff0 & (1 << sub_mb_type)) != 0
                    };
                    if is_8x8_like {
                        self.transform_8x8_mode_flag = 0;
                    }
                } else {
                    // 4xN
                    self.refidx4x4_c[i4x4] = (0xdc32 >> i4x4) as i8 & 15;
                    self.mvs_c[i4x4] = self.mvs_b[i4x4 + 1];
                    self.transform_8x8_mode_flag = 0;
                }
            }
        }

        // initialize direct prediction then parse all ref_idx values
        let direct_flags = (!((mvd_flags & 0xffff) | (mvd_flags >> 16)) & 0x1111) * 0xf000f;
        if direct_flags != 0 {
            let state = self.mvpred_state();
            let mut mc = mvpred::McOps::new();
            mb_apply(
                m,
                &mvpred::direct(
                    &state,
                    self.direct_spatial_mv_pred_flag != 0,
                    direct_flags,
                    &mut mc,
                ),
            );
            for (i, w, h) in mc.iter() {
                self.decode_inter(*i, *w, *h);
            }
        }
        if mvd_flags == 0 {
            self.parse_inter_residual();
            return;
        }
        self.parse_ref_idx(0x100 | mvd_flags2ref_idx(mvd_flags));

        // load neighbouring refIdx values and shuffle them into A/B/C/D (L0 + L1)
        let b = unsafe { &*self.mb_b };
        let c = unsafe { &*self.mb_c };
        let a = unsafe { &*self.mb_a };
        let d = unsafe { &*self.mb_d };
        let mut bc = [0i8; 16];
        bc[..8].copy_from_slice(&b.refIdx);
        bc[8..].copy_from_slice(&c.refIdx);
        let mut ar = [0i8; 16];
        ar[..8].copy_from_slice(&a.refIdx);
        ar[8..].copy_from_slice(&m.refIdx);
        let bcar0 = unziplo32_bytes(&bc, &ar);
        let bcar1 = unziphi32_bytes(&bc, &ar);
        const IDX_R: [i8; 16] = [
            12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
        ];
        const IDX_AB: [i8; 16] = [9, 12, 9, 12, 12, 13, 12, 13, 11, 14, 11, 14, 14, 15, 14, 15];
        const IDX_B: [i8; 16] = [2, 2, 12, 12, 3, 3, 13, 13, 12, 12, 14, 14, 13, 13, 15, 15];
        const IDX_D: [i8; 16] = [-1, 2, 9, 12, 2, 3, 12, 13, 9, 12, 11, 14, 12, 13, 14, 15];
        let r0 = shuffle_i8x16(&bcar0, &IDX_R);
        let r1 = shuffle_i8x16(&bcar1, &IDX_R);
        let a0 = shuffle_i8x16(&bcar0, &IDX_AB);
        let a1 = shuffle_i8x16(&bcar1, &IDX_AB);
        let b0 = shuffle_i8x16(&bcar0, &IDX_B);
        let b1 = shuffle_i8x16(&bcar1, &IDX_B);
        let c0 = shuffle_i8x16(&bcar0, &self.refidx4x4_c);
        let c1 = shuffle_i8x16(&bcar1, &self.refidx4x4_c);
        let mut d0 = shuffle_i8x16(&bcar0, &IDX_D);
        let mut d1 = shuffle_i8x16(&bcar1, &IDX_D);
        d0[0] = d.refIdx[3];
        d1[0] = d.refIdx[7];

        // 4-bit equality masks: L0 in [0..16], L1 in [16..32]
        let mut eqs = [0i8; 32];
        for i in 0..16 {
            let u = self.unavail4x4[i];
            let u_c = (u & 4) != 0;
            let eq_c0 = if u_c { r0[i] == d0[i] } else { r0[i] == c0[i] };
            let eq_b0 = r0[i] == b0[i];
            let eq_a0 = r0[i] == a0[i] || u == 14;
            eqs[i] = (eq_a0 as i32 | (eq_b0 as i32) << 1 | (eq_c0 as i32) << 2 | (u_c as i32) << 3)
                as i8;
            let eq_c1 = if u_c { r1[i] == d1[i] } else { r1[i] == c1[i] };
            let eq_b1 = r1[i] == b1[i];
            let eq_a1 = r1[i] == a1[i] || u == 14;
            eqs[16 + i] =
                (eq_a1 as i32 | (eq_b1 as i32) << 1 | (eq_c1 as i32) << 2 | (u_c as i32) << 3)
                    as i8;
        }

        // loop on mvs
        const MASKS: [i8; 16] = [0, 15, 10, 5, 12, 3, 0, 0, 8, 0, 0, 0, 4, 0, 2, 1];
        const SUB_WIDTHS: [u32; 16] = [0, 8, 4, 4, 8, 8, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4];
        const SUB_HEIGHTS: [u32; 16] = [0, 8, 8, 8, 4, 4, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4];
        let mut mf = mvd_flags;
        while mf != 0 {
            let i = mf.trailing_zeros() as usize;
            let i4x4 = i & 15;
            // mb->mvs_s[i] = 0 (value pointed to when A/B/C/D are unavailable)
            unsafe { *(self.mb() as *mut u8).add(176 + 4 * i).cast::<i32>() = 0 };
            let absmvd_off = (i & 16) as isize * 2;
            let mvd = self.parse_mvd_pair(i4x4, absmvd_off);

            let eq = eqs[i] as u32;
            let mvs_dc = if eq & 8 != 0 {
                self.mvs_d[i4x4]
            } else {
                self.mvs_c[i4x4]
            };
            let mvs_base_off = (i & 16) as isize; // mvs_p = mb->mvs_s + (i & 16)
            let mvp = if (0xe9e9u16 >> eq) & 1 != 0 {
                unsafe {
                    let base = (self.mb() as *const u8).offset(176 + 4 * mvs_base_off);
                    let mv_a = *(base.cast::<i32>().offset(self.mvs_a[i4x4] as isize));
                    let mv_b = *(base.cast::<i32>().offset(self.mvs_b[i4x4] as isize));
                    let mv_dc = *(base.cast::<i32>().offset(mvs_dc as isize));
                    median_pair_i32(mv_a, mv_b, mv_dc)
                }
            } else {
                let mvs_ab = if eq & 1 != 0 {
                    self.mvs_a[i4x4] as i32
                } else {
                    self.mvs_b[i4x4]
                };
                let off = if eq & 4 != 0 { mvs_dc } else { mvs_ab };
                unsafe {
                    *((self.mb() as *const u8)
                        .offset(176 + 4 * mvs_base_off)
                        .cast::<i32>()
                        .offset(off as isize))
                }
            };

            let type_ = ((mf >> (i & !3)) & 15) as usize;
            let msk = MASKS[type_] as i16;
            let i8x8 = i >> 2;
            let bits16: [i16; 8] = [1, 2, 4, 8, 0, 0, 0, 0];
            let abs_mask: [i16; 8] = (0..8)
                .map(|j| {
                    if (msk & bits16[j]) == bits16[j] {
                        -1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let p = pack_absmvd_bytes(mvd);
            let base = i8x8 * 8;
            for (lane, &am) in abs_mask[..4].iter().enumerate() {
                if am == -1 {
                    m.absMvd[base + 2 * lane] = p[0] as u8;
                    m.absMvd[base + 2 * lane + 1] = p[1] as u8;
                }
            }
            let mv = add_pair_i32(mvp, mvd);
            let mvx = pair_x(mv);
            let mvy = pair_y(mv);
            let mvs_mask = ziplo16_self(&abs_mask);
            for (lane, &mm) in mvs_mask.iter().enumerate() {
                if mm == -1 {
                    m.mvs[base + lane] = if lane & 1 == 0 { mvx } else { mvy };
                }
            }
            // C: decode_inter(ctx, i, widths[type], heights[type])
            self.decode_inter(i, SUB_WIDTHS[type_], SUB_HEIGHTS[type_]);
            mf &= mf - 1;
        }
        self.parse_inter_residual();
    }

    // ------------------------------------------------------------------
    // I-slice path + residual parsing (Tier D)
    //
    // Pixel kernels (`decode_intra*`, `add_idct*`, `transform_dc*`,
    // `add_dc*`) are no-op targets: they only touch reconstruction samples,
    // never the 304-byte record or entropy state.
    // ------------------------------------------------------------------

    /// C `parse_mb_qp_delta` (9.3.2.7 and 9.3.3.1.1.5).
    fn parse_mb_qp_delta(&mut self) {
        let delta: i32 = if !self.is_cabac {
            self.bits.get_se16(-26, 25)
        } else {
            let nz = self
                .cabac
                .get_ae(&mut self.bits, 60 + self.mb_qp_delta_nz as usize);
            self.mb_qp_delta_nz = nz as i8;
            if nz == 0 {
                0
            } else {
                let mut count: u32 = 1;
                let mut ctx_idx = 62;
                while self.cabac.get_ae(&mut self.bits, ctx_idx) != 0 && count < 52 {
                    count += 1;
                    ctx_idx = 63;
                }
                if count & 1 != 0 {
                    count as i32 / 2 + 1
                } else {
                    -(count as i32 / 2)
                }
            }
        };
        if delta != 0 {
            let sum = self.qp_s[0] as i32 + delta;
            let qp_y = if sum < 0 {
                sum + 52
            } else if sum >= 52 {
                sum - 52
            } else {
                sum
            };
            self.qp_s = [
                qp_y as u8,
                self.qp_c[0][qp_y as usize] as u8,
                self.qp_c[1][qp_y as usize] as u8,
                0,
            ];
            unsafe {
                (*self.mb()).QP = self.qp_s;
            }
        }
    }

    /// C `parse_intraNxN_pred_mode` (7.3.5.1, 7.4.5.1, 8.3.1.1). Returns the
    /// intra_pred_mode; `A4x4_int8`/`B4x4_int8` are byte offsets relative to
    /// `mb->Intra4x4PredMode` (negative ones reach into neighbour MBs).
    fn parse_intraNxN_pred_mode(&mut self, i4x4: usize) -> i32 {
        // C reads `*((int8_t *)mb->Intra4x4PredMode + off)` — signed!
        let base = unsafe { (self.mb() as *const u8).offset(16) }; // Intra4x4PredMode @ 16
        let a = unsafe { *base.offset(self.a4x4_int8[i4x4] as isize).cast::<i8>() } as i32;
        let b = unsafe { *base.offset(self.b4x4_int8[i4x4] as isize).cast::<i8>() } as i32;
        let mut mode = (a.min(b)).abs();
        let flag = if !self.is_cabac {
            self.bits.get_u1() == 0
        } else {
            self.cabac.get_ae(&mut self.bits, 68) == 0
        };
        if flag {
            let rem: i32 = if !self.is_cabac {
                self.bits.get_uv(3) as i32
            } else {
                self.cabac.get_ae(&mut self.bits, 69) as i32
                    + self.cabac.get_ae(&mut self.bits, 69) as i32 * 2
                    + self.cabac.get_ae(&mut self.bits, 69) as i32 * 4
            };
            mode = rem + (rem >= mode) as i32;
        }
        mode
    }

    /// C `parse_intra_chroma_pred_mode` (9.3.2.2 and 9.3.3.1.1.8).
    fn parse_intra_chroma_pred_mode(&mut self) {
        if self.chroma_array_type == 1 || self.chroma_array_type == 2 {
            let mode: i32 = if !self.is_cabac {
                self.bits.get_ue16(3) as i32
            } else {
                let mut ctx_idx = 64 + self.inc.intra_chroma_pred_mode_non_zero as usize;
                let mut mode: i32 = 0;
                while mode < 3 && self.cabac.get_ae(&mut self.bits, ctx_idx) != 0 {
                    mode += 1;
                    ctx_idx = 67;
                }
                unsafe {
                    (*self.mb()).f.intra_chroma_pred_mode_non_zero = (mode > 0) as i8;
                }
                mode
            };
            // C decode_intraChroma(samples_mb[1], stride[1] >> 1, ...)
            let off = self.plane_off(self.samples_mb[1] as *const u8);
            let stride = (self.stride[1] / 2) as usize;
            let variant = INTRA_CHROMA_MODES[mode as usize][self.unavail4x4[0] as usize & 3];
            let buf = self.plane_slice();
            intra_chroma(buf, off, stride, variant);
        }
    }

    /// C `parse_coded_block_pattern` (9.3.2.6 and 9.3.3.1.1.4). `map_me` is
    /// the CAVLC cbp->bits mapping table (`me_intra` or `me_inter`).
    fn parse_coded_block_pattern(&mut self, map_me: &[u8; 48]) {
        let m = unsafe { &mut *self.mb() };
        let mut cbp: i32 = 0;
        if !self.is_cabac {
            cbp = map_me[self.bits.get_ue16(47) as usize] as i32;
            m.bits[0] |= (cbp & 0xac) as u32;
        } else {
            let mut bits = m.bits[0];
            bits |= self.cabac.get_ae(&mut self.bits, 76 - (bits & 3) as usize) << 5;
            bits |= self
                .cabac
                .get_ae(&mut self.bits, 76 - ((bits >> 5) & 3) as usize)
                << 3;
            bits |= self
                .cabac
                .get_ae(&mut self.bits, 76 - ((bits >> 4) & 3) as usize)
                << 2;
            bits |= self
                .cabac
                .get_ae(&mut self.bits, 76 - ((bits >> 2) & 3) as usize)
                << 7;
            m.bits[0] = bits;
        }
        // Chroma suffix
        let dc: i32 = if !self.is_cabac {
            cbp & 3
        } else {
            self.cabac.get_ae(
                &mut self.bits,
                77 + self.inc.CodedBlockPatternChromaDC as usize,
            ) as i32
        };
        if (self.chroma_array_type == 1 || self.chroma_array_type == 2) && dc != 0 {
            m.f.CodedBlockPatternChromaDC = 1;
            m.f.CodedBlockPatternChromaAC = if !self.is_cabac {
                ((cbp >> 1) & 1) as i8
            } else {
                self.cabac.get_ae(
                    &mut self.bits,
                    81 + self.inc.CodedBlockPatternChromaAC as usize,
                ) as i8
            };
        }
    }

    /// C `parse_chroma_residual` — tail-called from the Intra residual paths.
    fn parse_chroma_residual(&mut self) {
        let m = unsafe { &mut *self.mb() };
        if m.f.CodedBlockPatternChromaDC != 0 {
            if !self.is_cabac {
                self.scan[..4].copy_from_slice(&[0, 4, 2, 6]);
                self.parse_residual_block_2x2_cavlc();
                self.scan[..4].copy_from_slice(&[1, 5, 3, 7]);
                self.parse_residual_block_2x2_cavlc();
            } else {
                self.ctx_idx_offsets = CTX_IDX_CHROMA_DC[0];
                self.coeff_abs_inc = [6, 7, 8, 8, 0, 0, 0, 0];
                if self.cabac.get_ae(
                    &mut self.bits,
                    (self.ctx_idx_offsets[0] + self.inc.coded_block_flags_16x16[1] as i16) as usize,
                ) != 0
                {
                    m.f.coded_block_flags_16x16[1] = 1;
                    self.scan[..4].copy_from_slice(&[0, 4, 2, 6]);
                    self.parse_residual_block_cabac(0, 3);
                }
                if self.cabac.get_ae(
                    &mut self.bits,
                    (self.ctx_idx_offsets[0] + self.inc.coded_block_flags_16x16[2] as i16) as usize,
                ) != 0
                {
                    m.f.coded_block_flags_16x16[2] = 1;
                    self.scan[..4].copy_from_slice(&[1, 5, 3, 7]);
                    self.parse_residual_block_cabac(0, 3);
                }
            }
            // C transform_dc2x2(ctx)
            {
                let m = unsafe { &mut *self.mb() };
                unsafe {
                    transform_dc2x2(
                        &mut self.c,
                        &self.ws4,
                        m.mbIsInterFlag != 0,
                        &self.qp_s,
                        m.f.CodedBlockPatternChromaAC != 0,
                        self.samples_mb[1],
                        self.stride[1] as usize,
                    );
                }
            }

            if m.f.CodedBlockPatternChromaAC != 0 {
                if self.is_cabac {
                    self.ctx_idx_offsets = CTX_IDX_CHROMA_AC[0];
                    self.coeff_abs_inc = [6, 7, 8, 9, 9, 0, 0, 0];
                }
                // C: ctx->scan_v[0] = scan_4x4[0]
                self.scan[..16].copy_from_slice(&SCAN_4X4[0]);
                let inter3 = (unsafe { (*self.mb()).mbIsInterFlag as usize }) * 3;
                for i4x4 in 0..8 {
                    let iycbcr = 1 + (i4x4 >> 2);
                    // C: samples_mb[iYCbCr] + y420[i4x4] * stride[1] + x420[i4x4]
                    let pix = unsafe {
                        self.samples_mb[iycbcr].add(
                            Y420[i4x4] as usize * self.stride[1] as usize + X420[i4x4] as usize,
                        )
                    };
                    let base = unsafe { (self.mb() as *const u8).offset(32 + 16) }; // nC + 16
                    let n_a = unsafe { *base.offset(self.acbcr_int8[i4x4] as isize) } as i32;
                    let n_b = unsafe { *base.offset(self.bcbcr_int8[i4x4] as isize) } as i32;
                    if !self.is_cabac {
                        if self.parse_residual_block_4x4_cavlc(1, 16 + i4x4 as i32, n_a, n_b) {
                            // C add_idct4x4(ctx, iYCbCr, i4x4, samples)
                            unsafe {
                                add_idct4x4(
                                    &mut self.c,
                                    self.qp_s[iycbcr],
                                    &self.ws4[iycbcr + inter3],
                                    i4x4 as i32,
                                    pix,
                                    self.stride[1] as usize,
                                );
                            }
                        } else {
                            unsafe {
                                add_dc4x4(&self.c, i4x4 as i32, pix, self.stride[1] as usize)
                            };
                        }
                    } else if self.cabac.get_ae(
                        &mut self.bits,
                        (self.ctx_idx_offsets[0]
                            + self.nc_inc[1][i4x4] as i16
                            + n_a as i16
                            + n_b as i16 * 2) as usize,
                    ) != 0
                    {
                        unsafe {
                            (*self.mb()).nC[16 + i4x4] = 1;
                        }
                        self.parse_residual_block_cabac(1, 15);
                        unsafe {
                            add_idct4x4(
                                &mut self.c,
                                self.qp_s[iycbcr],
                                &self.ws4[iycbcr + inter3],
                                i4x4 as i32,
                                pix,
                                self.stride[1] as usize,
                            );
                        }
                    } else {
                        unsafe { add_dc4x4(&self.c, i4x4 as i32, pix, self.stride[1] as usize) };
                    }
                }
            }
            // C `ctx->c_v[4] = ctx->c_v[5] = (i8x16){}`
            for i in 16..24 {
                self.c[i] = 0;
            }
        }
    }

    /// C `parse_Intra16x16_residual` — tail-called from `parse_I_mb`.
    fn parse_Intra16x16_residual(&mut self) {
        let m = unsafe { &mut *self.mb() };
        self.parse_mb_qp_delta();
        self.scan[..16].copy_from_slice(&SCAN_4X4[0]);
        let inter3 = (m.mbIsInterFlag as usize) * 3;
        for iycbcr in 0..3 {
            // DC block (parsed to c[0..15], then "transformed" to c[16..31])
            if !self.is_cabac {
                let a = unsafe { &*self.mb_a };
                let b = unsafe { &*self.mb_b };
                if self.parse_residual_block_4x4_cavlc(
                    0,
                    0,
                    a.nC[iycbcr * 16 + 5] as i32,
                    b.nC[iycbcr * 16 + 10] as i32,
                ) {
                    unsafe {
                        (*self.mb()).nC[0] = 0;
                    }
                    // C transform_dc4x4(ctx, iYCbCr)
                    unsafe {
                        transform_dc4x4(
                            &mut self.c,
                            &self.ws4[iycbcr],
                            self.qp_s[0],
                            m.bits[0] & (1 << 5) != 0,
                            self.samples_mb[iycbcr],
                            self.stride[iycbcr.min(1)] as usize,
                        );
                    }
                }
            } else {
                self.ctx_idx_offsets = CTX_IDX_16X16_DC[iycbcr][0];
                if self.cabac.get_ae(
                    &mut self.bits,
                    (self.ctx_idx_offsets[0] + self.inc.coded_block_flags_16x16[iycbcr] as i16)
                        as usize,
                ) != 0
                {
                    m.f.coded_block_flags_16x16[iycbcr] = 1;
                    self.coeff_abs_inc = [6, 7, 8, 9, 9, 0, 0, 0];
                    self.parse_residual_block_cabac(0, 15);
                    // C transform_dc4x4(ctx, iYCbCr)
                    unsafe {
                        transform_dc4x4(
                            &mut self.c,
                            &self.ws4[iycbcr],
                            self.qp_s[0],
                            m.bits[0] & (1 << 5) != 0,
                            self.samples_mb[iycbcr],
                            self.stride[iycbcr.min(1)] as usize,
                        );
                    }
                }
            }

            // All AC blocks pick a DC coeff, then go to c[1..15]
            if m.bits[0] & (1 << 5) != 0 {
                if self.is_cabac {
                    self.ctx_idx_offsets = CTX_IDX_16X16_AC[iycbcr][0];
                    self.coeff_abs_inc = [6, 7, 8, 9, 9, 0, 0, 0];
                }
                let stride = self.stride[iycbcr.min(1)] as usize;
                for i4x4 in 0..16 {
                    // C: samples_mb[iYCbCr] + y444[i4x4] * stride[iYCbCr] + x444[i4x4]
                    let pix = unsafe {
                        self.samples_mb[iycbcr]
                            .add(Y444[i4x4] as usize * stride + X444[i4x4] as usize)
                    };
                    let base = unsafe { (self.mb() as *const u8).add(32 + iycbcr * 16) };
                    let n_a = unsafe { *base.offset(self.a4x4_int8[i4x4] as isize) } as i32;
                    let n_b = unsafe { *base.offset(self.b4x4_int8[i4x4] as isize) } as i32;
                    if !self.is_cabac {
                        if self.parse_residual_block_4x4_cavlc(
                            1,
                            (iycbcr * 16 + i4x4) as i32,
                            n_a,
                            n_b,
                        ) {
                            // C add_idct4x4(ctx, iYCbCr, i4x4, samples)
                            unsafe {
                                add_idct4x4(
                                    &mut self.c,
                                    self.qp_s[iycbcr],
                                    &self.ws4[iycbcr + inter3],
                                    i4x4 as i32,
                                    pix,
                                    stride,
                                );
                            }
                        } else {
                            unsafe { add_dc4x4(&self.c, i4x4 as i32, pix, stride) };
                        }
                    } else if self.cabac.get_ae(
                        &mut self.bits,
                        (self.ctx_idx_offsets[0]
                            + self.nc_inc[iycbcr][i4x4] as i16
                            + n_a as i16
                            + n_b as i16 * 2) as usize,
                    ) != 0
                    {
                        unsafe {
                            (*self.mb()).nC[iycbcr * 16 + i4x4] = 1;
                        }
                        self.parse_residual_block_cabac(1, 15);
                        unsafe {
                            add_idct4x4(
                                &mut self.c,
                                self.qp_s[iycbcr],
                                &self.ws4[iycbcr + inter3],
                                i4x4 as i32,
                                pix,
                                stride,
                            );
                        }
                    } else {
                        unsafe { add_dc4x4(&self.c, i4x4 as i32, pix, stride) };
                    }
                }
            }
            // C `ctx->c_v[4..7] = (i32x4){}`
            for i in 16..32 {
                self.c[i] = 0;
            }

            // C `CAJUMP(parse_chroma_residual)`: tail call exits the function.
            if self.chroma_array_type < 3 {
                self.parse_chroma_residual();
                return;
            }
        }
    }

    /// C `parse_NxN_residual` — Intra_4x4 and Inter sub-MB residual blocks.
    fn parse_NxN_residual(&mut self) {
        let m = unsafe { &mut *self.mb() };
        if m.f.CodedBlockPatternChromaDC != 0 || (m.bits[0] & 0xac) != 0 {
            self.parse_mb_qp_delta();
        } else if self.is_cabac {
            self.mb_qp_delta_nz = 0;
        }

        for iycbcr in 0..3 {
            if m.f.transform_size_8x8_flag == 0 {
                // 4x4 blocks
                if self.is_cabac {
                    self.ctx_idx_offsets = CTX_IDX_4X4[iycbcr][0];
                    self.coeff_abs_inc = [6, 7, 8, 9, 9, 0, 0, 0];
                }
                self.scan[..16].copy_from_slice(&SCAN_4X4[0]);
                let inter3 = (m.mbIsInterFlag as usize) * 3;
                let stride = self.stride[iycbcr.min(1)] as usize;
                for i4x4 in 0..16 {
                    // C decode_intra4x4(samples, stride, Intra4x4Modes[mode][unavail])
                    if m.mbIsInterFlag == 0 {
                        let p = unsafe {
                            (self.samples_mb[iycbcr] as *const u8)
                                .add(Y444[i4x4] as usize * stride + X444[i4x4] as usize)
                        };
                        let off = self.plane_off(p);
                        let variant = INTRA4X4_MODES[m.Intra4x4PredMode[i4x4] as usize]
                            [self.unavail4x4[i4x4] as usize];
                        let buf = self.plane_slice();
                        intra4x4(buf, off, stride, variant);
                    }
                    if m.bits[0] & (1 << BIT_8X8[i4x4 >> 2] as usize) != 0 {
                        let pix = unsafe {
                            self.samples_mb[iycbcr]
                                .add(Y444[i4x4] as usize * stride + X444[i4x4] as usize)
                        };
                        let base = unsafe { (self.mb() as *const u8).add(32 + iycbcr * 16) };
                        let n_a = unsafe { *base.offset(self.a4x4_int8[i4x4] as isize) } as i32;
                        let n_b = unsafe { *base.offset(self.b4x4_int8[i4x4] as isize) } as i32;
                        if !self.is_cabac {
                            // C: parsed ? add_idct4x4(ctx, iYCbCr, -1, samples) : nothing
                            if self.parse_residual_block_4x4_cavlc(
                                0,
                                (iycbcr * 16 + i4x4) as i32,
                                n_a,
                                n_b,
                            ) {
                                unsafe {
                                    add_idct4x4(
                                        &mut self.c,
                                        self.qp_s[iycbcr],
                                        &self.ws4[iycbcr + inter3],
                                        -1,
                                        pix,
                                        stride,
                                    );
                                }
                            }
                        } else if self.cabac.get_ae(
                            &mut self.bits,
                            (self.ctx_idx_offsets[0]
                                + self.nc_inc[iycbcr][i4x4] as i16
                                + n_a as i16
                                + n_b as i16 * 2) as usize,
                        ) != 0
                        {
                            unsafe {
                                (*self.mb()).nC[iycbcr * 16 + i4x4] = 1;
                            }
                            self.parse_residual_block_cabac(0, 15);
                            unsafe {
                                add_idct4x4(
                                    &mut self.c,
                                    self.qp_s[iycbcr],
                                    &self.ws4[iycbcr + inter3],
                                    -1,
                                    pix,
                                    stride,
                                );
                            }
                        }
                    }
                }
            } else {
                // 8x8 blocks
                let stride = self.stride[iycbcr.min(1)] as usize;
                if self.is_cabac {
                    self.ctx_idx_offsets = CTX_IDX_8X8[iycbcr][0];
                    self.coeff_abs_inc = [6, 7, 8, 9, 9, 0, 0, 0];
                    // C resets only sig_inc_v[0]/last_inc_v[0]/scan_v[0]; the
                    // slice init already filled v[1..3] from the same tables,
                    // so the flat 64-entry result is identical.
                    self.sig_inc.copy_from_slice(&SIG_INC_8X8[0]);
                    self.last_inc.copy_from_slice(&LAST_INC_8X8);
                    self.scan.copy_from_slice(&SCAN_8X8_CABAC[0]);
                }
                for i8x8 in 0..4 {
                    // C decode_intra8x8(samples, stride, Intra8x8Modes[mode][unavail])
                    if m.mbIsInterFlag == 0 {
                        let p = unsafe {
                            (self.samples_mb[iycbcr] as *const u8)
                                .add(Y444[i8x8 * 4] as usize * stride + X444[i8x8 * 4] as usize)
                        };
                        let off = self.plane_off(p);
                        let variant = INTRA8X8_MODES[m.Intra4x4PredMode[i8x8 * 4] as usize]
                            [self.unavail4x4[i8x8 * 5] as usize];
                        let buf = self.plane_slice();
                        intra8x8(buf, off, stride, variant);
                    }
                    if m.bits[0] & (1 << BIT_8X8[i8x8] as usize) != 0 {
                        let pix = unsafe {
                            self.samples_mb[iycbcr]
                                .add(Y444[i8x8 * 4] as usize * stride + X444[i8x8 * 4] as usize)
                        };
                        if !self.is_cabac {
                            for i4x4 in 0..4 {
                                self.scan[..16].copy_from_slice(
                                    &SCAN_8X8_CAVLC[0][i4x4 * 16..(i4x4 + 1) * 16],
                                );
                                let base =
                                    unsafe { (self.mb() as *const u8).add(32 + iycbcr * 16) };
                                let n_a = unsafe {
                                    *base.offset(self.a4x4_int8[i8x8 * 4 + i4x4] as isize)
                                } as i32;
                                let n_b = unsafe {
                                    *base.offset(self.b4x4_int8[i8x8 * 4 + i4x4] as isize)
                                } as i32;
                                self.parse_residual_block_4x4_cavlc(
                                    0,
                                    (iycbcr * 16 + i8x8 * 4 + i4x4) as i32,
                                    n_a,
                                    n_b,
                                );
                            }
                            // C add_idct8x8(ctx, iYCbCr, samples)
                            unsafe {
                                add_idct8x8(
                                    &mut self.c,
                                    self.qp_s[iycbcr],
                                    &self.ws8[iycbcr * 2 + (m.mbIsInterFlag as usize)],
                                    pix,
                                    stride,
                                );
                            }
                        } else if self.chroma_array_type < 3
                            || self.cabac.get_ae(
                                &mut self.bits,
                                (self.ctx_idx_offsets[0]
                                    + ((m.bits[1] >> INC_8X8[iycbcr * 4 + i8x8] as usize) & 3)
                                        as i16) as usize,
                            ) != 0
                        {
                            m.bits[1] |= 1 << BIT_8X8[iycbcr * 4 + i8x8] as usize;
                            // C `mb->nC_s[iYCbCr * 4 + i8x8] = 0x01010101`
                            for j in 0..4 {
                                unsafe {
                                    (*self.mb()).nC[iycbcr * 16 + i8x8 * 4 + j] = 1;
                                }
                            }
                            self.parse_residual_block_8x8_cabac(0, 63);
                            // C add_idct8x8(ctx, iYCbCr, samples)
                            unsafe {
                                add_idct8x8(
                                    &mut self.c,
                                    self.qp_s[iycbcr],
                                    &self.ws8[iycbcr * 2 + (m.mbIsInterFlag as usize)],
                                    pix,
                                    stride,
                                );
                            }
                        }
                    }
                }
            }

            // C `CAJUMP(parse_chroma_residual)`: tail call exits the function.
            if self.chroma_array_type < 3 {
                self.parse_chroma_residual();
                return;
            }
        }
    }

    /// C `parse_inter_residual` — entry to residual parsing in Inter MBs
    /// (coded_block_pattern + transform_size_8x8_flag, parsed in a different
    /// order than for Intra).
    fn parse_inter_residual(&mut self) {
        const ME_INTER: [u8; 48] = [
            0, 1, 32, 8, 4, 128, 2, 40, 36, 136, 132, 172, 174, 44, 168, 164, 140, 12, 160, 173,
            42, 38, 138, 134, 34, 10, 6, 130, 46, 170, 166, 142, 33, 9, 5, 129, 41, 37, 137, 133,
            45, 169, 165, 141, 13, 161, 14, 162,
        ];
        let m = unsafe { &mut *self.mb() };
        self.parse_coded_block_pattern(&ME_INTER);
        if (m.bits[0] & 0xac) != 0 && self.transform_8x8_mode_flag != 0 {
            m.f.transform_size_8x8_flag = if !self.is_cabac {
                self.bits.get_u1() as i8
            } else {
                self.cabac.get_ae(
                    &mut self.bits,
                    399 + self.inc.transform_size_8x8_flag as usize,
                ) as i8
            };
        }
        if self.is_cabac {
            self.nc_inc = [[0; 16]; 3];
        }
        self.parse_NxN_residual();
    }

    /// C `parse_I_mb` (mb_type, transform_size_8x8_flag, intra modes, chroma
    /// mode, coded_block_pattern, PCM) with tail calls into the residual
    /// parsers.
    fn parse_I_mb(&mut self, mb_type_or_ctxIdx: i32) {
        const ME_INTRA: [u8; 48] = [
            174, 173, 172, 0, 45, 169, 165, 141, 44, 168, 164, 140, 46, 170, 166, 142, 1, 40, 36,
            136, 132, 41, 37, 137, 133, 42, 38, 138, 134, 32, 8, 4, 128, 33, 9, 5, 129, 12, 160,
            13, 161, 2, 34, 10, 6, 130, 14, 162,
        ];
        let m = unsafe { &mut *self.mb() };

        // Intra-specific initialisations
        if self.is_cabac {
            self.nc_inc = [[0; 16]; 3];
            if self.unavail4x4[0] & 1 != 0 {
                m.bits[1] |= 0x111111;
                const INC0: [i8; 16] = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0];
                const INC1: [i8; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                for i in 0..16 {
                    self.nc_inc[0][i] += INC0[i];
                    self.nc_inc[1][i] += INC1[i];
                }
                // C `ctx->inc.coded_block_flags_16x16_s |= 0x010101`
                for i in 0..3 {
                    self.inc.coded_block_flags_16x16[i] |= 1;
                }
            }
            if self.unavail4x4[0] & 2 != 0 {
                m.bits[1] |= 0x424242;
                const INC: [i8; 8] = [2, 2, 0, 0, 2, 2, 0, 0];
                for (i, inc) in INC.iter().enumerate() {
                    self.nc_inc[0][i] += *inc;
                    self.nc_inc[1][i] += *inc;
                }
                // C `ctx->inc.coded_block_flags_16x16_s |= 0x020202`
                for i in 0..3 {
                    self.inc.coded_block_flags_16x16[i] |= 2;
                }
            }
        }
        m.mbIsInterFlag = 0;
        m.f.inter_eqs = 0x1b5fbbff_u32.to_le_bytes();
        m.refIdx = [-1; 8];
        m.refPic = [-1; 8];
        for i in 0..64 {
            m.mvs[i] = 0;
        }

        // I_NxN
        let is_nxn = if !self.is_cabac {
            mb_type_or_ctxIdx == 0
        } else {
            self.cabac
                .get_ae(&mut self.bits, mb_type_or_ctxIdx as usize)
                == 0
        };
        if is_nxn {
            if self.is_cabac {
                m.f.mb_type_I_NxN = 1;
            }
            let ts88: i32 = if self.pps_transform_8x8_mode_flag != 0 {
                if !self.is_cabac {
                    self.bits.get_u1() as i32
                } else {
                    self.cabac.get_ae(
                        &mut self.bits,
                        399 + self.inc.transform_size_8x8_flag as usize,
                    ) as i32
                }
            } else {
                0
            };
            m.Intra4x4PredMode = [-2; 16];
            if ts88 != 0 {
                m.f.transform_size_8x8_flag = ts88 as i8;
                // C stores mode * 0x01010101 per 4x4 group (i32)
                for i in 0..4 {
                    let mode = self.parse_intraNxN_pred_mode(i * 4);
                    for j in 0..4 {
                        m.Intra4x4PredMode[i * 4 + j] = mode as i8;
                    }
                }
            } else {
                for i in 0..16 {
                    m.Intra4x4PredMode[i] = self.parse_intraNxN_pred_mode(i) as i8;
                }
            }
            self.parse_intra_chroma_pred_mode();
            self.parse_coded_block_pattern(&ME_INTRA);
            self.parse_NxN_residual();
        } else {
            // Intra_16x16 vs I_PCM
            let is_16x16 = if !self.is_cabac {
                mb_type_or_ctxIdx < 25
            } else {
                !self.cabac.terminate(&mut self.bits)
            };
            if is_16x16 {
                let mode: i32 = if !self.is_cabac {
                    let mut mt = mb_type_or_ctxIdx - 1;
                    // zeroes ref_idx_nz as byproduct
                    m.bits[0] = if mt > 11 { 0xac } else { 0 };
                    mt = if mt > 11 { mt - 12 } else { mt };
                    m.f.CodedBlockPatternChromaDC = (mt > 3) as i8;
                    m.f.CodedBlockPatternChromaAC = (mt >> 3) as i8;
                    mt & 3
                } else {
                    let mut ctx_idx = mb_type_or_ctxIdx.max(5);
                    // zeroes ref_idx_nz as byproduct
                    m.bits[0] = if self.cabac.get_ae(&mut self.bits, (ctx_idx + 1) as usize) != 0 {
                        0xac
                    } else {
                        0
                    };
                    let cbp_dc = self.cabac.get_ae(&mut self.bits, (ctx_idx + 2) as usize);
                    ctx_idx = ctx_idx.max(6);
                    if cbp_dc != 0 {
                        m.f.CodedBlockPatternChromaDC = 1;
                        m.f.CodedBlockPatternChromaAC =
                            self.cabac.get_ae(&mut self.bits, (ctx_idx + 2) as usize) as i8;
                    }
                    (self.cabac.get_ae(&mut self.bits, (ctx_idx + 3) as usize) << 1) as i32
                        + self
                            .cabac
                            .get_ae(&mut self.bits, (ctx_idx + 3).max(10) as usize)
                            as i32
                };
                m.Intra4x4PredMode = [2; 16];
                // C decode_intra16x16(samples_mb[0], stride[0], ...)
                {
                    let off = self.plane_off(self.samples_mb[0] as *const u8);
                    let stride = self.stride[0] as usize;
                    let variant = INTRA16X16_MODES[mode as usize][self.unavail4x4[0] as usize & 3];
                    let buf = self.plane_slice();
                    intra16x16(buf, off, stride, variant);
                }
                self.parse_intra_chroma_pred_mode();
                self.parse_Intra16x16_residual();
            } else {
                // I_PCM
                if !self.is_cabac {
                    // byte-align the CAVLC cache before the raw PCM samples
                    let bits = (63 - self.bits.lsb.trailing_zeros()) & 7;
                    self.bits.msb = shld(self.bits.lsb, self.bits.msb, bits);
                    self.bits.lsb <<= bits;
                }
                self.mb_qp_delta_nz = 0;
                // C `mb->f.v |= flags_twice.v` (ChromaDC, ChromaAC, cbf_16x16)
                m.f.CodedBlockPatternChromaDC |= 1;
                m.f.CodedBlockPatternChromaAC |= 1;
                for i in 0..3 {
                    m.f.coded_block_flags_16x16[i] |= 1;
                }
                // only mb->QP_s here (C does not touch ctx->t.QP_s for PCM)
                m.QP = [0, self.qp_c[0][0] as u8, self.qp_c[1][0] as u8, 0];
                m.bits[0] = 0xac;
                m.bits[1] = 0xacacac;
                let pcm_val: i8 = if !self.is_cabac { 16 } else { 1 };
                for i in 0..48 {
                    m.nC[i] = pcm_val;
                }
                m.Intra4x4PredMode = [2; 16];

                // PCM samples (C writes big-endian u32/u16 chunks in place).
                let mut mb_width: i32 = 16;
                let mut y: i32 = 16;
                for iycbcr in 0..3 {
                    let bd = self.bit_depth[iycbcr];
                    let mut p = self.samples_mb[iycbcr];
                    for _ in 0..y {
                        if bd == 8 {
                            unsafe {
                                *(p.cast::<u32>()) = (self.bits.get_uv(32) as u32).to_be();
                                *p.add(4).cast::<u32>() = (self.bits.get_uv(32) as u32).to_be();
                                if mb_width == 16 {
                                    *p.add(8).cast::<u32>() = (self.bits.get_uv(32) as u32).to_be();
                                    *p.add(12).cast::<u32>() =
                                        (self.bits.get_uv(32) as u32).to_be();
                                }
                            }
                        } else {
                            for x in 0..mb_width {
                                unsafe {
                                    *p.cast::<u16>().add(x as usize) =
                                        self.bits.get_uv(bd as usize) as u16;
                                }
                            }
                        }
                        p = unsafe { p.add(self.stride[iycbcr.min(1)] as usize) };
                    }
                    mb_width = if self.chroma_array_type < 3 { 8 } else { 16 };
                    y = [0, 8, 16, 16][self.chroma_array_type as usize];
                }
                if self.is_cabac {
                    let _ = self.cabac.start(&mut self.bits);
                }
            }
        }
    }

    /// C `parse_slice_data`: the macroblock loop over a slice. Initialises each
    /// macroblock (neighbours, `inc`, QP, unavailability), dispatches to
    /// `parse_{I/P/B}_mb`, emits the per-MB record (C `print_mb`), and advances
    /// the pointers. Deblocking is skipped: C's `deblock_mb` only writes
    /// `filter_edges`/samples after the record is emitted, so a no-deblock port
    /// is byte-equivalent for record capture.
    ///
    /// Precondition (C `initialize_context` + `sw264_decode_slice`): `mb_pos`,
    /// `curr_mb_addr`, `mbx`, `mby`, `mb_col` at the slice's first macroblock
    /// (mb_col points at the current buffer for P/I, the collocated buffer for
    /// B); offset arrays initialised; CAVLC: `mb_skip_run = -1`; CABAC:
    /// cabac start/init done and `mb_qp_delta_nz = 0`.
    pub(crate) fn parse_slice_data(&mut self) {
        // C `block_unavailability` (indexed by unavail16x16).
        const BLOCK_UNAVAILABILITY: [[i8; 16]; 16] = [
            [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [1, 0, 9, 4, 0, 0, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [6, 14, 0, 4, 14, 10, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [7, 14, 9, 4, 14, 10, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [0, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [1, 0, 9, 4, 0, 4, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [6, 14, 0, 4, 14, 14, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [7, 14, 9, 4, 14, 14, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [8, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [9, 0, 9, 4, 0, 0, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [14, 14, 0, 4, 14, 10, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [15, 14, 9, 4, 14, 10, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [8, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [9, 0, 9, 4, 0, 4, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
            [14, 14, 0, 4, 14, 14, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4],
            [15, 14, 9, 4, 14, 14, 0, 4, 9, 0, 9, 4, 0, 4, 0, 4],
        ];
        const UNAVAIL: *const RustMb = &UNAVAIL_MB as *const RustMb;
        let width = self.pic_width_in_mbs as i32;
        let smb = std::mem::size_of::<RustMb>() as i32; // 304
        let mut end_of_slice_flag = false;

        loop {
            let mbp = self.mb();

            // update flip_bit atomically to signal mb is a priori decoded,
            // otherwise end the slice
            let prev_recovery_bits = unsafe { (*mbp).recovery_bits };
            unsafe {
                (*mbp).recovery_bits = self.frame_flip_bit;
            }
            if prev_recovery_bits == self.frame_flip_bit {
                return;
            }

            // set and reset neighbouring pointers depending on their
            // availability. C: `(mbx == pic_width_in_mbs - 1) << 2` — the
            // comparison is parenthesised, so this is `((mbx == (w-1)) << 2)`
            // (set bit 2 when on the right edge), NOT `mbx == ((w-1) << 2)`.
            let mut unavail16x16 = (if self.mbx == 0 { 9 } else { 0 })
                | ((((self.mbx as i32) == (width - 1)) as i32) << 2)
                | (if self.mby == 0 { 14 } else { 0 });
            let mut filter_edges: i32 = 4 | 3 & !unavail16x16;
            let mut mb_a = unsafe { mbp.offset(-1) };
            let mut mb_b = unsafe { mb_a.offset(-width as isize) };
            let mut mb_c = unsafe { mb_b.offset(1) };
            let mut mb_d = unsafe { mb_b.offset(-1) };
            let decoded = self.curr_mb_addr - self.first_mb_in_slice as i32;
            if decoded <= width + 1 {
                if decoded == 1 {
                    // A becomes available
                    self.a4x4_int8[0] = (5 - smb) as i16;
                    self.a4x4_int8[2] = (7 - smb) as i16;
                    self.a4x4_int8[8] = (13 - smb) as i16;
                    self.a4x4_int8[10] = (15 - smb) as i16;
                    self.absmvd_a[0] = (10 - smb) as i16;
                    self.absmvd_a[2] = (14 - smb) as i16;
                    self.absmvd_a[8] = (26 - smb) as i16;
                    self.absmvd_a[10] = (30 - smb) as i16;
                    let smb_i32 = smb >> 2; // i16 elements per macroblock
                    self.mvs_a[0] = (5 - smb_i32) as i16;
                    self.mvs_a[2] = (7 - smb_i32) as i16;
                    self.mvs_a[8] = (13 - smb_i32) as i16;
                    self.mvs_a[10] = (15 - smb_i32) as i16;
                    self.mvs_d[2] = 5 - smb_i32;
                    self.mvs_d[8] = 7 - smb_i32;
                    self.mvs_d[10] = 13 - smb_i32;
                    if self.chroma_array_type == 1 {
                        self.acbcr_int8[0] = (1 - smb) as i16;
                        self.acbcr_int8[2] = (3 - smb) as i16;
                        self.acbcr_int8[4] = (5 - smb) as i16;
                        self.acbcr_int8[6] = (7 - smb) as i16;
                    }
                } else if decoded == 0 {
                    // A is unavailable
                    mb_a = UNAVAIL as *mut RustMb;
                    unavail16x16 |= 1;
                    filter_edges &= !(self.disable_deblocking_filter_idc as i32 >> 1); // impacts only bit 0
                }
                if decoded == width + 1 {
                    // D becomes available
                    let offD_int32 = ((width + 2) * smb) >> 2;
                    self.mvs_d[0] = 15 - offD_int32;
                } else {
                    // D is unavailable
                    mb_d = UNAVAIL as *mut RustMb;
                    unavail16x16 |= 8;
                    if decoded == width {
                        // B becomes available
                        let offB_int8 = (width + 1) * smb;
                        let offB_int32 = offB_int8 >> 2;
                        self.b4x4_int8[0] = 10 - offB_int8;
                        self.b4x4_int8[1] = 11 - offB_int8;
                        self.b4x4_int8[4] = 14 - offB_int8;
                        self.b4x4_int8[5] = 15 - offB_int8;
                        self.absmvd_b[0] = 20 - offB_int8;
                        self.absmvd_b[1] = 22 - offB_int8;
                        self.absmvd_b[4] = 28 - offB_int8;
                        self.absmvd_b[5] = 30 - offB_int8;
                        self.mvs_b[0] = 10 - offB_int32;
                        self.mvs_b[1] = 11 - offB_int32;
                        self.mvs_b[4] = 14 - offB_int32;
                        self.mvs_b[5] = 15 - offB_int32;
                        self.mvs_c[0] = 11 - offB_int32;
                        self.mvs_c[1] = 14 - offB_int32;
                        self.mvs_c[4] = 15 - offB_int32;
                        self.mvs_d[1] = 10 - offB_int32;
                        self.mvs_d[4] = 11 - offB_int32;
                        self.mvs_d[5] = 14 - offB_int32;
                        if self.chroma_array_type == 1 {
                            self.bcbcr_int8[0] = 2 - offB_int8;
                            self.bcbcr_int8[1] = 3 - offB_int8;
                            self.bcbcr_int8[4] = 6 - offB_int8;
                            self.bcbcr_int8[5] = 7 - offB_int8;
                        }
                    } else {
                        // B is unavailable
                        mb_b = UNAVAIL as *mut RustMb;
                        unavail16x16 |= 2;
                        filter_edges &= !(self.disable_deblocking_filter_idc as i32); // impacts only bit 1
                        if decoded == width - 1 {
                            // C becomes available
                            let offC_int32 = (width * smb) >> 2;
                            self.mvs_c[5] = 10 - offC_int32;
                        } else {
                            // C is unavailable
                            mb_c = UNAVAIL as *mut RustMb;
                            unavail16x16 |= 4;
                        }
                    }
                }
            }
            self.mb_a = mb_a as *const RustMb;
            self.mb_b = mb_b as *const RustMb;
            self.mb_c = mb_c as *const RustMb;
            self.mb_d = mb_d as *const RustMb;

            // initialize common macroblock values
            let a_mb = unsafe { *mb_a };
            let b_mb = unsafe { *mb_b };
            let unavail_row = BLOCK_UNAVAILABILITY[unavail16x16 as usize];
            self.unavail4x4[..16].copy_from_slice(&unavail_row);
            self.inc = flags_inc(a_mb.f, b_mb.f);
            unsafe {
                (*mbp).f = RustMbFlags::default();
                (*mbp).filter_edges = if self.disable_deblocking_filter_idc != 1 {
                    filter_edges as i8
                } else {
                    0
                };
                (*mbp).QP = self.qp_s;
            }
            if self.chroma_array_type == 1 {
                // FIXME 4:2:2 — _mm_shuffle_epi8 semantics: negative mask -> 0
                let v1 = shuffle_i8x16(
                    &unavail_row,
                    &[0, 4, 8, 12, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1],
                );
                self.unavail4x4[16..32].copy_from_slice(&v1);
                let a_bits = ((a_mb.bits[1] as u64) << 32) | a_mb.bits[0] as u64;
                let b_bits = ((b_mb.bits[1] as u64) << 32) | b_mb.bits[0] as u64;
                let bits_l = ((a_bits >> 3) & 0x0011_1111_0011_1111)
                    | ((b_bits >> 1) & 0x0042_4242_0042_4242);
                unsafe {
                    (*mbp).bits = [bits_l as u32, (bits_l >> 32) as u32];
                }
            }
            unsafe {
                (*mbp).nC = [0; 48];
            }

            // dispatch on slice type
            if self.slice_type == 0 {
                self.parse_P_mb();
            } else if self.slice_type == 1 {
                self.parse_B_mb();
            } else {
                let mb_type_or_ctxIdx: i32 = if !self.is_cabac {
                    self.bits.get_ue16(25) as i32
                } else {
                    5 - self.inc.mb_type_I_NxN as i32
                };
                self.parse_I_mb(mb_type_or_ctxIdx);
            }

            if self.is_cabac {
                end_of_slice_flag = self.cabac.terminate(&mut self.bits);
            }

            // C `print_mb`: append [CurrMbAddr][macroblock] while dump armed.
            if !self.mb_dump.is_null() && self.mb_dump_off + 4 + 304 <= self.mb_dump_cap {
                unsafe {
                    let p = self.mb_dump.add(self.mb_dump_off);
                    std::ptr::write(p.cast::<u32>(), self.curr_mb_addr as u32);
                    p.add(4).copy_from_nonoverlapping(mbp as *const u8, 304);
                }
                self.mb_dump_off += 4 + 304;
            }

            // C L1778-1795: deblock mbB (same column, previous row) while it is
            // still in cache, then point to the next macroblock. Net samples_mb
            // movement is +16/+8/+8 in both of C's branches.
            if self.curr_mb_addr - width == self.next_deblock_addr {
                self.next_deblock_addr += 1;
                self.deblock_mb_at(self.mb_pos - (width as usize + 1));
            }
            self.mb_pos += 1;
            unsafe {
                self.samples_mb[0] = self.samples_mb[0].add(16);
                self.samples_mb[1] = self.samples_mb[1].add(8);
                self.samples_mb[2] = self.samples_mb[2].add(8);
            }
            self.mbx += 1;
            self.curr_mb_addr += 1;
            unsafe {
                self.mb_col = self.mb_col.add(1);
            }
            if self.mbx >= self.pic_width_in_mbs {
                self.mb_pos += 1; // skip the empty macroblock at the edge
                unsafe {
                    self.mb_col = self.mb_col.add(1);
                }
                self.mby += 1;
                self.mbx = 0;
                // C L1802-1804: row-wrap stride correction.
                let s0 = (self.stride[0] as i64 * 16 - width as i64 * 16) as usize;
                let sc = (self.stride[1] as i64 * 8 - width as i64 * 8) as usize;
                unsafe {
                    self.samples_mb[0] = self.samples_mb[0].add(s0);
                    self.samples_mb[1] = self.samples_mb[1].add(sc);
                    self.samples_mb[2] = self.samples_mb[2].add(sc);
                }
                if self.mby >= self.pic_height_in_mbs {
                    return;
                }
            }

            // CAVLC: continue while a skip run remains or the rbsp trailing
            // bits have not been reached; CABAC: until end_of_slice_flag.
            let cont = if !self.is_cabac {
                self.mb_skip_run > 0 || !self.bits.rbsp_end(1)
            } else {
                !end_of_slice_flag
            };
            if !cont {
                break;
            }
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn mb_layout_matches_oracle() {
        assert_eq!(std::mem::size_of::<RustMbFlags>(), 16);
        assert_eq!(std::mem::size_of::<RustMb>(), 304);

        let off = |p: *const u8, base: *const u8| p as usize - base as usize;
        let m = RustMb::default();
        let b = &m as *const RustMb as *const u8;
        assert_eq!(off(&m.error_probability as *const _ as *const u8, b), 0);
        assert_eq!(off(&m.recovery_bits as *const _ as *const u8, b), 1);
        assert_eq!(off(&m.mbIsInterFlag as *const _ as *const u8, b), 2);
        assert_eq!(off(&m.filter_edges as *const _ as *const u8, b), 3);
        assert_eq!(off(m.QP.as_ptr(), b), 4);
        assert_eq!(off(m.bits.as_ptr() as *const u8, b), 8);
        assert_eq!(off(m.Intra4x4PredMode.as_ptr() as *const u8, b), 16);
        assert_eq!(off(m.nC.as_ptr() as *const u8, b), 32);
        assert_eq!(off(m.absMvd.as_ptr(), b), 80);
        assert_eq!(off(&m.f as *const _ as *const u8, b), 144);
        assert_eq!(off(m.refIdx.as_ptr() as *const u8, b), 160);
        assert_eq!(off(m.refPic.as_ptr() as *const u8, b), 168);
        assert_eq!(off(m.mvs.as_ptr() as *const u8, b), 176);

        let f = RustMbFlags::default();
        let fb = &f as *const RustMbFlags as *const u8;
        assert_eq!(
            off(&f.mb_field_decoding_flag as *const _ as *const u8, fb),
            0
        );
        assert_eq!(off(&f.mb_skip_flag as *const _ as *const u8, fb), 1);
        assert_eq!(off(&f.mb_type_I_NxN as *const _ as *const u8, fb), 2);
        assert_eq!(off(&f.mb_type_B_Direct as *const _ as *const u8, fb), 3);
        assert_eq!(
            off(&f.transform_size_8x8_flag as *const _ as *const u8, fb),
            4
        );
        assert_eq!(
            off(
                &f.intra_chroma_pred_mode_non_zero as *const _ as *const u8,
                fb
            ),
            5
        );
        assert_eq!(
            off(&f.CodedBlockPatternChromaDC as *const _ as *const u8, fb),
            6
        );
        assert_eq!(
            off(&f.CodedBlockPatternChromaAC as *const _ as *const u8, fb),
            7
        );
        assert_eq!(off(f.coded_block_flags_16x16.as_ptr(), fb), 8);
        assert_eq!(off(f.inter_eqs.as_ptr(), fb), 12);
    }

    #[test]
    fn serialize_deblock_roundtrip() {
        // Deterministic non-trivial values in the valid ranges.
        let mut m = RustMb {
            QP: [18, 20, 22, 0],
            mbIsInterFlag: 1,
            filter_edges: 3,
            nC: std::array::from_fn(|i| (i.is_multiple_of(7)) as i8 - 1), // mix of -1/0
            refIdx: std::array::from_fn(|i| (i as i8).wrapping_neg()),
            refPic: std::array::from_fn(|i| i as i8),
            mvs: std::array::from_fn(|i| ((i * 37) as i16).wrapping_neg()),
            ..Default::default()
        };
        m.f.inter_eqs = [0x1b, 0x5f, 0xbb, 0xff];
        m.f.transform_size_8x8_flag = 1;
        let parsed = crate::rust::deblock::parse_mb(&m.serialize_deblock());
        assert_eq!(m.serialize_deblock(), parsed.to_bytes());
    }
}
