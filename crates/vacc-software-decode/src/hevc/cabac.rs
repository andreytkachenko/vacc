//! Port of `hevc/decoding/cabac.{h,cpp}` — CABAC arithmetic decoder, spec §9.3.
//!
//! State machine over a [`BitstreamReader`]: context-coded bins
//! (`decode_decision`), bypass bins (`decode_bypass`, `decode_bypass_bins`),
//! terminate (`decode_terminate`), renormalization (§9.3.4.3.3), and context
//! initialization (§9.3.1.1). Tables come from `cabac_tables`.

use crate::hevc::bitreader::BitstreamReader;
use crate::hevc::cabac_tables::{
    CABAC_INIT_VALUES, NUM_CABAC_CONTEXTS, RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS,
};
use crate::hevc::types::clip3;

/// Single CABAC context (AD-005).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CabacContext {
    pub p_state_idx: u8,
    pub val_mps: u8,
}

/// CABAC arithmetic decoder engine.
pub struct CabacEngine<'bs> {
    contexts: [CabacContext; NUM_CABAC_CONTEXTS],
    range: u16,
    offset: u16,
    bs: &'bs mut BitstreamReader<'bs>,
    /// Test-only bin counter for differential tracing.
    dbg_bin_count: u32,
}

impl<'bs> CabacEngine<'bs> {
    pub fn new(bs: &'bs mut BitstreamReader<'bs>) -> Self {
        Self {
            contexts: [CabacContext::default(); NUM_CABAC_CONTEXTS],
            range: 0,
            offset: 0,
            bs,
            dbg_bin_count: 0,
        }
    }

    /// §9.3.4.3.1 — Initialize the arithmetic decoder.
    /// Must be called at the start of each independent slice segment.
    pub fn init_decoder(&mut self) {
        self.range = 510;
        self.offset = self.bs.read_bits(9).expect("cabac init: 9 bits") as u16;
    }

    /// §9.3.1.1 — Initialize all contexts for a slice type and QP.
    /// `slice_type`: 0=B, 1=P, 2=I (SliceType enum).
    pub fn init_contexts(&mut self, slice_type: i32, slice_qp_y: i32, cabac_init_flag: bool) {
        // Map SliceType enum (B=0, P=1, I=2) to init table index (I=0, P=1, B=2)
        let mut init_type = match slice_type {
            2 => 0, // I -> table index 0
            1 => 1, // P -> table index 1
            0 => 2, // B -> table index 2
            _ => 0,
        };

        // cabac_init_flag permutation (§9.2.1.1):
        // P slice with cabac_init_flag=1 uses B init values and vice versa.
        if cabac_init_flag {
            if init_type == 1 {
                init_type = 2; // P uses B
            } else if init_type == 2 {
                init_type = 1; // B uses P
            }
        }

        let qp = clip3(0, 51, slice_qp_y);

        for (i, init_row) in CABAC_INIT_VALUES.iter().enumerate() {
            let init_value = init_row[init_type];
            let slope = (init_value >> 4) as i32 * 5 - 45;
            let offset = (((init_value & 15) as i32) << 3) - 16;
            // C++: ((slope * qp) >> 4) + offset — parens required (Rust `+` binds
            // tighter than `>>`).
            let pre_ctx_state = clip3(1, 126, ((slope * qp) >> 4) + offset);

            if pre_ctx_state <= 63 {
                self.contexts[i] = CabacContext {
                    p_state_idx: (63 - pre_ctx_state) as u8,
                    val_mps: 0,
                };
            } else {
                self.contexts[i] = CabacContext {
                    p_state_idx: (pre_ctx_state - 64) as u8,
                    val_mps: 1,
                };
            }
        }
    }

    /// §9.3.4.3.2 — Decode a context-coded bin.
    pub fn decode_decision(&mut self, ctx_idx: usize) -> i32 {
        let p_state_idx = self.contexts[ctx_idx].p_state_idx;
        let val_mps = self.contexts[ctx_idx].val_mps;

        let lps_range = RANGE_TAB_LPS[p_state_idx as usize][(self.range >> 6) as usize & 3] as u16;
        self.range -= lps_range;

        let bin_val = if self.offset >= self.range {
            // LPS path
            self.offset -= self.range;
            self.range = lps_range;
            if p_state_idx == 0 {
                self.contexts[ctx_idx].val_mps = 1 - val_mps;
            }
            self.contexts[ctx_idx].p_state_idx = TRANS_IDX_LPS[p_state_idx as usize];
            1 - val_mps as i32
        } else {
            // MPS path
            self.contexts[ctx_idx].p_state_idx = TRANS_IDX_MPS[p_state_idx as usize];
            val_mps as i32
        };

        self.renormalize();
        if crate::hevc::coding_tree::hevc_trace() && self.dbg_bin_count < 20000 {
            eprintln!("RUST decision ctx={} bin={} range={} offset={}",
                ctx_idx, bin_val, self.range, self.offset);
        }
        self.dbg_bin_count += 1;
        bin_val
    }

    /// §9.3.4.3.4 — Bypass decoding (branchless).
    pub fn decode_bypass(&mut self) -> i32 {
        self.offset = ((self.offset as u32) << 1 | self.bs.read_bit_fast()) as u16;

        // Branchless: avoid unpredictable branch on 50/50 bypass bins.
        let val = (self.offset >= self.range) as i32;
        self.offset -= self.range & (-(val) as i16 as u16);
        if crate::hevc::coding_tree::hevc_trace() && self.dbg_bin_count < 20000 {
            eprintln!("RUST bypass bin={}", val);
        }
        self.dbg_bin_count += 1;
        val
    }

    /// Decode multiple bypass bins, MSB first (batched bit read).
    pub fn decode_bypass_bins(&mut self, num_bins: i32) -> i32 {
        // Read all bits at once — eliminates N-1 refill checks.
        let bits = self.bs.read_bits_safe(num_bins as usize);

        let mut value = 0;
        for i in (0..num_bins).rev() {
            // `bits` is u32: for i >= 32 the bit is 0 (matches C++ uint32 truncation).
            let bit = if (i as u32) < 32 { (bits >> i) & 1 } else { 0 };
            self.offset = ((self.offset as u32) << 1 | bit) as u16;
            let val = (self.offset >= self.range) as i32;
            self.offset -= self.range & (-(val) as i16 as u16);
            value = (value << 1) | val;
            if crate::hevc::coding_tree::hevc_trace() && self.dbg_bin_count < 20000 {
                eprintln!("RUST bypass bin={}", val);
            }
        }
        self.dbg_bin_count += num_bins as u32;
        value
    }

    /// §9.3.4.3.5 — Terminate decoding (cold path).
    pub fn decode_terminate(&mut self) -> i32 {
        self.range -= 2;

        if self.offset >= self.range {
            return 1;
        }
        self.renormalize();
        0
    }

    /// §9.3.4.3.6 — Alignment prior to bypass decoding of coeff_sign_flag
    /// and coeff_abs_level_remaining. Sets range to 256.
    pub fn align_bypass(&mut self) {
        self.range = 256;
    }

    /// Context access.
    pub fn context(&self, ctx_idx: usize) -> &CabacContext {
        &self.contexts[ctx_idx]
    }
    pub fn context_mut(&mut self, ctx_idx: usize) -> &mut CabacContext {
        &mut self.contexts[ctx_idx]
    }

    /// Save/restore contexts (for WPP).
    pub fn save_contexts(&self, dst: &mut [CabacContext]) {
        dst.copy_from_slice(&self.contexts);
    }
    pub fn load_contexts(&mut self, src: &[CabacContext]) {
        self.contexts.copy_from_slice(src);
    }

    /// Raw pointer to the context array (WPP save path).
    pub fn contexts_ptr(&self) -> *const CabacContext {
        self.contexts.as_ptr()
    }

    /// Access to the underlying bitstream reader (raw-bit reads such as PCM
    /// sample access and byte alignment).
    pub fn bitstream(&mut self) -> &mut BitstreamReader<'bs> {
        self.bs
    }

    /// Debug accessors (mirror the C++ `dbg_*`).
    pub fn dbg_range(&self) -> u16 {
        self.range
    }
    pub fn dbg_offset(&self) -> u16 {
        self.offset
    }

    /// §9.3.4.3.3 — Renormalization (batched clz + bulk read).
    fn renormalize(&mut self) {
        if self.range >= 256 {
            return;
        }
        debug_assert!(self.range >= 1, "cabac range must be >= 1");
        // C++: `__builtin_clz(range) - 23` — for range ∈ [1, 255] this is
        // 9 - bit_length(range).
        let bit_len = 32 - (self.range as u32).leading_zeros();
        let shift = 9 - bit_len as i32;
        self.range <<= shift;
        let bits = self.bs.read_bits_safe(shift as usize);
        self.offset = ((self.offset as u32) << shift | bits) as u16;
    }
}
