//! Scalar port of edge264's CABAC engine (`c/src/edge264_bitstream.c`).
//!
//! The context state storage and tables are a folded re-encoding of the spec's
//! (pStateIdx, valMPS) pairs that is self-consistent with `TRANS_IDX` /
//! `RANGE_TAB_LPS` — port the arithmetic verbatim, do not "normalize" to the
//! spec's Tables 9-1..9-5. `init` reproduces the C vectorized form:
//! `min = Clip3(1, 126, (lo*s + hi*o) >> 4)` with `lo/hi` the two bytes of
//! `max(QP, 0) + 4096` (== the spec PreCtxState for QP <= 51), packed as
//! `min < 64 ? (4*(255-min)) & 0xFF : (4*min+1) & 0xFF`, plus `ctx[276] = 252`.
//!
//! After [`Cabac::start`], the shared `SliceBits` msb/lsb pair holds the CABAC
//! offset/range; [`Cabac::terminate`] may restore CAVLC semantics (LPS path).

use super::bits::{SliceBits, shld};
use super::tables;

/// CABAC context states (1024 contexts, one byte each; see module docs).
pub struct Cabac {
    pub(crate) ctx: [u8; 1024],
}

impl Default for Cabac {
    fn default() -> Self {
        Self::new()
    }
}

impl Cabac {
    pub fn new() -> Self {
        Self { ctx: [0; 1024] }
    }

    /// cabac_alignment: reclaim whole bytes from the CAVLC cache back to a byte
    /// boundary, then prime offset/range. Returns true if the alignment bits
    /// were not all ones (bitstream error). Port of C `cabac_start`.
    ///
    /// Requires at least one prior CAVLC read (`lsb` must be non-zero).
    pub fn start(&mut self, bits: &mut SliceBits) -> bool {
        let mut extra_bits = (63 - bits.lsb.trailing_zeros()) as i32;
        while extra_bits >= 8 {
            let mut i: u32 = 0;
            if bits.cpb as i64 <= bits.end {
                let a = bits.base + bits.cpb - 4;
                i = u32::from_be_bytes([
                    bits.buf[a],
                    bits.buf[a + 1],
                    bits.buf[a + 2],
                    bits.buf[a + 3],
                ]);
            }
            // C checks (big_endian32(i) & 0xffffff) == 3: the 3 bytes before CPB
            // form a 00 00 03 escape, so step back over it too.
            bits.cpb -= 1 + ((i & 0x00ff_ffff) == 3) as usize;
            extra_bits -= 8;
        }
        let shift = (extra_bits & 7) as u32;
        let ret = shift > 0 && ((bits.msb as i64) >> (64 - shift)) != -1;
        bits.msb = shld(bits.lsb, bits.msb, shift); // -> offset
        bits.lsb = 510u64 << 55; // -> range
        if bits.msb >= bits.lsb {
            bits.msb = bits.lsb; // protection against invalid bitstreams
        }
        ret
    }

    /// Initialize all context states from QP and cabac_init_idc (0=I, 1..3=P/B).
    /// Port of C `cabac_init` (see module docs for the scalar form).
    pub fn init(&mut self, qp: u8, idc: usize) {
        debug_assert!(idc < 4);
        let q = (qp as i16).max(0) + 4096;
        let lo = q & 0xff;
        let hi = q >> 8;
        for (i, &(s, o)) in tables::CABAC_CONTEXT_INIT[idc].iter().enumerate() {
            // maddxbs: lo*s + hi*o with i16 saturating add (exact in QP range).
            let sum = lo
                .wrapping_mul(s as i16)
                .saturating_add(hi.wrapping_mul(o as i16));
            let minv = ((sum >> 4) as i32).clamp(1, 126);
            self.ctx[i] = if minv < 64 {
                (4 * (255 - minv)) as u8
            } else {
                (4 * minv + 1) as u8
            };
        }
        self.ctx[276] = 252;
    }

    /// Decode one context-coded bin. Port of C `get_ae` / `get_ae_inline`.
    pub fn get_ae(&mut self, bits: &mut SliceBits, ctx_idx: usize) -> u32 {
        let mut state = self.ctx[ctx_idx] as u64;
        let mut range = bits.lsb;
        let mut offset = bits.msb;
        let shift = range.leading_zeros(); // [0..55]; range >= 256 here
        // C indexes `(uint8_t*)rangeTabLPS - 4` by `idx`; the second term is in
        // [4..7] so `idx - 4` is always in-bounds.
        // range >= 2^8 here, so (range << clz(range)) >> 61 is its top 3 bits,
        // i.e. [4..7]; with state & !3 in [0..252], idx is [4..259] and
        // idx - 4 always indexes RANGE_TAB_LPS (256 entries) in bounds.
        let idx = (state & !3) + ((range << (shift & 63)) >> 61);
        debug_assert!(
            (4..=259).contains(&idx),
            "idx={idx} range={:#x} state={} ctx_idx={}",
            range,
            state,
            ctx_idx
        );
        let range_lps = (tables::RANGE_TAB_LPS[(idx - 4) as usize] as u64) << (55 - shift);
        range -= range_lps;
        if offset >= range {
            state ^= 255;
            offset -= range;
            range = range_lps;
        }
        bits.lsb = range;
        bits.msb = offset;
        self.ctx[ctx_idx] = tables::TRANS_IDX[state as usize];
        let bin = (state & 1) as u32;
        if range < 256 {
            self.renorm_fixed(bits);
        }
        bin
    }

    /// Decode one bypass bin. Port of C `get_bypass`.
    pub fn get_bypass(&mut self, bits: &mut SliceBits) -> u32 {
        if bits.lsb < 512 {
            // renorm_bits(SIZE_BIT - 9 = 55): 6 bytes, shift by 55 & -8 = 48.
            let bytes = bits.get_bytes(6);
            bits.msb = shld(bytes, bits.msb, 48);
            bits.lsb <<= 48;
        }
        bits.lsb >>= 1;
        let bin = (bits.msb >= bits.lsb) as u32;
        if bin != 0 {
            bits.msb -= bits.lsb;
        }
        bin
    }

    /// cabac_terminate: returns true on the LPS path (which refills the CAVLC
    /// cache for trailing-bit reads). Port of C `cabac_terminate`.
    pub fn terminate(&mut self, bits: &mut SliceBits) -> bool {
        // C: int extra = SIZE_BIT - 9 - clz(range); [0..55] for valid states
        // (range >= 256). Kept signed; shifts are masked like x86 hardware.
        let extra = 55i32 - bits.lsb.leading_zeros() as i32;
        bits.lsb = bits.lsb.wrapping_sub(2u64 << (extra as u64 & 63));
        if bits.msb < bits.lsb {
            if bits.lsb < 256 {
                self.renorm_fixed(bits);
            }
            false
        } else {
            // C: (offset * 2 + 1) << (SIZE_BIT - 1 - (extra & -8))
            let shift = (63i32 - (extra & !7)) as u64 & 63;
            bits.msb = (bits.msb.wrapping_mul(2).wrapping_add(1)) << shift;
            bits.refill();
            true
        }
    }

    /// Renormalize so the top `n` bits of offset are valid, loading whole
    /// bytes; returns the leftover sub-byte count. Port of C `renorm_bits`.
    pub(crate) fn renorm_bits(&self, bits: &mut SliceBits, n: i32) -> u32 {
        let bytes = bits.get_bytes((n >> 3) as usize);
        let shift = (n & !7) as u32;
        bits.msb = shld(bytes, bits.msb, shift);
        bits.lsb <<= shift & 63;
        (n & 7) as u32
    }

    /// Renormalize to SIZE_BIT - 8 extra bits in offset. Port of C `renorm_fixed`.
    pub(crate) fn renorm_fixed(&self, bits: &mut SliceBits) {
        let bytes = bits.get_bytes(7); // SIZE_BIT / 8 - 1
        bits.msb = shld(bytes, bits.msb, 56); // SIZE_BIT - 8
        bits.lsb <<= 56;
    }
}
