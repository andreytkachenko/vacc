//! Scalar port of edge264's bitstream reader (`c/src/edge264_bitstream.c`).
//!
//! One 128-bit state (two `u64` caches) serves both CAVLC and CABAC, mirroring
//! the C union: before a CABAC start `msb`/`lsb` are the CAVLC msb/lsb caches;
//! after `Cabac::start` they are the CABAC offset/range. The bit-level
//! semantics (including emulation-prevention handling in `get_bytes` and the
//! "void" refills that advance `cpb` past `end` without reading) must stay
//! bit-identical to the C code — see the differential tests in `tests.rs`.
//!
//! The buffer is expected to have >= 32 zero bytes before and after the
//! payload (`get_bytes` loads a 16-byte window from `cpb - 2`; `cabac_start`
//! reads 4 bytes at `cpb - 4`).

/// Shared CAVLC/CABAC bit cache over `buf[base..base+len]`.
pub struct SliceBits<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) base: usize, // payload start in buf (>= 4; headroom before it)
    pub(crate) cpb: usize,  // next byte to load, relative to payload start (may exceed end)
    /// Exclusive end, relative to payload start. A stop sequence (00 00 {0,1,2})
    /// can move it back below 0 (C does the same with its `end` pointer).
    pub(crate) end: i64,
    /// CAVLC: msb_cache (may be 0 mid-stream; refill then uses masked shifts);
    /// CABAC: offset.
    pub(crate) msb: u64,
    /// CAVLC: lsb_cache; CABAC: range.
    pub(crate) lsb: u64,
}

/// Left-rotate the (msb, lsb) 128-bit pair left by `i` bits, returning the new
/// msb half. Mirrors the C `shld(l, h, i)`; on x86-64 that is the SHLD
/// instruction, which masks the count to 6 bits (so callers may pass any u32).
pub(crate) fn shld(l: u64, h: u64, i: u32) -> u64 {
    let i = i & 63;
    if i == 0 {
        h
    } else {
        (h << i) | (l >> (64 - i))
    }
}

impl<'a> SliceBits<'a> {
    /// Creates a reader over `buf[base..base+len]`, primed exactly like
    /// production (`vacc_sw264.c` sets `msb = 1 << 63` then calls refill), so
    /// the first 8 bytes of the payload are consumed here.
    pub fn new(buf: &'a [u8], base: usize, len: usize) -> Self {
        // cabac_start's byte-reclaim loop walks cpb back up to 14 and reads at
        // cpb-4; keep >= 18 bytes of headroom before the payload.
        debug_assert!(
            base >= 18,
            "cabac_start reads up to 18 bytes before the payload start"
        );
        debug_assert!(
            buf.len() >= base + len + 16,
            "get_bytes tail reads up to 15 bytes past the payload end"
        );
        let mut s = Self {
            buf,
            base,
            cpb: 0,
            end: len as i64,
            msb: 1 << 63, // prime (see production init in vacc_sw264.c)
            lsb: 0,
        };
        s.refill();
        s
    }

    /// Extract `nbytes` (<= 8) from the bitstream as big-endian, removing
    /// emulation-prevention bytes on the fly. Port of C `get_bytes`.
    pub(crate) fn get_bytes(&mut self, nbytes: usize) -> u64 {
        let diff = self.cpb as i64 - self.end;
        if diff >= 2 {
            // Already past the payload: advance without reading (C does the same).
            self.cpb += nbytes;
            return 0;
        }

        // 16-byte window starting at cpb-2, zero-padded past the real end.
        let start = self.base + self.cpb - 2;
        let n_real = (self.end - self.cpb as i64 + 2).clamp(0, 16) as usize;
        let mut v = [0u8; 16];
        v[..n_real].copy_from_slice(&self.buf[start..start + n_real]);

        // x = v shifted left by 2 bytes (top two zeroed), as in C `shr128(v, 2)`.
        let mut x = [0u8; 16];
        x[..14].copy_from_slice(&v[2..16]);

        // Mask of positions i where v[i]==0 && v[i+1]==0 && v[i+2]<=3.
        let mut test: u16 = 0;
        for i in 0..14 {
            if v[i] == 0 && v[i + 1] == 0 && v[i + 2] <= 3 {
                test |= 1 << i;
            }
        }
        // C (SIZE_BIT==64) only enters the scan when a candidate sits in the
        // low 8 bytes of the window (`(i64x2)to_fix)[0]`); higher candidates
        // are deferred to a later call (which can matter when escapes between
        // them shift CPB).
        if test & 0x00ff != 0 {
            let three: u16 = x
                .iter()
                .enumerate()
                .filter(|&(_, b)| *b == 3)
                .map(|(j, _)| 1u16 << j)
                .sum();
            // A 00 00 {0,1,2} sequence terminates the payload (trailing zeros).
            let stop = test & !three;
            if stop != 0 {
                let i = stop.trailing_zeros() as usize;
                self.end = self.cpb as i64 + i as i64 - 2;
                x[i..].fill(0);
            }
            // Remove each 00 00 03 escape (the 03 is at x position i); positions
            // drift down by one per removal, which the shifted `esc` mirrors.
            let mask: u16 = (1u16 << nbytes) - 1;
            let mut esc = test & three;
            while esc & mask != 0 {
                let i = esc.trailing_zeros() as usize;
                for k in i..15 {
                    x[k] = x[k + 1];
                }
                x[15] = 0;
                self.cpb += 1;
                esc = (esc & (esc - 1)) >> 1;
            }
        }

        self.cpb += nbytes;
        u64::from_be_bytes(x[0..8].try_into().unwrap())
    }

    /// Refill the CAVLC cache from 8 more bytes. Mirrors C `refill`.
    ///
    /// C computes `ctz(msb)` and shifts by expressions of it; when `msb == 0`
    /// (reachable mid-stream) that is ctz(0) = 64 on x86-64 (tzcnt) and the
    /// shift counts are masked to 6 bits by the hardware. Emulated verbatim.
    pub(crate) fn refill(&mut self) {
        let bytes = self.get_bytes(8);
        let trailing = self.msb.trailing_zeros(); // [0..64]; 64 iff msb == 0
        let tm = trailing & 63;
        let sh = ((63i32 - trailing as i32) as u32) & 63; // (63 - trailing), masked
        self.msb = (self.msb ^ (1 << tm)) | (bytes >> sh);
        self.lsb = (bytes.wrapping_mul(2).wrapping_add(1)) << tm;
    }

    // ------------------------------------------------------------------ CAVLC

    /// Read one bit. Port of C `get_u1`.
    pub fn get_u1(&mut self) -> u32 {
        let ret = (self.msb >> 63) as u32;
        self.msb = shld(self.lsb, self.msb, 1);
        self.lsb <<= 1;
        if self.lsb != 0 {
            ret
        } else {
            self.refill();
            ret
        }
    }

    /// Read `n` (1..=64) fixed-length bits from the top of the 128-bit cache.
    /// Port of C `get_uv` (the CAVLC level codes consume up to 48 bits).
    pub fn get_uv(&mut self, n: usize) -> u64 {
        let n = n as u32;
        debug_assert!((1..=64).contains(&n));
        let ret = if n == 64 {
            self.msb
        } else if n < 64 {
            self.msb >> (64 - n)
        } else {
            // n > 64: top n bits of the 128-bit cache
            (self.msb << (n - 64)) | (self.lsb >> (128 - n))
        };
        self.msb = shld(self.lsb, self.msb, n);
        self.lsb <<= n & 63;
        if self.lsb != 0 {
            ret
        } else {
            self.refill();
            ret
        }
    }

    /// True when the cache holds exactly the rbsp trailing bits: a single
    /// `trailing_bit` 1-bit (or nothing) followed by zeros, with 0-7 filler
    /// bits left before the payload end. Port of C `rbsp_end`.
    pub(crate) fn rbsp_end(&self, trailing_bit: u32) -> bool {
        let bits_to_end = (self.end - self.cpb as i64) * 8 + 127
            - self.lsb.trailing_zeros() as i64
            - trailing_bit as i64;
        self.msb == (trailing_bit as u64) << 63
            && (self.lsb & (self.lsb - 1)) == 0
            && (bits_to_end as u32) <= 7 * trailing_bit
    }

    /// Read an unsigned Exp-Golomb code, clamped to `upper`. Port of C `get_ue16`.
    ///
    /// `v` is [1..65]; v == 65 iff `msb < 2^32`, where C's shift counts
    /// (64 - v) and the lsb shift are masked to 6 bits by the hardware.
    pub fn get_ue16(&mut self, upper: u32) -> u32 {
        // C: v = clz(msb | 1 << 32) * 2 + 1 — the guard bit caps clz at 32,
        // so v is [1..65]; v == 65 iff msb < 2^32, where C's shift counts
        // (64 - v) and the lsb shift are masked to 6 bits by the hardware.
        let v = (self.msb | (1u64 << 32)).leading_zeros() * 2 + 1; // [1..65]
        let sh = ((64i32 - v as i32) as u32) & 63; // (64 - v), masked
        // C: minu((msb >> (SIZE_BIT - v)) - 1, upper) — the 64-bit expression
        // is truncated to 32-bit `unsigned` at the argument, then min'd there.
        let ret = (((self.msb >> sh).wrapping_sub(1)) as u32).min(upper);
        self.msb = shld(self.lsb, self.msb, v);
        self.lsb <<= v & 63;
        if self.lsb != 0 {
            ret
        } else {
            self.refill();
            ret
        }
    }

    /// Read a signed Exp-Golomb code, clamped to [lower, upper]. Port of C `get_se16`.
    ///
    /// Like `get_ue16`, `v` is [1..65] and the v == 65 case needs masked shifts.
    pub fn get_se16(&mut self, lower: i32, upper: i32) -> i32 {
        // Like get_ue16: C computes v = clz(msb | 1 << 32) * 2 + 1 in [1..65].
        let v = (self.msb | (1u64 << 32)).leading_zeros() * 2 + 1; // [1..65]
        let sh = ((64i32 - v as i32) as u32) & 63; // (64 - v), masked
        // C: `unsigned ue` is 32-bit — the 64-bit Exp-Golomb value is truncated
        // to its low 32 bits before the signed conversion.
        let ue = (self.msb >> sh).wrapping_sub(1) as u32;
        // C: (ue & 1) ? (ue >> 1) + 1 : -(ue >> 1) — 32-bit unsigned wraparound,
        // then bitcast to int at the max(int, int) call, then clamped.
        let s_u: u32 = if ue & 1 != 0 {
            (ue >> 1).wrapping_add(1)
        } else {
            (ue >> 1).wrapping_neg()
        };
        let ret = (s_u as i32).clamp(lower, upper);
        self.msb = shld(self.lsb, self.msb, v);
        self.lsb <<= v & 63;
        if self.lsb != 0 {
            ret
        } else {
            self.refill();
            ret
        }
    }

    // ------------------------------------------------------- diff-test state

    #[cfg(test)]
    pub(crate) fn cpb_off(&self) -> i64 {
        self.cpb as i64
    }
    #[cfg(test)]
    pub(crate) fn cache0(&self) -> u64 {
        self.msb
    }
    #[cfg(test)]
    pub(crate) fn cache1(&self) -> u64 {
        self.lsb
    }
}
