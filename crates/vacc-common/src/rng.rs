//! Deterministic PRNGs for randomized differential tests (no external
//! dependencies).
//!
//! These streams historically existed as copy-pasted blocks in up to ten
//! test modules (`rust/tests.rs`, `rust/residual.rs`, `rust/deblock.rs`,
//! `hevc/{bitreader,transform,interpolation,intra_prediction,deblocking,
//! syntax_elements,sao}.rs`). The exact state updates and getters are
//! pinned by the test vectors below — changing the streams INVALIDATES the
//! deterministic test matrices that call with fixed seeds.

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64 (Vigna & Wysecki). `state` is the incrementing arithmetic
/// sequence seed.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// `state = seed` verbatim.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// `state = seed.wrapping_mul(GAMMA).wrapping_add(1)` — the seeding
    /// variant used by the hevc kernel test modules.
    #[inline]
    pub fn seeded(seed: u64) -> Self {
        Self::new(seed.wrapping_mul(SPLITMIX64_GAMMA).wrapping_add(1))
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `next_u64() % n` — uniform in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// `i32` in `[0, n)`.
    #[inline]
    pub fn i32(&mut self, n: u64) -> i32 {
        self.below(n) as i32
    }
}

/// xorshift64* — used by the H.264 (`rust/`) slice-decode test matrices.
#[derive(Clone, Copy, Debug)]
pub struct XorShift64Star(pub u64);

impl XorShift64Star {
    /// Next 64-bit output.
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 31;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `next() % n` — uniform in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Low byte of `next()`.
    #[inline]
    pub fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

/// 64-bit LCG with the Numerical Recipes constants
/// (`a = 6364136223846793005`, `c = 1442695040888963407`) — used by the
/// H.264 `rust/{residual,deblock}.rs` tests.
#[derive(Clone, Copy, Debug)]
pub struct Lcg(pub u64);

impl Lcg {
    /// Advance one step; high 32 bits of the state.
    #[inline]
    pub fn next32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    /// Advance one step; high 31 bits of the state.
    #[inline]
    pub fn next64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// `next32() % n` — uniform in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        self.next32() % n
    }

    /// Low byte of `next64()`.
    #[inline]
    pub fn byte(&mut self) -> u8 {
        self.next64() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_streams_pinned() {
        // Stream prefixes pinned at the time the per-module copies were
        // replaced — any mismatch means a test matrix would change.
        let mut r = SplitMix64::new(1);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [
                0x910a2dec89025cc1,
                0xbeeb8da1658eec67,
                0xf893a2eefb32555e
            ]
        );
        let mut r = SplitMix64::new(11);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [0x50f5647d2380309d, 0x432a5cd27a6b13a1, 0xa356be306e9b126d]
        );
        let mut r = SplitMix64::seeded(42);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [
                0x3c821fbf59108163,
                0xa7ff0d388687ffb2,
                0xde70d1019fc66081
            ]
        );
        // Determinism: two identical seeds produce identical full streams.
        let (mut a, mut b) = (SplitMix64::new(0), SplitMix64::new(0));
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(SplitMix64::new(0).i32(1), 0);
    }

    #[test]
    fn xorshift64star_streams_pinned() {
        let mut r = XorShift64Star(0x5eed_cabac);
        assert_eq!(
            [r.next(), r.next(), r.next()],
            [0x65265b76f4e82424, 0x0b69c73a6e744c8d, 0x6da86aab62d50c2a]
        );
        let mut r = XorShift64Star(1);
        r.byte();
        assert!(r.0 != 1);
    }

    #[test]
    fn lcg_streams_pinned() {
        let mut r = Lcg(0x01dc_755e);
        assert_eq!(
            [r.next32(), r.next32(), r.next32()],
            [0xfaf03e19, 0xeb1feb6c, 0xc3f2cb92]
        );
        let mut r = Lcg(0xd3b1_0cc5_e55e);
        assert_eq!([r.next64(), r.next64()], [0x0195_cfe0, 0x3f0c_0ee2]);
    }
}
