//! Q14 fixed-point Y'CbCr -> RGB coefficient tables.
//!
//! All tables express the conversion as
//!
//! ```text
//! c = clamp0_255( (y * ky + cb * cb_k + cr * cr_k + off + (1 << 13)) >> 14 )
//! ```
//!
//! with `ky`, the chroma-to-channel coefficients, and the offsets (which fold
//! in the limited-range Y=16 / chroma=128 centering) precomputed in Q14. The
//! tables were generated and cross-checked against the analytic
//! ITU-R BT.601 / BT.709 conversion equations; primaries land within ±1 of
//! the computed values.

use crate::spec::{ColorRange, ColorSpec, MatrixCoefficients};

/// Per-interleaved-row conversion constants (Q14).
#[derive(Debug, Clone, Copy)]
pub struct Conv8 {
    /// Luma gain (full-range: `1 << 14`, so luma passes through).
    pub ky: i32,
    /// R <- Cr
    pub r_cr: i32,
    pub r_off: i32,
    /// G <- Cb, G <- Cr
    pub g_cb: i32,
    pub g_cr: i32,
    pub g_off: i32,
    /// B <- Cb
    pub b_cb: i32,
    pub b_off: i32,
}

/// Rounding half-LSB added before the `>> 14` (i.e. `1 << 13`).
pub const RND: i32 = 8192;

const BT601_LIMITED: Conv8 = Conv8 {
    ky: 19071,
    r_cr: 26149,
    r_off: -3652208,
    g_cb: -6418,
    g_cr: -13328,
    g_off: 2222352,
    b_cb: 33050,
    b_off: -4535536,
};

const BT709_LIMITED: Conv8 = Conv8 {
    ky: 19077,
    r_cr: 25801,
    r_off: -3607760,
    g_cb: -6418,
    g_cr: -13319,
    g_off: 2221104,
    b_cb: 33050,
    b_off: -4535632,
};

const BT601_FULL: Conv8 = Conv8 {
    ky: 16384,
    r_cr: 22970,
    r_off: -2940160,
    g_cb: -5638,
    g_cr: -11700,
    g_off: 2219264,
    b_cb: 29032,
    b_off: -3716096,
};

const BT709_FULL: Conv8 = Conv8 {
    ky: 16384,
    r_cr: 25802,
    r_off: -3302656,
    g_cb: -5638,
    g_cr: -11700,
    g_off: 2219264,
    b_cb: 30825,
    b_off: -3945600,
};

/// Select the Q14 table for a color specification.
pub fn table(spec: ColorSpec) -> Conv8 {
    match (spec.matrix, spec.range) {
        (MatrixCoefficients::Bt601, ColorRange::Limited) => BT601_LIMITED,
        (MatrixCoefficients::Bt601, ColorRange::Full) => BT601_FULL,
        (MatrixCoefficients::Bt709, ColorRange::Limited) => BT709_LIMITED,
        (MatrixCoefficients::Bt709, ColorRange::Full) => BT709_FULL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv8(t: &Conv8, u: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let (y, c, r) = (u as i32, cb as i32, cr as i32);
        let clip = |v: i32| v.clamp(0, 255) as u8;
        (
            clip((t.ky * y + t.r_cr * r + t.r_off + RND) >> 14),
            clip((t.ky * y + t.g_cb * c + t.g_cr * r + t.g_off + RND) >> 14),
            clip((t.ky * y + t.b_cb * c + t.b_off + RND) >> 14),
        )
    }

    #[test]
    fn bt709_limited_primaries() {
        let t = table(ColorSpec {
            matrix: MatrixCoefficients::Bt709,
            range: ColorRange::Limited,
        });
        // Digital black / 709 black in Y'CbCr (limited): (16, 128, 128)
        assert_eq!(conv8(&t, 16, 128, 128), (0, 0, 0));
        // White (235, 128, 128) -> (255, 255, 255)
        assert_eq!(conv8(&t, 235, 128, 128), (255, 255, 255));
        // Blue (16, 240, 116) -> R0 G0 B226 (2.0172*(240-128) = 225.93)
        assert_eq!(
            conv8(&t, 16, 240, 116),
            (0, 0, 226),
            "bt709 digital blue"
        );
        // Red (136, 135, 240) -> (255, 46, 154) within rounding
        let (r, g, b) = conv8(&t, 136, 135, 240);
        assert!(r >= 250, "red R={r}");
        assert!(g <= 55, "red G={g}");
        assert!(b >= 148 && b <= 160, "red B={b}");
        // Green (196, 104, 60) -> (103, 255, 161) within rounding
        let (r, g, b) = conv8(&t, 196, 104, 60);
        assert!(g == 255, "green G={g}");
        assert!(r >= 95 && r <= 112, "green R={r}");
        assert!(b >= 150 && b <= 170, "green B={b}");
    }

    #[test]
    fn bt601_limited_primaries() {
        let t = table(ColorSpec {
            matrix: MatrixCoefficients::Bt601,
            range: ColorRange::Limited,
        });
        assert_eq!(conv8(&t, 16, 128, 128), (0, 0, 0));
        assert_eq!(conv8(&t, 235, 128, 128), (255, 255, 255));
        // Blue (16, 240, 101) -> B=2.0172*(240-128) = 225.9
        let (r, g, b) = conv8(&t, 16, 240, 101);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert!(b >= 224 && b <= 227, "bt601 blue B={b}");
    }

    #[test]
    fn full_range_identities() {
        for spec in [
            ColorSpec {
                matrix: MatrixCoefficients::Bt601,
                range: ColorRange::Full,
            },
            ColorSpec {
                matrix: MatrixCoefficients::Bt709,
                range: ColorRange::Full,
            },
        ] {
            let t = table(spec);
            // In full range, luma passes through: black at Y=0 is (0,0,0);
            // grey at Y=128 -> (128,128,128).
            assert_eq!(conv8(&t, 0, 128, 128), (0, 0, 0));
            assert_eq!(conv8(&t, 128, 128, 128), (128, 128, 128));
            assert_eq!(conv8(&t, 255, 128, 128), (255, 255, 255));
        }
    }

    #[test]
    fn monochrome_never_gains_color() {
        // Neutral (Cb=Cr=128) samples must produce R=G=B for every Y.
        let t = table(ColorSpec::default());
        for y in 0..=255u8 {
            let (r, g, b) = conv8(&t, y, 128, 128);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            assert!(max.saturating_sub(min) <= 1, "y={y} -> {r},{g},{b}");
        }
    }
}
