//! Bit-depth-agnostic saturation/clipping helpers shared by the software
//! decoders.
//!
//! Mirrors `Clip3` (H.264 §B.2.4 / H.265 §5.6) and the fixed-point
//! saturations of the reconstruction paths.

/// `Clip3(min_val, max_val, x)` — ITU-T H.264 §B.2.4 / H.265 §5.6.
#[inline]
pub fn clip3<T: Ord>(min_val: T, max_val: T, x: T) -> T {
    x.clamp(min_val, max_val)
}

/// Saturate `x` to the signed 16-bit range.
#[inline]
pub fn saturate_i16(x: i32) -> i16 {
    x.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Saturate `x` to the unsigned 8-bit range.
#[inline]
pub fn saturate_u8(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip3_cases() {
        assert_eq!(clip3(0, 9, 4), 4);
        assert_eq!(clip3(0, 9, -1), 0);
        assert_eq!(clip3(0, 9, 10), 9);
        assert_eq!(clip3(i16::MIN as i32, i16::MAX as i32, -32769), -32768i16 as i32);
    }

    #[test]
    fn saturate_cases() {
        assert_eq!(saturate_i16(40000), i16::MAX);
        assert_eq!(saturate_i16(-40000), i16::MIN);
        assert_eq!(saturate_i16(12345), 12345);
        assert_eq!(saturate_u8(-1), 0);
        assert_eq!(saturate_u8(256), 255);
        assert_eq!(saturate_u8(128), 128);
    }
}
