//! Bit-level writing and RBSP escaping shared by the encode backends.
//!
//! The writer is MSB-first, matching H.264/HEVC header syntax. Encoders use it to
//! serialize SPS/PPS/slice headers and residuals; `escape_rbsp` inserts
//! emulation-prevention bytes so no `00 00 0x` (x <= 3) sequence appears in a NAL payload.
//!
//! Adapted from the FilmCraft project (`filmcraft-bitstream`, MIT OR Apache-2.0,
//! https://github.com/storytold/filmcraft).

/// MSB-first bit writer.
#[derive(Default, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the low `n` bits of `value`, MSB first.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let v = (value as u64) & ((1u64 << n) - 1);
        self.acc = (self.acc << n) | v;
        self.nbits += n;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.buf.push((self.acc >> self.nbits) as u8);
        }
        self.acc &= (1u64 << self.nbits) - 1;
    }

    pub fn write_bit(&mut self, b: bool) {
        self.write_bits(b as u32, 1);
    }

    /// Unsigned Exp-Golomb `ue(v)`.
    pub fn write_ue(&mut self, v: u32) {
        let x = v as u64 + 1;
        let len = 64 - x.leading_zeros();
        let zeros = len - 1;
        for _ in 0..zeros {
            self.write_bit(false);
        }
        if len > 32 {
            self.write_bits((x >> 32) as u32, len - 32);
            self.write_bits(x as u32, 32);
        } else {
            self.write_bits(x as u32, len);
        }
    }

    /// Signed Exp-Golomb `se(v)`.
    pub fn write_se(&mut self, v: i32) {
        let k = if v > 0 { (v as u32) * 2 - 1 } else { (-(v as i64) as u32) * 2 };
        self.write_ue(k);
    }

    /// Total bits written so far (including the partial byte).
    pub fn bit_len(&self) -> usize {
        self.buf.len() * 8 + self.nbits as usize
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Pad with zero bits to a byte boundary.
    pub fn align_zero(&mut self) {
        if self.nbits > 0 {
            self.write_bits(0, 8 - self.nbits);
        }
    }

    /// `rbsp_trailing_bits()`: a one bit then zeros to alignment.
    pub fn rbsp_trailing(&mut self) {
        self.write_bit(true);
        self.align_zero();
    }

    /// Append whole bytes (bit-aligned only; otherwise written bit by bit).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        if self.nbits == 0 {
            self.buf.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.write_bits(b as u32, 8);
            }
        }
    }

    /// Consume the writer, returning the byte buffer (zero-padded to a byte boundary).
    pub fn finish(mut self) -> Vec<u8> {
        self.align_zero();
        self.buf
    }

    /// The whole bytes written so far (the partial trailing byte is not included).
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }
}

/// Insert emulation-prevention bytes so no `00 00 0x` (x <= 3) sequence appears.
pub fn escape_rbsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 64);
    let mut zeros = 0;
    for &b in data {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Remove emulation-prevention bytes (`00 00 03` -> `00 00`) from a NAL payload.
pub fn unescape_rbsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_and_alignment() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010_1100, 8);
        assert_eq!(w.bytes(), &[0b1010_1100]);
        w.write_bits(0b0101, 4);
        assert!(!w.is_byte_aligned());
        assert_eq!(w.bit_len(), 12);
        w.align_zero();
        assert_eq!(w.finish(), vec![0b1010_1100, 0b0101_0000]);
    }

    #[test]
    fn exp_golomb_known() {
        // 1 | 010 | 011 | 00100 | 00101 -> 0,1,2,3,4
        let mut w = BitWriter::new();
        for v in 0..5 {
            w.write_ue(v);
        }
        // 1 | 010 | 011 | 00100 | 00101 -> 17 bits
        assert_eq!(w.finish(), vec![0xA6, 0x42, 0x80]);

        let mut w = BitWriter::new();
        w.write_se(-2); // se(-2) = ue(4) = 00101
        assert_eq!(w.finish(), vec![0b00101000]);
    }

    #[test]
    fn rbsp_trailing() {
        let mut w = BitWriter::new();
        w.write_bits(0b10101, 5);
        w.rbsp_trailing();
        assert_eq!(w.finish(), vec![0b1010_1100]);
    }

    #[test]
    fn emulation_prevention() {
        let raw = [0, 0, 1, 0, 0, 0, 5, 0, 0, 3];
        let esc = escape_rbsp(&raw);
        assert_eq!(esc, vec![0, 0, 3, 1, 0, 0, 3, 0, 5, 0, 0, 3, 3]);
        assert_eq!(unescape_rbsp(&esc), raw);
    }

    #[test]
    fn escape_never_creates_pattern() {
        // worst case: all zero bytes
        let data = vec![0u8; 300];
        let e = escape_rbsp(&data);
        for w in e.windows(3) {
            assert!(!(w[0] == 0 && w[1] == 0 && w[2] < 3));
        }
        assert_eq!(unescape_rbsp(&e), data);
    }
}
