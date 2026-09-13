//! Port of `hevc/bitstream/bitstream_reader.{h,cpp}`.
//!
//! Bit-level reader over an RBSP buffer with a 64-bit cache (O(1) reads),
//! zero-padded reads for CABAC renormalization, and emulation-prevention
//! helpers. Field-update semantics are kept identical to the C++ original so
//! behavior matches bit-for-bit until the entropy layer is ported.

/// Error from a malformed or exhausted bitstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitstreamError {
    /// Fixed-length read past end of data.
    ReadPastEnd,
    /// Exp-Golomb code with more than 31 leading zeros.
    ExpGolombOverflow,
}

impl std::fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadPastEnd => write!(f, "read past end of bitstream"),
            Self::ExpGolombOverflow => write!(f, "Exp-Golomb overflow"),
        }
    }
}

impl std::error::Error for BitstreamError {}

pub type BitstreamResult<T> = Result<T, BitstreamError>;

/// Bit reader over an RBSP buffer. Spec refs: §7.2 (more_rbsp_data), §9.1.
pub struct BitstreamReader<'a> {
    data: &'a [u8],
    size: usize,
    bit_pos: usize, // current bit position (logical)

    // 64-bit read cache
    cache: u64,
    cache_bits: i32, // valid bits remaining in cache (MSB-aligned)
    byte_pos: usize, // next byte to load into cache

    // Precomputed position of rbsp_stop_one_bit (§7.2)
    last_one_bit_pos: usize,
}

impl<'a> BitstreamReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            size: data.len(),
            bit_pos: 0,
            cache: 0,
            cache_bits: 0,
            byte_pos: 0,
            last_one_bit_pos: 0,
        };
        r.last_one_bit_pos = find_last_one_bit(data);
        r.refill();
        r
    }

    /// Refill the 64-bit cache from the byte stream (MSB-aligned).
    fn refill(&mut self) {
        // Fast path: bulk load of 8 bytes when the cache is empty.
        if self.cache_bits <= 0 && self.byte_pos + 8 <= self.size {
            let raw = &self.data[self.byte_pos..self.byte_pos + 8];
            self.cache = u64::from_be_bytes(raw.try_into().unwrap());
            self.cache_bits = 64;
            self.byte_pos += 8;
            return;
        }

        // Slow path: byte-by-byte for the remaining bytes.
        while self.cache_bits <= 56 && self.byte_pos < self.size {
            self.cache |= u64::from(self.data[self.byte_pos]) << (56 - self.cache_bits);
            self.cache_bits += 8;
            self.byte_pos += 1;
        }
    }

    /// Fixed-length read (§7.2). Fails past end of data.
    pub fn read_bits(&mut self, n: usize) -> BitstreamResult<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }

        if self.bit_pos + n > self.size * 8 {
            return Err(BitstreamError::ReadPastEnd);
        }

        if self.cache_bits < n as i32 {
            self.refill();
        }

        let result = (self.cache >> (64 - n)) as u32;
        self.cache <<= n;
        self.cache_bits -= n as i32;
        self.bit_pos += n;

        Ok(result)
    }

    pub fn read_u(&mut self, n: usize) -> BitstreamResult<u32> {
        self.read_bits(n)
    }

    /// Fixed-length signed read (two's-complement sign extension).
    /// Caller guarantees 1 <= n <= 31.
    pub fn read_i(&mut self, n: usize) -> BitstreamResult<i32> {
        debug_assert!((1..=31).contains(&n));
        let val = self.read_bits(n)?;
        if val & (1 << (n - 1)) != 0 {
            Ok((val | (!0u32) << n) as i32)
        } else {
            Ok(val as i32)
        }
    }

    pub fn read_flag(&mut self) -> BitstreamResult<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read one byte (caller must be byte-aligned).
    pub fn read_byte(&mut self) -> BitstreamResult<u8> {
        debug_assert!(self.byte_aligned());
        Ok(self.read_bits(8)? as u8)
    }

    /// Read bits with zero-padding past end of data (for CABAC renormalization).
    #[inline]
    pub fn read_bits_safe(&mut self, n: usize) -> u32 {
        if n == 0 {
            return 0;
        }
        if n >= 64 {
            // More than a cache word (degenerate CABAC suffix). Drain in 32-bit
            // chunks to advance the position by exactly n, keeping the low 32 bits
            // (matches the C++ `uint32_t` truncation of `read_bits_safe`).
            let mut remaining = n;
            while remaining >= 32 {
                let _ = self.read_bits_safe(32);
                remaining -= 32;
            }
            return self.read_bits_safe(remaining);
        }
        if self.cache_bits < n as i32 {
            self.refill();
        }
        if self.cache_bits < n as i32 {
            // Past end — return zero-padded
            let result = if self.cache_bits > 0 {
                (self.cache >> (64 - n)) as u32
            } else {
                0
            };
            self.cache = 0;
            self.bit_pos += n;
            self.cache_bits = 0;
            return result;
        }
        let result = (self.cache >> (64 - n)) as u32;
        self.cache <<= n;
        self.cache_bits -= n as i32;
        self.bit_pos += n;
        result
    }

    /// Fast single-bit read for the CABAC hot path (no n==0 check).
    #[inline]
    pub fn read_bit_fast(&mut self) -> u32 {
        if self.cache_bits < 1 {
            self.refill();
        }
        let bit = (self.cache >> 63) as u32;
        self.cache <<= 1;
        self.cache_bits -= 1;
        self.bit_pos += 1;
        bit
    }

    /// Exp-Golomb unsigned (§9.2).
    pub fn read_ue(&mut self) -> BitstreamResult<u32> {
        let mut leading_zeros = 0;
        while !self.eof() && self.read_bits(1)? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(BitstreamError::ExpGolombOverflow);
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let suffix = self.read_bits(leading_zeros)?;
        Ok((1u32 << leading_zeros) - 1 + suffix)
    }

    /// Exp-Golomb signed (§9.2): 0->0, 1->1, 2->-1, 3->2, 4->-2, ...
    pub fn read_se(&mut self) -> BitstreamResult<i32> {
        let code = self.read_ue()?;
        let value = code.div_ceil(2) as i32;
        Ok(if code & 1 != 0 { value } else { -value })
    }

    pub fn byte_aligned(&self) -> bool {
        self.bit_pos.is_multiple_of(8)
    }

    /// §7.2 — read alignment_bit_equal_to_one, then zeros until byte aligned.
    pub fn byte_alignment(&mut self) -> BitstreamResult<()> {
        self.read_bits(1)?; // alignment_bit_equal_to_one
        while !self.byte_aligned() {
            self.read_bits(1)?; // alignment_bit_equal_to_zero
        }
        Ok(())
    }

    /// §7.2 — more_rbsp_data(): true if the current position is before the
    /// rbsp_stop_one_bit.
    pub fn more_rbsp_data(&self) -> bool {
        if self.bit_pos >= self.size * 8 {
            return false;
        }
        self.bit_pos < self.last_one_bit_pos
    }

    pub fn bits_read(&self) -> usize {
        self.bit_pos
    }

    /// Current byte position (bit position / 8).
    pub fn byte_position(&self) -> usize {
        self.bit_pos / 8
    }

    pub fn bits_remaining(&self) -> usize {
        (self.size * 8).saturating_sub(self.bit_pos)
    }

    pub fn eof(&self) -> bool {
        self.bit_pos >= self.size * 8
    }

    /// Seek to an absolute byte position (resets cache). Used by WPP to jump
    /// to the start of each substream.
    pub fn seek_to_byte(&mut self, pos: usize) {
        debug_assert!(pos <= self.size);
        self.bit_pos = pos * 8;
        self.byte_pos = pos;
        self.cache = 0;
        self.cache_bits = 0;
        self.refill();
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

/// Scan backward from end to find the last '1' bit (rbsp_stop_one_bit).
fn find_last_one_bit(data: &[u8]) -> usize {
    for i in (0..data.len()).rev() {
        let byte = data[i];
        if byte != 0 {
            // Lowest set bit in this byte = last '1' bit in stream order.
            let bit = byte.trailing_zeros();
            return i * 8 + (7 - bit) as usize;
        }
    }
    0
}

/// RBSP extraction — remove emulation prevention bytes (§7.3.1.1).
pub fn extract_rbsp(nal_data: &[u8]) -> Vec<u8> {
    extract_rbsp_with_epb(nal_data).0
}

/// RBSP extraction with EP byte position tracking.
/// `epb_positions` receives the byte positions (in the original NAL) of each
/// removed 0x03 byte.
pub fn extract_rbsp_with_epb(nal_data: &[u8]) -> (Vec<u8>, Vec<usize>) {
    let nal_size = nal_data.len();
    let mut rbsp = Vec::with_capacity(nal_size);
    let mut epb_positions = Vec::new();

    let mut i = 0;
    while i < nal_size {
        // Emulation prevention: 0x00 0x00 0x03 followed by 0x00-0x03, or at
        // the end of the NAL.
        if i + 2 < nal_size
            && nal_data[i] == 0x00
            && nal_data[i + 1] == 0x00
            && nal_data[i + 2] == 0x03
            && (i + 3 >= nal_size || nal_data[i + 3] <= 0x03)
        {
            rbsp.push(0x00);
            rbsp.push(0x00);
            epb_positions.push(i + 2); // position of the 0x03 byte
            i += 3; // skip emulation_prevention_three_byte
        } else {
            rbsp.push(nal_data[i]);
            i += 1;
        }
    }

    (rbsp, epb_positions)
}

/// Convert a byte offset in coded slice data (which counts EP bytes) to the
/// corresponding byte offset in the RBSP buffer.
/// `slice_data_start_coded` = byte offset of slice data start in the original NAL.
pub fn coded_to_rbsp_offset(
    coded_offset: usize,
    slice_data_start_coded: usize,
    epb_positions: &[usize],
) -> usize {
    // coded_offset is relative to slice data start in the coded (NAL) domain.
    // Subtract the EP bytes that fall before this position.
    let abs_coded = slice_data_start_coded + coded_offset;
    let epb_count = epb_positions.iter().take_while(|&&pos| pos < abs_coded).count();

    // RBSP offset of the slice data start.
    let mut slice_data_start_rbsp = slice_data_start_coded;
    for &pos in epb_positions {
        if pos < slice_data_start_coded {
            slice_data_start_rbsp -= 1;
        } else {
            break;
        }
    }
    (abs_coded - epb_count) - slice_data_start_rbsp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive MSB-first bit reference (independent of the cache logic).
    fn naive_bit(data: &[u8], bit: usize) -> u32 {
        ((data[bit / 8] >> (7 - bit % 8)) & 1) as u32
    }

    fn naive_bits(data: &[u8], start: usize, n: usize) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            v = (v << 1) | naive_bit(data, start + i);
        }
        v
    }

    /// Deterministic PRNG (splitmix64) — no external dependency.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Encode `values` as concatenated Exp-Golomb codes into a byte buffer.
    fn encode_ue_bytes(values: &[u32]) -> Vec<u8> {
        let mut bits: Vec<u32> = Vec::new();
        for &v in values {
            // L = floor(log2(v+1)); suffix = v - (2^L - 1) in L bits.
            let code_num = v + 1;
            let lz = 31 - code_num.leading_zeros();
            bits.extend(std::iter::repeat_n(0, lz as usize));
            bits.push(1);
            let suffix = v - ((1u32 << lz) - 1);
            for i in (0..lz).rev() {
                bits.push((suffix >> i) & 1);
            }
        }
        // Pad to a whole byte so the last chunk stays left-aligned (MSB-first).
        let pad = (8 - bits.len() % 8) % 8;
        bits.resize(bits.len() + pad, 0);
        bits.chunks(8)
            .map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b as u8))
            .collect()
    }

    /// Naive Exp-Golomb decode from `data` at bit `pos`.
    fn naive_ue(data: &[u8], pos: usize) -> (u32, usize) {
        let mut lz = 0usize;
        while naive_bit(data, pos + lz) == 0 {
            lz += 1;
        }
        if lz == 0 {
            return (0, pos + 1);
        }
        let suffix = naive_bits(data, pos + lz + 1, lz);
        ((1u32 << lz) - 1 + suffix, pos + lz + 1 + lz)
    }

    #[test]
    fn read_bits_matches_naive_reference() {
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let len = rng.below(40) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            if data.is_empty() {
                continue;
            }

            let total_bits = len * 8;
            let mut reader = BitstreamReader::new(&data);
            let mut pos = 0usize;
            while pos < total_bits {
                let n = (rng.below(31) as usize + 1).min(total_bits - pos);
                let got = reader.read_bits(n).unwrap();
                assert_eq!(got, naive_bits(&data, pos, n), "len={len} pos={pos} n={n}");
                pos += n;
            }
        }
    }

    #[test]
    fn read_past_end_errors() {
        let data = [0xA5u8, 0x51];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_bits(16).unwrap(), 0xA551);
        assert_eq!(r.read_bits(1), Err(BitstreamError::ReadPastEnd));

        let mut r = BitstreamReader::new(&[]);
        assert_eq!(r.read_bits(1), Err(BitstreamError::ReadPastEnd));
    }

    #[test]
    fn read_i_sign_extension() {
        // 0xFF as i8 = -1; 0x7F as i8 = 127; 0x80 as i8 = -128.
        let mut r = BitstreamReader::new(&[0xFF, 0x7F, 0x80]);
        assert_eq!(r.read_i(8).unwrap(), -1);
        assert_eq!(r.read_i(8).unwrap(), 127);
        assert_eq!(r.read_i(8).unwrap(), -128);
    }

    #[test]
    fn ue_roundtrip_0_to_63() {
        let values: Vec<u32> = (0..=63u32).collect();
        let bytes = encode_ue_bytes(&values);
        let mut r = BitstreamReader::new(&bytes);
        for expected in &values {
            assert_eq!(r.read_ue().unwrap(), *expected);
        }
    }

    #[test]
    fn ue_matches_naive_decode() {
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            // Values kept small so codes stay short on random-length streams.
            let values: Vec<u32> = (0..8).map(|_| rng.below(64) as u32).collect();
            let bytes = encode_ue_bytes(&values);
            let mut r = BitstreamReader::new(&bytes);
            let mut pos = 0usize;
            for expected in &values {
                let (ref_val, next_pos) = naive_ue(&bytes, pos);
                assert_eq!(r.read_ue().unwrap(), *expected);
                assert_eq!(ref_val, *expected, "pos={pos}");
                pos = next_pos;
            }
        }
    }

    #[test]
    fn se_roundtrip() {
        // se mapping: 0->0, 1->1, 2->-1, 3->2, 4->-2, 5->3
        let bytes = encode_ue_bytes(&[0u32, 1, 2, 3, 4, 5]);
        let mut r = BitstreamReader::new(&bytes);
        for expected in [0i32, 1, -1, 2, -2, 3] {
            assert_eq!(r.read_se().unwrap(), expected);
        }
    }

    #[test]
    fn ue_overflow() {
        // More than 31 leading zeros before end of data.
        let data = [0u8; 5];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_ue(), Err(BitstreamError::ExpGolombOverflow));
    }

    #[test]
    fn seek_and_reread() {
        let data: Vec<u8> = (0..24u8).map(|i| i.wrapping_mul(7)).collect();
        let mut r = BitstreamReader::new(&data);
        // Read something, then seek back and verify identical values.
        assert_eq!(r.read_bits(13).unwrap(), naive_bits(&data, 0, 13));
        for pos in [0usize, 1, 7, 8, 9, 16, 23] {
            r.seek_to_byte(pos);
            let n = (24 - pos) * 8;
            for k in 0..n {
                let bit = r.read_bit_fast();
                assert_eq!(bit, naive_bit(&data, pos * 8 + k), "pos={pos} k={k}");
            }
        }
    }

    #[test]
    fn read_bits_safe_zero_pads_past_end() {
        let data: Vec<u8> = (0..12u8).map(|i| i.wrapping_mul(3)).collect();
        for start in [0usize, 1, 5, 11] {
            let mut r = BitstreamReader::new(&data);
            r.seek_to_byte(start);
            let remaining = (data.len() - start) * 8;
            // Consume all remaining bits one at a time (cache ends clean).
            for k in 0..remaining {
                assert_eq!(r.read_bit_fast(), naive_bit(&data, start * 8 + k));
            }
            // Past end: zero-padded reads.
            for n in [1usize, 2, 3, 4, 5, 6, 7, 8] {
                assert_eq!(r.read_bits_safe(n), 0, "start={start} n={n}");
            }
        }
    }

    #[test]
    fn byte_alignment_and_more_rbsp_data() {
        // RBSP: payload 0xAB 0xCD, then rbsp_stop_one_bit (bit 22 of the last
        // byte 0b0000_0010) + one trailing zero bit.
        let data = [0xABu8, 0xCD, 0b0000_0010];
        let mut r = BitstreamReader::new(&data);
        assert!(r.more_rbsp_data());
        assert_eq!(r.read_bits(16).unwrap(), 0xABCD);
        assert!(r.more_rbsp_data()); // bit 16 < stop bit 22
        r.read_bits(6).unwrap();
        assert!(!r.more_rbsp_data()); // at the stop bit
        r.byte_alignment().unwrap();
        assert!(r.byte_aligned());
        assert!(r.eof());

        // Trailing zero bytes: last one bit stays at the payload end.
        let data = [0xABu8, 0x00, 0x00];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn extract_rbsp_cases() {
        // No EPB: passthrough.
        assert_eq!(extract_rbsp(&[1, 2, 3]), vec![1, 2, 3]);

        // 00 00 03 00 -> 03 removed.
        let (rbsp, epb) = extract_rbsp_with_epb(&[0x00, 0x00, 0x03, 0x00]);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x00]);
        assert_eq!(epb, vec![2]);

        // 00 00 03 04 -> NOT removed (next byte > 0x03).
        let (rbsp, epb) = extract_rbsp_with_epb(&[0x00, 0x00, 0x03, 0x04]);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x03, 0x04]);
        assert!(epb.is_empty());

        // 00 00 03 at end of NAL -> removed.
        let (rbsp, epb) = extract_rbsp_with_epb(&[0x5A, 0x00, 0x00, 0x03]);
        assert_eq!(rbsp, vec![0x5A, 0x00, 0x00]);
        assert_eq!(epb, vec![3]);

        // 00 00 03 03 -> removed; consecutive EPB sequences.
        let (rbsp, epb) = extract_rbsp_with_epb(&[0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x03, 0x01]);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x03, 0x00, 0x00, 0x01]);
        assert_eq!(epb, vec![2, 6]);

        // 00 00 02 00 -> not an EPB.
        let (rbsp, epb) = extract_rbsp_with_epb(&[0x00, 0x00, 0x02, 0x00]);
        assert_eq!(rbsp, vec![0x00, 0x00, 0x02, 0x00]);
        assert!(epb.is_empty());

        // Empty input.
        let (rbsp, epb) = extract_rbsp_with_epb(&[]);
        assert!(rbsp.is_empty() && epb.is_empty());
    }

    #[test]
    fn coded_to_rbsp_offset_cases() {
        // NAL: [0x5A, 0x00, 0x00, 0x03, 0x01, ...], slice data starts at
        // coded byte 2 (inside the EPB region).
        let epb = vec![3usize];
        // Coded offset 0 from slice start (coded byte 2): no EPB before it.
        assert_eq!(coded_to_rbsp_offset(0, 2, &epb), 0);
        // Coded offset 1 (coded byte 3 = the EPB byte itself): abs_coded 3,
        // no EPB strictly before it -> 3 - 0 = 3; relative to slice start
        // (rbsp byte 2) -> 1.
        assert_eq!(coded_to_rbsp_offset(1, 2, &epb), 1);
        // Coded offset 2 (coded byte 4 = 0x01): abs 4, one EPB before ->
        // rbsp byte 3; relative to slice start (rbsp 2) -> 1.
        assert_eq!(coded_to_rbsp_offset(2, 2, &epb), 1);
        // Coded offset 3 (coded byte 5): abs 5, one EPB before -> 4 - 1 = 3;
        // relative to slice start -> 2.
        assert_eq!(coded_to_rbsp_offset(3, 2, &epb), 2);

        // No EPBs: identity.
        assert_eq!(coded_to_rbsp_offset(7, 3, &[]), 7);

        // Slice start after the EPB: offset is identity again.
        assert_eq!(coded_to_rbsp_offset(5, 4, &epb), 5);
    }

    #[test]
    fn random_stream_roundtrip_vs_naive() {
        // Randomized mixed reads (u/i/flag/byte) against the naive reference.
        let mut rng = Rng::new(7);
        for _ in 0..100 {
            let len = (rng.below(30) as usize + 1) * 8; // whole bytes
            let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();

            let mut reader = BitstreamReader::new(&data);
            let mut pos = 0usize;
            while pos < len * 8 {
                match rng.below(4) {
                    0 => {
                        let n = (rng.below(31) as usize + 1).min(len * 8 - pos);
                        assert_eq!(reader.read_bits(n).unwrap(), naive_bits(&data, pos, n));
                        pos += n;
                    }
                    1 => {
                        assert_eq!(reader.read_flag().unwrap(), naive_bit(&data, pos) == 1);
                        pos += 1;
                    }
                    2 => {
                        let n = (rng.below(15) as usize + 1).min(len * 8 - pos);
                        let uv = naive_bits(&data, pos, n);
                        let expected = if uv & (1 << (n - 1)) != 0 {
                            (uv | (!0u32) << n) as i32
                        } else {
                            uv as i32
                        };
                        assert_eq!(reader.read_i(n).unwrap(), expected, "pos={pos} n={n}");
                        pos += n;
                    }
                    _ => {
                        if pos.is_multiple_of(8) && pos + 8 <= len * 8 {
                            let b = reader.read_byte().unwrap();
                            assert_eq!(b, data[pos / 8]);
                            pos += 8;
                        } else {
                            let n = (rng.below(7) as usize + 1).min(len * 8 - pos);
                            assert_eq!(reader.read_bits(n).unwrap(), naive_bits(&data, pos, n));
                            pos += n;
                        }
                    }
                }
            }
        }
    }
}
