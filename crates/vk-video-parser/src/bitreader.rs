// Adapted from cros-codecs (ChromiumOS) - BSD-2-Clause license
// Original: https://chromium.googlesource.com/chromiumos/platform/devel/cros-codecs/
//
//! Bit reader for H.264/H.265 bitstreams with inline emulation-prevention byte handling.
//!
//! Bits are read MSB-first within each byte. After loading a byte, bit 7 (MSB)
//! is read first, then bit 6, ..., down to bit 0 (LSB).

/// A bit reader for codec bitstreams. Properly handles emulation-prevention
/// bytes during reading.
pub struct BitReader<'a> {
    /// Data being read.
    data: &'a [u8],
    /// Current byte offset in data.
    pub pos: usize,
    /// Contents of the current byte.
    curr_byte: u8,
    /// Number of bits remaining (unread) in `curr_byte`.
    bits_left: usize,
    /// Previous two bytes for EPB detection (0x00 0x00 0x03 pattern).
    prev_two_bytes: u16,
    /// Whether to apply emulation-prevention byte removal.
    remove_epb: bool,
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader.
    pub fn new(data: &'a [u8], remove_epb: bool) -> Self {
        Self {
            data,
            pos: 0,
            curr_byte: 0,
            bits_left: 0,
            prev_two_bytes: 0xFFFF,
            remove_epb,
        }
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool, ParserError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read `n` fixed-width bits (1..=32), MSB-first.
    ///
    /// Algorithm (from cros-codecs): accumulate bits from the current byte
    /// while we don't have enough, then take the remaining bits from the
    /// top of the current byte.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, ParserError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(ParserError::InvalidBitstream);
        }

        let mut bits_left = n as usize;
        let mut out: u32 = 0;

        // Accumulate whole bytes while we need more bits than are available
        while self.bits_left < bits_left {
            out |= (self.curr_byte as u32) << (bits_left - self.bits_left);
            bits_left -= self.bits_left;
            self.load_byte()?;
        }

        // Take the remaining bits from the top of curr_byte
        out |= (self.curr_byte >> (self.bits_left as u32 - bits_left as u32)) as u32;
        // Handle n=32 specially to avoid UB with 1u32 << 32
        out &= if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.bits_left -= bits_left;

        Ok(out)
    }

    /// Read one byte (8 bits).
    pub fn read_byte(&mut self) -> Result<u8, ParserError> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Read an unsigned exponential-Golomb coded integer (ue(v)).
    pub fn read_ue(&mut self) -> Result<u32, ParserError> {
        let mut num_bits: u32 = 0;

        // Scan for leading zeros (one bit at a time)
        while self.read_bits(1)? == 0 {
            num_bits += 1;
            if num_bits > 31 {
                return Err(ParserError::InvalidBitstream);
            }
        }

        // Read the value bits
        let value = ((1u32 << num_bits) - 1)
            .checked_add(self.read_bits(num_bits as u8)?)
            .ok_or(ParserError::InvalidBitstream)?;

        Ok(value)
    }

    /// Read a signed exponential-Golomb coded integer (se(v)).
    ///
    /// H.264 spec maps: even ue → positive, odd ue → negative.
    pub fn read_se(&mut self) -> Result<i32, ParserError> {
        let code = self.read_ue()? as i32;
        if code % 2 == 0 {
            Ok(code / 2)
        } else {
            Ok(-(code / 2 + 1))
        }
    }

    /// Skip `n` bits (1..=31), handling EPB if enabled.
    pub fn skip_bits(&mut self, n: u8) -> Result<(), ParserError> {
        let _ = self.read_bits(n)?;
        Ok(())
    }

    /// Read an unsigned variable-length coded integer (uvlc) per AV1 spec 4.10.3.
    ///
    /// Format: leading_zeroes zeros, then a one bit, then leading_zeroes value bits.
    /// Value = 2^leading_zeroes - 1 + value_bits.
    pub fn read_uvlc(&mut self) -> Result<u32, ParserError> {
        let mut leading_zeroes = 0u8;

        loop {
            let done = self.read_bits(1)? != 0;
            if done {
                break;
            }
            leading_zeroes += 1;
        }

        if leading_zeroes >= 32 {
            return Ok(u32::MAX);
        }

        let value = self.read_bits(leading_zeroes)?;
        Ok(value + (1u32 << leading_zeroes) - 1)
    }

    /// Read a little-endian base-128 encoded integer (leb128) per AV1 spec 4.10.5.
    ///
    /// Each byte has 7 data bits and a continuation bit (MSB).
    /// Returns the decoded value.
    pub fn read_leb128(&mut self) -> Result<u32, ParserError> {
        let mut value = 0u64;

        for i in 0..8 {
            let byte = self.read_bits(8)? as u64;
            value |= (byte & 0x7F) << (i * 7);

            if (byte & 0x80) == 0 {
                return Ok(value as u32);
            }
        }

        Err(ParserError::InvalidBitstream)
    }

    /// Get current bit position in the stream.
    pub fn position(&self) -> u64 {
        (self.pos as u64) * 8 - (8 - self.bits_left) as u64
    }

    /// Check if there is more data to read.
    pub fn has_more_data(&self) -> bool {
        self.pos < self.data.len() || self.bits_left > 0
    }

    /// Check if there is more RBSP data to read.
    ///
    /// Returns false when we've reached the end of meaningful RBSP data
    /// (at or past the trailing bit and trailing zeros).
    ///
    /// RBSP format: data bits + one trailing bit (1) + zero or more padding bits (0).
    /// We need to detect when remaining bits are just the trailing bit + padding.
    pub fn has_more_rsbp_data(&self) -> bool {
        // If we've consumed all bytes, no more data
        if self.pos >= self.data.len() && self.bits_left == 0 {
            return false;
        }

        // Collect all remaining bits from current position to end of data
        // to check if they match the RBSP ending pattern: 1 followed by all 0s
        let mut remaining_value: u64 = 0;
        let mut remaining_bits: u32 = 0;

        // Bits remaining in current byte
        if self.bits_left > 0 {
            let mask = (1u8 << self.bits_left) - 1;
            remaining_value = (self.curr_byte & mask) as u64;
            remaining_bits = self.bits_left as u32;
        }

        // Include all subsequent bytes
        for &byte in &self.data[self.pos..] {
            remaining_value = (remaining_value << 8) | (byte as u64);
            remaining_bits += 8;
        }

        if remaining_bits == 0 {
            return false;
        }

        // RBSP ending pattern: 1 followed by all 0s
        // This means: remaining_value should be a power of 2 (single bit set at the top)
        // Example: 10000000 (trailing bit + 7 padding bits) = 0x80 = power of 2
        // Example: 1000000 (trailing bit + 6 padding bits) = 0x40 = power of 2
        if remaining_value > 0 && (remaining_value & (remaining_value - 1)) == 0 {
            // Remaining bits are just trailing bit + padding zeros
            return false;
        }

        true
    }

    /// Load the next byte into curr_byte, handling EPB if enabled.
    fn load_byte(&mut self) -> Result<(), ParserError> {
        if self.pos >= self.data.len() {
            return Err(ParserError::InvalidBitstream);
        }

        let byte = self.data[self.pos];
        self.pos += 1;

        if self.remove_epb && self.prev_two_bytes == 0 && byte == 0x03 {
            // EPB: skip 0x03 and read the actual data byte
            if self.pos >= self.data.len() {
                return Err(ParserError::InvalidBitstream);
            }
            let actual_byte = self.data[self.pos];
            self.pos += 1;
            // Update prev_two_bytes with the actual byte (matching cros-codecs)
            self.prev_two_bytes = (0xFFFFu16 << 8) | (actual_byte as u16);
            self.curr_byte = actual_byte;
        } else {
            self.prev_two_bytes = (self.prev_two_bytes << 8) | (byte as u16);
            self.curr_byte = byte;
        }

        self.bits_left = 8;
        Ok(())
    }

    /// Calculate the number of bytes consumed (rounded up to full byte).
    pub fn bytes_consumed(&self) -> usize {
        // pos tracks bytes loaded, bits_left tracks remaining bits in current byte
        if self.pos == 0 {
            0
        } else if self.bits_left == 8 {
            self.pos - 1 // Loaded but not consumed current byte
        } else {
            self.pos // Partially consumed current byte
        }
    }

    /// Debug: get current read position in bits.
    #[cfg(debug_assertions)]
    pub fn debug_bit_pos(&self) -> usize {
        if self.pos == 0 {
            0
        } else {
            (self.pos - 1) * 8 + (8 - self.bits_left)
        }
    }

    /// Debug: get current byte being read.
    #[cfg(debug_assertions)]
    pub fn debug_curr_byte(&self) -> u8 {
        self.curr_byte
    }
}

/// Errors from bit reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserError {
    /// The bitstream is invalid or truncated.
    InvalidBitstream,
}

impl std::fmt::Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParserError::InvalidBitstream => write!(f, "invalid bitstream"),
        }
    }
}

impl std::error::Error for ParserError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits_basic() {
        // 0b10101010 = 0xAA
        let data = [0xAA, 0xCC];
        let mut r = BitReader::new(&data, false);

        // Read 1 bit: MSB = 1
        assert_eq!(r.read_bits(1).unwrap(), 1);
        // Read 3 bits: 010 = 2
        assert_eq!(r.read_bits(3).unwrap(), 2);
        // Read 4 bits: 1010 = 10
        assert_eq!(r.read_bits(4).unwrap(), 10);
        // Read 8 bits from second byte: 11001100 = 204
        assert_eq!(r.read_bits(8).unwrap(), 0xCC);
    }

    #[test]
    fn test_read_bits_cross_byte() {
        // Read bits spanning across byte boundary
        // Continuous bit stream: 1111 0000 0000 1111
        let data = [0b11110000, 0b00001111];
        let mut r = BitReader::new(&data, false);
        // First 4 bits: 1111 = 15
        assert_eq!(r.read_bits(4).unwrap(), 0xF);
        // Next 8 bits: 0000 0000 = 0 (bottom 4 of byte1 + top 4 of byte2)
        assert_eq!(r.read_bits(8).unwrap(), 0);
        // Last 4 bits: 1111 = 15
        assert_eq!(r.read_bits(4).unwrap(), 0xF);
    }

    #[test]
    fn test_read_ue() {
        // ue(v)=0: 1xxxxxxx (0 leading zeros, no value bits)
        let data = [0b10000000];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 0);

        // ue(v)=1: 010xxxxx (1 leading zero, stop, value=0 → 1+0=1)
        let data = [0b01000000];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 1);

        // ue(v)=2: 011xxxxx (1 leading zero, stop, value=1 → 1+1=2)
        let data = [0b01100000];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 2);

        // ue(v)=5: 001101xx (2 leading zeros, stop, value=10=2 → 3+2=5)
        let data = [0b00110100];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 5);

        // ue(v)=12: 0001101x (3 leading zeros, stop, value=101=5 → 7+5=12)
        let data = [0b00011010];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 12);
    }

    #[test]
    fn test_epb_removal() {
        // Data: 0x00 0x00 0x03 0xAB -> after EPB removal: 0x00 0x00 0xAB
        let data = [0x00, 0x00, 0x03, 0xAB];
        let mut r = BitReader::new(&data, true);
        assert_eq!(r.read_bits(8).unwrap(), 0x00);
        assert_eq!(r.read_bits(8).unwrap(), 0x00);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB); // EPB (0x03) was skipped

        // Without EPB removal
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_bits(8).unwrap(), 0x00);
        assert_eq!(r.read_bits(8).unwrap(), 0x00);
        assert_eq!(r.read_bits(8).unwrap(), 0x03); // EPB NOT skipped
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
    }

    /// Test adapted from cros-codecs: read_stream_without_escape_and_trailing_zero_bytes
    #[test]
    fn test_cros_codecs_stream() {
        // First bit of 0x01 is 0. Remaining 7 bits of 0x01 + first bit of 0x23 = 0x02
        // Then 31 bits: 0x23456789, then 1 bit = 1 (from 0x89 = 10001001, after 31 bits we're at the 1)
        const RBSP: [u8; 6] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xa0];

        let mut r = BitReader::new(&RBSP, true);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.read_bits(8).unwrap(), 0x02);
        assert_eq!(r.read_bits(31).unwrap(), 0x23456789);
        assert_eq!(r.read_bits(1).unwrap(), 1);
        assert_eq!(r.read_bits(1).unwrap(), 0);
    }

    #[test]
    fn test_h264_sps_parse() {
        // Parse the SPS from born_trailer.h264
        // NAL header + SPS RBSP data (first 20 bytes)
        let data = [
            0x67, // NAL header: type=7 (SPS), ref_idc=3
            0x42, // profile_idc = 66 (Baseline)
            0xC0, // constraint_set flags
            0x29, // level_idc = 41 (Level 4.1)
            0x9A, // seq_parameter_set_id (ue(v))
            0x74, // log2_max_frame_num_minus4, pic_order_cnt_type
            0x03, // log2_max_pic_order_cnt_lsb_minus4, max_num_ref_frames
            0xC0, // gaps_in_frame_num, pic_width_in_mbs_minus1
            0x33, // pic_width_in_mbs_minus1 cont.
            0xD0, // pic_height_in_map_units_minus1
            0x80, // frame_mbs_only_flag, direct_8x8_inference_flag
            0x00, 0xCB, 0xA7, 0x80, 0x26, 0x25, 0xA0, 0x47, 0x8C,
        ];

        let mut r = BitReader::new(&data, true);

        // Skip NAL header (1 byte)
        let _ = r.read_byte().unwrap();

        // profile_idc
        let profile_idc = r.read_byte().unwrap();
        assert_eq!(profile_idc, 66, "expected Baseline profile");

        // constraint_set flags (8 bits)
        let _constraints = r.read_bits(8).unwrap();

        // level_idc
        let level_idc = r.read_byte().unwrap();
        assert_eq!(level_idc, 41, "expected Level 4.1");

        // seq_parameter_set_id (ue(v))
        let sps_id = r.read_ue().unwrap();
        assert_eq!(sps_id, 0, "expected sps_id=0");

        // Baseline profile: skip chroma_format_idc, bit_depth, scaling lists

        // log2_max_frame_num_minus4 (ue(v))
        let log2_max_fn = r.read_ue().unwrap();
        assert_eq!(log2_max_fn, 5, "expected log2_max_frame_num_minus4=5");

        // pic_order_cnt_type (ue(v))
        let poc_type = r.read_ue().unwrap();
        assert_eq!(poc_type, 0, "expected pic_order_cnt_type=0");

        // log2_max_pic_order_cnt_lsb_minus4 (ue(v))
        let log2_max_poc = r.read_ue().unwrap();
        assert_eq!(
            log2_max_poc, 6,
            "expected log2_max_pic_order_cnt_lsb_minus4=6"
        );

        // max_num_ref_frames (ue(v))
        let max_ref = r.read_ue().unwrap();
        assert_eq!(max_ref, 1, "expected max_num_ref_frames=1");

        // gaps_in_frame_num_value_allowed_flag (1 bit)
        let gaps = r.read_bit().unwrap();
        assert!(!gaps, "expected gaps_in_frame_num=false");

        // pic_width_in_mbs_minus1 (ue(v))
        let pic_w = r.read_ue().unwrap();
        assert_eq!(
            pic_w, 119,
            "expected pic_width_in_mbs_minus1=119 (1920/16-1)"
        );

        // pic_height_in_map_units_minus1 (ue(v))
        let pic_h = r.read_ue().unwrap();
        assert_eq!(pic_h, 50, "expected pic_height_in_map_units_minus1=50");

        // frame_mbs_only_flag (1 bit)
        let frame_mbs_only = r.read_bit().unwrap();
        assert!(frame_mbs_only, "expected frame_mbs_only_flag=true");

        let width = (pic_w + 1) * 16;
        let height = (pic_h + 1) * 16; // frame_mbs_only = 1
        assert_eq!(width, 1920);
        assert_eq!(height, 816);
    }

    #[test]
    fn test_h264_sps_parse_actual_born() {
        // Parse the ACTUAL SPS from born_trailer.h264
        // Data starts after NAL header (0x67)
        let data = [
            0x42, 0xC0, 0x29, 0x9A, 0x74, 0x03, 0xC0, 0x33, 0xD0, 0x80, 0x00, 0xCB, 0xA7, 0x80,
            0x26, 0x25, 0xA0, 0x47, 0x8C, 0x19,
        ];

        let mut r = BitReader::new(&data, true);

        let profile_idc = r.read_byte().unwrap();
        assert_eq!(profile_idc, 66);

        let _constraints = r.read_bits(8).unwrap();
        let level_idc = r.read_byte().unwrap();
        assert_eq!(level_idc, 41);

        let sps_id = r.read_ue().unwrap();
        println!("sps_id: {}", sps_id);

        let log2_max_fn = r.read_ue().unwrap();
        println!("log2_max_frame_num_minus4: {}", log2_max_fn);

        let poc_type = r.read_ue().unwrap();
        println!("pic_order_cnt_type: {}", poc_type);

        let log2_max_poc = r.read_ue().unwrap();
        println!("log2_max_pic_order_cnt_lsb_minus4: {}", log2_max_poc);

        let max_ref = r.read_ue().unwrap();
        println!("max_num_ref_frames: {}", max_ref);

        let gaps = r.read_bit().unwrap();
        println!("gaps_in_frame_num: {}", gaps);

        let pic_w = r.read_ue().unwrap();
        println!("pic_width_in_mbs_minus1: {}", pic_w);

        let pic_h = r.read_ue().unwrap();
        println!("pic_height_in_map_units_minus1: {}", pic_h);

        let frame_mbs_only = r.read_bit().unwrap();
        println!("frame_mbs_only_flag: {}", frame_mbs_only);

        // Verify the video dimensions
        let width = (pic_w + 1) * 16;
        let height = if frame_mbs_only {
            (pic_h + 1) * 16
        } else {
            (pic_h + 1) * 16 * 2
        };
        println!("Width: {}, Height: {}", width, height);
        assert_eq!(width, 1920, "expected width=1920");
        assert_eq!(height, 816, "expected height=816");
    }
}
