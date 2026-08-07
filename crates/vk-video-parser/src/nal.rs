//! NAL unit types and utilities.

/// H.264 NAL unit types (ITU-T H.264/AVC specification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum H264NalUnitType {
    Unspecified = 0,
    NonIdrSlice = 1,
    DataPartitionA = 2,
    DataPartitionB = 3,
    DataPartitionC = 4,
    IdrSlice = 5,
    Sei = 6,
    Sps = 7,
    Pps = 8,
    AccessUnitDelimiter = 9,
    SeqEnd = 10,
    StreamEnd = 11,
    FillerData = 12,
    SpsExt = 13,
    AuxiliaryCodecLayer = 14,
    CodedSliceExtension = 15,
    // 16-23 are reserved
    // 24-31 are undefined
}

impl H264NalUnitType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unspecified),
            1 => Some(Self::NonIdrSlice),
            2 => Some(Self::DataPartitionA),
            3 => Some(Self::DataPartitionB),
            4 => Some(Self::DataPartitionC),
            5 => Some(Self::IdrSlice),
            6 => Some(Self::Sei),
            7 => Some(Self::Sps),
            8 => Some(Self::Pps),
            9 => Some(Self::AccessUnitDelimiter),
            10 => Some(Self::SeqEnd),
            11 => Some(Self::StreamEnd),
            12 => Some(Self::FillerData),
            13 => Some(Self::SpsExt),
            14 => Some(Self::AuxiliaryCodecLayer),
            15 => Some(Self::CodedSliceExtension),
            _ => None,
        }
    }

    /// Check if this NAL unit type contains slice data.
    pub const fn is_slice(&self) -> bool {
        matches!(
            self,
            Self::NonIdrSlice
                | Self::DataPartitionA
                | Self::DataPartitionB
                | Self::DataPartitionC
                | Self::IdrSlice
        )
    }

    /// Check if this NAL unit type is a parameter set.
    pub const fn is_parameter_set(&self) -> bool {
        matches!(self, Self::Sps | Self::SpsExt | Self::Pps)
    }
}

/// H.265 NAL unit types (raw values from H.265 spec).
///
/// Per ITU-T H.265 Table 7-1:
/// 0: TRAIL_N, 1: TRAIL_R, 2: TSA_N, 3: TSA_R, 4: STSA_N, 5: STSA_R,
/// 6: RADL_N, 7: RADL_R, 8: RASL_N, 9: RASL_R,
/// 16: IDR_W_RADL, 17: IDR_N_LP, 18: BLA_W_LP, 19: BLA_W_RADL,
/// 20: BLA_N_LP, 21: CRA_NUT,
/// 32: VPS, 33: SPS, 34: PPS, 35: PREFIX_SEI, 36: SUFFIX_SEI,
/// 37: FD_H, 38: RESERVED, 39: RSV_IRAP_V39, 40: RSV_VCL_N40, 41: RSV_VCL_N41
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum H265NalUnitType {
    /// Trailing Coded slice segment without RASL (TRAIL_N) - non-reference
    TraiNalN = 0,
    /// Trailing Coded slice segment with RASL (TRAIL_R) - reference
    TraiNalR = 1,
    /// Coded slice segment of a TSA picture (TSA_N) - non-reference
    TsaN = 2,
    /// Coded slice segment of a TSA picture (TSA_R) - reference
    TsaR = 3,
    /// Coded slice segment of a STSA picture (STSA_N) - non-reference
    StsaN = 4,
    /// Coded slice segment of a STSA picture (STSA_R) - reference
    StsaR = 5,
    /// Coded slice segment of a RADL picture (RADL_N) - non-reference
    RadlN = 6,
    /// Coded slice segment of a RADL picture (RADL_R) - reference
    RadlR = 7,
    /// Coded slice segment of a RASL picture (RASL_N) - non-reference
    RaslN = 8,
    /// Coded slice segment of a RASL picture (RASL_R) - reference
    RaslR = 9,
    // 10-15: Reserved or extension types
    /// Coded slice segment of an IDR picture (IDR_W_RADL) - reference
    CodNalSliceIdrW = 16,
    /// Coded slice segment of an IDR picture (IDR_N_LP) - reference
    CodNalSliceIdrN = 17,
    /// Coded slice segment of a BLA picture (BLA_W_LP) - reference
    CodNalSliceBlaW = 18,
    /// Coded slice segment of a BLA picture (BLA_W_RADL) - reference
    CodNalSliceBlaRadl = 19,
    /// Coded slice segment of a BLA picture (BLA_N_LP) - reference
    CodNalSliceBlaN = 20,
    /// Coded slice segment of a CRA picture (CRA_NUT) - reference
    CodNalSliceCra = 21,
    // 22-25: Reserved
    // 26-31: Reserved
    /// Video parameter set
    Vps = 32,
    /// Sequence parameter set
    Sps = 33,
    /// Picture parameter set
    Pps = 34,
    /// Prefix Supplemental enhancement information
    SeiPrefix = 35,
    /// Suffix Supplemental enhancement information
    SeiSuffix = 36,
    /// Access unit delimiter
    AccessUnitDelimiter = 37,
    /// End of sequence
    EndOfSequence = 38,
    /// End of bitstream
    EndOfBitstream = 39,
    /// Filler data
    FillerData = 40,
    /// Prefix SEI NAL unit
    SeiPrefixN = 41,
    /// Suffix SEI NAL unit
    SeiSuffixN = 42,
    // 43-45: Reserved
    // 46-47: Reserved
    // 48-63: Undefined
}

impl H265NalUnitType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::TraiNalN),
            1 => Some(Self::TraiNalR),
            2 => Some(Self::TsaN),
            3 => Some(Self::TsaR),
            4 => Some(Self::StsaN),
            5 => Some(Self::StsaR),
            6 => Some(Self::RadlN),
            7 => Some(Self::RadlR),
            8 => Some(Self::RaslN),
            9 => Some(Self::RaslR),
            16 => Some(Self::CodNalSliceIdrW),
            17 => Some(Self::CodNalSliceIdrN),
            18 => Some(Self::CodNalSliceBlaW),
            19 => Some(Self::CodNalSliceBlaRadl),
            20 => Some(Self::CodNalSliceBlaN),
            21 => Some(Self::CodNalSliceCra),
            32 => Some(Self::Vps),
            33 => Some(Self::Sps),
            34 => Some(Self::Pps),
            35 => Some(Self::SeiPrefix),
            36 => Some(Self::SeiSuffix),
            37 => Some(Self::AccessUnitDelimiter),
            38 => Some(Self::EndOfSequence),
            39 => Some(Self::EndOfBitstream),
            40 => Some(Self::FillerData),
            41 => Some(Self::SeiPrefixN),
            42 => Some(Self::SeiSuffixN),
            _ => None,
        }
    }

    /// Check if this NAL unit type contains slice data.
    pub const fn is_slice(&self) -> bool {
        matches!(
            self,
            Self::TraiNalN | Self::TraiNalR
                | Self::TsaN | Self::TsaR | Self::StsaN | Self::StsaR
                | Self::RadlN | Self::RadlR | Self::RaslN | Self::RaslR
                | Self::CodNalSliceIdrW | Self::CodNalSliceIdrN
                | Self::CodNalSliceBlaW | Self::CodNalSliceBlaRadl
                | Self::CodNalSliceBlaN | Self::CodNalSliceCra
        )
    }

    /// Check if this NAL unit type is an IRAP picture (random access point).
    /// IRAP = IDR, BLA, CRA, or Reserved IRAP
    pub const fn is_irap(&self) -> bool {
        matches!(
            self,
            Self::CodNalSliceIdrW | Self::CodNalSliceIdrN
                | Self::CodNalSliceBlaW | Self::CodNalSliceBlaRadl
                | Self::CodNalSliceBlaN | Self::CodNalSliceCra
        )
    }

    /// Check if this NAL unit type is an IDR picture.
    pub const fn is_idr(&self) -> bool {
        matches!(self, Self::CodNalSliceIdrW | Self::CodNalSliceIdrN)
    }

    /// Check if this NAL unit type is a parameter set.
    pub const fn is_parameter_set(&self) -> bool {
        matches!(self, Self::Vps | Self::Sps | Self::Pps)
    }

    /// Check if this NAL unit type is a reference picture.
    /// Per H.265 spec: odd VCL NAL types (1,3,5,7,9,11,13,15,17,19,21) are reference.
    pub const fn is_reference(&self) -> bool {
        matches!(
            self,
            Self::TraiNalR | Self::TsaR | Self::StsaR | Self::RadlR | Self::RaslR
                | Self::CodNalSliceIdrW | Self::CodNalSliceIdrN
                | Self::CodNalSliceBlaW | Self::CodNalSliceBlaRadl
                | Self::CodNalSliceBlaN | Self::CodNalSliceCra
        )
    }
}

/// Generic NAL unit (codec-agnostic).
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// NAL unit type.
    pub nal_unit_type: u8,
    /// NAL unit data (without start code).
    pub data: Vec<u8>,
    /// Original offset in the bitstream.
    pub offset: usize,
    /// Size including start code.
    pub size: usize,
}

impl NalUnit {
    /// Create a new NAL unit.
    pub fn new(nal_unit_type: u8, data: Vec<u8>, offset: usize, size: usize) -> Self {
        Self {
            nal_unit_type,
            data,
            offset,
            size,
        }
    }

    /// Check if this is an H.264 NAL unit.
    pub fn is_h264(&self) -> bool {
        self.nal_unit_type < 32
    }

    /// Check if this is an H.265 NAL unit.
    pub fn is_h265(&self) -> bool {
        matches!(
            self.nal_unit_type,
            0..=14 | 32..=42
        )
    }

    /// Check if this NAL unit contains slice data.
    pub fn is_slice(&self, codec: CodecType) -> bool {
        match codec {
            CodecType::H264 => {
                H264NalUnitType::from_u8(self.nal_unit_type)
                    .map(|t| t.is_slice())
                    .unwrap_or(false)
            }
            CodecType::H265 => {
                H265NalUnitType::from_u8(self.nal_unit_type)
                    .map(|t| t.is_slice())
                    .unwrap_or(false)
            }
            CodecType::Av1 => false,
        }
    }

    /// Check if this NAL unit is a parameter set.
    pub fn is_parameter_set(&self, codec: CodecType) -> bool {
        match codec {
            CodecType::H264 => {
                H264NalUnitType::from_u8(self.nal_unit_type)
                    .map(|t| t.is_parameter_set())
                    .unwrap_or(false)
            }
            CodecType::H265 => {
                H265NalUnitType::from_u8(self.nal_unit_type)
                    .map(|t| t.is_parameter_set())
                    .unwrap_or(false)
            }
            CodecType::Av1 => false,
        }
    }
}

/// Codec type for NAL unit parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    H264,
    H265,
    Av1,
}

/// Start code patterns.
pub const START_CODE_3: &[u8] = b"\x00\x00\x01";
pub const START_CODE_4: &[u8] = b"\x00\x00\x00\x01";
pub const EMULATION_PREVENTION_BYTE: u8 = 0x03;

/// Find the next start code in a byte stream.
///
/// Returns the offset of the start code and its length (3 or 4 bytes).
/// Checks for 4-byte start code (0x00 0x00 0x00 0x01) before 3-byte (0x00 0x00 0x01).
///
/// IMPORTANT: This function only finds start codes that are NOT preceded by 0x00.
/// This prevents matching 0x00 0x00 0x01 sequences that appear inside RBSP data
/// (which should only match 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01 at the
/// beginning of a NAL unit, not inside RBSP data).
pub fn find_next_start_code(data: &[u8], start: usize) -> Option<(usize, usize)> {
    if start >= data.len() {
        return None;
    }

    let remaining = &data[start..];
    let mut i = 0;
    while i + 2 < remaining.len() {
        // Check for 4-byte start code first: 0x00 0x00 0x00 0x01
        // Must NOT be preceded by 0x00 (to avoid matching inside RBSP data)
        if i + 3 < remaining.len()
            && remaining[i] == 0
            && remaining[i + 1] == 0
            && remaining[i + 2] == 0
            && remaining[i + 3] == 1
        {
            if i == 0 || remaining[i - 1] != 0 {
                return Some((start + i, 4));
            }
        }
        // Check for 3-byte start code: 0x00 0x00 0x01
        // Must NOT be preceded by 0x00 (to avoid matching inside RBSP data)
        else if remaining[i] == 0 && remaining[i + 1] == 0 && remaining[i + 2] == 1 {
            if i == 0 || remaining[i - 1] != 0 {
                return Some((start + i, 3));
            }
        }
        i += 1;
    }
    None
}

/// Parse NAL unit header for H.264.
///
/// Returns (forbidden_zero_bit, nal_ref_idc, nal_unit_type).
pub fn parse_h264_nal_header(data: &[u8]) -> Option<(bool, u8, u8)> {
    if data.is_empty() {
        return None;
    }
    let first_byte = data[0];
    let forbidden_zero_bit = (first_byte & 0x80) != 0;
    let nal_ref_idc = (first_byte & 0x60) >> 5;
    let nal_unit_type = first_byte & 0x1F;
    Some((forbidden_zero_bit, nal_ref_idc, nal_unit_type))
}

 /// Parse NAL unit header for H.265.
 ///
 /// Returns (forbidden_zero_bit, nal_unit_type, nuh_layer_id, nuh_temporal_id_plus1).
 pub fn parse_h265_nal_header(data: &[u8]) -> Option<(bool, u8, u16, u8)> {
      if data.is_empty() {
          return None;
      }
      let first_byte = data[0];
      let second_byte = if data.len() > 1 { data[1] } else { 0 };
      let forbidden_zero_bit = (first_byte & 0x80) != 0;
      let nal_unit_type = (first_byte & 0x7E) >> 1;
      let nuh_layer_id: u16 = (((first_byte & 0x01) as u16) << 6) | (((second_byte & 0xFC) as u16) >> 2);
      let nuh_temporal_id_plus1 = second_byte & 0x07;
      Some((forbidden_zero_bit, nal_unit_type, nuh_layer_id, nuh_temporal_id_plus1))
  }

/// Remove emulation prevention bytes from RBSP data.
///
/// In H.264/H.265 bitstreams, the byte sequence `0x00 0x00 0x03` is used
/// to prevent confusion with start codes. The `0x03` is an emulation
/// prevention byte and should be removed when parsing RBSP data.
pub fn remove_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == EMULATION_PREVENTION_BYTE
        {
            result.push(data[i]);
            result.push(data[i + 1]);
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}

/// Add emulation prevention bytes to RBSP data.
pub fn add_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] <= EMULATION_PREVENTION_BYTE
        {
            result.push(data[i]);
            result.push(data[i + 1]);
            result.push(EMULATION_PREVENTION_BYTE);
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}

/// Generic NAL unit type alias (codec-agnostic).
pub type NalUnitType = u8;

/// Type alias for H.264 NAL unit type.
pub type H264NalUnitTypeAlias = H264NalUnitType;

/// Type alias for H.265 NAL unit type.
pub type H265NalUnitTypeAlias = H265NalUnitType;
