//! Port of `hevc/common/types.h` — shared enums, structs and helpers.

/// Clip3 — spec §5.9.
#[inline]
pub fn clip3<T: Ord>(min_val: T, max_val: T, x: T) -> T {
    x.clamp(min_val, max_val)
}

/// Pixel type — 16-bit for 8/10-bit support (AD-002).
pub type Pixel = u16;

/// Motion vector (1/4 pel precision).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Mv {
    pub x: i16,
    pub y: i16,
}

/// Chroma format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ChromaFormat {
    Monochrome = 0,
    Yuv420 = 1,
    Yuv422 = 2,
    Yuv444 = 3,
}

/// Derived chroma subsampling (spec §7.4.3.2.1).
#[inline]
pub fn sub_width_c(fmt: ChromaFormat) -> u32 {
    if fmt == ChromaFormat::Yuv444 {
        1
    } else {
        2
    }
}

/// Derived chroma subsampling (spec §7.4.3.2.1).
#[inline]
pub fn sub_height_c(fmt: ChromaFormat) -> u32 {
    if fmt == ChromaFormat::Yuv420 {
        2
    } else {
        1
    }
}

/// NAL unit type — spec §7.4.2.2, Table 7-1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    TrailN = 0,
    TrailR = 1,
    TsaN = 2,
    TsaR = 3,
    StsaN = 4,
    StsaR = 5,
    RadlN = 6,
    RadlR = 7,
    RaslN = 8,
    RaslR = 9,
    RsvVclN10 = 10,
    RsvVclR11 = 11,
    RsvVclN12 = 12,
    RsvVclR13 = 13,
    RsvVclN14 = 14,
    RsvVclR15 = 15,
    BlaWLP = 16,
    BlaWRadl = 17,
    BlaNLP = 18,
    IdrWRadl = 19,
    IdrNLP = 20,
    CraNut = 21,
    RsvIrap22 = 22,
    RsvIrap23 = 23,
    VpsNut = 32,
    SpsNut = 33,
    PpsNut = 34,
    AudNut = 35,
    EosNut = 36,
    EobNut = 37,
    FdNut = 38,
    PrefixSei = 39,
    SuffixSei = 40,
}

impl NalUnitType {
    pub const fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::TrailN,
            1 => Self::TrailR,
            2 => Self::TsaN,
            3 => Self::TsaR,
            4 => Self::StsaN,
            5 => Self::StsaR,
            6 => Self::RadlN,
            7 => Self::RadlR,
            8 => Self::RaslN,
            9 => Self::RaslR,
            10 => Self::RsvVclN10,
            11 => Self::RsvVclR11,
            12 => Self::RsvVclN12,
            13 => Self::RsvVclR13,
            14 => Self::RsvVclN14,
            15 => Self::RsvVclR15,
            16 => Self::BlaWLP,
            17 => Self::BlaWRadl,
            18 => Self::BlaNLP,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLP,
            21 => Self::CraNut,
            22 => Self::RsvIrap22,
            23 => Self::RsvIrap23,
            32 => Self::VpsNut,
            33 => Self::SpsNut,
            34 => Self::PpsNut,
            35 => Self::AudNut,
            36 => Self::EosNut,
            37 => Self::EobNut,
            38 => Self::FdNut,
            39 => Self::PrefixSei,
            40 => Self::SuffixSei,
            _ => return None,
        })
    }

    /// VCL NAL unit (types 0..=31).
    pub const fn is_vcl(self) -> bool {
        (self as u8) <= 31
    }

    /// IRAP NAL unit (types 16..=23).
    pub const fn is_irap(self) -> bool {
        matches!(self as u8, 16..=23)
    }

    pub const fn is_idr(self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLP)
    }

    pub const fn is_cra(self) -> bool {
        (self as u8) == 21
    }

    pub const fn is_bla(self) -> bool {
        matches!(self as u8, 16..=18)
    }

    pub const fn is_rasl(self) -> bool {
        matches!(self, Self::RaslN | Self::RaslR)
    }

    pub const fn is_radl(self) -> bool {
        matches!(self, Self::RadlN | Self::RadlR)
    }
}

/// Slice type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

/// Prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PredMode {
    Inter = 0,
    Intra = 1,
    Skip = 2,
}

/// Partition mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PartMode {
    Part2Nx2N = 0,
    Part2NxN = 1,
    PartNx2N = 2,
    PartNxN = 3,
    Part2NxnU = 4,
    Part2NxnD = 5,
    PartNlx2N = 6,
    PartNRx2N = 7,
}

/// NAL unit header.
#[derive(Clone, Copy, Debug)]
pub struct NalUnitHeader {
    pub nal_unit_type: NalUnitType,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
}

impl NalUnitHeader {
    pub const fn temporal_id(&self) -> u8 {
        self.nuh_temporal_id_plus1 - 1
    }
}

/// View over one component plane (data + dims) — used by the loop filters,
/// mirroring `Picture::planes[c]` / `width[c]` / `height[c]` / `stride[c]`.
pub struct Plane<'a> {
    pub data: &'a mut [u16],
    pub width: i32,
    pub height: i32,
    pub stride: i32,
}

/// Tile layout for loop-filter boundary checks (PPS-derived). `None` at the
/// call site means "no tiles" (empty `pps.TileId`).
#[derive(Clone, Copy, Debug)]
pub struct Tiles<'a> {
    /// `TileId[ts]` — tile id per tile-scan address.
    pub tile_id: &'a [i32],
    /// `CtbAddrRsToTs` — raster CTB address -> tile-scan address.
    pub ctb_addr_rs_to_ts: &'a [i32],
}

/// Sequence Parameter Set — subset of spec §7.4.3.2.1 fields consumed by the
/// Rust kernels. Extended as later sprints port more of the decoder.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sps {
    pub pic_width_in_luma_samples: i32,
    pub pic_height_in_luma_samples: i32,
    pub bit_depth_y: i32,
    pub bit_depth_c: i32,
    /// ChromaArrayType (0=monochrome, 1=4:2:0, 2=4:2:2, 3=4:4:4).
    pub chroma_array_type: i32,
    pub ctb_size_y: i32,
    pub min_tb_size_y: i32,
    pub pic_width_in_ctbs_y: i32,
    pub sub_width_c: i32,
    pub sub_height_c: i32,
    pub intra_smoothing_disabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_unit_type_roundtrip() {
        for v in 0..=255u8 {
            let t = NalUnitType::from_u8(v);
            assert_eq!(t.is_some(), matches!(v, 0..=23 | 32..=40), "v={v}");
            if let Some(t) = t {
                assert_eq!(t as u8, v);
            }
        }
    }

    #[test]
    fn nal_unit_type_helpers() {
        assert!(NalUnitType::TrailN.is_vcl());
        assert!(NalUnitType::RsvVclR15.is_vcl());
        assert!(!NalUnitType::VpsNut.is_vcl());
        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(!NalUnitType::TrailR.is_irap());
        assert!(NalUnitType::IdrNLP.is_idr());
        assert!(!NalUnitType::CraNut.is_idr());
        assert!(NalUnitType::BlaWLP.is_bla());
        assert!(!NalUnitType::IdrWRadl.is_bla());
        assert!(NalUnitType::RaslR.is_rasl());
        assert!(NalUnitType::RadlN.is_radl());
    }

    #[test]
    fn chroma_subsampling() {
        assert_eq!(sub_width_c(ChromaFormat::Yuv420), 2);
        assert_eq!(sub_width_c(ChromaFormat::Yuv444), 1);
        assert_eq!(sub_height_c(ChromaFormat::Yuv420), 2);
        assert_eq!(sub_height_c(ChromaFormat::Yuv422), 1);
    }

    #[test]
    fn clip3_bounds() {
        assert_eq!(clip3(0, 100, -5), 0);
        assert_eq!(clip3(0, 100, 150), 100);
        assert_eq!(clip3(0, 100, 50), 50);
    }
}
