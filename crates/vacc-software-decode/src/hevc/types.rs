//! Port of `hevc/common/types.h` — shared enums, structs and helpers.

/// Clip3 — spec §5.9 (single implementation in `vacc_common::clip`).
pub use vacc_common::clip::clip3;

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

impl PartMode {
    /// Inverse of `decode_part_mode`'s integer encoding (0..=7).
    pub const fn from_u32(v: u32) -> Self {
        match v {
            1 => PartMode::Part2NxN,
            2 => PartMode::PartNx2N,
            3 => PartMode::PartNxN,
            4 => PartMode::Part2NxnU,
            5 => PartMode::Part2NxnD,
            6 => PartMode::PartNlx2N,
            7 => PartMode::PartNRx2N,
            _ => PartMode::Part2Nx2N,
        }
    }
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

    // ---- Tier E (coding tree) derived fields (§7.4.3.2.1) ----
    pub min_cb_log2_size_y: i32,
    pub ctb_log2_size_y: i32,
    pub min_cb_size_y: i32,
    pub pic_height_in_ctbs_y: i32,
    pub pic_size_in_ctbs_y: i32,
    pub min_tb_log2_size_y: i32,
    pub max_tb_log2_size_y: i32,
    pub qp_bd_offset_y: i32,
    pub qp_bd_offset_c: i32,
    pub amp_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: i32,
    pub pcm_sample_bit_depth_chroma_minus1: i32,
    pub log2_min_ipcm_cb_size_y: i32,
    pub log2_max_ipcm_cb_size_y: i32,
    pub max_transform_hierarchy_depth_inter: i32,
    pub max_transform_hierarchy_depth_intra: i32,
    /// SPS RExt flag — Main profile infers 0.
    pub cabac_bypass_alignment_enabled_flag: bool,
    /// sample_adaptive_offset_enabled_flag (gates SAO application).
    pub sample_adaptive_offset_enabled_flag: bool,
    /// pcm_loop_filter_disabled_flag.
    pub pcm_loop_filter_disabled_flag: bool,
}

/// Picture Parameter Set — fields consumed by the coding-tree kernels.
#[derive(Clone, Debug)]
pub struct Pps {
    pub sign_data_hiding_enabled_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    /// Derived (§7.4.3.2.1): `CtbLog2SizeY - diff_cu_qp_delta_depth`.
    pub log2_min_cu_qp_delta_size: i32,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// loop_filter_across_tiles_enabled_flag.
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub log2_parallel_merge_level_minus2: i32,

    // Tile scan tables (derived, §6.5.1). Identity mapping when tiles are
    // disabled (mirrors C++ `derive_tile_scan`, always called at parse time).
    /// Raster CTB address -> tile-scan address.
    pub ctb_addr_rs_to_ts: Vec<i32>,
    /// Tile-scan address -> raster CTB address.
    pub ctb_addr_ts_to_rs: Vec<i32>,
    /// `TileId[tsAddr]` — tile id per tile-scan address.
    pub tile_id: Vec<i32>,

    // Tile layout (needed to rebuild the scan tables in tests/oracle).
    pub num_tile_columns_minus1: i32,
    pub num_tile_rows_minus1: i32,
    pub uniform_spacing_flag: bool,
    pub column_width_minus1: Vec<u32>,
    pub row_height_minus1: Vec<u32>,
}

impl Default for Pps {
    fn default() -> Self {
        Pps {
            sign_data_hiding_enabled_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: false,
            log2_min_cu_qp_delta_size: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            weighted_pred_flag: false,
            weighted_bipred_flag: false,
            transquant_bypass_enabled_flag: false,
            tiles_enabled_flag: false,
            entropy_coding_sync_enabled_flag: false,
            pps_loop_filter_across_slices_enabled_flag: false,
            loop_filter_across_tiles_enabled_flag: false,
            log2_parallel_merge_level_minus2: 0,
            ctb_addr_rs_to_ts: Vec::new(),
            ctb_addr_ts_to_rs: Vec::new(),
            tile_id: Vec::new(),
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
        }
    }
}

impl Pps {
    /// Rebuild the tile scan tables from the layout fields (§6.5.1),
    /// mirroring C++ `PPS::derive_tile_scan`. Identity mapping for a single
    /// tile (tiles disabled).
    pub fn derive_tile_scan(&mut self, sps: &Sps) {
        let num_ctbs_y = sps.pic_size_in_ctbs_y;
        let pic_w_ctb = sps.pic_width_in_ctbs_y;
        let pic_h_ctb = sps.pic_height_in_ctbs_y;
        let num_tile_cols = self.num_tile_columns_minus1 + 1;
        let num_tile_rows = self.num_tile_rows_minus1 + 1;

        let col_width = |i: i32| -> i32 {
            if self.uniform_spacing_flag {
                ((i + 1) * pic_w_ctb) / num_tile_cols - (i * pic_w_ctb) / num_tile_cols
            } else {
                (self.column_width_minus1[i as usize] as i32) + 1
            }
        };
        let row_height = |i: i32| -> i32 {
            if self.uniform_spacing_flag {
                ((i + 1) * pic_h_ctb) / num_tile_rows - (i * pic_h_ctb) / num_tile_rows
            } else {
                (self.row_height_minus1[i as usize] as i32) + 1
            }
        };

        let mut col_bd = vec![0i32; num_tile_cols as usize + 1];
        for i in 0..num_tile_cols {
            let w = if self.uniform_spacing_flag || i == num_tile_cols - 1 {
                pic_w_ctb - col_bd[i as usize]
            } else {
                col_width(i)
            };
            col_bd[(i + 1) as usize] = col_bd[i as usize] + w;
        }
        let mut row_bd = vec![0i32; num_tile_rows as usize + 1];
        for i in 0..num_tile_rows {
            let h = if self.uniform_spacing_flag || i == num_tile_rows - 1 {
                pic_h_ctb - row_bd[i as usize]
            } else {
                row_height(i)
            };
            row_bd[(i + 1) as usize] = row_bd[i as usize] + h;
        }

        self.ctb_addr_rs_to_ts = vec![0; num_ctbs_y as usize];
        self.ctb_addr_ts_to_rs = vec![0; num_ctbs_y as usize];
        self.tile_id = vec![0; num_ctbs_y as usize];

        let mut ts_idx = 0i32;
        for tile_row in 0..num_tile_rows {
            for tile_col in 0..num_tile_cols {
                let tile_id = tile_row * num_tile_cols + tile_col;
                for y in row_bd[tile_row as usize]..row_bd[(tile_row + 1) as usize] {
                    for x in col_bd[tile_col as usize]..col_bd[(tile_col + 1) as usize] {
                        let rs_addr = y * pic_w_ctb + x;
                        self.ctb_addr_rs_to_ts[rs_addr as usize] = ts_idx;
                        self.ctb_addr_ts_to_rs[ts_idx as usize] = rs_addr;
                        self.tile_id[ts_idx as usize] = tile_id;
                        ts_idx += 1;
                    }
                }
            }
        }
    }
}

/// Slice Segment Header — fields consumed by the coding-tree kernels.
#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub slice_segment_address: i32,
    pub dependent_slice_segment_flag: bool,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    pub slice_temporal_mvp_enabled_flag: bool,
    pub slice_sao_luma_flag: bool,
    pub slice_sao_chroma_flag: bool,
    pub num_ref_idx_l0_active_minus1: i32,
    pub num_ref_idx_l1_active_minus1: i32,
    pub mvd_l1_zero_flag: bool,
    pub cabac_init_flag: bool,
    pub collocated_from_l0_flag: bool,
    pub collocated_ref_idx: i32,
    pub five_minus_max_num_merge_cand: i32,
    pub slice_qp_delta: i32,
    pub slice_cb_qp_offset: i32,
    pub slice_cr_qp_offset: i32,
    /// Derived: `26 + slice_qp_delta`.
    pub slice_qp_y: i32,
    /// Derived: `5 - five_minus_max_num_merge_cand`.
    pub max_num_merge_cand: i32,
    /// Explicit weighted prediction table (§7.3.6.3).
    pub pred_weight_table: crate::hevc::interpolation::PredWeightTable,
    pub num_entry_point_offsets: i32,
    pub entry_point_offset_minus1: Vec<u32>,
}

impl Default for SliceHeader {
    fn default() -> Self {
        SliceHeader {
            slice_segment_address: 0,
            dependent_slice_segment_flag: false,
            slice_type: SliceType::I,
            pic_output_flag: true,
            slice_temporal_mvp_enabled_flag: false,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            mvd_l1_zero_flag: false,
            cabac_init_flag: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            five_minus_max_num_merge_cand: 0,
            slice_qp_delta: 0,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_qp_y: 26,
            max_num_merge_cand: 5,
            pred_weight_table: Default::default(),
            num_entry_point_offsets: 0,
            entry_point_offset_minus1: Vec::new(),
        }
    }
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
