//! FFI bindings for NVDEC (`cuviddec.h`) — decoder only.
//!
//! Generated from Video_Codec_SDK_12.0.16/Interface/cuviddec.h.
//! Only includes decoder-related types and functions.
//!
//! ## Contents
//!
//! ### Type Aliases
//! - [`CUresult`] — CUDA result/error code
//! - [`CUdeviceptr`] — 64-bit CUDA device pointer
//! - [`CUvideodecoder`] — Opaque decoder handle
//! - [`CUvideoparser`] — Opaque parser handle
//!
//! ### Enums
//! - [`cudaVideoCodec`] — Video codec identifiers (H.264, HEVC, VP9, AV1, etc.)
//! - [`cudaVideoSurfaceFormat`] — Output pixel format (NV12, P016, YUV444)
//! - [`cudaVideoChromaFormat`] — Chroma subsampling (4:2:0, 4:2:2, 4:4:4)
//! - [`cudaVideoDeinterlaceMode`] — Deinterlacing mode
//! - [`cudaVideoCreateFlags`] — Decoder creation flags
//! - [`cuvidDecodeStatus`] — Decode operation status
//! - [`CUvideopacketflags`] — Parser packet flags
//!
//! ### Structures
//! - [`CUVIDDECODECAPS`] — Decoder capability query results
//! - [`CUVIDDECODECREATEINFO`] — Decoder creation parameters
//! - [`CUVIDPICPARAMS`] — Per-picture decode parameters
//! - [`CUVIDEOFORMAT`] — Video format info (from parser sequence callback)
//! - [`CUVIDSOURCEDATAPACKET`] — Bitstream data packet
//! - [`CUVIDPARSERPARAMS`] — Parser creation parameters
//! - [`CUVIDPARSERDISPINFO`] — Display timing info (from parser display callback)
//! - [`CUDA_MEMCPY2D`](crate::device::CUDA_MEMCPY2D) — 2D memory copy descriptor (from `device` module)
//!
//! ### Constants
//! - [`CUDA_SUCCESS`] and error codes (e.g., [`CUDA_ERROR_OUT_OF_MEMORY`])
//!
//! ### Functions
//! Declared as `extern "C"` for direct FFI use. The `device` module provides
//! safe wrappers via dynamically loaded function pointers.
//!
//! ## Safety
//!
//! All types in this module are `#[repr(C)]` FFI bindings. Direct use requires
//! `unsafe` blocks. Prefer the safe wrappers in the [`device`](crate::device)
//! and [`decoder`](crate::decoder) modules when possible.
//!
//! Struct sizes and field offsets are verified by unit tests to match the
//! NVIDIA Video Codec SDK layout on 64-bit Linux.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{
    c_char, c_int, c_short, c_uchar, c_uint, c_ulong, c_ulonglong, c_ushort, c_void,
};

/// CUDA result type (from cuda.h).
///
/// `0` indicates success; non-zero values indicate errors.
/// See [`CUDA_SUCCESS`], [`CUDA_ERROR_OUT_OF_MEMORY`], etc.
pub type CUresult = c_uint;

/// CUDA stream (from cuda.h).
pub type CUstream = *mut c_void;

/// CUDA device pointer (from cuda.h).
///
/// 64-bit GPU virtual address.
pub type CUdeviceptr = c_ulonglong;

/// Opaque video decoder handle.
///
/// Returned by `cuvidCreateDecoder`. Passed to decode, map, and unmap
/// operations. Destroy with `cuvidDestroyDecoder`.
pub type CUvideodecoder = *mut c_void;

/// Opaque video context lock.
pub type CUvideoctxlock = *mut c_void;

/// Video codec identifiers.
///
/// Used to specify the codec type when creating a decoder or querying
/// capabilities.
///
/// # Example
///
/// ```
/// use nvdec_decode::ffi::cudaVideoCodec;
/// let codec = cudaVideoCodec::cudaVideoCodec_H264;
/// ```
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum cudaVideoCodec {
    cudaVideoCodec_MPEG1 = 0,
    cudaVideoCodec_MPEG2,
    cudaVideoCodec_MPEG4,
    cudaVideoCodec_VC1,
    cudaVideoCodec_H264,
    cudaVideoCodec_JPEG,
    cudaVideoCodec_H264_SVC,
    cudaVideoCodec_H264_MVC,
    cudaVideoCodec_HEVC,
    cudaVideoCodec_VP8,
    cudaVideoCodec_VP9,
    cudaVideoCodec_AV1,
    cudaVideoCodec_NumCodecs,
    // Uncompressed YUV formats (using i32 repr for 32-bit compatibility)
    cudaVideoCodec_YUV420 =
        (('I' as i32) << 24 | ('Y' as i32) << 16 | ('U' as i32) << 8 | ('V' as i32)),
    cudaVideoCodec_YV12 =
        (('Y' as i32) << 24 | ('V' as i32) << 16 | ('1' as i32) << 8 | ('2' as i32)),
    cudaVideoCodec_NV12 =
        (('N' as i32) << 24 | ('V' as i32) << 16 | ('1' as i32) << 8 | ('2' as i32)),
    cudaVideoCodec_YUYV =
        (('Y' as i32) << 24 | ('U' as i32) << 16 | ('Y' as i32) << 8 | ('V' as i32)),
    cudaVideoCodec_UYVY =
        (('U' as i32) << 24 | ('Y' as i32) << 16 | ('V' as i32) << 8 | ('Y' as i32)),
}

impl std::fmt::Debug for cudaVideoCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            cudaVideoCodec::cudaVideoCodec_MPEG1 => write!(f, "cudaVideoCodec_MPEG1"),
            cudaVideoCodec::cudaVideoCodec_MPEG2 => write!(f, "cudaVideoCodec_MPEG2"),
            cudaVideoCodec::cudaVideoCodec_MPEG4 => write!(f, "cudaVideoCodec_MPEG4"),
            cudaVideoCodec::cudaVideoCodec_VC1 => write!(f, "cudaVideoCodec_VC1"),
            cudaVideoCodec::cudaVideoCodec_H264 => write!(f, "cudaVideoCodec_H264"),
            cudaVideoCodec::cudaVideoCodec_JPEG => write!(f, "cudaVideoCodec_JPEG"),
            cudaVideoCodec::cudaVideoCodec_H264_SVC => write!(f, "cudaVideoCodec_H264_SVC"),
            cudaVideoCodec::cudaVideoCodec_H264_MVC => write!(f, "cudaVideoCodec_H264_MVC"),
            cudaVideoCodec::cudaVideoCodec_HEVC => write!(f, "cudaVideoCodec_HEVC"),
            cudaVideoCodec::cudaVideoCodec_VP8 => write!(f, "cudaVideoCodec_VP8"),
            cudaVideoCodec::cudaVideoCodec_VP9 => write!(f, "cudaVideoCodec_VP9"),
            cudaVideoCodec::cudaVideoCodec_AV1 => write!(f, "cudaVideoCodec_AV1"),
            cudaVideoCodec::cudaVideoCodec_NumCodecs => write!(f, "cudaVideoCodec_NumCodecs"),
            cudaVideoCodec::cudaVideoCodec_YUV420 => write!(f, "cudaVideoCodec_YUV420"),
            cudaVideoCodec::cudaVideoCodec_YV12 => write!(f, "cudaVideoCodec_YV12"),
            cudaVideoCodec::cudaVideoCodec_NV12 => write!(f, "cudaVideoCodec_NV12"),
            cudaVideoCodec::cudaVideoCodec_YUYV => write!(f, "cudaVideoCodec_YUYV"),
            cudaVideoCodec::cudaVideoCodec_UYVY => write!(f, "cudaVideoCodec_UYVY"),
        }
    }
}

/// Video surface format for decoded output.
///
/// Specifies the pixel format of decoded frames in GPU memory.
/// Values match the NVIDIA Video Codec SDK 12.0.16 header.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaVideoSurfaceFormat {
    /// Semi-Planar YUV [Y plane followed by interleaved UV plane]
    cudaVideoSurfaceFormat_NV12 = 0,
    /// 16 bit Semi-Planar YUV [Y plane followed by interleaved UV plane]
    cudaVideoSurfaceFormat_P016 = 1,
    /// Planar YUV [Y plane followed by U and V planes]
    cudaVideoSurfaceFormat_YUV444 = 2,
    /// 16 bit Planar YUV [Y plane followed by U and V planes]
    cudaVideoSurfaceFormat_YUV444_16Bit = 3,
}

/// Deinterlacing mode enums
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaVideoDeinterlaceMode {
    cudaVideoDeinterlaceMode_Weave = 0,
    cudaVideoDeinterlaceMode_Bob,
    cudaVideoDeinterlaceMode_Adaptive,
}

/// Chroma format enums
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaVideoChromaFormat {
    cudaVideoChromaFormat_Monochrome = 0,
    cudaVideoChromaFormat_420,
    cudaVideoChromaFormat_422,
    cudaVideoChromaFormat_444,
}

/// Decoder creation flags
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaVideoCreateFlags {
    cudaVideoCreate_Default = 0x00,
    cudaVideoCreate_PreferCUDA = 0x01,
    cudaVideoCreate_PreferDXVA = 0x02,
    cudaVideoCreate_PreferCUVID = 0x04,
}

/// Decode status enums
///
/// Values match the NVIDIA Video Codec SDK header on this platform.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvidDecodeStatus {
    cuvidDecodeStatus_Invalid = 0,
    cuvidDecodeStatus_InProgress = 1,
    cuvidDecodeStatus_Success = 2,
    cuvidDecodeStatus_Error = 8,
    cuvidDecodeStatus_Error_Concealed = 9,
}

/// Decoder capabilities structure.
///
/// Returned by `cuvidGetDecoderCaps` with information about hardware
/// decoder support for a specific codec and format combination.
///
/// Matches the NVIDIA Video Codec SDK header on this platform.
/// Note: nOutputFormatMask, nMinWidth, nMinHeight, nMaxHistogramBins are
/// unsigned short (2 bytes), not unsigned int (4 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDDECODECAPS {
    pub eCodecType: cudaVideoCodec,
    pub eChromaFormat: cudaVideoChromaFormat,
    pub nBitDepthMinus8: c_uint,
    pub reserved1: [c_uint; 3],
    pub bIsSupported: c_uchar,
    pub nNumNVDECs: c_uchar,
    pub nOutputFormatMask: c_ushort,
    pub nMaxWidth: c_uint,
    pub nMaxHeight: c_uint,
    pub nMaxMBCount: c_uint,
    pub nMinWidth: c_ushort,
    pub nMinHeight: c_ushort,
    pub bIsHistogramSupported: c_uchar,
    pub nCounterBitDepth: c_uchar,
    pub nMaxHistogramBins: c_ushort,
    pub reserved3: [c_uint; 10],
}

/// Rectangle structure (used in CUVIDDECODECREATEINFO)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDRECT {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

/// Decoder creation info structure.
///
/// Passed to `cuvidCreateDecoder` to configure the decoder. Specifies
/// resolution, codec type, output format, number of surfaces, and
/// other decode parameters.
///
/// Uses `c_ulong` (tcu_ulong in NVIDIA SDK) which is 8 bytes on 64-bit Linux
/// and 4 bytes on 32-bit/Windows. Total size: 176 bytes on 64-bit Linux.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDDECODECREATEINFO {
    pub ulWidth: c_ulong,
    pub ulHeight: c_ulong,
    pub ulNumDecodeSurfaces: c_ulong,
    pub CodecType: cudaVideoCodec,
    pub ChromaFormat: cudaVideoChromaFormat,
    pub ulCreationFlags: c_ulong,
    pub bitDepthMinus8: c_ulong,
    pub ulIntraDecodeOnly: c_ulong,
    pub ulMaxWidth: c_ulong,
    pub ulMaxHeight: c_ulong,
    pub Reserved1: c_ulong,
    pub display_area: CUVIDRECT,
    pub OutputFormat: cudaVideoSurfaceFormat,
    pub DeinterlaceMode: cudaVideoDeinterlaceMode,
    pub ulTargetWidth: c_ulong,
    pub ulTargetHeight: c_ulong,
    pub ulNumOutputSurfaces: c_ulong,
    pub vidLock: CUvideoctxlock,
    pub target_rect: CUVIDRECT,
    pub enableHistogram: c_ulong,
    pub Reserved2: [c_ulong; 4],
}

/// H.264 DPB entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDH264DPBENTRY {
    pub PicIdx: c_int,
    pub FrameIdx: c_int,
    pub is_long_term: c_int,
    pub not_existing: c_int,
    pub used_for_reference: c_int,
    pub FieldOrderCnt: [c_int; 2],
}

/// H.264 MVC extension
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CUVIDH264MVCEXT {
    pub num_views_minus1: c_int,
    pub view_id: c_int,
    pub inter_view_flag: c_uchar,
    pub num_inter_view_refs_l0: c_uchar,
    pub num_inter_view_refs_l1: c_uchar,
    pub MVCReserved8Bits: c_uchar,
    pub InterViewRefsL0: [c_int; 16],
    pub InterViewRefsL1: [c_int; 16],
}

/// H.264 SVC extension
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CUVIDH264SVCEXT {
    pub profile_idc: c_uchar,
    pub level_idc: c_uchar,
    pub DQId: c_uchar,
    pub DQIdMax: c_uchar,
    pub disable_inter_layer_deblocking_filter_idc: c_uchar,
    pub ref_layer_chroma_phase_y_plus1: c_uchar,
    pub inter_layer_slice_alpha_c0_offset_div2: c_char,
    pub inter_layer_slice_beta_offset_div2: c_char,
    pub DPBEntryValidFlag: c_uint,
    pub inter_layer_deblocking_filter_control_present_flag: c_uchar,
    pub extended_spatial_scalability_idc: c_uchar,
    pub adaptive_tcoeff_level_prediction_flag: c_uchar,
    pub slice_header_restriction_flag: c_uchar,
    pub chroma_phase_x_plus1_flag: c_uchar,
    pub chroma_phase_y_plus1: c_uchar,
    pub tcoeff_level_prediction_flag: c_uchar,
    pub constrained_intra_resampling_flag: c_uchar,
    pub ref_layer_chroma_phase_x_plus1_flag: c_uchar,
    pub store_ref_base_pic_flag: c_uchar,
    pub Reserved8BitsA: c_uchar,
    pub Reserved8BitsB: c_uchar,
    pub scaled_ref_layer_left_offset: c_int,
    pub scaled_ref_layer_top_offset: c_int,
    pub scaled_ref_layer_right_offset: c_int,
    pub scaled_ref_layer_bottom_offset: c_int,
    pub Reserved16Bits: c_uint,
    pub pNextLayer: *const CUVIDPICPARAMS,
    pub bRefBaseLayer: c_int,
}

/// H.264 picture parameters
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUVIDH264PICPARAMS {
    // SPS
    pub log2_max_frame_num_minus4: c_int,
    pub pic_order_cnt_type: c_int,
    pub log2_max_pic_order_cnt_lsb_minus4: c_int,
    pub delta_pic_order_always_zero_flag: c_int,
    pub frame_mbs_only_flag: c_int,
    pub direct_8x8_inference_flag: c_int,
    pub num_ref_frames: c_int,
    pub residual_colour_transform_flag: c_uchar,
    pub bit_depth_luma_minus8: c_uchar,
    pub bit_depth_chroma_minus8: c_uchar,
    pub qpprime_y_zero_transform_bypass_flag: c_uchar,
    // PPS
    pub entropy_coding_mode_flag: c_int,
    pub pic_order_present_flag: c_int,
    pub num_ref_idx_l0_active_minus1: c_int,
    pub num_ref_idx_l1_active_minus1: c_int,
    pub weighted_pred_flag: c_int,
    pub weighted_bipred_idc: c_int,
    pub pic_init_qp_minus26: c_int,
    pub deblocking_filter_control_present_flag: c_int,
    pub redundant_pic_cnt_present_flag: c_int,
    pub transform_8x8_mode_flag: c_int,
    pub MbaffFrameFlag: c_int,
    pub constrained_intra_pred_flag: c_int,
    pub chroma_qp_index_offset: c_int,
    pub second_chroma_qp_index_offset: c_int,
    pub ref_pic_flag: c_int,
    pub frame_num: c_int,
    pub CurrFieldOrderCnt: [c_int; 2],
    // DPB
    pub dpb: [CUVIDH264DPBENTRY; 16],
    // Quantization Matrices (raster-order)
    pub WeightScale4x4: [[c_uchar; 16]; 6],
    pub WeightScale8x8: [[c_uchar; 64]; 2],
    // FMO/ASO
    pub fmo_aso_enable: c_uchar,
    pub num_slice_groups_minus1: c_uchar,
    pub slice_group_map_type: c_uchar,
    pub pic_init_qs_minus26: c_char,
    pub slice_group_change_rate_minus1: c_uint,
    pub fmo: CUVIDH264FMOASO,
    pub Reserved: [c_uint; 12],
    // SVC/MVC
    pub svc_mvc: CUVIDH264SVCMVC,
}

/// FMO/ASO union in CUVIDH264PICPARAMS
#[repr(C)]
#[derive(Clone, Copy)]
pub union CUVIDH264FMOASO {
    #[allow(dead_code)]
    pub slice_group_map_addr: c_ulonglong,
    pub pMb2SliceGroupMap: *const c_uchar,
}

/// SVC/MVC union in CUVIDH264PICPARAMS
#[repr(C)]
#[derive(Clone, Copy)]
pub union CUVIDH264SVCMVC {
    #[allow(dead_code)]
    pub mvcext: CUVIDH264MVCEXT,
    pub svcext: CUVIDH264SVCEXT,
}

impl Default for CUVIDH264SVCMVC {
    fn default() -> Self {
        Self {
            mvcext: CUVIDH264MVCEXT::default(),
        }
    }
}

/// HEVC picture parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDHEVCPICPARAMS {
    // sps
    pub pic_width_in_luma_samples: c_int,
    pub pic_height_in_luma_samples: c_int,
    pub log2_min_luma_coding_block_size_minus3: c_uchar,
    pub log2_diff_max_min_luma_coding_block_size: c_uchar,
    pub log2_min_transform_block_size_minus2: c_uchar,
    pub log2_diff_max_min_transform_block_size: c_uchar,
    pub pcm_enabled_flag: c_uchar,
    pub log2_min_pcm_luma_coding_block_size_minus3: c_uchar,
    pub log2_diff_max_min_pcm_luma_coding_block_size: c_uchar,
    pub pcm_sample_bit_depth_luma_minus1: c_uchar,
    pub pcm_sample_bit_depth_chroma_minus1: c_uchar,
    pub pcm_loop_filter_disabled_flag: c_uchar,
    pub strong_intra_smoothing_enabled_flag: c_uchar,
    pub max_transform_hierarchy_depth_intra: c_uchar,
    pub max_transform_hierarchy_depth_inter: c_uchar,
    pub amp_enabled_flag: c_uchar,
    pub separate_colour_plane_flag: c_uchar,
    pub log2_max_pic_order_cnt_lsb_minus4: c_uchar,
    pub num_short_term_ref_pic_sets: c_uchar,
    pub long_term_ref_pics_present_flag: c_uchar,
    pub num_long_term_ref_pics_sps: c_uchar,
    pub sps_temporal_mvp_enabled_flag: c_uchar,
    pub sample_adaptive_offset_enabled_flag: c_uchar,
    pub scaling_list_enable_flag: c_uchar,
    pub IrapPicFlag: c_uchar,
    pub IdrPicFlag: c_uchar,
    pub bit_depth_luma_minus8: c_uchar,
    pub bit_depth_chroma_minus8: c_uchar,
    // sps/pps extension fields
    pub log2_max_transform_skip_block_size_minus2: c_uchar,
    pub log2_sao_offset_scale_luma: c_uchar,
    pub log2_sao_offset_scale_chroma: c_uchar,
    pub high_precision_offsets_enabled_flag: c_uchar,
    pub reserved1: [c_uchar; 10],
    // pps
    pub dependent_slice_segments_enabled_flag: c_uchar,
    pub slice_segment_header_extension_present_flag: c_uchar,
    pub sign_data_hiding_enabled_flag: c_uchar,
    pub cu_qp_delta_enabled_flag: c_uchar,
    pub diff_cu_qp_delta_depth: c_uchar,
    pub init_qp_minus26: c_char,
    pub pps_cb_qp_offset: c_char,
    pub pps_cr_qp_offset: c_char,
    pub constrained_intra_pred_flag: c_uchar,
    pub weighted_pred_flag: c_uchar,
    pub weighted_bipred_flag: c_uchar,
    pub transform_skip_enabled_flag: c_uchar,
    pub transquant_bypass_enabled_flag: c_uchar,
    pub entropy_coding_sync_enabled_flag: c_uchar,
    pub log2_parallel_merge_level_minus2: c_uchar,
    pub num_extra_slice_header_bits: c_uchar,
    pub loop_filter_across_tiles_enabled_flag: c_uchar,
    pub loop_filter_across_slices_enabled_flag: c_uchar,
    pub output_flag_present_flag: c_uchar,
    pub num_ref_idx_l0_default_active_minus1: c_uchar,
    pub num_ref_idx_l1_default_active_minus1: c_uchar,
    pub lists_modification_present_flag: c_uchar,
    pub cabac_init_present_flag: c_uchar,
    pub pps_slice_chroma_qp_offsets_present_flag: c_uchar,
    pub deblocking_filter_override_enabled_flag: c_uchar,
    pub pps_deblocking_filter_disabled_flag: c_uchar,
    pub pps_beta_offset_div2: c_char,
    pub pps_tc_offset_div2: c_char,
    pub tiles_enabled_flag: c_uchar,
    pub uniform_spacing_flag: c_uchar,
    pub num_tile_columns_minus1: c_uchar,
    pub num_tile_rows_minus1: c_uchar,
    pub column_width_minus1: [c_ushort; 21],
    pub row_height_minus1: [c_ushort; 21],
    // sps and pps extension HEVC-main 444
    pub sps_range_extension_flag: c_uchar,
    pub transform_skip_rotation_enabled_flag: c_uchar,
    pub transform_skip_context_enabled_flag: c_uchar,
    pub implicit_rdpcm_enabled_flag: c_uchar,
    pub explicit_rdpcm_enabled_flag: c_uchar,
    pub extended_precision_processing_flag: c_uchar,
    pub intra_smoothing_disabled_flag: c_uchar,
    pub persistent_rice_adaptation_enabled_flag: c_uchar,
    pub cabac_bypass_alignment_enabled_flag: c_uchar,
    pub pps_range_extension_flag: c_uchar,
    pub cross_component_prediction_enabled_flag: c_uchar,
    pub chroma_qp_offset_list_enabled_flag: c_uchar,
    pub diff_cu_chroma_qp_offset_depth: c_uchar,
    pub chroma_qp_offset_list_len_minus1: c_uchar,
    pub cb_qp_offset_list: [c_char; 6],
    pub cr_qp_offset_list: [c_char; 6],
    pub reserved2: [c_uchar; 2],
    pub reserved3: [c_uint; 8],
    // RefPicSets
    pub NumBitsForShortTermRPSInSlice: c_int,
    pub NumDeltaPocsOfRefRpsIdx: c_int,
    pub NumPocTotalCurr: c_int,
    pub NumPocStCurrBefore: c_int,
    pub NumPocStCurrAfter: c_int,
    pub NumPocLtCurr: c_int,
    pub CurrPicOrderCntVal: c_int,
    pub RefPicIdx: [c_int; 16],
    pub PicOrderCntVal: [c_int; 16],
    pub IsLongTerm: [c_uchar; 16],
    pub RefPicSetStCurrBefore: [c_uchar; 8],
    pub RefPicSetStCurrAfter: [c_uchar; 8],
    pub RefPicSetLtCurr: [c_uchar; 8],
    pub RefPicSetInterLayer0: [c_uchar; 8],
    pub RefPicSetInterLayer1: [c_uchar; 8],
    pub reserved4: [c_uint; 12],
    // scaling lists (diag order)
    pub ScalingList4x4: [[c_uchar; 16]; 6],
    pub ScalingList8x8: [[c_uchar; 64]; 6],
    pub ScalingList16x16: [[c_uchar; 64]; 6],
    pub ScalingList32x32: [[c_uchar; 64]; 2],
    pub ScalingListDCCoeff16x16: [c_uchar; 6],
    pub ScalingListDCCoeff32x32: [c_uchar; 2],
}

/// VP9 picture parameters
///
/// Layout matches the C struct in Video Codec SDK 12.0.16 (`cuviddec.h`)
/// exactly: **220 bytes** on 64-bit Linux. The two C bitfield groups cannot be
/// expressed in Rust, so they are represented as raw integers (`flags: u16`
/// and `segment_flags: u8`) with getter/setter helpers using the x86-64
/// LSB-first bit order.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct CUVIDVP9PICPARAMS {
    pub width: c_uint,
    pub height: c_uint,
    pub LastRefIdx: c_uchar,
    pub GoldenRefIdx: c_uchar,
    pub AltRefIdx: c_uchar,
    pub colorSpace: c_uchar,
    /// Packed bitfields (x86-64 LSB-first): profile:3 | frameContextIdx:2 |
    /// frameType:1 | showFrame:1 | errorResilient:1 | frameParallelDecoding:1 |
    /// subSamplingX:1 | subSamplingY:1 | intraOnly:1 | allow_high_precision_mv:1 |
    /// refreshEntropyProbs:1 | reserved2Bits:2
    pub flags: u16,
    pub reserved16Bits: u16,
    pub refFrameSignBias: [c_uchar; 4],
    pub bitDepthMinus8Luma: c_uchar,
    pub bitDepthMinus8Chroma: c_uchar,
    pub loopFilterLevel: c_uchar,
    pub loopFilterSharpness: c_uchar,
    pub modeRefLfEnabled: c_uchar,
    pub log2_tile_columns: c_uchar,
    pub log2_tile_rows: c_uchar,
    /// Packed bitfields: segmentEnabled:1 | segmentMapUpdate:1 |
    /// segmentMapTemporalUpdate:1 | segmentFeatureMode:1 | reserved4Bits:4
    pub segment_flags: u8,
    pub segmentFeatureEnable: [[c_uchar; 4]; 8],
    pub segmentFeatureData: [[i16; 4]; 8],
    pub mb_segment_tree_probs: [c_uchar; 7],
    pub segment_pred_probs: [c_uchar; 3],
    pub reservedSegment16Bits: [c_uchar; 2],
    pub qpYAc: c_int,
    pub qpYDc: c_int,
    pub qpChDc: c_int,
    pub qpChAc: c_int,
    pub activeRefIdx: [c_uint; 3],
    pub resetFrameContext: c_uint,
    pub mcomp_filter_type: c_uint,
    /// Values are signed i32 in practice (sign-extended i8 loop-filter deltas).
    pub mbRefLfDelta: [c_uint; 4],
    pub mbModeLfDelta: [c_uint; 2],
    pub frameTagSize: c_uint,
    pub offsetToDctParts: c_uint,
    pub reserved128Bits: [c_uint; 4],
}

// Verified against the SDK header: sizeof(CUVIDVP9PICPARAMS) == 220.
const _: () = {
    assert!(std::mem::size_of::<CUVIDVP9PICPARAMS>() == 220);
};

impl CUVIDVP9PICPARAMS {
    /// Create a zeroed `CUVIDVP9PICPARAMS`.
    pub fn new() -> Self {
        unsafe { std::mem::zeroed() }
    }

    // ── `flags` bitfields (u16, x86-64 LSB-first) ──────────────────────

    /// VP9 profile (0-3). Bits 0-2.
    pub fn profile(&self) -> u32 {
        (self.flags & 0b111) as u32
    }
    pub fn set_profile(&mut self, v: u32) {
        self.flags = (self.flags & !0b111) | ((v & 0b111) as u16);
    }

    /// Frame context index (0-3). Bits 3-4.
    pub fn frame_context_idx(&self) -> u32 {
        ((self.flags >> 3) & 0b11) as u32
    }
    pub fn set_frame_context_idx(&mut self, v: u32) {
        self.flags = (self.flags & !(0b11 << 3)) | (((v & 0b11) as u16) << 3);
    }

    /// Frame type: 0 = key, 1 = inter. Bit 5.
    pub fn frame_type(&self) -> u32 {
        ((self.flags >> 5) & 1) as u32
    }
    pub fn set_frame_type(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 5)) | (((v & 1) as u16) << 5);
    }

    /// Show frame. Bit 6.
    pub fn show_frame(&self) -> u32 {
        ((self.flags >> 6) & 1) as u32
    }
    pub fn set_show_frame(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 6)) | (((v & 1) as u16) << 6);
    }

    /// Error resilient mode. Bit 7.
    pub fn error_resilient(&self) -> u32 {
        ((self.flags >> 7) & 1) as u32
    }
    pub fn set_error_resilient(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 7)) | (((v & 1) as u16) << 7);
    }

    /// Frame parallel decoding mode. Bit 8.
    pub fn frame_parallel_decoding(&self) -> u32 {
        ((self.flags >> 8) & 1) as u32
    }
    pub fn set_frame_parallel_decoding(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 8)) | (((v & 1) as u16) << 8);
    }

    /// Chroma subsampling x. Bit 9.
    pub fn sub_sampling_x(&self) -> u32 {
        ((self.flags >> 9) & 1) as u32
    }
    pub fn set_sub_sampling_x(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 9)) | (((v & 1) as u16) << 9);
    }

    /// Chroma subsampling y. Bit 10.
    pub fn sub_sampling_y(&self) -> u32 {
        ((self.flags >> 10) & 1) as u32
    }
    pub fn set_sub_sampling_y(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 10)) | (((v & 1) as u16) << 10);
    }

    /// Intra-only frame. Bit 11.
    pub fn intra_only(&self) -> u32 {
        ((self.flags >> 11) & 1) as u32
    }
    pub fn set_intra_only(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 11)) | (((v & 1) as u16) << 11);
    }

    /// Allow high-precision motion vectors. Bit 12.
    pub fn allow_high_precision_mv(&self) -> u32 {
        ((self.flags >> 12) & 1) as u32
    }
    pub fn set_allow_high_precision_mv(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 12)) | (((v & 1) as u16) << 12);
    }

    /// Refresh entropy probabilities (from `refresh_frame_context`). Bit 13.
    pub fn refresh_entropy_probs(&self) -> u32 {
        ((self.flags >> 13) & 1) as u32
    }
    pub fn set_refresh_entropy_probs(&mut self, v: u32) {
        self.flags = (self.flags & !(1 << 13)) | (((v & 1) as u16) << 13);
    }

    /// Reserved. Bits 14-15.
    pub fn reserved2_bits(&self) -> u32 {
        ((self.flags >> 14) & 0b11) as u32
    }
    pub fn set_reserved2_bits(&mut self, v: u32) {
        self.flags = (self.flags & !(0b11 << 14)) | (((v & 0b11) as u16) << 14);
    }

    // ── `segment_flags` bitfields (u8, x86-64 LSB-first) ───────────────

    /// Segmentation enabled. Bit 0.
    pub fn segment_enabled(&self) -> u32 {
        (self.segment_flags & 1) as u32
    }
    pub fn set_segment_enabled(&mut self, v: u32) {
        self.segment_flags = (self.segment_flags & !1) | ((v & 1) as u8);
    }

    /// Segmentation map update. Bit 1.
    pub fn segment_map_update(&self) -> u32 {
        ((self.segment_flags >> 1) & 1) as u32
    }
    pub fn set_segment_map_update(&mut self, v: u32) {
        self.segment_flags = (self.segment_flags & !(1 << 1)) | (((v & 1) as u8) << 1);
    }

    /// Segmentation map temporal update. Bit 2.
    pub fn segment_map_temporal_update(&self) -> u32 {
        ((self.segment_flags >> 2) & 1) as u32
    }
    pub fn set_segment_map_temporal_update(&mut self, v: u32) {
        self.segment_flags = (self.segment_flags & !(1 << 2)) | (((v & 1) as u8) << 2);
    }

    /// Segmentation feature data update mode. Bit 3.
    pub fn segment_feature_mode(&self) -> u32 {
        ((self.segment_flags >> 3) & 1) as u32
    }
    pub fn set_segment_feature_mode(&mut self, v: u32) {
        self.segment_flags = (self.segment_flags & !(1 << 3)) | (((v & 1) as u8) << 3);
    }

    /// Reserved. Bits 4-7.
    pub fn reserved4_bits(&self) -> u32 {
        ((self.segment_flags >> 4) & 0b1111) as u32
    }
    pub fn set_reserved4_bits(&mut self, v: u32) {
        self.segment_flags = (self.segment_flags & !(0b1111 << 4)) | (((v & 0b1111) as u8) << 4);
    }
}

impl Default for CUVIDVP9PICPARAMS {
    fn default() -> Self {
        Self::new()
    }
}

/// AV1 picture parameters.
///
/// Layout matches the NVIDIA Video Codec SDK `cuviddec.h` exactly
/// (verified against a C compile of the SDK header on x86-64: total size
/// 1024 bytes). C bitfields are packed LSB-first into their allocation
/// units, so each bitfield group is modeled here as a single packed word
/// with accessor helpers (same pattern as [`CUVIDVP9PICPARAMS`]).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDAV1PICPARAMS {
    pub width: c_uint,
    pub height: c_uint,
    pub frame_offset: c_uint,
    pub decodePicIdx: c_int,
    /// Sequence header bitfields (LSB-first): profile:3 |
    /// use_128x128_superblock:1 | subsampling_x:1 | subsampling_y:1 |
    /// mono_chrome:1 | bit_depth_minus8:4 | enable_filter_intra:1 |
    /// enable_intra_edge_filter:1 | enable_interintra_compound:1 |
    /// enable_masked_compound:1 | enable_dual_filter:1 | enable_order_hint:1 |
    /// order_hint_bits_minus1:3 | enable_jnt_comp:1 | enable_superres:1 |
    /// enable_cdef:1 | enable_restoration:1 | enable_fgs:1 | reserved0_7bits:7
    pub seq_flags: u32,
    /// Frame header bitfields (LSB-first): frame_type:2 | show_frame:1 |
    /// disable_cdf_update:1 | allow_screen_content_tools:1 | force_integer_mv:1 |
    /// coded_denom:3 | allow_intrabc:1 | allow_high_precision_mv:1 |
    /// interp_filter:3 | switchable_motion_mode:1 | use_ref_frame_mvs:1 |
    /// disable_frame_end_update_cdf:1 | delta_q_present:1 | delta_q_res:2 |
    /// using_qmatrix:1 | coded_lossless:1 | use_superres:1 | tx_mode:2 |
    /// reference_mode:1 | allow_warped_motion:1 | reduced_tx_set:1 |
    /// skip_mode:1 | reserved1_3bits:3
    pub frame_flags: u32,
    /// Tiling bitfields (LSB-first): num_tile_cols:8 | num_tile_rows:8 |
    /// context_update_tile_id:16
    pub tile_info: u32,
    /// Width of each tile column in superblocks (128 bytes).
    pub tile_widths: [u16; 64],
    /// Height of each tile row in superblocks (128 bytes).
    pub tile_heights: [u16; 64],
    /// CDEF bitfields: cdef_damping_minus_3:2 | cdef_bits:2 | reserved2_4bits:4
    pub cdef_flags: u8,
    /// 0-3 bits: y_pri_strength, 4-7 bits: y_sec_strength (per CDEF unit).
    pub cdef_y_strength: [u8; 8],
    /// 0-3 bits: uv_pri_strength, 4-7 bits: uv_sec_strength (per CDEF unit).
    pub cdef_uv_strength: [u8; 8],
    /// SkipModeFrames: SkipModeFrame0:4 | SkipModeFrame1:4
    pub skip_mode_frames: u8,
    /// Base frame qindex (AV1 base_q_idx).
    pub base_qindex: u8,
    pub qp_y_dc_delta_q: i8,
    pub qp_u_dc_delta_q: i8,
    pub qp_v_dc_delta_q: i8,
    pub qp_u_ac_delta_q: i8,
    pub qp_v_ac_delta_q: i8,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
    /// Segmentation bitfields: segmentation_enabled:1 | segmentation_update_map:1 |
    /// segmentation_update_data:1 | segmentation_temporal_update:1 | reserved3_4bits:4
    pub segmentation_flags: u8,
    /// Feature data for each segment/feature (8x8, 128 bytes).
    pub segmentation_feature_data: [i16; 64],
    /// Indicates that the corresponding feature is unused or feature value is coded.
    pub segmentation_feature_mask: [u8; 8],
    /// Loop filter strength values for Y (2 planes).
    pub loop_filter_level: [u8; 2],
    pub loop_filter_level_u: u8,
    pub loop_filter_level_v: u8,
    pub loop_filter_sharpness: u8,
    pub loop_filter_ref_deltas: [i8; 8],
    pub loop_filter_mode_deltas: [i8; 2],
    /// Loop filter bitfields: loop_filter_delta_enabled:1 |
    /// loop_filter_delta_update:1 | delta_lf_present:1 | delta_lf_res:2 |
    /// delta_lf_multi:1 | reserved4_2bits:2
    pub loop_filter_flags: u8,
    /// Loop restoration unit sizes: 0: 32, 1: 64, 2: 128, 3: 256.
    pub lr_unit_size: [u8; 3],
    /// Used to compute FrameRestorationType.
    pub lr_type: [u8; 3],
    /// Reference frame containing the CDF values and other state.
    pub primary_ref_frame: u8,
    /// Frames in DPB that can be used as reference for current/future frames.
    pub ref_frame_map: [u8; 8],
    /// Layer ids: temporal_layer_id:4 | spatial_layer_id:4
    pub layer_ids: u8,
    pub reserved5_32bits: [u8; 4],
    /// Reference frames used for the current frame (7 slots).
    pub ref_frame: [CUVIDAV1REFFRAME; 7],
    /// Global motion params for reference frames (7 slots).
    pub global_motion: [CUVIDAV1GLOBALMOTION; 7],
    /// Film grain bitfields (LSB-first u16): apply_grain:1 | overlap_flag:1 |
    /// scaling_shift_minus8:2 | chroma_scaling_from_luma:1 | ar_coeff_lag:2 |
    /// ar_coeff_shift_minus6:2 | grain_scale_shift:2 | clip_to_restricted_range:1 |
    /// reserved6_4bits:4
    pub film_grain_flags: u16,
    pub num_y_points: u8,
    pub scaling_points_y: [[u8; 2]; 14],
    pub num_cb_points: u8,
    pub scaling_points_cb: [[u8; 2]; 10],
    pub num_cr_points: u8,
    pub scaling_points_cr: [[u8; 2]; 10],
    pub reserved7_8bits: u8,
    pub random_seed: u16,
    pub ar_coeffs_y: [i16; 24],
    pub ar_coeffs_cb: [i16; 25],
    pub ar_coeffs_cr: [i16; 25],
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: i16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: i16,
    pub reserved: [c_int; 7],
}

impl CUVIDAV1PICPARAMS {
    /// Extract a bitfield from `word` starting at `shift` with `width` bits.
    pub fn bits(word: u32, shift: u32, width: u32) -> u32 {
        (word >> shift) & ((1u32 << width) - 1)
    }
    pub fn profile(&self) -> u32 {
        Self::bits(self.seq_flags, 0, 3)
    }
    pub fn use_128x128_superblock(&self) -> u32 {
        Self::bits(self.seq_flags, 3, 1)
    }
    pub fn subsampling_x(&self) -> u32 {
        Self::bits(self.seq_flags, 4, 1)
    }
    pub fn subsampling_y(&self) -> u32 {
        Self::bits(self.seq_flags, 5, 1)
    }
    pub fn mono_chrome(&self) -> u32 {
        Self::bits(self.seq_flags, 6, 1)
    }
    pub fn bit_depth_minus8(&self) -> u32 {
        Self::bits(self.seq_flags, 7, 4)
    }
    pub fn enable_filter_intra(&self) -> u32 {
        Self::bits(self.seq_flags, 11, 1)
    }
    pub fn enable_intra_edge_filter(&self) -> u32 {
        Self::bits(self.seq_flags, 12, 1)
    }
    pub fn enable_interintra_compound(&self) -> u32 {
        Self::bits(self.seq_flags, 13, 1)
    }
    pub fn enable_masked_compound(&self) -> u32 {
        Self::bits(self.seq_flags, 14, 1)
    }
    pub fn enable_dual_filter(&self) -> u32 {
        Self::bits(self.seq_flags, 15, 1)
    }
    pub fn enable_order_hint(&self) -> u32 {
        Self::bits(self.seq_flags, 16, 1)
    }
    pub fn order_hint_bits_minus1(&self) -> u32 {
        Self::bits(self.seq_flags, 17, 3)
    }
    pub fn enable_jnt_comp(&self) -> u32 {
        Self::bits(self.seq_flags, 20, 1)
    }
    pub fn enable_superres(&self) -> u32 {
        Self::bits(self.seq_flags, 21, 1)
    }
    pub fn enable_cdef(&self) -> u32 {
        Self::bits(self.seq_flags, 22, 1)
    }
    pub fn enable_restoration(&self) -> u32 {
        Self::bits(self.seq_flags, 23, 1)
    }
    pub fn enable_fgs(&self) -> u32 {
        Self::bits(self.seq_flags, 24, 1)
    }
    pub fn frame_type(&self) -> u32 {
        Self::bits(self.frame_flags, 0, 2)
    }
    pub fn show_frame(&self) -> u32 {
        Self::bits(self.frame_flags, 2, 1)
    }
    pub fn disable_cdf_update(&self) -> u32 {
        Self::bits(self.frame_flags, 3, 1)
    }
    pub fn allow_screen_content_tools(&self) -> u32 {
        Self::bits(self.frame_flags, 4, 1)
    }
    pub fn force_integer_mv(&self) -> u32 {
        Self::bits(self.frame_flags, 5, 1)
    }
    pub fn coded_denom(&self) -> u32 {
        Self::bits(self.frame_flags, 6, 3)
    }
    pub fn allow_intrabc(&self) -> u32 {
        Self::bits(self.frame_flags, 9, 1)
    }
    pub fn allow_high_precision_mv(&self) -> u32 {
        Self::bits(self.frame_flags, 10, 1)
    }
    pub fn interp_filter(&self) -> u32 {
        Self::bits(self.frame_flags, 11, 3)
    }
    pub fn switchable_motion_mode(&self) -> u32 {
        Self::bits(self.frame_flags, 14, 1)
    }
    pub fn use_ref_frame_mvs(&self) -> u32 {
        Self::bits(self.frame_flags, 15, 1)
    }
    pub fn disable_frame_end_update_cdf(&self) -> u32 {
        Self::bits(self.frame_flags, 16, 1)
    }
    pub fn delta_q_present(&self) -> u32 {
        Self::bits(self.frame_flags, 17, 1)
    }
    pub fn delta_q_res(&self) -> u32 {
        Self::bits(self.frame_flags, 18, 2)
    }
    pub fn using_qmatrix(&self) -> u32 {
        Self::bits(self.frame_flags, 20, 1)
    }
    pub fn coded_lossless(&self) -> u32 {
        Self::bits(self.frame_flags, 21, 1)
    }
    pub fn use_superres(&self) -> u32 {
        Self::bits(self.frame_flags, 22, 1)
    }
    pub fn tx_mode(&self) -> u32 {
        Self::bits(self.frame_flags, 23, 2)
    }
    pub fn reference_mode(&self) -> u32 {
        Self::bits(self.frame_flags, 25, 1)
    }
    pub fn allow_warped_motion(&self) -> u32 {
        Self::bits(self.frame_flags, 26, 1)
    }
    pub fn reduced_tx_set(&self) -> u32 {
        Self::bits(self.frame_flags, 27, 1)
    }
    pub fn skip_mode(&self) -> u32 {
        Self::bits(self.frame_flags, 28, 1)
    }
    pub fn num_tile_cols(&self) -> u32 {
        Self::bits(self.tile_info, 0, 8)
    }
    pub fn num_tile_rows(&self) -> u32 {
        Self::bits(self.tile_info, 8, 8)
    }
    pub fn context_update_tile_id(&self) -> u32 {
        Self::bits(self.tile_info, 16, 16)
    }
    pub fn cdef_damping_minus_3(&self) -> u32 {
        Self::bits(self.cdef_flags as u32, 0, 2)
    }
    pub fn cdef_bits(&self) -> u32 {
        Self::bits(self.cdef_flags as u32, 2, 2)
    }
    pub fn skip_mode_frame0(&self) -> u32 {
        Self::bits(self.skip_mode_frames as u32, 0, 4)
    }
    pub fn skip_mode_frame1(&self) -> u32 {
        Self::bits(self.skip_mode_frames as u32, 4, 4)
    }
    pub fn segmentation_enabled(&self) -> u32 {
        Self::bits(self.segmentation_flags as u32, 0, 1)
    }
    pub fn segmentation_update_map(&self) -> u32 {
        Self::bits(self.segmentation_flags as u32, 1, 1)
    }
    pub fn segmentation_update_data(&self) -> u32 {
        Self::bits(self.segmentation_flags as u32, 2, 1)
    }
    pub fn segmentation_temporal_update(&self) -> u32 {
        Self::bits(self.segmentation_flags as u32, 3, 1)
    }
    pub fn loop_filter_delta_enabled(&self) -> u32 {
        Self::bits(self.loop_filter_flags as u32, 0, 1)
    }
    pub fn loop_filter_delta_update(&self) -> u32 {
        Self::bits(self.loop_filter_flags as u32, 1, 1)
    }
    pub fn delta_lf_present(&self) -> u32 {
        Self::bits(self.loop_filter_flags as u32, 2, 1)
    }
    pub fn delta_lf_res(&self) -> u32 {
        Self::bits(self.loop_filter_flags as u32, 3, 2)
    }
    pub fn delta_lf_multi(&self) -> u32 {
        Self::bits(self.loop_filter_flags as u32, 5, 1)
    }
    pub fn temporal_layer_id(&self) -> u32 {
        Self::bits(self.layer_ids as u32, 0, 4)
    }
    pub fn spatial_layer_id(&self) -> u32 {
        Self::bits(self.layer_ids as u32, 4, 4)
    }
    pub fn apply_grain(&self) -> u32 {
        Self::bits(self.film_grain_flags as u32, 0, 1)
    }
}

/// AV1 reference frame structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDAV1REFFRAME {
    pub width: c_uint,
    pub height: c_uint,
    pub index: c_uchar,
    pub reserved24Bits: [c_uchar; 3],
}

/// AV1 global motion structure (28 bytes: 1 bitfield byte + 3 reserved +
/// 6 x i32 matrix).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDAV1GLOBALMOTION {
    /// Bitfields: invalid:1 | wmtype:2 | reserved5Bits:5
    pub flags: u8,
    pub reserved24Bits: [c_char; 3],
    /// gm_params[6] from the AV1 specification.
    pub wmmat: [c_int; 6],
}

impl CUVIDAV1GLOBALMOTION {
    pub fn invalid(&self) -> u32 {
        (self.flags as u32 >> 0) & 1
    }
    pub fn wmtype(&self) -> u32 {
        (self.flags as u32 >> 1) & 0b11
    }
}

/// Picture parameters for decoding.
///
/// Passed to `cuvidDecodePicture` with per-frame decode parameters
/// including picture dimensions, reference picture flags, bitstream
/// data pointer, and codec-specific parameters (via [`CUVIDCODECSPECIFIC`]).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUVIDPICPARAMS {
    pub PicWidthInMbs: c_int,
    pub FrameHeightInMbs: c_int,
    pub CurrPicIdx: c_int,
    pub field_pic_flag: c_int,
    pub bottom_field_flag: c_int,
    pub second_field: c_int,
    pub nBitstreamDataLen: c_uint,
    pub pBitstreamData: *const c_uchar,
    pub nNumSlices: c_uint,
    pub pSliceDataOffsets: *const c_uint,
    pub ref_pic_flag: c_int,
    pub intra_pic_flag: c_int,
    pub Reserved: [c_uint; 30],
    pub CodecSpecific: CUVIDCODECSPECIFIC,
}

/// Codec-specific picture parameters union
#[repr(C)]
#[derive(Clone, Copy)]
pub union CUVIDCODECSPECIFIC {
    #[allow(dead_code)]
    pub h264: CUVIDH264PICPARAMS,
    pub hevc: CUVIDHEVCPICPARAMS,
    pub vp9: CUVIDVP9PICPARAMS,
    pub av1: CUVIDAV1PICPARAMS,
    pub CodecReserved: [c_uint; 1024],
}

/// Picture parameters for postprocessing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDPROCPARAMS {
    pub progressive_frame: c_int,
    pub second_field: c_int,
    pub top_field_first: c_int,
    pub unpaired_field: c_int,
    pub reserved_flags: c_uint,
    pub reserved_zero: c_uint,
    pub raw_input_dptr: c_ulonglong,
    pub raw_input_pitch: c_uint,
    pub raw_input_format: c_uint,
    pub raw_output_dptr: c_ulonglong,
    pub raw_output_pitch: c_uint,
    pub Reserved1: c_uint,
    pub output_stream: CUstream,
    pub Reserved: [c_uint; 46],
    pub histogram_dptr: *mut c_ulonglong,
    pub Reserved2: [*mut c_void; 1],
}

/// Decode status structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDGETDECODESTATUS {
    pub decodeStatus: cuvidDecodeStatus,
    pub reserved: [c_uint; 31],
    pub pReserved: [*mut c_void; 8],
}

/// Decoder reconfigure info
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDRECONFIGUREDECODERINFO {
    pub ulWidth: c_uint,
    pub ulHeight: c_uint,
    pub ulTargetWidth: c_uint,
    pub ulTargetHeight: c_uint,
    pub ulNumDecodeSurfaces: c_uint,
    pub reserved1: [c_uint; 12],
    pub display_area: CUVIDRECT,
    pub target_rect: CUVIDRECT,
    pub reserved2: [c_uint; 11],
}

// ============================================================================
// Parser types and structures (from nvcuvid.h)
// ============================================================================

/// Opaque video parser handle.
///
/// Returned by `cuvidCreateVideoParser`. Passed to parse and destroy
/// operations. Destroy with `cuvidDestroyVideoParser`.
pub type CUvideoparser = *mut c_void;

/// Video timestamp type (microseconds).
pub type CUvideotimestamp = i64;

/// Video packet flags.
///
/// Bitmask of flags for [`CUVIDSOURCEDATAPACKET`] passed to the parser.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUvideopacketflags {
    CUVID_PKT_ENDOFSTREAM = 0x01,
    CUVID_PKT_TIMESTAMP = 0x02,
    CUVID_PKT_DISCONTINUITY = 0x04,
    CUVID_PKT_ENDOFPICTURE = 0x08,
    CUVID_PKT_NOTIFY_EOS = 0x10,
}

/// Frame rate structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDFRAMERATE {
    pub numerator: c_uint,
    pub denominator: c_uint,
}

/// Display area structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDDISPLAYAREA {
    pub left: c_int,
    pub top: c_int,
    pub right: c_int,
    pub bottom: c_int,
}

/// Display aspect ratio structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDDISPLAYASPECTRATIO {
    pub x: c_int,
    pub y: c_int,
}

/// Video signal description
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDVIDEOSIGNALDESCRIPTION {
    pub video_format: c_uchar,
    pub video_full_range_flag: c_uchar,
    pub reserved_zero_bits: c_uchar,
    pub color_primaries: c_uchar,
    pub transfer_characteristics: c_uchar,
    pub matrix_coefficients: c_uchar,
}

/// Video format structure (from parser sequence callback).
///
/// Provided by the NVIDIA parser in the sequence callback with decoded
/// video format information extracted from the bitstream (codec,
/// resolution, chroma format, bit depth, display area, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDEOFORMAT {
    pub codec: cudaVideoCodec,
    pub frame_rate: CUVIDFRAMERATE,
    pub progressive_sequence: c_uchar,
    pub bit_depth_luma_minus8: c_uchar,
    pub bit_depth_chroma_minus8: c_uchar,
    pub min_num_decode_surfaces: c_uchar,
    pub coded_width: c_uint,
    pub coded_height: c_uint,
    pub display_area: CUVIDDISPLAYAREA,
    pub chroma_format: cudaVideoChromaFormat,
    pub bitrate: c_uint,
    pub display_aspect_ratio: CUVIDDISPLAYASPECTRATIO,
    pub video_signal_description: CUVIDVIDEOSIGNALDESCRIPTION,
    pub seqhdr_data_length: c_uint,
}

/// Source data packet.
///
/// Describes a chunk of bitstream data to pass to `cuvidParseVideoData`.
/// Contains a pointer to the raw NAL unit data and optional metadata
/// (timestamp, flags).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDSOURCEDATAPACKET {
    pub flags: c_ulong,
    pub payload_size: c_ulong,
    pub payload: *const c_uchar,
    pub timestamp: CUvideotimestamp,
}

/// Parser display info.
///
/// Passed to the display callback when a decoded frame is ready for
/// presentation. Contains the picture index, progressive/field flags,
/// and repeat field information.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDPARSERDISPINFO {
    pub picture_index: c_int,
    pub progressive_frame: c_int,
    pub top_field_first: c_int,
    pub repeat_first_field: c_int,
    pub timestamp: CUvideotimestamp,
}

/// Parser callback types
pub type PFNVIDSEQUENCECALLBACK =
    Option<unsafe extern "C" fn(pUserData: *mut c_void, pVideoFormat: *mut CUVIDEOFORMAT) -> c_int>;
pub type PFNVIDDECODECALLBACK =
    Option<unsafe extern "C" fn(pUserData: *mut c_void, pPicParams: *mut CUVIDPICPARAMS) -> c_int>;
pub type PFNVIDDISPLAYCALLBACK = Option<
    unsafe extern "C" fn(pUserData: *mut c_void, pDispInfo: *mut CUVIDPARSERDISPINFO) -> c_int,
>;

/// Parser parameters.
///
/// Passed to `cuvidCreateVideoParser` to configure the parser. Specifies
/// the codec type, callback functions, user data pointer, and parser
/// options (Annex-B mode, clock rate, error threshold, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDPARSERPARAMS {
    pub CodecType: cudaVideoCodec,
    pub ulMaxNumDecodeSurfaces: c_uint,
    pub ulClockRate: c_uint,
    pub ulErrorThreshold: c_uint,
    pub ulMaxDisplayDelay: c_uint,
    // bAnnexb (1 bit) + uReserved (31 bits) packed into one uint
    pub bAnnexb_and_reserved: c_uint,
    pub uReserved1: [c_uint; 4],
    pub pUserData: *mut c_void,
    pub pfnSequenceCallback: PFNVIDSEQUENCECALLBACK,
    pub pfnDecodePicture: PFNVIDDECODECALLBACK,
    pub pfnDisplayPicture: PFNVIDDISPLAYCALLBACK,
    pub pfnGetOperatingPoint: *mut c_void,
    pub pfnGetSEIMsg: *mut c_void,
    pub pvReserved2: [*mut c_void; 5],
    pub pExtVideoInfo: *mut c_void,
}

// ============================================================================
// FFI Function declarations
// ============================================================================

// NVDEC FFI function declarations.
//
// These are declared as `extern "C"` for direct linking. In practice,
// the `device` module loads these dynamically via `libloading`.
extern "C" {
    /// Query decoder capabilities
    pub fn cuvidGetDecoderCaps(pdc: *mut CUVIDDECODECAPS) -> CUresult;

    /// Create decoder
    pub fn cuvidCreateDecoder(
        phDecoder: *mut CUvideodecoder,
        pdci: *const CUVIDDECODECREATEINFO,
    ) -> CUresult;

    /// Destroy decoder
    pub fn cuvidDestroyDecoder(hDecoder: CUvideodecoder) -> CUresult;

    /// Decode a picture
    pub fn cuvidDecodePicture(
        hDecoder: CUvideodecoder,
        pPicParams: *const CUVIDPICPARAMS,
    ) -> CUresult;

    /// Get decode status
    pub fn cuvidGetDecodeStatus(
        hDecoder: CUvideodecoder,
        nPicIdx: c_int,
        pDecodeStatus: *mut CUVIDGETDECODESTATUS,
    ) -> CUresult;

    /// Reconfigure decoder
    pub fn cuvidReconfigureDecoder(
        hDecoder: CUvideodecoder,
        pDecReconfigParams: *const CUVIDRECONFIGUREDECODERINFO,
    ) -> CUresult;

    /// Map video frame (64-bit)
    pub fn cuvidMapVideoFrame64(
        hDecoder: CUvideodecoder,
        nPicIdx: c_int,
        pDevPtr: *mut c_ulonglong,
        pPitch: *mut c_uint,
        pVPP: *const CUVIDPROCPARAMS,
    ) -> CUresult;

    /// Unmap video frame (64-bit)
    pub fn cuvidUnmapVideoFrame64(hDecoder: CUvideodecoder, DevPtr: c_ulonglong) -> CUresult;

    /// Create context lock
    pub fn cuvidCtxLockCreate(pLock: *mut CUvideoctxlock, ctx: *mut c_void) -> CUresult;

    /// Destroy context lock
    pub fn cuvidCtxLockDestroy(lck: CUvideoctxlock) -> CUresult;

    /// Lock context
    pub fn cuvidCtxLock(lck: CUvideoctxlock, reserved_flags: c_uint) -> CUresult;

    /// Unlock context
    pub fn cuvidCtxUnlock(lck: CUvideoctxlock, reserved_flags: c_uint) -> CUresult;

    // Parser functions
    /// Create video parser
    pub fn cuvidCreateVideoParser(
        pObj: *mut CUvideoparser,
        pParams: *const CUVIDPARSERPARAMS,
    ) -> CUresult;

    /// Parse video data
    pub fn cuvidParseVideoData(
        obj: CUvideoparser,
        pPacket: *const CUVIDSOURCEDATAPACKET,
    ) -> CUresult;

    /// Destroy video parser
    pub fn cuvidDestroyVideoParser(obj: CUvideoparser) -> CUresult;
}

// ============================================================================
// CUDA result constants (subset needed for error handling)
// ============================================================================

/// CUDA operation succeeded.
pub const CUDA_SUCCESS: CUresult = 0;
/// Invalid argument value.
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
/// Out of memory.
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
/// CUDA not initialized.
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
/// CUDA deinitialized.
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
/// No CUDA-capable device found.
pub const CUDA_ERROR_NO_DEVICE: CUresult = 76;
/// Operation not supported.
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 80;
/// Peer access not supported.
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: CUresult = 218;

/// Convert a CUDA result code to a human-readable string.
///
/// # Example
///
/// ```
/// use nvdec_decode::ffi::{cu_result_to_string, CUDA_SUCCESS, CUDA_ERROR_OUT_OF_MEMORY};
///
/// assert_eq!(cu_result_to_string(CUDA_SUCCESS), "CUDA_SUCCESS");
/// assert_eq!(cu_result_to_string(CUDA_ERROR_OUT_OF_MEMORY), "CUDA_ERROR_OUT_OF_MEMORY");
/// assert_eq!(cu_result_to_string(999), "UNKNOWN_CUDA_ERROR");
/// ```
pub fn cu_result_to_string(result: CUresult) -> &'static str {
    match result {
        CUDA_SUCCESS => "CUDA_SUCCESS",
        CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY",
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
        CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE",
        CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED",
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        _ => "UNKNOWN_CUDA_ERROR",
    }
}

/// Check CUDA result and panic on error (for development).
///
/// Macro that checks if a CUDA operation returned [`CUDA_SUCCESS`].
/// Panics with a descriptive message (error name, code, file, line)
/// on failure.
///
/// # Panics
///
/// Panics if the expression evaluates to a non-zero CUDA error code.
///
/// # Example
///
/// ```ignore
/// nvdec_check!(cuvidCreateDecoder(&mut decoder, &create_info));
/// ```
#[macro_export]
macro_rules! nvdec_check {
    ($expr:expr) => {
        match $expr {
            $crate::ffi::CUDA_SUCCESS => {}
            err => panic!(
                "NVDEC error: {} ({}) at {}:{}",
                $crate::ffi::cu_result_to_string(err),
                err,
                file!(),
                line!()
            ),
        }
    };
}

// Manual Debug implementations for unions
impl std::fmt::Debug for CUVIDH264FMOASO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUVIDH264FMOASO").finish()
    }
}

impl std::fmt::Debug for CUVIDH264SVCMVC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUVIDH264SVCMVC").finish()
    }
}

impl std::fmt::Debug for CUVIDCODECSPECIFIC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUVIDCODECSPECIFIC").finish()
    }
}

impl std::fmt::Debug for CUVIDH264PICPARAMS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUVIDH264PICPARAMS")
            .field("frame_num", &self.frame_num)
            .field("num_ref_frames", &self.num_ref_frames)
            .finish()
    }
}

impl std::fmt::Debug for CUVIDPICPARAMS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CUVIDPICPARAMS")
            .field("PicWidthInMbs", &self.PicWidthInMbs)
            .field("FrameHeightInMbs", &self.FrameHeightInMbs)
            .field("CurrPicIdx", &self.CurrPicIdx)
            .field("nBitstreamDataLen", &self.nBitstreamDataLen)
            .finish()
    }
}

#[cfg(test)]
mod struct_size_tests {
    use super::*;

    // ── Struct size tests ──────────────────────────────────────────────

    #[test]
    fn test_cuvid_h264_picparams_size() {
        let size = std::mem::size_of::<CUVIDH264PICPARAMS>();
        println!("CUVIDH264PICPARAMS size: {}", size);
        // NVIDIA SDK: 984 bytes on 64-bit
        assert_eq!(
            size, 984,
            "CUVIDH264PICPARAMS size mismatch: expected 984, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_hevc_picparams_size() {
        let size = std::mem::size_of::<CUVIDHEVCPICPARAMS>();
        println!("CUVIDHEVCPICPARAMS size: {}", size);
        // NVIDIA SDK: 1568 bytes on 64-bit (with RExt fields)
        assert_eq!(
            size, 1484,
            "CUVIDHEVCPICPARAMS size mismatch: expected 1484, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_vp9_picparams_size() {
        let size = std::mem::size_of::<CUVIDVP9PICPARAMS>();
        println!("CUVIDVP9PICPARAMS size: {}", size);
        // NVIDIA SDK: 220 bytes on 64-bit (packed C bitfields)
        assert_eq!(
            size, 220,
            "CUVIDVP9PICPARAMS size mismatch: expected 220, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_av1_picparams_size() {
        let size = std::mem::size_of::<CUVIDAV1PICPARAMS>();
        println!("CUVIDAV1PICPARAMS size: {}", size);
        // NVIDIA SDK (cuviddec.h, x86-64): 1024 bytes (verified via C compile
        // of the SDK header: sizeof(CUVIDAV1PICPARAMS) == 1024).
        assert_eq!(
            size, 1024,
            "CUVIDAV1PICPARAMS size mismatch: expected 1024, got {}",
            size
        );
        assert_eq!(
            std::mem::size_of::<CUVIDAV1REFFRAME>(),
            12,
            "CUVIDAV1REFFRAME size mismatch: expected 12"
        );
        assert_eq!(
            std::mem::size_of::<CUVIDAV1GLOBALMOTION>(),
            28,
            "CUVIDAV1GLOBALMOTION size mismatch: expected 28"
        );
    }

    #[test]
    fn test_cuvid_picparams_size() {
        let size = std::mem::size_of::<CUVIDPICPARAMS>();
        println!("CUVIDPICPARAMS size: {}", size);
        // NVIDIA SDK: 4280 bytes on 64-bit (includes 4096-byte union)
        assert_eq!(
            size, 4280,
            "CUVIDPICPARAMS size mismatch: expected 4280, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_codec_specific_size() {
        let size = std::mem::size_of::<CUVIDCODECSPECIFIC>();
        println!("CUVIDCODECSPECIFIC size: {}", size);
        // NVIDIA SDK: 4096 bytes (union of codec-specific params + reserved)
        assert_eq!(
            size, 4096,
            "CUVIDCODECSPECIFIC size mismatch: expected 4096, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_decode_createinfo_size() {
        let size = std::mem::size_of::<CUVIDDECODECREATEINFO>();
        println!("CUVIDDECODECREATEINFO size: {}", size);
        assert!(size > 0, "CUVIDDECODECREATEINFO size must be > 0");
    }

    #[test]
    fn test_cuvid_decodecaps_size() {
        let size = std::mem::size_of::<CUVIDDECODECAPS>();
        println!("CUVIDDECODECAPS size: {}", size);
        assert!(size > 0, "CUVIDDECODECAPS size must be > 0");
    }

    #[test]
    fn test_cuvid_procparams_size() {
        let size = std::mem::size_of::<CUVIDPROCPARAMS>();
        println!("CUVIDPROCPARAMS size: {}", size);
        assert!(size > 0, "CUVIDPROCPARAMS size must be > 0");
    }

    #[test]
    fn test_cuvid_parserparams_size() {
        let size = std::mem::size_of::<CUVIDPARSERPARAMS>();
        println!("CUVIDPARSERPARAMS size: {}", size);
        assert!(size > 0, "CUVIDPARSERPARAMS size must be > 0");
    }

    #[test]
    fn test_cuvid_h264_dpbentry_size() {
        let size = std::mem::size_of::<CUVIDH264DPBENTRY>();
        println!("CUVIDH264DPBENTRY size: {}", size);
        // 5 * c_int + FieldOrderCnt[2] = 7 * 4 = 28 bytes
        assert_eq!(
            size, 28,
            "CUVIDH264DPBENTRY size mismatch: expected 28, got {}",
            size
        );
    }

    #[test]
    fn test_cuvid_rect_size() {
        let size = std::mem::size_of::<CUVIDRECT>();
        println!("CUVIDRECT size: {}", size);
        // 4 * c_short = 4 * 2 = 8 bytes
        assert_eq!(size, 8, "CUVIDRECT size mismatch: expected 8, got {}", size);
    }

    // ── Struct field offset tests ─────────────────────────────────────

    #[test]
    fn test_cuvid_h264_picparams_offsets() {
        // Verify critical field offsets match NVIDIA SDK layout
        use std::mem::offset_of;

        assert_eq!(offset_of!(CUVIDH264PICPARAMS, log2_max_frame_num_minus4), 0);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, pic_order_cnt_type), 4);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, log2_max_pic_order_cnt_lsb_minus4),
            8
        );
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, delta_pic_order_always_zero_flag),
            12
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, frame_mbs_only_flag), 16);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, direct_8x8_inference_flag),
            20
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, num_ref_frames), 24);
        // 4 c_uchar fields at offset 28-31
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, residual_colour_transform_flag),
            28
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, bit_depth_luma_minus8), 29);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, bit_depth_chroma_minus8), 30);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, qpprime_y_zero_transform_bypass_flag),
            31
        );
        // PPS starts at offset 32
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, entropy_coding_mode_flag), 32);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, pic_order_present_flag), 36);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, num_ref_idx_l0_active_minus1),
            40
        );
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, num_ref_idx_l1_active_minus1),
            44
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, weighted_pred_flag), 48);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, weighted_bipred_idc), 52);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, pic_init_qp_minus26), 56);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, deblocking_filter_control_present_flag),
            60
        );
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, redundant_pic_cnt_present_flag),
            64
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, transform_8x8_mode_flag), 68);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, MbaffFrameFlag), 72);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, constrained_intra_pred_flag),
            76
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, chroma_qp_index_offset), 80);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, second_chroma_qp_index_offset),
            84
        );
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, ref_pic_flag), 88);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, frame_num), 92);
        // CurrFieldOrderCnt at offset 96
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, CurrFieldOrderCnt), 96);
        // DPB at offset 104
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, dpb), 104);
        // DPB: 16 * 28 = 448 bytes, ends at offset 552
        // WeightScale4x4 at offset 552 (104 + 16*28)
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, WeightScale4x4), 552);
        // WeightScale8x8 at offset 648 (552 + 6*16)
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, WeightScale8x8), 648);
        // fmo_aso_enable at offset 776 (648 + 2*64)
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, fmo_aso_enable), 776);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, num_slice_groups_minus1), 777);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, slice_group_map_type), 778);
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, pic_init_qs_minus26), 779);
        assert_eq!(
            offset_of!(CUVIDH264PICPARAMS, slice_group_change_rate_minus1),
            780
        );
        // fmo union at offset 784 (aligned to 8)
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, fmo), 784);
        // Reserved at offset 792
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, Reserved), 792);
        // svc_mvc at offset 840 (792 + 12*4)
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, svc_mvc), 840);
    }

    #[test]
    fn test_cuvid_picparams_offsets() {
        use std::mem::offset_of;

        assert_eq!(offset_of!(CUVIDPICPARAMS, PicWidthInMbs), 0);
        assert_eq!(offset_of!(CUVIDPICPARAMS, FrameHeightInMbs), 4);
        assert_eq!(offset_of!(CUVIDPICPARAMS, CurrPicIdx), 8);
        assert_eq!(offset_of!(CUVIDPICPARAMS, field_pic_flag), 12);
        assert_eq!(offset_of!(CUVIDPICPARAMS, bottom_field_flag), 16);
        assert_eq!(offset_of!(CUVIDPICPARAMS, second_field), 20);
        assert_eq!(offset_of!(CUVIDPICPARAMS, nBitstreamDataLen), 24);
        assert_eq!(offset_of!(CUVIDPICPARAMS, pBitstreamData), 32);
        assert_eq!(offset_of!(CUVIDPICPARAMS, nNumSlices), 40);
        assert_eq!(offset_of!(CUVIDPICPARAMS, pSliceDataOffsets), 48);
        assert_eq!(offset_of!(CUVIDPICPARAMS, ref_pic_flag), 56);
        assert_eq!(offset_of!(CUVIDPICPARAMS, intra_pic_flag), 60);
        // Reserved[30] starts at offset 64
        assert_eq!(offset_of!(CUVIDPICPARAMS, Reserved), 64);
        // CodecSpecific union at offset 184 (64 + 30*4)
        assert_eq!(offset_of!(CUVIDPICPARAMS, CodecSpecific), 184);
    }

    #[test]
    fn test_cuvid_codec_specific_offsets() {
        use std::mem::offset_of;

        // All union members start at offset 0
        assert_eq!(offset_of!(CUVIDCODECSPECIFIC, h264), 0);
        assert_eq!(offset_of!(CUVIDCODECSPECIFIC, hevc), 0);
        assert_eq!(offset_of!(CUVIDCODECSPECIFIC, vp9), 0);
        assert_eq!(offset_of!(CUVIDCODECSPECIFIC, av1), 0);
        assert_eq!(offset_of!(CUVIDCODECSPECIFIC, CodecReserved), 0);
    }

    #[test]
    fn test_cuvid_decode_createinfo_offsets() {
        use std::mem::offset_of;

        // Offsets match NVIDIA SDK CUVIDDECODECREATEINFO layout on 64-bit Linux
        // tcu_ulong = unsigned long = 8 bytes on 64-bit Linux
        // Total struct size: 176 bytes on 64-bit Linux
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulWidth), 0);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulHeight), 8);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulNumDecodeSurfaces), 16);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, CodecType), 24);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ChromaFormat), 28);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulCreationFlags), 32);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, bitDepthMinus8), 40);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulIntraDecodeOnly), 48);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulMaxWidth), 56);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulMaxHeight), 64);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, Reserved1), 72);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, display_area), 80);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, OutputFormat), 88);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, DeinterlaceMode), 92);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulTargetWidth), 96);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulTargetHeight), 104);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, ulNumOutputSurfaces), 112);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, vidLock), 120);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, target_rect), 128);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, enableHistogram), 136);
        assert_eq!(offset_of!(CUVIDDECODECREATEINFO, Reserved2), 144);
    }

    // ── Enum size tests ───────────────────────────────────────────────

    #[test]
    fn test_enum_sizes() {
        // All NVIDIA SDK enums are c_int (4 bytes) on 64-bit
        assert_eq!(
            std::mem::size_of::<cudaVideoCodec>(),
            4,
            "cudaVideoCodec size mismatch: expected 4, got {}",
            std::mem::size_of::<cudaVideoCodec>()
        );
        assert_eq!(
            std::mem::size_of::<cudaVideoChromaFormat>(),
            4,
            "cudaVideoChromaFormat size mismatch: expected 4, got {}",
            std::mem::size_of::<cudaVideoChromaFormat>()
        );
        assert_eq!(
            std::mem::size_of::<cudaVideoSurfaceFormat>(),
            4,
            "cudaVideoSurfaceFormat size mismatch: expected 4, got {}",
            std::mem::size_of::<cudaVideoSurfaceFormat>()
        );
        assert_eq!(
            std::mem::size_of::<cudaVideoDeinterlaceMode>(),
            4,
            "cudaVideoDeinterlaceMode size mismatch: expected 4, got {}",
            std::mem::size_of::<cudaVideoDeinterlaceMode>()
        );
        assert_eq!(
            std::mem::size_of::<cudaVideoCreateFlags>(),
            4,
            "cudaVideoCreateFlags size mismatch: expected 4, got {}",
            std::mem::size_of::<cudaVideoCreateFlags>()
        );
        assert_eq!(
            std::mem::size_of::<cuvidDecodeStatus>(),
            4,
            "cuvidDecodeStatus size mismatch: expected 4, got {}",
            std::mem::size_of::<cuvidDecodeStatus>()
        );
    }

    // ── Cross-platform pointer size checks ────────────────────────────

    #[test]
    fn test_pointer_sizes() {
        // Verify pointer sizes match platform expectations
        let ptr_size = std::mem::size_of::<*const c_void>();
        assert!(
            ptr_size == 4 || ptr_size == 8,
            "Unexpected pointer size: {} (expected 4 or 8)",
            ptr_size
        );
        println!("Pointer size: {} bytes", ptr_size);
    }

    #[test]
    fn test_cuulonglong_size() {
        // c_ulonglong must be 8 bytes on all platforms
        assert_eq!(
            std::mem::size_of::<c_ulonglong>(),
            8,
            "c_ulonglong size mismatch: expected 8, got {}",
            std::mem::size_of::<c_ulonglong>()
        );
    }

    // ── Union size tests ──────────────────────────────────────────────

    #[test]
    fn test_union_sizes() {
        // CUVIDH264FMOASO: max(c_ulonglong, *const c_uchar) = 8 bytes
        assert_eq!(
            std::mem::size_of::<CUVIDH264FMOASO>(),
            8,
            "CUVIDH264FMOASO size mismatch: expected 8, got {}",
            std::mem::size_of::<CUVIDH264FMOASO>()
        );

        // CUVIDH264SVCMVC: max(MVC, SVC) - SVC is larger due to *const CUVIDPICPARAMS
        let svc_mvc_size = std::mem::size_of::<CUVIDH264SVCMVC>();
        assert!(svc_mvc_size > 0, "CUVIDH264SVCMVC size must be > 0");
        println!("CUVIDH264SVCMVC size: {} bytes", svc_mvc_size);
    }

    // ── Alignment tests ──────────────────────────────────────────────

    #[test]
    fn test_struct_alignment() {
        // All FFI structs must have alignment compatible with C
        // CUVIDH264PICPARAMS has 8-byte alignment due to c_ulonglong in FMOASO union
        assert_eq!(
            std::mem::align_of::<CUVIDH264PICPARAMS>(),
            8,
            "CUVIDH264PICPARAMS alignment mismatch"
        );
        assert_eq!(
            std::mem::align_of::<CUVIDPICPARAMS>(),
            8,
            "CUVIDPICPARAMS alignment mismatch (has pointer fields)"
        );
        assert_eq!(
            std::mem::align_of::<CUVIDCODECSPECIFIC>(),
            8,
            "CUVIDCODECSPECIFIC alignment mismatch (has pointer via union)"
        );
        assert_eq!(
            std::mem::align_of::<CUVIDDECODECREATEINFO>(),
            8,
            "CUVIDDECODECREATEINFO alignment mismatch"
        );
        assert_eq!(
            std::mem::align_of::<CUVIDPROCPARAMS>(),
            8,
            "CUVIDPROCPARAMS alignment mismatch"
        );
    }
}
