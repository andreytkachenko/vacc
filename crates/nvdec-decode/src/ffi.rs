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

use std::os::raw::{c_char, c_int, c_short, c_uchar, c_uint, c_ulong, c_ulonglong, c_ushort, c_void};

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
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDVP9PICPARAMS {
    pub width: c_uint,
    pub height: c_uint,
    pub LastRefIdx: c_uchar,
    pub GoldenRefIdx: c_uchar,
    pub AltRefIdx: c_uchar,
    pub colorSpace: c_uchar,
    pub profile: c_uint,
    pub frameContextIdx: c_uint,
    pub frameType: c_uint,
    pub showFrame: c_uint,
    pub errorResilient: c_uint,
    pub frameParallelDecoding: c_uint,
    pub subSamplingX: c_uint,
    pub subSamplingY: c_uint,
    pub intraOnly: c_uint,
    pub allow_high_precision_mv: c_uint,
    pub refreshEntropyProbs: c_uint,
    pub reserved2Bits: c_uint,
    pub reserved16Bits: c_uint,
    pub refFrameSignBias: [c_uchar; 4],
    pub bitDepthMinus8Luma: c_uchar,
    pub bitDepthMinus8Chroma: c_uchar,
    pub loopFilterLevel: c_uchar,
    pub loopFilterSharpness: c_uchar,
    pub modeRefLfEnabled: c_uchar,
    pub log2_tile_columns: c_uchar,
    pub log2_tile_rows: c_uchar,
    pub segmentEnabled: c_uint,
    pub segmentMapUpdate: c_uint,
    pub segmentMapTemporalUpdate: c_uint,
    pub segmentFeatureMode: c_uint,
    pub reserved4Bits: c_uint,
    pub segmentFeatureEnable: [[c_uchar; 4]; 8],
    pub segmentFeatureData: [[c_int; 4]; 8],
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
    pub mbRefLfDelta: [c_uint; 4],
    pub mbModeLfDelta: [c_uint; 2],
    pub frameTagSize: c_uint,
    pub offsetToDctParts: c_uint,
    pub reserved128Bits: [c_uint; 4],
}

/// AV1 picture parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDAV1PICPARAMS {
    pub width: c_uint,
    pub height: c_uint,
    pub frame_offset: c_uint,
    pub decodePicIdx: c_int,
    // sequence header
    pub profile: c_uint,
    pub use_128x128_superblock: c_uint,
    pub subsampling_x: c_uint,
    pub subsampling_y: c_uint,
    pub mono_chrome: c_uint,
    pub bit_depth_minus8: c_uint,
    pub enable_filter_intra: c_uint,
    pub enable_intra_edge_filter: c_uint,
    pub enable_interintra_compound: c_uint,
    pub enable_masked_compound: c_uint,
    pub enable_dual_filter: c_uint,
    pub enable_order_hint: c_uint,
    pub order_hint_bits_minus1: c_uint,
    pub enable_jnt_comp: c_uint,
    pub enable_superres: c_uint,
    pub enable_cdef: c_uint,
    pub enable_restoration: c_uint,
    pub enable_fgs: c_uint,
    pub reserved0_7bits: c_uint,
    // frame header
    pub frame_type: c_uint,
    pub show_frame: c_uint,
    pub disable_cdf_update: c_uint,
    pub allow_screen_content_tools: c_uint,
    pub force_integer_mv: c_uint,
    pub coded_denom: c_uint,
    pub allow_intrabc: c_uint,
    pub allow_high_precision_mv: c_uint,
    pub interp_filter: c_uint,
    pub switchable_motion_mode: c_uint,
    pub use_ref_frame_mvs: c_uint,
    pub disable_frame_end_update_cdf: c_uint,
    pub delta_q_present: c_uint,
    pub delta_q_res: c_uint,
    pub using_qmatrix: c_uint,
    pub coded_lossless: c_uint,
    pub use_superres: c_uint,
    pub tx_mode: c_uint,
    pub reference_mode: c_uint,
    pub allow_warped_motion: c_uint,
    pub reduced_tx_set: c_uint,
    pub skip_mode: c_uint,
    pub reserved1_3bits: c_uint,
    // tiling info
    pub num_tile_cols: c_uint,
    pub num_tile_rows: c_uint,
    pub context_update_tile_id: c_uint,
    pub tile_widths: [c_uint; 64],
    pub tile_heights: [c_uint; 64],
    // CDEF
    pub cdef_damping_minus_3: c_uint,
    pub cdef_bits: c_uint,
    pub reserved2_4bits: c_uint,
    pub cdef_y_strength: [c_uchar; 8],
    pub cdef_uv_strength: [c_uchar; 8],
    // SkipModeFrames
    pub SkipModeFrame0: c_uchar,
    pub SkipModeFrame1: c_uchar,
    // qp information
    pub base_qindex: c_uchar,
    pub qp_y_dc_delta_q: c_char,
    pub qp_u_dc_delta_q: c_char,
    pub qp_v_dc_delta_q: c_char,
    pub qp_u_ac_delta_q: c_char,
    pub qp_v_ac_delta_q: c_char,
    pub qm_y: c_uchar,
    pub qm_u: c_uchar,
    pub qm_v: c_uchar,
    // segmentation
    pub segmentation_enabled: c_uint,
    pub segmentation_update_map: c_uint,
    pub segmentation_update_data: c_uint,
    pub segmentation_temporal_update: c_uint,
    pub reserved3_4bits: c_uint,
    pub segmentation_feature_data: [[c_int; 8]; 8],
    pub segmentation_feature_mask: [c_uchar; 8],
    // loopfilter
    pub loop_filter_level: [c_uchar; 2],
    pub loop_filter_level_u: c_uchar,
    pub loop_filter_level_v: c_uchar,
    pub loop_filter_sharpness: c_uchar,
    pub loop_filter_ref_deltas: [c_char; 8],
    pub loop_filter_mode_deltas: [c_char; 2],
    pub loop_filter_delta_enabled: c_uint,
    pub loop_filter_delta_update: c_uint,
    pub delta_lf_present: c_uint,
    pub delta_lf_res: c_uint,
    pub delta_lf_multi: c_uint,
    pub reserved4_2bits: c_uint,
    // restoration
    pub lr_unit_size: [c_uchar; 3],
    pub lr_type: [c_uchar; 3],
    // reference frames
    pub primary_ref_frame: c_uchar,
    pub ref_frame_map: [c_uchar; 8],
    pub temporal_layer_id: c_uint,
    pub spatial_layer_id: c_uint,
    pub reserved5_32bits: [c_uchar; 4],
    // ref frame list
    pub ref_frame: [CUVIDAV1REFFRAME; 7],
    // global motion
    pub global_motion: [CUVIDAV1GLOBALMOTION; 7],
    // film grain params
    pub apply_grain: c_uint,
    pub overlap_flag: c_uint,
    pub scaling_shift_minus8: c_uint,
    pub chroma_scaling_from_luma: c_uint,
    pub ar_coeff_lag: c_uint,
    pub ar_coeff_shift_minus6: c_uint,
    pub grain_scale_shift: c_uint,
    pub clip_to_restricted_range: c_uint,
    pub reserved6_4bits: c_uint,
    pub num_y_points: c_uchar,
    pub scaling_points_y: [[c_uchar; 2]; 14],
    pub num_cb_points: c_uchar,
    pub scaling_points_cb: [[c_uchar; 2]; 10],
    pub num_cr_points: c_uchar,
    pub scaling_points_cr: [[c_uchar; 2]; 10],
    pub reserved7_8bits: c_uchar,
    pub random_seed: c_uint,
    pub ar_coeffs_y: [c_int; 24],
    pub ar_coeffs_cb: [c_int; 25],
    pub ar_coeffs_cr: [c_int; 25],
    pub cb_mult: c_uchar,
    pub cb_luma_mult: c_uchar,
    pub cb_offset: c_int,
    pub cr_mult: c_uchar,
    pub cr_luma_mult: c_uchar,
    pub cr_offset: c_int,
    pub reserved: [c_int; 7],
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

/// AV1 global motion structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDAV1GLOBALMOTION {
    pub invalid: c_uint,
    pub wmtype: c_uint,
    pub reserved5Bits: c_uint,
    pub reserved24Bits: [c_char; 3],
    pub wmmat: [c_int; 6],
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
        assert_eq!(size, 984, "CUVIDH264PICPARAMS size mismatch: expected 984, got {}", size);
    }

    #[test]
    fn test_cuvid_hevc_picparams_size() {
        let size = std::mem::size_of::<CUVIDHEVCPICPARAMS>();
        println!("CUVIDHEVCPICPARAMS size: {}", size);
        // NVIDIA SDK: 1568 bytes on 64-bit (with RExt fields)
        assert_eq!(size, 1484, "CUVIDHEVCPICPARAMS size mismatch: expected 1484, got {}", size);
    }

    #[test]
    fn test_cuvid_vp9_picparams_size() {
        let size = std::mem::size_of::<CUVIDVP9PICPARAMS>();
        println!("CUVIDVP9PICPARAMS size: {}", size);
        // NVIDIA SDK: 352 bytes on 64-bit
        assert_eq!(size, 352, "CUVIDVP9PICPARAMS size mismatch: expected 352, got {}", size);
    }

    #[test]
    fn test_cuvid_av1_picparams_size() {
        let size = std::mem::size_of::<CUVIDAV1PICPARAMS>();
        println!("CUVIDAV1PICPARAMS size: {}", size);
        // NVIDIA SDK: 1916 bytes on 64-bit (with film grain params)
        assert_eq!(size, 1916, "CUVIDAV1PICPARAMS size mismatch: expected 1916, got {}", size);
    }

    #[test]
    fn test_cuvid_picparams_size() {
        let size = std::mem::size_of::<CUVIDPICPARAMS>();
        println!("CUVIDPICPARAMS size: {}", size);
        // NVIDIA SDK: 4280 bytes on 64-bit (includes 4096-byte union)
        assert_eq!(size, 4280, "CUVIDPICPARAMS size mismatch: expected 4280, got {}", size);
    }

    #[test]
    fn test_cuvid_codec_specific_size() {
        let size = std::mem::size_of::<CUVIDCODECSPECIFIC>();
        println!("CUVIDCODECSPECIFIC size: {}", size);
        // NVIDIA SDK: 4096 bytes (union of codec-specific params + reserved)
        assert_eq!(size, 4096, "CUVIDCODECSPECIFIC size mismatch: expected 4096, got {}", size);
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
        assert_eq!(size, 28, "CUVIDH264DPBENTRY size mismatch: expected 28, got {}", size);
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
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, direct_8x8_inference_flag), 20);
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
        assert_eq!(offset_of!(CUVIDH264PICPARAMS, constrained_intra_pred_flag), 76);
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
        assert!(
            svc_mvc_size > 0,
            "CUVIDH264SVCMVC size must be > 0"
        );
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
