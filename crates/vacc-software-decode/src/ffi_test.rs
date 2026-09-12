//! Test-only FFI bindings to the C++ differential-testing oracles
//! (`hevc/hevc_test_api.h`). Only used from unit tests.

#[allow(non_snake_case)]
unsafe extern "C" {
    pub fn hevcdec_test_transform_inverse(
        log2TrafoSize: i32,
        cIdx: i32,
        is_intra: i32,
        transform_skip: i32,
        bit_depth: i32,
        scaled: *const i16,
        residual: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_dequant(
        bit_depth_luma: i32,
        bit_depth_chroma: i32,
        cIdx: i32,
        log2TrafoSize: i32,
        qp: i32,
        cu_is_intra: i32,
        use_scaling_list: i32,
        pps_scaling_list_present: i32,
        sl_data: *const u8,
        sl_dc: *const u8,
        pps_sl_data: *const u8,
        pps_sl_dc: *const u8,
        coefficients: *const i16,
        scaled: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_interpolate_luma(
        plane: *const u16,
        picW: i32,
        picH: i32,
        stride: i32,
        xInt: i32,
        yInt: i32,
        xFrac: i32,
        yFrac: i32,
        nPbW: i32,
        nPbH: i32,
        bitDepth: i32,
        pred: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_interpolate_chroma(
        plane: *const u16,
        cIdx: i32,
        picW: i32,
        picH: i32,
        stride: i32,
        xInt: i32,
        yInt: i32,
        xFrac: i32,
        yFrac: i32,
        nPbWC: i32,
        nPbHC: i32,
        bitDepth: i32,
        pred: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_weighted_pred_default(
        predL0: *const i16,
        predL1: *const i16,
        flagL0: i32,
        flagL1: i32,
        nSamples: i32,
        bitDepth: i32,
        output: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_weighted_pred_explicit(
        predL0: *const i16,
        predL1: *const i16,
        flagL0: i32,
        flagL1: i32,
        refIdxL0: i32,
        refIdxL1: i32,
        cIdx: i32,
        nSamples: i32,
        bitDepth: i32,
        luma_log2_weight_denom: u32,
        delta_chroma_log2_weight_denom: i32,
        w_luma: *const i16,
        o_luma: *const i16,
        w_chroma: *const i16,
        o_chroma: *const i16,
        output: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_intra_predict(
        picW: i32,
        picH: i32,
        ctbSizeY: i32,
        minTbSizeY: i32,
        bitDepth: i32,
        chromaArrayType: i32,
        subW: i32,
        subH: i32,
        intra_smoothing_disabled: i32,
        strong_smoothing_enabled: i32,
        plane: *const u16,
        stride: i32,
        x0: i32,
        y0: i32,
        log2PredSize: i32,
        cIdx: i32,
        intra_mode: i32,
        pred: *mut i16,
    ) -> i32;

    pub fn hevcdec_test_cabac_run(
        data: *const u8,
        len_bytes: i32,
        slice_type: i32,
        qp: i32,
        cabac_init_flag: i32,
        ops: *const u8,
        args: *const i32,
        n_ops: i32,
        out: *mut i32,
        max_out: i32,
        final_range: *mut u16,
        final_offset: *mut u16,
        final_ctx: *mut u8,
    ) -> i32;
}
