//! Tests for the Decoder trait implementation in NvdecH264Decoder.
//!
//! These tests verify that NvdecH264Decoder correctly implements the Decoder
//! trait from vacc-core, including info(), submit(), decode(), flush(),
//! reset(), and new_with_format().
//!
//! Note: Most tests require actual NVDEC hardware and are marked with #[ignore].
//! Run with `cargo test --test decoder_trait_tests -- --ignored` on a system
//! with NVIDIA hardware and proper drivers.

use vacc_nvdec_decode::{NvdecDecoder, NvdecError, NvdecH264Decoder};
use vacc_core::{
    codec::VideoCodec,
    decoder::Decoder,
    format::{ChromaSubsampling, ComponentBitDepth, H264PictureLayout, VideoFormat},
};

/// Path to the project root.
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load the born_trailer.h264 test file.
fn load_born_trailer() -> Vec<u8> {
    let path = format!("{}/assets/born_trailer.h264", PROJECT_ROOT);
    std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read test file: {}", path))
}

/// Load only the SPS/PPS portion of the stream (first ~650 bytes).
fn load_sps_pps_data() -> Vec<u8> {
    load_born_trailer()[..650].to_vec()
}

// ============================================================================
// Test 1: test_decoder_info_backend
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_backend() {
    let data = load_sps_pps_data();
    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");

    let info = decoder.info();
    assert_eq!(info.backend, "nvdec", "Decoder backend should be 'nvdec'");
}

// ============================================================================
// Test 2: test_decoder_info_codec
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_codec() {
    let data = load_sps_pps_data();
    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");

    let info = decoder.info();
    assert_eq!(
        info.codec,
        VideoCodec::DecodeH264,
        "Decoder codec should be DecodeH264"
    );
}

// ============================================================================
// Test 3: test_decoder_info_coded_size
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_coded_size() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected coded dimensions
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_width = 0u32;
    let mut expected_height = 0u32;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        expected_width = (h264_sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
                        expected_height = if h264_sps.frame_mbs_only_flag {
                            (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                        } else {
                            (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                        };
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.coded_size.width, expected_width,
        "Coded width should match SPS dimensions"
    );
    assert_eq!(
        info.coded_size.height, expected_height,
        "Coded height should match SPS dimensions"
    );
}

// ============================================================================
// Test 4: test_decoder_info_display_size
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_display_size() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected display dimensions
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_width = 0u32;
    let mut expected_height = 0u32;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        let coded_width = (h264_sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
                        let coded_height = if h264_sps.frame_mbs_only_flag {
                            (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                        } else {
                            (h264_sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                        };

                        if h264_sps.frame_cropping_flag {
                            let left = (h264_sps.frame_crop_left_offset as i32) * 2;
                            let right =
                                coded_width as i32 - (h264_sps.frame_crop_right_offset as i32) * 2;
                            let top = if h264_sps.frame_mbs_only_flag {
                                (h264_sps.frame_crop_top_offset as i32) * 2
                            } else {
                                (h264_sps.frame_crop_top_offset as i32) * 4
                            };
                            let bottom = coded_height as i32
                                - if h264_sps.frame_mbs_only_flag {
                                    (h264_sps.frame_crop_bottom_offset as i32) * 2
                                } else {
                                    (h264_sps.frame_crop_bottom_offset as i32) * 4
                                };
                            expected_width = (right - left) as u32;
                            expected_height = (bottom - top) as u32;
                        } else {
                            expected_width = coded_width;
                            expected_height = coded_height;
                        }
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.display_size.width, expected_width,
        "Display width should match cropped dimensions"
    );
    assert_eq!(
        info.display_size.height, expected_height,
        "Display height should match cropped dimensions"
    );
}

// ============================================================================
// Test 5: test_decoder_info_chroma_subsampling
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_chroma_subsampling() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected chroma format
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_chroma = ChromaSubsampling::_420;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        expected_chroma = match h264_sps.chroma_format_idc {
                            0 => ChromaSubsampling::Monochrome,
                            1 => ChromaSubsampling::_420,
                            2 => ChromaSubsampling::_422,
                            3 => ChromaSubsampling::_444,
                            _ => ChromaSubsampling::_420,
                        };
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.chroma_subsampling, expected_chroma,
        "Chroma subsampling should match SPS chroma_format_idc"
    );
}

// ============================================================================
// Test 6: test_decoder_info_bit_depth
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_bit_depth() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected bit depth
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_luma_depth = ComponentBitDepth::Bit8;
    let mut expected_chroma_depth = ComponentBitDepth::Bit8;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        let bit_depth_minus8 = h264_sps.bit_depth_luma_minus8;
                        expected_luma_depth = match bit_depth_minus8 {
                            0 => ComponentBitDepth::Bit8,
                            2 => ComponentBitDepth::Bit10,
                            4 => ComponentBitDepth::Bit12,
                            _ => ComponentBitDepth::Bit8,
                        };
                        let chroma_minus8 = h264_sps.bit_depth_chroma_minus8;
                        expected_chroma_depth = match chroma_minus8 {
                            0 => ComponentBitDepth::Bit8,
                            2 => ComponentBitDepth::Bit10,
                            4 => ComponentBitDepth::Bit12,
                            _ => ComponentBitDepth::Bit8,
                        };
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.luma_bit_depth, expected_luma_depth,
        "Luma bit depth should match SPS bit_depth_luma_minus8"
    );
    assert_eq!(
        info.chroma_bit_depth, expected_chroma_depth,
        "Chroma bit depth should match SPS bit_depth_chroma_minus8"
    );
}

// ============================================================================
// Test 7: test_decoder_info_profile_idc
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_profile_idc() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected profile_idc
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_profile_idc = None;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        expected_profile_idc = Some(h264_sps.profile_idc as u32);
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.profile_idc, expected_profile_idc,
        "Profile IDC should match SPS profile_idc"
    );
}

// ============================================================================
// Test 8: test_decoder_info_dpb_slots
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_info_dpb_slots() {
    use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

    let data = load_sps_pps_data();

    // Parse SPS to get expected DPB slots
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut expected_dpb_slots = 0u32;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(sps_box) = sps {
                    if let Some(h264_sps) = sps_box.downcast_ref::<vacc_core::picture::H264Sps>() {
                        expected_dpb_slots = h264_sps.max_num_ref_frames + 1;
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => {}
            Err(_) => break,
        }
    }

    let decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");
    let info = decoder.info();

    assert_eq!(
        info.dpb_slots, expected_dpb_slots,
        "DPB slots should be max_num_ref_frames + 1"
    );
}

// ============================================================================
// Test 9: test_decoder_submit_and_decode
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_submit_and_decode() {
    let data = load_sps_pps_data();
    let mut decoder = NvdecDecoder::new(data).expect("Failed to create decoder");

    // Submit more data containing slices
    let full_data = load_born_trailer();
    let slice_data = &full_data[650..std::cmp::min(full_data.len(), 100_000)];
    decoder.submit(slice_data).expect("Failed to submit data");

    // Decode should return a frame
    let frame = decoder.decode().expect("Failed to decode");
    assert!(
        frame.is_some(),
        "decode() should return a frame after submit()"
    );

    let frame = frame.unwrap();
    assert!(!frame.skipped, "Frame should not be skipped");
    assert!(
        frame.width > 0 && frame.height > 0,
        "Frame should have valid dimensions"
    );
}

// ============================================================================
// Test 10: test_decoder_flush
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_flush() {
    let data = load_sps_pps_data();
    let mut decoder = NvdecDecoder::new(data).expect("Failed to create decoder");

    // Submit a chunk of data with multiple frames
    let full_data = load_born_trailer();
    let slice_data = &full_data[650..std::cmp::min(full_data.len(), 50_000)];
    decoder.submit(slice_data).expect("Failed to submit data");

    // Drain any immediately available frames
    while decoder.decode().unwrap().is_some() {}

    // Flush should return any remaining pending frames
    let _flushed = decoder.flush().expect("Failed to flush");

    // The flush should succeed (may return empty vec if no pending frames)
    // We just verify the operation works without error

    // After flush, no more frames should be available
    assert!(
        decoder.decode().unwrap().is_none(),
        "No frames should be available after flush"
    );
}

// ============================================================================
// Test 11: test_decoder_reset
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_reset() {
    let data = load_sps_pps_data();
    let mut decoder = NvdecDecoder::new(data).expect("Failed to create decoder");

    // Submit and decode some data first
    let full_data = load_born_trailer();
    let slice_data = &full_data[650..std::cmp::min(full_data.len(), 10_000)];
    decoder.submit(slice_data).expect("Failed to submit data");

    // Decode at least one frame
    while decoder.decode().unwrap().is_some() {}

    // Reset the decoder
    decoder.reset().expect("Failed to reset decoder");

    // After reset, no frames should be available
    assert!(
        decoder.decode().unwrap().is_none(),
        "No frames should be available after reset"
    );

    // Flush after reset should return empty
    let flushed = decoder.flush().expect("Failed to flush after reset");
    assert!(
        flushed.is_empty(),
        "Flush after reset should return empty vec"
    );
}

// ============================================================================
// Test 12: test_decoder_new_with_format_h264
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_new_with_format_h264() {
    let data = load_sps_pps_data();

    let format = VideoFormat {
        codec: VideoCodec::DecodeH264,
        chroma_subsampling: ChromaSubsampling::_420,
        luma_bit_depth: ComponentBitDepth::Bit8,
        chroma_bit_depth: ComponentBitDepth::Bit8,
        profile_idc: None,
        film_grain_support: false,
        h264_picture_layout: H264PictureLayout::Progressive,
    };

    // new_with_format with H264 codec should succeed
    let decoder = NvdecDecoder::new_with_format(data, VideoCodec::DecodeH264, &format)
        .expect("new_with_format with H264 should succeed");

    let info = decoder.info();
    assert_eq!(info.codec, VideoCodec::DecodeH264);
}

// ============================================================================
// Test 13: test_decoder_new_with_format_unsupported
// ============================================================================

#[test]
fn test_decoder_new_with_format_unsupported() {
    let data = vec![0u8; 100]; // dummy data, never parsed

    let format = VideoFormat {
        codec: VideoCodec::DecodeH264,
        chroma_subsampling: ChromaSubsampling::_420,
        luma_bit_depth: ComponentBitDepth::Bit8,
        chroma_bit_depth: ComponentBitDepth::Bit8,
        profile_idc: None,
        film_grain_support: false,
        h264_picture_layout: H264PictureLayout::Progressive,
    };

    // Test with HEVC codec - should fail immediately without needing hardware
    let result = NvdecDecoder::new_with_format(data.clone(), VideoCodec::DecodeH265, &format);
    assert!(result.is_err(), "new_with_format with HEVC should fail");
    match result {
        Err(NvdecError::UnsupportedCodec(codec)) => {
            assert_eq!(codec, VideoCodec::DecodeH265);
        }
        Err(e) => panic!("Expected UnsupportedCodec error, got: {}", e),
        Ok(_) => panic!("new_with_format with HEVC should fail"),
    }

    // Test with VP9 codec - should also fail
    let result = NvdecDecoder::new_with_format(data.clone(), VideoCodec::DecodeVp9, &format);
    assert!(result.is_err(), "new_with_format with VP9 should fail");
    match result {
        Err(NvdecError::UnsupportedCodec(codec)) => {
            assert_eq!(codec, VideoCodec::DecodeVp9);
        }
        Err(e) => panic!("Expected UnsupportedCodec error, got: {}", e),
        Ok(_) => panic!("new_with_format with VP9 should fail"),
    }

    // Test with AV1 codec - should also fail
    let result = NvdecDecoder::new_with_format(data, VideoCodec::DecodeAv1, &format);
    assert!(result.is_err(), "new_with_format with AV1 should fail");
    match result {
        Err(NvdecError::UnsupportedCodec(codec)) => {
            assert_eq!(codec, VideoCodec::DecodeAv1);
        }
        Err(e) => panic!("Expected UnsupportedCodec error, got: {}", e),
        Ok(_) => panic!("new_with_format with AV1 should fail"),
    }
}

// ============================================================================
// Test 14: test_decoder_from_born_trailer
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_decoder_from_born_trailer() {
    let data = load_born_trailer();

    // Create decoder with initial data
    let mut decoder = NvdecDecoder::new(data).expect("Failed to create decoder from born_trailer");

    // Verify decoder info
    let info = decoder.info();
    assert_eq!(info.backend, "nvdec");
    assert_eq!(info.codec, VideoCodec::DecodeH264);
    assert!(info.coded_size.width > 0 && info.coded_size.height > 0);
    assert!(info.display_size.width > 0 && info.display_size.height > 0);
    assert!(info.profile_idc.is_some());
    assert!(info.dpb_slots > 0);

    // Decode frames from the stream
    let mut frame_count = 0;
    loop {
        match decoder.decode() {
            Ok(Some(frame)) => {
                frame_count += 1;
                assert!(frame.width > 0 && frame.height > 0);
                assert!(!frame.skipped);

                // Verify frame dimensions match decoder info
                assert_eq!(
                    frame.width, info.display_size.width,
                    "Frame width should match display width"
                );
                assert_eq!(
                    frame.height, info.display_size.height,
                    "Frame height should match display height"
                );
            }
            Ok(None) => break,
            Err(e) => {
                // EndOfStream is acceptable
                if !matches!(e, vacc_nvdec_decode::NvdecError::EndOfStream) {
                    panic!("Decode error: {:?}", e);
                }
                break;
            }
        }
    }

    assert!(
        frame_count > 0,
        "Should have decoded at least one frame from born_trailer.h264"
    );
}
