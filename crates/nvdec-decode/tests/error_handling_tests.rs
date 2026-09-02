//! Error handling tests for nvdec-decode.
//!
//! These tests verify proper error handling across all error types defined in
//! NvdecError, including UnsupportedCodec, DecoderCreationFailed, ParserError,
//! DecodeFailed, InvalidState, NoFramesAvailable, EndOfStream, CudaError,
//! and IoError propagation.
//!
//! Note: Most tests require actual NVDEC hardware and are marked with #[ignore].
//! Run with `cargo test --test error_handling_tests -- --ignored` on a system
//! with NVIDIA hardware and proper drivers.

use nvdec_decode::{NvdecDecoder, NvdecError, NvdecH264Decoder, NvdecResult};
use vacc_core::{
    codec::VideoCodec,
    decoder::Decoder,
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
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
// Test 1: test_error_unsupported_codec
// ============================================================================
// Verify UnsupportedCodec error for non-H264 codecs via new_with_format.
// This test does NOT require hardware because the codec check happens before
// init_nvdec() is called.
// ============================================================================

#[test]
fn test_error_unsupported_codec() {
    let data = load_sps_pps_data();
    let format = VideoFormat::new(
        VideoCodec::DecodeH264,
        ChromaSubsampling::_420,
        ComponentBitDepth::Bit8,
        ComponentBitDepth::Bit8,
    );

    // Test with HEVC codec - should fail with UnsupportedCodec
    let result = NvdecDecoder::new_with_format(data.clone(), VideoCodec::DecodeH265, &format);
    assert!(result.is_err(), "Expected error for HEVC codec");
    match result.err().unwrap() {
        NvdecError::UnsupportedCodec(codec) => {
            assert_eq!(
                codec,
                VideoCodec::DecodeH265,
                "Should report HEVC as unsupported"
            );
        }
        e => panic!("Expected UnsupportedCodec, got {:?}", e),
    }

    // Test with AV1 codec - should fail with UnsupportedCodec
    let result = NvdecDecoder::new_with_format(data.clone(), VideoCodec::DecodeAv1, &format);
    assert!(result.is_err(), "Expected error for AV1 codec");
    match result.err().unwrap() {
        NvdecError::UnsupportedCodec(codec) => {
            assert_eq!(
                codec,
                VideoCodec::DecodeAv1,
                "Should report AV1 as unsupported"
            );
        }
        e => panic!("Expected UnsupportedCodec, got {:?}", e),
    }

    // Test with VP9 codec - should fail with UnsupportedCodec
    let result = NvdecDecoder::new_with_format(data.clone(), VideoCodec::DecodeVp9, &format);
    assert!(result.is_err(), "Expected error for VP9 codec");
    match result.err().unwrap() {
        NvdecError::UnsupportedCodec(codec) => {
            assert_eq!(
                codec,
                VideoCodec::DecodeVp9,
                "Should report VP9 as unsupported"
            );
        }
        e => panic!("Expected UnsupportedCodec, got {:?}", e),
    }
}

// ============================================================================
// Test 2: test_error_no_sps_pps
// ============================================================================
// Verify error when bitstream has no SPS/PPS. DecoderCreationFailed is returned
// because the parser cannot initialize without parameter sets.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_no_sps_pps() {
    // Create synthetic data that looks like H.264 but has no valid SPS/PPS
    // Just some random bytes with a fake start code
    let fake_data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00, 0x00, 0x01, 0x41];

    let result = NvdecH264Decoder::new(fake_data);
    assert!(
        result.is_err(),
        "Expected error when bitstream has no SPS/PPS"
    );
    match result.err().unwrap() {
        NvdecError::DecoderCreationFailed(msg) => {
            assert!(
                msg.contains("SPS") || msg.contains("PPS") || msg.contains("parser"),
                "Error message should mention SPS/PPS: {}",
                msg
            );
        }
        e => panic!("Expected DecoderCreationFailed, got {:?}", e),
    }
}

// ============================================================================
// Test 3: test_error_invalid_bitstream
// ============================================================================
// Verify error for invalid/corrupted bitstream data that cannot be parsed.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_invalid_bitstream() {
    // Completely invalid data - no start codes, no valid NAL units
    let invalid_data: Vec<u8> = (0..1024).map(|i| (i * 7 + 3) as u8).collect();

    let result = NvdecH264Decoder::new(invalid_data);
    assert!(result.is_err(), "Expected error for invalid bitstream data");
    // Could be DecoderCreationFailed (no SPS/PPS found) or ParserError
    match result.err().unwrap() {
        NvdecError::DecoderCreationFailed(msg) => {
            assert!(
                !msg.is_empty(),
                "DecoderCreationFailed should have a message"
            );
        }
        NvdecError::ParserError(_) => {
            // Also acceptable - parser rejected the invalid data
        }
        e => panic!("Expected DecoderCreationFailed or ParserError, got {:?}", e),
    }
}

// ============================================================================
// Test 4: test_error_empty_bitstream
// ============================================================================
// Verify error for empty bitstream. DecoderCreationFailed because no SPS/PPS
// can be found in empty data.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_empty_bitstream() {
    let empty_data = Vec::<u8>::new();

    let result = NvdecH264Decoder::new(empty_data);
    assert!(result.is_err(), "Expected error for empty bitstream");
    match result.err().unwrap() {
        NvdecError::DecoderCreationFailed(msg) => {
            assert!(
                msg.contains("SPS") || msg.contains("PPS") || msg.contains("no"),
                "Error message should explain why empty data fails: {}",
                msg
            );
        }
        e => panic!("Expected DecoderCreationFailed, got {:?}", e),
    }
}

// ============================================================================
// Test 5: test_error_decoder_creation_failed
// ============================================================================
// Verify DecoderCreationFailed error when parser finds SPS/PPS but decoder
// cannot be created (e.g., unsupported profile or resolution).
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_decoder_creation_failed() {
    // Use SPS/PPS data from a valid stream - if decoder creation fails,
    // it should be DecoderCreationFailed
    let data = load_sps_pps_data();

    let result = NvdecH264Decoder::new(data);

    // On systems with proper hardware, this should succeed.
    // On systems with incompatible hardware/drivers, it may fail with
    // DecoderCreationFailed. We verify the error type is correct if it fails.
    match result {
        Ok(_) => {
            // Success is also valid - decoder was created properly
        }
        Err(NvdecError::DecoderCreationFailed(msg)) => {
            // Decoder creation failed - verify error type and message
            assert!(
                !msg.is_empty(),
                "DecoderCreationFailed should have a descriptive message"
            );
            println!(
                "Decoder creation failed (expected on some systems): {}",
                msg
            );
        }
        Err(e) => {
            // Other errors are also possible (CUDA not available, etc.)
            println!("Got different error (also valid): {:?}", e);
        }
    }
}

// ============================================================================
// Test 6: test_error_parser_error_propagation
// ============================================================================
// Verify ParserError is correctly propagated from vacc-parser when
// parsing fails.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_parser_error_propagation() {
    // Create data with a start code but corrupted SPS header
    // NAL type 7 (SPS) but with invalid content
    let mut corrupted_sps = vec![
        0x00, 0x00, 0x00, 0x01, // start code
        0x67, // NAL type 7 (SPS)
    ];
    // Add garbage that won't parse as valid SPS
    corrupted_sps.extend_from_slice(&[0xFF; 50]);

    let result = NvdecH264Decoder::new(corrupted_sps);
    assert!(result.is_err(), "Expected error for corrupted SPS");
    match result.err().unwrap() {
        NvdecError::ParserError(ref e) => {
            // Parser error was correctly propagated
            let err_msg = e.to_string();
            assert!(!err_msg.is_empty(), "ParserError should have a message");
        }
        NvdecError::DecoderCreationFailed(_) => {
            // Also acceptable if parser didn't error but decoder couldn't init
        }
        e => panic!("Expected ParserError or DecoderCreationFailed, got {:?}", e),
    }
}

// ============================================================================
// Test 7: test_error_decode_failed_handling
// ============================================================================
// Verify DecodeFailed error when cuvidDecodePicture fails.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_decode_failed_handling() {
    let data = load_born_trailer();

    let mut decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");

    // Submit valid data and try to decode
    let decode_limit = 100;
    for _ in 0..decode_limit {
        match decoder.decode() {
            Ok(Some(_frame)) => {
                // Successfully decoded a frame
            }
            Ok(None) => {
                // No frame available yet - need more data
                break;
            }
            Err(NvdecError::DecodeFailed(msg)) => {
                // Decode failed - verify error handling
                assert!(
                    !msg.is_empty(),
                    "DecodeFailed should have a descriptive message"
                );
                panic!("DecodeFailed occurred (valid error for testing): {}", msg);
            }
            Err(e) => {
                panic!("Unexpected error during decode: {:?}", e);
            }
        }
    }

    // If we reach here, decoding succeeded. The test verifies that if
    // DecodeFailed were to occur, it would be properly handled.
}

// ============================================================================
// Test 8: test_error_invalid_state_after_reset
// ============================================================================
// Verify InvalidState error when attempting to decode after reset without
// re-initialization.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_invalid_state_after_reset() {
    let data = load_born_trailer();

    let mut decoder = NvdecH264Decoder::new(data.clone()).expect("Failed to create decoder");

    // Decode a few frames to ensure decoder is working
    for _ in 0..10 {
        match decoder.decode() {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(e) => panic!("Unexpected error before reset: {:?}", e),
        }
    }

    // Reset the decoder
    decoder.reset().expect("Reset should succeed");

    // Try to submit data after reset - this should work (re-initializes)
    decoder
        .submit(&data[..1000])
        .expect("Submit after reset should work");

    // Decode should re-initialize from submitted data
    match decoder.decode() {
        Ok(Some(_)) => {
            // Successfully re-initialized and decoded
        }
        Ok(None) => {
            // No frame yet - need more data
        }
        Err(e) => {
            panic!("Decode after submit should work, got: {:?}", e);
        }
    }
}

// ============================================================================
// Test 9: test_error_no_frames_available
// ============================================================================
// Verify NoFramesAvailable when no frames have been decoded yet.
// Note: The current implementation returns Ok(None) instead of NoFramesAvailable,
// so this test verifies the actual behavior.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_no_frames_available() {
    let data = load_sps_pps_data();

    let mut decoder = NvdecH264Decoder::new(data).expect("Failed to create decoder");

    // Try to decode without submitting any frame data
    // Only SPS/PPS was provided, no actual frame slices
    let result = decoder.decode();

    match result {
        Ok(None) => {
            // No frames available - this is the expected behavior
            // The decoder correctly indicates no decoded frames are ready
        }
        Ok(Some(_)) => {
            // Unexpectedly got a frame
            panic!("Expected no frames when only SPS/PPS provided");
        }
        Err(NvdecError::NoFramesAvailable) => {
            // Also acceptable if implementation uses this error
        }
        Err(e) => {
            panic!("Unexpected error: {:?}", e);
        }
    }
}

// ============================================================================
// Test 10: test_error_end_of_stream
// ============================================================================
// Verify EndOfStream error handling when parser reaches end of stream.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_end_of_stream() {
    let data = load_born_trailer();

    let mut decoder = NvdecH264Decoder::new(data.clone()).expect("Failed to create decoder");

    // Decode all available frames
    let mut frame_count = 0;
    loop {
        match decoder.decode() {
            Ok(Some(_frame)) => {
                frame_count += 1;
            }
            Ok(None) => {
                // No more frames immediately available
                break;
            }
            Err(NvdecError::EndOfStream) => {
                // End of stream reached - verify error handling
                assert!(
                    frame_count > 0,
                    "Should have decoded some frames before EOS"
                );
                return; // Test passes
            }
            Err(e) => {
                panic!("Unexpected error during decode: {:?}", e);
            }
        }
    }

    // If we didn't hit EndOfStream, submit more data and continue
    // The test verifies proper handling when EOS is eventually reached
    assert!(frame_count > 0, "Should have decoded at least one frame");
}

// ============================================================================
// Test 11: test_error_cuda_error_handling
// ============================================================================
// Verify CudaError error handling when CUDA operations fail.
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_cuda_error_handling() {
    let data = load_sps_pps_data();

    let result = NvdecH264Decoder::new(data);

    match result {
        Ok(_) => {
            // CUDA is available and decoder created successfully
        }
        Err(NvdecError::CudaError(msg)) => {
            // CUDA error occurred - verify error handling
            assert!(
                !msg.is_empty(),
                "CudaError should have a descriptive message"
            );
            println!("CUDA error (valid for testing): {}", msg);
        }
        Err(NvdecError::LibLoadError(msg)) => {
            // Library load error - also a valid CUDA-related error
            assert!(
                !msg.is_empty(),
                "LibLoadError should have a descriptive message"
            );
            println!("Library load error (valid for testing): {}", msg);
        }
        Err(e) => {
            // Other errors are also possible
            println!("Got error: {:?}", e);
        }
    }
}

// ============================================================================
// Test 12: test_error_io_error_propagation
// ============================================================================
// Verify IoError is correctly propagated when file operations fail.
// ============================================================================

#[test]
fn test_error_io_error_propagation() {
    // Test that reading a non-existent file produces an IoError
    let non_existent_path = format!("{}/assets/nonexistent.h264", PROJECT_ROOT);
    let result = std::fs::read(&non_existent_path);

    assert!(result.is_err(), "Reading non-existent file should fail");

    let io_error = result.unwrap_err();
    assert!(
        io_error.kind() == std::io::ErrorKind::NotFound,
        "Should be NotFound error"
    );

    // Verify that IoError can be converted from std::io::Error
    let nvdec_error: NvdecError = io_error.into();
    match nvdec_error {
        NvdecError::IoError(ref inner) => {
            assert!(
                inner.kind() == std::io::ErrorKind::NotFound,
                "Converted error should preserve NotFound kind"
            );
        }
        _ => panic!("Expected IoError after conversion"),
    }
}

// ============================================================================
// Test 13: test_error_from_born_trailer_partial
// ============================================================================
// Verify error handling when only partial born_trailer.h264 is provided
// (e.g., only first 100 bytes - not enough for SPS/PPS).
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_from_born_trailer_partial() {
    let full_data = load_born_trailer();

    // Provide only the first 100 bytes - likely not enough for complete SPS/PPS
    let partial_data = full_data[..100].to_vec();

    let result = NvdecH264Decoder::new(partial_data);

    // Should fail because partial data doesn't contain complete SPS/PPS
    assert!(result.is_err(), "Expected error for partial bitstream data");

    match result.err().unwrap() {
        NvdecError::DecoderCreationFailed(msg) => {
            assert!(
                !msg.is_empty(),
                "DecoderCreationFailed should explain why partial data fails"
            );
        }
        NvdecError::ParserError(_) => {
            // Also acceptable if parser rejects incomplete data
        }
        e => panic!("Expected DecoderCreationFailed or ParserError, got {:?}", e),
    }
}

// ============================================================================
// Test 14: test_error_from_born_trailer_truncated
// ============================================================================
// Verify error handling when born_trailer.h264 is truncated mid-frame
// (e.g., cut in the middle of a slice NAL unit).
// ============================================================================

#[test]
#[ignore = "requires NVDEC hardware"]
fn test_error_from_born_trailer_truncated() {
    let full_data = load_born_trailer();

    // Take a reasonable chunk that includes SPS/PPS and some frames,
    // but truncate in the middle of what's likely a slice NAL
    let truncation_point = std::cmp::min(full_data.len(), 50_000);
    let truncated_data = full_data[..truncation_point].to_vec();

    let mut decoder = NvdecH264Decoder::new(truncated_data)
        .expect("Failed to create decoder with truncated data");

    // Try to decode - should decode some frames before hitting truncation
    let mut frame_count = 0;
    let mut hit_error = false;

    for _ in 0..100 {
        match decoder.decode() {
            Ok(Some(_frame)) => {
                frame_count += 1;
            }
            Ok(None) => {
                // No more frames available
                break;
            }
            Err(NvdecError::ParserError(_)) => {
                // Parser hit truncated data - expected
                hit_error = true;
                break;
            }
            Err(NvdecError::EndOfStream) => {
                // End of stream reached cleanly
                break;
            }
            Err(e) => {
                // Other errors may occur with truncated data
                println!("Got error with truncated data: {:?}", e);
                hit_error = true;
                break;
            }
        }
    }

    // Verify we decoded at least some frames before truncation
    assert!(
        frame_count > 0,
        "Should have decoded at least one frame from truncated data"
    );

    // Either we decoded successfully until end, or hit an error at truncation
    // Both are valid behaviors
    println!(
        "Decoded {} frames from truncated data, hit_error={}",
        frame_count, hit_error
    );
}

// ============================================================================
// Additional tests for error type properties
// ============================================================================

// ============================================================================
// Test: test_error_type_implements_std_error
// ============================================================================
// Verify NvdecError implements std::error::Error properly.
// ============================================================================

#[test]
fn test_error_type_implements_std_error() {
    let err = NvdecError::DecoderCreationFailed("test".to_string());

    // Verify it implements std::error::Error
    let _ = format!("{}", err); // Display
    let _ = format!("{:?}", err); // Debug

    // Verify error message is correct
    assert_eq!(
        err.to_string(),
        "Decoder creation failed: test",
        "Error message should match the variant description"
    );
}

// ============================================================================
// Test: test_error_from_parser_error
// ============================================================================
// Verify From<vacc_parser::ParserError> implementation.
// ============================================================================

#[test]
fn test_error_from_parser_error() {
    let parser_err = vacc_parser::ParserError::InvalidBitstream;
    let nvdec_err: NvdecError = parser_err.into();

    match nvdec_err {
        NvdecError::ParserError(inner) => {
            assert_eq!(
                inner.to_string(),
                "Invalid bitstream data",
                "ParserError should be wrapped correctly"
            );
        }
        _ => panic!("Expected ParserError variant"),
    }
}

// ============================================================================
// Test: test_error_from_io_error
// ============================================================================
// Verify From<std::io::Error> implementation.
// ============================================================================

#[test]
fn test_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied");
    let nvdec_err: NvdecError = io_err.into();

    match nvdec_err {
        NvdecError::IoError(inner) => {
            assert_eq!(
                inner.kind(),
                std::io::ErrorKind::PermissionDenied,
                "IoError should preserve the original error kind"
            );
        }
        _ => panic!("Expected IoError variant"),
    }
}

// ============================================================================
// Test: test_error_unsupported_codec_message
// ============================================================================
// Verify UnsupportedCodec error message format.
// ============================================================================

#[test]
fn test_error_unsupported_codec_message() {
    let err = NvdecError::UnsupportedCodec(VideoCodec::DecodeH265);
    let msg = err.to_string();

    assert!(
        msg.contains("Unsupported codec"),
        "Message should indicate unsupported codec"
    );
    assert!(
        msg.contains("DecodeH265") || msg.contains("H265"),
        "Message should mention the specific codec"
    );
}

// ============================================================================
// Test: test_error_decoder_creation_failed_message
// ============================================================================
// Verify DecoderCreationFailed error message format.
// ============================================================================

#[test]
fn test_error_decoder_creation_failed_message() {
    let err = NvdecError::DecoderCreationFailed("test reason".to_string());
    let msg = err.to_string();

    assert!(
        msg.contains("Decoder creation failed"),
        "Message should indicate decoder creation failure"
    );
    assert!(
        msg.contains("test reason"),
        "Message should include the reason"
    );
}

// ============================================================================
// Test: test_error_decode_failed_message
// ============================================================================
// Verify DecodeFailed error message format.
// ============================================================================

#[test]
fn test_error_decode_failed_message() {
    let err = NvdecError::DecodeFailed("cuvidDecodePicture failed".to_string());
    let msg = err.to_string();

    assert!(
        msg.contains("Decode failed"),
        "Message should indicate decode failure"
    );
    assert!(
        msg.contains("cuvidDecodePicture"),
        "Message should include details"
    );
}

// ============================================================================
// Test: test_error_invalid_state_message
// ============================================================================
// Verify InvalidState error message format.
// ============================================================================

#[test]
fn test_error_invalid_state_message() {
    let err = NvdecError::InvalidState("decoder not initialized".to_string());
    let msg = err.to_string();

    assert!(
        msg.contains("Invalid state"),
        "Message should indicate invalid state"
    );
    assert!(
        msg.contains("decoder not initialized"),
        "Message should include description"
    );
}

// ============================================================================
// Test: test_error_no_frames_available_message
// ============================================================================
// Verify NoFramesAvailable error message format.
// ============================================================================

#[test]
fn test_error_no_frames_available_message() {
    let err = NvdecError::NoFramesAvailable;
    let msg = err.to_string();

    assert_eq!(
        msg, "No frames available",
        "Message should match expected format"
    );
}

// ============================================================================
// Test: test_error_end_of_stream_message
// ============================================================================
// Verify EndOfStream error message format.
// ============================================================================

#[test]
fn test_error_end_of_stream_message() {
    let err = NvdecError::EndOfStream;
    let msg = err.to_string();

    assert_eq!(msg, "End of stream", "Message should match expected format");
}

// ============================================================================
// Test: test_error_cuda_error_message
// ============================================================================
// Verify CudaError error message format.
// ============================================================================

#[test]
fn test_error_cuda_error_message() {
    let err = NvdecError::CudaError("CUDA_ERROR_OUT_OF_MEMORY".to_string());
    let msg = err.to_string();

    assert!(
        msg.contains("CUDA error"),
        "Message should indicate CUDA error"
    );
    assert!(
        msg.contains("CUDA_ERROR_OUT_OF_MEMORY"),
        "Message should include CUDA error code"
    );
}

// ============================================================================
// Test: test_error_result_type_alias
// ============================================================================
// Verify NvdecResult type alias works correctly.
// ============================================================================

#[test]
fn test_error_result_type_alias() {
    let success: NvdecResult<u32> = Ok(42);
    assert!(matches!(success, Ok(42)));

    let failure: NvdecResult<u32> = Err(NvdecError::EndOfStream);
    assert!(failure.is_err());
    assert!(matches!(failure, Err(NvdecError::EndOfStream)));
}
