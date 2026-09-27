//! Common streaming tests for the software H.264 backend — implemented in
//! the backend-generic `vacc-golden-tests` framework.

use vacc_software_decode::SwH264Decoder;

#[test]
fn h264_common_stream_tests() {
    vacc_golden_tests::common_stream_tests::<SwH264Decoder>("h264_main.h264");
}
