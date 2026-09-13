//! Build script: compiles the hevc.js core (C++17, MIT license — see
//! `hevc/LICENSE`) and the vacc driver (`hevc_driver.cpp`) into a static
//! library. Bitstream parsing, POC computation and DPB management run in Rust
//! (vacc-parser); the C++ core handles pixel reconstruction (CABAC, prediction,
//! transform, deblocking, SAO).

use std::path::PathBuf;

fn main() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hevc");

    let mut cxx = cc::Build::new();
    cxx.cpp(true)
        .std("gnu++17")
        .include(&src)
        .flag_if_supported("-pthread")
        .warnings(false)
        .opt_level(2);

    // Test-only: enable HEVC_LOG diagnostics (HEVC_DEBUG_FILTER selects category).
    if std::env::var("HEVC_DEBUG").is_ok() {
        cxx.flag_if_supported("-DHEVC_DEBUG");
    }

    for file in [
        // hevc.js core (MIT, see hevc/LICENSE)
        "bitstream/bitstream_reader.cpp",
        "bitstream/nal_parser.cpp",
        "common/picture.cpp",
        "common/thread_pool.cpp",
        "decoding/cabac.cpp",
        "decoding/coding_tree.cpp",
        "decoding/decoder.cpp",
        "decoding/dpb.cpp",
        "decoding/interpolation.cpp",
        "decoding/inter_prediction.cpp",
        "decoding/intra_prediction.cpp",
        "decoding/residual_coding.cpp",
        "decoding/syntax_elements.cpp",
        "decoding/transform.cpp",
        "filters/deblocking.cpp",
        "filters/sao.cpp",
        "syntax/parameter_sets.cpp",
        "syntax/pps.cpp",
        "syntax/profile_tier_level.cpp",
        "syntax/slice_header.cpp",
        "syntax/sps.cpp",
        "syntax/vps.cpp",
        // vacc driver (C API over the hevc.js core)
        "hevc_driver.cpp",
        // Test-only oracles for differential Rust-vs-C++ kernel tests
        "hevc_test_api.cpp",
    ] {
        cxx.file(src.join(file));
    }

    cxx.compile("hevcsw");

    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed=hevc");
    println!("cargo:rerun-if-env-changed=HEVC_DEBUG");
}
