//! Builds the vendored edge264 C subset (see `c/LICENSE_BSD.txt`) as a single
//! translation unit.
//!
//! `src/vacc_sw264.c` includes all decoder modules itself (the same
//! "include all" scheme upstream uses), so exactly one file is compiled:
//! the slice/intra/inter/mvpred/residual/deblock/bitstream routines plus the
//! glue API consumed by the Rust decoder.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = PathBuf::from(&manifest_dir).join("c").join("src");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // Upstream only supports GCC/Clang (no MSVC).
    if cc::Build::new()
        .try_get_compiler()
        .expect("no C compiler found")
        .is_like_msvc()
    {
        panic!(
            "vacc-sw-decode requires a GCC or Clang toolchain \
             (on Windows use MinGW-w64); MSVC is not supported"
        );
    }

    // -march=native only for true native builds (same as the upstream Makefile).
    let native =
        target_arch == env::consts::ARCH && target_os == env::consts::OS && target_os != "android";

    let mut build = cc::Build::new();
    build.include(&src).flag("-std=gnu11").flag("-O3");
    build.flag_if_supported("-flax-vector-conversions"); // GCC-only
    build.flag_if_supported("-Wno-override-init");
    if matches!(target_os.as_str(), "linux" | "macos" | "android") {
        build.flag("-pthread");
    } else if target_os == "windows" {
        build.flag_if_supported("-pthread");
    }
    if native {
        build.flag("-march=native");
    }
    // Opt-in AddressSanitizer for debugging (SW264_ASAN=1 cargo build).
    if std::env::var("SW264_ASAN").is_ok() {
        build.flag("-fsanitize=address").flag("-O1").flag("-g");
        println!("cargo:rustc-link-lib=asan");
    }

    build.file(src.join("vacc_sw264.c"));
    build.compile("vacc_sw264");

    if target_os != "wasi" && target_arch != "wasm32" {
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:rerun-if-changed=build.rs");
    for entry in std::fs::read_dir(&src).unwrap() {
        println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
    }
}
