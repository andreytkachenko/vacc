//! Compile the WGSL compute shaders to SPIR-V at build time (naga, no
//! external tools). Targets SPIR-V 1.0 so any Vulkan 1.0+ driver accepts
//! the modules.

use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    for name in ["yuv2rgb", "resize_yuv"] {
        let source =
            std::fs::read_to_string(Path::new("src/shaders").join(format!("{name}.wgsl")))
                .unwrap_or_else(|e| panic!("reading {name}.wgsl: {e}"));
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("parsing {name}.wgsl: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            naga::valid::Capabilities::default(),
        );
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("validating {name}.wgsl: {e:?}"));
        let words = naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .unwrap_or_else(|e| panic!("SPIR-V for {name}.wgsl: {e}"));
        let mut spv = Vec::with_capacity(words.len() * 4);
        for w in words {
            spv.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(Path::new(&out_dir).join(format!("{name}.spv")), spv).unwrap();
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shaders");
}
