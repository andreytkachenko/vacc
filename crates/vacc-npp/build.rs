//! Build script: link the CUDA *driver* (`libcuda`) when it is available.
//!
//! The driver entry points are called directly from `ffi.rs` (normal PLT
//! calls), so no runtime `dlopen` is needed for `libcuda`. Linking it at build
//! time keeps the driver loaded from process start and lets us degrade
//! gracefully on hosts without a GPU: when the link cannot be made,
//! `vacc_npp_cuda_linked` is not set and the crate falls back to the software
//! pipeline at runtime.
//!
//! The NPP libraries themselves (`libnppig`, `libnppicc`) are still resolved
//! at runtime through `libloading`, so there is no build-time dependency on
//! NPP.
//!
//! Historical note: the `201`/`CUDA_ERROR_INVALID_CONTEXT` failure that first
//! blocked this backend was *not* caused by how `libcuda` is loaded (linking
//! vs `dlopen`). It was caused by calling the legacy v1 entry point
//! (`cuMemAlloc`) instead of `cuMemAlloc_v2`; on this driver the v1 path skips
//! fetching the thread's current context and fails in userspace. See
//! `ffi.rs::cuda_driver` for details.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Re-run when the environment or the usual install locations change.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    for p in [
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/local/cuda",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }

    // Always advertise the cfg we may set below (avoids `unexpected_cfgs`).
    println!("cargo:rustc-check-cfg=cfg(vacc_npp_cuda_linked)");

    if link_libcuda() {
        println!("cargo:rustc-cfg=vacc_npp_cuda_linked");
    }
}

/// Attempt to emit the linker flags for `libcuda`. Returns true on success.
fn link_libcuda() -> bool {
    // 1) Prefer a CUDA-toolkit stub (`libcuda.so`, SONAME `libcuda.so.1`). It is
    //    provided by the toolkit and is the canonical thing to link against.
    for stub_dir in toolkit_stub_dirs() {
        let stub = stub_dir.join("libcuda.so");
        if stub.is_file() {
            link_lib(&stub_dir, &stub);
            return true;
        }
    }

    // 2) Fall back to the driver's versioned library (`libcuda.so.1`), which is
    //    installed by the GPU driver even without the toolkit. Link `-lcuda`
    //    against a symlink we create in OUT_DIR, since `-l:libcuda.so.1`
    //    cannot be expressed through the propagating `rustc-link-lib` output.
    for dir in driver_dirs() {
        let lib = dir.join("libcuda.so.1");
        if lib.is_file() {
            match link_via_out_symlink(&lib) {
                Ok(()) => return true,
                Err(e) => eprintln!("vacc-npp: {e}; trying next location"),
            }
        }
    }

    eprintln!("vacc-npp: libcuda not found; NPP backend will be unavailable at runtime");
    false
}

/// Emit the linker flags for `libcuda`.
///
/// The driver entry points are referenced directly (PLT calls in `ffi.rs`),
/// so the linker keeps `libcuda` in `DT_NEEDED` under the default
/// `--as-needed` behavior — the driver is mapped at process start and the
/// primary context is live before any NPP call.
///
/// `rustc-link-lib` (not `rustc-link-arg`) is used because only it propagates
/// to downstream binaries: this crate is a library, and consumers (the `vacc`
/// facade, examples) must link `libcuda` through us. The search directory must
/// contain a `libcuda.so` symlink (the toolkit stub, or one we create).
fn link_lib(search_dir: &Path, resolved: &Path) {
    println!("cargo:rustc-link-search=native={}", search_dir.display());
    println!("cargo:rustc-link-lib=cuda");
    eprintln!("vacc-npp: linking libcuda via {}", resolved.display());
}

/// Create a `libcuda.so -> <resolved>` symlink in OUT_DIR and link `-lcuda`
/// through it (for driver-only hosts without the toolkit stub).
fn link_via_out_symlink(resolved: &Path) -> Result<(), String> {
    let out = env::var_os("OUT_DIR").ok_or("OUT_DIR not set")?;
    let link = Path::new(&out).join("libcuda.so");
    if link.exists() || link.symlink_metadata().is_ok() {
        std::fs::remove_file(&link).map_err(|e| e.to_string())?;
    }
    std::os::unix::fs::symlink(resolved, &link).map_err(|e| e.to_string())?;
    link_lib(Path::new(&out), resolved);
    Ok(())
}

/// Candidate toolkit stub directories (`.../targets/x86_64-linux/lib/stubs`).
fn toolkit_stub_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(v) = env::var_os(var) {
            dirs.push(Path::new(&v).join("targets/x86_64-linux/lib/stubs"));
        }
    }
    // Common fixed locations, including the `cuda` symlink.
    for root in cuda_roots() {
        dirs.push(root.join("targets/x86_64-linux/lib/stubs"));
    }
    dirs
}

/// `libcuda.so.1` is installed by the driver in these locations.
fn driver_dirs() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib64"),
    ]
}

/// CUDA toolkit roots under `/usr/local` (e.g. `cuda`, `cuda-13.3`).
fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = vec![PathBuf::from("/usr/local/cuda")];
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name == "cuda" || name.starts_with("cuda-") {
                roots.push(e.path());
            }
        }
    }
    roots
}
