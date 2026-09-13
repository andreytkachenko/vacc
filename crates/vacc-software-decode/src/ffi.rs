//! FFI bindings to the C++ hevc.js core driver (`hevc_driver.h`), retained
//! **only as the differential-test oracle** for the Rust `hevc` port (the
//! production decode path is fully Rust; see `h265.rs`).

use std::os::raw::c_int;

/// Opaque C++ decoder context (matches `typedef struct hevcdec_context` in
/// hevc_driver.h). Zero-sized on purpose: the C side owns the real layout.
#[repr(C)]
pub struct hevcdec_context;

pub const HEVCDEC_OK: c_int = 0;

// `hevcdec_context` is a C opaque type (zero-sized here); that is the standard
// Rust representation for such pointers.
#[allow(improper_ctypes)]
unsafe extern "C" {
    pub fn hevcdec_create(nthreads: c_int) -> *mut hevcdec_context;
    pub fn hevcdec_destroy(ctx: *mut hevcdec_context);

    pub fn hevcdec_decode_picture(
        ctx: *mut hevcdec_context,
        data: *const u8,
        len: usize,
        out_y: *mut u8,
        out_ystride: c_int,
        out_u: *mut u8,
        out_ustride: c_int,
        out_v: *mut u8,
        out_vstride: c_int,
    ) -> c_int;
}
