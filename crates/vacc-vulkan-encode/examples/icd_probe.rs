//! Directly dlopen the NVIDIA Vulkan ICD and ask it for vkCreateInstance,
//! bypassing the loader, to isolate ICD init failures.
use std::ffi::CString;

unsafe extern "system" {
    fn dlopen(filename: *const i8, flags: i32) -> *mut std::ffi::c_void;
    fn dlerror() -> *mut i8;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8) -> *mut std::ffi::c_void;
}

fn main() {
    let libname = std::env::args().nth(1).unwrap_or_else(|| "libGLX_nvidia.so.0".into());
    let cname = CString::new(libname.clone()).unwrap();
    let lib = unsafe { dlopen(cname.as_ptr(), 2) }; // RTLD_NOW
    if lib.is_null() {
        let err = unsafe { dlerror() };
        let msg = if err.is_null() {
            String::from("(null)")
        } else {
            unsafe { std::ffi::CStr::from_ptr(err) }.to_string_lossy().into_owned()
        };
        println!("dlopen({libname}) FAILED: {msg}");
        return;
    }
    println!("dlopen({libname}) ok");
    let sym = c"vk_icdGetInstanceProcAddr";
    let get_proc_ptr = unsafe { dlsym(lib, sym.as_ptr()) };
    if get_proc_ptr.is_null() {
        println!("vk_icdGetInstanceProcAddr NOT FOUND");
        return;
    }
    type GetProc = unsafe extern "system" fn(u64, *const i8) -> *const u8;
    let get_proc: GetProc = unsafe { std::mem::transmute(get_proc_ptr) };
    for name in ["vkCreateInstance", "vkEnumeratePhysicalDevices"] {
        let c = CString::new(name).unwrap();
        let p = unsafe { get_proc(0, c.as_ptr()) };
        println!("{name}: {p:?}");
    }
}
