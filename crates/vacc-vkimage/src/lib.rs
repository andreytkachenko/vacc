//! GPU image pipeline on Vulkan compute: Y'CbCr -> RGB conversion,
//! bilinear 4:2:0 resize, and affine warp (inverse mapping over RGBA32).
//!
//! A process-wide device context (lazily created, shared across decoders)
//! runs the same [`ImageConfig`] steps as the reference software pipeline in
//! `vacc-image`:
//!
//! - `rgb` only -> a single yuv2rgb pass
//! - `scale` only -> a single bilinear resize pass (8-bit output)
//! - `scale` + `rgb` -> resize into an 8-bit scratch, then yuv2rgb
//! - `affine` (+ `rgb`) -> yuv2rgb, then an inverse-mapping warp pass over
//!   the RGBA32 result (nearest / bilinear / bicubic)
//!
//! Only [`Interpolation::Bilinear`] resize and 8/10-bit sources are handled;
//! anything else (box/bicubic resize, box warp, 12-bit) returns
//! [`ImageError::Unsupported`] so the caller can fall back to software.
//!
//! All plane data is exchanged through one host-visible, host-coherent arena
//! buffer of `u32` words (one word per sample; naga has no 8-bit integer
//! type). The CPU copies source planes in — down-casting top-justified
//! 10-bit samples to 8 on the way — the GPU reads/writes it directly, and
//! after a fence wait the CPU repacks the result into tight bytes. There is
//! no device-local staging; the win is the ALU work (conversion/resampling)
//! on the GPU.

mod shaders;

use std::sync::{Mutex, OnceLock};

use ash::vk;
use vacc_image::{
    table, ImageConfig, ImageError, ImageResult, Interpolation, ProcessedFrame, RgbChannels,
    YuvImage, YuvLayout,
};

const FENCE_TIMEOUT_NS: u64 = 30 * 1_000 * 1_000 * 1_000;
const WORKGROUP: u32 = 8;
/// One uniform-buffer slot per pipeline (both parameter blocks fit).
const UNIFORM_SLOT: u64 = 256;
/// Command-buffer ring; frames are fully serialized by the fence, so one
/// in-flight buffer per slot suffices.
const CMD_SLOTS: usize = 8;
/// Arena growth step to avoid re-allocating every frame.
const ARENA_STEP: u64 = 1 << 20;

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// Whether a Vulkan compute device is available on this host (cached).
pub fn is_available() -> bool {
    global().is_some()
}

fn global() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| match create() {
        Ok(g) => Some(Mutex::new(g)),
        Err(e) => {
            log::debug!("vacc-vkimage: Vulkan unavailable: {e}");
            None
        }
    })
    .as_ref()
}

/// Run `cfg` on the GPU. See the module docs for supported combinations.
pub fn process(img: &YuvImage, cfg: &ImageConfig) -> ImageResult<ProcessedFrame> {
    let g = global().ok_or_else(|| ImageError::Unsupported("no Vulkan compute device".into()))?;
    let mut g = g.lock().unwrap();
    g.run(img, cfg)
}

/// yuv2rgb push-constant block (must mirror the WGSL `Params`). Offsets and
/// pitches are in words.
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    src_w: u32,
    src_h: u32,
    y_off: u32,
    y_pitch: u32,
    cb_off: u32,
    cb_pitch: u32,
    cr_off: u32,
    cr_pitch: u32,
    semi: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    ky: i32,
    r_cr: i32,
    r_off: i32,
    g_cb: i32,
    g_cr: i32,
    g_off: i32,
    b_cb: i32,
    b_off: i32,
}

/// resize_yuv push-constant block (must mirror the WGSL `Params`). Offsets
/// and pitches are in words; `chroma_stride` is 2 for interleaved input.
#[repr(C)]
#[derive(Clone, Copy)]
struct ResizeParams {
    src_w: u32,
    src_h: u32,
    y_off: u32,
    y_pitch: u32,
    cb_off: u32,
    cb_pitch: u32,
    cr_off: u32,
    cr_pitch: u32,
    chroma_stride: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
}

/// warp_rgb push-constant block (must mirror the WGSL `Params`). Offsets and
/// pitches are in words; `m*` is the *inverse* transform; `interp` is
/// 0 = nearest, 1 = bilinear, 2 = bicubic.
#[repr(C)]
#[derive(Clone, Copy)]
struct WarpParams {
    src_w: u32,
    src_h: u32,
    src_off: u32,
    src_pitch: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    interp: u32,
}

fn params_bytes<T>(p: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Format an ash error for messages.
fn vk_err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

/// Copy `height` rows of `row_samples` samples (from a plane with byte
/// `pitch`) into the arena, one u32 word per sample. 10-bit top-justified
/// samples are down-cast to 8 (`>> 6`), matching the reference pipeline's
/// exact down-cast.
fn copy_plane_words(
    src: &[u8],
    pitch: usize,
    row_samples: usize,
    height: usize,
    bpsb: usize,
    dst: *mut u32,
) {
    for r in 0..height {
        let row = &src[r * pitch..];
        let mut d = unsafe { dst.add(r * row_samples) };
        if bpsb == 1 {
            for c in 0..row_samples {
                unsafe {
                    *d = row[c] as u32;
                    d = d.add(1);
                }
            }
        } else {
            for c in 0..row_samples {
                let raw = u16::from_le_bytes([row[c * 2], row[c * 2 + 1]]);
                unsafe {
                    *d = (raw >> 6) as u32;
                    d = d.add(1);
                }
            }
        }
    }
}

// Raw pointers refer to device-mapped memory owned by this struct and are
// only ever dereferenced while holding the surrounding mutex.
unsafe impl Send for Gpu {}

struct Gpu {
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    cmd_buffers: Vec<vk::CommandBuffer>,
    cmd_slot: usize,
    fence: vk::Fence,
    pipeline_layout: vk::PipelineLayout,
    conv_pipeline: vk::Pipeline,
    resize_pipeline: vk::Pipeline,
    warp_pipeline: vk::Pipeline,
    desc_sets: [vk::DescriptorSet; 3], // [conv, resize, warp]
    uniform_buf: vk::Buffer,
    uniform_ptr: *mut u8,
    arena: vk::Buffer,
    arena_mem: vk::DeviceMemory,
    arena_ptr: *mut u8,
    arena_size: usize, // bytes
}

fn create() -> Result<Gpu, String> {
    let entry = unsafe { ash::Entry::load() }.map_err(vk_err)?;

    let app_name = std::ffi::CString::new("vacc-vkimage").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name.as_c_str())
        .api_version(vk::API_VERSION_1_0);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.map_err(vk_err)?;

    // Pick the best physical device with a compute queue (discrete GPU
    // preferred; llvmpipe-class CPU devices are a last resort).
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_err)?;
    let mut best: Option<(vk::PhysicalDevice, u32, u8)> = None;
    for pd in physicals {
        let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(qf) = qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            _ => 1,
        };
        if best.map_or(true, |(_, _, s)| score > s) {
            best = Some((pd, qf as u32, score));
        }
    }
    let (physical, queue_family, _) =
        best.ok_or_else(|| "no physical device with a compute queue".to_string())?;

    let queue_infos = vec![
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0f32])
    ];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
    let device = unsafe { instance.create_device(physical, &device_info, None) }
        .map_err(vk_err)?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family);
    let cmd_pool =
        unsafe { device.create_command_pool(&pool_info, None) }.map_err(vk_err)?;
    let cb_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(CMD_SLOTS as u32);
    let cmd_buffers =
        unsafe { device.allocate_command_buffers(&cb_alloc) }.map_err(vk_err)?;
    let fence =
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_err)?;

    // One descriptor set layout (storage src + storage dst) shared by both
    // pipelines; two sets, one per pipeline.
    let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = [0u32, 1].map(|b| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    })
    .into();
    bindings.push(
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    );
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(vk_err)?;
    let layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(vk_err)?;

    let make_pipeline = |spv: &[u8]| -> Result<vk::Pipeline, String> {
        // ash takes the module code as u32 words (little-endian on disk).
        let mut words = Vec::with_capacity(spv.len() / 4);
        for chunk in spv.chunks_exact(4) {
            words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )
        }
        .map_err(vk_err)?;
        let entry = std::ffi::CString::new("main").unwrap();
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module);
        stage.p_name = entry.as_ptr();
        let ps = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipes = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[ps], None)
        }
        .map_err(|(_, e)| format!("pipeline compile: {e:?}"))?;
        unsafe { device.destroy_shader_module(module, None) };
        Ok(pipes[0])
    };
    let conv_pipeline = make_pipeline(shaders::YUV2RGB)?;
    let resize_pipeline = make_pipeline(shaders::RESIZE_YUV)?;
    let warp_pipeline = make_pipeline(shaders::WARP_RGB)?;

    let arena_size = 4u64 * ARENA_STEP;
    let (arena, arena_mem, arena_ptr) =
        alloc_host_visible(&device, &instance, physical, arena_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(3)
        .pool_sizes(&[
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 6,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 3,
            },
        ]);
    let desc_pool =
        unsafe { device.create_descriptor_pool(&pool_info, None) }.map_err(vk_err)?;
    let desc_sets_arr = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&[set_layout; 3]),
        )
    }
    .map_err(vk_err)?;
    let desc_sets = [desc_sets_arr[0], desc_sets_arr[1], desc_sets_arr[2]];

    // Uniform ring: one slot per pass (conv, resize, warp).
    let (uniform_buf, _uniform_mem, uniform_ptr) =
        alloc_host_visible(&device, &instance, physical, 3 * UNIFORM_SLOT, vk::BufferUsageFlags::UNIFORM_BUFFER)?;

    let gpu = Gpu {
        instance,
        physical,
        device,
        queue,
        cmd_buffers,
        cmd_slot: 0,
        fence,
        pipeline_layout,
        conv_pipeline,
        resize_pipeline,
        warp_pipeline,
        desc_sets,
        uniform_buf,
        uniform_ptr,
        arena,
        arena_mem,
        arena_ptr,
        arena_size: arena_size as usize,
    };
    gpu.update_sets();
    Ok(gpu)
}

/// Create a host-visible coherent buffer and map it.
fn alloc_host_visible(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
    let bi = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&bi, None) }.map_err(vk_err)?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let idx = (0..mem_props.memory_type_count as u32)
        .find(|&t| {
            (reqs.memory_type_bits >> t) & 1 != 0
                && mem_props.memory_types[t as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        })
        .ok_or_else(|| "no host-visible coherent memory type for the arena".to_string())?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(idx);
    let memory = unsafe { device.allocate_memory(&alloc, None) }.map_err(vk_err)?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.map_err(vk_err)?;
    let ptr = unsafe {
        device.map_memory(memory, 0, reqs.size, vk::MemoryMapFlags::empty())
    }
    .map_err(vk_err)? as *mut u8;
    Ok((buffer, memory, ptr))
}

impl Gpu {
    fn arena_words(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(self.arena_ptr as *const u32, self.arena_size / 4)
        }
    }

    fn update_sets(&self) {
        let arena_info = vk::DescriptorBufferInfo::default()
            .buffer(self.arena)
            .range(u64::MAX);
        let arena_arr = [arena_info];

        // One buffer info per set (slot i of the uniform ring); must outlive
        // `writes`.
        let mut uniform_arrays: [[vk::DescriptorBufferInfo; 1]; 3] =
            std::array::from_fn(|_| [vk::DescriptorBufferInfo::default()]);
        for (i, _) in self.desc_sets.iter().enumerate() {
            uniform_arrays[i][0] = vk::DescriptorBufferInfo::default()
                .buffer(self.uniform_buf)
                .offset(i as u64 * UNIFORM_SLOT)
                .range(UNIFORM_SLOT);
        }
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for (i, set) in self.desc_sets.iter().enumerate() {
            let uniform_arr = &uniform_arrays[i];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&arena_arr),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&arena_arr),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(uniform_arr),
            );
        }
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    fn ensure_arena(&mut self, need_bytes: usize) -> Result<(), String> {
        if need_bytes <= self.arena_size {
            return Ok(());
        }
        let size = (need_bytes as u64).div_ceil(ARENA_STEP) * ARENA_STEP;
        unsafe {
            self.device.destroy_buffer(self.arena, None);
            self.device.free_memory(self.arena_mem, None);
        }
        let (arena, mem, ptr) =
            alloc_host_visible(&self.device, &self.instance, self.physical, size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        self.arena = arena;
        self.arena_mem = mem;
        self.arena_ptr = ptr;
        self.arena_size = size as usize;
        self.update_sets();
        Ok(())
    }

    fn run(&mut self, img: &YuvImage, cfg: &ImageConfig) -> ImageResult<ProcessedFrame> {
        if cfg.is_noop() {
            return Err(ImageError::Unsupported("no-op image config".into()));
        }
        let warp = cfg.affine;
        if let Some(w) = warp {
            if cfg.rgb.is_none() {
                return Err(ImageError::Unsupported(
                    "affine warp requires RGB output".into(),
                ));
            }
            if !w.interpolation.supports_warp() {
                return Err(ImageError::Unsupported(
                    "box interpolation is not supported for affine warp".into(),
                ));
            }
        }
        if !(8..=10).contains(&img.bits_per_sample) {
            return Err(ImageError::Unsupported(
                "vulkan path supports 8/10-bit sources".into(),
            ));
        }
        let scale = cfg.scale;
        if let Some(s) = scale {
            if s.filter != Interpolation::Bilinear {
                return Err(ImageError::Unsupported(
                    "vulkan resize supports bilinear only".into(),
                ));
            }
            if s.width == 0 || s.height == 0 {
                return Err(ImageError::InvalidDimensions("scale target must be non-zero".into()));
            }
        }
        let (sw, sh) = (img.width, img.height);
        if sw == 0 || sh == 0 {
            return Err(ImageError::InvalidDimensions("empty source".into()));
        }

        let semi_in = img.layout_hint == YuvLayout::Semi;
        // The reference pipeline down-casts 10-bit sources to I420 before
        // resizing, so scaled output is always planar.
        let out_semi = semi_in && img.bits_per_sample == 8 && scale.is_some();
        let (dw, dh) = match scale {
            Some(s) => (s.width as usize, s.height as usize),
            None => (sw, sh),
        };
        let (dwc, dhc) = ((dw + 1) / 2, (dh + 1) / 2);
        let yuv_out_size = dw * dh + 2 * dwc * dhc; // tight bytes

        let rgb = cfg.rgb;
        let rgb_size = rgb.map(|r| dw * dh * r.bytes() as usize).unwrap_or(0);

        // Arena layout, in words: [Y][Cb/Cr][Cr (planar)]
        // [scratch YUV (if scale)][rgb RGBA32 (if rgb)].
        let bpsb = img.bps_bytes();
        let (cw, chh) = ((sw + 1) / 2, (sh + 1) / 2);
        let mut off: u32 = 0;
        let y_off = off;
        off += sw as u32 * sh as u32;
        let cb_off = off;
        let cb_row_words = if semi_in { cw * 2 } else { cw };
        off += cb_row_words as u32 * chh as u32;
        let cr_off = if semi_in { cb_off + 1 } else { off };
        if !semi_in {
            off += cw as u32 * chh as u32;
        }
        let scratch_off = if scale.is_some() { off } else { 0 };
        let scratch_words = dw * dh + 2 * dwc * dhc;
        if scale.is_some() {
            off += scratch_words as u32;
        }
        // RGBA32 conversion output; the warp pass reads it when warping.
        let rgb_off = if rgb.is_some() { off } else { 0 };
        if rgb.is_some() {
            off += (dw * dh) as u32;
        }
        // Warped RGBA32 result (the readback target when warping).
        let warp_off = if warp.is_some() { off } else { 0 };
        if warp.is_some() {
            off += (dw * dh) as u32;
        }

        self.ensure_arena(off as usize * 4).map_err(ImageError::Pipeline)?;

        // Copy source planes (one word per sample; 10-bit down-cast inline).
        let wbase = self.arena_ptr as *mut u32;
        copy_plane_words(img.y, img.y_pitch, sw, sh, bpsb, unsafe { wbase.add(y_off as usize) });
        if semi_in {
            copy_plane_words(
                img.cb,
                img.cb_pitch,
                cw * 2,
                chh,
                bpsb,
                unsafe { wbase.add(cb_off as usize) },
            );
        } else {
            copy_plane_words(img.cb, img.cb_pitch, cw, chh, bpsb, unsafe {
                wbase.add(cb_off as usize)
            });
            let cr = img.cr.expect("planar source must carry a Cr plane");
            copy_plane_words(cr, img.cr_pitch, cw, chh, bpsb, unsafe {
                wbase.add(cr_off as usize)
            });
        }

        // Parameter blocks. The rgb pass reads the scratch (8-bit,
        // planar-style) when scaling happened, otherwise the input planes.
        let (csw, csh, cy_off, cy_pitch, ccb_off, ccb_pitch, ccr_off, ccr_pitch, csemi) =
            if scale.is_some() {
                (
                    dw as u32,
                    dh as u32,
                    scratch_off,
                    dw as u32,
                    scratch_off + (dw * dh) as u32,
                    dwc as u32,
                    scratch_off + (dw * dh + dwc * dhc) as u32,
                    dwc as u32,
                    0u32,
                )
            } else {
                (
                    sw as u32,
                    sh as u32,
                    y_off,
                    sw as u32,
                    cb_off,
                    cb_row_words as u32,
                    cr_off,
                    cw as u32,
                    semi_in as u32,
                )
            };
        let coeff = table(cfg.spec);
        let conv_params = ConvParams {
            src_w: csw,
            src_h: csh,
            y_off: cy_off,
            y_pitch: cy_pitch,
            cb_off: ccb_off,
            cb_pitch: ccb_pitch,
            cr_off: ccr_off,
            cr_pitch: ccr_pitch,
            semi: csemi,
            dst_off: rgb_off,
            dst_w: dw as u32,
            dst_h: dh as u32,
            ky: coeff.ky,
            r_cr: coeff.r_cr,
            r_off: coeff.r_off,
            g_cb: coeff.g_cb,
            g_cr: coeff.g_cr,
            g_off: coeff.g_off,
            b_cb: coeff.b_cb,
            b_off: coeff.b_off,
        };
        let resize_params = ResizeParams {
            src_w: sw as u32,
            src_h: sh as u32,
            y_off,
            y_pitch: sw as u32,
            cb_off,
            cb_pitch: cb_row_words as u32,
            cr_off,
            // Semi-planar Cr shares the interleaved CbCr rows.
            cr_pitch: if semi_in { cb_row_words as u32 } else { cw as u32 },
            chroma_stride: if semi_in { 2 } else { 1 },
            dst_off: scratch_off,
            dst_w: dw as u32,
            dst_h: dh as u32,
        };
        let warp_params = match warp {
            Some(w) => {
                // The shader samples the inverse map (matches the reference
                // pipeline's inverse mapping).
                let inv = w.transform.invert().ok_or_else(|| {
                    ImageError::Unsupported("singular affine matrix".into())
                })?;
                WarpParams {
                    src_w: dw as u32,
                    src_h: dh as u32,
                    src_off: rgb_off,
                    src_pitch: dw as u32,
                    dst_off: warp_off,
                    dst_w: dw as u32,
                    dst_h: dh as u32,
                    m00: inv.m00,
                    m01: inv.m01,
                    m02: inv.m02,
                    m10: inv.m10,
                    m11: inv.m11,
                    m12: inv.m12,
                    interp: match w.interpolation {
                        Interpolation::Nearest => 0,
                        Interpolation::Bilinear => 1,
                        Interpolation::Bicubic => 2,
                        Interpolation::Box => unreachable!("rejected above"),
                    },
                }
            }
            None => WarpParams {
                src_w: 0, src_h: 0, src_off: 0, src_pitch: 0,
                dst_off: 0, dst_w: 0, dst_h: 0,
                m00: 1.0, m01: 0.0, m02: 0.0,
                m10: 0.0, m11: 1.0, m12: 0.0,
                interp: 1,
            },
        };

        // Stage the parameter blocks in the uniform ring.
        unsafe {
            std::ptr::copy_nonoverlapping(
                params_bytes(&conv_params).as_ptr(),
                self.uniform_ptr,
                std::mem::size_of::<ConvParams>(),
            );
            std::ptr::copy_nonoverlapping(
                params_bytes(&resize_params).as_ptr(),
                self.uniform_ptr.add(UNIFORM_SLOT as usize),
                std::mem::size_of::<ResizeParams>(),
            );
            if warp.is_some() {
                std::ptr::copy_nonoverlapping(
                    params_bytes(&warp_params).as_ptr(),
                    self.uniform_ptr.add(2 * UNIFORM_SLOT as usize),
                    std::mem::size_of::<WarpParams>(),
                );
            }
        }

        let slot = self.cmd_slot;
        self.cmd_slot = (slot + 1) % CMD_SLOTS;
        let cb = self.cmd_buffers[slot];
        unsafe {
            self.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| ImageError::Pipeline(vk_err(e)))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| ImageError::Pipeline(vk_err(e)))?;
            self.device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| ImageError::Pipeline(vk_err(e)))?;

            // Passes run in pipeline order (resize -> yuv2rgb -> warp); a
            // full memory barrier orders every hand-off over the arena.
            let gx = (dw as u32).div_ceil(WORKGROUP);
            let gy = (dh as u32).div_ceil(WORKGROUP);
            let mut pass_recorded = false;
            if scale.is_some() {
                self.bind_pass(cb, self.resize_pipeline, 1);
                self.device.cmd_dispatch(cb, gx, gy, 1);
                pass_recorded = true;
            }
            if rgb.is_some() {
                if pass_recorded {
                    self.record_barrier(cb);
                }
                self.bind_pass(cb, self.conv_pipeline, 0);
                self.device.cmd_dispatch(cb, gx, gy, 1);
                pass_recorded = true;
            }
            if warp.is_some() {
                if pass_recorded {
                    self.record_barrier(cb);
                }
                self.bind_pass(cb, self.warp_pipeline, 2);
                self.device.cmd_dispatch(cb, gx, gy, 1);
            }

            self.device
                .end_command_buffer(cb)
                .map_err(|e| ImageError::Pipeline(vk_err(e)))?;
            let cbsub = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbsub);
            self.device
                .queue_submit(self.queue, &[submit], self.fence)
                .map_err(|e| ImageError::Pipeline(vk_err(e)))?;
            let w = self.device
                .wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS);
            w.map_err(|e| ImageError::Pipeline(vk_err(e)))?;
        }

        // Repack words into tight bytes.
        match rgb {
            None => {
                let words = self.arena_words();
                let scratch = &words[scratch_off as usize..scratch_off as usize + scratch_words];
                let y_words = &scratch[..dw * dh];
                let cb_words = &scratch[dw * dh..dw * dh + dwc * dhc];
                let cr_words = &scratch[dw * dh + dwc * dhc..];
                let mut data = vec![0u8; yuv_out_size];
                for (i, w) in y_words.iter().enumerate() {
                    data[i] = (*w & 0xFF) as u8;
                }
                if out_semi {
                    for k in 0..dwc * dhc {
                        data[dw * dh + 2 * k] = (cb_words[k] & 0xFF) as u8;
                        data[dw * dh + 2 * k + 1] = (cr_words[k] & 0xFF) as u8;
                    }
                } else {
                    for (i, w) in cb_words.iter().enumerate() {
                        data[dw * dh + i] = (*w & 0xFF) as u8;
                    }
                    for (i, w) in cr_words.iter().enumerate() {
                        data[dw * dh + dwc * dhc + i] = (*w & 0xFF) as u8;
                    }
                }
                Ok(ProcessedFrame::Yuv(vacc_image::YuvOutput {
                    width: dw as u32,
                    height: dh as u32,
                    layout: if out_semi { YuvLayout::Semi } else { YuvLayout::Planar },
                    data,
                }))
            }
            Some(ch) => {
                let words = self.arena_words();
                // Warping reads the conv output and writes a separate region;
                // read back whichever holds the final RGBA32 pixels.
                let read_off = if warp.is_some() { warp_off } else { rgb_off };
                let rgb_words = &words[read_off as usize..read_off as usize + dw * dh];
                let mut data = vec![0u8; rgb_size];
                if ch == RgbChannels::Rgba32 {
                    for (i, w) in rgb_words.iter().enumerate() {
                        data[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                    }
                } else {
                    for (i, w) in rgb_words.iter().enumerate() {
                        data[i * 3] = (*w & 0xFF) as u8;
                        data[i * 3 + 1] = ((*w >> 8) & 0xFF) as u8;
                        data[i * 3 + 2] = ((*w >> 16) & 0xFF) as u8;
                    }
                }
                Ok(ProcessedFrame::Rgb(vacc_image::RgbOutput {
                    width: dw as u32,
                    height: dh as u32,
                    channels: ch.bytes(),
                    data,
                }))
            }
        }
    }

    fn bind_pass(&self, cb: vk::CommandBuffer, pipeline: vk::Pipeline, set_idx: usize) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.desc_sets[set_idx]],
                &[],
            );
        }
    }

    /// Order shader writes before subsequent shader reads on the arena.
    fn record_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use vacc_image::{
        Affine, ColorRange, ImageConfig, Interpolation, MatrixCoefficients, RgbChannels, Scale,
        Warp, YuvImage,
    };

    /// Serialize GPU tests (shared device context).
    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: StdMutex<()> = StdMutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn require_gpu() -> Option<std::sync::MutexGuard<'static, ()>> {
        if !is_available() {
            return None;
        }
        Some(gpu_lock())
    }

    /// Deterministic gradient 4:2:0 source (owned tight buffer + static view),
    /// mirroring the NPP test vectors.
    fn grad_yuv(w: usize, h: usize, planar: bool) -> (Vec<u8>, YuvImage<'static>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as u32 * 255 / w.max(1) as u32) as u8)
            .collect();
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        let cb: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 400 % 256) as u8).collect();
        let cr: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 900 % 256) as u8).collect();
        if planar {
            let mut buf = Vec::new();
            buf.extend_from_slice(&y);
            buf.extend_from_slice(&cb);
            buf.extend_from_slice(&cr);
            let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
            let img = YuvImage::planar(
                &data[..w * h],
                w,
                &data[w * h..w * h + cw * chh],
                cw,
                &data[w * h + cw * chh..],
                cw,
                w,
                h,
                8,
            );
            (buf, img)
        } else {
            let mut uv = vec![0u8; cw * 2 * chh];
            for r in 0..chh {
                for x in 0..cw {
                    uv[r * cw * 2 + x * 2] = cb[r * cw + x];
                    uv[r * cw * 2 + x * 2 + 1] = cr[r * cw + x];
                }
            }
            let mut buf = Vec::new();
            buf.extend_from_slice(&y);
            buf.extend_from_slice(&uv);
            let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
            let img = YuvImage::semi(&data[..w * h], w, &data[w * h..], cw * 2, w, h, 8);
            (buf, img)
        }
    }

    fn diff(a: &[u8], b: &[u8]) -> (u32, f64) {
        assert_eq!(a.len(), b.len());
        let mut max = 0u32;
        let mut sum = 0.0f64;
        for i in 0..a.len() {
            let d = (a[i] as i32 - b[i] as i32).unsigned_abs() as f64;
            max = max.max(d as u32);
            sum += d;
        }
        (max, sum / a.len() as f64)
    }

    fn sw_process(img: &YuvImage, cfg: &ImageConfig) -> ProcessedFrame {
        vacc_image::process(img, cfg, vacc_image::Kernel::Auto).unwrap()
    }

    #[test]
    fn yuv2rgb_matches_sw() {
        let Some(_guard) = require_gpu() else { return };
        for planar in [false, true] {
            for (matrix, range) in [
                (MatrixCoefficients::Bt601, ColorRange::Limited),
                (MatrixCoefficients::Bt709, ColorRange::Limited),
                (MatrixCoefficients::Bt709, ColorRange::Full),
            ] {
                for ch in [RgbChannels::Rgb24, RgbChannels::Rgba32] {
                    let (_, img) = grad_yuv(320, 240, planar);
                    let cfg = ImageConfig {
                        rgb: Some(ch),
                        scale: None,
                        affine: None,
                        spec: vacc_image::ColorSpec { matrix, range },
                    };
                    let vk_res = process(&img, &cfg).unwrap();
                    let sw_res = sw_process(&img, &cfg);
                    match (vk_res, sw_res) {
                        (ProcessedFrame::Rgb(a), ProcessedFrame::Rgb(b)) => {
                            let (max, mean) = diff(&a.data, &b.data);
                            assert!(
                                max <= 1 && mean < 0.05,
                                "convert drift planar={planar} {matrix:?}/{range:?} {:?}: max={max} mean={mean}",
                                ch
                            );
                        }
                        _ => panic!("expected rgb outputs"),
                    }
                }
            }
        }
    }

    #[test]
    fn resize_matches_sw() {
        let Some(_guard) = require_gpu() else { return };
        for planar in [false, true] {
            let (_, img) = grad_yuv(320, 240, planar);
            for (tw, th) in [(160usize, 120), (256, 144), (400, 300)] {
                let cfg = ImageConfig {
                    rgb: None,
                    scale: Some(Scale::new(tw as u32, th as u32, Interpolation::Bilinear)),
                    affine: None,
                    spec: Default::default(),
                };
                let vk_res = process(&img, &cfg).unwrap();
                let sw_res = sw_process(&img, &cfg);
                match (vk_res, sw_res) {
                    (ProcessedFrame::Yuv(a), ProcessedFrame::Yuv(b)) => {
                        assert_eq!(a.layout, b.layout);
                        let (max, mean) = diff(&a.data, &b.data);
                        assert!(
                            max <= 4 && mean < 1.5,
                            "resize drift planar={planar} {tw}x{th}: max={max} mean={mean}"
                        );
                    }
                    _ => panic!("expected yuv outputs"),
                }
            }
        }
    }

    #[test]
    fn scale_and_rgb_matches_sw() {
        let Some(_guard) = require_gpu() else { return };
        for planar in [false, true] {
            let (_, img) = grad_yuv(320, 240, planar);
            let cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgba32),
                scale: Some(Scale::new(160, 120, Interpolation::Bilinear)),
                affine: None,
                spec: vacc_image::ColorSpec::auto(1080),
            };
            let vk_res = process(&img, &cfg).unwrap();
            let sw_res = sw_process(&img, &cfg);
            match (vk_res, sw_res) {
                (ProcessedFrame::Rgb(a), ProcessedFrame::Rgb(b)) => {
                    assert_eq!((a.width, a.height), (160, 120));
                    let (max, mean) = diff(&a.data, &b.data);
                    assert!(
                        max <= 4 && mean < 1.5,
                        "pipeline drift planar={planar}: max={max} mean={mean}"
                    );
                }
                _ => panic!("expected rgb outputs"),
            }
        }
    }

    #[test]
    fn unsupported_combinations_rejected() {
        let Some(_guard) = require_gpu() else { return };
        let (_, img) = grad_yuv(64, 32, true);
        // Box filter -> software fallback.
        let cfg = ImageConfig {
            rgb: None,
            scale: Some(Scale::new(32, 16, Interpolation::Box)),
            affine: None,
            spec: Default::default(),
        };
        assert!(matches!(process(&img, &cfg), Err(ImageError::Unsupported(_))));
        // Box warp -> software fallback.
        let cfg = ImageConfig {
            rgb: Some(RgbChannels::Rgba32),
            scale: None,
            affine: Some(Warp::new(Affine::identity(), Interpolation::Box)),
            spec: Default::default(),
        };
        assert!(matches!(process(&img, &cfg), Err(ImageError::Unsupported(_))));
        // Warp without RGB output -> software fallback.
        let cfg = ImageConfig {
            rgb: None,
            scale: None,
            affine: Some(Warp::bilinear(Affine::identity())),
            spec: Default::default(),
        };
        assert!(matches!(process(&img, &cfg), Err(ImageError::Unsupported(_))));
    }

    #[test]
    fn warp_matches_sw() {
        let Some(_guard) = require_gpu() else { return };
        let (w, h) = (160usize, 96);
        for planar in [false, true] {
            let (_, img) = grad_yuv(w, h, planar);
            let transforms: [(Affine, &str); 3] = [
                (Affine::identity(), "identity"),
                (Affine::translate(10.0, 5.0), "translate"),
                (
                    Affine::rotate_around(std::f32::consts::PI / 6.0, w as f32 / 2.0, h as f32 / 2.0),
                    "rotate",
                ),
            ];
            for interp in [
                Interpolation::Nearest,
                Interpolation::Bilinear,
                Interpolation::Bicubic,
            ] {
                for (t, label) in transforms {
                    for ch in [RgbChannels::Rgba32, RgbChannels::Rgb24] {
                        let cfg = ImageConfig {
                            rgb: Some(ch),
                            scale: None,
                            affine: Some(Warp::new(t, interp)),
                            spec: Default::default(),
                        };
                        let vk_res = process(&img, &cfg).unwrap();
                        let sw_res = sw_process(&img, &cfg);
                        match (vk_res, sw_res) {
                            (ProcessedFrame::Rgb(a), ProcessedFrame::Rgb(b)) => {
                                assert_eq!((a.width, a.height), (w as u32, h as u32));
                                let (max, mean) = diff(&a.data, &b.data);
                                assert!(
                                    max <= 4 && mean < 1.5,
                                    "warp drift planar={planar} {interp:?} {label} {:?}: max={max} mean={mean}",
                                    ch
                                );
                            }
                            _ => panic!("expected rgb outputs"),
                        }
                    }
                }
            }
            // Scale + warp: resize first, then warp at the scaled size.
            let cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgba32),
                scale: Some(Scale::new(80, 48, Interpolation::Bilinear)),
                affine: Some(Warp::bilinear(Affine::translate(4.0, 2.0))),
                spec: Default::default(),
            };
            let vk_res = process(&img, &cfg).unwrap();
            let sw_res = sw_process(&img, &cfg);
            match (vk_res, sw_res) {
                (ProcessedFrame::Rgb(a), ProcessedFrame::Rgb(b)) => {
                    assert_eq!((a.width, a.height), (80, 48));
                    let (max, mean) = diff(&a.data, &b.data);
                    assert!(
                        max <= 4 && mean < 1.5,
                        "scale+warp drift planar={planar}: max={max} mean={mean}"
                    );
                }
                _ => panic!("expected rgb outputs"),
            }
        }
    }
}
