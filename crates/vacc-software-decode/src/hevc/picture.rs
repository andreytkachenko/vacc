//! Port of `hevc/common/picture.{h,cpp}` — planar YUV picture buffer (AD-002).
//!
//! Only the buffer management used by the kernels is ported here. File I/O
//! (`write_yuv`) and the TMVP motion-info storage belong to the pipeline and
//! TMVP (inter prediction) layers respectively, and are ported later.

use crate::hevc::types::{sub_height_c, sub_width_c, ChromaFormat};

/// Picture buffer — planar YUV layout (AD-002), spec §6.1.
#[derive(Clone)]
pub struct Picture {
    /// Plane data: 0=Y, 1=Cb, 2=Cr (row-major, `stride[c]` samples per row).
    pub planes: [Vec<u16>; 3],
    /// Width per plane in samples.
    pub width: [i32; 3],
    /// Height per plane in samples.
    pub height: [i32; 3],
    /// Stride per plane in samples.
    pub stride: [i32; 3],

    /// Luma dimensions and bit depths.
    pub pic_width_in_luma: i32,
    pub pic_height_in_luma: i32,
    pub bit_depth_luma: i32,
    pub bit_depth_chroma: i32,
    pub chroma_format: ChromaFormat,

    /// Conformance window (in luma samples, spec §7.4.3.2.1).
    pub conf_win_left: i32,
    pub conf_win_right: i32,
    pub conf_win_top: i32,
    pub conf_win_bottom: i32,

    /// Picture Order Count.
    pub poc: i32,
    /// Coded Video Sequence ID (incremented at each IRAP with NoRaslOutputFlag).
    pub cvs_id: i32,

    /// Reference status.
    pub used_for_short_term_ref: bool,
    pub used_for_long_term_ref: bool,
    pub needed_for_output: bool,
}

impl Default for Picture {
    fn default() -> Self {
        Self {
            planes: [Vec::new(), Vec::new(), Vec::new()],
            width: [0; 3],
            height: [0; 3],
            stride: [0; 3],
            pic_width_in_luma: 0,
            pic_height_in_luma: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format: ChromaFormat::Yuv420,
            conf_win_left: 0,
            conf_win_right: 0,
            conf_win_top: 0,
            conf_win_bottom: 0,
            poc: 0,
            cvs_id: 0,
            used_for_short_term_ref: false,
            used_for_long_term_ref: false,
            needed_for_output: false,
        }
    }
}

impl Picture {
    /// Allocate planes based on dimensions and chroma format.
    pub fn allocate(&mut self, w: i32, h: i32, fmt: ChromaFormat, bd_luma: i32, bd_chroma: i32) {
        self.pic_width_in_luma = w;
        self.pic_height_in_luma = h;
        self.bit_depth_luma = bd_luma;
        self.bit_depth_chroma = bd_chroma;
        self.chroma_format = fmt;

        let sub_w = sub_width_c(fmt);
        let sub_h = sub_height_c(fmt);

        // Luma plane
        self.width[0] = w;
        self.height[0] = h;
        self.stride[0] = w;

        // Chroma planes
        if fmt == ChromaFormat::Monochrome {
            self.width[1] = 0;
            self.width[2] = 0;
            self.height[1] = 0;
            self.height[2] = 0;
            self.stride[1] = 0;
            self.stride[2] = 0;
        } else {
            let cw = w / sub_w as i32;
            let ch = h / sub_h as i32;
            self.width[1] = cw;
            self.width[2] = cw;
            self.height[1] = ch;
            self.height[2] = ch;
            self.stride[1] = cw;
            self.stride[2] = cw;
        }

        for c in 0..3 {
            if self.width[c] > 0 && self.height[c] > 0 {
                self.planes[c] = vec![0u16; (self.stride[c] * self.height[c]) as usize];
            } else {
                self.planes[c].clear();
            }
        }
    }

    /// Get sample at position (x, y) in plane c.
    #[inline]
    pub fn sample(&self, c: usize, x: i32, y: i32) -> u16 {
        self.planes[c][((y * self.stride[c]) + x) as usize]
    }

    /// Mutable access to sample at position (x, y) in plane c.
    #[inline]
    pub fn sample_mut(&mut self, c: usize, x: i32, y: i32) -> &mut u16 {
        &mut self.planes[c][((y * self.stride[c]) + x) as usize]
    }

    /// Is this picture a reference?
    #[inline]
    pub fn is_reference(&self) -> bool {
        self.used_for_short_term_ref || self.used_for_long_term_ref
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_420_dims() {
        let mut pic = Picture::default();
        pic.allocate(64, 32, ChromaFormat::Yuv420, 8, 8);
        assert_eq!(pic.width[0], 64);
        assert_eq!(pic.height[0], 32);
        assert_eq!(pic.stride[0], 64);
        assert_eq!(pic.width[1], 32);
        assert_eq!(pic.height[1], 16);
        assert_eq!(pic.stride[1], 32);
        assert_eq!(pic.width[2], 32);
        assert_eq!(pic.planes[0].len(), 64 * 32);
        assert_eq!(pic.planes[1].len(), 32 * 16);
    }

    #[test]
    fn allocate_monochrome_has_no_chroma() {
        let mut pic = Picture::default();
        pic.allocate(16, 16, ChromaFormat::Monochrome, 10, 10);
        assert_eq!(pic.width[1], 0);
        assert_eq!(pic.height[2], 0);
        assert!(pic.planes[1].is_empty());
    }

    #[test]
    fn sample_roundtrip() {
        let mut pic = Picture::default();
        pic.allocate(8, 4, ChromaFormat::Yuv420, 8, 8);
        *pic.sample_mut(0, 3, 2) = 512;
        assert_eq!(pic.sample(0, 3, 2), 512);
        assert!(!pic.is_reference());
        pic.used_for_short_term_ref = true;
        assert!(pic.is_reference());
    }
}
