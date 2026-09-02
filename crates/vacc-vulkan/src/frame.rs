//! Decoded frame representation.

/// YCbCr plane data.
#[derive(Debug, Clone)]
pub struct YCbCrPlane {
    pub data_ptr: u64,
    pub length: usize,
    pub row_pitch: usize,
    pub width: usize,
    pub height: usize,
}

/// Decoded frame with YCbCr output.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub frame_index: u32,
    pub coded_width: u32,
    pub coded_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub timestamp: i64,
    pub poc: i32,
    pub y_plane: YCbCrPlane,
    pub uv_plane: YCbCrPlane,
    pub skipped: bool,
    pub is_reference: bool,
    pub field_flags: u32,
}

impl DecodedFrame {
    pub fn new(
        frame_index: u32,
        coded_width: u32,
        coded_height: u32,
        display_width: u32,
        display_height: u32,
        y_plane: YCbCrPlane,
        uv_plane: YCbCrPlane,
    ) -> Self {
        Self {
            frame_index,
            coded_width,
            coded_height,
            display_width,
            display_height,
            timestamp: 0,
            poc: 0,
            y_plane,
            uv_plane,
            skipped: false,
            is_reference: true,
            field_flags: 0,
        }
    }

    pub fn skipped(frame_index: u32) -> Self {
        Self {
            frame_index,
            coded_width: 0,
            coded_height: 0,
            display_width: 0,
            display_height: 0,
            timestamp: 0,
            poc: 0,
            y_plane: YCbCrPlane {
                data_ptr: 0,
                length: 0,
                row_pitch: 0,
                width: 0,
                height: 0,
            },
            uv_plane: YCbCrPlane {
                data_ptr: 0,
                length: 0,
                row_pitch: 0,
                width: 0,
                height: 0,
            },
            skipped: true,
            is_reference: false,
            field_flags: 0,
        }
    }

    pub const fn chroma_divisor(&self) -> u32 {
        2
    }

    pub fn frame_size(&self) -> usize {
        let y_size = self.y_plane.row_pitch * self.coded_height as usize;
        let chroma_height = (self.coded_height as usize).div_ceil(2);
        let _chroma_width = (self.coded_width as usize).div_ceil(2);
        let uv_size = self.uv_plane.row_pitch * chroma_height;
        y_size + uv_size
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct YCbCrConversionParams {
    pub model_conversion: u32,
    pub range: u32,
    pub x_chroma_offset: u32,
    pub y_chroma_offset: u32,
    pub matrix_coefficients: u32,
    pub video_full_range: bool,
}

/// Decoded frame pool.
pub struct DecodedFramePool {
    frames: Vec<DecodedFrame>,
    available: Vec<usize>,
}

impl DecodedFramePool {
    pub fn new(capacity: usize) -> Self {
        let mut frames = Vec::with_capacity(capacity);
        let mut available = Vec::with_capacity(capacity);

        for i in 0..capacity {
            frames.push(DecodedFrame::skipped(i as u32));
            available.push(i);
        }

        Self { frames, available }
    }

    pub fn acquire(&mut self) -> Option<&mut DecodedFrame> {
        if let Some(index) = self.available.pop() {
            Some(&mut self.frames[index])
        } else {
            None
        }
    }

    pub fn release(&mut self, frame_index: u32) {
        if (frame_index as usize) < self.frames.len() {
            self.available.push(frame_index as usize);
        }
    }

    pub fn get(&self, index: usize) -> Option<&DecodedFrame> {
        self.frames.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut DecodedFrame> {
        self.frames.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn available_count(&self) -> usize {
        self.available.len()
    }
}
