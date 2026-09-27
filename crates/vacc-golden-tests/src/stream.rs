//! Golden pinning of decoded streams, generic over any `Decoder` backend.
//!
//! [`golden_stream`] drains a decoder (decode + flush) and pins, per display
//! frame, the SHA-256 of the canonical pixels (see [`crate::canonical`]) to
//! key `{sample}::f{index}`, plus the POC sequence (`{sample}::pocs`) and
//! the first frame's display size (`{sample}::size`). A missing/extra/reordered
//! frame or any pixel change changes the pinned hashes.
//!
//! Backends whose pixels are not in `DecodedFrame::pixel_data` (e.g. Vulkan,
//! which reads back through its own frame type) drive [`FrameSink`] directly
//! with their canonical bytes.

use std::error::Error;

use vacc_core::decoder::Decoder;
use vacc_core::frame::DecodedFrame;

use crate::canonical::canonical_pixels;
use crate::goldens;

/// Drain `d`: repeated `decode()` until exhausted, then `flush()`.
pub fn drain<D: Decoder>(d: &mut D) -> Vec<DecodedFrame>
where
    D::Error: Error + 'static,
{
    let mut frames = Vec::new();
    while let Some(f) = d.decode().expect("decode failed") {
        frames.push(f);
    }
    frames.extend(d.flush().expect("flush failed"));
    frames
}

/// Incremental per-frame pinning for backends that canonicalize pixels
/// themselves (e.g. Vulkan readback). Feed one `on_frame` per display frame,
/// then `finish()` to pin the POC sequence and size.
pub struct FrameSink<'t> {
    table: &'t [(&'t str, &'t str)],
    sample: String,
    pocs: Vec<i32>,
    size: Option<(u32, u32)>,
    n: usize,
}

impl<'t> FrameSink<'t> {
    pub fn new(sample: &str, table: &'t [(&'t str, &'t str)]) -> Self {
        Self {
            table,
            sample: sample.to_string(),
            pocs: Vec::new(),
            size: None,
            n: 0,
        }
    }

    /// Pin one display frame. `pixels` is the canonical planar Y+U+V form
    /// (see [`crate::canonical`]); `None` pins the literal `"none"` (skipped
    /// frame).
    pub fn on_frame(&mut self, poc: i32, width: u32, height: u32, pixels: Option<&[u8]>) {
        let i = self.n;
        if self.size.is_none() {
            self.size = Some((width, height));
        }
        self.pocs.push(poc);
        let key = format!("{}::f{i}", self.sample);
        match pixels {
            Some(px) => pin(self.table, &key, px),
            None => pin(self.table, &key, b"none"),
        }
        self.n += 1;
    }

    /// Pin the POC sequence and display size; returns the frame count.
    /// Panics if no frames were pinned.
    pub fn finish(mut self) -> usize {
        assert!(self.n > 0, "{}: no frames decoded", self.sample);
        let mut pocs_buf = Vec::new();
        for p in &self.pocs {
            goldens::push_i32(&mut pocs_buf, *p);
        }
        pin(self.table, &format!("{}::pocs", self.sample), &pocs_buf);
        if let Some((w, h)) = self.size.take() {
            pin(
                self.table,
                &format!("{}::size", self.sample),
                format!("{w}x{h}").as_bytes(),
            );
        }
        eprintln!("{}: {} frames pinned to goldens", self.sample, self.n);
        self.n
    }
}

/// Assert `data` against the golden for `key` and record it while a
/// regeneration run is collecting (see [`goldens::collecting`]).
fn pin(table: &[(&str, &str)], key: &str, data: &[u8]) {
    goldens::assert_data(table, key, data);
    goldens::record(key, data);
}

/// Decode `d` fully and pin every display frame (canonical pixels), the POC
/// sequence, and the display size to `table`. Frames without `pixel_data`
/// pin the literal `"none"`. Returns the frame count.
pub fn golden_stream<D: Decoder>(mut d: D, sample: &str, table: &[(&str, &str)]) -> usize
where
    D::Error: Error + 'static,
{
    let mut sink = FrameSink::new(sample, table);
    for f in drain(&mut d) {
        let pixels = f.pixel_data.as_ref().map(canonical_pixels);
        sink.on_frame(f.poc, f.width, f.height, pixels.as_deref());
    }
    sink.finish()
}
