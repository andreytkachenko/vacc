//! Vulkan video decoder implementing the Decoder trait.

use vacc_core::{
    codec::VideoCodec as CoreVideoCodec,
    decoder::{Decoder, DecoderInfo},
    frame::DecodedFrame,
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
};

use super::{Error, Result};

/// Vulkan video decoder implementing the Decoder trait.
pub struct VulkanDecoder {
    inner: vacc_vulkan::VideoDecoder,
}

impl VulkanDecoder {
    /// Create a new video decoder from bitstream data.
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let inner = vacc_vulkan::VideoDecoder::new(data, 64)
            .map_err(|e| Error::Vulkan(e))?;
        Ok(Self { inner })
    }

    /// Decode all frames from the bitstream.
    pub fn decode_all(&mut self, max_frames: usize) -> Result<Vec<vacc_vulkan::DecodedFrame>> {
        self.inner.decode_all(max_frames)
            .map_err(|e| Error::Vulkan(e))
    }

    /// Reorder frames from decoding order to presentation order (by POC).
    pub fn reorder_to_presentation(frames: Vec<vacc_vulkan::DecodedFrame>) -> Vec<vacc_vulkan::DecodedFrame> {
        vacc_vulkan::VideoDecoder::reorder_to_presentation(frames)
    }
}

impl Decoder for VulkanDecoder {
    type Error = Error;

    fn new(data: Vec<u8>) -> Result<Self> {
        Self::new(data)
    }

    fn new_with_format(
        _data: Vec<u8>,
        _codec: CoreVideoCodec,
        _format: &VideoFormat,
    ) -> Result<Self> {
        Err(Error::Vulkan(vacc_vulkan::VideoError::DecoderInit(
            "new_with_format not yet implemented".to_string()
        )))
    }

    fn info(&self) -> DecoderInfo {
        // Extract info from the inner decoder's decoded frames or parsed state
        // For now, return basic info
        DecoderInfo {
            backend: "vulkan".to_string(),
            codec: CoreVideoCodec::DecodeH264, // Would need to query from inner decoder
            coded_size: Extent2D::new(0, 0),
            display_size: Extent2D::new(0, 0),
            chroma_subsampling: ChromaSubsampling::_420,
            luma_bit_depth: ComponentBitDepth::Bit8,
            chroma_bit_depth: ComponentBitDepth::Bit8,
            profile_idc: None,
            dpb_slots: 0,
        }
    }

    fn submit(&mut self, _data: &[u8]) -> Result<()> {
        // The inner decoder works on full bitstream data, not incremental submit
        // This would require modifying the inner decoder's API
        Err(Error::Vulkan(vacc_vulkan::VideoError::InvalidState(
            "submit not supported - use decode_all instead".to_string()
        )))
    }

    fn decode(&mut self) -> Result<Option<DecodedFrame>> {
        // The inner decoder returns all frames at once via decode_all
        // This would require modifying the inner decoder's API for incremental decode
        Err(Error::Vulkan(vacc_vulkan::VideoError::InvalidState(
            "decode not supported - use decode_all instead".to_string()
        )))
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
