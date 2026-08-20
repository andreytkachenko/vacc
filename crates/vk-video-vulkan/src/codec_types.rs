//! Re-export StdVideo* types from ash::vk::native for convenience.
//!
//! ash 0.38 provides these via bindgen from the Vulkan headers in the
//! `ash::vk::native` submodule. We re-export them here so the rest of the
//! crate uses a single import path.

pub use ash::vk::native::StdVideoDecodeH264PictureInfo;
pub use ash::vk::native::StdVideoDecodeH264PictureInfoFlags;
pub use ash::vk::native::StdVideoH264PictureParameterSet;
pub use ash::vk::native::StdVideoH264PpsFlags;
pub use ash::vk::native::StdVideoH264SequenceParameterSet;
pub use ash::vk::native::StdVideoH264SpsFlags;

pub use ash::vk::native::StdVideoDecodeH265PictureInfo;
pub use ash::vk::native::StdVideoDecodeH265PictureInfoFlags;
pub use ash::vk::native::StdVideoH265PictureParameterSet;
pub use ash::vk::native::StdVideoH265SequenceParameterSet;
pub use ash::vk::native::StdVideoH265VideoParameterSet;

pub use ash::vk::native::StdVideoAV1ColorConfig;
pub use ash::vk::native::StdVideoAV1SequenceHeader;
pub use ash::vk::native::StdVideoAV1TimingInfo;
pub use ash::vk::native::StdVideoDecodeAV1PictureInfo;
pub use ash::vk::native::StdVideoDecodeAV1PictureInfoFlags;
// Note: StdVideoAV1FilmGrainParams does not exist in ash::vk::native.
// Film grain is represented via StdVideoAV1FilmGrain instead.
