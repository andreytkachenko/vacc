//! VK_KHR_video_maintenance1 inline-queries structures.
//!
//! ash 0.38 exposes `VideoSessionCreateFlagsKHR::INLINE_QUERIES` but does NOT
//! generate `VkVideoInlineQueryInfoKHR`. Our sessions are created with the
//! INLINE_QUERIES flag (required as a workaround for
//! vkUpdateVideoSessionKHR being unresolvable on this NVIDIA driver), so every
//! `VkVideoDecodeInfoKHR` pNext chain includes one.
//!
//! CRITICAL: the layout must match the C struct exactly — there is NO padding
//! between fields, and `queryPool` is a 64-bit handle at offset 16. A u32
//! field at that offset leaks uninitialized stack padding into the pool
//! handle, which the NVIDIA driver rejects with an intentional trap (0x168).
//!
//! queryPool=VK_NULL_HANDLE + queryCount=0 means "no queries" (legal: all
//! VkVideoInlineQueryInfoKHR VUIDs are conditional on a non-null pool).

/// VK_STRUCTURE_TYPE_VIDEO_INLINE_QUERY_INFO_KHR
pub const VIDEO_INLINE_QUERY_INFO_KHR: u32 = 1000515001;

/// Exact C layout of VkVideoInlineQueryInfoKHR (ash 0.38 lacks it):
/// sType@0, pNext@8, queryPool@16, firstQuery@24, queryCount@28 (32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VideoInlineQueryInfoKHR {
    pub s_type: u32,
    pub p_next: *const std::ffi::c_void,
    pub query_pool: u64,
    pub first_query: u32,
    pub query_count: u32,
}

/// Build an inline-queries structure with no queries enabled, chained in front
/// of `tail` (the rest of the `VkVideoDecodeInfoKHR` pNext chain).
pub fn empty_inline_queries(tail: *const std::ffi::c_void) -> VideoInlineQueryInfoKHR {
    VideoInlineQueryInfoKHR {
        s_type: VIDEO_INLINE_QUERY_INFO_KHR,
        p_next: tail,
        query_pool: 0,
        first_query: 0,
        query_count: 0,
    }
}
