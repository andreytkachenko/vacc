//! DpbManager integration tests for nvdec-decode.
//!
//! These tests verify that DpbManager from vacc-vulkan works correctly
//! when used by nvdec-decode for reference frame tracking during decode.

use vacc_vulkan::{DpbManager, H264MmcoCommand};

// ============================================================================
// Helper: Create a DpbManager with configured slots and populate entries
// ============================================================================

fn create_dpb_with_entries(max_slots: u32, entries: &[(u32, u32, [i32; 2])]) -> DpbManager {
    let mut dpb = DpbManager::new(max_slots);
    for (slot, frame_num, poc) in entries {
        dpb.entries[*slot as usize].frame_num = *frame_num;
        dpb.entries[*slot as usize].pic_order_cnt = *poc;
        dpb.entries[*slot as usize].is_valid = true;
    }
    dpb
}

fn count_valid_entries(dpb: &DpbManager) -> usize {
    dpb.entries.iter().filter(|e| e.is_valid).count()
}

// ============================================================================
// 1. test_dpb_manager_new_initialization
// ============================================================================

#[test]
fn test_dpb_manager_new_initialization() {
    // Verify new DpbManager has all entries invalid
    let dpb = DpbManager::new(8);

    assert_eq!(dpb.entries.len(), 8);
    assert_eq!(dpb.max_num_ref_frames(), 16); // default
    assert_eq!(dpb.max_frame_num(), 64); // default
    assert_eq!(dpb.next_slot, 0);

    for i in 0..8 {
        assert!(
            !dpb.entries[i].is_valid,
            "Entry {} should be invalid on creation",
            i
        );
        assert_eq!(dpb.entries[i].frame_num, 0);
        assert_eq!(dpb.entries[i].pic_order_cnt, [0, 0]);
        assert_eq!(dpb.entries[i].slot_index, i as u32);
    }
}

// ============================================================================
// 2. test_dpb_manager_set_max_frame_num
// ============================================================================

#[test]
fn test_dpb_manager_set_max_frame_num() {
    let mut dpb = DpbManager::new(8);

    // Default is 64
    assert_eq!(dpb.max_frame_num(), 64);

    // Set to common values based on log2_max_frame_num_minus4
    dpb.set_max_frame_num(128);
    assert_eq!(dpb.max_frame_num(), 128);

    dpb.set_max_frame_num(256);
    assert_eq!(dpb.max_frame_num(), 256);

    dpb.set_max_frame_num(65536);
    assert_eq!(dpb.max_frame_num(), 65536);
}

// ============================================================================
// 3. test_dpb_manager_set_max_num_ref_frames
// ============================================================================

#[test]
fn test_dpb_manager_set_max_num_ref_frames() {
    let mut dpb = DpbManager::new(8);

    // Default is 16
    assert_eq!(dpb.max_num_ref_frames(), 16);

    dpb.set_max_num_ref_frames(4);
    assert_eq!(dpb.max_num_ref_frames(), 4);

    dpb.set_max_num_ref_frames(16);
    assert_eq!(dpb.max_num_ref_frames(), 16);

    dpb.set_max_num_ref_frames(32);
    assert_eq!(dpb.max_num_ref_frames(), 32);
}

// ============================================================================
// 4. test_dpb_manager_find_or_recycle_slot_empty
// ============================================================================

#[test]
fn test_dpb_manager_find_or_recycle_slot_empty() {
    // Find slot when all empty
    let mut dpb = DpbManager::new(4);

    // All slots invalid, should return slot 0
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(0));

    // Mark slot 0 as valid
    dpb.entries[0].is_valid = true;

    // Should return slot 1 (first empty)
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(1));

    // Mark slots 0,1,3 as valid
    dpb.entries[1].is_valid = true;
    dpb.entries[3].is_valid = true;

    // Should return slot 2 (first empty)
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(2));
}

// ============================================================================
// 5. test_dpb_manager_find_or_recycle_slot_with_valid
// ============================================================================

#[test]
fn test_dpb_manager_find_or_recycle_slot_with_valid() {
    // Find empty slot among valid ones
    let mut dpb = DpbManager::new(6);

    // Set up valid entries at slots 0, 2, 4
    dpb.entries[0].is_valid = true;
    dpb.entries[0].pic_order_cnt = [10, 10];
    dpb.entries[2].is_valid = true;
    dpb.entries[2].pic_order_cnt = [20, 20];
    dpb.entries[4].is_valid = true;
    dpb.entries[4].pic_order_cnt = [30, 30];

    // Should find slot 1 first (lowest index empty)
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(1));

    // Mark slot 1 valid, should find slot 3
    dpb.entries[1].is_valid = true;
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(3));

    // Mark slot 3 valid, should find slot 5
    dpb.entries[3].is_valid = true;
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(5));
}

// ============================================================================
// 6. test_dpb_manager_find_or_recycle_slot_recycle
// ============================================================================

#[test]
fn test_dpb_manager_find_or_recycle_slot_recycle() {
    // Recycle oldest slot when full
    let mut dpb = create_dpb_with_entries(
        4,
        &[
            (0, 1, [10, 10]), // oldest POC
            (1, 2, [20, 20]),
            (2, 3, [30, 30]),
            (3, 4, [40, 40]), // newest POC
        ],
    );

    // All slots valid, should recycle slot with lowest POC (slot 0, POC=10)
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(0));

    // Slot 0 should now be invalid
    assert!(!dpb.entries[0].is_valid);
    assert_eq!(count_valid_entries(&dpb), 3);

    // Next call finds slot 0 as empty (already invalidated)
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(0));

    // To test recycling slot 1, restore slot 0 and recycle again
    dpb.entries[0].is_valid = true;
    dpb.entries[0].pic_order_cnt = [5, 5]; // POC=5, now oldest
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(0)); // slot 0 has POC=5, still oldest

    // Restore slot 0 with higher POC, now slot 1 is oldest
    dpb.entries[0].is_valid = true;
    dpb.entries[0].pic_order_cnt = [25, 25];
    let slot = dpb.find_or_recycle_slot(&[]);
    assert_eq!(slot, Some(1)); // slot 1 has POC=20, now oldest
    assert!(!dpb.entries[1].is_valid);
}

// ============================================================================
// 7. test_dpb_manager_find_or_recycle_slot_protected
// ============================================================================

#[test]
fn test_dpb_manager_find_or_recycle_slot_protected() {
    // Don't recycle protected POCs
    let mut dpb = create_dpb_with_entries(
        4,
        &[
            (0, 1, [10, 10]),
            (1, 2, [20, 20]),
            (2, 3, [30, 30]),
            (3, 4, [40, 40]),
        ],
    );

    // Protect POCs 10 and 30 - these cannot be recycled
    let protected = vec![10, 30];

    // Should recycle POC 20 (slot 1), not POC 10 (protected)
    let slot = dpb.find_or_recycle_slot(&protected);
    assert_eq!(slot, Some(1));
    assert!(!dpb.entries[1].is_valid);
    assert!(dpb.entries[0].is_valid); // POC 10 still valid
    assert!(dpb.entries[2].is_valid); // POC 30 still valid

    // Next call finds slot 1 as empty
    let slot = dpb.find_or_recycle_slot(&protected);
    assert_eq!(slot, Some(1));

    // Restore slot 1 with POC=15, protect only POC 30
    dpb.entries[1].is_valid = true;
    dpb.entries[1].pic_order_cnt = [15, 15];
    let protected = vec![30];

    // Should recycle POC 10 (slot 0) since it's no longer protected and oldest
    let slot = dpb.find_or_recycle_slot(&protected);
    assert_eq!(slot, Some(0));
    assert!(!dpb.entries[0].is_valid);
    assert!(dpb.entries[1].is_valid); // POC 15 still valid
    assert!(dpb.entries[2].is_valid); // POC 30 still valid
}

// ============================================================================
// 8. test_dpb_manager_apply_sliding_window
// ============================================================================

#[test]
fn test_dpb_manager_apply_sliding_window() {
    // Sliding window removes oldest ref when exceeding max_num_ref_frames
    let mut dpb = create_dpb_with_entries(
        8,
        &[
            (0, 1, [10, 10]),
            (1, 2, [20, 20]),
            (2, 3, [30, 30]),
            (3, 4, [40, 40]),
        ],
    );

    dpb.set_max_num_ref_frames(3);

    // Current frame_num=5, 4 refs exist, max is 3
    // Should remove oldest ref (frame_num=1, POC=10)
    dpb.apply_sliding_window(5);

    assert!(!dpb.entries[0].is_valid); // frame_num=1 removed
    assert!(dpb.entries[1].is_valid); // frame_num=2 kept
    assert!(dpb.entries[2].is_valid); // frame_num=3 kept
    assert!(dpb.entries[3].is_valid); // frame_num=4 kept
    assert_eq!(count_valid_entries(&dpb), 3);

    // Add another ref and apply again
    dpb.entries[0].is_valid = true;
    dpb.entries[0].frame_num = 6;
    dpb.entries[0].pic_order_cnt = [50, 50];

    // Now refs are: 2,3,4,6 (frame_num). Oldest is 2.
    // Current is 7, should remove frame_num=2
    dpb.apply_sliding_window(7);
    assert!(!dpb.entries[1].is_valid); // frame_num=2 removed
    assert_eq!(count_valid_entries(&dpb), 3);
}

// ============================================================================
// 9. test_dpb_manager_apply_sliding_window_no_wrap
// ============================================================================

#[test]
fn test_dpb_manager_apply_sliding_window_noop_when_under_limit() {
    // Sliding window should NOT remove refs when under limit
    let mut dpb = create_dpb_with_entries(8, &[(0, 1, [10, 10]), (1, 2, [20, 20])]);

    dpb.set_max_num_ref_frames(4);

    // Only 2 refs, max is 4, nothing should be removed
    dpb.apply_sliding_window(3);

    assert!(dpb.entries[0].is_valid);
    assert!(dpb.entries[1].is_valid);
    assert_eq!(count_valid_entries(&dpb), 2);
}

// ============================================================================
// 10. test_dpb_manager_apply_mmco_unmark_short_term
// ============================================================================

#[test]
fn test_dpb_manager_apply_mmco_unmark_short_term() {
    // MMCO 1: Mark specific short-term reference as unused
    let mut dpb = create_dpb_with_entries(
        8,
        &[
            (0, 5, [10, 10]),
            (1, 10, [20, 20]),
            (2, 15, [30, 30]),
            (3, 20, [40, 40]),
        ],
    );

    // Current frame_num=25
    // MMCO 1 with difference_of_pic_nums_minus1=4
    // picNumX = 25 - (4+1) = 20
    // Should invalidate frame_num=20 (slot 3)
    let mmco = vec![H264MmcoCommand::UnmarkShortTerm {
        difference_of_pic_nums_minus1: 4,
    }];
    dpb.apply_mmco(25, 4, &mmco);

    assert!(dpb.entries[0].is_valid);
    assert!(dpb.entries[1].is_valid);
    assert!(dpb.entries[2].is_valid);
    assert!(!dpb.entries[3].is_valid); // frame_num=20 invalidated
}

// ============================================================================
// 11. test_dpb_manager_apply_mmco_unmark_all
// ============================================================================

#[test]
fn test_dpb_manager_apply_mmco_unmark_all() {
    // MMCO 5: Invalidate all references
    let mut dpb =
        create_dpb_with_entries(8, &[(0, 5, [10, 10]), (1, 10, [20, 20]), (2, 15, [30, 30])]);

    let mmco = vec![H264MmcoCommand::UnmarkAll];
    dpb.apply_mmco(20, 3, &mmco);

    for i in 0..8 {
        assert!(
            !dpb.entries[i].is_valid,
            "Entry {} should be invalid after MMCO UnmarkAll",
            i
        );
    }
}

// ============================================================================
// 12. test_dpb_manager_apply_mmco_multiple_commands
// ============================================================================

#[test]
fn test_dpb_manager_apply_mmco_multiple_commands() {
    // Multiple MMCO commands in sequence
    let mut dpb = create_dpb_with_entries(
        8,
        &[
            (0, 5, [10, 10]),
            (1, 10, [20, 20]),
            (2, 15, [30, 30]),
            (3, 20, [40, 40]),
        ],
    );

    // Current frame_num=25
    // MMCO 1: unmark picNumX = 25 - (0+1) = 24 (no match)
    // MMCO 1: unmark picNumX = 25 - (5+1) = 19 (no match)
    // MMCO 1: unmark picNumX = 25 - (10+1) = 14 (no match)
    // MMCO 1: unmark picNumX = 25 - (15+1) = 9 (no match)
    // MMCO 1: unmark picNumX = 25 - (20+1) = 4 (no match)
    // Actually let's use exact matches:
    // MMCO 1: unmark picNumX = 25 - (5+1) = 19 -> no match
    // Let's use: difference_of_pic_nums_minus1=5 -> picNumX=19 (no)
    // difference_of_pic_nums_minus1=10 -> picNumX=14 (no)
    // difference_of_pic_nums_minus1=15 -> picNumX=9 (no)
    // difference_of_pic_nums_minus1=20 -> picNumX=4 (no)
    //
    // Let me recalculate:
    // frame_num=5: diff=20 -> picNumX=25-21=4 (no)
    // frame_num=10: diff=15 -> picNumX=25-16=9 (no)
    // frame_num=15: diff=10 -> picNumX=25-11=14 (no)
    // frame_num=20: diff=5 -> picNumX=25-6=19 (no)
    //
    // Hmm, I need to use the right differences:
    // For frame_num=20: picNumX=20, so diff = 25-20-1 = 4
    // For frame_num=15: picNumX=15, so diff = 25-15-1 = 9
    // For frame_num=10: picNumX=10, so diff = 25-10-1 = 14
    // For frame_num=5: picNumX=5, so diff = 25-5-1 = 19

    let mmco = vec![
        H264MmcoCommand::UnmarkShortTerm {
            difference_of_pic_nums_minus1: 4,
        }, // unmark 20
        H264MmcoCommand::UnmarkShortTerm {
            difference_of_pic_nums_minus1: 9,
        }, // unmark 15
    ];
    dpb.apply_mmco(25, 4, &mmco);

    assert!(dpb.entries[0].is_valid); // frame_num=5 still valid
    assert!(dpb.entries[1].is_valid); // frame_num=10 still valid
    assert!(!dpb.entries[2].is_valid); // frame_num=15 invalidated
    assert!(!dpb.entries[3].is_valid); // frame_num=20 invalidated
}

// ============================================================================
// 13. test_dpb_manager_invalidate_all
// ============================================================================

#[test]
fn test_dpb_manager_invalidate_all() {
    // IDR clears all entries
    let mut dpb = create_dpb_with_entries(
        8,
        &[
            (0, 1, [10, 10]),
            (1, 2, [20, 20]),
            (2, 3, [30, 30]),
            (3, 4, [40, 40]),
        ],
    );

    assert_eq!(count_valid_entries(&dpb), 4);

    dpb.invalidate_all();

    assert_eq!(count_valid_entries(&dpb), 0);

    // Also verify layout and access reset
    for i in 0..8 {
        assert_eq!(
            dpb.entries[i].current_layout,
            ash::vk::ImageLayout::UNDEFINED
        );
        assert_eq!(
            dpb.entries[i].last_access,
            vacc_vulkan::LastAccessType::None
        );
    }
}

// ============================================================================
// 14. test_dpb_manager_invalidate_slot
// ============================================================================

#[test]
fn test_dpb_manager_invalidate_slot() {
    // Single slot invalidation
    let mut dpb =
        create_dpb_with_entries(8, &[(0, 1, [10, 10]), (1, 2, [20, 20]), (2, 3, [30, 30])]);

    dpb.entries[1].current_layout = ash::vk::ImageLayout::GENERAL;
    dpb.entries[1].last_access = vacc_vulkan::LastAccessType::DecodeWrite;

    dpb.invalidate_slot(1);

    assert!(dpb.entries[0].is_valid);
    assert!(!dpb.entries[1].is_valid);
    assert!(dpb.entries[2].is_valid);

    // Verify access reset. `current_layout` is intentionally preserved on
    // single-slot invalidation (the image's Vulkan layout is physical state;
    // see DpbManager::invalidate_slot docs).
    assert_eq!(dpb.entries[1].current_layout, ash::vk::ImageLayout::GENERAL);
    assert_eq!(
        dpb.entries[1].last_access,
        vacc_vulkan::LastAccessType::None
    );

    // Out of bounds should be safe
    dpb.invalidate_slot(100); // should not panic
}

// ============================================================================
// 15. test_dpb_manager_frame_num_wrap_via_sliding_window
// ============================================================================

#[test]
fn test_dpb_manager_frame_num_wrap_via_sliding_window() {
    // Wraparound-aware frame_num comparison via sliding window
    let mut dpb = DpbManager::new(8);
    dpb.set_max_frame_num(64);
    dpb.set_max_num_ref_frames(2);

    // Simulate wraparound: frames near the end of cycle and beginning of next
    // frame_num=60,61,62 are from current cycle
    // frame_num=2,3 are from next cycle (wrapped)
    let entries = vec![
        (0, 60, [10, 10]), // oldest in wrap terms
        (1, 61, [20, 20]),
        (2, 62, [30, 30]),
        (3, 2, [40, 40]), // wrapped, but newer
    ];

    for (slot, frame_num, poc) in &entries {
        dpb.entries[*slot as usize].frame_num = *frame_num;
        dpb.entries[*slot as usize].pic_order_cnt = *poc;
        dpb.entries[*slot as usize].is_valid = true;
    }

    // Current frame_num=4 (wrapped cycle)
    // With wrap logic: frame_num=60 wraps to 60-64=-4, frame_num=61 wraps to -3, etc.
    // frame_num=2 wraps to 2 (no wrap needed since 2 < 4)
    // Oldest wrapped is frame_num=60 (-4), should be removed
    dpb.apply_sliding_window(4);

    // frame_num=60 should be removed (oldest in wrap terms)
    assert!(!dpb.entries[0].is_valid);
    assert_eq!(count_valid_entries(&dpb), 3);

    // Apply again - now frame_num=61 is oldest
    dpb.apply_sliding_window(4);
    assert!(!dpb.entries[1].is_valid);
    assert_eq!(count_valid_entries(&dpb), 2);
}

// ============================================================================
// 16. test_dpb_manager_compute_pic_num_via_mmco
// ============================================================================

#[test]
fn test_dpb_manager_compute_pic_num_via_mmco() {
    // Verify picNumX computation with wraparound via MMCO
    let mut dpb = DpbManager::new(8);
    dpb.set_max_frame_num(64);

    // Current frame_num=5 (wrapped cycle)
    // Frame at frame_num=60 is from previous cycle
    // picNumX = current - (diff+1) with wraparound
    // For frame_num=60: we need picNumX=60
    // If current=5, diff+1 = (64+5) - 60 = 9, so diff=8
    dpb.entries[0].frame_num = 60;
    dpb.entries[0].pic_order_cnt = [10, 10];
    dpb.entries[0].is_valid = true;

    // MMCO 1 with difference_of_pic_nums_minus1=8
    // picNumX = (64 + 5) - 9 = 60
    let mmco = vec![H264MmcoCommand::UnmarkShortTerm {
        difference_of_pic_nums_minus1: 8,
    }];
    dpb.apply_mmco(5, 1, &mmco);

    assert!(!dpb.entries[0].is_valid); // frame_num=60 should be invalidated
}

// ============================================================================
// 17. test_dpb_manager_full_decode_flow
// ============================================================================

#[test]
fn test_dpb_manager_full_decode_flow() {
    // Simulate full decode: allocate → add ref → MMCO → recycle
    let mut dpb = DpbManager::new(4);
    dpb.set_max_frame_num(64);
    dpb.set_max_num_ref_frames(3);

    // Frame 0: IDR, invalidate all, allocate slot
    dpb.invalidate_all();
    let slot = dpb.find_or_recycle_slot(&[]).unwrap();
    assert_eq!(slot, 0);
    dpb.entries[0].frame_num = 0;
    dpb.entries[0].pic_order_cnt = [0, 0];
    dpb.entries[0].is_valid = true;
    assert_eq!(count_valid_entries(&dpb), 1);

    // Frame 1: P-frame, uses frame 0 as ref (protected)
    let protected = vec![0]; // POC of frame 0
    let slot = dpb.find_or_recycle_slot(&protected).unwrap();
    dpb.entries[slot as usize].frame_num = 1;
    dpb.entries[slot as usize].pic_order_cnt = [1, 1];
    dpb.entries[slot as usize].is_valid = true;
    assert_eq!(count_valid_entries(&dpb), 2);

    // Frame 2: P-frame, uses frames 0 and 1 as refs
    let protected = vec![0, 1];
    let slot = dpb.find_or_recycle_slot(&protected).unwrap();
    dpb.entries[slot as usize].frame_num = 2;
    dpb.entries[slot as usize].pic_order_cnt = [2, 2];
    dpb.entries[slot as usize].is_valid = true;
    assert_eq!(count_valid_entries(&dpb), 3);

    // Frame 3: P-frame with MMCO to drop frame 0
    // picNumX = 3 - (3+1) = -1? No: frame_num=0, diff = 3-0-1 = 2
    let protected = vec![1, 2]; // frames 1 and 2 are refs
    let mmco = vec![H264MmcoCommand::UnmarkShortTerm {
        difference_of_pic_nums_minus1: 2,
    }]; // unmark frame_num=0
    dpb.apply_mmco(3, 3, &mmco);

    // MMCO invalidated frame_num=0 (slot 0), so find_or_recycle_slot returns slot 0
    let slot = dpb.find_or_recycle_slot(&protected).unwrap();
    assert_eq!(slot, 0); // slot 0 was invalidated by MMCO
    dpb.entries[slot as usize].frame_num = 3;
    dpb.entries[slot as usize].pic_order_cnt = [3, 3];
    dpb.entries[slot as usize].is_valid = true;

    // Verify no entry has frame_num=0 anymore (MMCO worked)
    assert!(dpb.find_by_frame_num(0).is_none());
    assert!(dpb.entries[1].is_valid); // frame_num=1
    assert!(dpb.entries[2].is_valid); // frame_num=2
    assert!(dpb.entries[0].is_valid); // frame_num=3 (reused slot 0)
    assert_eq!(count_valid_entries(&dpb), 3);

    // Frame 4: Fill remaining empty slot (slot 3)
    let protected = vec![1, 2, 3]; // frames 1, 2, 3 are refs
    let slot = dpb.find_or_recycle_slot(&protected).unwrap();
    assert_eq!(slot, 3); // slot 3 was empty
    dpb.entries[slot as usize].frame_num = 4;
    dpb.entries[slot as usize].pic_order_cnt = [4, 4];
    dpb.entries[slot as usize].is_valid = true;
    assert_eq!(count_valid_entries(&dpb), 4);

    // Frame 5: All slots full, protected=[3,4], should recycle frame 1 (POC=1)
    let protected = vec![3, 4];
    let slot = dpb.find_or_recycle_slot(&protected);
    assert_eq!(slot, Some(1)); // frame_num=1, POC=1 should be recycled (oldest unprotected)

    dpb.entries[1].frame_num = 5;
    dpb.entries[1].pic_order_cnt = [5, 5];
    dpb.entries[1].is_valid = true;

    assert!(dpb.entries[0].is_valid); // frame_num=3
    assert!(dpb.entries[1].is_valid); // frame_num=5
    assert!(dpb.entries[2].is_valid); // frame_num=2
    assert!(dpb.entries[3].is_valid); // frame_num=4
    assert_eq!(count_valid_entries(&dpb), 4);
}

// ============================================================================
// 18. test_dpb_manager_get_references
// ============================================================================

#[test]
fn test_dpb_manager_get_references() {
    let dpb = create_dpb_with_entries(8, &[(0, 1, [10, 10]), (1, 2, [20, 20]), (3, 4, [40, 40])]);

    let refs = dpb.get_references();
    assert_eq!(refs.len(), 3);
    assert_eq!(refs[0].frame_num, 1);
    assert_eq!(refs[1].frame_num, 2);
    assert_eq!(refs[2].frame_num, 4);
}

// ============================================================================
// 19. test_dpb_manager_find_by_frame_num
// ============================================================================

#[test]
fn test_dpb_manager_find_by_frame_num() {
    let dpb = create_dpb_with_entries(
        8,
        &[(0, 1, [10, 10]), (1, 5, [50, 50]), (2, 10, [100, 100])],
    );

    // Find existing frame
    let (idx, entry) = dpb.find_by_frame_num(5).unwrap();
    assert_eq!(idx, 1);
    assert_eq!(entry.frame_num, 5);
    assert_eq!(entry.pic_order_cnt, [50, 50]);

    // Find non-existing frame
    assert!(dpb.find_by_frame_num(99).is_none());

    // Find invalid frame (should not match)
    let mut dpb = dpb;
    dpb.entries[0].is_valid = false;
    assert!(dpb.find_by_frame_num(1).is_none());
}

// ============================================================================
// 20. test_dpb_manager_layout_tracking
// ============================================================================

#[test]
fn test_dpb_manager_layout_tracking() {
    let mut dpb = DpbManager::new(4);

    // Set layout for slot 0
    dpb.set_slot_layout(0, ash::vk::ImageLayout::GENERAL);
    assert_eq!(dpb.get_slot_layout(0), ash::vk::ImageLayout::GENERAL);

    // Set layout for slot 1
    dpb.set_slot_layout(1, ash::vk::ImageLayout::PRESENT_SRC_KHR);
    assert_eq!(
        dpb.get_slot_layout(1),
        ash::vk::ImageLayout::PRESENT_SRC_KHR
    );

    // Slot 2 should be UNDEFINED by default
    assert_eq!(dpb.get_slot_layout(2), ash::vk::ImageLayout::UNDEFINED);
}

// ============================================================================
// 21. test_dpb_manager_last_access_tracking
// ============================================================================

#[test]
fn test_dpb_manager_last_access_tracking() {
    let mut dpb = DpbManager::new(4);

    // Set last access for slot 0
    dpb.set_slot_last_access(0, vacc_vulkan::LastAccessType::DecodeWrite);
    assert_eq!(
        dpb.get_slot_last_access(0),
        vacc_vulkan::LastAccessType::DecodeWrite
    );

    // Set last access for slot 1
    dpb.set_slot_last_access(1, vacc_vulkan::LastAccessType::TransferRead);
    assert_eq!(
        dpb.get_slot_last_access(1),
        vacc_vulkan::LastAccessType::TransferRead
    );

    // Slot 2 should be None by default
    assert_eq!(
        dpb.get_slot_last_access(2),
        vacc_vulkan::LastAccessType::None
    );
}

// ============================================================================
// 22. test_dpb_manager_register_frame
// ============================================================================

#[test]
fn test_dpb_manager_register_frame() {
    let mut dpb = DpbManager::new(4);

    // Register frame at slot 0
    dpb.register_frame(0, 100);
    assert!(dpb.entries[0].is_valid);
    assert_eq!(dpb.entries[0].frame_num, 100);
    assert_eq!(dpb.entries[0].slot_index, 0);

    // Register frame at slot 2
    dpb.register_frame(2, 101);
    assert!(dpb.entries[2].is_valid);
    assert_eq!(dpb.entries[2].frame_num, 101);
}

// ============================================================================
// 23. test_dpb_manager_find_or_recycle_slot_excluding
// ============================================================================

#[test]
fn test_dpb_manager_find_or_recycle_slot_excluding() {
    let mut dpb = DpbManager::new(4);

    // All empty, exclude slot 0
    let exclude = vec![0i32];
    let slot = dpb.find_or_recycle_slot_excluding(&exclude);
    assert_eq!(slot, Some(1)); // should skip excluded slot 0

    // Mark slots 1,2,3 as valid, exclude 2
    dpb.entries[1].is_valid = true;
    dpb.entries[2].is_valid = true;
    dpb.entries[3].is_valid = true;
    let exclude = vec![2i32];

    // Should find slot 0 (empty, not excluded)
    let slot = dpb.find_or_recycle_slot_excluding(&exclude);
    assert_eq!(slot, Some(0));

    // All non-excluded slots valid, should recycle oldest non-excluded
    dpb.entries[0].is_valid = true;
    dpb.entries[0].frame_num = 10;
    dpb.entries[1].frame_num = 5;
    dpb.entries[3].frame_num = 15;

    // Exclude slot 2, recycle oldest among 0,1,3 -> slot 1 (frame_num=5)
    let exclude = vec![2i32];
    let slot = dpb.find_or_recycle_slot_excluding(&exclude);
    assert_eq!(slot, Some(1));
}
