//! Golden-hash test infrastructure (replaces the C++ differential oracle).
//!
//! Every byte-exact output of a deterministic test case is reduced to a
//! SHA-256 digest and pinned in the generated `golden_data.rs` (include!'d
//! below). The goldens were generated from the build that verified
//! byte-exact agreement with the C++ oracle; they now pin that behavior
//! permanently, so no C++ sources are needed at test time.
//!
//! The digest, lookup, record/collect, and `golden_data.rs` writer live in
//! `vacc_common::goldens`; this module keeps only this crate's `GOLDENS`
//! table, the module registry (`collect_entries`), and the regeneration
//! test.
//!
//! Regeneration (only after an intentional, re-verified behavior change,
//! with the sample streams available):
//! ```sh
//! H265_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens
//! cargo test -p vacc-software-decode   # rebuild picks up the new file
//! ```

include!("golden_data.rs");

pub(crate) fn golden(key: &str) -> Option<&'static str> {
    vacc_common::goldens::lookup(GOLDENS, key)
}

/// SHA-256 of `data` as lowercase hex (shared implementation).
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    vacc_common::goldens::sha256_hex(data)
}

/// Assert that a precomputed `actual` hash matches the golden for `key`.
pub(crate) fn assert_hash(key: &str, actual: &str) {
    vacc_common::goldens::assert_hash(GOLDENS, key, actual)
}

/// Assert that `data` hashes to the golden pinned for `key`.
pub(crate) fn assert_golden(key: &str, data: &[u8]) {
    vacc_common::goldens::assert_data(GOLDENS, key, data)
}

/// Canonical serialization helpers for grid/plane outputs (little-endian).
pub(crate) fn push_i16(out: &mut Vec<u8>, v: i16) {
    vacc_common::goldens::push_i16(out, v)
}

pub(crate) fn push_u16(out: &mut Vec<u8>, v: u16) {
    vacc_common::goldens::push_u16(out, v)
}

pub(crate) fn push_i32(out: &mut Vec<u8>, v: i32) {
    vacc_common::goldens::push_i32(out, v)
}

// ============================================================
// Regeneration
// ============================================================

/// Collect (key, sha256) from every golden-producing test module.
pub(crate) fn collect_entries() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    // See the `golden_entries` re-exports at the bottom of each converted
    // test module.
    v.extend(super::deblocking::golden_entries());
    v.extend(super::interpolation::golden_entries());
    v.extend(super::intra_prediction::golden_entries());
    v.extend(super::sao::golden_entries());
    v.extend(super::syntax_elements::golden_entries());
    v.extend(super::tier_e::golden_entries());
    v.extend(super::tier_f::golden_entries());
    v.extend(super::transform::golden_entries());
    v.extend(crate::h265::golden_entries());
    v
}

#[test]
fn regenerate_goldens() {
    if std::env::var_os("H265_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set H265_REGEN_GOLDENS=1 to rewrite golden_data.rs");
        return;
    }
    let entries = collect_entries();
    assert!(!entries.is_empty(), "no modules produced golden entries");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hevc/golden_data.rs");
    let n = vacc_common::goldens::write_golden_file(
        entries,
        &path,
        "H265_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens",
    );
    eprintln!("wrote {n} golden entries to {}", path.display());
}
