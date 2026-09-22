//! Golden-hash test infrastructure (replaces the C differential oracle).
//!
//! Every byte-exact output of a deterministic test case is reduced to a
//! SHA-256 digest and pinned in the generated `golden_data.rs` (include!'d
//! below). The goldens were generated from the build that verified
//! byte-exact agreement with the C oracle (`c/src/vacc_sw264_test.c`,
//! since deleted); they now pin that behavior permanently, so no C sources
//! are needed at test time.
//!
//! The digest, lookup, record/collect, and `golden_data.rs` writer live in
//! `vacc_common::goldens`; this module keeps only this crate's `GOLDENS`
//! table, the module registry (`collect_entries`), and the regeneration
//! test.
//!
//! Regeneration (only after an intentional, re-verified behavior change,
//! with the sample streams available):
//! ```sh
//! SW264_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens
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

/// Assert that `data` hashes to the golden pinned for `key`. Skipped while
/// collection is armed (`regenerate_goldens`): the new values are accepted by
/// the pending rewrite, and normal runs assert against the committed file.
pub(crate) fn assert_golden(key: &str, data: &[u8]) {
    vacc_common::goldens::assert_data(GOLDENS, key, data)
}

/// Record one case's Rust output under `key`. No-op unless collection is
/// armed on the current thread (i.e. inside [`collect`]).
pub(crate) fn record(key: &str, data: &[u8]) {
    vacc_common::goldens::record(key, data)
}

/// Run `f` with golden collection armed on the current thread and return the
/// recorded `(key, sha256)` pairs in recording order. The flag is reset even
/// if `f` panics.
pub(crate) fn collect<R: FnOnce()>(f: R) -> Vec<(String, String)> {
    vacc_common::goldens::collect(f)
}

// ============================================================
// Regeneration
// ============================================================

/// Collect (key, sha256) from every golden-producing test module.
pub(crate) fn collect_entries() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    // See the `golden_entries` re-exports in the test modules.
    v.extend(crate::rust::tests::golden_entries());
    v.extend(crate::rust::deblock::golden_entries());
    v.extend(crate::decoder::golden_entries());
    v
}

#[test]
fn regenerate_goldens() {
    if std::env::var_os("SW264_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set SW264_REGEN_GOLDENS=1 to rewrite golden_data.rs");
        return;
    }
    let entries = collect_entries();
    assert!(!entries.is_empty(), "no modules produced golden entries");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rust/golden_data.rs");
    let n = vacc_common::goldens::write_golden_file(
        entries,
        &path,
        "SW264_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens",
    );
    eprintln!("wrote {n} golden entries to {}", path.display());
}
