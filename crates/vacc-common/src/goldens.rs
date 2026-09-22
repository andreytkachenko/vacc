//! Golden-hash test infrastructure shared by the software decoders.
//!
//! Every byte-exact output of a deterministic test case is reduced to a
//! SHA-256 digest and pinned in a per-crate generated `golden_data.rs`
//! (a `pub(crate) const GOLDENS: &[(&str, &str)]` table, `include!`d by the
//! crate's `#[cfg(test)] goldens` module, e.g. `rust/goldens.rs` /
//! `hevc/goldens.rs`). The goldens pin behavior that was originally
//! verified byte-exact against the C/C++ oracles (since deleted); no oracle
//! sources are needed at test time.
//!
//! This module is the single implementation of the digest, the keyed
//! assertions, the thread-local record/collect capture, and the
//! `golden_data.rs` writer; the per-crate modules only keep their own
//! `GOLDENS` table, module registry (`collect_entries`), and regeneration
//! env var.

use std::cell::{Cell, RefCell};
use std::path::Path;

// ============================================================
// Self-contained SHA-256 (test-only content digest; not for security)
// ============================================================

struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    len_bytes: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            len_bytes: 0,
        }
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([
                block[4 * t],
                block[4 * t + 1],
                block[4 * t + 2],
                block[4 * t + 3],
            ]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        );
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    fn update(&mut self, data: &[u8]) {
        self.len_bytes += data.len() as u64;
        let mut rest = data;
        if self.buf_len > 0 {
            let n = (64 - self.buf_len).min(rest.len());
            self.buf[self.buf_len..self.buf_len + n].copy_from_slice(&rest[..n]);
            self.buf_len += n;
            rest = &rest[n..];
            if self.buf_len == 64 {
                Self::compress(&mut self.state, &self.buf);
                self.buf_len = 0;
            }
        }
        let full = rest.len() / 64;
        for i in 0..full {
            let mut block = [0u8; 64];
            block.copy_from_slice(&rest[i * 64..i * 64 + 64]);
            Self::compress(&mut self.state, &block);
        }
        let rem = rest.len() - full * 64;
        if rem > 0 {
            self.buf[..rem].copy_from_slice(&rest[full * 64..]);
            self.buf_len = rem;
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for i in self.buf_len..64 {
                self.buf[i] = 0;
            }
            Self::compress(&mut self.state, &self.buf);
            self.buf = [0u8; 64];
            self.buf_len = 0;
        }
        for i in self.buf_len..56 {
            self.buf[i] = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress(&mut self.state, &self.buf);
        let mut out = [0u8; 32];
        for (i, w) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

/// SHA-256 of `data` as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ============================================================
// Golden lookup / assertions
// ============================================================

/// Look up the golden digest pinned for `key` in `table`.
#[must_use]
pub fn lookup<'t>(table: &'t [(&'t str, &'t str)], key: &str) -> Option<&'t str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Assert that a precomputed `actual` hash matches the golden for `key`.
/// Skipped while golden collection is armed on the current thread (i.e.
/// during a regeneration run): new values are written by the collector.
pub fn assert_hash(table: &[(&str, &str)], key: &str, actual: &str) {
    if collecting() {
        return;
    }
    match lookup(table, key) {
        Some(expected) => assert_eq!(
            actual, expected,
            "golden mismatch for {key} (expected {expected}, got {actual})"
        ),
        None => panic!("no golden for {key}; run the regeneration test (see module docs)"),
    }
}

/// Assert that `data` hashes to the golden pinned for `key`. Same skip
/// semantics as [`assert_hash`].
pub fn assert_data(table: &[(&str, &str)], key: &str, data: &[u8]) {
    assert_hash(table, key, &sha256_hex(data));
}

// ============================================================
// Canonical serialization helpers for grid/plane outputs (little-endian)
// ============================================================

pub fn push_i16(out: &mut Vec<u8>, v: i16) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

// ============================================================
// Collection (used by `regenerate_goldens`)
// ============================================================

// Per-thread collection state: while armed, `record` appends
// `(key, sha256)` pairs to the per-thread collector.
thread_local! {
    static GOLDEN_COLLECTING: Cell<bool> = const { Cell::new(false) };
    static GOLDEN_COLLECTOR: RefCell<Vec<(String, String)>> =
        const { RefCell::new(Vec::new()) };
}

/// True while [`collect`] is armed on the current thread.
pub fn collecting() -> bool {
    GOLDEN_COLLECTING.with(Cell::get)
}

/// Record one case's output (hashed) under `key`. No-op unless collection
/// is armed on the current thread (i.e. inside [`collect`]).
pub fn record(key: &str, data: &[u8]) {
    if !GOLDEN_COLLECTING.with(Cell::get) {
        return;
    }
    GOLDEN_COLLECTOR.with(|c| c.borrow_mut().push((key.to_string(), sha256_hex(data))));
}

/// Run `f` with golden collection armed on the current thread and return
/// the recorded `(key, sha256)` pairs in recording order. The flag is reset
/// even if `f` panics.
pub fn collect<R: FnOnce()>(f: R) -> Vec<(String, String)> {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            GOLDEN_COLLECTING.with(|c| c.set(false));
            GOLDEN_COLLECTOR.with(|c| c.borrow_mut().clear());
        }
    }
    GOLDEN_COLLECTING.with(|c| c.set(true));
    let _guard = Guard;
    f();
    GOLDEN_COLLECTOR.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

// ============================================================
// Regeneration
// ============================================================

/// Format and write `entries` (in any order) to `out_path` as a
/// `golden_data.rs` body (`// DO NOT EDIT BY HAND ...` + `GOLDENS` table).
/// `regen_command` is the doc line naming the regeneration command. Panics
/// on duplicate keys. Returns the number of entries written.
#[must_use]
pub fn write_golden_file(
    entries: Vec<(String, String)>,
    out_path: &Path,
    regen_command: &str,
) -> usize {
    let mut entries = entries;
    entries.sort();
    let mut seen = std::collections::BTreeSet::new();
    for (k, _) in &entries {
        assert!(seen.insert(k.as_str()), "duplicate golden key {k}");
    }
    let mut out = String::from("// DO NOT EDIT BY HAND — generated by:\n//   ");
    out.push_str(regen_command);
    out.push_str("\n\npub(crate) const GOLDENS: &[(&str, &str)] = &[\n");
    for (k, v) in &entries {
        out.push_str(&format!("    ({k:?}, {v:?}),\n"));
    }
    out.push_str("];\n");
    let n = entries.len();
    std::fs::write(out_path, out).expect("write golden_data.rs");
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_test_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 1,000,000 x 'a' (multi-block padding edge cases).
        let data = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&data),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn record_collect_roundtrip() {
        let pairs = collect(|| {
            record("a", b"foo");
            record("b", b"barbar");
        });
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1, sha256_hex(b"foo"));
        assert!(!collecting());
        // Outside collection, record is a no-op even when a prior arm exists.
        record("c", b"baz");
        assert!(collect(|| {}).is_empty());
    }

    #[test]
    fn lookup_and_assert_hash() {
        let good = sha256_hex(b"known");
        let table = [("k", good.as_str())];
        assert_eq!(lookup(&table, "k"), Some(good.as_str()));
        assert!(lookup(&table, "other").is_none());
        assert_hash(&table, "k", &good);
    }

    #[test]
    #[should_panic(expected = "golden mismatch")]
    fn assert_hash_mismatch() {
        let good = sha256_hex(b"good");
        let bad = sha256_hex(b"bad");
        let table = [("k", good.as_str())];
        assert_hash(&table, "k", &bad);
    }

    #[test]
    #[should_panic(expected = "no golden for missing")]
    fn assert_hash_missing() {
        let table: [(&str, &str); 0] = [];
        assert_hash(&table, "missing", "x");
    }

    #[test]
    fn assert_data_matches() {
        let good = sha256_hex(b"known");
        let table = [("k", good.as_str())];
        assert_data(&table, "k", b"known");
    }

    #[test]
    fn push_serialization() {
        let mut out = Vec::new();
        push_i16(&mut out, -2);
        push_u16(&mut out, 0xABCD);
        push_i32(&mut out, -1);
        assert_eq!(out, vec![0xFE, 0xFF, 0xCD, 0xAB, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn write_golden_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vacc_common_goldens_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("golden_data_generated.rs");
        let entries = vec![
            (String::from("zeta"), String::from("ff")),
            (String::from("alpha"), String::from("ee")),
        ];
        let n = write_golden_file(entries, &path, "TEST=1 cargo test regenerate_goldens");
        assert_eq!(n, 2);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with(
            "// DO NOT EDIT BY HAND — generated by:\n//   TEST=1 cargo test regenerate_goldens\n\n"
        ));
        let table_line = text.lines().nth(3).unwrap();
        assert!(table_line.starts_with("pub(crate) const GOLDENS: &[(&str, &str)] = &["));
        assert!(text.contains("(\"alpha\", \"ee\"),"));
        assert!(text.contains("(\"zeta\", \"ff\"),"));
        // Sorted: alpha before zeta.
        assert!(text
            .find("(\"alpha\"")
            .unwrap()
            < text.find("(\"zeta\"").unwrap());
        assert!(text.ends_with("];\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "duplicate golden key")]
    fn write_golden_file_rejects_duplicates() {
        let dir = std::env::temp_dir().join(format!("vacc_common_goldens_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("golden_data_dup.rs");
        let entries = vec![
            (String::from("a"), String::from("1")),
            (String::from("a"), String::from("2")),
        ];
        write_golden_file(entries, &path, "TEST=1 cargo test regenerate_goldens");
    }
}
