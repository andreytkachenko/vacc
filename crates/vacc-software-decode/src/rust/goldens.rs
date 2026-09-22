//! Golden-hash test infrastructure (replaces the C differential oracle).
//!
//! Every byte-exact output of a deterministic test case is reduced to a
//! SHA-256 digest and pinned in the generated `golden_data.rs` (include!'d
//! below). The goldens were generated from the build that verified
//! byte-exact agreement with the C oracle (`c/src/vacc_sw264_test.c`,
//! since deleted); they now pin that behavior permanently, so no C sources
//! are needed at test time.
//!
//! Regeneration (only after an intentional, re-verified behavior change,
//! with the sample streams available):
//! ```sh
//! SW264_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens
//! cargo test -p vacc-software-decode   # rebuild picks up the new file
//! ```

include!("golden_data.rs");

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
pub(crate) fn sha256_hex(data: &[u8]) -> String {
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

pub(crate) fn golden(key: &str) -> Option<&'static str> {
    GOLDENS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Assert that `data` hashes to the golden pinned for `key`. Skipped while
/// collection is armed (`regenerate_goldens`): the new values are accepted by
/// the pending rewrite, and normal runs assert against the committed file.
pub(crate) fn assert_golden(key: &str, data: &[u8]) {
    if GOLDEN_COLLECTING.with(std::cell::Cell::get) {
        return;
    }
    let actual = sha256_hex(data);
    match golden(key) {
        Some(expected) => assert_eq!(
            actual, expected,
            "golden mismatch for {key} (expected {expected}, got {actual})"
        ),
        None => panic!("no golden for {key}; run the regeneration test (see module docs)"),
    }
}

// ============================================================
// Collection (used by `regenerate_goldens`)
// ============================================================

// Per-thread collection state: while armed, `record` appends
// `(key, sha256)` pairs to the per-thread collector.
thread_local! {
    static GOLDEN_COLLECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static GOLDEN_COLLECTOR: std::cell::RefCell<Vec<(String, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record one case's Rust output under `key`. No-op unless collection is
/// armed on the current thread (i.e. inside [`collect`]).
pub(crate) fn record(key: &str, data: &[u8]) {
    if !GOLDEN_COLLECTING.with(std::cell::Cell::get) {
        return;
    }
    GOLDEN_COLLECTOR.with(|c| c.borrow_mut().push((key.to_string(), sha256_hex(data))));
}

/// Run `f` with golden collection armed on the current thread and return the
/// recorded `(key, sha256)` pairs in recording order. The flag is reset even
/// if `f` panics.
pub(crate) fn collect<R: FnOnce()>(f: R) -> Vec<(String, String)> {
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
fn regenerate_goldens() {
    if std::env::var_os("SW264_REGEN_GOLDENS").is_none() {
        eprintln!("regenerate_goldens: set SW264_REGEN_GOLDENS=1 to rewrite golden_data.rs");
        return;
    }
    let mut entries = collect_entries();
    assert!(!entries.is_empty(), "no modules produced golden entries");
    entries.sort();
    let mut seen = std::collections::BTreeSet::new();
    for (k, _) in &entries {
        assert!(seen.insert(k.as_str()), "duplicate golden key {k}");
    }
    let mut out = String::from(
        "// DO NOT EDIT BY HAND — generated by:\n//   SW264_REGEN_GOLDENS=1 cargo test -p vacc-software-decode regenerate_goldens\n\n",
    );
    out.push_str("pub(crate) const GOLDENS: &[(&str, &str)] = &[\n");
    for (k, v) in &entries {
        out.push_str(&format!("    ({k:?}, {v:?}),\n"));
    }
    out.push_str("];\n");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rust/golden_data.rs");
    std::fs::write(&path, out).expect("write golden_data.rs");
    eprintln!(
        "wrote {} golden entries to {}",
        entries.len(),
        path.display()
    );
}
