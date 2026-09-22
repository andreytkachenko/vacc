//! Golden-pinned fuzz tests: run CAVLC/CABAC/deblock/mvpred op sequences
//! through the Rust core and pin every value read plus the final machine
//! state (cpb offset, both cache halves, all 1024 context states) to its
//! golden hash.

#![cfg(test)]

use crate::rust::bits::SliceBits;
use crate::rust::cabac::Cabac;
use crate::rust::deblock::{
    DEBLOCK_LC_ROWS, DEBLOCK_LC_SIZE, DEBLOCK_LC_STRIDE, DEBLOCK_LY_ROWS, DEBLOCK_LY_SIZE,
    DEBLOCK_LY_STRIDE, DEBLOCK_MB_STATE_LEN, DEBLOCK_Y_COL,
};
use crate::rust::goldens;
use crate::rust::mvpred::{MVPRED_IN_LEN, MVPRED_MB_LEN};

const PAD: usize = 32;

use vacc_common::rng::XorShift64Star;

/// Run `ops` through the Rust port on a zero-padded copy of `payload`.
fn run_r(payload: &[u8], ops: &[i32]) -> (Vec<i32>, i64, u64, u64, [u8; 1024]) {
    let mut buf = vec![0u8; PAD + payload.len() + PAD];
    buf[PAD..PAD + payload.len()].copy_from_slice(payload);
    let mut bits = SliceBits::new(&buf, PAD, payload.len());
    let mut cabac = Cabac::new();
    let mut out: Vec<i32> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        match ops[i] {
            0 => {
                out.push(bits.get_u1() as i32);
                i += 1;
            }
            1 => {
                let n = ops[i + 1] as usize;
                out.push(bits.get_uv(n) as i32);
                out.push(ops[i + 1]);
                i += 2;
            }
            2 => {
                out.push(bits.get_ue16(ops[i + 1] as u32) as i32);
                out.push(ops[i + 1]);
                i += 2;
            }
            3 => {
                out.push(bits.get_se16(ops[i + 1], ops[i + 2]));
                out.push(ops[i + 1]);
                out.push(ops[i + 2]);
                i += 3;
            }
            4 => {
                out.push(cabac.start(&mut bits) as i32);
                i += 1;
            }
            5 => {
                cabac.init(ops[i + 1] as u8, ops[i + 2] as usize);
                out.push(ops[i + 1]);
                out.push(ops[i + 2]);
                i += 3;
            }
            6 => {
                out.push(cabac.get_ae(&mut bits, ops[i + 1] as usize) as i32);
                out.push(ops[i + 1]);
                i += 2;
            }
            7 => {
                out.push(cabac.get_bypass(&mut bits) as i32);
                i += 1;
            }
            8 => {
                out.push(cabac.terminate(&mut bits) as i32);
                i += 1;
            }
            _ => panic!("bad op {}", ops[i]),
        }
    }
    (out, bits.cpb_off(), bits.cache0(), bits.cache1(), cabac.ctx)
}

fn compare(payload: &[u8], ops: &[i32], tag: &str) {
    let (r_out, r_cpb, r_c0, r_c1, r_ctx) = run_r(payload, ops);
    // Golden pin: every read value + the full final machine state.
    let mut data = Vec::with_capacity(r_out.len() * 4 + 24 + r_ctx.len());
    for v in &r_out {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&r_cpb.to_le_bytes());
    data.extend_from_slice(&r_c0.to_le_bytes());
    data.extend_from_slice(&r_c1.to_le_bytes());
    data.extend_from_slice(&r_ctx);
    goldens::assert_golden(tag, &data);
    goldens::record(tag, &data);
}

/// Random payload with EPB / stop sequences sprinkled in.
fn random_payload(rng: &mut XorShift64Star, min_len: usize, max_len: usize) -> Vec<u8> {
    let len = min_len + rng.below((max_len - min_len) as u64) as usize;
    let mut v: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
    // Inject a few 00 00 03 (EPB) and 00 00 00 (stop) sequences.
    for _ in 0..4 {
        if v.len() >= 4 + 3 {
            let at = 1 + rng.below((v.len() - 4) as u64) as usize;
            let kind = rng.below(2);
            v[at] = 0;
            v[at + 1] = 0;
            v[at + 2] = if kind == 0 { 3 } else { rng.below(3) as u8 };
        }
    }
    v
}

fn random_cavlc_ops(rng: &mut XorShift64Star, payload_len: usize) -> Vec<i32> {
    let mut ops: Vec<i32> = Vec::new();
    // Budget roughly the payload in bits plus some overread.
    let budget = (payload_len * 8 / 3 + 40) as u64;
    let mut spent: u64 = 0;
    while spent < budget {
        match rng.below(10) {
            0 | 1 => {
                ops.push(0); // get_u1
                spent += 1;
            }
            2..=4 => {
                let n = [1usize, 2, 4, 7, 8, 16, 31, 32][rng.below(8) as usize];
                ops.extend([1, n as i32]); // get_uv
                spent += n as u64;
            }
            5..=7 => {
                let upper = [0u32, 1, 15, 255, 65535][rng.below(5) as usize];
                ops.extend([2, upper as i32]); // get_ue16
                spent += 4;
            }
            _ => {
                let lower = -100 + rng.below(120) as i32;
                let upper = lower + 5 + rng.below(200) as i32;
                ops.extend([3, lower, upper]); // get_se16
                spent += 4;
            }
        }
    }
    ops
}

#[test]
fn cavlc_fuzz() {
    cavlc_fuzz_body();
}

fn cavlc_fuzz_body() {
    let mut rng = XorShift64Star(0x5eed_cabac);
    for iter in 0..300 {
        let payload = random_payload(&mut rng, 1, 64);
        let ops = random_cavlc_ops(&mut rng, payload.len());
        compare(&payload, &ops, &format!("cavlc #{iter}"));
    }
}

#[test]
fn cavlc_structured() {
    cavlc_structured_body();
}

fn cavlc_structured_body() {
    // Hand-built edge cases: EPB chains, stop sequences, full-width reads,
    // long Exp-Golomb codes crossing refills.
    let cases: &[&[u8]] = &[
        &[0x00, 0x00, 0x03, 0xFF, 0xFF, 0x00, 0x00, 0x03],
        &[0x00, 0x00, 0x00],
        &[0x00, 0x00, 0x01, 0xFF],
        &[0x00; 40],
        &[0xFF; 40],
        &[
            0b0000_0000,
            0b0000_0000,
            0b0000_0001,
            0b0000_0001,
            0b0000_0001,
        ],
        &[
            0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0xAA,
        ],
        &[0x01],
        &[
            0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
    ];
    let mut rng = XorShift64Star(0x5eed_57a5);
    for (ci, payload) in cases.iter().enumerate() {
        let ops = random_cavlc_ops(&mut rng, payload.len());
        compare(payload, &ops, &format!("cavlc-structured #{ci}"));
    }
}

/// CABAC op sequence: some CAVLC reads, cabac_start, cabac_init, a mix of
/// get_ae/get_bypass, cabac_terminate, then trailing CAVLC reads (exercising
/// the terminate LPS refill path). The cache is already primed by SliceBits::new.
fn random_cabac_ops(rng: &mut XorShift64Star) -> Vec<i32> {
    let mut ops: Vec<i32> = Vec::new();
    ops.extend(std::iter::repeat_n(0, (1 + rng.below(4)) as usize)); // get_u1
    ops.push(4); // cabac_start
    let qp = rng.below(52) as i32;
    let idc = rng.below(4) as i32;
    ops.extend([5, qp, idc]); // cabac_init
    let n_bins = 4 + rng.below(60) as usize;
    for _ in 0..n_bins {
        if rng.below(10) < 8 {
            // Bias ctx indices toward low values and 276 (the hardcoded one).
            let c = if rng.below(8) == 0 {
                276
            } else {
                rng.below(400) as i32
            };
            ops.extend([6, c]); // get_ae
        } else {
            ops.push(7); // get_bypass
        }
    }
    ops.push(8); // cabac_terminate
    ops.extend(std::iter::repeat_n(0, (1 + rng.below(5)) as usize)); // trailing CAVLC reads
    ops
}

#[test]
fn cabac_fuzz() {
    cabac_fuzz_body();
}

fn cabac_fuzz_body() {
    let mut rng = XorShift64Star(0x5eed_cabac2);
    for iter in 0..300 {
        let payload = random_payload(&mut rng, 8, 96);
        let ops = random_cabac_ops(&mut rng);
        compare(&payload, &ops, &format!("cabac #{iter}"));
    }
}

#[test]
fn cabac_init_full_domain() {
    cabac_init_full_domain_body();
}

fn cabac_init_full_domain_body() {
    // Every (idc, qp) pair: init and compare all 1024 context states.
    let mut rng = XorShift64Star(0x5eed_cab4);
    for idc in 0..4i32 {
        for qp in 0..=51i32 {
            // A few CAVLC bits so cabac_start has a primed cache, then start+init.
            let payload = random_payload(&mut rng, 8, 24);
            let mut ops: Vec<i32> = vec![0, 0];
            ops.push(4);
            ops.extend([5, qp, idc]);
            compare(&payload, &ops, &format!("init idc={idc} qp={qp}"));
        }
    }
}

// ------------------------------------------------------- Tier B: intra

const INTRA_BUF: usize = 18 * 48;
const INTRA_O: usize = 2 * 48 + 16; // block origin within the neighborhood buf
const INTRA_STRIDE: usize = 48;

/// Valid internal mode range per kind (C enum sizes; switches are unguarded).
fn intra_max_mode(kind: i32) -> u32 {
    match kind {
        0 => 13, // I4x4_*: 14 modes
        1 => 31, // I8x8_*: 32 modes
        _ => 6,  // I16x16_* / IC8x8_*: 7 modes
    }
}

fn compare_intra(kind: i32, mode: u32, buf: &[u8; INTRA_BUF], tag: &str) {
    // Rust run on the same inputs.
    let mut b = *buf;
    match kind {
        0 => crate::rust::intra::intra4x4(&mut b, INTRA_O, INTRA_STRIDE, mode),
        1 => crate::rust::intra::intra8x8(&mut b, INTRA_O, INTRA_STRIDE, mode),
        2 => crate::rust::intra::intra16x16(&mut b, INTRA_O, INTRA_STRIDE, mode),
        3 => crate::rust::intra::intra_chroma(&mut b, INTRA_O, INTRA_STRIDE, mode),
        _ => panic!("{tag}: bad kind {kind}"),
    }
    let rows = match kind {
        0 => 4usize,
        1 => 8,
        _ => 16,
    };
    let cols = if kind == 3 { 8 } else { rows }; // chroma writes only the low 8 bytes/row
    let mut r_out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            r_out.push(b[INTRA_O + r * INTRA_STRIDE + c]);
        }
    }

    goldens::assert_golden(tag, &r_out);
    goldens::record(tag, &r_out);
}

#[test]
fn intra_fuzz_all_modes() {
    intra_fuzz_all_modes_body();
}

fn intra_fuzz_all_modes_body() {
    let mut rng = XorShift64Star(0x5eed_13a1);
    for kind in 0..4i32 {
        for mode in 0..=intra_max_mode(kind) {
            for iter in 0..50 {
                let buf: [u8; INTRA_BUF] = std::array::from_fn(|_| rng.byte());
                compare_intra(
                    kind,
                    mode,
                    &buf,
                    &format!("intra k={kind} m={mode} #{iter}"),
                );
            }
        }
    }
}

#[test]
fn intra_structured() {
    intra_structured_body();
}

fn intra_structured_body() {
    // Structured neighborhoods: flat (left==top — the common DC/edge case),
    // row/col/diagonal ramps, checkerboard, all-zero/all-max.
    let mut bufs: Vec<[u8; INTRA_BUF]> = Vec::new();
    for v in [0u8, 255, 128] {
        bufs.push([v; INTRA_BUF]);
    }
    bufs.push(std::array::from_fn(|i| (i % 48) as u8)); // column ramp
    bufs.push(std::array::from_fn(|i| ((i / 48) * 16) as u8)); // row ramp
    bufs.push(std::array::from_fn(|i| ((i / 48 + i % 48) % 256) as u8)); // diagonal ramp
    bufs.push(std::array::from_fn(|i| {
        if (i / 48 + i % 48) & 1 == 0 { 0 } else { 255 }
    })); // checkerboard
    // Left half and top row constant, interior random (typical MB boundary).
    let mut rng = XorShift64Star(0x5eed_13a2);
    for _ in 0..8 {
        let v = rng.byte();
        let mut b = [0u8; INTRA_BUF];
        for r in 0..18 {
            for c in 0..48 {
                // top row (r<2) and left col (c==15) = v; rest random.
                b[r * 48 + c] = if r < 2 || c == 15 { v } else { rng.byte() };
            }
        }
        bufs.push(b);
    }
    for (bi, buf) in bufs.iter().enumerate() {
        for kind in 0..4i32 {
            for mode in 0..=intra_max_mode(kind) {
                compare_intra(
                    kind,
                    mode,
                    buf,
                    &format!("intra-struct #{bi} k={kind} m={mode}"),
                );
            }
        }
    }
}

const RES_PLANE: usize = 200; // per-plane pixel region (4x4 + 8x8 blocks)
const RES_PIX: usize = RES_PLANE * 3;

#[allow(clippy::too_many_arguments)]
fn compare_residual(
    coeffs: &[i32; 64],
    ws4: &[i8; 96],
    ws8: &[i8; 384],
    qp: &[u8; 3],
    inter: i32,
    pix_in: &[u8; RES_PIX],
    ops: &[i32],
    tag: &str,
) {
    // Rust run on the same inputs.
    let mut st = crate::rust::residual::Residual {
        c: *coeffs,
        qp: *qp,
        ws4: std::array::from_fn(|i| std::array::from_fn(|j| ws4[i * 16 + j])),
        ws8: std::array::from_fn(|i| std::array::from_fn(|j| ws8[i * 64 + j])),
        inter: inter != 0,
    };
    let mut pix = *pix_in;
    let mut r_out: Vec<i32> = Vec::new();
    for chunk in ops.chunks(3) {
        let (op, plane, dcidx) = (chunk[0], chunk[1] as usize, chunk[2]);
        // C oracle resets ctx->c from coeffs before each op (per-block
        // decoding fills the coeff buffer fresh from the entropy decoder).
        st.c = *coeffs;
        let base = plane * RES_PLANE + if op == 2 { 64 } else { 0 };
        // Kernels take the block origin (C passes p = pix + base).
        match op {
            0 => st.add_idct4x4(plane, dcidx, &mut pix[base..], 16),
            1 => st.add_dc4x4(plane, dcidx, &mut pix[base..], 16),
            2 => st.add_idct8x8(plane, &mut pix[base..], 16),
            _ => panic!("{tag}: bad op {op}"),
        }
        let n = if op == 2 { 8 } else { 4 };
        for r in 0..n {
            for c in 0..n {
                r_out.push(pix[base + r * 16 + c] as i32);
            }
        }
    }

    let mut data = Vec::with_capacity(r_out.len() * 4 + RES_PIX + st.c.len() * 4);
    for v in &r_out {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&pix);
    for v in &st.c {
        data.extend_from_slice(&v.to_le_bytes());
    }
    goldens::assert_golden(tag, &data);
    goldens::record(tag, &data);
}

#[test]
fn residual_fuzz() {
    residual_fuzz_body();
}

fn residual_fuzz_body() {
    let mut rng = XorShift64Star(0x5eed_4c42);
    for iter in 0..1000 {
        // Coeffs: mostly realistic magnitude, occasionally extreme to exercise
        // i32 wraparound in the dequant.
        let coeffs: [i32; 64] = std::array::from_fn(|k| {
            if k == 0 && rng.below(8) == 0 {
                if rng.below(2) == 0 {
                    i32::MAX
                } else {
                    i32::MIN
                }
            } else {
                (rng.next() % 32768) as i32 - 16384
            }
        });
        let ws4: [i8; 96] = std::array::from_fn(|_| rng.byte() as i8);
        let ws8: [i8; 384] = std::array::from_fn(|_| rng.byte() as i8);
        let qp: [u8; 3] = [
            rng.below(52) as u8,
            rng.below(52) as u8,
            rng.below(52) as u8,
        ];
        let inter = rng.below(2) as i32;
        let pix: [u8; RES_PIX] = std::array::from_fn(|_| rng.byte());
        let n_ops = 1 + rng.below(6) as usize;
        let ops: Vec<i32> = (0..n_ops)
            .flat_map(|_| {
                let op = rng.below(3) as i32;
                let plane = rng.below(3) as i32;
                // DCidx: -1 (no override) only valid for add_idct4x4, which is
                // the only kernel guarding it (C add_dc4x4 reads c[16+DCidx]
                // unguarded). add_idct8x8 ignores it.
                let dcidx = if op == 0 {
                    if rng.below(4) == 0 {
                        -1
                    } else {
                        rng.below(8) as i32
                    }
                } else if op == 1 {
                    rng.below(8) as i32
                } else {
                    0
                };
                [op, plane, dcidx]
            })
            .collect();
        compare_residual(
            &coeffs,
            &ws4,
            &ws8,
            &qp,
            inter,
            &pix,
            &ops,
            &format!("residual #{iter}"),
        );
    }
}

// --- DC transforms (Tier E0) ---

const DC_PIX: usize = 1152; // plane bases 0/320/640 (16x16, stride 16), dc2x2 base 960 (16 rows x 8, stride 8)

fn compare_transform_dc(
    coeffs: &[i32; 64],
    ws4: &[i8; 96],
    qp: &[u8; 3],
    inter: i32,
    pix_in: &[u8; DC_PIX],
    ops: &[i32],
    tag: &str,
) {
    // Rust run on the same inputs. op: 0 = transform_dc4x4(plane, flag),
    // 1 = transform_dc2x2(flag); flag = store_later (C guard bit).
    let mut st = crate::rust::residual::Residual {
        c: *coeffs,
        qp: *qp,
        ws4: std::array::from_fn(|i| std::array::from_fn(|j| ws4[i * 16 + j])),
        ws8: [[0i8; 64]; 6],
        inter: inter != 0,
    };
    let mut pix = *pix_in;
    let mut r_out: Vec<i32> = Vec::new();
    for chunk in ops.chunks(3) {
        let (op, plane, flag) = (chunk[0], chunk[1] as usize, chunk[2]);
        // C oracle resets ctx->c from coeffs before each op.
        st.c = *coeffs;
        match op {
            0 => {
                let base = plane * 320;
                st.transform_dc4x4(plane, flag != 0, &mut pix[base..], 16);
                if flag == 0 {
                    for r in 0..16 {
                        for c in 0..16 {
                            r_out.push(pix[base + r * 16 + c] as i32);
                        }
                    }
                }
            }
            1 => {
                st.transform_dc2x2(flag != 0, &mut pix[960..], 16);
                if flag == 0 {
                    for r in 0..16 {
                        for c in 0..8 {
                            r_out.push(pix[960 + r * 8 + c] as i32);
                        }
                    }
                }
            }
            _ => panic!("{tag}: bad op {op}"),
        }
    }

    let mut data = Vec::with_capacity(r_out.len() * 4 + DC_PIX + st.c.len() * 4);
    for v in &r_out {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&pix);
    for v in &st.c {
        data.extend_from_slice(&v.to_le_bytes());
    }
    goldens::assert_golden(tag, &data);
    goldens::record(tag, &data);
}

#[test]
fn transform_dc_handcheck() {
    // Hand-verified against C (edge264_residual.c) with ws4 = all 1s, qp = 0:
    // LS4x4 = 1*normAdjust4x4[0][0] = 10.
    let ws4: [i8; 96] = [1; 96];
    let qp: [u8; 3] = [0, 0, 0];
    // dc4x4: c[0]=1024 -> every f lane 1024 -> dc = (1024*10+32)>>6 = 160
    // -> r = (160+32)>>6 = 3 -> every pixel +3.
    let mut coeffs: [i32; 64] = [0; 64];
    coeffs[0] = 1024;
    let mut pix = [100u8; DC_PIX];
    let mut st = crate::rust::residual::Residual {
        c: coeffs,
        qp,
        ws4: std::array::from_fn(|i| std::array::from_fn(|j| ws4[i * 16 + j])),
        ws8: [[0i8; 64]; 6],
        inter: false,
    };
    st.transform_dc4x4(0, false, &mut pix[..256], 16);
    assert!(pix[..256].iter().all(|&p| p == 103), "dc4x4 store-now");
    assert_eq!(&st.c[0..16], &[0; 16], "dc4x4 zeroes c[0..16]");
    // store-later: c[16..32] = the 16 first-stage dc values (all 160).
    st.c = coeffs;
    st.transform_dc4x4(0, true, &mut pix[..256], 16);
    assert_eq!(&st.c[16..32], &[160i32; 16], "dc4x4 store-later");
    // dc2x2: c[0]=1024 (cb_a), c[4]=2048 (cr_a). d0=[3072,0,0,0],
    // d1=[-1024,0,0,0] -> f0=f1=[3072,0,-1024,0] -> dcCb = [960,-320,960,-320]
    // (>>5 plain), dcCr = 0 -> rb = [15,-5,15,-5], rr = 0. Cb rows (even) get
    // [+15 x4, -5 x4]; Cr rows (odd) unchanged.
    coeffs = [0; 64];
    coeffs[0] = 1024;
    coeffs[4] = 2048;
    let mut cpix = [50u8; 128];
    st.c = coeffs;
    st.transform_dc2x2(false, &mut cpix, 16);
    for r in 0..16 {
        if r % 2 == 0 {
            assert_eq!(&cpix[r * 8..r * 8 + 4], &[65; 4], "cb row {r} lo");
            assert_eq!(&cpix[r * 8 + 4..r * 8 + 8], &[45; 4], "cb row {r} hi");
        } else {
            assert_eq!(&cpix[r * 8..r * 8 + 8], &[50; 8], "cr row {r}");
        }
    }
    // store-later: c[16..20] = dcCb, c[20..24] = dcCr.
    st.c = coeffs;
    st.transform_dc2x2(true, &mut cpix, 16);
    assert_eq!(&st.c[16..20], &[960, -320, 960, -320], "dc2x2 Cb stored");
    assert_eq!(&st.c[20..24], &[0; 4], "dc2x2 Cr stored");
}

#[test]
fn transform_dc_fuzz() {
    transform_dc_fuzz_body();
}

fn transform_dc_fuzz_body() {
    let mut rng = XorShift64Star(0x5eed_dc42);
    for iter in 0..1000 {
        // Coeffs: realistic magnitude, occasionally extreme to exercise the
        // i32 wraparound in f*LS and the (dc+32)>>6 stage.
        let coeffs: [i32; 64] = std::array::from_fn(|k| {
            if k == 0 && rng.below(8) == 0 {
                if rng.below(2) == 0 {
                    i32::MAX
                } else {
                    i32::MIN
                }
            } else {
                (rng.next() % 32768) as i32 - 16384
            }
        });
        let ws4: [i8; 96] = std::array::from_fn(|_| rng.byte() as i8);
        let qp: [u8; 3] = [
            rng.below(52) as u8,
            rng.below(52) as u8,
            rng.below(52) as u8,
        ];
        let inter = rng.below(2) as i32;
        let pix: [u8; DC_PIX] = std::array::from_fn(|_| rng.byte());
        let n_ops = 1 + rng.below(6) as usize;
        let ops: Vec<i32> = (0..n_ops)
            .flat_map(|_| {
                let op = rng.below(2) as i32;
                // plane is only used by dc4x4 (C ignores it for dc2x2);
                // flag = store_later guard bit.
                let plane = if op == 0 { rng.below(3) as i32 } else { 0 };
                let flag = rng.below(2) as i32;
                [op, plane, flag]
            })
            .collect();
        compare_transform_dc(
            &coeffs,
            &ws4,
            &qp,
            inter,
            &pix,
            &ops,
            &format!("transform_dc #{iter}"),
        );
    }
}

// --- Inter MC (Tier B) ---

#[allow(clippy::too_many_arguments)]
fn compare_inter(
    mode: u32,
    w: usize,
    h: usize,
    sstride: usize,
    src: &[u8],
    dstride: usize,
    dst_in: &[u8],
    wod: &[i16; 8],
    tag: &str,
) {
    // Rust: dst starts as a copy of dst_in (luma MC reads+writes in place).
    let mut dst = vec![0u8; h * dstride];
    for r in 0..h {
        dst[r * dstride..r * dstride + w].copy_from_slice(&dst_in[r * dstride..r * dstride + w]);
    }
    // `src` is the (h+5) x (w+5) neighborhood; block top-left at src+2*ss+2.
    crate::rust::inter::inter_luma(src, &mut dst, w, h, mode, sstride, dstride, wod);

    let r_out: Vec<u8> = dst
        .chunks(dstride)
        .flat_map(|row| row[..w].iter().copied())
        .collect();
    goldens::assert_golden(tag, &r_out);
    goldens::record(tag, &r_out);
}

#[test]
fn inter_fuzz_all_modes() {
    inter_fuzz_all_modes_body();
}

fn inter_fuzz_all_modes_body() {
    // All 48 luma QPEL modes, no_weight, random fills.
    let mut rng = XorShift64Star(0x5eed_132b);
    for &(w, base) in &[(4usize, 0u32), (8, 16), (16, 32)] {
        let sstride = w + 8; // >= w+7 neighborhood cols; <= 48
        let dstride = w + 8;
        let hs: &[usize] = if base == 32 { &[8, 16] } else { &[4, 8, 16] };
        for y in 0..4u32 {
            for x in 0..4u32 {
                let mode = base + y * 4 + x;
                for &h in hs {
                    for iter in 0..5 {
                        let src: Vec<u8> = (0..(h + 5) * sstride + 8).map(|_| rng.byte()).collect();
                        let dst_in: Vec<u8> = (0..h * dstride).map(|_| rng.byte()).collect();
                        compare_inter(
                            mode,
                            w,
                            h,
                            sstride,
                            &src,
                            dstride,
                            &dst_in,
                            &crate::rust::inter::WOD_NO_WEIGHT,
                            &format!("inter {w}x{h} mode={mode} #{iter}"),
                        );
                    }
                }
            }
        }
    }
}

fn mk_wod(wq: i8, wp: i8, oy: i16, l: i16) -> [i16; 8] {
    let wy = (((wp as u16) & 0xFF) << 8) | ((wq as u16) & 0xFF);
    [wy as i16, oy, l, l, 256, 256, l, 0]
}

#[test]
fn inter_fuzz_weighted() {
    inter_fuzz_weighted_body();
}

fn inter_fuzz_weighted_body() {
    // Weighted prediction: per-pixel weighting is mode-independent, but the
    // per-lane shift (wod[2] vs wod[6]) needs w>=8; include one pattern with
    // different shifts to pin the exact C lane behavior.
    let mut rng = XorShift64Star(0x5eed_7a11);
    let pats: [([i16; 8], &str); 5] = [
        (mk_wod(1, 1, 1, 1), "wq=1 wp=1 o=1 L=1"),
        (mk_wod(-2, 3, -5, 4), "wq=-2 wp=3 o=-5 L=4"),
        (mk_wod(0, 2, 7, 2), "wq=0 wp=2 o=7 L=2"),
        (mk_wod(-1, -1, 3, 5), "wq=-1 wp=-1 o=3 L=5"),
        ([0x0201, 3, 2, 0, 256, 256, 5, 0], "wq=1 wp=2 o=3 L2=2 L6=5"),
    ];
    for &(w, base) in &[(4usize, 0u32), (8, 16), (16, 32)] {
        let sstride = w + 8;
        let dstride = w + 8;
        for y in 0..4u32 {
            for x in 0..4u32 {
                let mode = base + y * 4 + x;
                for &(wod, tag) in &pats {
                    let src: Vec<u8> = (0..9 * sstride + 8).map(|_| rng.byte()).collect();
                    let dst_in: Vec<u8> = (0..4 * dstride).map(|_| rng.byte()).collect();
                    compare_inter(
                        mode,
                        w,
                        4,
                        sstride,
                        &src,
                        dstride,
                        &dst_in,
                        &wod,
                        &format!("interW {w} mode={mode} {tag}"),
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_chroma(
    w: usize,
    h: usize,
    sstride: usize,
    src: &[u8],
    dstride: usize,
    dst_in: &[u8],
    x_frac: u32,
    y_frac: u32,
    wod: &[i16; 8],
    tag: &str,
) {
    let cw = w / 2;
    // Rust: dst starts as a copy of dst_in (chroma MC reads+writes in place).
    let mut dst = vec![0u8; h * dstride];
    for r in 0..h {
        dst[r * dstride..r * dstride + cw].copy_from_slice(&dst_in[r * dstride..r * dstride + cw]);
    }
    crate::rust::inter::inter_chroma(src, &mut dst, w, h, x_frac, y_frac, sstride, dstride, wod);

    let r_out: Vec<u8> = dst
        .chunks(dstride)
        .flat_map(|row| row[..cw].iter().copied())
        .collect();
    goldens::assert_golden(tag, &r_out);
    goldens::record(tag, &r_out);
}

#[test]
fn chroma_fuzz_all_fracs() {
    chroma_fuzz_all_fracs_body();
}

fn chroma_fuzz_all_fracs_body() {
    // All 64 sub-chroma-pel positions, no_weight, random fills.
    let mut rng = XorShift64Star(0x5eed_c1a4);
    for &w in &[4usize, 8, 16] {
        let cw = w / 2;
        let sstride = cw + 6; // >= cw+1; <= 48
        let dstride = cw + 6;
        for &h in &[4usize, 8] {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    for iter in 0..3 {
                        let src: Vec<u8> = (0..(h + 2) * sstride).map(|_| rng.byte()).collect();
                        let dst_in: Vec<u8> = (0..h * dstride).map(|_| rng.byte()).collect();
                        compare_chroma(
                            w,
                            h,
                            sstride,
                            &src,
                            dstride,
                            &dst_in,
                            x,
                            y,
                            &crate::rust::inter::WOD_NO_WEIGHT,
                            &format!("chroma {w}x{h} x{x} y{y} #{iter}"),
                        );
                    }
                }
            }
        }
    }
}

fn mk_wodc(wq: i8, wp: i8, sh: i16, ocb: i16, ocr: i16) -> [i16; 8] {
    let w = (((wp as u16 & 0xFF) << 8) | (wq as u16 & 0xFF)) as i16;
    [256, 0, sh, sh, w, w, ocb, ocr]
}

#[test]
fn chroma_fuzz_weighted() {
    chroma_fuzz_weighted_body();
}

fn chroma_fuzz_weighted_body() {
    // Weighted prediction: signed weight bytes (pmaddubsw second operand),
    // per-plane offsets, and shift counts that pin this machine's
    // floor-division sra quirk (sh >= 15 saturates to sign, not i16 -32768).
    let mut rng = XorShift64Star(0x5eed_7c2d);
    let pats: [([i16; 8], &str); 9] = [
        (crate::rust::inter::WOD_NO_WEIGHT, "no_weight"),
        (mk_wodc(1, 1, 1, 1, 1), "wq=1 wp=1 sh=1"),
        (mk_wodc(-2, 3, -5, 4, -4), "signed wq/wp, sh=-5"),
        (mk_wodc(127, -128, 6, 32767, -32768), "extremes+sat"),
        (
            [0x0201, 3, 2, 1, 0x0403, 0x0605, 7, 9],
            "fitter WOD_WT (sh3=1 sh7=9)",
        ),
        (mk_wodc(1, 1, 14, 0, 0), "sh=14 (last plain shift)"),
        (mk_wodc(1, -128, 15, 0, 0), "sh=15 (floor boundary)"),
        (mk_wodc(1, -128, 16, 0, 0), "sh=16"),
        (mk_wodc(-1, 127, 32, 0, 0), "sh=32"),
    ];
    for &w in &[4usize, 8, 16] {
        let cw = w / 2;
        let sstride = cw + 6;
        let dstride = cw + 6;
        for &h in &[4usize, 8] {
            for y in (0..8u32).step_by(2) {
                for x in (0..8u32).step_by(2) {
                    for &(wod, tag) in &pats {
                        let src: Vec<u8> = (0..(h + 2) * sstride).map(|_| rng.byte()).collect();
                        let dst_in: Vec<u8> = (0..h * dstride).map(|_| rng.byte()).collect();
                        compare_chroma(
                            w,
                            h,
                            sstride,
                            &src,
                            dstride,
                            &dst_in,
                            x,
                            y,
                            &wod,
                            &format!("chromaW {w}x{h} x{x} y{y} {tag}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn residual_full_qp_domain() {
    residual_full_qp_domain_body();
}

fn residual_full_qp_domain_body() {
    // Every QP x inter x op combination with fixed non-trivial inputs.
    let coeffs: [i32; 64] = std::array::from_fn(|k| {
        if k % 5 == 0 {
            1234 - k as i32
        } else {
            k as i32 * 7
        }
    });
    let ws4: [i8; 96] = std::array::from_fn(|k| ((k % 256) as u16 * 7 + 3) as u8 as i8);
    let ws8: [i8; 384] = std::array::from_fn(|k| ((k % 256) as u16 * 5 + 11) as u8 as i8);
    let pix: [u8; RES_PIX] = std::array::from_fn(|k| ((k as u16 * 13 + 91) % 256) as u8);
    for inter in 0..2i32 {
        for op in 0..3i32 {
            for qp in 0..=51u8 {
                let ops = vec![op, 0, 3];
                compare_residual(
                    &coeffs,
                    &ws4,
                    &ws8,
                    &[qp, qp, qp],
                    inter,
                    &pix,
                    &ops,
                    &format!("residual op={op} inter={inter} qp={qp}"),
                );
            }
        }
    }
}

#[test]
fn deblock_oracle_smoke() {
    use crate::rust::deblock::run_rust_deblock;
    let mut rng = XorShift64Star(0xdeb1_0ca7);
    // Zero MB state (intra, QP=18) for all three slots; only the current MB's
    // filter_edges is varied. QP[3]@0, mbIsInterFlag@3, filter_edges@4.
    let mut mb_state = vec![0u8; 3 * DEBLOCK_MB_STATE_LEN];
    // QP=18 for ALL three MBs: alpha/beta are averaged over current + left/top
    // neighbours, so zero-QP neighbours would drag the averaged index below the
    // alpha table's nonzero range and disable filtering.
    for slot in 0..3usize {
        let b = slot * DEBLOCK_MB_STATE_LEN;
        mb_state[b] = 18;
        mb_state[b + 1] = 18;
        mb_state[b + 2] = 18;
    }
    let cur = 2 * DEBLOCK_MB_STATE_LEN;
    let y_in: Vec<u8> = (0..DEBLOCK_LY_SIZE).map(|_| rng.byte()).collect();
    let c_in: Vec<u8> = (0..DEBLOCK_LC_SIZE).map(|_| rng.byte()).collect();

    // fe=0 is a guaranteed no-op: deblock_mb returns before touching pixels.
    mb_state[cur + 4] = 0;
    let (y_out, c_out) = run_rust_deblock(&mb_state, 0, 0, 0, 0, &y_in, &c_in);
    assert_eq!(y_out, y_in, "fe=0 luma must be untouched");
    assert_eq!(c_out, c_in, "fe=0 chroma must be untouched");

    // fe in {1,2,3}: deterministic across repeated calls.
    for &fe in &[1i32, 2, 3] {
        mb_state[cur + 4] = fe as u8;
        let (a_y, a_c) = run_rust_deblock(&mb_state, fe, 0, 0, 0, &y_in, &c_in);
        let (b_y, b_c) = run_rust_deblock(&mb_state, fe, 0, 0, 0, &y_in, &c_in);
        assert_eq!(a_y, b_y, "fe={fe} luma not deterministic");
        assert_eq!(a_c, b_c, "fe={fe} chroma not deterministic");
    }

    // A hard step exactly at the current MB's left boundary must be filtered
    // when bit 0 is set (proves the filter actually fires, not just no-ops).
    // Deblocking is gated by bS, which needs coded blocks: mark every luma
    // block in the current and left MBs as non-zero-coeff (nC[0..15]).
    for slot in [1usize, 2] {
        let b = slot * DEBLOCK_MB_STATE_LEN;
        for k in 0..16 {
            mb_state[b + 10 + k] = 1; // nC luma 4x4 flags
        }
    }
    // A weak edge at the current MB's left boundary: |p0-q0| < alpha and flat
    // sides (max(|p1-p0|,|q1-q0|) < beta) so the filter actually modifies it.
    // (A hard 0/255 step is deliberately NOT filtered by design.)
    let mut y_step = vec![0u8; DEBLOCK_LY_SIZE];
    for r in 0..DEBLOCK_LY_ROWS {
        let row = &mut y_step[r * DEBLOCK_LY_STRIDE..(r + 1) * DEBLOCK_LY_STRIDE];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = if c >= DEBLOCK_Y_COL { 131 } else { 128 };
        }
    }
    let c_zero = vec![0u8; DEBLOCK_LC_SIZE];
    mb_state[cur + 4] = 1;
    let (ys, _) = run_rust_deblock(&mb_state, 1, 0, 0, 0, &y_step, &c_zero);
    assert_ne!(ys, y_step, "weak coded edge with fe=1 must be filtered");
}

/// Run one deblock through the Rust port and pin the output to its golden.
#[allow(clippy::too_many_arguments)]
fn compare_deblock(
    mb_state: &[u8],
    fe: i32,
    entropy: i32,
    off_a: i16,
    off_b: i16,
    y_in: &[u8],
    c_in: &[u8],
    tag: &str,
) -> bool {
    let (r_y, r_c) = crate::rust::deblock::run_rust_deblock(
        mb_state,
        fe,
        entropy,
        off_a as i32,
        off_b as i32,
        y_in,
        c_in,
    );
    let mut data = Vec::with_capacity(r_y.len() + r_c.len());
    data.extend_from_slice(&r_y);
    data.extend_from_slice(&r_c);
    goldens::assert_golden(tag, &data);
    goldens::record(tag, &data);
    r_y != y_in || r_c != c_in // true if the filter actually modified pixels
}

#[test]
fn deblock_fuzz() {
    deblock_fuzz_body();
}

fn deblock_fuzz_body() {
    use DEBLOCK_MB_STATE_LEN as L;
    let mut rng = XorShift64Star(0xdb10_c7a5_2690);
    let mut fired = 0usize;
    for iter in 0..400usize {
        // Branch profile: 0=P, 1=B-16x16, 2=B-8x8, 3=intra, 4=random.
        let prof = rng.below(5);
        let mut mb_state = vec![0u8; 3 * L];
        for slot in 0..3usize {
            let b = slot * L;
            mb_state[b] = rng.below(52) as u8; // qy
            mb_state[b + 1] = rng.below(52) as u8; // qcb
            mb_state[b + 2] = rng.below(52) as u8; // qcr
            let inter = match prof {
                0..=2 => 1,
                3 => 0,
                _ => rng.below(2) as u8,
            };
            mb_state[b + 3] = inter;
            mb_state[b + 4] = rng.below(4) as u8; // fe (overridden for slot 2)
            let eq = if prof == 1 && slot == 2 {
                0x1b5fbbff
            } else {
                rng.next() as u32
            };
            mb_state[b + 5..b + 9].copy_from_slice(&eq.to_le_bytes());
            mb_state[b + 9] = rng.below(2) as u8; // ts8x8
            for k in 0..48 {
                mb_state[b + 10 + k] = if rng.below(5) == 0 { 0 } else { 1 };
            }
            for k in 0..8 {
                // P profile: all refIdx=-1 so refIdx_s[1]==-1 for every slot.
                let v = match prof {
                    0 => -1i8,
                    _ => [-1, -1, 0, 1][rng.below(4) as usize] as i8,
                };
                mb_state[b + 58 + k] = v as u8;
            }
            for k in 0..8 {
                mb_state[b + 66 + k] = rng.below(3) as u8;
            }
            for k in 0..64 {
                let mv = (rng.below(21) as i16) - 10;
                mb_state[b + 74 + 2 * k..b + 74 + 2 * k + 2].copy_from_slice(&mv.to_le_bytes());
            }
        }
        let fe = (1 + rng.below(7)) as i32; // 1..7
        let entropy = rng.below(2) as i32;
        let off_a = (rng.below(3) as i16) - 2; // -2..0
        let off_b = (rng.below(3) as i16) - 2;
        let y_in: Vec<u8> = (0..DEBLOCK_LY_SIZE).map(|_| rng.byte()).collect();
        let c_in: Vec<u8> = (0..DEBLOCK_LC_SIZE).map(|_| rng.byte()).collect();
        if compare_deblock(
            &mb_state,
            fe,
            entropy,
            off_a,
            off_b,
            &y_in,
            &c_in,
            &format!("deblock #{iter}"),
        ) {
            fired += 1;
        }
    }
    assert!(
        fired > 50,
        "too few deblock runs actually filtered pixels: {fired}/400"
    );
}

/// C-vs-Rust for fe=1..7 with a filter-firing pattern (random data rarely
/// fires the strong-filter path; this weak-step pattern makes every luma edge
/// fire). Intra state like frame 0 of h264_baseline.
#[test]
fn deblock_fe_bits_firing_pattern() {
    deblock_fe_bits_firing_pattern_body();
}

fn deblock_fe_bits_firing_pattern_body() {
    use DEBLOCK_MB_STATE_LEN as L;
    for &fe in &[1i32, 2, 3, 4, 5, 6, 7] {
        let mut mb_state = vec![0u8; 3 * L];
        for slot in 0..3usize {
            let b = slot * L;
            mb_state[b] = 22;
            mb_state[b + 1] = 22;
            mb_state[b + 2] = 22;
            for k in 0..48 {
                mb_state[b + 10 + k] = 1;
            }
        }
        // Weak steps (3) at every 4th col/row boundary: |p0-q0|=3 < alpha and
        // max(|p1-p0|,|q1-q0|)=0 < beta, so all luma edges fire.
        let mut y_in = vec![0u8; DEBLOCK_LY_SIZE];
        for r in 0..DEBLOCK_LY_ROWS {
            for (c, slot) in y_in[r * DEBLOCK_LY_STRIDE..(r + 1) * DEBLOCK_LY_STRIDE]
                .iter_mut()
                .enumerate()
            {
                *slot = 128 + (((c / 4) % 2) + ((r / 4) % 2)) as u8 * 3;
            }
        }
        // Chroma: steps at every 2nd buffer-row/col boundary.
        let mut c_in = vec![0u8; DEBLOCK_LC_SIZE];
        for r in 0..DEBLOCK_LC_ROWS {
            for (c, slot) in c_in[r * DEBLOCK_LC_STRIDE..(r + 1) * DEBLOCK_LC_STRIDE]
                .iter_mut()
                .enumerate()
            {
                *slot = 64 + (((c / 2) % 2) + ((r / 2) % 2)) as u8 * 3;
            }
        }
        let (r_y, r_c) =
            crate::rust::deblock::run_rust_deblock(&mb_state, fe, 0, 0, 0, &y_in, &c_in);
        assert_ne!(
            r_y, y_in,
            "fe={fe}: no luma pixels filtered — pattern not firing"
        );
        let mut data = Vec::with_capacity(r_y.len() + r_c.len());
        data.extend_from_slice(&r_y);
        data.extend_from_slice(&r_c);
        let key = format!("deblock-fe fe={fe}");
        goldens::assert_golden(&key, &data);
        goldens::record(&key, &data);
    }
}

#[test]
fn deblock_structured() {
    deblock_structured_body();
}

fn deblock_structured_body() {
    use DEBLOCK_MB_STATE_LEN as L;
    let mut rng = XorShift64Star(0x510c_7a5e_deb1);
    // A flat-sides step pattern (constant within each 4x4 block, stepped
    // across every boundary) so internal and boundary edges all cross the
    // alpha/beta gates and actually filter (see the y_in note below).
    let mut y_in = vec![0u8; DEBLOCK_LY_SIZE];
    for r in 0..DEBLOCK_LY_ROWS {
        for c in 0..DEBLOCK_LY_STRIDE {
            // Flat within each 4x4 block, stepped across every 4x4 boundary
            // (+3 per column block, +2 per row block): at QP=22/off=-1 the
            // tables give alpha'=8, beta'=3, so |p0-q0| <= 3 < 8 with flat
            // sides (max(|p1-p0|,|q1-q0|)=0 < 3) makes every coded edge fire
            // the soft filter with a nonzero correction. Monotonic (no modulo
            // wrap: a wrapped step would exceed alpha and gate the edge off).
            y_in[r * DEBLOCK_LY_STRIDE + c] = (128 + (r / 4) * 2 + (c / 4) * 3) as u8;
        }
    }
    let c_in: Vec<u8> = (0..DEBLOCK_LC_SIZE).map(|_| rng.byte()).collect();

    // Profiles: (name, current_inter, neighbor_inter, ts8x8, all_nC, expect_fire).
    // "mixed nC" uses the k%3 pattern below (all_nC=false routes there).
    //
    // NOTE: for this input the SOFT and HARD boundary kernels produce
    // identical output (a step-3 edge between flat 4x4 blocks maps to
    // p1/p0/q0/q1 = [v+1, v+1, v+2, v+2] under both — verified by probe),
    // so the "both-inter SOFT", "cur-intra HARD" and "all-intra intra-branch"
    // goldens collide per fe. That is expected branch-selection coverage, not
    // a bug; the kernels are pinned apart by the random-pixel
    // `deblock-inter #N` fuzz (soft) and the all-intra stream goldens (hard).
    let cases: [(&str, u8, u8, u8, bool, bool); 6] = [
        ("both-inter SOFT", 1, 1, 0, true, true),
        ("cur-intra HARD", 0, 1, 0, true, true),
        ("all-intra intra-branch", 0, 0, 0, true, true),
        ("inter ts8x8", 1, 1, 1, true, true),
        ("no coded blocks", 1, 1, 0, false, false),
        ("mixed nC", 1, 1, 0, false, true),
    ];
    for (ci, &(name, cinter, ninter, ts, allnc, expect_fire)) in cases.iter().enumerate() {
        for &fe in &[1i32, 2, 3] {
            let mut mb_state = vec![0u8; 3 * L];
            for slot in 0..3usize {
                let b = slot * L;
                mb_state[b] = 22; // qy
                mb_state[b + 1] = 22; // qcb
                mb_state[b + 2] = 22; // qcr
                mb_state[b + 3] = if slot == 2 { cinter } else { ninter };
                mb_state[b + 9] = ts;
                if allnc {
                    for k in 0..48 {
                        mb_state[b + 10 + k] = 1;
                    }
                } else if ci == 5 {
                    for k in 0..48 {
                        mb_state[b + 10 + k] = (k % 3) as u8; // mixed pattern
                    }
                }
            }
            let fired = compare_deblock(
                &mb_state,
                fe,
                0,
                -1,
                -1,
                &y_in,
                &c_in,
                &format!("{name} fe={fe}"),
            );
            // Coded-block profiles must actually filter pixels (the flat-sides
            // step pattern crosses the alpha/beta gates); bS=0 everywhere
            // ("no coded blocks") must stay a no-op.
            assert_eq!(
                fired, expect_fire,
                "{name} fe={fe}: filter firing unexpected"
            );
        }
    }
}

// ---- Tier C: mvpred ---------------------------------------------------------

/// Random mv component kept in [-100, 80]: final mvs (mvp+mvd / temporal
/// scale) then stay within [-200, 160] quarter-pel, so every decode_inter MC
/// fetch inside the C oracle stays inside its scratch plane.
fn mvpred_mv_comp(rng: &mut XorShift64Star) -> i16 {
    (rng.below(181) as i32 - 100) as i16
}

fn mvpred_mv_pair_le(rng: &mut XorShift64Star) -> [u8; 4] {
    let p = (mvpred_mv_comp(rng) as i32) | ((mvpred_mv_comp(rng) as i32) << 16);
    p.to_le_bytes()
}

#[test]
fn mvpred_fuzz() {
    mvpred_fuzz_body();
}

fn mvpred_fuzz_body() {
    use MVPRED_MB_LEN as L;
    let mut rng = XorShift64Star(0x2690_db10_c7a5);
    for iter in 0..400usize {
        let op = rng.below(13) as i32;
        let mut inb = vec![0u8; MVPRED_IN_LEN];
        // 6 macroblocks: current, A, B, C, D, Col.
        for slot in 0..6usize {
            let b = slot * L;
            for i in 0..8 {
                inb[b + i] = rng.byte(); // refIdx: any byte (mask/compare only)
            }
            for i in 0..32 {
                inb[b + 8 + 4 * i..b + 12 + 4 * i].copy_from_slice(&mvpred_mv_pair_le(&mut rng));
            }
            for i in 0..8 {
                // current-MB refPic flows straight into decode_inter's
                // samples_buffers[refPic] index; keep it a valid picture.
                inb[b + 136 + i] = if slot == 0 {
                    rng.below(32) as u8
                } else {
                    rng.byte()
                };
            }
            let ie = if slot == 5 && op == 12 {
                // The C temporal do-while peels flags via
                // extract_neighbours(inter_eqs >> 2i): on this BMI2 build
                // _pext_u32(f, 0x27) can return up to 39 (OOB on masks[16]),
                // so every 2-bit block code of col's inter_eqs must be <= 1
                // (all odd bits clear). 8x8 inference ORs in 0x1b1b1b1b, which
                // sets code high bits, so with it on the flags must be zero
                // (the loop is skipped entirely).
                rng.next() as u32 & 0x5555_5555
            } else {
                rng.next() as u32
            };
            inb[b + 144..b + 148].copy_from_slice(&ie.to_le_bytes());
        }
        let o = 6 * L;
        for i in 0..48 {
            inb[o + i] = rng.byte(); // unavail4x4
        }
        for i in 0..64 {
            inb[o + 48 + i] = rng.below(32) as u8; // RefPicList[2][32]
        }
        for i in 0..32 {
            inb[o + 112 + i] = rng.below(32) as u8; // MapPicToList0
        }
        for i in 0..32 {
            let d = (rng.below(256) as i32 - 128) as i16;
            inb[o + 144 + 2 * i..o + 146 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        inb[o + 208] = rng.below(2) as u8; // col_short_term
        let infer8x8 = rng.below(2) as u8;
        inb[o + 209] = infer8x8; // direct_8x8_inference_flag
        inb[o + 210] = rng.below(2) as u8; // direct_spatial_mv_pred_flag
        inb[o + 211..o + 215].copy_from_slice(&mvpred_mv_pair_le(&mut rng)); // mvd{dx,dy}
        let direct_flags: u32 = if op == 12 {
            if infer8x8 != 0 {
                0
            } else {
                // Derive (Col inter_eqs, direct_flags) consistently from the
                // same per-8x8 sub_mb_types, mirroring slice.c:1192. Arbitrary
                // direct_flags make the temporal do-while's `masks[type] << i`
                // XOR set bits >= 16, so a later ctz yields i>=16 and reads
                // refPic[i8x8] out of bounds (SIGSEGV). Consistent derivation
                // keeps every mask within its run (verified 0/2M unsafe).
                const F: [u32; 13] = [
                    0, 0x00001, 0x10000, 0x10001, 0x00005, 0x00003, 0x50000, 0x30000, 0x50005,
                    0x30003, 0x0000f, 0xf0000, 0xf000f,
                ];
                const E: [u8; 13] = [
                    0, 0x1b, 0x1b, 0x1b, 0x11, 0x0a, 0x11, 0x0a, 0x11, 0x0a, 0, 0, 0,
                ];
                let mut mvd = 0u32;
                let mut eb = [0u8; 4];
                for (b, slot) in eb.iter_mut().enumerate() {
                    let t = rng.below(13) as usize;
                    mvd |= F[t] << (4 * b);
                    *slot = E[t];
                }
                let ie = u32::from_le_bytes(eb);
                inb[5 * L + 144..5 * L + 148].copy_from_slice(&ie.to_le_bytes());
                ((!((mvd & 0xffff) | (mvd >> 16))) & 0x1111) * 0xf000f
            }
        } else if op == 11 {
            // slice.c: (~(...)&0x1111) * 0xf000f, or the all-direct 0xffffffff.
            if rng.below(4) == 0 {
                0xffffffff
            } else {
                (rng.below(16) as u32) * 0xf000f
            }
        } else {
            0
        };
        inb[o + 215..o + 219].copy_from_slice(&direct_flags.to_le_bytes());

        // Spatial-direct corner coverage: unavailable neighbours (refIdx -1,
        // XOR-zero path) and near-zero collocated mvs (colZeroFlags path).
        if op == 11 && rng.below(8) == 0 {
            let slots: &[usize] = match rng.below(4) {
                0 => &[1, 2, 3, 4], // all unavailable
                1 => &[3],
                2 => &[1, 3],
                _ => &[2, 3],
            };
            for &slot in slots {
                for i in 0..8 {
                    inb[slot * L + i] = -1i8 as u8;
                }
            }
        }
        if op == 11 && rng.below(4) == 0 {
            let b = 5 * L;
            for i in 0..32 {
                let p = (rng.below(3) as i32 - 1) | ((rng.below(3) as i32 - 1) << 16);
                inb[b + 8 + 4 * i..b + 12 + 4 * i].copy_from_slice(&p.to_le_bytes());
            }
        }

        let r_out = crate::rust::mvpred::run_rust_mvpred(&inb, op);
        let key = format!("mvpred #{iter} op={op}");
        goldens::assert_golden(&key, &r_out);
        goldens::record(&key, &r_out);
    }
}

/// Re-run every golden-producing body with collection armed (for
/// `regenerate_goldens`).
pub(crate) fn golden_entries() -> Vec<(String, String)> {
    goldens::collect(|| {
        cavlc_fuzz_body();
        cavlc_structured_body();
        cabac_fuzz_body();
        cabac_init_full_domain_body();
        intra_fuzz_all_modes_body();
        intra_structured_body();
        residual_fuzz_body();
        transform_dc_fuzz_body();
        inter_fuzz_all_modes_body();
        inter_fuzz_weighted_body();
        chroma_fuzz_all_fracs_body();
        chroma_fuzz_weighted_body();
        residual_full_qp_domain_body();
        deblock_fuzz_body();
        deblock_fe_bits_firing_pattern_body();
        deblock_structured_body();
        mvpred_fuzz_body();
    })
}
