// Standalone bit-by-bit slice-header tracer for H.264.
// Reads the RBSP bits of a target picture (matched by frame_num) and parses the
// slice header field-by-field with a manual bit index, printing each field,
// value, and exact bit range. Avoids the BitReader position() issues.
use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

struct Bits<'a> {
    b: &'a [u8],
    i: usize, // bit index
}

impl<'a> Bits<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn bit(&mut self) -> u32 {
        let v = ((self.b[self.i / 8] >> (7 - (self.i % 8))) & 1) as u32;
        self.i += 1;
        v
    }
    fn bits(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit();
        }
        v
    }
    fn ue(&mut self) -> u32 {
        let mut z = 0u32;
        while self.bit() == 0 {
            z += 1;
            if z > 30 {
                break;
            }
        }
        let v = self.bits(z);
        (1u32 << z) - 1 + v
    }
    fn se(&mut self) -> i32 {
        let c = self.ue() as i32;
        if c % 2 == 0 { -(c / 2) } else { (c + 1) / 2 }
    }
    fn show(&self, start: usize, len: usize) -> String {
        let mut s = String::new();
        for k in start..start + len {
            s.push(if (self.b[k / 8] >> (7 - (k % 8))) & 1 == 1 { '1' } else { '0' });
        }
        s
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "assets/test_baseline.h264".into());
    let want_fn: u32 = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let data = std::fs::read(&path).unwrap();
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);
    let mut found = false;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() {
                    break;
                }
                for sl in &slices {
                    let nal = &sl.nal_data;
                    // EPB removal
                    let mut rbsp = Vec::new();
                    let mut i = 1;
                    while i < nal.len() {
                        if i + 2 < nal.len() && nal[i] == 0 && nal[i + 1] == 0 && nal[i + 2] == 3 {
                            rbsp.push(0);
                            rbsp.push(0);
                            i += 3;
                        } else {
                            rbsp.push(nal[i]);
                            i += 1;
                        }
                    }
                    let nal_hdr = nal[0];
                    let nal_ref_idc = (nal_hdr >> 5) & 3;
                    let nal_unit_type = nal_hdr & 0x1f;
                    // quick frame_num peek: skip first_mb(ue), slice_type(ue), pps_id(ue) then 4 bits
                    let mut peek = Bits::new(&rbsp);
                    let _ = peek.ue();
                    let _ = peek.ue();
                    let _ = peek.ue();
                    let peek_fn = peek.bits(4);
                    if peek_fn == want_fn {
                        println!(
                            "=== pic with frame_num={} nal_hdr=0x{:02x} ref_idc={} unit_type={} ===",
                            want_fn, nal_hdr, nal_ref_idc, nal_unit_type
                        );
                        let mut r = Bits::new(&rbsp);
                        let s = r.i;
                        let first_mb = r.ue();
                        println!("first_mb_in_slice ue = {}  bits[{}:{}]={}", first_mb, s, r.i, r.show(s, r.i - s));
                        let s = r.i;
                        let st = r.ue();
                        println!("slice_type ue = {} (mod5={})  bits[{}:{}]={}", st, st % 5, s, r.i, r.show(s, r.i - s));
                        let s = r.i;
                        let pps = r.ue();
                        println!("pps_id ue = {}  bits[{}:{}]={}", pps, s, r.i, r.show(s, r.i - s));
                        let s = r.i;
                        let fn_ = r.bits(4);
                        println!("frame_num u(4) = {}  bits[{}:{}]={}", fn_, s, r.i, r.show(s, r.i - s));
                        // assume frame_mbs_only, poc_type=2, no redundant, no colour_plane
                        let s = r.i;
                        let flag = r.bit();
                        println!(
                            "adaptive_ref_pic_marking_mode_flag = {}  bits[{}:{}]={}",
                            flag, s, r.i, r.show(s, r.i - s)
                        );
                        if flag == 1 {
                            loop {
                                let s = r.i;
                                let op = r.bits(4);
                                print!("  MMCO op u(4) = {}  bits[{}:{}]={}", op, s, r.i, r.show(s, r.i - s));
                                if op == 0 {
                                    println!("  (END)");
                                    break;
                                }
                                match op {
                                    1 | 4 => {
                                        let s2 = r.i;
                                        let v = r.ue();
                                        println!(" | value ue = {}  bits[{}:{}]={}", v, s2, r.i, r.show(s2, r.i - s2));
                                    }
                                    2 | 3 | 5 | 6 | 7 | 9 => {
                                        let s2 = r.i;
                                        let v = r.bits(5);
                                        println!(" | value u(5) = {}  bits[{}:{}]={}", v, s2, r.i, r.show(s2, r.i - s2));
                                    }
                                    8 => println!(" | (no value)"),
                                    _ => {
                                        println!("  *** INVALID OP {} ***", op);
                                        break;
                                    }
                                }
                            }
                        }
                        // continue: num_ref_idx etc.
                        let s = r.i;
                        let override_flag = r.bit();
                        println!("num_ref_idx_active_override_flag = {}  bits[{}:{}]={}", override_flag, s, r.i, r.show(s, r.i - s));
                        if override_flag == 1 {
                            let s2 = r.i;
                            let n0 = r.ue();
                            println!("  num_ref_idx_l0_active_minus1 ue = {}  bits[{}:{}]={}", n0, s2, r.i, r.show(s2, r.i - s2));
                        }
                        let s = r.i;
                        let rplm = r.bit();
                        println!("ref_pic_list_modification_flag = {}  bits[{}:{}]={}", rplm, s, r.i, r.show(s, r.i - s));
                        let s = r.i;
                        let cabac_init = r.bit();
                        println!("cabac_init_flag = {}  bits[{}:{}]={}", cabac_init, s, r.i, r.show(s, r.i - s));
                        let s = r.i;
                        let qp = r.se();
                        println!("slice_qp_delta se = {}  bits[{}:{}]={}", qp, s, r.i, r.show(s, r.i - s));
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("err: {e}");
                break;
            }
        }
    }
    if !found {
        eprintln!("frame_num {} not found", want_fn);
    }
}
