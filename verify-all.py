#!/usr/bin/env python3
"""verify-all.py — full hardware-decoder verification matrix (hash-based).

For every applicable (sample, backend) combination this:
  1. decodes N frames with the project's unified `decode` example binary,
     which prints per display frame an FNV-1a 64 hash of the canonical
     planar YUV pixels,
  2. decodes the same N frames with ffmpeg into the sample's native pix_fmt,
  3. computes the IDENTICAL canonical hash in Python and compares frame by
     frame (hash equality == byte-exact pixels).

Canonical form (must stay in sync with crates/examples/src/decode.rs):
packed planar Y+U+V rows cropped to the display size; 16-bit samples are
bottom-justified (the decoder shifts P016 top-justified values >> 6);
semi-planar UV (P010/P012) is de-interleaved into planar U then V.

Backends / binary (see crates/examples/src/decode.rs):
  vulkan   H.264/H.265/VP9
  vaapi    H.264/H.265/VP9
  nvdec    H.264/H.265/VP9 (requires an NVIDIA GPU)

NOTE: the example binary is NOT built automatically. Run
    cargo build --release --examples
before invoking this script, or every result silently reflects stale code.

Usage:
  python3 verify-all.py                 # full matrix, 300 frames
  python3 verify-all.py --max-frames 30 # quick smoke
  python3 verify-all.py --backends vulkan,vaapi --samples hevc_main.h265
"""
import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "assets" / "samples"
EX = ROOT / "target/release/examples"
WORK = Path("/tmp/verify_all")
NFRAMES_DEFAULT = 300

# (filename, codec, pix_fmt) — the committed sample set in assets/samples/.
SAMPLES = [
    ("h264_baseline.h264",          "h264", "yuv420p"),
    ("h264_constrained_baseline.h264", "h264", "yuv420p"),
    ("h264_main.h264",              "h264", "yuv420p"),
    ("h264_high.h264",              "h264", "yuv420p"),
    ("h264_tC.h264",                "h264", "yuv420p"),
    ("h264_tD.h264",                "h264", "yuv420p"),
    ("h264_tN.h264",                "h264", "yuv420p"),
    ("h264_tW.h264",                "h264", "yuv420p"),
    ("h264_xallI.h264",             "h264", "yuv420p"),
    ("h264_xfd.h264",               "h264", "yuv420p"),
    ("h264_high10.h264",            "h264", "yuv420p10le"),
    ("h264_high422.h264",           "h264", "yuv422p"),
    ("h264_high444.h264",           "h264", "yuv444p"),
    ("h265_main.h265",              "hevc", "yuv420p"),
    ("h265_cra.h265",               "hevc", "yuv420p"),
    ("h265_msp.h265",               "hevc", "yuv420p"),
    ("h265_main10.h265",            "hevc", "yuv420p10le"),
    ("vp9_profile0.ivf",            "vp9",  "yuv420p"),
    ("vp9_profile1_444.ivf",        "vp9",  "yuv444p"),
    ("vp9_profile1.ivf",            "vp9",  "yuv420p10le"),
    ("vp9_profile2.ivf",            "vp9",  "yuv420p12le"),
    ("av1_main.ivf",                "av1",  "yuv420p"),
    ("av1_high.ivf",                "av1",  "yuv420p10le"),
    ("av1_professional.ivf",        "av1",  "yuv422p10le"),
    # aomenc --film-grain-table: SPS film_grain_params_present=1 and a grain
    # block at the end of every showable frame header. Exercises parser issue
    # 3 (film_grain_params) end-to-end. Decoded without grain application on
    # all backends (apply_grain forced 0) to stay pixel-identical to ffmpeg.
    ("av1_grain.ivf",               "av1",  "yuv420p"),
    # rav1e (ffmpeg -c:v librav1e -rav1e-params segmentation_temporal=1:
    # segmentation_spatial=1): SPS frame_id_numbers_present_flag=1, ~half the
    # pictures are show-existing-frame OBUs, and two frames update signed
    # segmentation feature data (ALT_Q). Exercises parser issues 4
    # (su(1+bits) signed features), 7 (per-ref delta_frame_id_minus_1) and 9
    # (show-existing display_frame_id) end-to-end.
    ("av1_seg.ivf",                 "av1",  "yuv420p"),
]

# backend -> supported codecs (all use the single unified `decode` binary)
BACKENDS = {
    "vulkan": {"h264", "hevc", "vp9", "av1"},
    "vaapi":  {"h264", "hevc", "vp9", "av1"},
    "nvdec":  {"h264", "hevc", "vp9", "av1"},
}

# (sample, backend) cells where the hardware/driver genuinely cannot decode
# the stream — NOT a bug in our decoders. Evidence (verified 2026-08-31 on
# RTX 3060 GA106 + Meteor Lake iGPU; AV1 re-verified 2026-09-08 on GA106):
HW_UNSUPPORTED = {
    # H.264 High 10-bit: GA106 NVDEC caps report it unsupported (create fails
    # with 801); MTL Vulkan driver rejects the spec-legal profile+depth combo
    # with ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR; iHD VAAPI rejects
    # RTFormat=YUV420_10 for H.264 (its AVC caps list is 8-bit only).
    ("h264_high10.h264", "nvdec"),
    ("h264_high10.h264", "vulkan"),
    ("h264_high10.h264", "vaapi"),
    # GA106 NVDEC: cuvidGetDecoderCaps reports H.264 as 8-bit 4:2:0 only
    # (10-bit, 4:2:2 and 4:4:4 all "unsupported"; create fails with 801).
    ("h264_high422.h264", "nvdec"),
    ("h264_high444.h264", "nvdec"),
    # H.264 4:2:2 / 4:4:4 on Vulkan Video: the spec exposes no 4:2:2/4:4:4
    # formats for H.264 decode (only HEVC has them); the MTL driver rejects
    # the profile with ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR.
    ("h264_high422.h264", "vulkan"),
    ("h264_high444.h264", "vulkan"),
    # iHD (Gen12/MTL): AVC VLD pipeline is NV12-only. Config creation with
    # RTFormat=YUV422 is accepted (lenient caps fallback) but decode fails at
    # vaEndPicture; RTFormat=YUV444 config is rejected outright. No-attr
    # configs expose only NV12 surface pixfmts for every H.264 profile.
    ("h264_high422.h264", "vaapi"),
    ("h264_high444.h264", "vaapi"),
    # VP9 4:4:4: Vulkan Video spec exposes only 4:2:0 formats for VP9 decode
    # and the iGPU driver rejects the 4:4:4 profile with
    # ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR; GA106 NVDEC caps report
    # VP9 P1 4:4:4 unsupported.
    ("vp9_profile1_444.ivf", "vulkan"),
    ("vp9_profile1_444.ivf", "nvdec"),
    # AV1 Professional (profile 2, 10-bit 4:2:2): unsupported everywhere —
    # iHD caps list only AV1 Profile0/1, GA106 NVDEC caps say AV1 is
    # main/high 4:2:0 only, and the Vulkan driver rejects profile 2.
    ("av1_professional.ivf", "vulkan"),
    ("av1_professional.ivf", "nvdec"),
    ("av1_professional.ivf", "vaapi"),
}

# pix_fmt -> (bytes per frame / (w*h), bytes per sample, semi-planar UV?)
# NOTE: ffmpeg's yuv420pNNle are PLANAR (Y, U, V separate planes); only the
# p0NNle names are semi-planar (Y + interleaved UV).
PXFMT = {
    "yuv420p":      (1.5, 1, False),
    "yuv422p":      (2.0, 1, False),
    "yuv444p":      (3.0, 1, False),
    "yuv420p10le":  (1.5, 2, False),  # planar 10-bit (bottom-justified)
    "yuv420p12le":  (1.5, 2, False),  # planar 12-bit (bottom-justified)
    "p010le":       (1.5, 2, True),   # semi-planar: Y + interleaved UV
    "p012le":       (1.5, 2, True),
    "yuv422p10le":  (2.0, 2, False),
    "yuv444p10le":  (3.0, 2, False),
}

def frame_size(pixfmt, w, h):
    ratio, bps, _semi = PXFMT[pixfmt]
    return int(round(w * h * ratio)) * bps

def probe(path):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",str(path)],
        capture_output=True, text=True)
    w,h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)

def fnv1a64(data: bytes) -> int:
    h = 0xcbf29ce484222325
    m = (1 << 64) - 1
    for b in data:
        h ^= b
        h = (h * 0x00000100000001b3) & m
    return h

def canonicalize(data: bytes, pixfmt: str, w: int, h: int) -> bytes:
    """Normalize ffmpeg rawvideo to the decoder's canonical planar Y+U+V."""
    ratio, bps, semi = PXFMT[pixfmt]
    if not semi:
        # ffmpeg planar layouts (I420 / I210-style / 4:4:4) are already
        # packed planar rows in Y, U, V order.
        return data
    # Semi-planar (P010/P012): Y plane first, then UV interleaved per row.
    y_size = w * h * bps
    uv_row = w * 2 * bps
    y = data[:y_size]
    uv = data[y_size:]
    out = bytearray(y)
    for row in range(0, len(uv), uv_row):
        r = uv[row:row + uv_row]
        out += r[0::2 * bps]  # U samples (every pair)
    for row in range(0, len(uv), uv_row):
        r = uv[row:row + uv_row]
        out += r[bps::2 * bps]  # V samples
    return bytes(out)

def run_backend(sample_path, backend, nframes):
    """Run the unified decode binary; return list of (w, h, hash_hex|None)."""
    binp = EX / "decode"
    if not binp.exists():
        return None, "binary missing (run: cargo build --release --examples)"
    r = subprocess.run([str(binp), "-b", backend, "-i", str(sample_path),
                        "-n", str(nframes)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return None, "rc=%d: %s" % (r.returncode, " | ".join(tail))
    frames = []
    pat = re.compile(r"^frame \d+: pts=\S+ size=(\d+)x(\d+) hash=([0-9a-f]{16}|-)\s*$")
    for line in r.stdout.splitlines():
        m = pat.match(line)
        if not m:
            continue
        frames.append((int(m.group(1)), int(m.group(2)),
                       None if m.group(3) == "-" else int(m.group(3), 16)))
    if not frames:
        return None, "no output frames"
    return frames, None

def _input_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def get_reference(sample_path, pixfmt, nframes, w, h):
    """Decode reference with ffmpeg to rawvideo in pixfmt; cached by content."""
    ih = _input_hash(Path(sample_path))
    cache = WORK / "ref" / f"{Path(sample_path).stem}__{ih}__{pixfmt}.raw"
    if not cache.exists() or cache.stat().st_size != frame_size(pixfmt, w, h) * nframes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i",str(sample_path),"-frames:v",str(nframes),
            "-f","rawvideo","-pix_fmt",pixfmt,str(cache)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None, (r.stderr.strip().splitlines()[-2:] or ["ffmpeg failed"])
    return cache.read_bytes(), None

def compare_one(sample, codec, pixfmt, backend, nframes):
    """Return a result dict for one (sample, backend)."""
    sample_path = SAMPLES_DIR / sample
    if not sample_path.exists():
        return dict(status="MISSING_SAMPLE")
    if (sample, backend) in HW_UNSUPPORTED:
        return dict(status="HW_UNSUPPORTED")
    if pixfmt not in PXFMT:
        return dict(status="SKIP_NO_REF_FMT", detail=pixfmt)

    w, h = probe(sample_path)
    fsize = frame_size(pixfmt, w, h)

    frames, err = run_backend(sample_path, backend, nframes)
    if err:
        return dict(status="BACKEND_FAIL", detail=err[:160])

    ref, rerr = get_reference(sample_path, pixfmt, nframes, w, h)
    if rerr:
        return dict(status="REF_FAIL", detail=" ".join(rerr)[:160])

    count = min(len(frames), nframes)
    exact = 0
    first_diff_frame = None
    for i in range(count):
        fw, fh, fhash = frames[i]
        r0, r1 = i * fsize, (i + 1) * fsize
        rchunk = ref[r0:r1]
        if len(rchunk) != fsize:
            return dict(status="REF_SHORT", detail="reference truncated at frame %d" % i)
        if (fw, fh) != (w, h):
            if first_diff_frame is None:
                first_diff_frame = (i, "size %dx%d vs %dx%d" % (fw, fh, w, h))
            continue
        if fhash is None:
            if first_diff_frame is None:
                first_diff_frame = (i, "no pixel data")
            continue
        rhash = fnv1a64(canonicalize(rchunk, pixfmt, w, h))
        if fhash == rhash:
            exact += 1
        elif first_diff_frame is None:
            first_diff_frame = (i, "pixel hash mismatch")

    got = len(frames)
    short = got < nframes
    if count == 0:
        return dict(status="NO_FRAMES")
    if short:
        status = "SHORT"
    elif exact == count:
        status = "PASS"
    else:
        status = "FAIL"
    res = dict(status=status, exact=exact, total=nframes, got=got)
    if first_diff_frame:
        res["first_diff"] = "frame %d: %s" % first_diff_frame
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=NFRAMES_DEFAULT)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--samples", default="")  # comma list of filenames (substring ok)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    sample_filter = [s.strip() for s in args.samples.split(",") if s.strip()]

    rows = []   # (sample, backend, result)
    for fname, codec, pixfmt in SAMPLES:
        if sample_filter and not any(f in fname for f in sample_filter):
            continue
        for backend in backends:
            if backend not in BACKENDS or codec not in BACKENDS[backend]:
                rows.append((fname, backend, dict(status="N/A")))
                continue
            print(f"[{backend}] {fname} ({pixfmt}) ...", flush=True)
            try:
                res = compare_one(fname, codec, pixfmt, backend, args.max_frames)
            except Exception as e:
                res = dict(status="ERROR", detail=repr(e)[:160])
            rows.append((fname, backend, res))

    # ---- summary matrix ----
    print("\n" + "="*78)
    print(f"VERIFICATION MATRIX  (max_frames={args.max_frames})")
    print("="*78)
    hdr = f"{'sample':<22}" + "".join(f"{b[:11]:>12}" for b in backends)
    print(hdr)
    print("-"*len(hdr))
    by_sample = {}
    for fname, backend, res in rows:
        by_sample.setdefault(fname, {})[backend] = res
    for fname, _, _ in SAMPLES:
        if sample_filter and not any(f in fname for f in sample_filter):
            continue
        cells = []
        for b in backends:
            res = by_sample.get(fname, {}).get(b, dict(status="N/A"))
            st = res["status"]
            if st == "PASS":
                c = f"{res['exact']}/{res['total']}"
            elif st in ("FAIL", "SHORT") and "exact" in res:
                c = f"{res['exact']}/{res['total']}"
            elif st == "HW_UNSUPPORTED":
                c = "hw-unsuppt"[:11]
            else:
                c = st[:11]
            cells.append(f"{c:>12}")
        print(f"{fname:<22}" + "".join(cells))

    hw_na = sum(1 for _, _, r in rows if r["status"] == "HW_UNSUPPORTED")
    n_pass = sum(1 for _, _, r in rows if r["status"] == "PASS")
    print(f"\n{n_pass} PASS, {hw_na} HW-unsupported (skipped by design)")

    # ---- details ----
    print("\nDETAILS")
    for fname, backend, res in rows:
        if res["status"] in ("PASS", "N/A", "HW_UNSUPPORTED"):
            continue
        extra = ""
        if res.get("first_diff"):
            extra += f"  first_diff={res['first_diff']}"
        if res.get("detail"):
            extra += f"  detail={res['detail']}"
        print(f"  {backend:<12} {fname:<22} {res['status']}{extra}")

if __name__ == "__main__":
    main()
