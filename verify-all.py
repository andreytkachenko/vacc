#!/usr/bin/env python3
"""verify-all.py — full hardware-decoder verification matrix (300 frames).

For every (sample, backend) combination this:
  1. decodes N display frames with the unified `decode` example binary
     (canonical planar YUV dumped per frame via -o),
  2. decodes the same N frames with ffmpeg into rawvideo in the sample's
     NATIVE pixel format (software decode = reference),
  3. byte-compares every frame (exactness is the primary metric; first
     diff location, max xor-diff and luma PSNR are reported for misses).

Backends: vulkan | nvdec | vaapi — all via target/release/examples/decode.

Usage:
  python3 verify-all.py                        # full matrix, all frames
  python3 verify-all.py --max-frames 30        # quick smoke
  python3 verify-all.py --backends vulkan --samples h264_main,hevc
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
EXE = ROOT / "target/release/examples/decode"
WORK = Path("/tmp/verify_all")
NFRAMES_DEFAULT = 300

# (filename, ffmpeg reference pixel format — native depth/chroma, planar)
SAMPLES = [
    ("h264_baseline.h264",         "yuv420p"),
    ("h264_constrained_baseline.h264", "yuv420p"),
    ("h264_main.h264",             "yuv420p"),
    ("h264_high.h264",             "yuv420p"),
    ("h264_tC.h264",               "yuv420p"),
    ("h264_tD.h264",               "yuv420p"),
    ("h264_tN.h264",               "yuv420p"),
    ("h264_tW.h264",               "yuv420p"),
    ("h264_xallI.h264",            "yuv420p"),
    ("h264_xfd.h264",              "yuv420p"),
    ("h264_high10.h264",           "yuv420p10le"),
    ("h264_high422.h264",          "yuv422p"),
    ("h264_high444.h264",          "yuv444p"),
    ("h265_main.h265",             "yuv420p"),
    ("h265_cra.h265",              "yuv420p"),
    ("h265_msp.h265",              "yuv420p"),
    ("h265_main10.h265",           "yuv420p10le"),
    ("vp9_profile0.ivf",           "yuv420p"),
    ("vp9_profile1_444.ivf",       "yuv444p"),
    ("vp9_profile1.ivf",           "yuv420p10le"),
    ("vp9_profile2.ivf",           "yuv420p12le"),
    ("av1_main.ivf",               "yuv420p"),
    ("av1_high.ivf",               "yuv420p10le"),
    ("av1_professional.ivf",       "yuv422p10le"),
]

BACKENDS = ["vulkan", "nvdec", "vaapi"]

# (sample, backend) cells where the hardware/driver genuinely cannot decode
# the stream — NOT a bug in our decoders. Evidence (verified 2026-08-31 on
# RTX 3060 GA106 + Meteor Lake iGPU):
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

# pix_fmt -> (plane ratio vs luma, bytes per sample, bits)
PXFMT = {
    "yuv420p":      (1.5, 1, 8),
    "yuv420p10le":  (1.5, 2, 10),
    "yuv420p12le":  (1.5, 2, 12),
    "yuv422p":      (2.0, 1, 8),
    "yuv422p10le":  (2.0, 2, 10),
    "yuv444p":      (3.0, 1, 8),
    "yuv444p10le":  (3.0, 2, 10),
}


def frame_size(pixfmt, w, h):
    ratio, bps, _ = PXFMT[pixfmt]
    return int(round(w * h * ratio)) * bps


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def _num(s):
    m = re.search(r"frame_(\d+)$", Path(s).stem)
    return int(m.group(1)) if m else 0


def run_backend(sample_path, backend, nframes, workdir):
    """Run the unified decode example; return (n_frames_dumped, err_or_none)."""
    for f in workdir.glob("*.yuv"):
        f.unlink()
    r = subprocess.run(
        [str(EXE), "-b", backend, "-i", str(sample_path), "-n", str(nframes),
         "-o", str(workdir)],
        cwd=str(workdir), capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return 0, "rc=%d: %s" % (r.returncode, " | ".join(tail))
    files = sorted(workdir.glob("*.yuv"), key=_num)
    if not files:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return 0, "no output yuv: %s" % " | ".join(tail)
    return len(files), None


def _input_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_reference(sample_path, ref_pixfmt, nframes, w, h):
    """Decode reference with ffmpeg to rawvideo; cached per (content, fmt, N)."""
    ih = _input_hash(Path(sample_path))
    cache = WORK / "ref" / f"{Path(sample_path).stem}__{ih}__{ref_pixfmt}__{nframes}.raw"
    if not cache.exists() or cache.stat().st_size != frame_size(ref_pixfmt, w, h) * nframes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(sample_path), "-frames:v", str(nframes),
                            "-f", "rawvideo", "-pix_fmt", ref_pixfmt, str(cache)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[-2:] or ["ffmpeg failed"]
    return cache.read_bytes(), None


def luma_psnr(a, b, w, h, bits):
    """PSNR on the luma plane (first w*h*bps bytes of each frame)."""
    bps = 2 if bits > 8 else 1
    n = w * h
    peak = (1 << bits) - 1
    if bps == 1:
        mse = sum((x - y) * (x - y) for x, y in zip(a[:n], b[:n])) / n
    else:
        va = [int.from_bytes(a[i:i + 2], "little") for i in range(0, n * 2, 2)]
        vb = [int.from_bytes(b[i:i + 2], "little") for i in range(0, n * 2, 2)]
        mse = sum((x - y) * (x - y) for x, y in zip(va, vb)) / n
    if mse == 0:
        return float("inf")
    return 10 * math.log10(peak * peak / mse)


def compare_one(sample, ref_pixfmt, backend, nframes):
    cfg = PXFMT[ref_pixfmt]
    sample_path = SAMPLES_DIR / sample
    if not sample_path.exists():
        return dict(status="MISSING_SAMPLE")
    if (sample, backend) in HW_UNSUPPORTED:
        return dict(status="HW_UNSUPPORTED")

    w, h = probe(sample_path)
    fsize = frame_size(ref_pixfmt, w, h)

    workdir = WORK / "out" / f"{Path(sample).stem}__{backend}"
    workdir.mkdir(parents=True, exist_ok=True)
    n_got, err = run_backend(sample_path, backend, nframes, workdir)
    if err:
        return dict(status="BACKEND_FAIL", detail=err[:200])

    ref, rerr = get_reference(sample_path, ref_pixfmt, nframes, w, h)
    if rerr:
        return dict(status="REF_FAIL", detail=" ".join(rerr)[:200])

    files = sorted(workdir.glob("*.yuv"), key=_num)
    count = min(len(files), nframes)
    exact = 0
    first_diff = None
    worst = (0, 0)          # (xor diff, byte offset within frame)
    min_psnr = float("inf")
    for i in range(count):
        fdata = files[i].read_bytes()
        rchunk = ref[i * fsize:(i + 1) * fsize]
        if len(fdata) != fsize:
            if first_diff is None:
                first_diff = (i, "size %d vs %d" % (len(fdata), fsize))
            continue
        if fdata == rchunk:
            exact += 1
            continue
        if first_diff is None:
            db = sum(1 for x, y in zip(fdata, rchunk) if x != y)
            off = next(j for j, (x, y) in enumerate(zip(fdata, rchunk)) if x != y)
            first_diff = (i, "%d/%d bytes differ @%d" % (db, fsize, off))
        for j, (x, y) in enumerate(zip(fdata, rchunk)):
            d = x ^ y
            if d > worst[0]:
                worst = (d, j)
                if d == 255:
                    break
        p = luma_psnr(fdata, rchunk, w, h, cfg[2])
        min_psnr = min(min_psnr, p)

    short = n_got < nframes
    if count == 0:
        return dict(status="NO_FRAMES")
    if exact == count and not short:
        status = "PASS"
    elif short and exact == count:
        # Sample shorter than requested; every available frame verified exact.
        status = "SHORT"
    elif min_psnr >= 45:
        status = "NEAR"
    else:
        status = "FAIL"
    res = dict(status=status, exact=exact, total=nframes, got=n_got,
               min_psnr=(round(min_psnr, 2) if min_psnr != float("inf") else None),
               w=w, h=h, ref_fmt=ref_pixfmt)
    if short:
        res["detail"] = "decoded %d/%d frames" % (n_got, nframes)
    if first_diff:
        res["first_diff"] = "frame %d: %s" % first_diff
    if worst[0]:
        res["worst"] = "xor %d @byte %d" % worst
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=NFRAMES_DEFAULT)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--samples", default="")  # comma list of filename substrings
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    sample_filter = [s.strip() for s in args.samples.split(",") if s.strip()]

    rows = []  # (sample, backend, result)
    for fname, pixfmt in SAMPLES:
        if sample_filter and not any(f in fname for f in sample_filter):
            continue
        for backend in backends:
            print(f"[{backend}] {fname} ({pixfmt}) ...", flush=True)
            try:
                res = compare_one(fname, pixfmt, backend, args.max_frames)
            except Exception as e:
                res = dict(status="ERROR", detail=repr(e)[:160])
            rows.append((fname, backend, res))

    print("\n" + "=" * 92)
    print(f"VERIFICATION MATRIX  (max_frames={args.max_frames})")
    print("=" * 92)
    hdr = f"{'sample':<28}" + "".join(f"{b:>16}" for b in backends)
    print(hdr)
    print("-" * len(hdr))
    pass_tot = fail_tot = 0
    for fname, pixfmt in SAMPLES:
        if sample_filter and not any(f in fname for f in sample_filter):
            continue
        line = f"{fname:<28}"
        for backend in backends:
            res = next(r for f, b, r in rows if f == fname and b == backend)
            st = res["status"]
            if st == "PASS":
                cell = f"{res['exact']}/{res['total']} OK"
                pass_tot += 1
            elif st == "SHORT":
                # All available frames verified exact; sample shorter than N.
                cell = f"S({res.get('got')}/{res['total']})"
            elif st in ("NEAR", "FAIL"):
                cell = f"{st[:4]}~{res.get('min_psnr')}"
                fail_tot += 1
            elif st == "HW_UNSUPPORTED":
                cell = "HW-n/a"
            else:
                cell = st[:16]
                if st in ("BACKEND_FAIL", "REF_FAIL", "ERROR", "NO_FRAMES"):
                    fail_tot += 1
            line += f"{cell:>16}"
        print(line)
    print("-" * len(hdr))
    hw_na = sum(1 for _, _, r in rows if r["status"] == "HW_UNSUPPORTED")
    print(f"\nPASS cells: {pass_tot}   non-PASS: {fail_tot}   HW-unsupported (n/a): {hw_na}")

    bad = [(f, b, r) for f, b, r in rows
           if r["status"] in ("NEAR", "FAIL", "BACKEND_FAIL",
                              "REF_FAIL", "ERROR", "NO_FRAMES")]
    if bad:
        print("\n--- non-PASS detail ---")
        for f, b, r in bad:
            extra = r.get("detail") or r.get("first_diff") or ""
            if r.get("worst"):
                extra += (("; " + extra) if extra else "") + " " + r["worst"]
            print(f"  {b:<8} {f:<28} {r['status']:<10} {extra}")


if __name__ == "__main__":
    main()
