#!/usr/bin/env python3
"""verify-all.py — full hardware-decoder verification matrix.

For every applicable (sample, backend) combination this:
  1. decodes N frames with the project's hardware decoder example binary,
  2. decodes the same N frames with ffmpeg into the *matching* pixel layout,
  3. byte-compares every frame (exactness is the primary metric; PSNR on the
     luma plane is reported as a fallback for near-misses).

A final matrix summarises PASS/FAIL per (sample, backend).

Backends / binaries (see crates/examples/src/):
  vulkan       vulkan_decode        H.264/H.265  planar native depth/chroma
  vulkan_vp9   vulkan_decode_vp9    VP9          always 8-bit planar yuv420p
  vaapi        decode_vaapi         H.264/H.265/VP9 planar native
  nvdec        decode_nvdec         H.264        8-bit planar yuv420p only
  nvdec_h265   decode_nvdec_h265    H.265        4:2:0 NV12/P010 semi-planar, 4:4:4 planar
  nvdec_vp9    decode_nvdec_vp9     VP9          8-bit planar yuv420p

Usage:
  python3 verify-all.py                 # full matrix, 300 frames
  python3 verify-all.py --max-frames 30 # quick smoke
  python3 verify-all.py --backends vulkan,vaapi --samples hevc_main.h265
"""
import argparse
import glob
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
EX = ROOT / "target/release/examples"
WORK = Path("/tmp/verify_all")
NFRAMES_DEFAULT = 300

# (filename, codec, pix_fmt)  — AV1 excluded from verification (HW unsupported).
SAMPLES = [
    ("h264_baseline.h264",  "h264", "yuv420p"),
    ("h264_main.h264",      "h264", "yuv420p"),
    ("h264_high.h264",      "h264", "yuv420p"),
    ("h264_high10.h264",    "h264", "yuv420p10le"),
    ("h264_high422.h264",   "h264", "yuv422p10le"),
    ("h264_high444_8.h264", "h264", "yuv444p"),
    ("h264_high444_10.h264","h264", "yuv444p10le"),
    ("h264_gop1.h264",      "h264", "yuv420p"),
    ("h264_gop100.h264",    "h264", "yuv420p"),
    ("h264_cra.h264",       "h264", "yuv420p"),
    ("hevc_main.h265",      "hevc", "yuv420p"),
    ("hevc_main10.h265",    "hevc", "yuv420p10le"),
    ("hevc_main422.h265",   "hevc", "yuv422p10le"),
    ("hevc_main444_8.h265", "hevc", "yuv444p"),
    ("hevc_main444_10.h265","hevc", "yuv444p10le"),
    ("hevc_gop1.h265",      "hevc", "yuv420p"),
    ("hevc_gop100.h265",    "hevc", "yuv420p"),
    ("hevc_cra.h265",       "hevc", "yuv420p"),
    ("vp9_p0_8bit.ivf",     "vp9",  "yuv420p"),
    ("vp9_p2_10bit.ivf",    "vp9",  "yuv420p10le"),
    ("vp9_p2_12bit.ivf",    "vp9",  "yuv420p12le"),
    ("vp9_gop1.ivf",        "vp9",  "yuv420p"),
    ("vp9_gop100.ivf",      "vp9",  "yuv420p"),
]

# backend -> dict(binary, codecs, min_bitdepth_ok, refpix(fmt)->ffmpeg pix_fmt)
def _vulkan_ref(f):   return f                      # planar native
def _vaapi_ref(f):    return f                      # planar native
def _vp9v_ref(f):     return "yuv420p"              # always 8-bit planar
def _nvdec_ref(f):    return "yuv420p"              # 8-bit only
def _nvdec265_ref(f):
    return {"yuv420p": "nv12", "yuv420p10le": "p010le",
            "yuv444p": "yuv444p", "yuv444p10le": "yuv444p10le"}.get(f, f)
def _nvdecvp9_ref(f): return "yuv420p"

BACKENDS = {
    "vulkan":      dict(bin="vulkan_decode",      codecs={"h264","hevc"}, ref=_vulkan_ref),
    "vulkan_vp9":  dict(bin="vulkan_decode_vp9",  codecs={"vp9"},         ref=_vp9v_ref),
    "vaapi":       dict(bin="decode_vaapi",       codecs={"h264","hevc","vp9"}, ref=_vaapi_ref),
    "nvdec":       dict(bin="decode_nvdec",       codecs={"h264"},        ref=_nvdec_ref,  eight_only=True),
    "nvdec_h265":  dict(bin="decode_nvdec_h265",  codecs={"hevc"},        ref=_nvdec265_ref),
    "nvdec_vp9":   dict(bin="decode_nvdec_vp9",   codecs={"vp9"},         ref=_nvdecvp9_ref),
}

# pix_fmt -> (chroma plane ratio vs luma, bytes per sample)
PXFMT = {
    "yuv420p":      (1.5, 1), "nv12":        (1.5, 1),
    "yuv420p10le":  (1.5, 2), "p010le":      (1.5, 2),
    "yuv420p12le":  (1.5, 2), "p012le":      (1.5, 2),
    "yuv422p10le":  (2.0, 2),
    "yuv444p":      (3.0, 1), "yuv444p10le": (3.0, 2),
}

def frame_size(pixfmt, w, h):
    ratio, bps = PXFMT[pixfmt]
    return int(round(w * h * ratio)) * bps

def probe(path):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",str(path)],
        capture_output=True, text=True)
    w,h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)

def _num(s):
    m = re.search(r"frame_(\d+)", Path(s).stem)
    return int(m.group(1)) if m else 0

def run_backend(sample_path, backend, nframes, workdir):
    """Run a decoder binary; return sorted list of per-frame YUV files."""
    cfg = BACKENDS[backend]
    binp = EX / cfg["bin"]
    if not binp.exists():
        return None, "binary missing"
    for f in workdir.glob("*.yuv"):
        f.unlink()
    env = dict(os.environ)
    # run with explicit max_frames; default out prefixes are known per binary
    r = subprocess.run([str(binp), str(sample_path), str(nframes)],
                       cwd=str(workdir), capture_output=True, text=True, env=env,
                       timeout=600)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return None, "rc=%d: %s" % (r.returncode, " | ".join(tail))
    files = sorted(workdir.glob("*.yuv"), key=_num)
    if not files:
        return None, "no output yuv"
    return files, None

def _input_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_reference(sample_path, ref_pixfmt, nframes, w, h):
    """Decode reference with ffmpeg to rawvideo in ref_pixfmt; cached.

    The cache key includes a hash of the input content: a regenerated sample
    (same name and frame count) must not reuse stale reference pixels.
    """
    ih = _input_hash(Path(sample_path))
    cache = WORK / "ref" / f"{Path(sample_path).stem}__{ih}__{ref_pixfmt}.raw"
    if not cache.exists() or cache.stat().st_size != frame_size(ref_pixfmt,w,h)*nframes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i",str(sample_path),"-frames:v",str(nframes),
            "-f","rawvideo","-pix_fmt",ref_pixfmt,str(cache)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None, (r.stderr.strip().splitlines()[-2:] or ["ffmpeg failed"])
    data = cache.read_bytes()
    return data, None

def luma_psnr(a, b, w, h, bps):
    """PSNR on luma plane. a,b raw frame bytes; luma = first w*h*bps bytes."""
    n = w * h
    if bps == 1:
        sa, sb = a[:n], b[:n]
        mse = sum((x-y)*(x-y) for x,y in zip(sa,sb)) / n
        peak = 255.0
    else:
        import struct
        va = struct.unpack("<%dH" % (n//1), a[:n*2]) if False else None
        # unpack 16-bit LE luma
        va = [int.from_bytes(a[i:i+2],'little') for i in range(0, n*2, 2)]
        vb = [int.from_bytes(b[i:i+2],'little') for i in range(0, n*2, 2)]
        mse = sum((x-y)*(x-y) for x,y in zip(va,vb)) / n
        peak = 1023.0
    if mse == 0:
        return float("inf")
    return 10*math.log10(peak*peak/mse)

def compare_one(sample, codec, pixfmt, backend, nframes):
    """Return a result dict for one (sample, backend)."""
    cfg = BACKENDS[backend]
    sample_path = SAMPLES_DIR / sample
    if not sample_path.exists():
        return dict(status="MISSING_SAMPLE")
    if cfg.get("eight_only") and pixfmt != "yuv420p":
        return dict(status="SKIP_8BIT_ONLY")
    ref_fmt = cfg["ref"](pixfmt)
    if ref_fmt not in PXFMT:
        return dict(status="SKIP_NO_REF_FMT", detail=ref_fmt)

    w, h = probe(sample_path)
    fsize = frame_size(ref_fmt, w, h)

    workdir = WORK / "out" / f"{Path(sample).stem}__{backend}"
    workdir.mkdir(parents=True, exist_ok=True)
    frames, err = run_backend(sample_path, backend, nframes, workdir)
    if err:
        return dict(status="BACKEND_FAIL", detail=err[:160])
    if len(frames) < nframes:
        # decode fewer than requested; compare what we have but flag it
        pass

    ref, rerr = get_reference(sample_path, ref_fmt, nframes, w, h)
    if rerr:
        return dict(status="REF_FAIL", detail=" ".join(rerr)[:160])

    bps = PXFMT[ref_fmt][1]
    count = min(len(frames), nframes)
    exact = 0
    first_diff_frame = None
    min_psnr = float("inf")
    for i in range(count):
        fdata = frames[i].read_bytes()
        r0, r1 = i*fsize, (i+1)*fsize
        rchunk = ref[r0:r1]
        if len(fdata) != fsize:
            # layout mismatch -> record and try luma PSNR on min length
            if first_diff_frame is None:
                first_diff_frame = (i, "size %d vs %d" % (len(fdata), fsize))
            continue
        if fdata == rchunk:
            exact += 1
        else:
            if first_diff_frame is None:
                db = sum(1 for x,y in zip(fdata,rchunk) if x!=y)
                first_diff_frame = (i, "%d/%d bytes differ" % (db, fsize))
            p = luma_psnr(fdata, rchunk, w, h, bps)
            min_psnr = min(min_psnr, p)

    got = len(frames)
    short = got < nframes
    if count == 0:
        return dict(status="NO_FRAMES")
    if short:
        status = "SHORT"
    elif exact == count:
        status = "PASS"
    else:
        status = "NEAR" if min_psnr >= 45 else "FAIL"
    res = dict(status=status, exact=exact, total=nframes, got=got,
               min_psnr=(round(min_psnr,2) if min_psnr!=float("inf") else None),
               w=w, h=h, ref_fmt=ref_fmt)
    if short:
        res["detail"] = "decoded %d/%d frames" % (got, nframes)
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
            if backend not in BACKENDS or codec not in BACKENDS[backend]["codecs"]:
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
    print(hdr); print("-"*len(hdr))
    pass_tot = fail_tot = 0
    for fname, codec, pixfmt in SAMPLES:
        if sample_filter and not any(f in fname for f in sample_filter):
            continue
        line = f"{fname:<22}"
        for backend in backends:
            res = next(r for f,b,r in rows if f==fname and b==backend)
            st = res["status"]
            if st == "PASS":
                cell = f"{res['exact']}/{res['total']}"
                pass_tot += 1
            elif st == "SHORT":
                cell = f"S({res.get('got')}/{res['total']})"
                fail_tot += 1
            elif st == "NEAR":
                cell = f"~{res.get('min_psnr')}"
                fail_tot += 1
            elif st == "FAIL":
                cell = f"F({res.get('min_psnr')})"
                fail_tot += 1
            else:
                cell = st[:11]
            line += f"{cell:>12}"
        print(line)
    print("-"*len(hdr))
    print(f"\nPASS cells: {pass_tot}   non-PASS (NEAR/FAIL): {fail_tot}")

    # detail for non-PASS
    bad = [(f,b,r) for f,b,r in rows if r["status"] in ("NEAR","FAIL","SHORT","BACKEND_FAIL","REF_FAIL","ERROR")]
    if bad:
        print("\n--- non-PASS detail ---")
        for f,b,r in bad:
            extra = r.get("detail") or r.get("first_diff") or ""
            print(f"  {b:<12} {f:<22} {r['status']:<10} {extra}")

if __name__ == "__main__":
    main()
