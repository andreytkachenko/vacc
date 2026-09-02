#!/usr/bin/env python3
"""verify-all.py — sample generation + full hardware-decoder verification.

The single video source is assets/big_buck_bunney.h265 (Big Buck Bunny,
1920x1080, 300 frames). Every sample in assets/samples/ is a codec variant
of it: scaled to 640x360 @ 30 fps, 300 frames (10 s) — except the six t*/x*
H.264 stress samples, which are 30-frame files.

Sample generation (--generate / --regen):
  ffmpeg re-encodes the master with the per-sample recipe below into
  assets/samples/. Recipes were recovered from the encoder option strings
  embedded in the committed samples' SEI units (x264/x265), so regenerated
  files are structurally equivalent (same profile/chroma/depth/resolution/
  fps/frame count and stress properties). They are NOT byte-identical to
  the committed set: the original encode ran on a different machine, and
  for the H.264 stress samples the vcpkg x264 build is not available here.
  Consequence: the parser tests that pin bitstream-derived data (h264 NAL
  constants in h264_slice_inline.rs, VP9 golden anchors, the AV1 expected
  table) correspond to the COMMITTED samples; regenerating them invalidates
  those anchors. `--generate` only fills in missing files and never touches
  existing ones; `--regen` overwrites everything.

Verification: for every (sample, backend) combination this:
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
  python3 verify-all.py --generate             # fill in missing samples, then verify
  python3 verify-all.py --regen                # re-encode all samples from master, then verify
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
SAMPLES_DIR = Path(os.environ.get("VACC_SAMPLES_DIR", str(ROOT / "assets" / "samples")))
MASTER = ROOT / "assets" / "big_buck_bunney.h265"
EXE = ROOT / "target/release/examples/decode"
WORK = Path("/tmp/verify_all")
NFRAMES_DEFAULT = 300

# Sample generation recipes. Every sample is encoded from MASTER through the
# common pipeline:  -vf scale=640:360 -r 30 -t <seconds> -an
# (master is 1920x1080 @ 25 fps; resampling to 30 fps for 10 s yields exactly
# 300 frames, matching the committed samples). Encoder options were recovered
# from the x264/x265 option strings in the samples' SEI units. VP9/AV1 have no
# such SEI; their settings are equivalent (same profile/depth/chroma) but not
# byte-reproducible.
#   name: (ffmpeg encoder args, seconds of source, post-step or None)
# Post-steps: "plain-baseline" clears the SPS constraint flags (this x264
# build always sets constraint_set1 for the baseline profile; the committed
# h264_baseline sample is plain Baseline with 0x00 constraints).
RECIPES = {
    "h264_baseline.h264":            ("-c:v libx264 -profile:v baseline -preset medium -crf 23", 10, "plain-baseline"),
    # (this build emits constraint flags 0xc0 for baseline = Constrained Baseline)
    "h264_constrained_baseline.h264": ("-c:v libx264 -profile:v baseline -preset medium -crf 23", 10),
    "h264_main.h264":                ("-c:v libx264 -profile:v main -preset medium -crf 23", 10),
    "h264_high.h264":                ("-c:v libx264 -profile:v high -preset medium -crf 23", 10),
    # 10-bit / 4:2:2 / 4:4:4: do NOT force a profile name (this ffmpeg build
    # either rejects the combo or silently falls back to 4:2:0); -pix_fmt alone
    # makes x264 auto-select high10/high422/high444.
    "h264_high10.h264":              ("-c:v libx264 -pix_fmt yuv420p10le -preset medium -crf 23", 10),
    "h264_high422.h264":             ("-c:v libx264 -pix_fmt yuv422p -preset medium -crf 23", 10),
    "h264_high444.h264":             ("-c:v libx264 -pix_fmt yuv444p -preset medium -crf 23", 10),
    # 30-frame stress samples (transform/deblock variants; original used a
    # vcpkg x264 build — regenerated with the distro x264, same options).
    # weightp=0 / crf=18 match the originals (verified via PPS fields: this
    # x264 build defaults to weightp=1 and crf 23, older builds did not).
    "h264_tC.h264":                  ("-c:v libx264 -profile:v main -preset medium -x264-params crf=18:deblock=0:0:0:mbtree=0:bframes=0:weightp=0", 1),
    "h264_tD.h264":                  ("-c:v libx264 -profile:v baseline -preset medium -x264-params crf=18:deblock=1:0:0:bframes=0", 1),
    "h264_tN.h264":                  ("-c:v libx264 -profile:v baseline -preset medium -x264-params crf=18:deblock=0:0:0:bframes=0", 1),
    "h264_tW.h264":                  ("-c:v libx264 -profile:v baseline -preset medium -x264-params crf=18:deblock=0:0:0:bframes=0:weightp=2", 1),
    # all-IDR: keyint=1 + per-frame SPS/PPS (repeat_headers).
    "h264_xallI.h264":               ("-c:v libx264 -profile:v main -preset medium -x264-params keyint=1:keyint_min=1:scenecut=0:ref=1:mixed_ref=0:bframes=0:mbtree=0:crf=18:repeat_headers=1", 1),
    # frame-dup stress: 30 frames > 2^4 frame_num space with no IDR reset,
    # so frame_num wraps mid-stream (exercises DPB wraparound handling).
    # Original is Main profile + CAVLC (not Baseline as the name suggests).
    "h264_xfd.h264":                 ("-c:v libx264 -profile:v main -coder cavlc -preset medium -x264-params deblock=0:0:0:bframes=3:weightb=0:weightp=0:crf=18", 1),
    "h265_main.h265":                ("-c:v libx265 -crf 28 -x265-params open-gop=1:repeat-headers=1:keyint=250:min-keyint=25", 10),
    "h265_cra.h265":                 ("-c:v libx265 -crf 28 -x265-params open-gop=1:repeat-headers=1:keyint=25:min-keyint=2", 10),
    "h265_msp.h265":                 ("-c:v libx265 -crf 28 -x265-params open-gop=1:repeat-headers=1:keyint=30:min-keyint=3:slices=4", 10),
    "h265_main10.h265":              ("-c:v libx265 -pix_fmt yuv420p10le -crf 28 -x265-params open-gop=1:repeat-headers=1:keyint=250:min-keyint=25", 10),
    "vp9_profile0.ivf":              ("-c:v libvpx-vp9 -b:v 300k -g 128", 10),
    "vp9_profile1_444.ivf":          ("-c:v libvpx-vp9 -pix_fmt yuv444p -b:v 300k -g 128", 10),
    "vp9_profile1.ivf":              ("-c:v libvpx-vp9 -pix_fmt yuv420p10 -b:v 300k -g 128", 10),
    "vp9_profile2.ivf":              ("-c:v libvpx-vp9 -pix_fmt yuv420p12 -b:v 300k -g 128", 10),
    "av1_main.ivf":                  ("-c:v libaom-av1 -crf 30 -b:v 0 -cpu-used 2 -g 600", 10),
    "av1_high.ivf":                  ("-c:v libaom-av1 -pix_fmt yuv420p10 -crf 30 -b:v 0 -cpu-used 2 -g 600", 10),
    "av1_professional.ivf":          ("-c:v libaom-av1 -pix_fmt yuv422p10 -crf 30 -b:v 0 -cpu-used 2 -g 600", 10),
}

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


def clear_sps_constraints(path):
    """Zero the constraint-flags byte of every SPS NAL (plain Baseline).
    NAL layout: <start code> 67 <profile_idc> <constraint_flags> <level_idc> ..."""
    d = bytearray(Path(path).read_bytes())
    i = 0
    while True:
        j = d.find(b"\x00\x00\x01", i)
        if j < 0 or j + 5 > len(d):
            break
        if d[j + 3] & 0x1f == 7:  # SPS NAL
            d[j + 5] = 0
        i = j + 3
    Path(path).write_bytes(bytes(d))


def generate_samples(force=False):
    """Encode samples from MASTER into SAMPLES_DIR via the RECIPES table.

    force=False: only missing files. force=True (--regen): overwrite all.
    """
    import shlex
    if not MASTER.exists():
        print(f"ERROR: master source {MASTER} not found")
        return False
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for fname in (name for name, _ in SAMPLES):
        recipe = RECIPES.get(fname)
        if recipe is None:
            print(f"  FAIL: {fname}: no generation recipe")
            ok = False
            continue
        args_str, seconds = recipe[0], recipe[1]
        post = recipe[2] if len(recipe) > 2 else None
        out = SAMPLES_DIR / fname
        if out.exists() and not force:
            print(f"  [skip] {fname} (exists; --regen to overwrite)")
            continue
        cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(MASTER), "-vf", "scale=640:360", "-r", "30",
                "-t", str(seconds), "-an"]
               + shlex.split(args_str) + [str(out)])
        print(f"  [gen]  {fname} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            print(f"  FAIL: {fname}: {r.stderr.strip().splitlines()[-1] if r.stderr else 'no output'}")
            ok = False
            continue
        if post == "plain-baseline":
            clear_sps_constraints(out)
    if force:
        print("WARNING: regenerated samples are structurally equivalent but NOT "
              "byte-identical to the committed set; parser tests with pinned "
              "bitstream data (h264 NAL constants, VP9 golden anchors, AV1 table) "
              "correspond to the committed samples and will mismatch.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=NFRAMES_DEFAULT)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--samples", default="")  # comma list of filename substrings
    ap.add_argument("--generate", action="store_true",
                    help="encode missing samples from assets/big_buck_bunney.h265 first")
    ap.add_argument("--regen", action="store_true",
                    help="re-encode ALL samples from the master (overwrites)")
    args = ap.parse_args()

    if args.generate or args.regen:
        print("Generating samples from %s ..." % MASTER.name, flush=True)
        generate_samples(force=args.regen)

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
