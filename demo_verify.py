#!/usr/bin/env python3
"""demo_verify.py — decode every ~/demo mp4 (as extracted elementary stream)
with every applicable backend and verify per-frame pixel hashes are identical
across backends.

Streams live in ~/scratch_demo (extracted with `ffmpeg -c:v copy`):
  h264 -> *.h264, hevc -> *.h265, vp9/av1 -> *.ivf

Backends per codec (the sw backend only implements H.264/H.265):
  h264/h265: sw nvdec vaapi vulkan
  vp9/av1:   nvdec vaapi vulkan

Usage:
  python3 demo_verify.py --max-frames 30            # smoke, all streams
  python3 demo_verify.py                            # full decode, all streams
  python3 demo_verify.py --max-frames 30 forest     # filter by filename substring
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "target/release/examples/decode"
STREAMS = Path.home() / "scratch_demo"
HASHDIR = STREAMS / "hashes"

BACKENDS = {
    "h264": ["sw", "nvdec", "vaapi", "vulkan"],
    "h265": ["sw", "nvdec", "vaapi", "vulkan"],
    "vp9":  ["nvdec", "vaapi", "vulkan"],
    "av1":  ["nvdec", "vaapi", "vulkan"],
}

FR = re.compile(r"^frame (\d+): pts=\S+ size=(\d+)x(\d+) hash=([0-9a-f]{16}|-)\s*$")
DONE = re.compile(r"^decoded (\d+) frames in")


def stream_key(p: Path) -> str:
    """Unique per-file cache key (office.h264 / office.h265 share a stem)."""
    return p.name.replace(".", "_")


def stream_codec(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".h264":
        return "h264"
    if ext == ".h265":
        return "h265"
    if ext == ".ivf":
        head = p.read_bytes()[:16]
        return "av1" if b"AV01" in head else "vp9"
    raise ValueError(f"unknown stream type: {p}")


def run_backend(p: Path, backend: str, nframes: int) -> tuple[int, str | None]:
    """Decode with one backend; store per-frame hashes. Return (nframes, err)."""
    suffix = "" if nframes <= 0 else f"__n{nframes}"
    hf = HASHDIR / f"{stream_key(p)}__{backend}{suffix}.hashes"
    if hf.exists():
        lines = hf.read_text().splitlines()
        return len(lines), None

    cmd = [str(EXE), "-b", backend, "-i", str(p)]
    if nframes > 0:
        cmd += ["-n", str(nframes)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=21600)
    dt = time.time() - t0
    hashes = []
    n_decoded = None
    for line in (r.stdout or "").splitlines():
        m = FR.match(line)
        if m:
            hashes.append(m.group(4))
            continue
        d = DONE.match(line)
        if d:
            n_decoded = int(d.group(1))
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return 0, f"rc={r.returncode}: " + " | ".join(tail)[:200]
    if not hashes:
        return 0, "no frames printed"
    if n_decoded is not None and n_decoded != len(hashes):
        return len(hashes), f"decoded {n_decoded} but printed {len(hashes)}"
    HASHDIR.mkdir(parents=True, exist_ok=True)
    hf.write_text("\n".join(hashes) + "\n")
    print(f"   ({backend}: {len(hashes)} frames in {dt:.1f}s)", flush=True)
    return len(hashes), None


def load(p: Path, suffix: str) -> list[str] | None:
    f = HASHDIR / f"{stream_key(p)}{suffix}.hashes"
    if not f.exists():
        return None
    return f.read_text().splitlines()


def compare(p: Path, nframes: int) -> dict:
    suffix = "" if nframes <= 0 else f"__n{nframes}"
    codec = stream_codec(p)
    res = {"codec": codec}
    for b in BACKENDS[codec]:
        h = load(p, f"__{b}{suffix}")
        res[b] = h
    present = [b for b in BACKENDS[codec] if res[b] is not None]
    if not present:
        res["status"] = "NO_DATA"
        return res
    ref_b = present[0]
    ref = res[ref_b]
    problems = []
    for b in present[1:]:
        h = res[b]
        if len(h) != len(ref):
            problems.append(f"{b}: {len(h)} frames vs {len(ref)} ({ref_b})")
            continue
        for i, (a, c) in enumerate(zip(ref, h)):
            if a != c:
                problems.append(f"{b}: first hash mismatch at frame {i} "
                                f"({a} vs {c})")
                break
    res["nframes"] = len(ref)
    res["status"] = "PASS" if not problems else "FAIL"
    res["problems"] = problems
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limit display frames per run (0 = all)")
    ap.add_argument("filters", nargs="*", help="filename substrings")
    args = ap.parse_args()

    streams = sorted(STREAMS.glob("*.h264")) + sorted(STREAMS.glob("*.h265")) \
        + sorted(STREAMS.glob("*.ivf"))
    if args.filters:
        streams = [s for s in streams
                   if any(f in s.name for f in args.filters)]
    if not streams:
        sys.exit("no streams found")

    suffix = "" if args.max_frames <= 0 else f"__n{args.max_frames}"
    rows = []
    for p in streams:
        codec = stream_codec(p)
        backends = BACKENDS[codec]
        for b in backends:
            tag = f"{p.name} [{b}]"
            t0 = time.time()
            print(f"== {tag} ...", flush=True)
            n, err = run_backend(p, b, args.max_frames)
            print(f"   done in {time.time() - t0:.1f}s", flush=True)
            if err:
                print(f"   ERROR: {err}", flush=True)
        res = compare(p, args.max_frames)
        rows.append((p.name, res))
        st = res["status"]
        extra = f" ({res['nframes']} frames)" if "nframes" in res else ""
        print(f"   -> {st}{extra}", flush=True)
        for pr in res.get("problems", []):
            print(f"      {pr}", flush=True)

    print("\n" + "=" * 70)
    hdr = f"{'stream':<38}" + "".join(f"{b:>10}" for b in ["sw", "nvdec", "vaapi", "vulkan"])
    print(hdr)
    print("-" * len(hdr))
    for name, res in rows:
        cells = []
        for b in ["sw", "nvdec", "vaapi", "vulkan"]:
            h = res.get(b)
            if h is None:
                cells.append(f"{'-':>10}")
            else:
                cells.append(f"{len(h):>10}")
        mark = "" if res["status"] == "PASS" else "  <== FAIL"
        print(f"{name:<38}" + "".join(cells) + mark)
    nfail = sum(1 for _, r in rows if r["status"] != "PASS")
    print(f"\n{len(rows) - nfail}/{len(rows)} streams PASS cross-backend")


if __name__ == "__main__":
    main()
