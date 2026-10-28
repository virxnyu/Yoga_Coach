"""Stage your correct-pose photos into the layout extract_dataset.py needs, then turn them into clips.

    python scripts/prepare_photos.py                 # uses the default folder names below
    python scripts/prepare_photos.py --stage-only    # only copy/sort the photos, show counts

Default source folders (relative to data/real):
    Vrukshasana      -> tree      UtkataKonasana -> goddess
    Veerabhadrasana  -> warrior2  Triangle       -> triangle
Photos are searched recursively. Images whose path contains 'point', 'skeleton', 'landmark', 'kp', 'annot'
or 'overlay' are skipped (pictures with drawn body points must not be used; MediaPipe finds its own).
Staged photos go to data/photos/<pose>/correct/, clips to data/clips/ (train with --real-dir data/clips).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
SKIP = ("point", "skeleton", "landmark", "keypoint", "kp", "annot", "overlay")
MAP = {"Vrukshasana": "tree", "Veerabhadrasana": "warrior2", "Triangle": "triangle", "UtkataKonasana": "goddess"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "real"))
    ap.add_argument("--stage", default=os.path.join(ROOT, "data", "photos"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--label", default="correct")
    args = ap.parse_args()

    total = 0
    for folder, pose in MAP.items():
        d = os.path.join(args.src, folder)
        if not os.path.isdir(d):
            print(f"[missing] {d}")
            continue
        dest = os.path.join(args.stage, pose, args.label)
        os.makedirs(dest, exist_ok=True)
        n = skipped = 0
        for dp, _, fs in os.walk(d):
            for f in sorted(fs):
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, d).lower()
                if not f.lower().endswith(IMG):
                    continue
                if any(w in rel for w in SKIP):
                    skipped += 1
                    continue
                shutil.copyfile(full, os.path.join(dest, f"{n:03d}_{f}"))
                n += 1
        print(f"{folder:16s} -> {pose:9s}: {n} photos staged, {skipped} skipped (look like body-point drawings)")
        total += n
    if not total:
        raise SystemExit("No photos found.")
    if args.stage_only:
        return
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "extract_dataset.py"), "--root", args.stage, "--out", args.out]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
