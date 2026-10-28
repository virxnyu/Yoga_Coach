"""Convert a folder of labelled photos / videos into training clips.

Folder layout (label folder = 'correct' or error codes joined with '+'):

    my_media/
      tree/correct/*.jpg|*.mp4
      tree/bent_knee_shallow/*.mp4
      warrior2/arms_not_level+torso_lean/*.png
      other/none/*.mp4

    python scripts/extract_dataset.py --root my_media --out data/real
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yogacoach import dataio  # noqa: E402
from yogacoach.config import POSES  # noqa: E402
from yogacoach.pose_backend import PoseBackend  # noqa: E402

IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--every", type=int, default=2, help="use every Nth video frame")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--pose-variant", default="full")
    ap.add_argument("--pose-model", default=None)
    args = ap.parse_args()

    img_backend = PoseBackend(args.pose_variant, args.pose_model, video=False)
    saved = skipped = 0
    for pose in POSES:
        for ldir in sorted(glob.glob(os.path.join(args.root, pose, "*"))):
            if not os.path.isdir(ldir):
                continue
            label = os.path.basename(ldir).replace("+", ",")
            dataio.parse_labels(pose, label)   # fail early on typos
            files = sorted(f for f in glob.glob(os.path.join(ldir, "*")) if f.lower().endswith(IMG + VID))
            for f in files:
                frames = []
                if f.lower().endswith(IMG):
                    im = cv2.imread(f)
                    kp, _ = img_backend.detect(im) if im is not None else (None, 0)
                    if kp is not None:
                        frames = [kp] * 12
                else:
                    vb = PoseBackend(args.pose_variant, args.pose_model, video=True)
                    cap = cv2.VideoCapture(f)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    i = 0
                    while len(frames) < args.max_frames:
                        ok, im = cap.read()
                        if not ok:
                            break
                        if i % args.every == 0:
                            kp, _ = vb.detect(im, int(1000 * i / fps))
                            if kp is not None:
                                frames.append(kp)
                        i += 1
                    cap.release(); vb.close()
                if len(frames) >= 12:
                    dataio.save_clip(args.out, np.stack(frames), pose, label, 30.0 / args.every, "extract")
                    saved += 1
                else:
                    skipped += 1
                    print("skipped (no/too few detections):", f)
    print(f"saved {saved} clips, skipped {skipped}")


if __name__ == "__main__":
    main()
