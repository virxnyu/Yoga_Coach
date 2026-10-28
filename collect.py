"""Record labelled keypoint clips from your webcam (or a video file).

Every clip is one pose performed one way:

    python collect.py --pose tree --label correct --seconds 15
    python collect.py --pose tree --label bent_knee_shallow --seconds 15
    python collect.py --pose warrior2 --label arms_not_level,torso_lean
    python collect.py --pose other --label none            # standing, walking, random movement
    python collect.py --pose goddess --label correct --video clip.mp4   # label an existing video
    python collect.py --list                                # valid pose / error names

Clips are saved to data/real/ and picked up automatically by `python train.py --real-dir data/real`.
Tip: record each (pose, label) 3+ times, vary clothing/background/distance, move slightly during the clip.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from yogacoach import dataio
from yogacoach.config import POSE_ERRORS, POSES, TARGET_POSES, WINDOW
from yogacoach.draw import draw_skeleton
from yogacoach.pose_backend import PoseBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=POSES)
    ap.add_argument("--label", default="correct", help="'correct', 'none' (for pose other) or comma-separated error codes")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--countdown", type=int, default=6)
    ap.add_argument("--source", default="0")
    ap.add_argument("--video", default=None, help="label an existing video file instead of the webcam")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--pose-variant", default="full")
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.pose:
        print("poses:", ", ".join(POSES))
        for p in TARGET_POSES:
            print(f"  {p:9s} errors: {', '.join(POSE_ERRORS[p])}")
        return
    dataio.parse_labels(args.pose, args.label)          # validate before opening the camera

    src = args.video or args.source
    is_cam = args.video is None and str(src).isdigit()
    cap = cv2.VideoCapture(int(src) if is_cam else src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open '{src}'")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    backend = PoseBackend(args.pose_variant, args.pose_model, video=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mirror = is_cam and not args.no_mirror

    frames, t_start, n = [], time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        ts = int(time.time() * 1000) if is_cam else int(1000 * n / fps)
        kp, _ = backend.detect(frame, ts)
        n += 1
        el = time.time() - t_start
        if is_cam:
            if el < args.countdown:
                msg, col, rec = f"Get into {args.pose} ({args.label}) - starting in {args.countdown - el:0.0f}s", (40, 190, 250), False
            else:
                msg, col, rec = f"RECORDING {el - args.countdown:0.1f}/{args.seconds:.0f}s  frames={len(frames)}", (60, 60, 235), True
        else:
            msg, col, rec = f"processing frame {n}", (40, 190, 250), True
        if rec and kp is not None:
            frames.append(kp)
        view = frame.copy()
        if kp is not None:
            draw_skeleton(view, kp)
        else:
            cv2.putText(view, "no body detected - step back", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 190, 250), 2)
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, col, 2, cv2.LINE_AA)
        cv2.imshow("collect", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if is_cam and el >= args.countdown + args.seconds:
            break
    cap.release(); backend.close(); cv2.destroyAllWindows()

    if len(frames) < WINDOW:
        raise SystemExit(f"Only {len(frames)} usable frames (< {WINDOW}). Nothing saved - make sure your whole body is visible.")
    path = dataio.save_clip(args.out, np.stack(frames), args.pose, args.label, fps, "video" if args.video else "webcam")
    print(f"saved {len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
