"""Real-time yoga pose coach.

    python realtime.py                          # webcam 0, auto-detect pose, hybrid mode
    python realtime.py --pose tree              # judge Tree pose specifically
    python realtime.py --source my_video.mp4 --save out.mp4 --no-display

Keys: q quit | 0 auto-detect | 1 Tree | 2 Warrior II | 3 Triangle | 4 Goddess | l leniency | d debug numbers | s screenshot
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from yogacoach.config import TARGET_POSES
from yogacoach.draw import draw_overlay
from yogacoach.engine import CoachEngine
from yogacoach.pose_backend import PoseBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or path to a video file")
    ap.add_argument("--pose", default="auto", choices=["auto"] + TARGET_POSES)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dl", "rules"],
                    help="hybrid = YogaNet + geometric rules (default)")
    ap.add_argument("--leniency", default="normal", choices=["strict", "normal", "lenient"],
                    help="how far from the ideal angles still counts as correct (default: lenient)")
    ap.add_argument("--check-pose", action="store_true",
                    help="also warn when the network thinks you are not doing the selected pose")
    ap.add_argument("--debug", action="store_true", help="show raw angles and pose scores on screen")
    ap.add_argument("--model", default="models/yogacoach.pt")
    ap.add_argument("--pose-variant", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--pose-model", default=None, help="path to a pose_landmarker_*.task file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam image")
    ap.add_argument("--mirror", action="store_true", help="mirror a video file as well")
    ap.add_argument("--save", default=None, help="write the annotated video to this .mp4")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--shots", default="outputs/screenshots")
    args = ap.parse_args()

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source '{args.source}'")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    mirror = (is_cam and not args.no_mirror) or (not is_cam and args.mirror)

    engine = CoachEngine(None if args.mode == "rules" else args.model, target=args.pose, mode=args.mode,
                         leniency=args.leniency, check_pose=args.check_pose)
    debug, levels = args.debug, ["strict", "normal", "lenient"]
    backend = PoseBackend(args.pose_variant, args.pose_model, video=True)
    writer, target, n, t_prev, fps = None, args.pose, 0, time.time(), 0.0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(args.shots, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        ts = int(1000 * n / src_fps) if not is_cam else int(time.time() * 1000)
        kp, _ = backend.detect(frame, ts)
        res = engine.update(kp)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if n else 0.0
        t_prev = now
        out = draw_overlay(frame, kp, res, fps, target, args.mode, debug, levels[engine.level])
        if args.save:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps if not is_cam else 25, (w, h))
            writer.write(out)
        if not args.no_display:
            cv2.imshow("Yoga Coach", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
                target = "auto" if key == ord("0") else TARGET_POSES[key - ord("1")]
                engine.target = target
                engine.reset()
            if key == ord("l"):                       # cycle strict -> normal -> lenient
                engine.level = (engine.level + 1) % 3
            if key == ord("d"):
                debug = not debug
            if key == ord("s"):
                p = os.path.join(args.shots, f"shot_{int(time.time())}.png")
                cv2.imwrite(p, out)
                print("saved", p)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    backend.close()
    if writer:
        writer.release()
        print("saved", args.save)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
