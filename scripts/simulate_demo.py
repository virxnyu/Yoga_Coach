"""Render the coaching overlay on SIMULATED skeleton input (no camera, no MediaPipe).

Used to (a) test the engine + overlay end-to-end on any machine and (b) make
clearly-labelled illustrative figures.  These are NOT real-world screenshots;
for the report take real ones with realtime.py (press 's').

    python scripts/simulate_demo.py --model models/yogacoach.pt --out outputs/simulated
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yogacoach import rules, synth  # noqa: E402
from yogacoach.config import ERRORS, TARGET_POSES  # noqa: E402
from yogacoach.draw import draw_overlay  # noqa: E402
from yogacoach.engine import CoachEngine  # noqa: E402
from yogacoach.geometry import measure  # noqa: E402

W, H = 1280, 720


def background():
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        img[y, :] = (int(40 + 25 * y / H), int(34 + 20 * y / H), int(30 + 15 * y / H))
    cv2.putText(img, "SIMULATED SKELETON INPUT (no camera)", (W - 560, H - 44), cv2.FONT_HERSHEY_SIMPLEX, .6, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def pick(pose, rng, correct):
    for _ in range(2000):
        p = synth.SAMPLERS[pose](rng)
        kp = synth._transform(synth.build_skeleton(p), rng)
        m = measure(kp[None])
        if not rules.unambiguous(pose, m)[0]:
            continue
        n = int(rules.labels(pose, m)[0].sum())
        if correct and n == 0:
            return kp
        if not correct and 1 <= n <= 2:
            return kp
    raise RuntimeError("no sample")


def fit(seq):
    """Scale/translate a (T,13,2) pixel sequence so the body fits the frame right of the panel."""
    b0 = seq[0]
    h = b0[:, 1].max() - b0[:, 1].min()
    w = b0[:, 0].max() - b0[:, 0].min()
    s = min(520.0 / h, 700.0 / w)
    c = np.array([(b0[:, 0].max() + b0[:, 0].min()) / 2, (b0[:, 1].max() + b0[:, 1].min()) / 2])
    return ((seq - c) * s + np.array([W * 0.62, H * 0.53])).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--out", default="outputs/simulated")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    mode = args.mode or ("hybrid" if args.model else "rules")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for pose in TARGET_POSES:
        for correct in (True, False):
            seq = fit(synth._to_sequence(pick(pose, rng, correct), rng, 30, 1.0))
            eng = CoachEngine(args.model, target=pose, mode=mode)
            res = None
            for f in seq:
                res = eng.update(f)
            img = draw_overlay(background(), seq[-1], res, 30.0, pose, mode)
            path = os.path.join(args.out, f"{pose}_{'correct' if correct else 'incorrect'}.png")
            cv2.imwrite(path, img)
            print(path, res.status, [c for c, _, _ in res.errors])


if __name__ == "__main__":
    main()
