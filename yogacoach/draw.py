"""OpenCV overlay: skeleton, status banner, feedback list, score bar."""
from __future__ import annotations

import textwrap

import cv2
import numpy as np

from .config import EDGES, K, POSE_TITLES
from .engine import FrameResult

GREEN, RED, AMBER, WHITE, GREY = (80, 200, 90), (60, 60, 235), (40, 190, 250), (245, 245, 245), (170, 170, 170)


def ascii_text(s: str) -> str:
    """OpenCV's built-in fonts cannot draw the degree sign etc."""
    return s.replace("°", " deg").replace("–", "-").replace("—", "-").replace("’", "'")


def draw_skeleton(img, kp, highlight=()):
    hl = set(highlight)
    for a, b in EDGES:
        bad = a in hl or b in hl
        p1, p2 = tuple(int(v) for v in kp[K[a]]), tuple(int(v) for v in kp[K[b]])
        cv2.line(img, p1, p2, RED if bad else GREEN, 4, cv2.LINE_AA)
    for n, i in K.items():
        cv2.circle(img, tuple(int(v) for v in kp[i]), 7 if n in hl else 5, RED if n in hl else WHITE, -1, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.62):
    sub = img[y:y + h, x:x + w]
    sub[:] = (sub * (1 - alpha)).astype(np.uint8)


def draw_debug(img, res, s):
    """Right-hand panel with the raw numbers behind the decision."""
    H, W = img.shape[:2]
    lines = []
    if res.pose_probs:
        lines.append("pose scores:")
        lines += [f" {p:9s}{v:5.2f}" for p, v in sorted(res.pose_probs.items(), key=lambda kv: -kv[1])]
    m = res.measures
    if m:
        lines += ["", f"knees   {m['knee_l']:4.0f} {m['knee_r']:4.0f}", f"elbows  {m['elbow_l']:4.0f} {m['elbow_r']:4.0f}",
                  f"shoulder{m['sh_l']:4.0f} {m['sh_r']:4.0f}", f"lean    {m['lean']:4.0f}", f"stance  {m['stance']:4.2f}",
                  f"wr>nose {m['wrist_above_nose']:4.2f}", f"wr line {m['wrist_line']:4.0f}"]
    if not lines:
        return
    pw, lh = int(250 * s), int(24 * s)
    x0 = W - pw - 12
    _panel(img, x0, 12, pw, lh * (len(lines) + 1))
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x0 + 10, 12 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, .5 * s, WHITE, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, kp: np.ndarray | None, res: FrameResult, fps: float = 0.0,
                 target: str = "auto", mode: str = "hybrid", debug: bool = False,
                 leniency: str = "") -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    s = max(0.6, min(W, H) / 900)               # font scale follows the resolution
    if kp is not None:
        draw_skeleton(img, kp, res.highlight)

    lines = []
    if res.status in ("warming", "no_person", "wrong_pose"):
        colour, head = AMBER, {"warming": "GETTING READY", "no_person": "BODY NOT VISIBLE",
                               "wrong_pose": "CHECK YOUR POSE"}[res.status]
        lines = textwrap.wrap(ascii_text(res.note), 46)
    elif res.status == "correct":
        colour, head = GREEN, "CORRECT POSE"
        lines = textwrap.wrap(ascii_text(res.note), 46)
    else:
        colour, head = RED, "INCORRECT POSE"
        for _, _, msg in res.errors[:4]:
            lines += textwrap.wrap("- " + ascii_text(msg), 46)

    pw = int(560 * s)
    ph = int((95 + 30 * (len(lines) + 1) + (40 if res.score is not None else 0)) * s)
    _panel(img, 12, 12, pw, ph)
    title = POSE_TITLES.get(res.pose, "Auto-detect") if res.pose else (
        POSE_TITLES.get(target, "Auto-detect") if target != "auto" else "Auto-detect")
    cv2.putText(img, title, (28, int(46 * s)), cv2.FONT_HERSHEY_SIMPLEX, .72 * s, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, head, (28, int(88 * s)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * s, colour, 2, cv2.LINE_AA)
    y = int(124 * s)
    for ln in lines:
        cv2.putText(img, ln, (28, y), cv2.FONT_HERSHEY_SIMPLEX, .58 * s, WHITE, 1, cv2.LINE_AA)
        y += int(30 * s)
    if res.score is not None:
        bw = int(300 * s)
        cv2.rectangle(img, (28, y + 4), (28 + bw, y + int(18 * s) + 4), GREY, 1)
        cv2.rectangle(img, (28, y + 4), (28 + int(bw * res.score / 100), y + int(18 * s) + 4), colour, -1)
        txt = f"alignment {res.score:.0f}/100"
        if res.status == "correct" and res.hold_seconds > 0.5:
            txt += f"   held {res.hold_seconds:.0f}s"
        cv2.putText(img, txt, (28 + bw + 12, y + int(17 * s) + 2), cv2.FONT_HERSHEY_SIMPLEX, .55 * s, WHITE, 1, cv2.LINE_AA)

    if debug:
        draw_debug(img, res, s)
    foot = (f"target: {target}  mode: {mode}  {leniency}  fps: {fps:4.1f}   [q] quit [0] auto [1-4] pose "
            f"[l] leniency [d] debug [s] shot")
    cv2.putText(img, foot, (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, .5 * s, WHITE, 1, cv2.LINE_AA)
    return img
