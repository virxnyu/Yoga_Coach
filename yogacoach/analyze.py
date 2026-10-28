"""Single-image analysis (used by the image checker web page and CLI).

The same CoachEngine that runs on video is fed the photo's keypoints repeatedly
(a photo is a "video" in which nothing moves), so a photo gets exactly the
decision the live system would give for that posture.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import rules
from .config import K, MIN_VISIBILITY_IMAGE, POSE_TITLES, TARGET_POSES, WINDOW
from .draw import GREEN, RED, AMBER, ascii_text, draw_skeleton
from .engine import CoachEngine

MAX_SIDE = 1000


@dataclass
class ImageReport:
    status: str                       # correct | incorrect | wrong_pose | no_person
    pose: str | None = None
    score: float | None = None
    feedback: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    probs: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)       # (label, value, target, status, code)
    text: str = ""
    annotated: np.ndarray | None = None            # BGR
    leniency: str = "lenient"
    mode: str = "hybrid"
    visibility: float = 0.0

    def to_json(self, image_data_url: str) -> dict:
        return dict(status=self.status, pose=self.pose, pose_title=POSE_TITLES.get(self.pose, ""),
                    score=self.score, feedback=self.feedback, warnings=self.warnings, probs=self.probs,
                    rows=[dict(label=a, value=b, target=c, status=d, code=e) for a, b, c, d, e in self.rows],
                    text=self.text, image=image_data_url, leniency=self.leniency, mode=self.mode,
                    visibility=round(self.visibility, 2))


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode any common image format; honours phone-photo EXIF rotation. Returns BGR uint8."""
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def _shrink(img: np.ndarray, max_side: int = MAX_SIDE):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def _warnings(kp, vis, shape):
    h, w = shape[:2]
    out = []
    if vis < 0.5:
        out.append(f"Low landmark visibility ({vis:.2f}): part of the body may be cut off or hidden - the result may be unreliable.")
    if (kp[:, 0] < -0.01 * w).any() or (kp[:, 0] > 1.01 * w).any() or (kp[:, 1] < -0.01 * h).any() or (kp[:, 1] > 1.01 * h).any():
        out.append("Some joints lie outside the photo (hands or feet cut off) - use a photo showing the whole body.")
    bh = kp[:, 1].max() - kp[:, 1].min()
    if bh < 0.35 * h:
        out.append("The person is small in the photo; a closer full-body shot is more accurate.")
    L = np.linalg.norm((kp[K["l_sh"]] + kp[K["r_sh"]]) / 2 - (kp[K["l_hip"]] + kp[K["r_hip"]]) / 2) + 1e-6
    sw = np.linalg.norm(kp[K["l_sh"]] - kp[K["r_sh"]]) / L
    if sw < 0.3:
        out.append("The body looks turned sideways to the camera. This system expects a FRONT view (facing the camera), "
                   "so a side-view photo can give wrong results.")
    return out


def _annotate(img, kp, res, scale):
    out = img.copy()
    h, w = out.shape[:2]
    s = max(0.7, w / 900)
    if kp is not None:
        k = kp * scale
        draw_skeleton(out, k, res.highlight)
        m = res.measures
        for name, key in (("l_kn", "knee_l"), ("r_kn", "knee_r"), ("l_el", "elbow_l"), ("r_el", "elbow_r")):
            if key in m:
                p = (int(k[K[name]][0]) + 10, int(k[K[name]][1]) - 8)
                txt = f"{m[key]:.0f}"
                cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, .6 * s, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, .6 * s, (255, 255, 255), 1, cv2.LINE_AA)
    colour, head = {"correct": (GREEN, "CORRECT"), "incorrect": (RED, "INCORRECT"),
                    "wrong_pose": (AMBER, "NO TARGET POSE RECOGNISED"), "no_person": (AMBER, "NO BODY DETECTED")}[res.status]
    if res.pose and res.status in ("correct", "incorrect"):
        head += " - " + POSE_TITLES[res.pose].split(" (")[0]
    bar = int(46 * s)
    cv2.rectangle(out, (0, 0), (w, bar), colour, -1)
    cv2.putText(out, ascii_text(head), (12, int(bar * 0.68)), cv2.FONT_HERSHEY_DUPLEX, .8 * s, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def analyze_image(img_bgr: np.ndarray, backend, model_path: str | None, target: str = "auto",
                  mode: str = "hybrid", leniency: str = "lenient") -> ImageReport:
    img, scale = _shrink(img_bgr)
    kp, vis = backend.detect(img, None, min_vis=MIN_VISIBILITY_IMAGE)
    rep = ImageReport(status="no_person", leniency=leniency, mode=mode, visibility=float(vis))
    if kp is None:
        rep.text = (f"No body detected (landmark visibility {vis:.2f}).\n"
                    "Use a clear photo with the whole body visible, facing the camera, and not too small in the frame.")
        res0 = CoachEngine(None, "auto", "rules").update(None)
        rep.annotated = _annotate(img, None, res0, 1.0)
        rep.warnings = ["No body detected."]
        return rep

    eng = CoachEngine(model_path if mode != "rules" else None, target=target, mode=mode, leniency=leniency)
    res = None
    for i in range(WINDOW + 8):                     # a still photo: identical frames
        res = eng.update(kp, t=i / 30.0)
    rep.status, rep.pose, rep.score = res.status, res.pose, res.score
    rep.probs = {p: round(v, 3) for p, v in res.pose_probs.items()}
    rep.warnings = _warnings(kp, vis, img.shape)
    level = rules.level_of(leniency)
    rep.feedback = [msg for _, _, msg in res.errors] if res.status == "incorrect" else (
        [res.note] if res.note else [])

    table_pose = res.pose
    if table_pose is None and res.pose_probs:       # unrecognised: show the closest target pose
        cand = {p: v for p, v in res.pose_probs.items() if p in TARGET_POSES}
        table_pose = max(cand, key=cand.get) if cand else None
    if table_pose and res.measures:
        rep.rows = rules.report_rows(table_pose, res.measures, level)

    lines = []
    verdict = {"correct": "CORRECT", "incorrect": "INCORRECT", "wrong_pose": "NO TARGET POSE RECOGNISED"}[res.status]
    lines.append(f"Verdict: {verdict}")
    how = "selected by you" if target != "auto" else "auto-detected"
    if res.pose:
        lines.append(f"Pose judged: {POSE_TITLES[res.pose]}  ({how})")
    if res.score is not None:
        lines.append(f"Alignment score: {res.score:.0f}/100")
    lines.append(f"Mode: {mode}   Leniency: {leniency}")
    lines.append("Pose scores: " + ", ".join(f"{p} {v:.2f}" for p, v in sorted(res.pose_probs.items(), key=lambda kv: -kv[1])))
    lines.append("")
    if res.status == "incorrect":
        lines.append("What to fix:")
        lines += [f"  - {m}" for m in rep.feedback]
    elif res.status == "correct":
        lines.append(res.note)
    else:
        lines.append(res.note)
    if rep.rows:
        lines += ["", f"Measurements for {POSE_TITLES[table_pose]}:"]
        for label, val, tgt, st, _ in rep.rows:
            lines.append(f"  {label:42s} {val:>12s}   target {tgt:24s} {st}")
    if rep.warnings:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in rep.warnings]
    rep.text = "\n".join(lines)
    rep.annotated = _annotate(img, kp, res, 1.0)
    return rep
