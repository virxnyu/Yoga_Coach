"""Reading / writing real recorded clips (keypoint sequences saved as .npz)."""
from __future__ import annotations

import glob
import os
import time

import numpy as np

from .config import ERRORS, NUM_ERR, NUM_KP, POSE_ERRORS, POSES, WINDOW
from .geometry import frame_features, mirror


def parse_labels(pose: str, errors: str) -> np.ndarray:
    """'' / 'correct' / 'none' -> all zeros, otherwise comma separated error codes."""
    if pose not in POSES:
        raise ValueError(f"unknown pose '{pose}', expected one of {POSES}")
    vec = np.zeros(NUM_ERR, np.float32)
    codes = [c.strip() for c in str(errors).split(",") if c.strip() and c.strip() not in ("correct", "none")]
    for c in codes:
        if c not in POSE_ERRORS.get(pose, []):
            raise ValueError(f"error '{c}' is not valid for pose '{pose}' (valid: {POSE_ERRORS.get(pose)})")
        vec[ERRORS.index(c)] = 1
    return vec


def save_clip(out_dir: str, kp: np.ndarray, pose: str, errors: str, fps: float, source: str = "webcam") -> str:
    """kp: (T,13,2) pixel coordinates."""
    parse_labels(pose, errors)  # validate
    os.makedirs(out_dir, exist_ok=True)
    tag = "correct" if not str(errors).strip() or errors in ("correct", "none") else str(errors).replace(",", "+")
    path = os.path.join(out_dir, f"{pose}__{tag}__{int(time.time() * 1000)}.npz")
    np.savez_compressed(path, kp=kp.astype(np.float32), pose=pose, errors=str(errors), fps=fps, source=source)
    return path


def load_real(directory: str, T: int = WINDOW, stride: int = 3, mirror_aug: bool = True):
    """Load all clips -> windows. Returns dict like synth.generate plus 'file' (clip index)."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        return None
    kps, poses, errs, fidx = [], [], [], []
    for fi, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        kp = z["kp"].astype(np.float32)
        pose, errors = str(z["pose"]), str(z["errors"])
        lab = parse_labels(pose, errors)
        if kp.ndim != 3 or kp.shape[1:] != (NUM_KP, 2):
            raise ValueError(f"{f}: expected (T,{NUM_KP},2) keypoints, got {kp.shape}")
        if len(kp) < T:   # single images / very short clips -> repeat last frame
            kp = np.concatenate([kp, np.repeat(kp[-1:], T - len(kp), axis=0)])
        for s in range(0, len(kp) - T + 1, stride):
            w = kp[s:s + T]
            for aug in ((False, True) if mirror_aug else (False,)):
                kps.append(mirror(w) if aug else w)
                poses.append(POSES.index(pose))
                errs.append(lab)
                fidx.append(fi)
    kp = np.stack(kps).astype(np.float32)
    poses = np.array(poses, np.int64)
    return dict(kp=kp, X=frame_features(kp), pose=poses, errors=np.stack(errs).astype(np.float32),
                is_target=(poses != POSES.index("other")).astype(np.float32),
                file=np.array(fidx), files=files)
