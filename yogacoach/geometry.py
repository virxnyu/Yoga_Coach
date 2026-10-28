"""Vectorised pose geometry: angles, alignment measurements and model features.

Convention: image coordinates, x to the right, y DOWN, isotropic units
(pixels, NOT normalised 0-1 values, otherwise angles are distorted by the
frame aspect ratio).  All functions accept arrays shaped (..., 13, 2).
"""
from __future__ import annotations

import numpy as np

from .config import K

EPS = 1e-6
UP = np.array([0.0, -1.0])

# Number of features produced by frame_features()
FEATURE_NAMES = (
    [f"{n}_{a}" for n in ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr",
                          "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"] for a in "xy"]
    + ["knee_l", "knee_r", "elbow_l", "elbow_r", "sh_l", "sh_r", "hip_l", "hip_r",
       "lean", "stance", "wrist_above_nose", "wrist_above_elbow", "wrist_line"]
)
NUM_FEATURES = len(FEATURE_NAMES)  # 39


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle in degrees at b for points a-b-c. Shapes (..., 2) -> (...)."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / (n1 * n2 + EPS)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def torso_length(kp: np.ndarray) -> np.ndarray:
    shm = (kp[..., K["l_sh"], :] + kp[..., K["r_sh"], :]) / 2
    hipm = (kp[..., K["l_hip"], :] + kp[..., K["r_hip"], :]) / 2
    return np.linalg.norm(shm - hipm, axis=-1) + EPS


def measure(kp: np.ndarray) -> dict[str, np.ndarray]:
    """Alignment measurements used by rules, the generator and as model features."""
    g = lambda n: kp[..., K[n], :]  # noqa: E731
    shm = (g("l_sh") + g("r_sh")) / 2
    hipm = (g("l_hip") + g("r_hip")) / 2
    torso = shm - hipm
    L = np.linalg.norm(torso, axis=-1) + EPS

    lean = np.degrees(np.arccos(np.clip(np.sum(torso * UP, axis=-1) / L, -1, 1)))

    knee_l = angle3(g("l_hip"), g("l_kn"), g("l_an"))
    knee_r = angle3(g("r_hip"), g("r_kn"), g("r_an"))
    elbow_l = angle3(g("l_sh"), g("l_el"), g("l_wr"))
    elbow_r = angle3(g("r_sh"), g("r_el"), g("r_wr"))
    sh_l = angle3(g("l_hip"), g("l_sh"), g("l_el"))
    sh_r = angle3(g("r_hip"), g("r_sh"), g("r_el"))
    hip_l = angle3(g("l_sh"), g("l_hip"), g("l_kn"))
    hip_r = angle3(g("r_sh"), g("r_hip"), g("r_kn"))

    stance = np.linalg.norm(g("l_an") - g("r_an"), axis=-1) / L
    nose_y = g("nose")[..., 1]
    wan = np.minimum((nose_y - g("l_wr")[..., 1]), (nose_y - g("r_wr")[..., 1])) / L
    wae = np.minimum((g("l_el")[..., 1] - g("l_wr")[..., 1]),
                     (g("r_el")[..., 1] - g("r_wr")[..., 1])) / L
    wv = g("l_wr") - g("r_wr")
    wline = np.degrees(np.arccos(np.clip(np.abs(wv[..., 1]) / (np.linalg.norm(wv, axis=-1) + EPS), 0, 1)))

    return dict(
        L=L, lean=lean, knee_l=knee_l, knee_r=knee_r,
        straight_knee=np.maximum(knee_l, knee_r), bent_knee=np.minimum(knee_l, knee_r),
        knee_diff=np.abs(knee_l - knee_r),
        elbow_l=elbow_l, elbow_r=elbow_r,
        elbow_min=np.minimum(elbow_l, elbow_r), elbow_max=np.maximum(elbow_l, elbow_r),
        sh_l=sh_l, sh_r=sh_r, hip_l=hip_l, hip_r=hip_r,
        stance=stance, wrist_above_nose=wan, wrist_above_elbow=wae, wrist_line=wline,
    )


def normalize(kp: np.ndarray) -> np.ndarray:
    """Translate so the hip centre is the origin and scale by torso length."""
    hipm = (kp[..., K["l_hip"], :] + kp[..., K["r_hip"], :]) / 2
    L = torso_length(kp)
    return (kp - hipm[..., None, :]) / L[..., None, None]


def frame_features(kp: np.ndarray) -> np.ndarray:
    """(..., 13, 2) -> (..., 39) float32 feature vector (scale/translation invariant)."""
    m = measure(kp)
    nk = normalize(kp)
    flat = nk.reshape(*nk.shape[:-2], -1)
    extra = np.stack([
        m["knee_l"] / 180, m["knee_r"] / 180, m["elbow_l"] / 180, m["elbow_r"] / 180,
        m["sh_l"] / 180, m["sh_r"] / 180, m["hip_l"] / 180, m["hip_r"] / 180,
        m["lean"] / 90, np.clip(m["stance"], 0, 4) / 3,
        np.clip(m["wrist_above_nose"], -3, 3) / 3, np.clip(m["wrist_above_elbow"], -3, 3) / 3,
        m["wrist_line"] / 90,
    ], axis=-1)
    return np.concatenate([flat, extra], axis=-1).astype(np.float32)


def mirror(kp: np.ndarray) -> np.ndarray:
    """Horizontal flip with left/right keypoint swap (data augmentation)."""
    out = kp.copy()
    out[..., 0] = -out[..., 0]
    pairs = [("l_sh", "r_sh"), ("l_el", "r_el"), ("l_wr", "r_wr"),
             ("l_hip", "r_hip"), ("l_kn", "r_kn"), ("l_an", "r_an")]
    for a, b in pairs:
        out[..., [K[a], K[b]], :] = out[..., [K[b], K[a]], :]
    return out
