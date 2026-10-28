"""Synthetic keypoint generator (forward-kinematics skeleton).

Why: a project like this needs thousands of *labelled incorrect* poses, which
do not exist in any public dataset.  We bootstrap training with a kinematic
skeleton whose joint angles are sampled inside ("correct") or outside
("incorrect") the alignment ranges in config.RANGES.  Samples that lie within a
safety margin of a threshold are rejected so labels are unambiguous.  Sensor
effects (roll, foreshortening, jitter, mirroring, scale) are then added so the
network sees MediaPipe-like noisy sequences.  Real recordings made with
scripts/collect.py can be mixed in (and should be, see README).
"""
from __future__ import annotations

import numpy as np

from . import rules
from .config import ERRORS, K, NUM_ERR, NUM_KP, POSE_ERRORS, POSES, WINDOW
from .geometry import frame_features, measure, mirror

S_OUT = {"l": -1.0, "r": 1.0}   # image-left limbs point towards -x when "outward"


def _vec(psi_deg: float) -> np.ndarray:
    """Unit vector for an angle measured from straight DOWN, positive towards +x."""
    a = np.radians(psi_deg)
    return np.array([np.sin(a), np.cos(a)])


def build_skeleton(p: dict) -> np.ndarray:
    """Forward kinematics -> (13, 2) keypoints in torso-length units (y down)."""
    lean = np.radians(p["lean"])
    dir_t = np.array([np.sin(lean), -np.cos(lean)])
    perp = np.array([np.cos(lean), np.sin(lean)])
    shm = dir_t * 1.0
    kp = np.zeros((NUM_KP, 2))
    kp[K["l_hip"]] = (-p["hw"] / 2, 0.0)
    kp[K["r_hip"]] = (p["hw"] / 2, 0.0)
    kp[K["l_sh"]] = shm - perp * p["sw"] / 2
    kp[K["r_sh"]] = shm + perp * p["sw"] / 2
    kp[K["nose"]] = shm + dir_t * p["head"]
    for s in "lr":
        psi_u, psi_f = p["arm"][s]
        el = kp[K[f"{s}_sh"]] + _vec(psi_u) * p["uarm"]
        kp[K[f"{s}_el"]] = el
        kp[K[f"{s}_wr"]] = el + _vec(psi_f) * p["farm"]
        psi_t, psi_s = p["leg"][s]
        kn = kp[K[f"{s}_hip"]] + _vec(psi_t) * p["thigh"]
        kp[K[f"{s}_kn"]] = kn
        kp[K[f"{s}_an"]] = kn + _vec(psi_s) * p["shin"]
    return kp


def _body(rng):
    return dict(sw=rng.uniform(.72, .92), hw=rng.uniform(.42, .58), thigh=rng.uniform(.85, 1.05),
                shin=rng.uniform(.85, 1.05), uarm=rng.uniform(.52, .68), farm=rng.uniform(.42, .58),
                head=rng.uniform(.45, .65))


def _flags(rng, names, p_all_good=0.4, p_bad=0.35):
    if rng.random() < p_all_good:
        return {n: False for n in names}
    f = {n: rng.random() < p_bad for n in names}
    if not any(f.values()):
        f[names[rng.integers(len(names))]] = True
    return f


def _sgn(rng):
    return 1.0 if rng.random() < 0.5 else -1.0


def _leg(side, alpha_out, beta_out):
    return (S_OUT[side] * alpha_out, S_OUT[side] * beta_out)


def _arm(side, psi_u_out, bend, direction=1.0):
    """Arm given outward angle of the upper arm and elbow flexion (deg)."""
    u = S_OUT[side] * psi_u_out
    return (u, u + S_OUT[side] * bend * direction)


# --------------------------------------------------------------------------- poses
def sample_tree(rng):
    f = _flags(rng, ["standing_knee", "lifted_knee", "arms", "elbows", "torso"])
    p = _body(rng)
    p["lean"] = _sgn(rng) * (rng.uniform(16, 30) if f["torso"] else rng.uniform(0, 5))
    st = "l" if rng.random() < .5 else "r"
    lf = "r" if st == "l" else "l"
    a_s = rng.uniform(0, 5)
    bend_s = rng.uniform(22, 50) if f["standing_knee"] else rng.uniform(0, 6)
    leg = {st: _leg(st, a_s, a_s - _sgn(rng) * bend_s)}
    k = rng.uniform(112, 150) if f["lifted_knee"] else rng.uniform(35, 90)
    a_l = rng.uniform(40, 75)
    leg[lf] = _leg(lf, a_l, a_l - (180 - k))
    arm = {}
    for s in "lr":
        u = rng.uniform(40, 110) if f["arms"] else rng.uniform(150, 178)
        bend = rng.uniform(25, 55) if f["elbows"] else rng.uniform(0, 10)
        arm[s] = _arm(s, u, bend, 1.0 if rng.random() < .8 else -1.0)
    p.update(arm=arm, leg=leg)
    return p


def sample_warrior2(rng):
    f = _flags(rng, ["front_knee", "back_knee", "arms", "elbows", "torso", "stance"])
    p = _body(rng)
    p["lean"] = _sgn(rng) * (rng.uniform(14, 26) if f["torso"] else rng.uniform(0, 4))
    fr = "l" if rng.random() < .5 else "r"
    bk = "r" if fr == "l" else "l"
    k = rng.uniform(132, 162) if f["front_knee"] else rng.uniform(92, 118)
    shin_out = rng.uniform(-6, 8)
    a_f = 180 - k + shin_out
    bend_b = rng.uniform(22, 50) if f["back_knee"] else rng.uniform(0, 6)
    a_b = rng.uniform(25, 45)
    g = rng.uniform(.3, .55) if f["stance"] else 1.0
    leg = {fr: _leg(fr, a_f * g, shin_out), bk: _leg(bk, a_b * g, a_b * g - _sgn(rng) * bend_b)}
    arm = {}
    for s in "lr":
        if f["arms"]:
            u = rng.uniform(20, 65) if rng.random() < .5 else rng.uniform(115, 160)
        else:
            u = rng.uniform(82, 98)
        bend = rng.uniform(25, 50) if f["elbows"] else rng.uniform(0, 8)
        arm[s] = _arm(s, u, bend, _sgn(rng))
    p.update(arm=arm, leg=leg)
    return p


def sample_triangle(rng):
    f = _flags(rng, ["knees", "stance", "lean", "arms", "elbows"])
    p = _body(rng)
    sgn = _sgn(rng)
    p["lean"] = sgn * (rng.uniform(4, 28) if f["lean"] else rng.uniform(48, 80))
    g = rng.uniform(.25, .5) if f["stance"] else 1.0
    leg = {}
    for s in "lr":
        a = rng.uniform(22, 36) * g
        bend = rng.uniform(22, 45) if (f["knees"] and rng.random() < .6) else rng.uniform(0, 5)
        leg[s] = _leg(s, a, a - _sgn(rng) * bend)
    gam = _sgn(rng) * rng.uniform(32, 65) if f["arms"] else rng.uniform(-6, 6)
    top, bot = ("l", "r") if sgn > 0 else ("r", "l")
    arm = {}
    for s, base in ((top, 180.0), (bot, 0.0)):
        u = base + gam + rng.uniform(-4, 4)
        bend = rng.uniform(25, 50) if f["elbows"] else rng.uniform(0, 6)
        arm[s] = (u, u + S_OUT[s] * bend * _sgn(rng))
    p.update(arm=arm, leg=leg)
    return p


def sample_goddess(rng):
    f = _flags(rng, ["knees", "uneven", "stance", "arms", "elbows", "torso"])
    p = _body(rng)
    p["lean"] = _sgn(rng) * (rng.uniform(14, 26) if f["torso"] else rng.uniform(0, 4))
    kb = rng.uniform(142, 165) if f["knees"] else rng.uniform(104, 128)
    ks = {"l": kb + rng.normal(0, 2), "r": kb + rng.normal(0, 2)}
    if f["uneven"]:
        s = "l" if rng.random() < .5 else "r"
        ks[s] = min(168.0, ks[s] + rng.uniform(28, 50))
    g = rng.uniform(.3, .5) if f["stance"] else 1.0
    leg = {}
    for s in "lr":
        sh = rng.uniform(-5, 8)
        leg[s] = _leg(s, (180 - ks[s] + sh) * g, sh)
    arm = {}
    for s in "lr":
        if f["arms"]:
            u = rng.uniform(25, 62) if rng.random() < .5 else rng.uniform(118, 150)
        else:
            u = rng.uniform(82, 98)
        if f["elbows"]:
            bend = rng.uniform(5, 45) if rng.random() < .6 else rng.uniform(130, 150)
        else:
            bend = rng.uniform(82, 98)
        arm[s] = _arm(s, u, bend, 1.0)
    p.update(arm=arm, leg=leg)
    return p


def sample_other(rng):
    p = _body(rng)
    p["lean"] = _sgn(rng) * rng.uniform(0, 30)
    leg = {}
    for s in "lr":
        a = rng.uniform(0, 16)
        leg[s] = _leg(s, a, a - _sgn(rng) * rng.uniform(0, 22))
    arm = {}
    mode = rng.integers(4)
    for s in "lr":
        if mode == 0:        # arms by the side
            u, b = rng.uniform(0, 35), rng.uniform(0, 25)
        elif mode == 1:      # arms raised
            u, b = rng.uniform(150, 178), rng.uniform(0, 20)
        elif mode == 2:      # hands on hips / chest
            u, b = rng.uniform(25, 55), rng.uniform(70, 140)
        else:                # anything
            u, b = rng.uniform(0, 180), rng.uniform(0, 120)
        arm[s] = _arm(s, u, b, _sgn(rng))
    p.update(arm=arm, leg=leg)
    return p


SAMPLERS = {"tree": sample_tree, "warrior2": sample_warrior2, "triangle": sample_triangle,
            "goddess": sample_goddess, "other": sample_other}


# --------------------------------------------------------------------------- sequences
def _transform(kp, rng):
    """Camera roll, foreshortening, mirroring (noise-free; labels computed after this)."""
    roll = np.radians(rng.normal(0, 2.5))
    R = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    kp = kp @ R.T
    kp = kp * np.array([rng.uniform(0.9, 1.0), 1.0])
    if rng.random() < .5:
        kp = mirror(kp)
    return kp


def _to_sequence(kp, rng, T, noise=1.0):
    """Add per-sequence pixel scale/offset, smooth drift and iid jitter -> (T,13,2) pixels."""
    sigma = rng.uniform(.004, .025) * noise
    walk = np.cumsum(rng.normal(0, .004 * noise, (T, NUM_KP, 2)), axis=0)
    walk -= walk.mean(0, keepdims=True)
    seq = kp[None] + walk + rng.normal(0, sigma, (T, NUM_KP, 2))
    scale = rng.uniform(120, 350)
    return seq * scale + rng.uniform(200, 600, size=(1, 1, 2))


def generate(n_per_pose: int, seed: int, T: int = WINDOW, noise: float = 1.0, max_tries: int = 60):
    """Return dict of arrays: X (N,T,F), kp (N,T,13,2), pose (N,), errors (N,E), is_target (N,)."""
    rng = np.random.default_rng(seed)
    kps, poses, errs = [], [], []
    for pi, pose in enumerate(POSES):
        got = 0
        tries = 0
        while got < n_per_pose:
            tries += 1
            if tries > n_per_pose * max_tries:
                raise RuntimeError(f"generator stuck on {pose}")
            p = SAMPLERS[pose](rng)
            kp0 = _transform(build_skeleton(p), rng)
            m = measure(kp0[None])
            if pose == "other":
                if not (m["stance"][0] < 1.2 and m["bent_knee"][0] > 152):
                    continue
                lab = np.zeros(NUM_ERR, np.float32)
            else:
                if not bool(rules.unambiguous(pose, m)[0]):
                    continue
                lab = rules.labels(pose, m)[0]
            kps.append(_to_sequence(kp0, rng, T, noise))
            poses.append(pi)
            errs.append(lab)
            got += 1
    kp = np.stack(kps).astype(np.float32)
    poses = np.array(poses, np.int64)
    errs = np.stack(errs).astype(np.float32)
    perm = rng.permutation(len(poses))
    kp, poses, errs = kp[perm], poses[perm], errs[perm]
    return dict(kp=kp, X=frame_features(kp), pose=poses, errors=errs,
                is_target=(poses != POSES.index("other")).astype(np.float32))


def describe(data) -> str:
    lines = []
    for i, pose in enumerate(POSES):
        sel = data["pose"] == i
        n = int(sel.sum())
        if pose == "other":
            lines.append(f"  {pose:9s} n={n}")
            continue
        corr = float((data["errors"][sel].sum(1) == 0).mean())
        per = {e: int(data["errors"][sel][:, ERRORS.index(e)].sum()) for e in POSE_ERRORS[pose]}
        lines.append(f"  {pose:9s} n={n} correct={corr:.0%} errors={per}")
    return "\n".join(lines)
