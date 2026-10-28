"""Rule-based alignment checker.

Each check returns an *excess*: how far (in degrees, or torso-lengths for ratio
metrics) a measurement lies outside its target range (<= 0 means inside).
The same table drives:
  * the synthetic generator's labels (with an ambiguity margin),
  * the numeric half of the real-time hybrid feedback,
  * the human readable corrections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import E, ERRORS, NUM_ERR, POSE_ERRORS, RANGES

DEG, RATIO = "deg", "ratio"
SEV_SCALE = {DEG: 15.0, RATIO: 0.30}
MARGIN = {DEG: 5.0, RATIO: 0.08}

# Leniency: how far OUTSIDE the ideal range a measurement may be before it counts as a mistake.
# level 0 = strict (used for training labels), 1 = normal, 2 = lenient (default in the app).
LENIENCY = {"strict": 0, "normal": 1, "lenient": 2}
TOL_PER_LEVEL = {DEG: 6.0, RATIO: 0.06}


def level_of(leniency) -> int:
    return LENIENCY[leniency] if isinstance(leniency, str) else int(leniency)


def _hi(v, hi):
    return v - hi


def _lo(v, lo):
    return lo - v


def _checks(pose: str):
    r = RANGES[pose]
    if pose == "tree":
        return {
            "straight_knee_bent": (lambda m: _lo(m["straight_knee"], r["straight_knee"][0]), DEG),
            "bent_knee_shallow": (lambda m: _hi(m["bent_knee"], r["bent_knee"][1]), DEG),
            "arms_low": (lambda m: _lo(m["wrist_above_nose"], r["wrist_above_nose"][0]), RATIO),
            "elbows_bent": (lambda m: _lo(m["elbow_min"], r["elbow_min"][0]), DEG),
            "torso_lean": (lambda m: _hi(m["lean"], r["lean"][1]), DEG),
        }
    if pose == "warrior2":
        lo, hi = r["sh_both"]
        return {
            "bent_knee_shallow": (lambda m: _hi(m["bent_knee"], r["bent_knee"][1]), DEG),
            "straight_knee_bent": (lambda m: _lo(m["straight_knee"], r["straight_knee"][0]), DEG),
            "arms_not_level": (lambda m: np.maximum(_lo(np.minimum(m["sh_l"], m["sh_r"]), lo),
                                                    _hi(np.maximum(m["sh_l"], m["sh_r"]), hi)), DEG),
            "elbows_bent": (lambda m: _lo(m["elbow_min"], r["elbow_min"][0]), DEG),
            "torso_lean": (lambda m: _hi(m["lean"], r["lean"][1]), DEG),
            "stance_narrow": (lambda m: _lo(m["stance"], r["stance"][0]), RATIO),
        }
    if pose == "triangle":
        return {
            "straight_knee_bent": (lambda m: _lo(m["bent_knee"], r["knee_min"][0]), DEG),
            "stance_narrow": (lambda m: _lo(m["stance"], r["stance"][0]), RATIO),
            "lean_insufficient": (lambda m: _lo(m["lean"], r["lean"][0]), DEG),
            "arms_not_stacked": (lambda m: _hi(m["wrist_line"], r["wrist_line"][1]), DEG),
            "elbows_bent": (lambda m: _lo(m["elbow_min"], r["elbow_min"][0]), DEG),
        }
    if pose == "goddess":
        lo, hi = r["sh_both"]
        elo, ehi = r["elbow_both"]
        return {
            "bent_knee_shallow": (lambda m: _hi(m["straight_knee"], r["bent_knee"][1]), DEG),
            "knees_uneven": (lambda m: _hi(m["knee_diff"], r["knee_diff"][1]), DEG),
            "stance_narrow": (lambda m: _lo(m["stance"], r["stance"][0]), RATIO),
            "arms_not_level": (lambda m: np.maximum(_lo(np.minimum(m["sh_l"], m["sh_r"]), lo),
                                                    _hi(np.maximum(m["sh_l"], m["sh_r"]), hi)), DEG),
            "elbow_angle_wrong": (lambda m: np.maximum(_lo(m["elbow_min"], elo),
                                                       _hi(m["elbow_max"], ehi)), DEG),
            "torso_lean": (lambda m: _hi(m["lean"], r["lean"][1]), DEG),
        }
    raise KeyError(pose)


_CACHE: dict[str, dict] = {}


def checks(pose: str):
    if pose not in _CACHE:
        _CACHE[pose] = _checks(pose)
        assert set(_CACHE[pose]) == set(POSE_ERRORS[pose]), pose
    return _CACHE[pose]


_MARGIN_VEC = np.array([0.0] * NUM_ERR)


def excess_matrix(pose: str, m: dict, level: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(..., NUM_ERR) excess matrix (-1e3 where an error is not applicable to the pose)
    and the per-error ambiguity margin vector.  `level` > 0 relaxes every threshold."""
    ref = np.asarray(m["lean"])
    X = np.full(ref.shape + (NUM_ERR,), -1e3, dtype=np.float64)
    marg = np.zeros(NUM_ERR)
    for code, (fn, unit) in checks(pose).items():
        X[..., E[code]] = fn(m) - TOL_PER_LEVEL[unit] * level
        marg[E[code]] = MARGIN[unit]
    for code in ERRORS:                       # margins for non-applicable codes irrelevant
        if marg[E[code]] == 0:
            marg[E[code]] = 1.0
    return X, marg


def labels(pose: str, m: dict) -> np.ndarray:
    """Multi-hot error labels (strict thresholds)."""
    X, _ = excess_matrix(pose, m)
    return (X > 0).astype(np.float32)


def unambiguous(pose: str, m: dict) -> np.ndarray:
    """True where every measurement is clearly inside or clearly outside its range."""
    X, marg = excess_matrix(pose, m)
    strict = X > marg
    loose = X > -marg
    return ~np.any(strict != loose, axis=-1)


# --------------------------------------------------------------------------- feedback text
ADVICE = {
    "tree": {
        "straight_knee_bent": "Straighten your standing leg (knee {straight_knee:.0f}°, aim for 165°+).",
        "bent_knee_shallow": "Open your lifted knee out and place the foot higher on the inner thigh (knee {bent_knee:.0f}°, aim under 100°).",
        "arms_low": "Raise both arms and join your hands above your head.",
        "elbows_bent": "Straighten your elbows and reach up through the fingertips.",
        "torso_lean": "Stand tall: stack your shoulders over your hips (leaning {lean:.0f}°, aim under 10°).",
    },
    "warrior2": {
        "bent_knee_shallow": "Bend your front knee deeper, towards 90-110° (now {bent_knee:.0f}°).",
        "straight_knee_bent": "Straighten your back leg (knee {straight_knee:.0f}°, aim for 165°+).",
        "arms_not_level": "Bring both arms to shoulder height, parallel to the floor.",
        "elbows_bent": "Straighten both elbows and stretch through the fingertips.",
        "torso_lean": "Keep your torso upright over your hips (leaning {lean:.0f}°, aim under 10°).",
        "stance_narrow": "Step your feet wider apart.",
    },
    "triangle": {
        "straight_knee_bent": "Straighten both legs (weaker knee at {bent_knee:.0f}°, aim for 160°+).",
        "stance_narrow": "Widen your stance.",
        "lean_insufficient": "Tilt your torso further sideways, hinging at the hip (now {lean:.0f}°, aim 40°+).",
        "arms_not_stacked": "Stack your arms in one vertical line: top hand directly above the bottom hand.",
        "elbows_bent": "Straighten both elbows.",
    },
    "goddess": {
        "bent_knee_shallow": "Sink deeper and bend both knees (straighter knee {straight_knee:.0f}°, aim 95-135°).",
        "knees_uneven": "Bend both knees by the same amount (difference {knee_diff:.0f}°).",
        "stance_narrow": "Widen your stance with the toes turned out.",
        "arms_not_level": "Lift your upper arms to shoulder height.",
        "elbow_angle_wrong": "Bend your elbows to about 90 degrees (cactus arms).",
        "torso_lean": "Keep your torso upright.",
    },
}

GOOD_MESSAGE = "Great alignment - hold the pose and breathe."


@dataclass
class Violation:
    code: str
    excess: float
    severity: float   # 0..1
    message: str


def _scalar(m: dict) -> dict:
    return {k: float(np.asarray(v)) for k, v in m.items()}


def violations(pose: str, m: dict, level: int = 0) -> list[Violation]:
    """Rule violations for one frame, worst first, with formatted messages."""
    ms = _scalar(m)
    out = []
    for code, (fn, unit) in checks(pose).items():
        ex = float(fn(ms)) - TOL_PER_LEVEL[unit] * level
        if ex > 0:
            sev = float(min(1.0, ex / SEV_SCALE[unit]))
            out.append(Violation(code, ex, sev, ADVICE[pose][code].format(**ms)))
    out.sort(key=lambda v: -v.severity)
    return out


def confidence_matrix(pose: str, m: dict, level: int = 0) -> np.ndarray:
    """Soft per-error confidence in [0,1] (vectorised): 0.5 exactly at the threshold,
    saturating at +/- one tolerance band (2 x the ambiguity margin)."""
    X, marg = excess_matrix(pose, m, level)
    return np.clip(0.5 + X / (4.0 * marg), 0.0, 1.0)


def rule_confidence(pose: str, m: dict, level: int = 0) -> np.ndarray:
    """Single-frame version used by the real-time engine -> (NUM_ERR,)."""
    return confidence_matrix(pose, {k: np.asarray(float(np.asarray(v))) for k, v in m.items()}, level)


def pose_score(pose: str, m: dict, level: int = 0) -> float:
    """0-100 alignment score: mean over applicable checks of (1 - severity)."""
    ms = _scalar(m)
    sev = []
    for code, (fn, unit) in checks(pose).items():
        ex = float(fn(ms)) - TOL_PER_LEVEL[unit] * level
        sev.append(min(1.0, max(0.0, ex) / SEV_SCALE[unit]))
    return float(100 * (1 - np.mean(sev)))


# --------------------------------------------------------------------------- human-readable measurement table
def _one(key, fmt="{:.0f}°"):
    return lambda m: fmt.format(m[key])


def _two(k1, k2):
    return lambda m: f"{m[k1]:.0f}° / {m[k2]:.0f}°"


_R = "{:.2f}"
REPORT = {
    "tree": {
        "straight_knee_bent": ("standing-leg knee angle", _one("straight_knee"), ">= 165°"),
        "bent_knee_shallow": ("lifted-leg knee angle", _one("bent_knee"), "<= 100°"),
        "arms_low": ("wrists above nose (torso lengths)", _one("wrist_above_nose", _R), ">= 0.10"),
        "elbows_bent": ("smaller elbow angle", _one("elbow_min"), ">= 150°"),
        "torso_lean": ("torso lean from vertical", _one("lean"), "<= 10°"),
    },
    "warrior2": {
        "bent_knee_shallow": ("front-knee angle", _one("bent_knee"), "<= 125° (ideal 90-110°)"),
        "straight_knee_bent": ("back-knee angle", _one("straight_knee"), ">= 165°"),
        "arms_not_level": ("arm height, left / right shoulder angle", _two("sh_l", "sh_r"), "75-105°"),
        "elbows_bent": ("smaller elbow angle", _one("elbow_min"), ">= 160°"),
        "torso_lean": ("torso lean from vertical", _one("lean"), "<= 10°"),
        "stance_narrow": ("stance width (torso lengths)", _one("stance", _R), ">= 1.70"),
    },
    "triangle": {
        "straight_knee_bent": ("weaker knee angle", _one("bent_knee"), ">= 160°"),
        "stance_narrow": ("stance width (torso lengths)", _one("stance", _R), ">= 1.60"),
        "lean_insufficient": ("torso lean from vertical", _one("lean"), ">= 35°"),
        "arms_not_stacked": ("wrist line vs vertical", _one("wrist_line"), "<= 22°"),
        "elbows_bent": ("smaller elbow angle", _one("elbow_min"), ">= 160°"),
    },
    "goddess": {
        "bent_knee_shallow": ("straighter knee angle", _one("straight_knee"), "<= 135°"),
        "knees_uneven": ("difference between knees", _one("knee_diff"), "<= 22°"),
        "stance_narrow": ("stance width (torso lengths)", _one("stance", _R), ">= 1.40"),
        "arms_not_level": ("arm height, left / right shoulder angle", _two("sh_l", "sh_r"), "70-110°"),
        "elbow_angle_wrong": ("elbow angles, left / right", _two("elbow_l", "elbow_r"), "70-110°"),
        "torso_lean": ("torso lean from vertical", _one("lean"), "<= 10°"),
    },
}


def report_rows(pose: str, m: dict, level: int = 0):
    """[(label, value, target, status)] with status 'OK', 'ok (within tolerance)' or 'OUT'."""
    ms = _scalar(m)
    rows = []
    for code, (fn, unit) in checks(pose).items():
        label, val, target = REPORT[pose][code]
        ex = float(fn(ms))
        if ex <= 0:
            st = "OK"
        elif ex - TOL_PER_LEVEL[unit] * level <= 0:
            st = "ok (within tolerance)"
        else:
            st = "OUT"
        rows.append((label, val(ms), target, st, code))
    return rows
