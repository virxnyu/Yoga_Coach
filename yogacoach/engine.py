"""Real-time coaching engine (no camera / MediaPipe dependency -> unit-testable).

Feed it one (13,2) pixel-keypoint array per video frame; it keeps a sliding
window, runs YogaNet, fuses the network's per-error probabilities with the
rule-based geometric evidence and returns stable, human readable feedback.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import rules
from .config import (AUTO_POSE_MIN, ERROR_JOINTS, ERRORS, NUM_ERR, POSE_CONF_MIN, POSE_ERRORS, POSE_TITLES,
                     POSES, TARGET_POSES, WINDOW)
from .geometry import frame_features, measure

FUSE_W_MODEL = 0.6     # weight of the neural network in the hybrid decision
FLAG_THRESHOLD = 0.5


def fuse(p_model: np.ndarray, rule_conf: np.ndarray, w_model: float = FUSE_W_MODEL) -> np.ndarray:
    return w_model * p_model + (1.0 - w_model) * rule_conf


def load_model(path: str):
    import torch
    from .model import YogaNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["poses"] != POSES or ck["errors"] != ERRORS:
        raise ValueError("checkpoint was trained with a different pose/error vocabulary")
    net = YogaNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


@dataclass
class FrameResult:
    status: str                      # warming | no_person | wrong_pose | correct | incorrect
    pose: str | None = None          # pose being judged (target or detected)
    pose_probs: dict = field(default_factory=dict)
    correct_prob: float | None = None
    score: float | None = None       # 0-100 geometric alignment score
    errors: list = field(default_factory=list)     # [(code, fused_score, message)]
    highlight: list = field(default_factory=list)  # joint names to colour red
    measures: dict = field(default_factory=dict)
    hold_seconds: float = 0.0
    note: str = ""


class CoachEngine:
    def __init__(self, model_path: str | None = None, target: str = "auto", mode: str = "hybrid",
                 window: int = WINDOW, smooth: float = 0.45, vote_len: int = 12,
                 wrong_pose_thr: float = 0.15, leniency="lenient", check_pose: bool = False):
        assert mode in ("hybrid", "dl", "rules")
        assert target == "auto" or target in TARGET_POSES, f"target must be 'auto' or one of {TARGET_POSES}"
        self.mode, self.target, self.T = mode, target, window
        self.vote_len = vote_len
        self.smooth, self.wrong_pose_thr = smooth, wrong_pose_thr
        self.level = rules.level_of(leniency)      # 0 strict, 1 normal, 2 lenient
        self.check_pose = check_pose               # warn "this doesn't look like <pose>" in target mode
        self.net = None
        if mode != "rules":
            if not model_path:
                raise ValueError("model_path required unless mode='rules'")
            self.net, _ = load_model(model_path)
        self.reset()

    def reset(self):
        self.kp_buf: deque = deque(maxlen=self.T)
        self.feat_buf: deque = deque(maxlen=self.T)
        self.ema_pose = None
        self.ema_err = None
        self.votes = deque(maxlen=self.vote_len)
        self.missing = 0
        self.correct_since = None
        self.wrong_count = 0

    # ------------------------------------------------------------------ main entry
    def update(self, kp: np.ndarray | None, t: float | None = None) -> FrameResult:
        t = time.monotonic() if t is None else t
        if kp is None or not np.isfinite(kp).all():
            self.missing += 1
            if self.missing > 8:
                self.reset()
            self.correct_since = None
            return FrameResult("no_person", note="Step back so your whole body is visible.")
        self.missing = 0
        kp = np.asarray(kp, np.float32)
        self.kp_buf.append(kp)
        self.feat_buf.append(frame_features(kp))
        if len(self.kp_buf) < self.T:
            return FrameResult("warming", note="Hold still...")

        win_kp = np.stack(self.kp_buf)                         # (T,13,2)
        med_kp = np.median(win_kp, axis=0)
        meas = measure(med_kp)
        meas_s = {k: float(v) for k, v in meas.items()}

        # ---- neural network
        pose_p = np.zeros(len(POSES)); pose_p[POSES.index("other")] = 1.0
        err_p = np.zeros(NUM_ERR)
        correct_p = None
        if self.net is not None:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(np.stack(self.feat_buf))[None]
                pl, cl, el = self.net(x)
            pose_p = torch.softmax(pl, -1)[0].numpy()
            correct_p = float(torch.sigmoid(cl)[0])
            err_p = torch.sigmoid(el)[0].numpy()
        self.ema_pose = pose_p if self.ema_pose is None else self.smooth * pose_p + (1 - self.smooth) * self.ema_pose
        self.ema_err = err_p if self.ema_err is None else self.smooth * err_p + (1 - self.smooth) * self.ema_err

        # ---- which pose are we judging?
        probs = {p: float(self.ema_pose[i]) for i, p in enumerate(POSES)}
        if self.target == "auto":
            rs = {p: rules.pose_score(p, meas_s, self.level) / 100 for p in TARGET_POSES}
            if self.net is None:      # rules-only auto mode: pick the pose with the best geometric score
                pose = max(TARGET_POSES, key=lambda p: rs[p])
                probs = dict(rs)
                best_score = rs[pose]
            elif self.level == 0:     # strict: trust the network only
                pose = max(TARGET_POSES, key=lambda p: probs[p])
                best_score = probs[pose] if POSES[int(np.argmax(self.ema_pose))] != "other" else 0.0
            else:                     # lenient: network probability blended with geometric fit
                comb = {p: 0.6 * probs[p] + 0.4 * rs[p] for p in TARGET_POSES}
                pose = max(TARGET_POSES, key=lambda p: comb[p])
                best_score = comb[pose]
                probs = {**{p: comb[p] for p in TARGET_POSES}, "other": probs["other"]}
            need = AUTO_POSE_MIN if self.level else POSE_CONF_MIN
            if self.net is None:
                need = 0.55
            if best_score < need:
                self.correct_since = None
                return FrameResult("wrong_pose", pose_probs=probs, correct_prob=correct_p, measures=meas_s,
                                   note="No target pose recognised - get into Tree, Warrior II, Triangle or Goddess.")
        else:
            pose = self.target
            if self.check_pose and self.net is not None and probs[pose] < self.wrong_pose_thr:
                self.wrong_count += 1
            else:
                self.wrong_count = 0
            if self.wrong_count >= 10:
                best = max(probs, key=probs.get)
                self.correct_since = None
                nice = POSE_TITLES[best] if best != "other" else "something else"
                return FrameResult("wrong_pose", pose=pose, pose_probs=probs, correct_prob=correct_p,
                                   measures=meas_s, note=f"This doesn't look like {POSE_TITLES[pose]} yet (looks like {nice}).")

        # ---- errors: rules + network fused
        applic = np.zeros(NUM_ERR)
        for c in POSE_ERRORS[pose]:
            applic[ERRORS.index(c)] = 1
        rc = rules.rule_confidence(pose, meas_s, self.level)
        if self.mode == "rules":
            fused = rc
        elif self.mode == "dl":
            fused = self.ema_err
        else:
            fused = fuse(self.ema_err, rc)
        fused = fused * applic
        flagged = fused >= FLAG_THRESHOLD
        if self.mode == "hybrid" and self.level > 0:
            flagged &= rc >= 0.5          # the (relaxed) geometry must agree; the network confirms
        self.votes.append(flagged)
        stable = np.mean(np.stack(self.votes), axis=0) >= 0.5 if len(self.votes) >= 5 else flagged

        errors = []
        for ei in np.argsort(-fused):
            if stable[ei] and applic[ei]:
                code = ERRORS[ei]
                msg = rules.ADVICE[pose][code].format(**meas_s)
                errors.append((code, float(fused[ei]), msg))
        highlight = sorted({j for c, _, _ in errors for j in ERROR_JOINTS[c]})
        score = rules.pose_score(pose, meas_s, self.level)

        if errors:
            self.correct_since = None
            hold = 0.0
        else:
            if self.correct_since is None:
                self.correct_since = t
            hold = t - self.correct_since
        return FrameResult("incorrect" if errors else "correct", pose=pose, pose_probs=probs,
                           correct_prob=correct_p, score=score, errors=errors, highlight=highlight,
                           measures=meas_s, hold_seconds=hold,
                           note="" if errors else rules.GOOD_MESSAGE)
