"""Unit / integration tests (no camera, no MediaPipe needed).

    python -m pytest -q tests          # or: python tests/test_pipeline.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yogacoach import dataio, rules, synth  # noqa: E402
from yogacoach.config import ERRORS, POSE_ERRORS, POSES, TARGET_POSES, WINDOW  # noqa: E402
from yogacoach.engine import CoachEngine  # noqa: E402
from yogacoach.geometry import angle3, frame_features, measure, mirror, normalize  # noqa: E402


def test_angle3_known_values():
    a, b, c = np.array([1., 0]), np.array([0., 0]), np.array([0., 1])
    assert abs(float(angle3(a, b, c)) - 90) < 0.1
    assert abs(float(angle3(np.array([-1., 0]), b, a)) - 180) < 0.5


def test_features_are_scale_translation_invariant():
    d = synth.generate(4, seed=1, T=3)
    kp = d["kp"][:6]
    f1 = frame_features(kp)
    f2 = frame_features(kp * 3.7 + np.array([55., -12.]))
    assert np.allclose(f1, f2, atol=1e-4)


def test_mirror_swaps_sides_but_not_symmetric_measures():
    d = synth.generate(3, seed=2, T=2)
    kp = d["kp"][:8, 0]
    m1, m2 = measure(kp), measure(mirror(kp))
    for key in ("lean", "stance", "straight_knee", "bent_knee", "elbow_min", "wrist_line", "knee_diff"):
        assert np.allclose(m1[key], m2[key], atol=1e-3), key
    assert np.allclose(m1["knee_l"], m2["knee_r"], atol=1e-3)


def test_generator_labels_match_rules_and_shapes():
    d = synth.generate(40, seed=3)
    assert d["X"].shape == (40 * len(POSES), WINDOW, 39)
    assert d["errors"].shape[1] == len(ERRORS)
    for pi, pose in enumerate(POSES):
        if pose == "other":
            continue
        sel = d["pose"] == pi
        # errors only from the pose's own vocabulary
        allowed = {ERRORS.index(e) for e in POSE_ERRORS[pose]}
        used = set(np.where(d["errors"][sel].sum(0) > 0)[0])
        assert used <= allowed, (pose, used - allowed)
        # both correct and incorrect examples exist
        has_err = d["errors"][sel].sum(1) > 0
        assert has_err.any() and (~has_err).any()


def test_rules_accept_ideal_and_reject_broken():
    rng = np.random.default_rng(0)
    for pose in TARGET_POSES:
        ok = bad = 0
        for _ in range(60):
            p = synth.SAMPLERS[pose](rng)
            kp = synth._transform(synth.build_skeleton(p), rng)
            m = measure(kp[None])
            if not rules.unambiguous(pose, m)[0]:
                continue
            n = int(rules.labels(pose, m)[0].sum())
            ok += n == 0
            bad += n > 0
        assert ok > 3 and bad > 3, pose


def test_dataio_roundtrip_and_validation():
    rng = np.random.default_rng(0)
    kp = (rng.normal(size=(40, 13, 2)) * 50 + 300).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        dataio.save_clip(td, kp, "tree", "arms_low,torso_lean", 30.0)
        dataio.save_clip(td, kp, "other", "none", 30.0)
        d = dataio.load_real(td)
        assert d["X"].shape[1:] == (WINDOW, 39)
        assert d["errors"][:, ERRORS.index("arms_low")].max() == 1
        try:
            dataio.save_clip(td, kp, "tree", "knees_uneven", 30.0)
            raise AssertionError("invalid label accepted")
        except ValueError:
            pass


def _run_engine(pose, mode="rules", frames=40, seed=0, want_correct=True, model=None):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        p = synth.SAMPLERS[pose](rng)
        kp0 = synth._transform(synth.build_skeleton(p), rng)
        m = measure(kp0[None])
        if not rules.unambiguous(pose, m)[0]:
            continue
        lab = rules.labels(pose, m)[0]
        if (lab.sum() == 0) == want_correct:
            break
    seq = synth._to_sequence(kp0, rng, frames, noise=1.0)
    eng = CoachEngine(model, target=pose, mode=mode)
    res = None
    for f in seq:
        res = eng.update(f)
    return res, lab


def test_engine_rules_mode_correct_and_incorrect():
    for pose in TARGET_POSES:
        res, _ = _run_engine(pose, want_correct=True)
        assert res.status == "correct", (pose, res.status, res.errors)
        res, lab = _run_engine(pose, want_correct=False)
        assert res.status == "incorrect", (pose, res.status)
        flagged = {c for c, _, _ in res.errors}
        truth = {ERRORS[i] for i in np.where(lab)[0]}
        assert flagged & truth, (pose, flagged, truth)


def test_engine_handles_missing_person():
    eng = CoachEngine(None, target="tree", mode="rules")
    assert eng.update(None).status == "no_person"
    assert eng.update(np.full((13, 2), np.nan)).status == "no_person"
    assert eng.update(np.random.rand(13, 2).astype(np.float32) * 100).status == "warming"


def test_engine_with_trained_model():
    ck = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yogacoach.pt")
    if not os.path.isfile(ck):
        return
    for pose in TARGET_POSES:
        res, _ = _run_engine(pose, mode="hybrid", want_correct=True, model=ck)
        assert res.status == "correct", (pose, res.status, res.errors)
        res, lab = _run_engine(pose, mode="hybrid", want_correct=False, model=ck)
        assert res.status == "incorrect", (pose, res.status)
        assert {c for c, _, _ in res.errors} & {ERRORS[i] for i in np.where(lab)[0]}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
