"""Image checker tests with a fake pose backend (no MediaPipe / model download needed)."""
import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import check_image  # noqa: E402
from yogacoach import rules, synth  # noqa: E402
from yogacoach.analyze import analyze_image, load_image_bytes  # noqa: E402
from yogacoach.geometry import measure  # noqa: E402

MODEL = os.path.join(ROOT, "models", "yogacoach.pt")


class FakeBackend:
    """Returns a fixed skeleton scaled into the image, ignoring pixels."""
    def __init__(self, kp_norm):
        self.kp = kp_norm

    def detect(self, img, ts=None, min_vis=None):
        h, w = img.shape[:2]
        k = self.kp - self.kp.mean(0)
        s = 0.8 * h / (k[:, 1].max() - k[:, 1].min())
        return (k * s + np.array([w / 2, h / 2])).astype(np.float32), 0.9


def skeleton(pose, correct, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        p = synth.SAMPLERS[pose](rng)
        kp = synth._transform(synth.build_skeleton(p), rng)
        m = measure(kp[None])
        if not rules.unambiguous(pose, m)[0]:
            continue
        if (rules.labels(pose, m)[0].sum() == 0) == correct:
            return kp
    raise RuntimeError


def png_bytes(w=480, h=640):
    ok, b = cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))
    return b.tobytes()


def test_analyze_image_correct_and_incorrect():
    img = load_image_bytes(png_bytes())
    for pose in ("tree", "warrior2", "triangle", "goddess"):
        rep = analyze_image(img, FakeBackend(skeleton(pose, True)), MODEL, pose, "hybrid", "lenient")
        assert rep.status == "correct", (pose, rep.text)
        assert rep.rows and rep.annotated.shape[:2] == img.shape[:2]
        rep = analyze_image(img, FakeBackend(skeleton(pose, False, 1)), MODEL, pose, "hybrid", "strict")
        assert rep.status == "incorrect", (pose, rep.text)
        assert rep.feedback and any(r[3] == "OUT" for r in rep.rows)


def test_auto_detect_and_no_person():
    img = load_image_bytes(png_bytes())
    rep = analyze_image(img, FakeBackend(skeleton("warrior2", True)), MODEL, "auto", "hybrid", "lenient")
    assert rep.pose == "warrior2" and rep.status == "correct", rep.text

    class Nobody:
        def detect(self, *a, **k):
            return None, 0.0
    rep = analyze_image(img, Nobody(), MODEL)
    assert rep.status == "no_person" and rep.annotated is not None


def test_lenient_accepts_slightly_off_but_strict_rejects():
    """Raise a warrior's front knee just outside the ideal range: lenient ok, strict flags it."""
    rng = np.random.default_rng(5)
    from yogacoach.synth import build_skeleton, sample_warrior2, _transform
    for _ in range(3000):
        p = sample_warrior2(rng)
        kp = _transform(build_skeleton(p), rng)
        m = measure(kp[None])
        v = float(m["bent_knee"][0])
        lab = rules.labels("warrior2", m)[0].sum()
        # front knee between 125 and 133 deg and nothing else wrong -> within 8 deg of the range
        names = [c for c, f in zip(rules.ERRORS, rules.labels("warrior2", m)[0]) if f]
        if names == ["bent_knee_shallow"] and 127 < v < 133:
            break
    else:
        raise RuntimeError("no sample")
    img = load_image_bytes(png_bytes())
    strict = analyze_image(img, FakeBackend(kp), MODEL, "warrior2", "rules", "strict")
    lenient = analyze_image(img, FakeBackend(kp), MODEL, "warrior2", "rules", "lenient")
    assert strict.status == "incorrect" and lenient.status == "correct", (strict.text, lenient.text)


def test_http_roundtrip():
    backend = FakeBackend(skeleton("tree", True))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), check_image.make_handler(backend, MODEL, td))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            assert "Yoga Image Checker" in html
            req = urllib.request.Request(f"http://127.0.0.1:{port}/analyze?pose=tree&leniency=lenient&mode=hybrid&name=a%20b.png",
                                         data=png_bytes(), method="POST")
            r = json.loads(urllib.request.urlopen(req).read())
            assert r["status"] == "correct" and r["image"].startswith("data:image/jpeg;base64,") and r["rows"]
            assert os.path.isfile(os.path.join(td, "a_b_checked.jpg"))
            bad = urllib.request.Request(f"http://127.0.0.1:{port}/analyze?pose=nope", data=png_bytes(), method="POST")
            try:
                urllib.request.urlopen(bad)
                raise AssertionError("bad pose accepted")
            except urllib.error.HTTPError as e:
                assert e.code == 400
            junk = urllib.request.Request(f"http://127.0.0.1:{port}/analyze?pose=tree", data=b"not an image", method="POST")
            r = json.loads(urllib.request.urlopen(junk).read())
            assert r["status"] == "no_person"
        finally:
            srv.shutdown()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)
