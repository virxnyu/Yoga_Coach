"""MediaPipe Pose Landmarker wrapper (Tasks API, mediapipe >= 0.10.x).

Returns the 13 keypoints used by the project in *pixel* coordinates
(x * width, y * height) so that angles are not distorted by the aspect ratio.
The .task model file is downloaded once (about 9 MB for 'full').
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from .config import KP_IDS, MIN_VISIBILITY

_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def ensure_model(variant: str = "full", model_path: str | None = None) -> str:
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        return model_path
    assert variant in ("lite", "full", "heavy")
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.isfile(path) or os.path.getsize(path) < 1_000_000:
        url = _URL.format(v=variant)
        print(f"[pose] downloading {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # noqa: BLE001
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(
                f"Could not download the MediaPipe pose model ({e}).\n"
                f"Download it manually from:\n  {url}\nand save it as:\n  {path}\n"
                f"(or pass --pose-model /path/to/file.task)") from e
    return path


class PoseBackend:
    def __init__(self, variant: str = "full", model_path: str | None = None, video: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.video = video
        path = ensure_model(variant, model_path)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=0.5 if video else 0.3,
            min_pose_presence_confidence=0.5 if video else 0.3,
            min_tracking_confidence=0.5)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None, min_vis: float | None = None):
        """-> (kp (13,2) float32 pixels or None, mean visibility)."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = int(timestamp_ms or 0)
            ts = max(ts, self._last_ts + 1)       # must be strictly increasing
            self._last_ts = ts
            res = self.landmarker.detect_for_video(img, ts)
        else:
            res = self.landmarker.detect(img)
        if not res.pose_landmarks:
            return None, 0.0
        lm = res.pose_landmarks[0]
        kp = np.array([[lm[i].x * w, lm[i].y * h] for i in KP_IDS], np.float32)
        vis = float(np.mean([lm[i].visibility for i in KP_IDS]))
        if vis < (MIN_VISIBILITY if min_vis is None else min_vis):
            return None, vis
        return kp, vis

    def close(self):
        self.landmarker.close()
