# Real-Time Yoga Pose Correction (Deep Learning Techniques project)

**Task:** identify the *incorrect* yoga pose (4 poses) from real-time video and give feedback.

Poses: **Tree**, **Warrior II**, **Triangle**, **Goddess** (front-facing camera, whole body visible).

Team: Viranyu Srinivasan, Jeffrey Jayapackiyaraj, Aditya Agrawal - NIT Tiruchirappalli

## How it works

```
webcam -> MediaPipe Pose Landmarker (33 landmarks -> 13 joints, pixel coords)
       -> 39 scale-invariant features / frame -> sliding window of 12 frames
       -> YogaNet (conv + Bi-GRU, 0.23 M params)  : pose | correct? | 11 error types
       -> + geometric rule checker (angles vs target ranges)
       -> fusion (0.6 net + 0.4 rules) + temporal voting
       -> overlay: CORRECT / INCORRECT, red joints, "Bend your front knee deeper (now 148°)"
```

Architecture figure: `docs/architecture.png`. Code map:

| File | Purpose |
|---|---|
| `yogacoach/config.py` | keypoints, poses, error vocabulary, **target angle ranges** (edit to tune) |
| `yogacoach/geometry.py` | angles, measurements, 39-D features |
| `yogacoach/rules.py` | rule checker + feedback texts |
| `yogacoach/synth.py` | synthetic kinematic data generator (correct/incorrect) |
| `yogacoach/model.py` | YogaNet and multi-task loss |
| `yogacoach/engine.py` | real-time engine (window, fusion, smoothing) |
| `yogacoach/pose_backend.py` | MediaPipe wrapper (auto-downloads `pose_landmarker_full.task`) |
| `train.py` | training + evaluation + plots |
| `realtime.py` | live demo (webcam or video file) |
| `collect.py`, `scripts/extract_dataset.py` | record / convert your own labelled data |
| `scripts/build_report.py` | builds the Word report from `team.json` + `outputs/` |
| `tests/test_pipeline.py` | unit/integration tests (no camera needed) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py                           # 8 tests should pass
```

## Run the live demo

A trained checkpoint is in `models/yogacoach.pt` (trained on synthetic data - see "Improving it with real data").
The first run downloads the MediaPipe pose model (~9 MB, needs internet; or download
`pose_landmarker_full.task` manually into `models/` or pass `--pose-model`).

```bash
python realtime.py                    # webcam, auto-detect the pose
python realtime.py --pose warrior2    # judge a specific pose
python realtime.py --source video.mp4 --save out.mp4 --no-display
python realtime.py --mode rules --model none   # geometry-only fallback (no PyTorch needed)
```

Keys: `q` quit, `0` auto, `1` Tree, `2` Warrior II, `3` Triangle, `4` Goddess, `l` cycle leniency, `d` show raw numbers, `s` screenshot (saved to `outputs/screenshots/`).

**Leniency.** The default is `--leniency lenient`: a pose counts as correct even if angles are up to ~12 degrees
(0.12 torso lengths) outside the ideal ranges. Use `--leniency normal` (6 deg) or `strict` (0) to tighten it.
In auto-detect mode the pose is chosen from the network probability blended with how well the body geometry fits each pose.
`--check-pose` additionally warns when the network thinks you are in a different pose than the one selected.
`--debug` (or key `d`) draws the measured angles and pose scores so you can see *why* it decided what it did.
Stand 2-3 m away with your **whole body in frame, facing the camera**, good lighting.

## Check photos (image upload)

```bash
python check_image.py            # opens http://127.0.0.1:8765 : drag & drop, choose, or paste (Ctrl+V) photos
python check_image.py photo.jpg --pose tree          # command line report
python check_image.py --folder my_photos --pose warrior2   # whole folder -> summary + outputs/image_checks/summary.csv
```
The page shows the annotated photo (joint angles in white, wrong joints in red), the verdict, what to fix, the pose scores
and a table of every measured angle against its target. You can switch pose, leniency and decision mode and re-check.
Use clear **front-view, full-body** photos; side views, cropped bodies or tiny figures can give wrong answers (the page warns you).
Results are also saved to `outputs/image_checks/`.

## Improving it with real data (strongly recommended before the demo)

The shipped model is bootstrapped on synthetic skeletons. Real people differ, so record some clips:

```bash
python collect.py --list                                  # valid pose/error names
python collect.py --pose tree --label correct --seconds 15
python collect.py --pose tree --label bent_knee_shallow
python collect.py --pose warrior2 --label arms_not_level,torso_lean
python collect.py --pose other --label none               # standing / walking / random movement
python train.py --real-dir data/real                      # retrains and evaluates on held-out real clips
```

Aim for >= 3 clips per (pose, label), different people, clothes and distances. With >= 8 clips the script holds out
clips for validation/test and reports real-world metrics (`real_test`). Deliberately perform each mistake;
the label names are in `yogacoach/config.py` (`POSE_ERRORS`).

## Report

1. Put names/roll numbers/e-mails/phones/GitHub link/contributions in `team.json`.
2. Take real screenshots with `realtime.py` (key `s`).
3. `python scripts/build_report.py` -> `Yoga_Pose_Correction_Report.docx` (sections in the order of the assignment sheet).

## Honest notes / limitations

* Synthetic-test numbers measure label recovery on generated skeletons, not accuracy on real people.
  Synthetic labels use the same target ranges as the rule checker, so rule-only scores there are not independent evidence.
* 2-D, single camera, front view. Target ranges are common alignment cues, not instructor-certified; tune them in `config.py`.
* The live MediaPipe path needs internet once (model download) and a camera; everything else is covered by tests.
