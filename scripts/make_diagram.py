"""Draw the proposed architecture diagram -> docs/architecture.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "architecture.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8.2))
ax.set_xlim(0, 160); ax.set_ylim(-4, 82); ax.axis("off")

C = dict(io="#E8F1FB", cv="#E7F6EA", dl="#FFF1D6", rule="#F3E5F5", out="#FDE8E8", train="#EEEEEE")


def box(x, y, w, h, title, sub="", color="#fff", fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#444", lw=1.3))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=fs - 2.3, color="#222", linespacing=1.35)


def arrow(x1, y1, x2, y2, txt="", rad=0.0, color="#333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color, connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.8, txt, ha="center", fontsize=8, color=color)


# ---- real-time path (top row)
box(2, 50, 20, 24, "Camera / video", "webcam frame\n(mirrored for\nthe user)", C["io"])
box(30, 50, 25, 24, "MediaPipe Pose", "Landmarker: 33 landmarks\n-> keep 13 joints\n(pixel coordinates)", C["cv"])
box(63, 50, 25, 24, "Feature extractor", "hip-centred, torso-scaled\ncoordinates (26)\n+ 8 joint angles\n+ 5 ratios  = 39 / frame", C["cv"])
box(96, 50, 20, 24, "Sliding window", "last T = 12 frames\n(~0.4 s)\nmedian keypoints", C["cv"])
box(124, 50, 34, 24, "Pose measurements", "knee / elbow / shoulder angles\ntorso lean, stance width,\nwrist-line, wrist height", C["rule"])
for x1, x2 in [(22, 30), (55, 63), (88, 96)]:
    arrow(x1 + 0.4, 62, x2 - 0.4, 62)
arrow(116.4, 62, 123.6, 62)

# ---- YogaNet (middle left)
box(63, 18, 53, 26, "YogaNet  (PyTorch, 0.2 M parameters)",
    "LayerNorm + Linear(39->128)  ->  residual 1-D temporal Conv x2\n->  Bi-GRU(128)  ->  mean+max pooling  ->  shared trunk",
    C["dl"])
box(66, 20.5, 14.5, 8, "Pose head", "4 poses + other", "#fff", fs=8.8)
box(82.5, 20.5, 14.5, 8, "Correct head", "correct / incorrect", "#fff", fs=8.8)
box(99, 20.5, 14.5, 8, "Error head", "11 multi-label", "#fff", fs=8.8)
arrow(106, 50, 106, 44.6)

# ---- rules + fusion
box(124, 18, 34, 26, "Rule checker", "target ranges per pose\n(config.RANGES)\nsoft confidence per error\nnumeric feedback (deg)", C["rule"])
arrow(141, 50, 141, 44.6)

box(63, -2, 53, 15, "Hybrid fusion + temporal voting",
    "score = 0.6 * P_net(error) + 0.4 * C_rule(error)  >=  0.5  ->  error flagged\nmajority vote over the last 12 frames (stable, flicker-free feedback)",
    C["out"], fs=10)
arrow(96, 18.2, 96, 13.6)
arrow(135, 18.2, 116.4, 8, rad=-0.15)

box(124, -2, 34, 15, "Coach overlay", "CORRECT / INCORRECT banner\nred joints + text corrections\nalignment score, hold timer", C["out"], fs=10)
arrow(116.4, 4, 123.6, 4)

# ---- training path (left bottom)
box(2, 18, 52, 26, "Training data (offline)", "", C["train"])
ax.text(28, 36.6, "1. Synthetic kinematic skeleton generator\n   (correct / incorrect joint angles, sensor noise,\n   roll, foreshortening, mirroring)\n2. Real clips recorded with collect.py\n   (labelled correct / error type)", ha="center", va="top", fontsize=8.7, linespacing=1.35)
arrow(54.4, 31, 62.6, 31, "train.py")
ax.text(28, 6, "Offline:  train.py -> models/yogacoach.pt", ha="center", fontsize=9, style="italic", color="#444")
ax.text(80, 79, "Proposed system: real-time yoga pose correction (pose estimation + deep temporal network + geometric rules)",
        ha="center", fontsize=12.5, fontweight="bold")
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", OUT)
