"""Central configuration: keypoints, poses, error vocabulary, alignment ranges.

All geometry downstream works on 13 body keypoints taken from MediaPipe's
33-landmark BlazePose topology.  Ranges below are *configurable* alignment
tolerances derived from standard yoga alignment cues; they are used (a) by the
rule-based checker, (b) by the synthetic-data generator to decide what counts
as "correct" vs "incorrect", and (c) to give numeric feedback ("target 85-125").
"""
from __future__ import annotations

# ----------------------------------------------------------------- keypoints
# MediaPipe BlazePose landmark ids we keep
KP_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
KP_NAMES = ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr",
            "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"]
K = {n: i for i, n in enumerate(KP_NAMES)}
NUM_KP = len(KP_NAMES)

# Skeleton edges (indices into KP_NAMES) used for drawing
EDGES = [("l_sh", "r_sh"), ("l_hip", "r_hip"), ("l_sh", "l_hip"), ("r_sh", "r_hip"),
         ("l_sh", "l_el"), ("l_el", "l_wr"), ("r_sh", "r_el"), ("r_el", "r_wr"),
         ("l_hip", "l_kn"), ("l_kn", "l_an"), ("r_hip", "r_kn"), ("r_kn", "r_an")]

# ----------------------------------------------------------------- classes
POSES = ["tree", "warrior2", "triangle", "goddess", "other"]
NUM_POSES = len(POSES)
POSE_TITLES = {
    "tree": "Tree Pose (Vrikshasana)",
    "warrior2": "Warrior II (Virabhadrasana II)",
    "triangle": "Triangle Pose (Trikonasana)",
    "goddess": "Goddess Pose (Utkata Konasana)",
    "other": "No target pose",
}
TARGET_POSES = POSES[:4]          # the four yoga poses that are coached

# Global error vocabulary (multi-label head)
ERRORS = [
    "straight_knee_bent",   # a leg that should be straight is bent
    "bent_knee_shallow",    # a leg that should be bent is not bent enough
    "arms_low",             # arms not raised enough
    "arms_not_level",       # arms not horizontal (warrior)
    "elbows_bent",          # arms should be straight
    "torso_lean",           # torso should be upright
    "stance_narrow",        # feet too close together
    "lean_insufficient",    # triangle: torso not tilted far enough
    "arms_not_stacked",     # triangle: arms not on one vertical line
    "knees_uneven",         # goddess: two knees bent differently
    "elbow_angle_wrong",    # goddess: cactus arms not ~90 deg
]
NUM_ERR = len(ERRORS)
E = {n: i for i, n in enumerate(ERRORS)}

# Which errors are meaningful for which pose
POSE_ERRORS = {
    "tree": ["straight_knee_bent", "bent_knee_shallow", "arms_low", "elbows_bent", "torso_lean"],
    "warrior2": ["straight_knee_bent", "bent_knee_shallow", "arms_not_level",
                 "elbows_bent", "torso_lean", "stance_narrow"],
    "triangle": ["straight_knee_bent", "stance_narrow", "lean_insufficient", "arms_not_stacked",
                 "elbows_bent"],
    "goddess": ["bent_knee_shallow", "knees_uneven", "stance_narrow",
                "arms_not_level", "elbow_angle_wrong", "torso_lean"],
}

# ----------------------------------------------------------------- ranges
# Measured quantities (see geometry.measure):
#   straight_knee, bent_knee      : interior knee angle (deg), straight = larger of the two
#   sh_l, sh_r                    : hip-shoulder-elbow angle (deg); 0=arm along torso, 90=horizontal, 180=overhead
#   elbow_min                     : smaller elbow angle (deg); 180 = straight arm
#   lean                          : torso angle from vertical (deg)
#   stance                        : ankle distance / torso length
#   wrist_above_nose              : (nose_y - wrist_y)/torso length, smaller of the two wrists
#   wrist_line                    : angle of wrist-to-wrist line from vertical (deg)
#   knee_diff                     : |knee_l - knee_r| (deg)
#   wrist_above_elbow             : (elbow_y - wrist_y)/torso length, smaller of the two arms
RANGES = {
    "tree": {
        "straight_knee": (165, 180),
        "bent_knee": (20, 100),
        "wrist_above_nose": (0.10, 9.0),
        "elbow_min": (150, 180),
        "lean": (0, 10),
    },
    "warrior2": {
        "bent_knee": (85, 125),
        "straight_knee": (165, 180),
        "sh_both": (75, 105),
        "elbow_min": (160, 180),
        "lean": (0, 10),
        "stance": (1.7, 9.0),
    },
    "triangle": {
        "straight_knee": (165, 180),   # applies to the smaller (weaker) knee too
        "knee_min": (160, 180),
        "lean": (35, 95),
        "stance": (1.6, 9.0),
        "wrist_line": (0, 22),
        "elbow_min": (160, 180),
    },
    "goddess": {
        "bent_knee": (95, 135),         # applies to both knees
        "knee_diff": (0, 22),
        "stance": (1.4, 9.0),
        "sh_both": (70, 110),
        "elbow_both": (70, 110),
        "lean": (0, 10),
    },
}

# ----------------------------------------------------------------- model / data
WINDOW = 12            # frames per model window
MIN_VISIBILITY = 0.4   # mean landmark visibility needed to trust a video frame
MIN_VISIBILITY_IMAGE = 0.25  # more forgiving for still photos
POSE_CONF_MIN = 0.55   # (strict) below this the engine says "no pose recognised"
AUTO_POSE_MIN = 0.40   # lenient auto-detect threshold on the combined pose score

# Error -> joints to highlight in red on the overlay
ERROR_JOINTS = {
    "straight_knee_bent": ["l_kn", "r_kn"],
    "bent_knee_shallow": ["l_kn", "r_kn"],
    "arms_low": ["l_wr", "r_wr"],
    "arms_not_level": ["l_wr", "r_wr"],
    "elbows_bent": ["l_el", "r_el"],
    "torso_lean": ["l_sh", "r_sh", "l_hip", "r_hip"],
    "stance_narrow": ["l_an", "r_an"],
    "lean_insufficient": ["l_sh", "r_sh"],
    "arms_not_stacked": ["l_wr", "r_wr"],
    "knees_uneven": ["l_kn", "r_kn"],
    "elbow_angle_wrong": ["l_el", "r_el"],
}
