"""Build COCO-style HALPE26 26kp annotation JSON for mmpose training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

HALPE26_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "left_big_toe",
    "left_small_toe",
    "right_heel",
    "right_big_toe",
    "right_small_toe",
    "left_eye_inner",
    "right_eye_inner",
    "mouth",
]

HALPE26_SKELETON = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # head
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],  # upper body
    [5, 11],
    [6, 12],
    [11, 12],  # torso
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],  # legs
    [15, 17],
    [16, 20],
    [17, 18],
    [18, 19],  # left foot
    [20, 21],
    [21, 22],  # right foot
    [23, 24],
    [24, 25],  # face dupes
]

DEFAULT_CATEGORY = {
    "supercategory": "person",
    "id": 1,
    "name": "person",
    "keypoint_names": HALPE26_KEYPOINT_NAMES,
    "skeleton": HALPE26_SKELETON,
}

# Face dupe mapping: HALPE26 idx 23-25 copy from these indices
_FACE_DUPE_SOURCE = {23: 1, 24: 2, 25: 0}


def merge_coco_foot_keypoints(
    coco_2d: NDArray[np.float64],
    foot_2d: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Merge COCO 17kp + foot 6kp + face 3dupes into HALPE26 26kp.

    Args:
        coco_2d: (17, 2) COCO keypoints from _coco.npy (may contain NaN).
        foot_2d: (6, 2) projected foot keypoints (may contain NaN).

    Returns:
        pts: (26, 2) merged HALPE26 coordinates.
        vis: (26,) visibility per keypoint.
    """
    pts = np.zeros((26, 2), dtype=np.float32)
    vis = np.zeros(26, dtype=np.float32)

    # COCO 17kp (indices 0-16): visibility based on NaN check
    for i in range(17):
        if not np.isnan(coco_2d[i]).any():
            pts[i] = coco_2d[i]
            vis[i] = 2.0  # COCO standard: visible
        else:
            vis[i] = 0.0

    # Foot 6kp (indices 17-22)
    for i in range(6):
        if not np.isnan(foot_2d[i]).any():
            pts[17 + i] = foot_2d[i]
            vis[17 + i] = 2.0
        else:
            vis[17 + i] = 0.0

    # Face duplicates (indices 23-25): copy from existing, low visibility
    for dup_idx, src_idx in _FACE_DUPE_SOURCE.items():
        pts[dup_idx] = pts[src_idx]
        vis[dup_idx] = 0.3 if vis[src_idx] > 0 else 0.0

    return pts, vis


def format_keypoints(
    pts: NDArray[np.float32],
    vis: NDArray[np.float32],
) -> list[float]:
    """Flatten keypoints to COCO format: [x1, y1, v1, x2, y2, v2, ...]."""
    kp: list[float] = []
    for i in range(26):
        kp.extend([float(pts[i, 0]), float(pts[i, 1]), float(vis[i])])
    return kp


def build_coco_json(
    images: list[dict],
    annotations: list[dict],
) -> dict:
    """Build a COCO-style annotation dict."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": [DEFAULT_CATEGORY],
    }


def save_coco_json(data: dict, output_path: str) -> None:
    """Save COCO JSON to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(data, f)
