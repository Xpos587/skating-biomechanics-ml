"""Comprehensive biomechanics angle computation (Sports2D-inspired).

Provides:
- ANGLE_DEFS: Joint angle definitions (12 angles)
- SEGMENT_DEFS: Segment angle definitions (14 angles)
- compute_joint_angles(): Compute all joint angles from 2D pose
- compute_segment_angles(): Compute all segment angles from 2D pose
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.types import H36Key
from src.utils.geometry import angle_3pt, segment_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# ANGLE DEFINITIONS
# =============================================================================

# Joint angles: 3-point angles at the middle joint
ANGLE_DEFS: list[dict] = [
    {"name": "R Ankle", "a": H36Key.RKNEE, "v": H36Key.RFOOT, "c": None},
    {"name": "L Ankle", "a": H36Key.LKNEE, "v": H36Key.LFOOT, "c": None},
    {"name": "R Knee", "a": H36Key.RHIP, "v": H36Key.RKNEE, "c": H36Key.RFOOT},
    {"name": "L Knee", "a": H36Key.LHIP, "v": H36Key.LKNEE, "c": H36Key.LFOOT},
    {"name": "R Hip", "a": H36Key.THORAX, "v": H36Key.RHIP, "c": H36Key.RKNEE},
    {"name": "L Hip", "a": H36Key.THORAX, "v": H36Key.LHIP, "c": H36Key.LKNEE},
    {"name": "R Shoulder", "a": None, "v": H36Key.RSHOULDER, "c": H36Key.RELBOW},
    {"name": "L Shoulder", "a": None, "v": H36Key.LSHOULDER, "c": H36Key.LELBOW},
    {"name": "R Elbow", "a": H36Key.RSHOULDER, "v": H36Key.RELBOW, "c": H36Key.RWRIST},
    {"name": "L Elbow", "a": H36Key.LSHOULDER, "v": H36Key.LELBOW, "c": H36Key.LWRIST},
    {"name": "R Wrist", "a": H36Key.RELBOW, "v": H36Key.RWRIST, "c": None},
    {"name": "L Wrist", "a": H36Key.LELBOW, "v": H36Key.LWRIST, "c": None},
]

# Segment angles: angle of a body segment relative to horizontal
SEGMENT_DEFS: list[dict] = [
    {"name": "R Foot", "start": H36Key.RFOOT, "end": None},
    {"name": "L Foot", "start": H36Key.LFOOT, "end": None},
    {"name": "R Shank", "start": H36Key.RKNEE, "end": H36Key.RFOOT},
    {"name": "L Shank", "start": H36Key.LKNEE, "end": H36Key.LFOOT},
    {"name": "R Thigh", "start": H36Key.RHIP, "end": H36Key.RKNEE},
    {"name": "L Thigh", "start": H36Key.LHIP, "end": H36Key.LKNEE},
    {"name": "Pelvis", "start": H36Key.LHIP, "end": H36Key.RHIP},
    {"name": "Trunk", "start": None, "end": None},
    {"name": "Shoulders", "start": H36Key.LSHOULDER, "end": H36Key.RSHOULDER},
    {"name": "Head", "start": None, "end": None},
    {"name": "R Arm", "start": H36Key.RSHOULDER, "end": H36Key.RELBOW},
    {"name": "L Arm", "start": H36Key.LSHOULDER, "end": H36Key.LELBOW},
    {"name": "R Forearm", "start": H36Key.RELBOW, "end": H36Key.RWRIST},
    {"name": "L Forearm", "start": H36Key.LELBOW, "end": H36Key.LWRIST},
]


def compute_joint_angles(pose: NDArray[np.float32]) -> dict[str, float]:
    """Compute all joint angles from a 2D pose.

    Args:
        pose: (17, 2) pose array in pixel or normalized coordinates.

    Returns:
        Dict mapping angle name to degrees [0, 180].
    """
    angles: dict[str, float] = {}

    mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
    mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2

    for definition in ANGLE_DEFS:
        name = definition["name"]
        v_idx = definition["v"]
        a_idx = definition["a"]
        c_idx = definition["c"]

        vertex = pose[v_idx]

        if "Shoulder" in name:
            a = mid_shoulder
            c = pose[c_idx]
        elif a_idx is None or c_idx is None:
            angles[name] = np.nan
            continue
        else:
            a = pose[a_idx]
            c = pose[c_idx]

        try:
            angles[name] = angle_3pt(
                np.asarray(a, dtype=np.float64),
                np.asarray(vertex, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
            )
        except (ValueError, ZeroDivisionError):
            angles[name] = np.nan

    return angles


def compute_segment_angles(pose: NDArray[np.float32]) -> dict[str, float]:
    """Compute all segment angles from a 2D pose.

    Segment angle = angle of the segment relative to horizontal.

    Args:
        pose: (17, 2) pose array in pixel or normalized coordinates.

    Returns:
        Dict mapping segment name to degrees [-180, 180].
    """
    angles: dict[str, float] = {}

    mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
    mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2

    for definition in SEGMENT_DEFS:
        name = definition["name"]
        start = definition["start"]
        end = definition["end"]

        try:
            if name == "Trunk":
                angles[name] = segment_angle(mid_hip, mid_shoulder)
            elif name == "Head":
                angles[name] = segment_angle(mid_shoulder, pose[H36Key.HEAD])
            elif name in ("R Foot", "L Foot"):
                idx = H36Key.RFOOT if "R" in name else H36Key.LFOOT
                knee_idx = H36Key.RKNEE if "R" in name else H36Key.LKNEE
                angles[name] = segment_angle(pose[knee_idx], pose[idx])
            else:
                angles[name] = segment_angle(pose[start], pose[end])
        except (ValueError, ZeroDivisionError, IndexError):
            angles[name] = np.nan

    return angles
