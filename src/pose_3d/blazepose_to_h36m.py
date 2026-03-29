"""Keypoint mapping from BlazePose 33 to H3.6M 17 format.

This module converts BlazePose (MediaPipe) 33-keypoint format to the
H3.6M 17-keypoint format used by AthletePose3D models.

BlazePose Keypoints (33):
    0: nose
    1: left_eye_inner, 2: left_eye, 3: left_eye_outer
    4: right_eye_inner, 5: right_eye, 6: right_eye_outer
    7: left_ear, 8: right_ear
    9: mouth_left, 10: mouth_right
    11: left_shoulder, 12: right_shoulder
    13: left_elbow, 14: right_elbow
    15: left_wrist, 16: right_wrist
    17: left_pinky, 18: right_pinky
    19: left_index, 20: right_index
    21: left_thumb, 22: right_thumb
    23: left_hip, 24: right_hip
    25: left_knee, 26: right_knee
    27: left_ankle, 28: right_ankle
    29: left_heel, 30: right_heel
    31: left_foot_index, 32: right_foot_index

H3.6M Keypoints (17):
    0: hip (center)
    1: rhip, 2: rknee, 3: rfoot
    4: lhip, 5: lknee, 6: lfoot
    7: spine, 8: thorax, 9: neck, 10: head
    11: lshoulder, 12: lelbow, 13: lwrist
    14: rshoulder, 15: relbow, 16: rwrist
"""

import numpy as np


# BlazePose keypoint indices
class BKey:
    """BlazePose keypoint indices (33 total)."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# H3.6M keypoint indices
class H36Key:
    """H3.6M keypoint indices (17 total)."""

    HIP_CENTER = 0
    RHIP = 1
    RKNEE = 2
    RFOOT = 3
    LHIP = 4
    LKNEE = 5
    LFOOT = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LSHOULDER = 11
    LELBOW = 12
    LWRIST = 13
    RSHOULDER = 14
    RELBOW = 15
    RWRIST = 16


def _mid_point(pose: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
    """Calculate midpoint between two keypoints."""
    return (pose[idx1] + pose[idx2]) / 2


def blazepose_to_h36m(blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints.

    Args:
        blazepose_pose: (33, 2), (33, 3), or (N, 33, 2) array
            - If 2D: Last dim is (x, y)
            - If 3D: Last dim is (x, y, confidence)
            - If batched: First dim is frame count

    Returns:
        h36m_pose: (17, 2), (17, 3), or (N, 17, 2) array
            - Same format as input, but with 17 keypoints

    Raises:
        ValueError: If input shape is invalid
    """
    # Handle batched input
    if blazepose_pose.ndim == 3:
        n_frames = blazepose_pose.shape[0]
        h36m_poses = np.zeros((n_frames, 17, blazepose_pose.shape[2]), dtype=blazepose_pose.dtype)
        for i in range(n_frames):
            h36m_poses[i] = blazepose_to_h36m(blazepose_pose[i])
        return h36m_poses

    # Single frame processing
    if blazepose_pose.shape[0] != 33:
        raise ValueError(f"Expected 33 keypoints, got {blazepose_pose.shape[0]}")

    has_confidence = blazepose_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=blazepose_pose.dtype)

    # Midpoints calculation
    mid_hip = _mid_point(blazepose_pose, BKey.LEFT_HIP, BKey.RIGHT_HIP)
    mid_shoulder = _mid_point(blazepose_pose, BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER)

    # Map each H3.6M keypoint
    mapping = [
        # H3.6M index: (source_func, fallback_indices)
        (H36Key.HIP_CENTER, lambda: mid_hip, [BKey.LEFT_HIP, BKey.RIGHT_HIP]),
        (H36Key.RHIP, lambda: blazepose_pose[BKey.RIGHT_HIP], [BKey.RIGHT_HIP]),
        (H36Key.RKNEE, lambda: blazepose_pose[BKey.RIGHT_KNEE], [BKey.RIGHT_KNEE]),
        (H36Key.RFOOT, lambda: blazepose_pose[BKey.RIGHT_ANKLE], [BKey.RIGHT_ANKLE]),
        (H36Key.LHIP, lambda: blazepose_pose[BKey.LEFT_HIP], [BKey.LEFT_HIP]),
        (H36Key.LKNEE, lambda: blazepose_pose[BKey.LEFT_KNEE], [BKey.LEFT_KNEE]),
        (H36Key.LFOOT, lambda: blazepose_pose[BKey.LEFT_ANKLE], [BKey.LEFT_ANKLE]),
        (
            H36Key.SPINE,
            lambda: mid_shoulder * 0.5 + mid_hip * 0.5,
            [BKey.LEFT_HIP, BKey.RIGHT_HIP, BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER],
        ),
        (H36Key.THORAX, lambda: mid_shoulder, [BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER]),
        (H36Key.NECK, lambda: blazepose_pose[BKey.NOSE], [BKey.NOSE]),
        (H36Key.HEAD, lambda: blazepose_pose[BKey.NOSE], [BKey.NOSE]),
        (H36Key.LSHOULDER, lambda: blazepose_pose[BKey.LEFT_SHOULDER], [BKey.LEFT_SHOULDER]),
        (H36Key.LELBOW, lambda: blazepose_pose[BKey.LEFT_ELBOW], [BKey.LEFT_ELBOW]),
        (H36Key.LWRIST, lambda: blazepose_pose[BKey.LEFT_WRIST], [BKey.LEFT_WRIST]),
        (H36Key.RSHOULDER, lambda: blazepose_pose[BKey.RIGHT_SHOULDER], [BKey.RIGHT_SHOULDER]),
        (H36Key.RELBOW, lambda: blazepose_pose[BKey.RIGHT_ELBOW], [BKey.RIGHT_ELBOW]),
        (H36Key.RWRIST, lambda: blazepose_pose[BKey.RIGHT_WRIST], [BKey.RIGHT_WRIST]),
    ]

    for h36_idx, source_func, fallback_indices in mapping:
        h36m_pose[h36_idx] = source_func()

    return h36m_pose


def h36m_to_blazepose(h36m_pose: np.ndarray) -> np.ndarray:
    """Convert H3.6M 17 keypoints back to BlazePose 33 format.

    Missing keypoints are filled with zeros or interpolated.

    Args:
        h36m_pose: (17, 2), (17, 3), or (N, 17, 2) array

    Returns:
        blazepose_pose: (33, 2), (33, 3), or (N, 33, 2) array
    """
    # Handle batched input
    if h36m_pose.ndim == 3:
        n_frames = h36m_pose.shape[0]
        blazepose_poses = np.zeros((n_frames, 33, h36m_pose.shape[2]), dtype=h36m_pose.dtype)
        for i in range(n_frames):
            blazepose_poses[i] = h36m_to_blazepose(h36m_pose[i])
        return blazepose_poses

    has_confidence = h36m_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    blazepose_pose = np.zeros((33, n_channels), dtype=h36m_pose.dtype)

    # Direct mappings
    direct_mapping = [
        (H36Key.HEAD, BKey.NOSE),
        (H36Key.LSHOULDER, BKey.LEFT_SHOULDER),
        (H36Key.RSHOULDER, BKey.RIGHT_SHOULDER),
        (H36Key.LELBOW, BKey.LEFT_ELBOW),
        (H36Key.RELBOW, BKey.RIGHT_ELBOW),
        (H36Key.LWRIST, BKey.LEFT_WRIST),
        (H36Key.RWRIST, BKey.RIGHT_WRIST),
        (H36Key.LHIP, BKey.LEFT_HIP),
        (H36Key.RHIP, BKey.RIGHT_HIP),
        (H36Key.LKNEE, BKey.LEFT_KNEE),
        (H36Key.RKNEE, BKey.RIGHT_KNEE),
        (H36Key.LFOOT, BKey.LEFT_ANKLE),
        (H36Key.RFOOT, BKey.RIGHT_ANKLE),
    ]

    for h36_idx, bp_idx in direct_mapping:
        blazepose_pose[bp_idx] = h36m_pose[h36_idx]

    # Interpolate missing keypoints (eyes, ears, mouth, hands)
    # Use neighboring joints with small offsets
    blazepose_pose[BKey.LEFT_EYE] = h36m_pose[H36Key.HEAD] + np.array([-0.02, 0.02, 0])[:n_channels]
    blazepose_pose[BKey.RIGHT_EYE] = h36m_pose[H36Key.HEAD] + np.array([0.02, 0.02, 0])[:n_channels]
    blazepose_pose[BKey.LEFT_EAR] = h36m_pose[H36Key.HEAD] + np.array([-0.05, 0, 0])[:n_channels]
    blazepose_pose[BKey.RIGHT_EAR] = h36m_pose[H36Key.HEAD] + np.array([0.05, 0, 0])[:n_channels]

    return blazepose_pose


# H3.6M keypoint names for visualization/debugging
H36M_KEYPOINT_NAMES = [
    "hip_center",
    "rhip",
    "rknee",
    "rfoot",
    "lhip",
    "lknee",
    "lfoot",
    "spine",
    "thorax",
    "neck",
    "head",
    "lshoulder",
    "lelbow",
    "lwrist",
    "rshoulder",
    "relbow",
    "rwrist",
]

# H3.6M skeleton connections for visualization
H36M_SKELETON_EDGES = [
    # Torso
    (H36Key.HIP_CENTER, H36Key.SPINE),
    (H36Key.SPINE, H36Key.THORAX),
    (H36Key.THORAX, H36Key.NECK),
    (H36Key.NECK, H36Key.HEAD),
    # Right arm
    (H36Key.THORAX, H36Key.RSHOULDER),
    (H36Key.RSHOULDER, H36Key.RELBOW),
    (H36Key.RELBOW, H36Key.RWRIST),
    # Left arm
    (H36Key.THORAX, H36Key.LSHOULDER),
    (H36Key.LSHOULDER, H36Key.LELBOW),
    (H36Key.LELBOW, H36Key.LWRIST),
    # Right leg
    (H36Key.HIP_CENTER, H36Key.RHIP),
    (H36Key.RHIP, H36Key.RKNEE),
    (H36Key.RKNEE, H36Key.RFOOT),
    # Left leg
    (H36Key.HIP_CENTER, H36Key.LHIP),
    (H36Key.LHIP, H36Key.LKNEE),
    (H36Key.LKNEE, H36Key.LFOOT),
]
