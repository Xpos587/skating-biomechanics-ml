"""HALPE26 keypoint constants and H3.6M mapping.

HALPE26 is the output format of RTMPose ``BodyWithFeet`` model (rtmlib).
It extends COCO 17 keypoints with 6 foot keypoints (heel, big toe, small toe)
and 3 face keypoints.

References:
    - rtmlib: https://github.com/Tau-J/rtmlib
    - RTMPose: https://arxiv.org/abs/2303.07399
    - HALPE26: COCO 17 + feet (6) + face (3) = 26 keypoints
"""

from enum import IntEnum

import numpy as np

from .h36m_extractor import H36Key


class HALPE26Key(IntEnum):
    """HALPE26 keypoint indices (26 total).

    Indices 0-16 are standard COCO keypoints.
    Indices 17-22 are foot keypoints (critical for blade edge detection).
    Indices 23-25 are face keypoints (not used in our pipeline).
    """

    # Standard COCO 17
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Foot keypoints (NEW — not in COCO)
    LEFT_HEEL = 17
    LEFT_BIG_TOE = 18
    LEFT_SMALL_TOE = 19
    RIGHT_HEEL = 20
    RIGHT_BIG_TOE = 21
    RIGHT_SMALL_TOE = 22

    # Face keypoints (not used)
    LEFT_EYE_INNER = 23  # not reliable
    RIGHT_EYE_INNER = 24  # not reliable
    MOUTH = 25  # not reliable


# Foot keypoint indices in order: [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
FOOT_KEYPOINT_INDICES = [
    HALPE26Key.LEFT_HEEL,
    HALPE26Key.LEFT_BIG_TOE,
    HALPE26Key.LEFT_SMALL_TOE,
    HALPE26Key.RIGHT_HEEL,
    HALPE26Key.RIGHT_BIG_TOE,
    HALPE26Key.RIGHT_SMALL_TOE,
]


def halpe26_to_h36m(halpe26_pose: np.ndarray) -> np.ndarray:
    """Convert HALPE26 (26kp) to H3.6M (17kp) format.

    HALPE26 indices 0-16 are identical to COCO, so the mapping is the same
    geometric computation as _coco_to_h36m_single() in h36m_extractor.py.

    Args:
        halpe26_pose: (26, 2) or (26, 3) array in HALPE26 format.

    Returns:
        h36m_pose: (17, 2) or (17, 3) array in H3.6M format.
    """
    has_confidence = halpe26_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=halpe26_pose.dtype)

    # Midpoints
    mid_hip = (halpe26_pose[HALPE26Key.LEFT_HIP] + halpe26_pose[HALPE26Key.RIGHT_HIP]) / 2
    mid_shoulder = (
        halpe26_pose[HALPE26Key.LEFT_SHOULDER] + halpe26_pose[HALPE26Key.RIGHT_SHOULDER]
    ) / 2

    # Direct mapping (same as COCO since indices 0-16 are identical)
    h36m_pose[H36Key.HIP_CENTER] = mid_hip
    h36m_pose[H36Key.RHIP] = halpe26_pose[HALPE26Key.RIGHT_HIP]
    h36m_pose[H36Key.RKNEE] = halpe26_pose[HALPE26Key.RIGHT_KNEE]
    h36m_pose[H36Key.RFOOT] = halpe26_pose[HALPE26Key.RIGHT_ANKLE]
    h36m_pose[H36Key.LHIP] = halpe26_pose[HALPE26Key.LEFT_HIP]
    h36m_pose[H36Key.LKNEE] = halpe26_pose[HALPE26Key.LEFT_KNEE]
    h36m_pose[H36Key.LFOOT] = halpe26_pose[HALPE26Key.LEFT_ANKLE]
    h36m_pose[H36Key.SPINE] = mid_shoulder * 0.5 + mid_hip * 0.5
    h36m_pose[H36Key.THORAX] = mid_shoulder
    h36m_pose[H36Key.NECK] = halpe26_pose[HALPE26Key.NOSE]

    # HEAD: midpoint of eyes
    left_eye = halpe26_pose[HALPE26Key.LEFT_EYE]
    right_eye = halpe26_pose[HALPE26Key.RIGHT_EYE]
    if has_confidence:
        eye_conf_ok = left_eye[2] >= 0.3 and right_eye[2] >= 0.3
    else:
        eye_conf_ok = True

    if eye_conf_ok:
        head_pos = (left_eye[:2] + right_eye[:2]) / 2
        if has_confidence:
            head_conf = (left_eye[2] + right_eye[2]) / 2
            h36m_pose[H36Key.HEAD, :2] = head_pos
            h36m_pose[H36Key.HEAD, 2] = head_conf
        else:
            h36m_pose[H36Key.HEAD] = head_pos
    else:
        nose_pos = halpe26_pose[HALPE26Key.NOSE, :2]
        shoulder_to_nose = nose_pos - mid_shoulder[:2]
        offset_dist = np.linalg.norm(shoulder_to_nose) * 0.1
        direction = shoulder_to_nose / (np.linalg.norm(shoulder_to_nose) + 1e-8)
        head_pos = nose_pos + direction * offset_dist
        if has_confidence:
            h36m_pose[H36Key.HEAD, :2] = head_pos
            h36m_pose[H36Key.HEAD, 2] = halpe26_pose[HALPE26Key.NOSE, 2]
        else:
            h36m_pose[H36Key.HEAD] = head_pos

    h36m_pose[H36Key.LSHOULDER] = halpe26_pose[HALPE26Key.LEFT_SHOULDER]
    h36m_pose[H36Key.LELBOW] = halpe26_pose[HALPE26Key.LEFT_ELBOW]
    h36m_pose[H36Key.LWRIST] = halpe26_pose[HALPE26Key.LEFT_WRIST]
    h36m_pose[H36Key.RSHOULDER] = halpe26_pose[HALPE26Key.RIGHT_SHOULDER]
    h36m_pose[H36Key.RELBOW] = halpe26_pose[HALPE26Key.RIGHT_ELBOW]
    h36m_pose[H36Key.RWRIST] = halpe26_pose[HALPE26Key.RIGHT_WRIST]

    return h36m_pose


def extract_foot_keypoints(halpe26_pose: np.ndarray) -> np.ndarray:
    """Extract foot keypoints from HALPE26 pose.

    Args:
        halpe26_pose: (26, 3) array [x, y, confidence] in normalized or pixel coords.

    Returns:
        (6, 3) array: [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
    """
    return halpe26_pose[FOOT_KEYPOINT_INDICES].copy()
