"""H3.6M 17-keypoint constants, skeleton, and COCO-to-H3.6M conversion.

This is a pure data/conversion module shared by both RTMPoseExtractor and
other modules that need H3.6M definitions (visualization, metrics, etc.).

No pose estimation classes live here -- only constants and utility functions.
"""

import numpy as np


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


# COCO keypoint indices (for internal mapping)
class _COCOKey:
    """COCO 17 keypoint indices (internal use only)."""

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


# H3.6M keypoint names
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


def coco_to_h36m(coco_pose: np.ndarray) -> np.ndarray:
    """Convert COCO 17 keypoints to H3.6M 17 keypoints (single frame).

    Args:
        coco_pose: (17, 2) or (17, 3) array in COCO format

    Returns:
        h36m_pose: (17, 2) or (17, 3) array in H3.6M format
    """
    has_confidence = coco_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=coco_pose.dtype)

    # Midpoints
    mid_hip = (coco_pose[_COCOKey.LEFT_HIP] + coco_pose[_COCOKey.RIGHT_HIP]) / 2
    mid_shoulder = (coco_pose[_COCOKey.LEFT_SHOULDER] + coco_pose[_COCOKey.RIGHT_SHOULDER]) / 2

    # Direct mapping from COCO to H3.6M
    h36m_pose[H36Key.HIP_CENTER] = mid_hip
    h36m_pose[H36Key.RHIP] = coco_pose[_COCOKey.RIGHT_HIP]
    h36m_pose[H36Key.RKNEE] = coco_pose[_COCOKey.RIGHT_KNEE]
    h36m_pose[H36Key.RFOOT] = coco_pose[_COCOKey.RIGHT_ANKLE]
    h36m_pose[H36Key.LHIP] = coco_pose[_COCOKey.LEFT_HIP]
    h36m_pose[H36Key.LKNEE] = coco_pose[_COCOKey.LEFT_KNEE]
    h36m_pose[H36Key.LFOOT] = coco_pose[_COCOKey.LEFT_ANKLE]
    h36m_pose[H36Key.SPINE] = mid_shoulder * 0.5 + mid_hip * 0.5
    h36m_pose[H36Key.THORAX] = mid_shoulder
    h36m_pose[H36Key.NECK] = coco_pose[_COCOKey.NOSE]

    # HEAD: use midpoint of eyes (indices 1, 2) for better head position
    left_eye = coco_pose[_COCOKey.LEFT_EYE]  # index 1
    right_eye = coco_pose[_COCOKey.RIGHT_EYE]  # index 2
    if has_confidence:
        eye_conf_ok = left_eye[2] >= 0.3 and right_eye[2] >= 0.3
    else:
        eye_conf_ok = True  # no confidence channel, assume ok

    if eye_conf_ok:
        # Midpoint of eyes for position, average confidence
        head_pos = (left_eye[:2] + right_eye[:2]) / 2
        if has_confidence:
            head_conf = (left_eye[2] + right_eye[2]) / 2
            h36m_pose[H36Key.HEAD, :2] = head_pos
            h36m_pose[H36Key.HEAD, 2] = head_conf
        else:
            h36m_pose[H36Key.HEAD] = head_pos
    else:
        # Fallback: nose position offset upward by 10% of shoulder-to-nose distance
        nose_pos = coco_pose[_COCOKey.NOSE, :2]
        shoulder_to_nose = nose_pos - mid_shoulder[:2]
        offset_dist = np.linalg.norm(shoulder_to_nose) * 0.1
        # Offset in direction from mid-shoulder to nose (upward)
        direction = shoulder_to_nose / (np.linalg.norm(shoulder_to_nose) + 1e-8)
        head_pos = nose_pos + direction * offset_dist
        if has_confidence:
            h36m_pose[H36Key.HEAD, :2] = head_pos
            h36m_pose[H36Key.HEAD, 2] = coco_pose[_COCOKey.NOSE, 2]
        else:
            h36m_pose[H36Key.HEAD] = head_pos

    h36m_pose[H36Key.LSHOULDER] = coco_pose[_COCOKey.LEFT_SHOULDER]
    h36m_pose[H36Key.LELBOW] = coco_pose[_COCOKey.LEFT_ELBOW]
    h36m_pose[H36Key.LWRIST] = coco_pose[_COCOKey.LEFT_WRIST]
    h36m_pose[H36Key.RSHOULDER] = coco_pose[_COCOKey.RIGHT_SHOULDER]
    h36m_pose[H36Key.RELBOW] = coco_pose[_COCOKey.RIGHT_ELBOW]
    h36m_pose[H36Key.RWRIST] = coco_pose[_COCOKey.RIGHT_WRIST]

    return h36m_pose


# Backward compatibility alias
_coco_to_h36m_single = coco_to_h36m


def _biometric_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Compute biometric distance between two H3.6M poses.

    Uses anatomical ratios (scale-invariant) to match the same person
    even when track IDs change. Returns 0.0 for identical proportions.

    Args:
        pose_a: H3.6M pose (17, 3) -- normalized coordinates.
        pose_b: H3.6M pose (17, 3) -- normalized coordinates.

    Returns:
        Distance metric (lower = more similar).
    """
    # Joint pairs for anatomical ratios
    pairs = [
        (H36Key.LSHOULDER, H36Key.RSHOULDER),  # shoulder width
        (H36Key.LHIP, H36Key.RHIP),  # hip width
        (H36Key.LSHOULDER, H36Key.LELBOW),  # left upper arm
        (H36Key.LELBOW, H36Key.LWRIST),  # left forearm
        (H36Key.RSHOULDER, H36Key.RELBOW),  # right upper arm
        (H36Key.RELBOW, H36Key.RWRIST),  # right forearm
        (H36Key.LHIP, H36Key.LKNEE),  # left femur
        (H36Key.LKNEE, H36Key.LFOOT),  # left tibia
        (H36Key.RHIP, H36Key.RKNEE),  # right femur
        (H36Key.RKNEE, H36Key.RFOOT),  # right tibia
    ]

    ratios_a = []
    ratios_b = []
    for _i, (j1, j2) in enumerate(pairs):
        len_a = np.linalg.norm(pose_a[j1, :2] - pose_a[j2, :2])
        len_b = np.linalg.norm(pose_b[j1, :2] - pose_b[j2, :2])
        # Skip if either joint has low confidence
        if pose_a[j1, 2] < 0.3 or pose_a[j2, 2] < 0.3:
            continue
        if pose_b[j1, 2] < 0.3 or pose_b[j2, 2] < 0.3:
            continue
        ratios_a.append(len_a)
        ratios_b.append(len_b)

    if len(ratios_a) < 3:
        # Not enough confident joints -- reject rather than match on position alone
        return float("inf")

    ratios_a = np.array(ratios_a)
    ratios_b = np.array(ratios_b)

    # Normalize by total body size (scale-invariant)
    total_a = ratios_a.sum()
    total_b = ratios_b.sum()
    if total_a > 1e-6 and total_b > 1e-6:
        ratios_a /= total_a
        ratios_b /= total_b

    return float(np.linalg.norm(ratios_a - ratios_b))


# Public alias for backward compatibility (maps to COCO)
BKey = _COCOKey
