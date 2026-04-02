"""Foot keypoint 3D→2D camera projection for AthletePose3D dataset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# AthletePose3D 142kp indices for HALPE26 foot keypoints (17-22)
FOOT_AP3D_INDICES = np.array([49, 26, 10, 112, 93, 77], dtype=np.intp)
# LHEL, L_Toe, L_5th_MTP, RHEL, R_Toe, R_5th_MTP


def _load_camera(cam: dict) -> tuple[NDArray, NDArray, NDArray, float, float, float, float]:
    """Parse camera dict into projection components."""
    K = np.array(cam["affine_intrinsics_matrix"], dtype=np.float64)
    rot_mat = np.array(cam["extrinsic_matrix"], dtype=np.float64)
    rot_mat[1:, :] *= -1  # Critical: flip Y and Z rows (AthletePose3D convention)
    t = np.array(cam["xyz"], dtype=np.float64)
    return K, rot_mat, t, K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def project_point(
    p_world: NDArray[np.float64],
    cam: dict,
) -> tuple[float, float]:
    """Project a single 3D world point to 2D pixel coordinates.

    Uses AthletePose3D convention: rot_mat[1:, :] *= -1 before projection.
    """
    if np.isnan(p_world).any():
        return (np.nan, np.nan)

    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    translated = p_world - t
    kpts_camera = rot_mat @ translated

    if kpts_camera[2] <= 0:
        return (np.nan, np.nan)

    u = fu * (kpts_camera[0] / kpts_camera[2]) + cu
    v = fv * (kpts_camera[1] / kpts_camera[2]) + cv
    return (float(u), float(v))


def project_foot_frame(
    keypoints_3d: NDArray[np.float64],
    cam: dict,
) -> NDArray[np.float32]:
    """Project foot keypoints from 3D to 2D.

    Args:
        keypoints_3d: (N, 3) array of 3D world coordinates (N >= 113 for all foot kps).
        cam: Camera dict from cam_param.json.

    Returns:
        (6, 2) projected 2D coordinates for HALPE26 foot keypoints 17-22.
        Out-of-range indices get NaN.
    """
    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    pts = np.zeros((6, 2), dtype=np.float32)

    for i, ap3d_idx in enumerate(FOOT_AP3D_INDICES):
        if ap3d_idx >= len(keypoints_3d):
            pts[i] = [np.nan, np.nan]
            continue

        p = keypoints_3d[ap3d_idx]
        if np.isnan(p).any():
            pts[i] = [np.nan, np.nan]
            continue

        translated = p - t
        kc = rot_mat @ translated
        if kc[2] <= 0:
            pts[i] = [np.nan, np.nan]
            continue

        pts[i] = [fu * kc[0] / kc[2] + cu, fv * kc[1] / kc[2] + cv]

    return pts


def validate_foot_projection(
    foot_2d: NDArray[np.float32],
    coco_2d: NDArray[np.float64],
) -> None:
    """Validate projected foot keypoints against reference ankle positions.

    Invalid points are set to NaN in-place (merge_coco_foot_keypoints will
    set vis=0 for NaN input).

    Rules (foot_2d indices 0-5):
        0,3 = heels   : max 60px from ankle AND must not be above ankle
        1,2,4,5 = toes: max 80px from ankle

    Args:
        foot_2d: (6, 2) projected foot coordinates (indices 0-5).
            Indices 0-2 = left foot, 3-5 = right foot.
            0,3 = heel, 1,4 = big toe, 2,5 = small toe.
        coco_2d: (17, 2) COCO keypoints. Index 15 = left_ankle, 16 = right_ankle.
    """
    l_ankle = coco_2d[15]
    r_ankle = coco_2d[16]

    for i in range(6):
        if np.isnan(foot_2d[i]).any():
            continue

        reference_ankle = l_ankle if i < 3 else r_ankle
        if np.isnan(reference_ankle).any():
            foot_2d[i] = [np.nan, np.nan]
            continue

        dist = float(np.linalg.norm(foot_2d[i] - reference_ankle))
        is_heel = i in (0, 3)

        if is_heel:
            # Heel: max 60px from ankle, must not be above ankle
            if dist > 60 or foot_2d[i, 1] < reference_ankle[1] - 30:
                foot_2d[i] = [np.nan, np.nan]
        else:
            # Toe: max 80px from ankle
            if dist > 80:
                foot_2d[i] = [np.nan, np.nan]
