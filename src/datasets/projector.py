"""Foot keypoint 3D→2D camera projection for AthletePose3D dataset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# AthletePose3D 142kp indices for HALPE26 foot keypoints (17-22)
FOOT_AP3D_INDICES = np.array([49, 26, 10, 112, 93, 77], dtype=np.intp)
# LHEL, L_Toe, L_5th_MTP, RHEL, R_Toe, R_5th_MTP

# AthletePose3D 142kp indices for ankle keypoints (used as weak-perspective anchors)
ANKLE_AP3D_INDICES = np.array([33, 95], dtype=np.intp)
# L_ankle, R_ankle — verified <3px distance to _coco.npy across all frames


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
    """Project foot keypoints from 3D to 2D using hybrid projection.

    The AthletePose3D 142kp has two marker coordinate systems:
    - Processed markers (e.g., LANK=33, LHEL=49) — same frame, used for heels
    - Raw mocap markers (e.g., L_Toe=26, L_HEEL=12) — different frame

    Strategy:
    - Heels: weak-perspective using processed markers (consistent frame).
      Uses ankle depth Z_ankle as reference depth for the heel, preserving
      parallel foot geometry and eliminating the "heel above ankle" paradox.
    - Big toes: full-perspective using raw mocap markers (projected independently).
    - Small toes: always NaN (invisible inside skate boots).

    Args:
        keypoints_3d: (N, 3) array of 3D world coordinates (N >= 113 for all foot kps).
        cam: Camera dict from cam_param.json.

    Returns:
        (6, 2) projected 2D coordinates for HALPE26 foot keypoints 17-22.
        Indices 2 (L_small_toe) and 5 (R_small_toe) are always NaN.
    """
    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    pts = np.full((6, 2), np.nan, dtype=np.float32)

    # --- Heels: weak-perspective (processed system) ---
    # Processed markers share the same coordinate frame, so camera-space
    # offsets are meaningful. Using Z_ankle preserves parallel foot geometry.
    heel_pairs = [
        (0, 49, 33),  # foot_2d[0]=LHEL, AP3D[49]=LHEL, AP3D[33]=LANK
        (3, 112, 95),  # foot_2d[3]=RHEL, AP3D[112]=RHEL, AP3D[95]=RANK
    ]

    for foot_idx, heel_ap3d, ankle_ap3d in heel_pairs:
        if ankle_ap3d >= len(keypoints_3d):
            continue

        ankle_3d = keypoints_3d[ankle_ap3d]
        if np.isnan(ankle_3d).any():
            continue

        ankle_translated = ankle_3d - t
        ankle_cam = rot_mat @ ankle_translated

        if ankle_cam[2] <= 0:
            continue

        ankle_u = fu * (ankle_cam[0] / ankle_cam[2]) + cu
        ankle_v = fv * (ankle_cam[1] / ankle_cam[2]) + cv
        z_ankle = ankle_cam[2]

        if heel_ap3d >= len(keypoints_3d):
            continue

        heel_3d = keypoints_3d[heel_ap3d]
        if np.isnan(heel_3d).any():
            continue

        # Camera-space offset from ankle (both processed → same frame)
        heel_translated = heel_3d - t
        heel_cam = rot_mat @ heel_translated
        delta_cam = heel_cam - ankle_cam

        pts[foot_idx] = [
            ankle_u + fu * delta_cam[0] / z_ankle,
            ankle_v + fv * delta_cam[1] / z_ankle,
        ]

    # --- Big toes: full-perspective (raw mocap markers) ---
    # Raw mocap markers (L_Toe=26, R_Toe=93) are in a different coordinate
    # frame than processed markers. Project them independently with full
    # perspective — they happen to produce reasonable 2D positions.
    toe_pairs = [
        (1, 26),   # foot_2d[1]=L_big_toe, AP3D[26]=L_Toe
        (4, 93),   # foot_2d[4]=R_big_toe, AP3D[93]=R_Toe
    ]

    for foot_idx, toe_ap3d in toe_pairs:
        if toe_ap3d >= len(keypoints_3d):
            continue

        toe_3d = keypoints_3d[toe_ap3d]
        if np.isnan(toe_3d).any():
            continue

        toe_translated = toe_3d - t
        toe_cam = rot_mat @ toe_translated

        if toe_cam[2] <= 0:
            continue

        pts[foot_idx] = [
            fu * (toe_cam[0] / toe_cam[2]) + cu,
            fv * (toe_cam[1] / toe_cam[2]) + cv,
        ]

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
            # Toe: max 80px from ankle, must not be above ankle
            if dist > 80 or foot_2d[i, 1] < reference_ankle[1] - 30:
                foot_2d[i] = [np.nan, np.nan]
