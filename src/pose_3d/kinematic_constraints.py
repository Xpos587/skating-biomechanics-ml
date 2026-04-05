"""Kinematic constraints for anatomically plausible 3D pose sequences.

Applies bone length consistency and joint angle limits to 3D pose
sequences in H3.6M 17-keypoint format.  This module is part of the
"corrective lens" pipeline that uses 3D information to improve the
quality of skeleton overlays.

Pipeline order (inside ``apply_kinematic_constraints``):
    temporal smooth  ->  bone length enforcement  ->  angle limit clamping
"""

import numpy as np
from numpy.typing import NDArray

from ..pose_estimation import H36Key
from ..utils.smoothing import PoseSmoother, get_skating_optimized_config

# ---------------------------------------------------------------------------
# Skeleton topology — kinematic-chain order for iterative correction
# ---------------------------------------------------------------------------

_KINEMATIC_CHAIN_EDGES: list[tuple[int, int]] = [
    # Spine: hip_center -> head
    (H36Key.HIP_CENTER, H36Key.SPINE),
    (H36Key.SPINE, H36Key.THORAX),
    (H36Key.THORAX, H36Key.NECK),
    (H36Key.NECK, H36Key.HEAD),
    # Right leg
    (H36Key.HIP_CENTER, H36Key.RHIP),
    (H36Key.RHIP, H36Key.RKNEE),
    (H36Key.RKNEE, H36Key.RFOOT),
    # Left leg
    (H36Key.HIP_CENTER, H36Key.LHIP),
    (H36Key.LHIP, H36Key.LKNEE),
    (H36Key.LKNEE, H36Key.LFOOT),
    # Right arm
    (H36Key.THORAX, H36Key.RSHOULDER),
    (H36Key.RSHOULDER, H36Key.RELBOW),
    (H36Key.RELBOW, H36Key.RWRIST),
    # Left arm
    (H36Key.THORAX, H36Key.LSHOULDER),
    (H36Key.LSHOULDER, H36Key.LELBOW),
    (H36Key.LELBOW, H36Key.LWRIST),
]

# Joint angle limit definitions (parent_idx, joint_idx, child_idx, min_deg, max_deg)
# ---------------------------------------------------------------------------

JOINT_LIMITS: list[tuple[int, int, int, float, float]] = [
    (H36Key.RHIP, H36Key.RKNEE, H36Key.RFOOT, 0.0, 180.0),
    (H36Key.LHIP, H36Key.LKNEE, H36Key.LFOOT, 0.0, 180.0),
    (H36Key.RSHOULDER, H36Key.RELBOW, H36Key.RWRIST, 0.0, 160.0),
    (H36Key.LSHOULDER, H36Key.LELBOW, H36Key.LWRIST, 0.0, 160.0),
    (H36Key.THORAX, H36Key.RHIP, H36Key.RKNEE, 30.0, 180.0),
    (H36Key.THORAX, H36Key.LHIP, H36Key.LKNEE, 30.0, 180.0),
]

# Small epsilon to avoid degenerate division
_EPS: float = 1e-8

# Number of bone-length correction iterations
_N_BONE_ITERS: int = 3


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _angle_3d(a: NDArray[np.float32], b: NDArray[np.float32], c: NDArray[np.float32]) -> float:
    """Angle at point *b* formed by rays b->a and b->c in 3D space.

    Args:
        a: First point (3,).
        b: Vertex point (3,).
        c: Third point (3,).

    Returns:
        Angle in degrees in the range [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < _EPS or norm_bc < _EPS:
        return 180.0
    cos_angle = float(np.dot(ba, bc)) / (norm_ba * norm_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _rodrigues_rotate(
    v: NDArray[np.float32],
    k: NDArray[np.float32],
    theta: float,
) -> NDArray[np.float32]:
    """Rotate vector *v* around unit axis *k* by *theta* radians (Rodrigues).

    Args:
        v: Vector to rotate, shape (3,).
        k: Unit rotation axis, shape (3,).
        theta: Rotation angle in radians.

    Returns:
        Rotated vector, shape (3,).
    """
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    one_minus_cos = 1.0 - cos_t
    dot_kv = float(np.dot(k, v))
    cross_kv = np.cross(k, v)
    result = v * cos_t + cross_kv * sin_t + k * (dot_kv * one_minus_cos)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enforce_bone_lengths(poses_3d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Enforce consistent bone lengths across all frames.

    Reference bone lengths are computed as the median length across the
    entire sequence for each skeleton edge.  Each frame is then iteratively
    corrected so that every bone matches its reference length while keeping
    the parent joint fixed.

    Processing follows kinematic-chain order (root -> spine -> head,
    root -> legs, thorax -> arms) so that corrections propagate
    correctly from proximal to distal joints.

    Args:
        poses_3d: Root-relative 3D poses, shape ``(N, 17, 3)`` in metres.

    Returns:
        Corrected poses with consistent bone lengths, same shape.

    Raises:
        ValueError: If input shape is not ``(N, 17, 3)``.
    """
    if poses_3d.ndim != 3 or poses_3d.shape[1] != 17 or poses_3d.shape[2] != 3:
        msg = f"Expected shape (N, 17, 3), got {poses_3d.shape}"
        raise ValueError(msg)

    corrected = poses_3d.copy()
    n_frames = corrected.shape[0]

    # Compute reference bone lengths as median across all frames
    ref_lengths: dict[tuple[int, int], float] = {}
    for parent, child in _KINEMATIC_CHAIN_EDGES:
        diffs = corrected[:, parent, :] - corrected[:, child, :]
        lengths = np.linalg.norm(diffs, axis=1)  # (N,)
        ref_lengths[(parent, child)] = float(np.median(lengths))

    # Iterative correction
    for _ in range(_N_BONE_ITERS):
        for parent, child in _KINEMATIC_CHAIN_EDGES:
            ref_len = ref_lengths[(parent, child)]
            if ref_len < _EPS:
                continue
            for frame in range(n_frames):
                diff = corrected[frame, child] - corrected[frame, parent]
                cur_len = float(np.linalg.norm(diff))
                if cur_len < _EPS or abs(cur_len - ref_len) < _EPS:
                    continue
                direction = diff / cur_len
                corrected[frame, child] = corrected[frame, parent] + direction * ref_len

    return corrected


def enforce_joint_angle_limits(poses_3d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Clamp joint angles to anatomically valid ranges.

    For each joint defined in ``JOINT_LIMITS``, the 3D angle formed by
    the parent-joint-child triplet is checked against allowed bounds.
    If the angle exceeds the limits, the child point is rotated around
    the joint (using Rodrigues' formula) to bring the angle to the
    nearest valid value while preserving the parent-joint bone length.

    Joints corrected:
        - Knees (0-180 degrees, no hyperextension)
        - Elbows (0-160 degrees)
        - Hips (30-180 degrees)

    Args:
        poses_3d: Root-relative 3D poses, shape ``(N, 17, 3)`` in metres.

    Returns:
        Corrected poses with clamped joint angles, same shape.

    Raises:
        ValueError: If input shape is not ``(N, 17, 3)``.
    """
    if poses_3d.ndim != 3 or poses_3d.shape[1] != 17 or poses_3d.shape[2] != 3:
        msg = f"Expected shape (N, 17, 3), got {poses_3d.shape}"
        raise ValueError(msg)

    corrected = poses_3d.copy()
    n_frames = corrected.shape[0]

    for parent, joint, child, min_deg, max_deg in JOINT_LIMITS:
        for frame in range(n_frames):
            p = corrected[frame, parent]
            j = corrected[frame, joint]
            c = corrected[frame, child]

            angle = _angle_3d(p, j, c)

            if min_deg <= angle <= max_deg:
                continue  # within valid range

            # Determine target angle
            target_angle = min_deg if angle < min_deg else max_deg
            target_rad = np.radians(target_angle)

            # Vectors from joint
            ba = p - j
            bc = c - j
            norm_ba = float(np.linalg.norm(ba))
            norm_bc = float(np.linalg.norm(bc))

            if norm_ba < _EPS or norm_bc < _EPS:
                continue

            ba_unit = ba / norm_ba
            bc_unit = bc / norm_bc

            # Rotation axis: perpendicular to the plane formed by ba and bc
            axis = np.cross(ba_unit, bc_unit)
            axis_norm = float(np.linalg.norm(axis))

            if axis_norm < _EPS:
                # ba and bc are (anti-)parallel — pick arbitrary perpendicular axis
                arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if abs(float(np.dot(ba_unit, arbitrary))) > 0.9:
                    arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                axis = np.cross(ba_unit, arbitrary)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < _EPS:
                    continue

            axis = (axis / axis_norm).astype(np.float32)

            # Current angle between ba and bc
            current_rad = np.radians(angle)

            # How much to rotate bc around joint to reach target
            delta = target_rad - current_rad  # negative if clamping to max

            # Rotate the full bc vector (not just unit) and restore magnitude
            bc_rotated = _rodrigues_rotate(bc_unit, axis, delta)
            corrected[frame, child] = j + bc_rotated * norm_bc

    return corrected


def apply_kinematic_constraints(
    poses_3d: NDArray[np.float32],
    fps: float,
    confidences: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Apply the full kinematic constraint pipeline to a 3D pose sequence.

    The pipeline applies three stages in order:

    1. **Temporal smoothing** via One-Euro Filter (``PoseSmoother``)
       with skating-optimized parameters.
    2. **Bone length enforcement** — iteratively adjusts child joints
       so every bone matches the sequence-median length.
    3. **Joint angle clamping** — rotates out-of-range joints to the
       nearest anatomically valid angle.

    Args:
        poses_3d: Root-relative 3D poses, shape ``(N, 17, 3)`` in metres.
        fps: Video frame rate in Hz (used for temporal smoothing).
        confidences: Per-joint confidence scores ``(N, 17)``, currently
            reserved for future weighting.  Unused at present.

    Returns:
        Fully corrected 3D pose sequence, same shape ``(N, 17, 3)``.

    Raises:
        ValueError: If input shape is not ``(N, 17, 3)``.
    """
    _ = confidences  # reserved for future use

    if poses_3d.ndim != 3 or poses_3d.shape[1] != 17 or poses_3d.shape[2] != 3:
        msg = f"Expected shape (N, 17, 3), got {poses_3d.shape}"
        raise ValueError(msg)

    # Stage 1: temporal smoothing
    config = get_skating_optimized_config(fps=fps)
    smoother = PoseSmoother(config=config, freq=fps)
    smoothed = smoother.smooth_3d(poses_3d)

    # Stage 2: bone length enforcement
    bone_corrected = enforce_bone_lengths(smoothed)

    # Stage 3: joint angle clamping
    constrained = enforce_joint_angle_limits(bone_corrected)

    return constrained
