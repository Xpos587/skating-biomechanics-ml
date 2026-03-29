"""Geometric utilities for pose analysis."""

import numpy as np
from numpy.typing import NDArray

from .types import H36Key, FrameKeypoints, NormalizedPose, TimeSeries


def angle_3pt(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    """Calculate angle ABC in degrees.

    Args:
        a: Point A coordinates (x, y).
        b: Vertex point B coordinates (x, y).
        c: Point C coordinates (x, y).

    Returns:
        Angle in degrees [0, 180].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Cosine of angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

    # Clamp to [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))


def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Euclidean distance between two points.

    Args:
        a: Point A coordinates (x, y).
        b: Point B coordinates (x, y).

    Returns:
        Distance in same units as input.
    """
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def normalize_poses(
    raw: FrameKeypoints,
    spine_indices: tuple[int, int] = (H36Key.LSHOULDER, H36Key.LHIP),
    target_spine_length: float = 0.4,
) -> NormalizedPose:
    """Normalize poses via root-centering and scale normalization.

    1. Center pose at mid-hip (root) -> origin (0, 0)
    2. Scale so spine length equals target_spine_length

    Args:
        raw: Raw keypoints (num_frames, 17, 3) with x, y, confidence.
        spine_indices: (shoulder_idx, hip_idx) for spine length calculation.
        target_spine_length: Target spine length after normalization.

    Returns:
        NormalizedPose (num_frames, 17, 2) with centered, scaled coordinates.
    """
    if raw.shape[1] != 17:
        raise ValueError(f"Expected 17 keypoints (H3.6M format), got {raw.shape[1]}")

    num_frames = raw.shape[0]
    normalized = np.zeros((num_frames, 17, 2), dtype=np.float32)

    # Mid-hip point (between left and right hip)
    mid_hip_raw = (raw[:, H36Key.LHIP, :2] + raw[:, H36Key.RHIP, :2]) / 2

    for frame_idx in range(num_frames):
        frame_raw = raw[frame_idx]

        # 1. Root-centering: shift mid-hip to origin
        mid_hip = mid_hip_raw[frame_idx]
        centered = frame_raw[:, :2] - mid_hip

        # 2. Scale normalization
        shoulder_idx, hip_idx = spine_indices
        spine_vector = centered[shoulder_idx] - centered[hip_idx]
        spine_length = np.linalg.norm(spine_vector)

        if spine_length < 1e-6:
            # Degenerate case: use identity scale
            scale = 1.0
        else:
            scale = target_spine_length / spine_length

        normalized[frame_idx] = centered * scale

    return normalized


def smooth_signal(signal: TimeSeries, window: int = 5) -> TimeSeries:
    """Apply moving average smoothing to signal.

    Args:
        signal: Input signal (num_frames,).
        window: Window size for moving average (must be odd).

    Returns:
        Smoothed signal (num_frames,).
    """
    if window < 1:
        return signal

    if window % 2 == 0:
        window += 1

    if len(signal) < window:
        return signal.copy()

    # Use numpy convolution for efficient moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")

    return smoothed.astype(np.float32)


def get_mid_hip(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate mid-hip point for each frame.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        Mid-hip coordinates (num_frames, 2).
    """
    return (poses[:, H36Key.LHIP, :] + poses[:, H36Key.RHIP, :]) / 2


def get_mid_shoulder(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate mid-shoulder point for each frame.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        Mid-shoulder coordinates (num_frames, 2).
    """
    return (poses[:, H36Key.LSHOULDER, :] + poses[:, H36Key.RSHOULDER, :]) / 2


def calculate_center_of_mass(poses: NormalizedPose, frame_idx: int) -> float:
    """Calculate Center of Mass (CoM) Y-coordinate for a single frame.

    Uses anthropometric segment mass ratios from Dempster (1955) and
    Zatsiorsky (2002). The CoM is the weighted average of body segment
    positions: CoM = (1/M) × Σ(mᵢ × pᵢ)

    This provides a physics-accurate measure of jump height, independent
    of landing pose. The hip-only method has 60% error for low jumps due
    to bent-knee landings artificially increasing flight time.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        frame_idx: Frame index to calculate CoM for.

    Returns:
        CoM Y-coordinate in normalized units (lower = higher position).

    Segment mass ratios (relative to total body mass):
        - Head: 0.081
        - Torso: 0.497
        - Upper arms: 0.050 each
        - Forearms+hands: 0.030 each
        - Thighs: 0.100 each
        - Shins+feet: 0.161 each
    """
    pose = poses[frame_idx]

    # Head (HEAD keypoint in H3.6M format)
    head = pose[H36Key.HEAD]
    head_mass = 0.081

    # Torso (mid-shoulder to mid-hip midpoint)
    torso = (
        pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER] + pose[H36Key.LHIP] + pose[H36Key.RHIP]
    ) / 4
    torso_mass = 0.497

    # Arms (elbow-wrist midpoint for upper arm, wrist for forearm)
    l_upper_arm = (pose[H36Key.LSHOULDER] + pose[H36Key.LELBOW]) / 2
    r_upper_arm = (pose[H36Key.RSHOULDER] + pose[H36Key.RELBOW]) / 2
    l_forearm = (pose[H36Key.LELBOW] + pose[H36Key.LWRIST]) / 2
    r_forearm = (pose[H36Key.RELBOW] + pose[H36Key.RWRIST]) / 2
    arm_mass_each = 0.050

    # Thighs (hip-knee midpoint)
    l_thigh = (pose[H36Key.LHIP] + pose[H36Key.LKNEE]) / 2
    r_thigh = (pose[H36Key.RHIP] + pose[H36Key.RKNEE]) / 2
    thigh_mass_each = 0.100

    # Shins+feet (knee-foot midpoint)
    l_leg = (pose[H36Key.LKNEE] + pose[H36Key.LFOOT]) / 2
    r_leg = (pose[H36Key.RKNEE] + pose[H36Key.RFOOT]) / 2
    leg_mass_each = 0.161

    # Weighted sum of Y-coordinates only (for height)
    com_y = (
        head_mass * head[1]
        + torso_mass * torso[1]
        + arm_mass_each * (l_upper_arm[1] + r_upper_arm[1] + l_forearm[1] + r_forearm[1])
        + thigh_mass_each * (l_thigh[1] + r_thigh[1])
        + leg_mass_each * (l_leg[1] + r_leg[1])
    )

    return float(com_y)


def calculate_com_trajectory(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate Center of Mass trajectory for entire pose sequence.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        CoM Y-coordinates (num_frames,) in normalized units.
    """
    num_frames = len(poses)
    com_trajectory = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        com_trajectory[i] = calculate_center_of_mass(poses, i)

    return com_trajectory
