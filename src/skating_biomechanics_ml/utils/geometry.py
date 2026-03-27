"""Geometric utilities for pose analysis."""

import numpy as np
from numpy.typing import NDArray

from skating_biomechanics_ml.types import BKey, FrameKeypoints, NormalizedPose, TimeSeries


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
    spine_indices: tuple[int, int] = (BKey.LEFT_SHOULDER, BKey.LEFT_HIP),
    target_spine_length: float = 0.4,
) -> NormalizedPose:
    """Normalize poses via root-centering and scale normalization.

    1. Center pose at mid-hip (root) -> origin (0, 0)
    2. Scale so spine length equals target_spine_length

    Args:
        raw: Raw keypoints (num_frames, 33, 3) with x, y, confidence.
        spine_indices: (shoulder_idx, hip_idx) for spine length calculation.
        target_spine_length: Target spine length after normalization.

    Returns:
        NormalizedPose (num_frames, 33, 2) with centered, scaled coordinates.
    """
    if raw.shape[1] != 33:
        raise ValueError(f"Expected 33 keypoints, got {raw.shape[1]}")

    num_frames = raw.shape[0]
    normalized = np.zeros((num_frames, 33, 2), dtype=np.float32)

    # Mid-hip point (between left and right hip)
    mid_hip_raw = (raw[:, BKey.LEFT_HIP, :2] + raw[:, BKey.RIGHT_HIP, :2]) / 2

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
        poses: NormalizedPose (num_frames, 33, 2).

    Returns:
        Mid-hip coordinates (num_frames, 2).
    """
    return (poses[:, BKey.LEFT_HIP, :] + poses[:, BKey.RIGHT_HIP, :]) / 2


def get_mid_shoulder(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate mid-shoulder point for each frame.

    Args:
        poses: NormalizedPose (num_frames, 33, 2).

    Returns:
        Mid-shoulder coordinates (num_frames, 2).
    """
    return (poses[:, BKey.LEFT_SHOULDER, :] + poses[:, BKey.RIGHT_SHOULDER, :]) / 2
