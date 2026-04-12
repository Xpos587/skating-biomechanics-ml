"""Pose normalization for camera-invariant analysis.

This module provides normalization utilities to make poses invariant to:
- Root position (centering at hip_center)
- Scale (spine length normalization)
- Anthropometry (body proportions)

Updated for H3.6M 17-keypoint format (3D-only pipeline).
"""

import numpy as np

from ..types import H36Key, NormalizedPose, Pose3D


class PoseNormalizer:
    """Normalize poses for camera-invariant analysis.

    Applies root-centering and scale normalization to make poses
    comparable across different videos and athletes.
    """

    def __init__(self, target_spine_length: float = 0.4) -> None:
        """Initialize pose normalizer.

        Args:
            target_spine_length: Target spine length after normalization.
                Spine is measured from thorax to hip_center.
                Default 0.4 is typical for adult athletes.
        """
        self._target_spine_length = target_spine_length

    def normalize(self, poses: Pose3D) -> NormalizedPose:
        """Normalize 3D poses via root-centering and scale normalization.

        Normalization steps:
        1. Center each frame at hip_center (root) -> origin (0, 0, 0)
        2. Scale so spine length equals target_spine_length
        3. Project to 2D (x, y) for normalized 2D poses

        Args:
            poses: 3D poses (num_frames, 17, 3) with x, y, z in meters.

        Returns:
            NormalizedPose (num_frames, 17, 2) with centered, scaled coordinates.

        Raises:
            ValueError: If poses shape is invalid.
        """
        if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            raise ValueError(f"Expected poses shape (N, 17, 3), got {poses.shape}")

        num_frames = poses.shape[0]
        normalized = np.zeros((num_frames, 17, 2), dtype=np.float32)

        # Process each frame
        for frame_idx in range(num_frames):
            frame = poses[frame_idx]

            # Get hip_center position (root joint in H3.6M)
            hip_center = frame[H36Key.HIP_CENTER]

            # 1. Root-centering: shift hip_center to origin
            centered = frame - hip_center

            # 2. Scale normalization
            thorax = frame[H36Key.THORAX]

            spine_vector = thorax - hip_center
            spine_length = np.linalg.norm(spine_vector)

            scale = 1.0 if spine_length < 1e-6 else self._target_spine_length / spine_length

            # 3. Project to 2D (x, y) - drop z coordinate
            normalized[frame_idx] = centered[:, :2] * scale

        return normalized

    def get_spine_length(self, poses: Pose3D) -> float:
        """Calculate average spine length across frames.

        Args:
            poses: 3D poses (num_frames, 17, 3).

        Returns:
            Average spine length in original coordinate units (meters).
        """
        hip_center = poses[:, H36Key.HIP_CENTER]
        thorax = poses[:, H36Key.THORAX]

        spine_lengths = np.linalg.norm(thorax - hip_center, axis=1)
        return float(np.mean(spine_lengths))

    def is_valid_frame(
        self,
        frame: np.ndarray,
        min_visible: float = 0.7,
    ) -> bool:
        """Check if frame has enough visible keypoints.

        Args:
            frame: Single frame keypoints (17, 3) with x, y, z.
            min_visible: Minimum ratio of visible keypoints [0, 1].

        Returns:
            True if frame is valid for analysis.
        """
        if frame.shape != (17, 3):
            return False

        # Count keypoints with non-zero position (valid)
        visible = np.sum(np.abs(frame).max(axis=1) > 0.01)
        ratio = visible / 17

        return bool(ratio >= min_visible)


def get_hip_center(poses: Pose3D) -> np.ndarray:
    """Get hip_center position for each frame.

    Args:
        poses: 3D poses (num_frames, 17, 3).

    Returns:
        Hip_center coordinates (num_frames, 3).
    """
    return poses[:, H36Key.HIP_CENTER]


def get_thorax(poses: Pose3D) -> np.ndarray:
    """Get thorax position for each frame.

    Args:
        poses: 3D poses (num_frames, 17, 3).

    Returns:
        Thorax coordinates (num_frames, 3).
    """
    return poses[:, H36Key.THORAX]


# Convenience aliases
get_mid_hypot = get_hip_center
get_mid_shoulder_raw = get_thorax
