"""3D Pose normalization for camera-invariant analysis.

This module provides normalization utilities for 3D poses to make them invariant to:
- Root position (centering at hip_center)
- Scale (body height normalization)
- Anthropometry (body proportions)
"""

import numpy as np

from ..types import H36Key, Pose3D


class Pose3DNormalizer:
    """Normalize 3D poses for camera-invariant analysis.

    Applies root-centering and scale normalization to make 3D poses
    comparable across different videos and athletes.
    """

    def __init__(self, target_height: float = 1.7) -> None:
        """Initialize 3D pose normalizer.

        Args:
            target_height: Target body height after normalization (meters).
                Default 1.7m is typical for adult athletes.
        """
        self._target_height = target_height

    def normalize(self, poses_3d: Pose3D) -> Pose3D:
        """Normalize 3D poses via root-centering and scale normalization.

        Normalization steps:
        1. Center each frame at hip_center -> origin (0, 0, 0)
        2. Scale so body height equals target_height
        3. Preserve 3D structure (x, y, z coordinates)

        Args:
            poses_3d: Raw 3D poses (num_frames, 17, 3) with x, y, z in meters.

        Returns:
            Normalized 3D poses (num_frames, 17, 3) with centered, scaled coordinates.

        Raises:
            ValueError: If poses shape is invalid.
        """
        if poses_3d.ndim != 3 or poses_3d.shape[1] != 17 or poses_3d.shape[2] != 3:
            raise ValueError(f"Expected poses_3d shape (N, 17, 3), got {poses_3d.shape}")

        num_frames = poses_3d.shape[0]
        normalized = np.zeros_like(poses_3d, dtype=np.float32)

        # Process each frame
        for frame_idx in range(num_frames):
            frame = poses_3d[frame_idx]

            # Get hip_center position (root joint)
            hip_center = frame[H36Key.HIP_CENTER]

            # 1. Root-centering: shift hip_center to origin
            centered = frame - hip_center

            # 2. Scale normalization based on body height
            # Body height = distance from lowest foot to highest head point
            head_y = frame[H36Key.HEAD, 1]  # y is typically vertical
            left_foot_y = frame[H36Key.LFOOT, 1]
            right_foot_y = frame[H36Key.RFOOT, 1]

            # Find lowest foot and highest head
            lowest_foot = min(left_foot_y, right_foot_y)
            body_height = head_y - lowest_foot

            if body_height < 0.1:  # Less than 10cm - degenerate case
                scale = 1.0
            else:
                scale = self._target_height / body_height

            normalized[frame_idx] = centered * scale

        return normalized

    def get_body_height(self, poses_3d: Pose3D) -> float:
        """Calculate average body height across frames.

        Args:
            poses_3d: 3D poses (num_frames, 17, 3).

        Returns:
            Average body height in meters.
        """
        num_frames = poses_3d.shape[0]
        heights = []

        for frame_idx in range(num_frames):
            frame = poses_3d[frame_idx]

            head_y = frame[H36Key.HEAD, 1]
            left_foot_y = frame[H36Key.LFOOT, 1]
            right_foot_y = frame[H36Key.RFOOT, 1]

            lowest_foot = min(left_foot_y, right_foot_y)
            body_height = head_y - lowest_foot

            if body_height > 0.1:  # Filter degenerate cases
                heights.append(body_height)

        if not heights:
            return 1.7  # Default height

        return float(np.mean(heights))

    def is_valid_frame(
        self,
        frame_3d: np.ndarray,
        min_visible: float = 0.7,
    ) -> bool:
        """Check if 3D frame has enough valid keypoints.

        Args:
            frame_3d: Single frame 3D keypoints (17, 3) with x, y, z.
            min_visible: Minimum ratio of valid keypoints [0, 1].

        Returns:
            True if frame is valid for analysis.
        """
        if frame_3d.shape != (17, 3):
            return False

        # Count keypoints that are not at origin (likely invalid)
        # A valid keypoint should have non-zero position
        valid = np.sum(np.abs(frame_3d).max(axis=1) > 0.01)
        ratio = valid / 17

        return bool(ratio >= min_visible)


def get_hip_center_3d(poses_3d: Pose3D) -> np.ndarray:
    """Get hip_center position for each frame.

    Args:
        poses_3d: 3D poses (num_frames, 17, 3).

    Returns:
        Hip_center coordinates (num_frames, 3).
    """
    return poses_3d[:, H36Key.HIP_CENTER]


def get_head_center_3d(poses_3d: Pose3D) -> np.ndarray:
    """Get head position for each frame.

    Args:
        poses_3d: 3D poses (num_frames, 17, 3).

    Returns:
        Head coordinates (num_frames, 3).
    """
    return poses_3d[:, H36Key.HEAD]


def calculate_body_heights(poses_3d: Pose3D) -> np.ndarray:
    """Calculate body height for each frame.

    Args:
        poses_3d: 3D poses (num_frames, 17, 3).

    Returns:
        Body heights (num_frames,) in meters.
    """
    head_y = poses_3d[:, H36Key.HEAD, 1]
    left_foot_y = poses_3d[:, H36Key.LFOOT, 1]
    right_foot_y = poses_3d[:, H36Key.RFOOT, 1]

    lowest_foot = np.minimum(left_foot_y, right_foot_y)
    return head_y - lowest_foot
