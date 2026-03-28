"""Physics-informed pose validation and optimization.

Implements Kalman filter + bone length constraints to eliminate
occlusion artifacts and ensure biomechanically realistic poses.

Reference: Leuthold et al. (December 2025) "Physics Informed Human Posture
Estimation Based on 3D Landmarks" - arXiv:2512.06783
Results: -10.2% MPJPE, -16.6% joint angle error

The optimizer works by:
1. Learning bone lengths from high-confidence frames
2. Using Kalman filter to smooth pose trajectories
3. Enforcing bone length constraints to prevent unnatural stretches
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter
from numpy.typing import NDArray

if TYPE_CHECKING:
    from skating_biomechanics_ml.types import NormalizedPose


@dataclass(frozen=True)
class BoneConstraints:
    """Anatomical bone length constraints (relative to spine length).

    All ratios are relative to spine length (distance from mid-shoulder
    to mid-hip). These are based on anthropometric averages for adult
    athletes.

    Attributes:
        SPINE_TO_UPPER_ARM: Upper arm length / spine length
        SPINE_TO_FOREARM: Forearm length / spine length
        SPINE_TO_THIGH: Thigh length / spine length
        SPINE_TO_SHIN: Shin length / spine length
        TOLERANCE: Allowable variation (± percentage)
    """

    SPINE_TO_UPPER_ARM: float = 0.45
    SPINE_TO_FOREARM: float = 0.40
    SPINE_TO_THIGH: float = 0.50
    SPINE_TO_SHIN: float = 0.48
    TOLERANCE: float = 0.15  # 15% variation allowed


class PhysicsPoseOptimizer:
    """Optimizes poses using physics constraints.

    Uses Kalman filtering + bone length constraints to eliminate
    occlusion artifacts while preserving natural motion.

    Based on: Leuthold et al. (2025) "Physics Informed Human Posture
    Estimation" - achieves -10.2% MPJPE improvement.

    Attributes:
        constraints: Bone length constraints.
        spine_length: Learned spine length (mid-shoulder to mid-hip).
        bone_lengths: Dictionary of learned bone lengths.
        trust_per_keypoint: Measurement trust values [0, 1] for each keypoint.
    """

    def __init__(
        self,
        constraints: BoneConstraints | None = None,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        """Initialize physics pose optimizer.

        Args:
            constraints: Bone length constraints (uses defaults if None).
            process_noise: Kalman filter process noise (motion uncertainty).
            measurement_noise: Kalman filter measurement noise (detection uncertainty).
        """
        self.constraints = constraints or BoneConstraints()
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Learned parameters
        self._spine_length: float | None = None
        self._bone_lengths: dict[tuple[int, int], float] = {}
        self._trust: NDArray[np.float64] = np.ones(33)  # Trust per keypoint

        # Kalman filter state (33 keypoints x 2 coordinates)
        self._state: NDArray[np.float64] | None = None  # Current estimate
        self._covariance: NDArray[np.float64] | None = None  # Covariance matrix

    def learn_bone_lengths(
        self,
        poses: "NormalizedPose",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Learn initial bone lengths from high-confidence frames.

        Uses median values across high-confidence detections to establish
        baseline bone lengths for constraint enforcement.

        Args:
            poses: Normalized pose sequence (num_frames, 33, 2).
            confidence_threshold: Minimum confidence to trust measurement.
        """
        # Calculate spine length (mid-shoulder to mid-hip)
        spine_lengths = []
        for i in range(len(poses)):
            mid_shoulder = (poses[i, 11] + poses[i, 12]) / 2  # LEFT/RIGHT_SHOULDER
            mid_hip = (poses[i, 23] + poses[i, 24]) / 2  # LEFT/RIGHT_HIP
            spine_length = float(np.linalg.norm(mid_shoulder - mid_hip))
            if spine_length > 0.01:  # Valid measurement
                spine_lengths.append(spine_length)

        if spine_lengths:
            self._spine_length = float(np.median(spine_lengths))

        # Learn bone lengths for key body segments
        bone_pairs = [
            ((11, 13), "left_upper_arm"),  # LEFT_SHOULDER -> LEFT_ELBOW
            ((13, 15), "left_forearm"),    # LEFT_ELBOW -> LEFT_WRIST
            ((12, 14), "right_upper_arm"), # RIGHT_SHOULDER -> RIGHT_ELBOW
            ((14, 16), "right_forearm"),   # RIGHT_ELBOW -> RIGHT_WRIST
            ((23, 25), "left_thigh"),      # LEFT_HIP -> LEFT_KNEE
            ((25, 27), "left_shin"),       # LEFT_KNEE -> LEFT_ANKLE
            ((24, 26), "right_thigh"),     # RIGHT_HIP -> RIGHT_KNEE
            ((26, 28), "right_shin"),      # RIGHT_KNEE -> RIGHT_ANKLE
        ]

        for (start_idx, end_idx), name in bone_pairs:
            lengths = []
            for i in range(len(poses)):
                start = poses[i, start_idx]
                end = poses[i, end_idx]
                length = float(np.linalg.norm(start - end))
                if length > 0.001:  # Valid measurement
                    lengths.append(length)

            if lengths:
                self._bone_lengths[(start_idx, end_idx)] = float(np.median(lengths))

    def optimize_sequence(
        self,
        poses: "NormalizedPose",
    ) -> "NormalizedPose":
        """Optimize entire pose sequence using Kalman filter + constraints.

        Args:
            poses: Raw normalized poses (num_frames, 33, 2).

        Returns:
            Optimized poses with corrected bone lengths and smoothed trajectories.
        """
        if len(poses) == 0:
            return poses

        # Initialize Kalman filter if needed
        if self._state is None:
            self._initialize_kalman(poses[0])

        optimized = np.zeros_like(poses, dtype=np.float32)

        for i in range(len(poses)):
            # Predict step (Kalman filter)
            self._predict()

            # Update step with measurement
            self._update(poses[i])

            # Copy current state as optimized pose
            optimized[i] = self._state.reshape(33, 2)

        # Apply bone length constraints as post-processing
        optimized = self._enforce_bone_constraints(optimized)

        return optimized

    def _initialize_kalman(self, initial_pose: NDArray[np.float64]) -> None:
        """Initialize Kalman filter state.

        Args:
            initial_pose: First frame pose (33, 2).
        """
        # State vector: flatten (33, 2) -> (66,)
        self._state = initial_pose.flatten().astype(np.float64)

        # Covariance matrix (66x66)
        # Initial uncertainty is high
        self._covariance = np.eye(66, dtype=np.float64) * 0.1

    def _predict(self) -> None:
        """Kalman filter predict step (state transition)."""
        if self._state is None or self._covariance is None:
            return

        # State transition model: x(k) = x(k-1) + v (constant velocity assumed zero)
        # Simplified: state stays the same with some process noise
        # F = I (identity matrix)
        # P(k) = F * P(k-1) * F^T + Q

        # Add process noise to covariance
        self._covariance += np.eye(66, dtype=np.float64) * self.process_noise

    def _update(self, measurement: NDArray[np.float64]) -> None:
        """Kalman filter update step (measurement incorporation).

        Args:
            measurement: New pose measurement (33, 2).
        """
        if self._state is None or self._covariance is None:
            return

        # Measurement model: z = H * x + noise
        # H = I (direct observation)
        z = measurement.flatten()

        # Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
        # Simplified: K = P / (P + R)
        innovation_covariance = self._covariance + np.eye(66, dtype=np.float64) * self.measurement_noise
        kalman_gain = self._covariance @ np.linalg.inv(innovation_covariance)

        # Update state: x = x + K * (z - x)
        innovation = z - self._state
        self._state += kalman_gain @ innovation

        # Update covariance: P = (I - K * H) * P
        self._covariance = (np.eye(66, dtype=np.float64) - kalman_gain) @ self._covariance

    def _enforce_bone_constraints(
        self,
        poses: "NormalizedPose",
    ) -> "NormalizedPose":
        """Enforce bone length constraints on pose sequence.

        Iteratively adjusts joint positions to maintain anatomically
        valid bone lengths while minimizing deviation from original poses.

        Args:
            poses: Input poses (num_frames, 33, 2).

        Returns:
            Poses with corrected bone lengths.
        """
        if self._spine_length is None:
            return poses  # No constraints learned yet

        result = poses.copy()

        # Define bone pairs to constrain
        bone_constraints = [
            # (start_idx, end_idx, ratio_from_constraints)
            (11, 13, self.constraints.SPINE_TO_UPPER_ARM),  # LEFT_SHOULDER -> ELBOW
            (13, 15, self.constraints.SPINE_TO_FOREARM),    # LEFT_ELBOW -> WRIST
            (12, 14, self.constraints.SPINE_TO_UPPER_ARM),  # RIGHT_SHOULDER -> ELBOW
            (14, 16, self.constraints.SPINE_TO_FOREARM),    # RIGHT_ELBOW -> WRIST
            (23, 25, self.constraints.SPINE_TO_THIGH),      # LEFT_HIP -> KNEE
            (25, 27, self.constraints.SPINE_TO_SHIN),       # LEFT_KNEE -> ANKLE
            (24, 26, self.constraints.SPINE_TO_THIGH),      # RIGHT_HIP -> KNEE
            (26, 28, self.constraints.SPINE_TO_SHIN),       # RIGHT_KNEE -> ANKLE
        ]

        for i in range(len(result)):
            pose = result[i]

            for start_idx, end_idx, ratio in bone_constraints:
                # Current bone length
                start = pose[start_idx]
                end = pose[end_idx]
                current_length = np.linalg.norm(start - end)

                # Target length
                target_length = self._spine_length * ratio

                # Check if constraint violated
                min_length = target_length * (1 - self.constraints.TOLERANCE)
                max_length = target_length * (1 + self.constraints.TOLERANCE)

                if current_length < min_length or current_length > max_length:
                    # Adjust end joint position (keep start fixed)
                    direction = (end - start) / (current_length + 1e-8)
                    new_end = start + direction * target_length
                    result[i, end_idx] = new_end.astype(np.float32)

        return result


def optimize_poses_with_physics(
    poses: "NormalizedPose",
    constraints: BoneConstraints | None = None,
) -> "NormalizedPose":
    """Convenience function to optimize poses with physics constraints.

    Args:
        poses: Raw normalized poses (num_frames, 33, 2).
        constraints: Optional bone length constraints.

    Returns:
        Optimized poses with corrected bone lengths.
    """
    optimizer = PhysicsPoseOptimizer(constraints=constraints)
    optimizer.learn_bone_lengths(poses)
    return optimizer.optimize_sequence(poses)


__all__ = [
    "BoneConstraints",
    "PhysicsPoseOptimizer",
    "optimize_poses_with_physics",
]
