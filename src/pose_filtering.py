"""Pose filtering using Hampel outlier rejection and enhanced Kalman filtering.

Implements improved pose trajectory filtering based on Pose2Sim findings:
- Hampel filter: Median Absolute Deviation (MAD) based outlier rejection
- Enhanced Kalman: 6-state model [x, vx, ax, y, vy, ay] with RTS smoothing
- Combined pipeline: Hampel → Kalman → RTS for optimal trajectory estimation

Reference: Leuthold et al. (2025) "Physics Informed Human Posture Estimation"
- Results: -10.2% MPJPE, -16.6% joint angle error

Dependencies:
    - filterpy>=1.4.5: KalmanFilter, RTS_smoother
    - scipy: signal processing, median filtering
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from filterpy.kalman import KalmanFilter, RTS_smoother
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from .types import NormalizedPose


def hampel_filter(
    poses: np.ndarray,
    window_size: int = 7,
    n_sigma: float = 2.0,
    min_samples: int = 3,
) -> np.ndarray:
    """Remove outliers from pose sequence using Hampel filter.

    The Hampel filter identifies outliers using the Median Absolute Deviation (MAD)
    and replaces them with the median value from the sliding window.

    Args:
        poses: (N, 33, 2) pose sequence in normalized coordinates [0, 1]
        window_size: Size of sliding window for median/MAD calculation (default: 7)
        n_sigma: Modified z-score threshold (default: 2.0 = 95% confidence interval)
        min_samples: Minimum samples required for filtering (default: 3)

    Returns:
        Filtered poses with outliers replaced by local median values.

    Reference:
        Pose2Sim/filtering.py lines 63-84
        Hampel, F. R. "The influence curve and its role in outlier detection"
        Proceedings of the IMS 1971.
    """
    if window_size < min_samples:
        window_size = min_samples

    poses_filtered = poses.copy()
    n_frames, n_kpts, n_coords = poses.shape

    # Process each keypoint and coordinate independently
    for kp_idx in range(n_kpts):
        for coord_idx in range(n_coords):
            signal = poses[:, kp_idx, coord_idx]

            # Apply median filter for sliding window median
            median = median_filter(signal, size=window_size, mode='nearest')

            # Calculate Median Absolute Deviation (MAD)
            mad = median_filter(np.abs(signal - median), size=window_size, mode='nearest')

            # Modified z-score (0.6745 makes it consistent with standard deviation for normal distributions)
            with np.errstate(divide='ignore', invalid='ignore'):
                z_score = 0.6745 * (signal - median) / (mad + 1e-8)

            # Identify and replace outliers
            outlier_mask = np.abs(z_score) > n_sigma
            poses_filtered[outlier_mask, kp_idx, coord_idx] = median[outlier_mask]

    return poses_filtered


@dataclass
class EnhancedKalmanConfig:
    """Configuration for enhanced Kalman filter.

    Attributes:
        fps: Frame rate of video (for dt calculation)
        process_noise: Motion uncertainty (Q matrix)
        measurement_noise: Detection uncertainty (R matrix)
        use_rts: Whether to use RTS (Rauch-Tung-Striebel) smoother
    """

    fps: float = 30.0
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    use_rts: bool = True


class EnhancedPoseFilter:
    """Enhanced 6-state Kalman filter for pose trajectory smoothing.

    Uses a constant acceleration model with state vector [x, vx, ax, y, vy, ay]
    for each of the 33 BlazePose keypoints. This provides:
    - Velocity estimation for motion prediction
    - Acceleration estimation for dynamics
    - Better tracking during fast motions (jumps, spins)
    - RTS smoother for bidirectional trajectory optimization

    State transition model (constant acceleration):
        x(t+dt) = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2
        vx(t+dt) = vx(t) + ax(t)*dt
        ax(t+dt) = ax(t)

    Based on:
        - Leuthold et al. (2025) "Physics Informed Human Posture Estimation"
        - Pose2Sim/filtering.py Kalman filter implementation

    Attributes:
        config: Filter configuration parameters.
        kf: KalmanFilter instance from filterpy.
        dt: Time step (1/fps).
        state_dim: Total state dimension (33 keypoints * 6 states = 198).
        meas_dim: Measurement dimension (33 keypoints * 2 coords = 66).
    """

    def __init__(self, config: EnhancedKalmanConfig | None = None):
        """Initialize enhanced pose filter.

        Args:
            config: Filter configuration (uses defaults if None).
        """
        self.config = config or EnhancedKalmanConfig()
        self.dt = 1.0 / self.config.fps

        # State dimension: [x, vx, ax, y, vy, ay] for each of 33 keypoints
        self.state_dim = 33 * 6  # 198 dimensions total
        self.meas_dim = 33 * 2  # 66 dimensions (x, y for each keypoint)

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=self.state_dim, dim_z=self.meas_dim)

        # Initial state (all zeros)
        self.kf.x = np.zeros(self.state_dim)

        # State transition matrix (constant acceleration model)
        self.kf.F = self._build_transition_matrix()

        # Measurement matrix (observe positions only)
        self.kf.H = self._build_measurement_matrix()

        # Noise matrices
        self.kf.Q = np.eye(self.state_dim) * self.config.process_noise
        self.kf.R = np.eye(self.meas_dim) * self.config.measurement_noise

        # Initial covariance (large uncertainty)
        self.kf.P = np.eye(self.state_dim) * 1.0

        # Store measurements for RTS smoothing
        self._measurements: list[np.ndarray] = []
        self._is_initialized = False

    def _build_transition_matrix(self) -> np.ndarray:
        """Build state transition matrix F for constant acceleration model.

        For each keypoint (33 total), the 6x6 block is:
            [1  dt  0.5*dt^2]   [x]
            [0  1   dt      ]   [vx]
            [0  0   1      ]   [ax]

        Returns:
            F: (198, 198) transition matrix
        """
        dt = self.dt
        F = np.eye(self.state_dim)

        for i in range(33):
            base = i * 6
            # x += vx * dt + 0.5 * ax * dt^2
            F[base, base + 1] = dt
            F[base, base + 2] = 0.5 * dt**2

            # vx += ax * dt
            F[base + 1, base + 2] = dt

            # Same for y dimension
            y_base = base + 99  # y state starts after x states (33 * 3 = 99)
            F[y_base, y_base + 1] = dt
            F[y_base, y_base + 2] = 0.5 * dt**2
            F[y_base + 1, y_base + 2] = dt

        return F

    def _build_measurement_matrix(self) -> np.ndarray:
        """Build measurement matrix H (observe positions only).

        Maps 198-dim state to 66-dim measurement [x, y] for each keypoint.

        Returns:
            H: (66, 198) measurement matrix
        """
        H = np.zeros((self.meas_dim, self.state_dim))

        for i in range(33):
            # x position measurement
            H[i * 2, i * 6] = 1.0  # Observe x state
            # y position measurement
            H[i * 2 + 1, i * 6 + 99] = 1.0  # Observe y state

        return H

    def init_from_poses(self, poses: np.ndarray) -> None:
        """Initialize filter state from first poses.

        Args:
            poses: (N_init, 33, 2) initial poses for state initialization
        """
        if len(poses) < 2:
            raise ValueError("Need at least 2 frames for initialization")

        # Use average position and velocity from first frames
        mean_pos = poses[:5].mean(axis=0) if len(poses) >= 5 else poses[0]

        # Initialize positions
        for i in range(33):
            self.kf.x[i * 6] = mean_pos[i, 0]  # x
            self.kf.x[i * 6 + 99] = mean_pos[i, 1]  # y

        # Estimate velocity from first two frames
        if len(poses) >= 2:
            velocity = (poses[1] - poses[0]) / self.dt
            for i in range(33):
                self.kf.x[i * 6 + 1] = velocity[i, 0]  # vx
                self.kf.x[i * 6 + 100] = velocity[i, 1]  # vy

        self._is_initialized = True

    def filter_sequence(self, poses: np.ndarray) -> np.ndarray:
        """Filter a sequence of poses using Kalman filter with optional RTS smoothing.

        Args:
            poses: (N, 33, 2) pose sequence to filter

        Returns:
            Filtered poses with same shape as input
        """
        if not self._is_initialized:
            self.init_from_poses(poses)

        n_frames = poses.shape[0]
        poses_filtered = np.zeros_like(poses)

        # Forward pass: predict and update for each frame
        state_estimates = []

        for t in range(n_frames):
            # Flatten pose to measurement vector
            z = poses[t].reshape(-1)  # (66,)

            if t > 0:
                # Predict step
                self.kf.predict()

            # Update step
            self.kf.update(z)
            state_estimates.append(self.kf.x.copy())

        # Apply RTS smoothing if enabled
        if self.config.use_rts and len(state_estimates) > 1:
            smoothed_states = self._apply_rts_smoother(state_estimates)
        else:
            smoothed_states = state_estimates

        # Extract positions from state estimates
        for t, state in enumerate(smoothed_states):
            # Extract x and y for each keypoint
            for i in range(33):
                poses_filtered[t, i, 0] = state[i * 6]  # x
                poses_filtered[t, i, 1] = state[i * 6 + 99]  # y

        return poses_filtered

    def _apply_rts_smoother(
        self, state_estimates: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Apply Rauch-Tung-Striebel (RTS) smoother for bidirectional filtering.

        RTS smoothing uses future measurements to improve past estimates,
        resulting in smoother trajectories with less lag.

        Args:
            state_estimates: List of Kalman filter states from forward pass

        Returns:
            List of RTS-smoothed state estimates
        """
        # Convert list of states to array (T, state_dim)
        states = np.stack(state_estimates)

        # Create RTS smoother
        smoother = RTS_smoother(self.kf)

        # Apply smoothing (processes all states at once)
        smoothed_states, _ = smoother.smooth(states, dt=self.dt)

        # Convert back to list
        return [smoothed_states[t] for t in range(smoothed_states.shape[0])]


def filter_pose_sequence(
    poses: np.ndarray,
    fps: float,
    use_hampel: bool = True,
    hampel_window: int = 7,
    hampel_sigma: float = 2.0,
    kalman_config: EnhancedKalmanConfig | None = None,
) -> np.ndarray:
    """Apply complete filtering pipeline to pose sequence.

    Pipeline: Hampel (outlier rejection) → Kalman (smoothing) → RTS (bidirectional)

    Args:
        poses: (N, 33, 2) pose sequence to filter
        fps: Frame rate for dt calculation
        use_hampel: Whether to apply Hampel filter first
        hampel_window: Window size for Hampel filter
        hampel_sigma: Sigma threshold for Hampel filter
        kalman_config: Configuration for Kalman filter

    Returns:
        Filtered poses with same shape as input

    Example:
        >>> poses_raw = extract_poses(video)
        >>> poses_clean = filter_pose_sequence(poses_raw, fps=30.0)
    """
    # Step 1: Hampel filter for outlier rejection
    if use_hampel:
        poses = hampel_filter(poses, window_size=hampel_window, n_sigma=hampel_sigma)

    # Step 2: Enhanced Kalman filter with RTS smoothing
    if kalman_config is None:
        kalman_config = EnhancedKalmanConfig(fps=fps)

    kalman_filter = EnhancedPoseFilter(kalman_config)
    poses_filtered = kalman_filter.filter_sequence(poses)

    return poses_filtered
