"""Temporal smoothing using One-Euro Filter for pose sequences.

The One-Euro Filter (Casiez et al., 2012) is an adaptive low-pass filter
that combines noise reduction at low speeds with minimal lag at high speeds.
Ideal for smoothing human motion capture data from BlazePose.

Reference: https://github.com/jaantollander/OneEuroFilter
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..types import NormalizedPose


@dataclass(frozen=True)
class OneEuroFilterConfig:
    """Configuration for One-Euro Filter parameters.

    Defaults optimized for figure skating motion at 25-60 fps.

    Attributes:
        min_cutoff: Minimum cutoff frequency (Hz) - controls jitter reduction
            at low speeds. Lower = more smoothing but more lag. Range: [0.1, 10.0]
        beta: Speed coefficient - reduces lag at high speeds.
            Higher = less lag but more jitter. Range: [0.0, 1.0]
        derivative_cutoff: Cutoff frequency for velocity estimation. Range: [0.1, 10.0]
        freq: Sampling frequency in Hz (frames per second).
    """

    min_cutoff: float = 1.0
    beta: float = 0.007
    derivative_cutoff: float = 1.0
    freq: float = 30.0


class OneEuroFilter:
    """One-Euro Filter for smoothing noisy 1D signals.

    Implements the adaptive low-pass filter from Casiez et al. (2012).
    Filters a single time series (e.g., x-coordinate of one joint).
    Stateful: processes samples incrementally.

    Example:
        >>> filter = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
        >>> filtered = filter.reset_and_filter(x_sequence)
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        derivative_cutoff: float = 1.0,
    ) -> None:
        """Initialize One-Euro Filter.

        Args:
            freq: Sampling frequency in Hz (frames per second).
            min_cutoff: Minimum cutoff frequency in Hz.
            beta: Speed coefficient for adaptive cutoff.
            derivative_cutoff: Cutoff for derivative filtering.
        """
        self.freq: Final[float] = freq
        self.min_cutoff: Final[float] = min_cutoff
        self.beta: Final[float] = beta
        self.derivative_cutoff: Final[float] = derivative_cutoff

        # State variables (reset between sequences)
        self._x_prev: float = 0.0
        self._dx_prev: float = 0.0
        self._t_prev: float = 0.0
        self._initialized: bool = False

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        """Compute smoothing factor alpha from time interval and cutoff frequency.

        Args:
            te: Time interval since last sample.
            cutoff: Cutoff frequency in Hz.

        Returns:
            Smoothing factor alpha in [0, 1].
        """
        r = 2.0 * np.pi * cutoff * te
        return r / (r + 1.0)

    @staticmethod
    def _exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing filter.

        Args:
            alpha: Smoothing factor.
            x: Current input value.
            x_prev: Previous filtered value.

        Returns:
            Filtered output value.
        """
        return alpha * x + (1.0 - alpha) * x_prev

    def reset(self) -> None:
        """Reset filter state for new sequence."""
        self._x_prev = 0.0
        self._dx_prev = 0.0
        self._t_prev = 0.0
        self._initialized = False

    def filter_sample(self, t: float, x: float) -> float:
        """Filter a single sample (stateful incremental processing).

        Args:
            t: Timestamp in seconds.
            x: Input value to filter.

        Returns:
            Filtered value.

        Raises:
            ValueError: If timestamps are not monotonically increasing.
        """
        if not self._initialized:
            # First sample - pass through
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            self._initialized = True
            return x

        # Check monotonic timestamps
        if t <= self._t_prev:
            msg = f"Timestamps must be monotonically increasing: {t} <= {self._t_prev}"
            raise ValueError(msg)

        # Time interval
        te = t - self._t_prev

        # Filter the derivative (velocity)
        dx = (x - self._x_prev) / te
        alpha_d = self._smoothing_factor(te, self.derivative_cutoff)
        dx_hat = self._exponential_smoothing(alpha_d, dx, self._dx_prev)

        # Adaptive cutoff based on filtered velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the signal
        alpha = self._smoothing_factor(te, cutoff)
        x_hat = self._exponential_smoothing(alpha, x, self._x_prev)

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def filter_sequence(
        self,
        x: NDArray[np.float32],
        timestamps: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Filter a complete sequence (batch processing).

        Args:
            x: Input sequence (num_samples,).
            timestamps: Optional timestamps (num_samples,). If None, uses uniform spacing.

        Returns:
            Filtered sequence (num_samples,).
        """
        if timestamps is None:
            timestamps = np.arange(len(x), dtype=np.float32) / self.freq

        if len(x) != len(timestamps):
            msg = f"Length mismatch: {len(x)} != {len(timestamps)}"
            raise ValueError(msg)

        self.reset()
        filtered = np.zeros_like(x)

        for i in range(len(x)):
            filtered[i] = self.filter_sample(float(timestamps[i]), float(x[i]))

        return filtered.astype(np.float32)

    def reset_and_filter(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convenience method: reset and filter sequence with uniform timestamps.

        Args:
            x: Input sequence (num_samples,).

        Returns:
            Filtered sequence (num_samples,).
        """
        return self.filter_sequence(x, None)


class PoseSmoother:
    """Smooth pose sequences using One-Euro Filter.

    Applies One-Euro Filter to all pose keypoints independently.
    Each joint's x and y coordinates are filtered as separate time series.

    Supports H3.6M 17-keypoint format for 3D-only pipeline.

    Integration point in pipeline:
        AthletePose3DExtractor → PoseNormalizer → PoseSmoother → PhaseDetector
                                                                     ↓
                                                                BiomechanicsAnalyzer
    """

    def __init__(
        self,
        config: OneEuroFilterConfig | None = None,
        freq: float = 30.0,
    ) -> None:
        """Initialize pose smoother.

        Args:
            config: Filter configuration. If None, uses skating-optimized defaults.
            freq: Sampling frequency in Hz (video FPS).
        """
        if config is None:
            config = OneEuroFilterConfig(freq=freq)

        self.config = config
        self.freq = freq

        # Create filter for each dimension (33 joints x 2 coords)
        # We'll create filters on-demand to save memory
        self._filters: dict[tuple[int, int], OneEuroFilter] = {}

    def _get_filter(self, joint_idx: int, coord_idx: int) -> OneEuroFilter:
        """Get or create filter for specific joint and coordinate.

        Args:
            joint_idx: BlazePose joint index (0-32).
            coord_idx: Coordinate index (0=x, 1=y).

        Returns:
            OneEuroFilter instance for this time series.
        """
        key = (joint_idx, coord_idx)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                freq=self.freq,
                min_cutoff=self.config.min_cutoff,
                beta=self.config.beta,
                derivative_cutoff=self.config.derivative_cutoff,
            )
        return self._filters[key]

    def smooth(self, poses: NormalizedPose) -> NormalizedPose:
        """Smooth pose sequence using One-Euro Filter.

        Args:
            poses: NormalizedPose (num_frames, num_joints, 2).
                   Supports H3.6M (17 joints) or BlazePose (33 joints).

        Returns:
            Smoothed poses (num_frames, num_joints, 2).
        """
        _num_frames, num_joints, num_coords = poses.shape

        if num_coords != 2:
            msg = f"Expected shape (N, J, 2), got {poses.shape}"
            raise ValueError(msg)

        # Support both H3.6M (17) and BlazePose (33) formats
        if num_joints not in (17, 33):
            msg = f"Expected 17 or 33 joints, got {num_joints}"
            raise ValueError(msg)

        # Create output array
        smoothed = np.zeros_like(poses)

        # Filter each joint and coordinate independently
        for joint_idx in range(num_joints):
            for coord_idx in range(num_coords):
                # Extract time series for this joint/coordinate
                series = poses[:, joint_idx, coord_idx]

                # Get filter and apply
                filter_obj = self._get_filter(joint_idx, coord_idx)
                smoothed[:, joint_idx, coord_idx] = filter_obj.reset_and_filter(series)

        return smoothed

    def smooth_3d(self, poses_3d: NDArray[np.float32]) -> NDArray[np.float32]:
        """Smooth 3D pose sequence using One-Euro Filter.

        Processes x, y, z coordinates independently for each joint.

        Args:
            poses_3d: 3D poses (num_frames, 17, 3) with x, y, z in meters.

        Returns:
            Smoothed 3D poses (num_frames, 17, 3).
        """
        _num_frames, num_joints, num_coords = poses_3d.shape

        if num_joints != 17 or num_coords != 3:
            msg = f"Expected shape (N, 17, 3), got {poses_3d.shape}"
            raise ValueError(msg)

        # Create output array
        smoothed = np.zeros_like(poses_3d)

        # Filter each joint and coordinate independently
        for joint_idx in range(num_joints):
            for coord_idx in range(num_coords):  # x, y, z
                # Extract time series for this joint/coordinate
                series = poses_3d[:, joint_idx, coord_idx]

                # Get filter and apply
                filter_obj = self._get_filter(joint_idx, coord_idx)
                smoothed[:, joint_idx, coord_idx] = filter_obj.reset_and_filter(series)

        return smoothed

    def smooth_phase_aware(
        self,
        poses: NormalizedPose,
        phase_boundaries: list[int],
    ) -> NormalizedPose:
        """Smooth poses with phase-aware processing.

        Resets filter at each phase boundary to avoid smoothing across
        rapid transitions (e.g., takeoff, landing).

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phase_boundaries: List of frame indices where phases change.
                E.g., [takeoff, peak, landing] for jumps.

        Returns:
            Smoothed poses (num_frames, 17, 2).
        """
        if not phase_boundaries:
            return self.smooth(poses)

        # Sort boundaries and add start/end
        boundaries = sorted([0, *phase_boundaries, len(poses)])

        # Create output array
        smoothed = np.zeros_like(poses)

        # Process each phase independently
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end <= start:
                continue

            # Extract phase
            phase_poses = poses[start:end]

            # Smooth phase
            smoothed_phase = self.smooth(phase_poses)

            # Copy to output
            smoothed[start:end] = smoothed_phase

        return smoothed

    def set_frequency(self, freq: float) -> None:
        """Update sampling frequency and reset filters.

        Args:
            freq: New sampling frequency in Hz.
        """
        self.freq = freq
        self._filters.clear()


def get_skating_optimized_config(fps: float = 30.0) -> OneEuroFilterConfig:
    """Get One-Euro Filter config optimized for figure skating.

    Figure skating has specific characteristics:
    - Slow preparatory movements (crossovers, setup)
    - Very fast rotations (jumps: 300-600 deg/s)
    - Sudden transitions (takeoff, landing)

    This config balances jitter reduction for slow movements
    with minimal lag for fast rotations.

    Args:
        fps: Video frame rate (affects frequency scaling).

    Returns:
        Optimized configuration for figure skating.
    """
    # Scale parameters based on FPS
    # Higher FPS → slightly higher min_cutoff (less smoothing needed)
    # Lower FPS → slightly lower min_cutoff (more smoothing needed)
    base_min_cutoff = 1.0
    fps_scaling = fps / 30.0  # Normalize to 30 FPS
    min_cutoff = base_min_cutoff * fps_scaling

    # Beta: controls lag reduction at high speeds
    # Skating needs fast response during rotations
    beta = 0.007  # Conservative: reduce jitter more than lag

    return OneEuroFilterConfig(
        min_cutoff=min_cutoff,
        beta=beta,
        derivative_cutoff=1.0,
        freq=fps,
    )
