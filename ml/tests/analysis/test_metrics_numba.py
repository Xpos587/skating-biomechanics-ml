"""Tests for Numba-jitted metrics functions."""

import numpy as np
import pytest

from skating_ml.analysis.metrics import (
    _angle_3pt_rad_numba,
    _compute_knee_angle_series_numba,
    _compute_trunk_lean_series_numba,
)
from skating_ml.types import H36Key


class TestMetricsNumba:
    """Tests for Numba-jitted metrics functions."""

    def test_angle_3pt_rad_basic(self):
        """Basic angle calculation in radians."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])

        angle = _angle_3pt_rad_numba(a, b, c)
        assert np.isclose(angle, np.pi / 2)

    def test_angle_3pt_rad_straight(self):
        """Straight line = π radians."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])

        angle = _angle_3pt_rad_numba(a, b, c)
        assert np.isclose(angle, np.pi, rtol=1e-4)

    def test_compute_knee_angle_series_shape(self):
        """Output shape should match input."""
        poses = np.random.randn(50, 17, 2).astype(np.float32)
        angles = _compute_knee_angle_series_numba(
            poses,
            int(H36Key.LHIP),
            int(H36Key.LKNEE),
            int(H36Key.LFOOT),
        )

        assert angles.shape == (50,)

    def test_compute_knee_angle_series_straight_leg(self):
        """Straight leg should give ~180 degrees."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        # Set hip, knee, foot in a straight line
        poses[0, H36Key.LHIP] = [0, 0]
        poses[0, H36Key.LKNEE] = [1, 0]
        poses[0, H36Key.LFOOT] = [2, 0]

        angles = _compute_knee_angle_series_numba(
            poses,
            int(H36Key.LHIP),
            int(H36Key.LKNEE),
            int(H36Key.LFOOT),
        )

        assert np.isclose(angles[0], 180.0, rtol=1e-4)

    def test_compute_knee_angle_series_bent_knee(self):
        """Bent knee should give < 180 degrees."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        # Hip at origin, knee forward, foot below
        poses[0, H36Key.LHIP] = [0, 0]
        poses[0, H36Key.LKNEE] = [1, 0]
        poses[0, H36Key.LFOOT] = [1, -1]

        angles = _compute_knee_angle_series_numba(
            poses,
            int(H36Key.LHIP),
            int(H36Key.LKNEE),
            int(H36Key.LFOOT),
        )

        assert angles[0] <= 180.0
        assert angles[0] >= 90.0

    def test_compute_knee_angle_series_right_knee(self):
        """Right knee should work too."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        poses[0, H36Key.RHIP] = [0, 0]
        poses[0, H36Key.RKNEE] = [1, 0]
        poses[0, H36Key.RFOOT] = [2, 0]

        angles = _compute_knee_angle_series_numba(
            poses,
            int(H36Key.RHIP),
            int(H36Key.RKNEE),
            int(H36Key.RFOOT),
        )

        assert np.isclose(angles[0], 180.0, rtol=1e-4)

    def test_compute_trunk_lean_series_shape(self):
        """Output shape should match input."""
        poses = np.random.randn(50, 17, 2).astype(np.float32)
        leans = _compute_trunk_lean_series_numba(poses)

        assert leans.shape == (50,)

    def test_compute_trunk_lean_upright(self):
        """Upright pose should give ~0 degrees."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        # Shoulders directly above hips
        poses[0, H36Key.LHIP] = [0, 1]
        poses[0, H36Key.RHIP] = [0, 1]
        poses[0, H36Key.LSHOULDER] = [0, 0]
        poses[0, H36Key.RSHOULDER] = [0, 0]

        leans = _compute_trunk_lean_series_numba(poses)

        # Upright = 0 degrees (with some tolerance for float)
        assert np.abs(leans[0]) < 5.0

    def test_compute_trunk_lean_forward(self):
        """Forward lean should give positive angle."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        # Shoulders forward of hips
        poses[0, H36Key.LHIP] = [0, 1]
        poses[0, H36Key.RHIP] = [0, 1]
        poses[0, H36Key.LSHOULDER] = [0.5, 0]
        poses[0, H36Key.RSHOULDER] = [0.5, 0]

        leans = _compute_trunk_lean_series_numba(poses)

        # Forward lean = positive angle
        assert leans[0] > 0

    def test_compute_trunk_lean_backward(self):
        """Backward lean should give negative angle."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        # Shoulders behind hips
        poses[0, H36Key.LHIP] = [0, 1]
        poses[0, H36Key.RHIP] = [0, 1]
        poses[0, H36Key.LSHOULDER] = [-0.5, 0]
        poses[0, H36Key.RSHOULDER] = [-0.5, 0]

        leans = _compute_trunk_lean_series_numba(poses)

        # Backward lean = negative angle
        assert leans[0] < 0

    def test_jit_compilation_speed(self):
        """Verify JIT makes subsequent calls fast."""
        import time

        poses = np.random.randn(100, 17, 2).astype(np.float32)

        # Compilation call
        _ = _compute_knee_angle_series_numba(
            poses,
            int(H36Key.LHIP),
            int(H36Key.LKNEE),
            int(H36Key.LFOOT),
        )

        # Compiled calls
        start = time.perf_counter()
        for _ in range(100):
            _compute_knee_angle_series_numba(
                poses,
                int(H36Key.LHIP),
                int(H36Key.LKNEE),
                int(H36Key.LFOOT),
            )
        elapsed = time.perf_counter() - start

        # Should be fast
        assert elapsed < 0.5  # 100 calls in < 500ms
