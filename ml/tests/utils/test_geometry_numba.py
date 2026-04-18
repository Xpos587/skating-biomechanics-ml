"""Tests for Numba-jitted geometry functions."""

import numpy as np
import pytest

from skating_ml.utils.geometry import (
    _angle_3pt_rad,
    _distance_numba,
    angle_3pt,
    angle_3pt_batch,
    distance,
)


class TestGeometryNumba:
    """Tests for Numba-jitted geometry functions."""

    def test_angle_3pt_basic(self):
        """Basic angle calculation."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])

        angle = angle_3pt(a, b, c)
        assert np.isclose(angle, 90.0)

    def test_angle_3pt_180_degrees(self):
        """Straight line = 180 degrees."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])

        angle = angle_3pt(a, b, c)
        assert np.isclose(angle, 180.0, rtol=1e-4)

    def test_angle_3pt_jitted_core(self):
        """Test jitted core function returns radians."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])

        angle_rad = _angle_3pt_rad(a, b, c)
        assert np.isclose(angle_rad, np.pi / 2)

    def test_distance_basic(self):
        """Distance calculation."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])

        dist = distance(p1, p2)
        assert np.isclose(dist, 5.0)

    def test_distance_jitted(self):
        """Test jitted distance function."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])

        dist = _distance_numba(p1, p2)
        assert np.isclose(dist, 5.0)

    def test_angle_3pt_batch(self):
        """Batch angle calculation."""
        triplets = np.array(
            [
                [[0, 0], [1, 0], [1, 1]],  # 90 deg
                [[0, 0], [1, 0], [2, 0]],  # 180 deg
                [[0, 1], [1, 0], [0, -1]],  # 90 deg (vertical)
            ],
            dtype=np.float64,
        )

        angles = angle_3pt_batch(triplets)
        assert np.isclose(angles[0], 90.0)
        assert np.isclose(angles[1], 180.0, rtol=1e-4)
        assert np.isclose(angles[2], 90.0)

    def test_angle_3pt_batch_single(self):
        """Batch with single triplet."""
        triplets = np.array([[[0, 0], [1, 0], [1, 1]]], dtype=np.float64)

        angles = angle_3pt_batch(triplets)
        assert angles.shape == (1,)
        assert np.isclose(angles[0], 90.0)

    def test_jit_compilation_speed(self):
        """Verify JIT compilation happened (second call should be fast)."""
        import time

        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])

        # First call (compilation)
        _ = _angle_3pt_rad(a, b, c)

        # Second call (compiled)
        start = time.perf_counter()
        for _ in range(10000):
            _angle_3pt_rad(a, b, c)
        elapsed = time.perf_counter() - start

        # Should be very fast
        assert elapsed < 0.1  # 10k calls in < 100ms

    def test_batch_vs_loop_consistency(self):
        """Batch should give same results as loop."""
        triplets = np.random.randn(100, 3, 2)

        # Batch version
        angles_batch = angle_3pt_batch(triplets)

        # Loop version
        angles_loop = np.empty(100, dtype=np.float64)
        for i in range(100):
            angles_loop[i] = angle_3pt(triplets[i, 0], triplets[i, 1], triplets[i, 2])

        np.testing.assert_allclose(angles_batch, angles_loop, rtol=1e-6)

    def test_angle_3pt_zero_vector(self):
        """Handle zero-length vectors gracefully."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])

        # Should not crash due to 1e-8 epsilon in denominator
        angle = angle_3pt(a, b, c)
        # Result is undefined but should be finite
        assert np.isfinite(angle)
