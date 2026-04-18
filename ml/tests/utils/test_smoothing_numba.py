"""Tests for Numba-jitted smoothing functions."""

import numpy as np
import pytest

from skating_ml.utils.smoothing import (
    _one_euro_filter_sequence_numba,
    _smoothing_factor_numba,
    _exponential_smoothing_numba,
    smooth_trajectory_2d_numba,
)


class TestSmoothingNumba:
    """Tests for Numba-jitted smoothing functions."""

    def test_smoothing_factor_basic(self):
        """Basic smoothing factor calculation."""
        # te=1.0, cutoff=1.0 → r=2π → alpha = 2π/(2π+1) ≈ 0.863
        alpha = _smoothing_factor_numba(1.0, 1.0)
        expected = 2 * np.pi / (2 * np.pi + 1.0)
        assert np.isclose(alpha, expected)

    def test_smoothing_factor_zero_time(self):
        """Zero time interval should give zero smoothing (pass-through)."""
        alpha = _smoothing_factor_numba(0.0, 1.0)
        assert alpha == 0.0

    def test_exponential_smoothing_basic(self):
        """Basic exponential smoothing."""
        # alpha=0.5, x=10, x_prev=0 → output=5
        result = _exponential_smoothing_numba(0.5, 10.0, 0.0)
        assert result == 5.0

    def test_exponential_smoothing_no_filter(self):
        """alpha=1 means no filtering (pass-through)."""
        result = _exponential_smoothing_numba(1.0, 10.0, 0.0)
        assert result == 10.0

    def test_exponential_smoothing_full_filter(self):
        """alpha=0 means no update (keep previous)."""
        result = _exponential_smoothing_numba(0.0, 10.0, 5.0)
        assert result == 5.0

    def test_one_euro_filter_sequence_constant(self):
        """Constant signal should pass through with minimal lag."""
        x = np.ones(100, dtype=np.float64)
        filtered = _one_euro_filter_sequence_numba(x, freq=30.0, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0)

        # First sample passes through unchanged
        assert filtered[0] == 1.0

        # Rest should be close to 1.0 (some smoothing expected)
        assert np.allclose(filtered[1:], 1.0, rtol=0.1)

    def test_one_euro_filter_sequence_length(self):
        """Output length should match input."""
        x = np.random.randn(50).astype(np.float64)
        filtered = _one_euro_filter_sequence_numba(x, freq=30.0, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0)

        assert len(filtered) == len(x)

    def test_one_euro_filter_reduces_noise(self):
        """Filter should reduce noise variance."""
        # Create constant signal with noise
        np.random.seed(42)
        clean = np.ones(100) * 5.0
        noise = np.random.randn(100) * 0.1
        noisy = clean + noise

        # Filter
        filtered = _one_euro_filter_sequence_numba(
            noisy.astype(np.float64),
            freq=30.0,
            min_cutoff=0.3,
            beta=0.007,
            derivative_cutoff=1.0,
        )

        # Filter should reduce variance (noise)
        assert np.var(filtered) < np.var(noisy)

    def test_smooth_trajectory_2d_shape(self):
        """Output shape should match input."""
        trajectory = np.random.randn(50, 2).astype(np.float64)
        smoothed = smooth_trajectory_2d_numba(trajectory, fps=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        assert smoothed.shape == trajectory.shape

    def test_smooth_trajectory_2d_first_unchanged(self):
        """First point should be unchanged (initialization)."""
        trajectory = np.random.randn(50, 2).astype(np.float64)
        smoothed = smooth_trajectory_2d_numba(trajectory, fps=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        assert np.allclose(smoothed[0], trajectory[0])

    def test_smooth_trajectory_2d_reduces_noise(self):
        """Smoothing should reduce noise variance."""
        # Create constant trajectory with noise
        np.random.seed(42)
        clean = np.ones((100, 2)) * 5.0
        noise = np.random.randn(100, 2) * 0.1
        noisy = clean + noise

        # Smooth
        smoothed = smooth_trajectory_2d_numba(
            noisy.astype(np.float64),
            fps=30.0,
            min_cutoff=0.3,
            beta=0.007,
            d_cutoff=1.0,
        )

        # Variance should be reduced for both dimensions
        assert np.var(smoothed[:, 0]) < np.var(noisy[:, 0])
        assert np.var(smoothed[:, 1]) < np.var(noisy[:, 1])

    def test_jit_compilation_speed(self):
        """Verify JIT makes subsequent calls fast."""
        import time

        x = np.random.randn(100).astype(np.float64)

        # Compilation call
        _ = _one_euro_filter_sequence_numba(x, freq=30.0, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0)

        # Compiled calls
        start = time.perf_counter()
        for _ in range(100):
            _one_euro_filter_sequence_numba(x, freq=30.0, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0)
        elapsed = time.perf_counter() - start

        # Should be fast
        assert elapsed < 0.5  # 100 calls in < 500ms
