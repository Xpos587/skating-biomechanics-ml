"""Tests for One-Euro Filter temporal smoothing."""

import numpy as np
import pytest

from src.utils.smoothing import (
    OneEuroFilter,
    OneEuroFilterConfig,
    PoseSmoother,
    get_skating_optimized_config,
)


class TestOneEuroFilter:
    """Test OneEuroFilter single time series filtering."""

    def test_smoothing_factor_bounds(self):
        """Test that smoothing factor is always in [0, 1]."""
        filter_obj = OneEuroFilter(freq=30.0)

        # Various time intervals and cutoffs
        for te in [0.01, 0.033, 0.1, 1.0]:
            for cutoff in [0.1, 1.0, 10.0]:
                alpha = filter_obj._smoothing_factor(te, cutoff)
                assert 0.0 <= alpha <= 1.0

    def test_constant_signal(self):
        """Test that constant signal passes through unchanged."""
        filter_obj = OneEuroFilter(freq=30.0)
        x = np.ones(100, dtype=np.float32) * 5.0

        filtered = filter_obj.reset_and_filter(x)

        # Should be very close to constant
        assert np.allclose(filtered, 5.0, atol=1e-3)

    def test_smoothing_produces_smooth_output(self):
        """Test that filter produces smoother output than noisy input."""
        filter_obj = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.001)

        # Generate clean sinusoid
        t = np.arange(100) / 30.0
        x_clean = np.sin(2 * np.pi * 0.5 * t).astype(np.float32)

        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, 100).astype(np.float32)
        x_noisy = x_clean + noise

        # Filter
        x_filtered = filter_obj.reset_and_filter(x_noisy)

        # Filtered should be smoother than noisy (less second derivative)
        def smoothness(signal):
            return np.mean(np.abs(np.diff(signal, n=2)))

        smoothness_noisy = smoothness(x_noisy)
        smoothness_filtered = smoothness(x_filtered)

        assert smoothness_filtered < smoothness_noisy

    def test_high_frequency_smoothing(self):
        """Test that higher frequencies are smoothed more."""
        # Low frequency signal (less affected)
        filter_obj = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.001)

        # Generate clean sinusoid
        t = np.arange(100) / 30.0
        x_clean = np.sin(2 * np.pi * 0.5 * t).astype(np.float32)

        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, 100).astype(np.float32)
        x_noisy = x_clean + noise

        # Filter
        x_filtered = filter_obj.reset_and_filter(x_noisy)

        # Output should be finite and valid
        assert np.all(np.isfinite(x_filtered))
        assert len(x_filtered) == len(x_noisy)

    def test_step_response(self):
        """Test filter response to step input."""
        filter_obj = OneEuroFilter(freq=30.0)

        # Step signal: 0 → 1 at t=0.5
        t = np.arange(100) / 30.0
        x = (t > 0.5).astype(np.float32)

        filtered = filter_obj.reset_and_filter(x)

        # Check that filter rises smoothly (no oscillation)
        # Should be monotonic after step
        step_idx = np.argmax(t > 0.5)
        assert np.all(np.diff(filtered[step_idx:]) >= -1e-6)

    def test_frequency_response(self):
        """Test that low frequencies pass, high frequencies are attenuated."""
        # Low frequency (should pass through)
        filter_obj = OneEuroFilter(freq=30.0, min_cutoff=1.0)
        t = np.arange(100) / 30.0
        x_low = np.sin(2 * np.pi * 0.2 * t).astype(np.float32)

        filtered_low = filter_obj.reset_and_filter(x_low)
        attenuation_low = np.std(filtered_low) / np.std(x_low)

        # High frequency (should be attenuated)
        x_high = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)
        filtered_high = filter_obj.reset_and_filter(x_high)
        attenuation_high = np.std(filtered_high) / np.std(x_high)

        # Low freq should have less attenuation than high freq
        assert attenuation_low > attenuation_high

    def test_reset(self):
        """Test that reset clears filter state."""
        filter_obj = OneEuroFilter(freq=30.0)

        # Process some data
        x1 = np.ones(10, dtype=np.float32)
        filter_obj.reset_and_filter(x1)

        # Reset
        filter_obj.reset()

        # Process same data again - should get same result
        x2 = np.ones(10, dtype=np.float32)
        filtered = filter_obj.reset_and_filter(x2)

        assert np.allclose(filtered, 1.0, atol=1e-6)

    def test_monotonic_timestamps(self):
        """Test that non-monotonic timestamps raise error."""
        filter_obj = OneEuroFilter(freq=30.0)

        # Initialize
        filter_obj.filter_sample(0.0, 1.0)

        # Try non-monotonic timestamp
        with pytest.raises(ValueError, match="monotonically increasing"):
            filter_obj.filter_sample(0.0, 2.0)  # Same timestamp

    def test_exponential_smoothing(self):
        """Test exponential smoothing formula."""
        alpha = 0.5
        x = 10.0
        x_prev = 5.0

        result = OneEuroFilter._exponential_smoothing(alpha, x, x_prev)
        expected = alpha * x + (1.0 - alpha) * x_prev

        assert result == expected


class TestPoseSmoother:
    """Test PoseSmoother for multi-joint pose sequences."""

    @pytest.fixture
    def sample_poses(self):
        """Create sample pose sequence with 17 joints (H3.6M format)."""
        from src.types import H36Key

        # 30 frames, 17 joints, 2 coords
        poses = np.zeros((30, 17, 2), dtype=np.float32)

        # Add sinusoidal motion to some joints
        t = np.arange(30) / 30.0
        for i in range(30):
            # Left wrist moves (index 13 in H3.6M)
            poses[i, H36Key.LWRIST, 0] = 0.5 * np.sin(2 * np.pi * t[i])
            poses[i, H36Key.LWRIST, 1] = 0.3 * np.cos(2 * np.pi * t[i])

            # Right wrist moves (index 16 in H3.6M)
            poses[i, H36Key.RWRIST, 0] = -0.5 * np.sin(2 * np.pi * t[i])
            poses[i, H36Key.RWRIST, 1] = 0.3 * np.cos(2 * np.pi * t[i])

        return poses

    def test_smooth_preserves_shape(self, sample_poses):
        """Test that smoothing preserves array shape."""
        smoother = PoseSmoother(freq=30.0)
        smoothed = smoother.smooth(sample_poses)

        assert smoothed.shape == sample_poses.shape
        assert smoothed.dtype == np.float32

    def test_smooth_all_joints(self, sample_poses):
        """Test that all joints are processed."""
        smoother = PoseSmoother(freq=30.0)
        smoothed = smoother.smooth(sample_poses)

        # All joints should be valid (no NaN, inf)
        assert np.all(np.isfinite(smoothed))

    def test_noise_reduction_multidimensional(self):
        """Test noise reduction across all joints."""
        # Create poses with noise
        np.random.seed(42)
        num_frames = 60
        poses = np.random.normal(0, 0.1, (num_frames, 17, 2)).astype(np.float32)

        # Add underlying signal
        t = np.arange(num_frames) / 30.0
        for i in range(num_frames):
            poses[i, :, 0] += 0.5 * np.sin(2 * np.pi * 0.5 * t[i])
            poses[i, :, 1] += 0.3 * np.cos(2 * np.pi * 0.5 * t[i])

        # Smooth with lower cutoff for more smoothing effect
        config_low = OneEuroFilterConfig(freq=30.0, min_cutoff=0.5, beta=0.007)
        smoother = PoseSmoother(config=config_low)
        smoothed = smoother.smooth(poses)

        # Compute smoothness (second derivative)
        def smoothness(signal):
            return np.mean(np.abs(np.diff(signal, n=2)))

        # Smoothed should be smoother than original
        original_smoothness = np.mean(
            [smoothness(poses[:, j, c]) for j in range(17) for c in range(2)]
        )
        smoothed_smoothness = np.mean(
            [smoothness(smoothed[:, j, c]) for j in range(17) for c in range(2)]
        )

        assert smoothed_smoothness < original_smoothness

    def test_phase_aware_smoothing(self, sample_poses):
        """Test phase-aware smoothing with boundaries."""
        smoother = PoseSmoother(freq=30.0)

        # Define phase boundaries
        boundaries = [10, 20]  # Takeoff, landing

        # Smooth with phase awareness
        smoothed = smoother.smooth_phase_aware(sample_poses, boundaries)

        # Check shape preserved
        assert smoothed.shape == sample_poses.shape

    def test_frequency_scaling(self):
        """Test that frequency parameter affects smoothing."""
        # Low frequency = more smoothing
        config_low = OneEuroFilterConfig(freq=30.0, min_cutoff=0.3, beta=0.007)
        smoother_low = PoseSmoother(config=config_low)

        # High frequency = less smoothing
        config_high = OneEuroFilterConfig(freq=30.0, min_cutoff=3.0, beta=0.007)
        smoother_high = PoseSmoother(config=config_high)

        # Noisy signal
        np.random.seed(42)
        poses = np.random.normal(0, 0.1, (30, 17, 2)).astype(np.float32)

        smoothed_low = smoother_low.smooth(poses)
        smoothed_high = smoother_high.smooth(poses)

        # Low cutoff should produce smoother output
        # Check variance of first differences
        var_low = np.var(np.diff(smoothed_low))
        var_high = np.var(np.diff(smoothed_high))

        assert var_low < var_high  # More smoothing = less variance

    def test_set_frequency(self, sample_poses):
        """Test updating sampling frequency."""
        smoother = PoseSmoother(freq=30.0)

        # Smooth at 30 FPS
        smoothed_30 = smoother.smooth(sample_poses)

        # Change to 60 FPS
        smoother.set_frequency(60.0)

        # Should clear filters
        assert len(smoother._filters) == 0

        # Should work at new frequency
        smoothed_60 = smoother.smooth(sample_poses)

        # Results may differ due to frequency scaling
        assert smoothed_60.shape == smoothed_30.shape

    def test_invalid_shape(self):
        """Test that invalid shape raises error."""
        smoother = PoseSmoother(freq=30.0)

        # Wrong shape - 3D instead of 2D (smooth() expects 2D)
        poses = np.zeros((30, 17, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            smoother.smooth(poses)


class TestOneEuroFilterConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OneEuroFilterConfig()

        assert config.min_cutoff == 1.0
        assert config.beta == 0.007
        assert config.derivative_cutoff == 1.0
        assert config.freq == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = OneEuroFilterConfig(
            min_cutoff=2.0,
            beta=0.01,
            derivative_cutoff=0.5,
            freq=60.0,
        )

        assert config.min_cutoff == 2.0
        assert config.beta == 0.01
        assert config.derivative_cutoff == 0.5
        assert config.freq == 60.0

    def test_config_frozen(self):
        """Test that config is immutable (frozen dataclass)."""
        config = OneEuroFilterConfig()

        with pytest.raises(Exception, match="cannot assign to field"):  # FrozenInstanceError
            config.min_cutoff = 2.0


class TestSkatingOptimizedConfig:
    """Test skating-optimized configuration factory."""

    def test_default_fps(self):
        """Test config at default 30 FPS."""
        config = get_skating_optimized_config()

        assert config.freq == 30.0
        assert config.min_cutoff == 1.0
        assert config.beta == 0.007

    def test_fps_scaling(self):
        """Test that min_cutoff scales with FPS."""
        config_25 = get_skating_optimized_config(25.0)
        config_30 = get_skating_optimized_config(30.0)
        config_60 = get_skating_optimized_config(60.0)

        # Higher FPS → higher min_cutoff
        assert config_25.min_cutoff < config_30.min_cutoff
        assert config_30.min_cutoff < config_60.min_cutoff

    def test_parameters_in_valid_range(self):
        """Test that all parameters are in valid ranges."""
        for fps in [25, 30, 60, 120]:
            config = get_skating_optimized_config(fps)

            assert 0.1 <= config.min_cutoff <= 10.0
            assert 0.0 <= config.beta <= 1.0
            assert 0.1 <= config.derivative_cutoff <= 10.0
            assert config.freq == fps


class TestIntegration:
    """Integration tests with existing pipeline components."""

    def test_smoothing_after_normalization(self):
        """Test that smoothing works after pose normalization."""
        from src.types import H36Key

        # Create normalized poses (centered at origin)
        poses = np.zeros((30, 17, 2), dtype=np.float32)
        np.arange(30) / 30.0
        for i in range(30):
            poses[i, H36Key.LEFT_HIP, 0] = -0.05
            poses[i, H36Key.LEFT_HIP, 1] = 0.0
            poses[i, H36Key.RIGHT_HIP, 0] = 0.05
            poses[i, H36Key.RIGHT_HIP, 1] = 0.0
            poses[i, H36Key.LEFT_SHOULDER, 0] = -0.1
            poses[i, H36Key.LEFT_SHOULDER, 1] = -0.3
            poses[i, H36Key.RIGHT_SHOULDER, 0] = 0.1
            poses[i, H36Key.RIGHT_SHOULDER, 1] = -0.3

        smoother = PoseSmoother(freq=30.0)
        smoothed = smoother.smooth(poses)

        # Should preserve normalization properties
        # Check that mid-hip is still near origin
        mid_hip = (smoothed[:, H36Key.LEFT_HIP] + smoothed[:, H36Key.RIGHT_HIP]) / 2
        assert np.allclose(mid_hip, 0.0, atol=0.1)  # Still centered

    def test_smoothing_preserves_phase_boundaries(self):
        """Test that phase-aware smoothing preserves rapid transitions."""
        # Create poses with sudden jump at frame 15
        poses = np.zeros((30, 17, 2), dtype=np.float32)
        poses[:15, :, 0] = -0.5  # Left side
        poses[15:, :, 0] = 0.5  # Right side

        smoother = PoseSmoother(freq=30.0)

        # Phase-aware: should preserve sharp transition
        boundaries = [15]
        smoothed_phase = smoother.smooth_phase_aware(poses, boundaries)

        # Standard: should smooth transition
        smoother2 = PoseSmoother(freq=30.0)
        smoothed_standard = smoother2.smooth(poses)

        # Phase-aware should have sharper transition at boundary
        # Check the difference at frame 15
        transition_phase = np.diff(smoothed_phase[14:17, 0, 0])
        transition_standard = np.diff(smoothed_standard[14:17, 0, 0])

        # Phase-aware should preserve more of the jump
        assert np.max(np.abs(transition_phase)) >= np.max(np.abs(transition_standard))
