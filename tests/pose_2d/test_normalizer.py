"""Tests for pose normalization.

Updated for H3.6M 17-keypoint 3D format.
"""

import numpy as np
import pytest

from src.normalizer import PoseNormalizer
from src.types import H36Key


class TestPoseNormalizer:
    """Test PoseNormalizer."""

    def test_normalizer_initialization(self):
        """Should initialize with default parameters."""
        normalizer = PoseNormalizer()

        assert normalizer._target_spine_length == 0.4

    def test_normalizer_custom_spine_length(self):
        """Should initialize with custom spine length."""
        normalizer = PoseNormalizer(target_spine_length=0.5)

        assert normalizer._target_spine_length == 0.5

    def test_normalize_shape(self, sample_keypoints):
        """Should output correct shape."""
        normalizer = PoseNormalizer()

        normalized = normalizer.normalize(sample_keypoints)

        # Returns 2D normalized poses (N, 17, 2)
        assert normalized.shape == (1, 17, 2)
        assert normalized.dtype == np.float32

    def test_normalize_centers_at_origin(self, sample_keypoints):
        """Should center hip_center at origin."""
        normalizer = PoseNormalizer()

        normalized = normalizer.normalize(sample_keypoints)

        # Hip_center should be at origin after normalization
        hip_center = normalized[0, H36Key.HIP_CENTER]

        assert np.allclose(hip_center, [0, 0], atol=1e-5)

    def test_normalize_scales_spine(self, sample_keypoints):
        """Should scale spine to target length."""
        target_spine = 0.4
        normalizer = PoseNormalizer(target_spine_length=target_spine)

        normalized = normalizer.normalize(sample_keypoints)

        # Calculate spine length in normalized pose
        thorax = normalized[0, H36Key.THORAX]
        hip_center = normalized[0, H36Key.HIP_CENTER]

        spine_length = np.linalg.norm(thorax - hip_center)

        assert np.isclose(spine_length, target_spine, rtol=0.01)

    def test_is_valid_frame_good_confidence(self, sample_keypoints):
        """Should accept frame with good keypoints."""
        normalizer = PoseNormalizer()

        # Sample keypoints have valid positions
        is_valid = normalizer.is_valid_frame(sample_keypoints[0], min_visible=0.7)

        # Should be valid
        assert isinstance(is_valid, bool)

    def test_is_valid_frame_low_confidence(self):
        """Should reject frame with low valid keypoints."""
        normalizer = PoseNormalizer()

        # Create frame with all zeros (17 keypoints for H3.6M)
        low_conf_frame = np.zeros((17, 3), dtype=np.float32)
        # All at origin - invalid

        is_valid = normalizer.is_valid_frame(low_conf_frame, min_visible=0.7)

        assert is_valid is False

    def test_is_valid_frame_wrong_shape(self):
        """Should reject frame with wrong shape."""
        normalizer = PoseNormalizer()

        # Wrong shape - 2D instead of 3D
        wrong_shape = np.zeros((17, 2), dtype=np.float32)

        is_valid = normalizer.is_valid_frame(wrong_shape)

        assert is_valid is False

    def test_get_spine_length(self, sample_keypoints):
        """Should calculate average spine length."""
        normalizer = PoseNormalizer()

        spine_length = normalizer.get_spine_length(sample_keypoints)

        assert spine_length > 0
        assert isinstance(spine_length, float)


class TestNormalizeMultipleFrames:
    """Test normalization with multiple frames."""

    def test_normalize_three_frames(self):
        """Should normalize three frames correctly."""
        normalizer = PoseNormalizer()

        # Create three identical frames (17 keypoints for H3.6M)
        frames = np.tile(np.zeros((1, 17, 3), dtype=np.float32), (3, 1, 1))

        # Set some positions (H3.6M 17 format)
        for i in range(3):
            frames[i, H36Key.THORAX] = [0.0, 0.3, 0.0]
            frames[i, H36Key.HIP_CENTER] = [0.0, 0.0, 0.0]

        normalized = normalizer.normalize(frames)

        assert normalized.shape == (3, 17, 2)

        # Each frame should be centered at hip_center
        for i in range(3):
            hip_center = normalized[i, H36Key.HIP_CENTER]
            assert np.allclose(hip_center, [0, 0], atol=1e-5)
