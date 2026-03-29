"""Tests for YOLO-Pose pose extractor."""

import numpy as np
import pytest

from src.pose_extractor import PoseExtractor
from src.types import H36Key


@pytest.mark.slow
class TestPoseExtractor:
    """Test PoseExtractor with YOLO-Pose model."""

    def test_extractor_initialization(self):
        """Should initialize with default parameters."""
        extractor = PoseExtractor()

        assert extractor._model_size == "s"
        assert extractor._min_confidence == 0.5
        assert extractor._model is None  # Lazy load

    def test_model_lazy_load(self):
        """Should load model on first access."""
        extractor = PoseExtractor()

        assert extractor._model is None
        _ = extractor.model  # Access property
        assert extractor._model is not None

    def test_detect_empty_frame(self):
        """Should return None for empty frame."""
        extractor = PoseExtractor()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        kp = extractor.extract_frame(empty_frame)

        assert kp is None

    def test_extract_returns_correct_shape(self, sample_frame):
        """May return keypoints with correct shape if person detected."""
        extractor = PoseExtractor()

        kp = extractor.extract_frame(sample_frame)

        # May return None if no person detected in sample frame
        if kp is not None:
            assert kp.shape == (17, 3)
            assert kp.dtype == np.float32


@pytest.mark.slow
class TestPoseExtractorKeypoints:
    """Test extracted keypoint structure."""

    def test_keypoint_indices(self):
        """Should have exactly 17 keypoints (H3.6M format)."""
        # This tests the constant definition
        assert len(list(BKey)) == 17

    def test_keypoint_names(self):
        """Should have expected keypoint names (H3.6M format)."""
        # H3.6M 17kp indices
        assert H36Key.HIP_CENTER == 0
        assert H36Key.HEAD == 10  # NOSE maps to HEAD in backward compat
        assert H36Key.LSHOULDER == 11
        assert H36Key.RSHOULDER == 14
        assert H36Key.LHIP == 4
        assert H36Key.RHIP == 1
