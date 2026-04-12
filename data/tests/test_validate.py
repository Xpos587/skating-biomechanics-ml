"""Tests for skeleton validation utilities."""

import numpy as np
import pytest

from data_tools.validate import ValidationError, validate_skeleton


def test_valid_skeleton_passes():
    """A valid skeleton should pass all validation checks."""
    poses = np.random.randn(100, 17, 2)
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) == 0


def test_wrong_keypoint_count_fails():
    """Wrong number of keypoints should produce an error."""
    poses = np.random.randn(100, 15, 2)
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "keypoints" for e in errors)


def test_nan_detection():
    """NaN values should be detected."""
    poses = np.random.randn(100, 17, 2)
    poses[50, 8, 0] = np.nan  # Insert NaN
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "nan" for e in errors)


def test_inf_detection():
    """Inf values should be detected."""
    poses = np.random.randn(100, 17, 2)
    poses[50, 8, 0] = np.inf  # Insert Inf
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "inf" for e in errors)


def test_all_zeros_detection():
    """All-zero skeleton should be detected."""
    poses = np.zeros((100, 17, 2))
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "zeros" for e in errors)


def test_min_frames():
    """Too few frames should be detected."""
    poses = np.random.randn(5, 17, 2)
    errors = validate_skeleton(poses, sample_id="test_sample", min_frames=10)
    assert len(errors) > 0
    assert any(e.field == "frames" for e in errors)


def test_4d_squeeze():
    """4D array with trailing dimension should be auto-squeezed."""
    poses = np.random.randn(100, 17, 2, 1)
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) == 0


def test_wrong_ndim():
    """Wrong number of dimensions should be caught."""
    poses = np.random.randn(100, 17)  # 2D instead of 3D
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "ndim" for e in errors)


def test_too_few_channels():
    """Too few channels should be detected."""
    poses = np.random.randn(100, 17, 1)
    errors = validate_skeleton(poses, sample_id="test_sample")
    assert len(errors) > 0
    assert any(e.field == "channels" for e in errors)


def test_coordinate_bounds():
    """Coordinates exceeding bounds should be detected."""
    poses = np.random.randn(100, 17, 2) * 100  # Scale up to exceed bounds
    errors = validate_skeleton(poses, sample_id="test_sample", max_coord=10.0)
    assert len(errors) > 0
    assert any(e.field == "bounds" for e in errors)


def test_multiple_errors():
    """Multiple validation errors should all be reported."""
    poses = np.random.randn(3, 15, 2)  # Wrong frame count, wrong keypoint count
    poses[0, 0, 0] = np.nan  # Add NaN
    errors = validate_skeleton(poses, sample_id="test_sample", min_frames=10)
    assert len(errors) >= 3
    error_fields = {e.field for e in errors}
    assert "frames" in error_fields
    assert "keypoints" in error_fields
    assert "nan" in error_fields


def test_validation_error_repr():
    """ValidationError should have a useful string representation."""
    error = ValidationError(
        sample_id="test_123",
        field="keypoints",
        message="Wrong number of keypoints: 15, expected 17",
    )
    repr_str = repr(error)
    assert "test_123" in repr_str
    assert "keypoints" in repr_str
    assert "Wrong number of keypoints" in repr_str
