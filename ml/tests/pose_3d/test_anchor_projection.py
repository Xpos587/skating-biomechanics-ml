"""Tests for anchor_projection module (anchor-based 2D projection, confidence blending)."""

import numpy as np

from skating_ml.pose_3d.anchor_projection import anchor_project, blend_by_confidence


def make_test_data(n_frames=10, width=1920, height=1080):
    """Create synthetic 2D and 3D pose data for testing."""
    rng = np.random.RandomState(42)

    # 2D normalized poses: person standing at center
    poses_2d = np.full((n_frames, 17, 2), 0.5, dtype=np.float32)
    for i in range(n_frames):
        poses_2d[i, :, 0] += rng.randn(17) * 0.02  # x
        poses_2d[i, :, 1] += rng.randn(17) * 0.02  # y
    poses_2d = np.clip(poses_2d, 0.0, 1.0)

    # 3D root-relative poses: person in meters (y-up convention, negative y = up)
    poses_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i in range(n_frames):
        poses_3d[i, 0] = [0, 0, 0]  # hip center at origin
        poses_3d[i, 1] = [0.15, 0.1, 0]  # RHIP
        poses_3d[i, 4] = [-0.15, 0.1, 0]  # LHIP
        poses_3d[i, 7] = [0, -0.15, 0]  # SPINE
        poses_3d[i, 8] = [0, -0.35, 0]  # THORAX
        poses_3d[i, 9] = [0, -0.45, 0]  # NECK
        poses_3d[i, 10] = [0, -0.6, 0]  # HEAD
        poses_3d[i, 11] = [-0.2, -0.35, 0]  # LSHOULDER
        poses_3d[i, 14] = [0.2, -0.35, 0]  # RSHOULDER
        poses_3d[i, 12] = [-0.55, -0.35, 0]  # LELBOW
        poses_3d[i, 15] = [0.55, -0.35, 0]  # RELBOW
        poses_3d[i, 13] = [-0.85, -0.35, 0]  # LWRIST
        poses_3d[i, 16] = [0.85, -0.35, 0]  # RWRIST
        poses_3d[i, 2] = [0.15, 0.45, 0]  # RKNEE
        poses_3d[i, 5] = [-0.15, 0.45, 0]  # LKNEE
        poses_3d[i, 3] = [0.15, 0.9, 0]  # RFOOT
        poses_3d[i, 6] = [-0.15, 0.9, 0]  # LFOOT

    return poses_3d, poses_2d, width, height


class TestAnchorProject:
    def test_output_shape(self):
        poses_3d, poses_2d, w, h = make_test_data()
        result = anchor_project(poses_3d, poses_2d, w, h)
        assert result.shape == (10, 17, 2)
        assert result.dtype == np.float32

    def test_output_in_range(self):
        poses_3d, poses_2d, w, h = make_test_data()
        result = anchor_project(poses_3d, poses_2d, w, h)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_hip_anchored(self):
        """Projected hip should be close to the original 2D hip."""
        poses_3d, poses_2d, w, h = make_test_data(n_frames=5)
        result = anchor_project(poses_3d, poses_2d, w, h)
        # Original 2D hip center (midpoint of LHIP=4 and RHIP=1)
        for i in range(5):
            orig_hip = (poses_2d[i, 1, :2] + poses_2d[i, 4, :2]) / 2
            proj_hip = (result[i, 1, :2] + result[i, 4, :2]) / 2
            assert np.linalg.norm(proj_hip - orig_hip) < 0.1

    def test_output_is_finite(self):
        poses_3d, poses_2d, w, h = make_test_data()
        result = anchor_project(poses_3d, poses_2d, w, h)
        assert np.all(np.isfinite(result))

    def test_single_frame(self):
        """Should work with a single frame."""
        poses_3d, poses_2d, w, h = make_test_data(n_frames=1)
        result = anchor_project(poses_3d, poses_2d, w, h)
        assert result.shape == (1, 17, 2)

    def test_different_resolutions(self):
        """Should work with different width/height."""
        poses_3d, poses_2d, _, _ = make_test_data(n_frames=3)
        for w, h in [(640, 480), (1280, 720), (3840, 2160)]:
            result = anchor_project(poses_3d, poses_2d, w, h)
            assert result.shape == (3, 17, 2)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)


class TestBlendByConfidence:
    def test_high_confidence_trusts_raw(self):
        """When confidence is high, output should be close to raw."""
        rng = np.random.RandomState(123)
        raw = rng.rand(5, 17, 2).astype(np.float32)
        corrected = rng.rand(5, 17, 2).astype(np.float32)
        conf = np.full((5, 17), 0.95, dtype=np.float32)  # very high confidence
        blended = blend_by_confidence(raw, corrected, conf, threshold=0.5)
        # With high confidence, weight_3d should be ~0, so blended ≈ raw
        np.testing.assert_allclose(blended, raw, atol=0.05)

    def test_low_confidence_trusts_corrected(self):
        """When confidence is low, output should be close to corrected."""
        rng = np.random.RandomState(456)
        raw = rng.rand(5, 17, 2).astype(np.float32)
        corrected = rng.rand(5, 17, 2).astype(np.float32)
        conf = np.full((5, 17), 0.05, dtype=np.float32)  # very low confidence
        blended = blend_by_confidence(raw, corrected, conf, threshold=0.5)
        # With low confidence, weight_3d should be ~1, so blended ≈ corrected
        np.testing.assert_allclose(blended, corrected, atol=0.05)

    def test_shape_preserved(self):
        rng = np.random.RandomState(789)
        raw = rng.rand(5, 17, 2).astype(np.float32)
        corrected = rng.rand(5, 17, 2).astype(np.float32)
        conf = rng.rand(5, 17).astype(np.float32)
        blended = blend_by_confidence(raw, corrected, conf, threshold=0.5)
        assert blended.shape == raw.shape

    def test_mid_confidence_is_interpolation(self):
        """At exactly the threshold, blending should produce a weighted mix."""
        rng = np.random.RandomState(101)
        raw = rng.rand(5, 17, 2).astype(np.float32)
        corrected = rng.rand(5, 17, 2).astype(np.float32)
        conf = np.full((5, 17), 0.5, dtype=np.float32)  # at threshold
        blended = blend_by_confidence(raw, corrected, conf, threshold=0.5)
        # weight_3d = 1 - clip((0.5 - 0.5 + 0.2) / 0.4, 0, 1) = 1 - clip(0.5, 0, 1) = 0.5
        expected = 0.5 * raw + 0.5 * corrected
        np.testing.assert_allclose(blended, expected.astype(np.float32), atol=0.01)

    def test_output_dtype_float32(self):
        raw = np.random.rand(3, 17, 2).astype(np.float32)
        corrected = np.random.rand(3, 17, 2).astype(np.float32)
        conf = np.random.rand(3, 17).astype(np.float32)
        blended = blend_by_confidence(raw, corrected, conf, threshold=0.5)
        assert blended.dtype == np.float32
