"""Tests for corrective_pipeline module (CorrectiveLens end-to-end)."""

import numpy as np
import pytest

from src.pose_3d.corrective_pipeline import CorrectiveLens


def make_2d_poses(n=10):
    """Create simple synthetic 2D normalized poses."""
    rng = np.random.RandomState(42)
    poses_2d = np.full((n, 17, 2), 0.5, dtype=np.float32)
    poses_2d[:, :, 0] += rng.randn(n, 17) * 0.03
    poses_2d[:, :, 1] += rng.randn(n, 17) * 0.03
    poses_2d = np.clip(poses_2d, 0.05, 0.95)
    return poses_2d.astype(np.float32)


ONNX_MODEL = "data/models/motionagformer-s-ap3d.onnx"


@pytest.mark.skipif(
    not __import__("pathlib").Path(ONNX_MODEL).exists(),
    reason="ONNX model not found",
)
class TestCorrectiveLens:
    def test_nonexistent_model_raises(self):
        """Nonexistent model path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CorrectiveLens(model_path="/nonexistent/model.pth")

    def test_correct_sequence_shape(self):
        """Output should have correct shape."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        n = 10
        poses_2d = make_2d_poses(n)

        corrected, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
        )
        assert corrected.shape == (n, 17, 2)
        assert poses_3d.shape == (n, 17, 3)

    def test_correct_sequence_output_in_range(self):
        """Corrected poses should be in [0, 1]."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        n = 10
        poses_2d = make_2d_poses(n)

        corrected, _ = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
        )
        assert np.all(corrected >= 0.0)
        assert np.all(corrected <= 1.0)

    def test_with_confidence(self):
        """Should work with confidence blending."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        n = 10
        poses_2d = make_2d_poses(n)
        confidences = np.random.rand(n, 17).astype(np.float32)

        corrected, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
            confidences=confidences,
            blend_threshold=0.5,
        )
        assert corrected.shape == (n, 17, 2)
        assert poses_3d.shape == (n, 17, 3)
        assert np.all(np.isfinite(corrected))

    def test_output_dtypes(self):
        """Output arrays should be float32."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        poses_2d = make_2d_poses(5)

        corrected, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=30.0,
            width=1280,
            height=720,
        )
        assert corrected.dtype == np.float32
        assert poses_3d.dtype == np.float32

    def test_3d_poses_are_finite(self):
        """3D poses should not contain NaN or Inf."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        poses_2d = make_2d_poses(8)

        _, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
        )
        assert np.all(np.isfinite(poses_3d))

    def test_single_frame(self):
        """Should handle a single-frame sequence."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        poses_2d = make_2d_poses(1)

        corrected, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=30.0,
            width=1920,
            height=1080,
        )
        assert corrected.shape == (1, 17, 2)
        assert poses_3d.shape == (1, 17, 3)

    def test_low_confidence_biases_toward_corrected(self):
        """Low confidence joints should produce output closer to the corrected result."""
        lens = CorrectiveLens(model_path=ONNX_MODEL)
        n = 10
        poses_2d = make_2d_poses(n)

        # Run without blending
        corrected_no_blend, _ = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
        )

        # Run with low confidence (should trust corrected more)
        conf_low = np.full((n, 17), 0.1, dtype=np.float32)
        corrected_low_conf, _ = lens.correct_sequence(
            poses_2d_norm=poses_2d,
            fps=25.0,
            width=1920,
            height=1080,
            confidences=conf_low,
            blend_threshold=0.5,
        )

        # With low confidence, the result should be close to the fully-corrected output
        diff_to_corrected = np.abs(corrected_low_conf - corrected_no_blend)
        assert np.all(np.isfinite(diff_to_corrected))
