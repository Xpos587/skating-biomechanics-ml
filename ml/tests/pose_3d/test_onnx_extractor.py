# tests/pose_3d/test_onnx_extractor.py
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def onnx_model_path():
    p = Path("data/models/motionagformer-s-ap3d.onnx")
    if not p.exists():
        pytest.skip("ONNX model not exported yet")
    # External data file required by ONNX Runtime (large, may be excluded from CI)
    data_file = p.with_name("motionagformer-s-ap3d.onnx.data")
    if not data_file.exists():
        pytest.skip("ONNX external data file not available")
    return p


def test_onnx_extractor_init(onnx_model_path):
    from skating_ml.pose_3d.onnx_extractor import ONNXPoseExtractor

    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    assert ext.temporal_window == 81


def test_onnx_extractor_single_window(onnx_model_path):
    from skating_ml.pose_3d.onnx_extractor import ONNXPoseExtractor

    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input: (81, 17, 2) normalized 2D poses
    poses_2d = np.random.rand(81, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (81, 17, 3)
    # Z coordinates should be reasonable (not all zeros, not huge)
    assert not np.allclose(result[:, :, 2], 0)
    assert np.nanmax(np.abs(result)) < 10


def test_onnx_extractor_long_sequence(onnx_model_path):
    from skating_ml.pose_3d.onnx_extractor import ONNXPoseExtractor

    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input longer than 81 frames — should be windowed
    poses_2d = np.random.rand(200, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (200, 17, 3)


def test_onnx_extractor_short_sequence(onnx_model_path):
    from skating_ml.pose_3d.onnx_extractor import ONNXPoseExtractor

    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input shorter than 81 frames — should be padded
    poses_2d = np.random.rand(30, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (30, 17, 3)
