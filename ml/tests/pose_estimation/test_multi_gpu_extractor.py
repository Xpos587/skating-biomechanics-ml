# tests/pose_estimation/test_multi_gpu_extractor.py
"""Tests for multi-GPU pose extraction."""

import pytest

from skating_ml.device import MultiGPUConfig
from skating_ml.pose_estimation import MultiGPUPoseExtractor


def test_multi_gpu_extractor_init():
    """Test MultiGPUPoseExtractor initialization."""
    config = MultiGPUConfig()
    extractor = MultiGPUPoseExtractor(config=config)

    assert extractor.config == config
    assert extractor.output_format == "normalized"
    assert extractor.conf_threshold == 0.5
    assert extractor.mode == "balanced"


def test_multi_gpu_extractor_single_gpu_fallback():
    """Test that single GPU falls back to sequential extraction."""
    # Force single GPU mode
    config = MultiGPUConfig(gpu_ids=[0])
    extractor = MultiGPUPoseExtractor(config=config)

    # Should initialize without error
    assert extractor is not None

    # If only one GPU enabled, should use sequential extraction
    if len(config.enabled_gpus) <= 1:
        assert extractor.config.enabled_gpus == config.enabled_gpus


def test_multi_gpu_extractor_custom_params():
    """Test MultiGPUPoseExtractor with custom parameters."""
    config = MultiGPUConfig(memory_reserve_mb=1024)
    extractor = MultiGPUPoseExtractor(
        config=config,
        output_format="pixels",
        conf_threshold=0.3,
        mode="lightweight",
    )

    assert extractor.output_format == "pixels"
    assert extractor.conf_threshold == 0.3
    assert extractor.mode == "lightweight"


@pytest.mark.skipif(len(MultiGPUConfig().enabled_gpus) < 2, reason="Requires at least 2 GPUs")
def test_multi_gpu_extraction_returns_valid_result():
    """Test that multi-GPU extraction produces valid output structure."""
    from pathlib import Path

    video_path = Path("data/videos/test_15s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    config = MultiGPUConfig()
    extractor = MultiGPUPoseExtractor(config=config)

    result = extractor.extract_video_tracked(video_path)

    # Check result structure
    assert result.poses.ndim == 3
    assert result.poses.shape[1] == 17  # H3.6M keypoints
    assert result.poses.shape[2] == 3  # x, y, confidence
    assert len(result.frame_indices) > 0


def test_multi_gpu_extract_chunk_static_method():
    """Test _extract_chunk static method structure."""
    # This is a basic smoke test for the static method
    # Full integration test would require actual video file
    from pathlib import Path

    video_path = Path("data/videos/test_15s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    # Test that the method exists and is callable
    assert callable(MultiGPUPoseExtractor._extract_chunk)


def test_merge_chunks():
    """Test chunk merging logic."""
    from pathlib import Path

    from skating_ml.types import VideoMeta

    # Create mock results
    chunk1_result = {
        "poses": [[[1.0, 1.0, 0.9]] * 17],
        "frame_indices": [0],
        "start_frame": 0,
        "end_frame": 1,
    }
    chunk2_result = {
        "poses": [[[2.0, 2.0, 0.8]] * 17],
        "frame_indices": [1],
        "start_frame": 1,
        "end_frame": 2,
    }

    results = [(0, chunk1_result), (1, chunk2_result)]

    # Create mock metadata (path is required)
    meta = VideoMeta(
        path=Path("dummy.mp4"),
        num_frames=2,
        fps=30.0,
        width=1920,
        height=1080,
    )

    extractor = MultiGPUPoseExtractor()

    # Merge chunks
    merged = extractor._merge_chunks(results, meta, person_click=None)

    # Check merged result
    assert merged.poses.shape == (2, 17, 3)
    assert merged.poses[0, 0, 0] == 1.0
    assert merged.poses[1, 0, 0] == 2.0
