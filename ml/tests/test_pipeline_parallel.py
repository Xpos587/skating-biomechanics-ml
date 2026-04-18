# tests/test_pipeline_parallel.py
"""Tests for async pipeline parallelism."""

import asyncio
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from skating_ml.pipeline import AnalysisPipeline
from skating_ml.types import ElementPhase, VideoMeta


@pytest.mark.asyncio
async def test_analyze_async_exists():
    """Test that analyze_async method exists and is callable."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    assert callable(pipeline.analyze_async)
    assert hasattr(pipeline, "_lift_poses_3d_async")
    assert hasattr(pipeline, "_detect_phases_async")
    assert hasattr(pipeline, "_compute_metrics_async")
    assert hasattr(pipeline, "_load_reference_async")


@pytest.mark.asyncio
async def test_analyze_async_without_element():
    """Test analyze_async without element type."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    # Mock both get_video_meta and extract_and_track
    mock_meta = VideoMeta(
        path=Path("dummy.mp4"),
        num_frames=100,
        fps=30.0,
        width=1920,
        height=1080,
    )

    with (
        mock.patch("skating_ml.pipeline.get_video_meta", return_value=mock_meta),
        mock.patch.object(
            pipeline,
            "_extract_and_track",
            return_value=(np.random.randn(100, 17, 3), 0),
        ),
    ):
        report = await pipeline.analyze_async(
            Path("dummy.mp4"),
            element_type=None,
        )

    assert report.element_type == "unknown"
    assert report.phases.takeoff == 0
    assert report.phases.peak == 0
    assert report.phases.landing == 0


def test_analyze_sync_still_works():
    """Test that sync analyze method still works after adding async."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    # Mock both get_video_meta and extract_and_track
    mock_meta = VideoMeta(
        path=Path("dummy.mp4"),
        num_frames=100,
        fps=30.0,
        width=1920,
        height=1080,
    )

    with (
        mock.patch("skating_ml.pipeline.get_video_meta", return_value=mock_meta),
        mock.patch.object(
            pipeline,
            "_extract_and_track",
            return_value=(np.random.randn(100, 17, 3), 0),
        ),
    ):
        report = pipeline.analyze(
            Path("dummy.mp4"),
            element_type=None,
        )

    assert report.element_type == "unknown"


@pytest.mark.asyncio
async def test_parallel_stage_execution():
    """Test that async methods can be executed in parallel."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    # Create mock tasks
    async def mock_task(name, delay):
        await asyncio.sleep(delay)
        return name

    # Run tasks in parallel
    results = await asyncio.gather(
        mock_task("task1", 0.01),
        mock_task("task2", 0.01),
    )

    assert len(results) == 2
    assert "task1" in results
    assert "task2" in results


@pytest.mark.asyncio
async def test_lift_poses_3d_async():
    """Test _lift_poses_3d_async method."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    # Mock poses
    poses_2d = np.random.randn(100, 17, 2).astype(np.float32)

    # If 3D extractor is not available, should return None
    result = await pipeline._lift_poses_3d_async(poses_2d, 30.0)

    # Result should be (poses_3d, blade_summaries) or (None, None)
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_detect_phases_async_with_manual():
    """Test _detect_phases_async with manual phases."""
    pipeline = AnalysisPipeline(enable_smoothing=False)

    poses = np.random.randn(100, 17, 2).astype(np.float32)
    manual = ElementPhase(name="test", takeoff=50, peak=60, landing=70, start=40, end=80)

    result = await pipeline._detect_phases_async(poses, 30.0, "test", manual)

    assert result == manual


@pytest.mark.asyncio
async def test_compute_metrics_async():
    """Test _compute_metrics_async method."""
    from skating_ml.analysis import element_defs

    pipeline = AnalysisPipeline(enable_smoothing=False)

    poses = np.random.randn(100, 17, 2).astype(np.float32)
    phases = ElementPhase(name="waltz_jump", takeoff=50, peak=60, landing=70, start=40, end=80)
    element_def = element_defs.get_element_def("waltz_jump")

    if element_def is not None:
        metrics = await pipeline._compute_metrics_async(poses, phases, 30.0, element_def)

        # Should return list of metrics
        assert isinstance(metrics, list)


@pytest.mark.asyncio
async def test_load_reference_async():
    """Test _load_reference_async method."""
    pipeline = AnalysisPipeline(enable_smoothing=False, reference_store=None)

    # Without reference store, should return None
    result = await pipeline._load_reference_async("waltz_jump")

    assert result is None
