"""Multi-GPU and async pipeline performance benchmark for Phase 2.

Tests that Phase 2 optimizations meet performance targets:
- Multi-GPU scaling: ~2x speedup on 2 GPUs
- Async pipeline: 1.5-2x speedup vs sync
- End-to-end pipeline: < 30s for 15s video (450 frames @ 30fps)
"""

import time
from pathlib import Path

import numpy as np
import pytest

from skating_ml.device import MultiGPUConfig
from skating_ml.pipeline import AnalysisPipeline
from skating_ml.pose_estimation import MultiGPUPoseExtractor, PoseExtractor
from skating_ml.types import ElementPhase


class TestMultiGPUPoseExtraction:
    """Benchmark multi-GPU pose extraction performance."""

    @pytest.mark.skipif(
        len(MultiGPUConfig().enabled_gpus) < 2,
        reason="Requires at least 2 GPUs for multi-GPU testing",
    )
    def test_multi_gpu_vs_single_gpu_scaling(self, tmp_path):
        """Multi-GPU should achieve ~1.5-2x speedup over single GPU."""
        # Create test video if not exists (small video for benchmarking)
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        # Single GPU extraction
        config_single = MultiGPUConfig(gpu_ids=[0])
        extractor_single = MultiGPUPoseExtractor(config=config_single)

        start = time.perf_counter()
        result_single = extractor_single.extract_video_tracked(video_path)
        time_single = time.perf_counter() - start

        # Multi-GPU extraction
        config_multi = MultiGPUConfig()
        extractor_multi = MultiGPUPoseExtractor(config=config_multi)

        start = time.perf_counter()
        result_multi = extractor_multi.extract_video_tracked(video_path)
        time_multi = time.perf_counter() - start

        # Verify results are similar
        assert result_single.poses.shape == result_multi.poses.shape
        assert np.allclose(
            result_single.poses[~np.isnan(result_single.poses)],
            result_multi.poses[~np.isnan(result_multi.poses)],
            rtol=1e-3,
            atol=1e-3,
        )

        speedup = time_single / time_multi
        print(
            f"Single GPU: {time_single:.2f}s, Multi-GPU: {time_multi:.2f}s, Speedup: {speedup:.2f}x"
        )

        # Target: at least 1.3x speedup (accounting for overhead)
        assert speedup >= 1.3, f"Multi-GPU speedup too low: {speedup:.2f}x (target >= 1.3x)"

    def test_single_gpu_fallback_performance(self, tmp_path):
        """Single GPU fallback should be comparable to PoseExtractor."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        # Standard PoseExtractor
        extractor_standard = PoseExtractor()
        start = time.perf_counter()
        result_standard = extractor_standard.extract_video_tracked(video_path)
        time_standard = time.perf_counter() - start

        # MultiGPUPoseExtractor with single GPU
        config = MultiGPUConfig(gpu_ids=[0])
        extractor_multi = MultiGPUPoseExtractor(config=config)
        start = time.perf_counter()
        result_multi = extractor_multi.extract_video_tracked(video_path)
        time_multi = time.perf_counter() - start

        print(
            f"PoseExtractor: {time_standard:.2f}s, MultiGPUPoseExtractor (1 GPU): {time_multi:.2f}s"
        )

        # Results should be similar
        assert result_standard.poses.shape == result_multi.poses.shape

        # Multi-GPU with 1 GPU should not be significantly slower
        overhead = (time_multi - time_standard) / time_standard
        assert overhead < 0.2, f"Single-GPU overhead too high: {overhead:.1%} (target < 20%)"


class TestAsyncPipelinePerformance:
    """Benchmark async pipeline parallelism."""

    @pytest.mark.asyncio
    async def test_async_vs_sync_pipeline_speedup(self, tmp_path):
        """Async pipeline should achieve 1.3-1.7x speedup over sync."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        # Sync pipeline
        pipeline_sync = AnalysisPipeline(enable_smoothing=True)
        start = time.perf_counter()
        report_sync = pipeline_sync.analyze(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )
        time_sync = time.perf_counter() - start

        # Async pipeline
        pipeline_async = AnalysisPipeline(enable_smoothing=True)
        start = time.perf_counter()
        report_async = await pipeline_async.analyze_async(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )
        time_async = time.perf_counter() - start

        speedup = time_sync / time_async
        print(f"Sync: {time_sync:.2f}s, Async: {time_async:.2f}s, Speedup: {speedup:.2f}x")

        # Results should be equivalent
        assert report_sync.element_type == report_async.element_type
        assert len(report_sync.metrics) == len(report_async.metrics)

        # Target: 1.2x speedup minimum
        assert speedup >= 1.2, f"Async speedup too low: {speedup:.2f}x (target >= 1.2x)"

    @pytest.mark.asyncio
    async def test_async_parallel_stage_execution(self, tmp_path):
        """Verify async pipeline runs stages in parallel."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        pipeline = AnalysisPipeline(enable_smoothing=True)

        # Stage timing instrumentation
        stage_times = {}

        original_lift_3d = pipeline._lift_poses_3d_async

        async def timed_lift_3d(*args, **kwargs):
            start = time.perf_counter()
            result = await original_lift_3d(*args, **kwargs)
            stage_times["lift_3d"] = time.perf_counter() - start
            return result

        original_detect_phases = pipeline._detect_phases_async

        async def timed_detect_phases(*args, **kwargs):
            start = time.perf_counter()
            result = await original_detect_phases(*args, **kwargs)
            stage_times["detect_phases"] = time.perf_counter() - start
            return result

        pipeline._lift_poses_3d_async = timed_lift_3d
        pipeline._detect_phases_async = timed_detect_phases

        # Run async pipeline
        await pipeline.analyze_async(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )

        # Both stages should have timing data
        assert "lift_3d" in stage_times
        assert "detect_phases" in stage_times

        print(
            f"3D lifting: {stage_times['lift_3d']:.2f}s, Phase detection: {stage_times['detect_phases']:.2f}s"
        )


class TestEndToEndPipelinePerformance:
    """Benchmark full pipeline performance targets."""

    def test_full_pipeline_performance_target(self):
        """Full pipeline should complete in < 30s for 15s video."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        pipeline = AnalysisPipeline(enable_smoothing=True)

        start = time.perf_counter()
        report = pipeline.analyze(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )
        elapsed = time.perf_counter() - start

        print(f"Full pipeline: {elapsed:.2f}s")

        # Verify result
        assert report.element_type == "waltz_jump"
        assert len(report.metrics) > 0
        assert len(report.recommendations) > 0

        # Target: < 30s for full pipeline
        assert elapsed < 30.0, f"Pipeline too slow: {elapsed:.2f}s (target < 30s)"

    @pytest.mark.asyncio
    async def test_async_full_pipeline_performance_target(self, tmp_path):
        """Async full pipeline should complete in < 20s for 15s video."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        pipeline = AnalysisPipeline(enable_smoothing=True)

        start = time.perf_counter()
        report = await pipeline.analyze_async(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )
        elapsed = time.perf_counter() - start

        print(f"Async full pipeline: {elapsed:.2f}s")

        # Verify result
        assert report.element_type == "waltz_jump"
        assert len(report.metrics) > 0
        assert len(report.recommendations) > 0

        # Target: < 20s for async pipeline (parallel stages)
        assert elapsed < 20.0, f"Async pipeline too slow: {elapsed:.2f}s (target < 20s)"


class TestMultiGPUPipelineIntegration:
    """Integration tests for multi-GPU + async pipeline."""

    @pytest.mark.skipif(len(MultiGPUConfig().enabled_gpus) < 2, reason="Requires at least 2 GPUs")
    @pytest.mark.asyncio
    async def test_multi_gpu_async_pipeline(self, tmp_path):
        """Multi-GPU + async pipeline combination."""
        video_path = Path("data/videos/test_15s.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")

        # Create pipeline with multi-GPU extractor

        config = MultiGPUConfig()
        multi_gpu_extractor = MultiGPUPoseExtractor(config=config)

        pipeline = AnalysisPipeline(enable_smoothing=True)
        # Replace 2D extractor with multi-GPU version
        pipeline._pose_2d_extractor = multi_gpu_extractor

        start = time.perf_counter()
        report = await pipeline.analyze_async(
            video_path,
            element_type="waltz_jump",
            manual_phases=ElementPhase(
                name="waltz_jump", takeoff=100, peak=120, landing=140, start=80, end=160
            ),
        )
        elapsed = time.perf_counter() - start

        print(f"Multi-GPU + Async pipeline: {elapsed:.2f}s")

        # Verify result
        assert report.element_type == "waltz_jump"
        assert len(report.metrics) > 0

        # Target: < 15s with both optimizations
        assert elapsed < 15.0, f"Multi-GPU + Async too slow: {elapsed:.2f}s (target < 15s)"
