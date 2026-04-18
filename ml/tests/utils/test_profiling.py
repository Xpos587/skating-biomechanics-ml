"""Tests for pipeline profiling utilities."""

import json
import time

from skating_ml.utils.profiling import PipelineProfiler, profile_stage


class FakePipeline:
    """Fake pipeline class for testing @profile_stage decorator."""

    def __init__(self, profiler: PipelineProfiler | None = None) -> None:
        """Initialize fake pipeline.

        Args:
            profiler: Optional profiler to attach. If None, methods run without profiling.
        """
        self._profiler = profiler

    @profile_stage("extract")
    def extract_poses(self) -> str:
        """Simulate pose extraction stage."""
        time.sleep(0.01)
        return "poses"

    @profile_stage("analyze")
    def analyze_poses(self) -> str:
        """Simulate analysis stage."""
        time.sleep(0.015)
        return "analysis"

    @profile_stage  # Uses method name
    def normalize(self) -> str:
        """Simulate normalization stage."""
        time.sleep(0.005)
        return "normalized"


def test_profiler_records_stages() -> None:
    """Test that profiler records multiple stages with wall_time > 0."""
    profiler = PipelineProfiler()
    pipeline = FakePipeline(profiler)

    # Run profiled methods
    pipeline.extract_poses()
    pipeline.analyze_poses()

    # Check that 2 stages were recorded
    stages = profiler.stages
    assert len(stages) == 2

    # Check that each stage has positive wall time
    for stage in stages:
        assert stage.wall_time_s > 0
        assert stage.call_count == 1

    # Check stage names
    stage_names = {s.name for s in stages}
    assert stage_names == {"extract", "analyze"}


def test_profiler_total_time() -> None:
    """Test that total_wall_time_s >= sum of individual stage times."""
    profiler = PipelineProfiler()
    pipeline = FakePipeline(profiler)

    # Run profiled methods
    pipeline.extract_poses()
    pipeline.analyze_poses()
    pipeline.normalize()

    # Total should be at least sum of stages
    stage_sum = sum(s.wall_time_s for s in profiler.stages)
    assert profiler.total_wall_time_s >= stage_sum

    # Total should be positive
    assert profiler.total_wall_time_s > 0


def test_profiler_summary_table() -> None:
    """Test that summary_table() contains stage names and TOTAL row."""
    profiler = PipelineProfiler()
    pipeline = FakePipeline(profiler)

    # Run profiled methods
    pipeline.extract_poses()
    pipeline.analyze_poses()

    # Get summary table
    table = profiler.summary_table()

    # Check that stage names are present
    assert "extract" in table
    assert "analyze" in table

    # Check that TOTAL row is present
    assert "TOTAL" in table

    # Check table structure (headers)
    assert "Stage" in table
    assert "Time (s)" in table
    assert "%" in table
    assert "Calls" in table


def test_profiler_to_dict() -> None:
    """Test that to_dict() returns JSON-serializable data with stage names."""
    profiler = PipelineProfiler()
    pipeline = FakePipeline(profiler)

    # Run profiled methods
    pipeline.extract_poses()
    pipeline.analyze_poses()

    # Export to dict
    data = profiler.to_dict()

    # Check structure
    assert "total_wall_time_s" in data
    assert "stages" in data
    assert isinstance(data["stages"], list)

    # Check that we can JSON serialize
    json_str = json.dumps(data)
    assert len(json_str) > 0

    # Check stage names in dict
    stage_names = {s["name"] for s in data["stages"]}
    assert "extract" in stage_names
    assert "analyze" in stage_names

    # Check each stage has required fields
    for stage in data["stages"]:
        assert "name" in stage
        assert "wall_time_s" in stage
        assert "call_count" in stage


def test_profiler_reset() -> None:
    """Test that reset() clears all recorded stages."""
    profiler = PipelineProfiler()
    pipeline = FakePipeline(profiler)

    # Run profiled methods
    pipeline.extract_poses()
    pipeline.analyze_poses()

    # Verify data was recorded
    assert len(profiler.stages) == 2
    assert profiler.total_wall_time_s > 0

    # Reset
    profiler.reset()

    # Verify cleared
    assert len(profiler.stages) == 0
    assert profiler.total_wall_time_s == 0


def test_profiler_context_manager() -> None:
    """Test that profiler works as a context manager."""
    profiler = PipelineProfiler()

    with profiler:
        # Simulate work
        time.sleep(0.02)
        profiler.record("manual_stage", 0.01)

    # Total time should be tracked from context manager
    assert profiler.total_wall_time_s >= 0.02
    assert profiler.total_wall_time_s < 1.0  # Should be fast

    # Manual stage should also be recorded
    assert len(profiler.stages) >= 1
    assert any(s.name == "manual_stage" for s in profiler.stages)


def test_profiler_merge_duplicate_stages() -> None:
    """Test that recording same stage name merges timing and increments call_count."""
    profiler = PipelineProfiler()

    # Record same stage twice
    profiler.record("test_stage", 0.1)
    profiler.record("test_stage", 0.05)

    stages = profiler.stages
    assert len(stages) == 1

    stage = stages[0]
    assert stage.name == "test_stage"
    assert abs(stage.wall_time_s - 0.15) < 1e-6  # Summed (with float tolerance)
    assert stage.call_count == 2  # Incremented


def test_profile_stage_without_profiler() -> None:
    """Test that @profile_stage works when no profiler is attached."""
    pipeline = FakePipeline(profiler=None)

    # Should not raise an error
    result = pipeline.extract_poses()
    assert result == "poses"

    result = pipeline.analyze_poses()
    assert result == "analysis"


def test_profiler_negative_time_error() -> None:
    """Test that recording negative time raises ValueError."""
    profiler = PipelineProfiler()

    try:
        profiler.record("test", -1.0)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "non-negative" in str(e)


def test_profiler_empty_summary() -> None:
    """Test summary_table() when no data recorded."""
    profiler = PipelineProfiler()
    table = profiler.summary_table()
    assert "No profiling data available" in table
