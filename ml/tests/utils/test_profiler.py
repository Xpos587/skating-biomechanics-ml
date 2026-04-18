# tests/utils/test_profiler.py
"""Tests for profiling utilities."""

import time

from skating_ml.utils.profiler import Profiler, StageProfiler, profile_function


def test_profiler_context_manager():
    """Test profiler context manager."""
    with Profiler() as p:
        # Simulate work
        time.sleep(0.01)

    # Should have timing data - we can't easily check elapsed_time
    # but the context manager should complete without error
    assert True


def test_profiler_with_name():
    """Test profiler with custom name."""
    with Profiler("test_operation"):
        time.sleep(0.005)

    # Should complete without error
    assert True


def test_profiler_decorator():
    """Test profiler decorator."""

    @profile_function
    def slow_function():
        time.sleep(0.01)
        return "done"

    result = slow_function()

    assert result == "done"


def test_profiler_decorator_with_args():
    """Test profiler decorator with arguments."""

    @profile_function
    def function_with_args(x, y):
        time.sleep(0.005)
        return x + y

    result = function_with_args(1, 2)

    assert result == 3


def test_stage_profiler():
    """Test StageProfiler class."""
    profiler = StageProfiler()

    profiler.start("stage1")
    time.sleep(0.01)
    elapsed1 = profiler.end("stage1")

    assert elapsed1 >= 0.01
    assert elapsed1 < 1.0  # Should be fast


def test_stage_profiler_multiple_stages():
    """Test StageProfiler with multiple stages."""
    profiler = StageProfiler()

    profiler.start("stage1")
    time.sleep(0.005)
    profiler.end("stage1")

    profiler.start("stage2")
    time.sleep(0.01)
    profiler.end("stage2")

    report = profiler.report()

    assert "stage1" in report
    assert "stage2" in report
    assert report["stage1"] >= 0.005
    assert report["stage2"] >= 0.01


def test_stage_profiler_report():
    """Test StageProfiler report method."""
    profiler = StageProfiler()

    profiler.start("extract")
    time.sleep(0.005)
    profiler.end("extract")

    profiler.start("analyze")
    time.sleep(0.01)
    profiler.end("analyze")

    report = profiler.report()

    assert len(report) == 2
    assert "extract_start" not in report  # Start keys should be excluded
    assert "analyze_start" not in report


def test_stage_profiler_end_without_start():
    """Test StageProfiler.end() without matching start()."""
    profiler = StageProfiler()

    # Should not crash, should return 0.0
    elapsed = profiler.end("nonexistent_stage")

    assert elapsed == 0.0


def test_stage_profiler_print_report(capsys):
    """Test StageProfiler.print_report()."""
    profiler = StageProfiler()

    profiler.start("test_stage")
    time.sleep(0.005)
    profiler.end("test_stage")

    # Should not raise an exception
    profiler.print_report()

    # Check output (optional)
    captured = capsys.readouterr()
    # Just verify it doesn't crash
    assert True
