"""Profiling utilities for performance analysis.

Provides context managers and decorators for timing code execution.
Useful for identifying bottlenecks in the pipeline.
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def Profiler(name: str = "operation"):
    """Context manager for profiling code blocks.

    Usage:
        with Profiler("pose_extraction"):
            poses = extractor.extract_video(video)

    Args:
        name: Name of the operation being profiled.

    Yields:
        None
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name}: {elapsed:.4f}s")


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time.

    Usage:
        @profile_function
        def slow_function():
            ...

    Args:
        func: Function to profile.

    Returns:
        Wrapped function with timing.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper


class StageProfiler:
    """Profile pipeline stages separately.

    Tracks timing for each stage and generates a summary report.

    Usage:
        profiler = StageProfiler()
        profiler.start("pose_extraction")
        # ... do work ...
        profiler.end("pose_extraction")
        profiler.print_report()
    """

    def __init__(self) -> None:
        """Initialize stage profiler."""
        self.timings: dict[str, float] = {}

    def start(self, stage: str) -> None:
        """Start timing a stage.

        Args:
            stage: Stage name identifier.
        """
        self.timings[f"{stage}_start"] = time.perf_counter()

    def end(self, stage: str) -> float:
        """End timing a stage and return elapsed time.

        Args:
            stage: Stage name identifier (must match start call).

        Returns:
            Elapsed time in seconds.
        """
        start_key = f"{stage}_start"
        if start_key not in self.timings:
            logger.warning(f"Stage {stage} was not started")
            return 0.0

        elapsed = time.perf_counter() - self.timings[start_key]
        self.timings[stage] = elapsed
        return elapsed

    def report(self) -> dict[str, float]:
        """Get timing report for all stages.

        Returns:
            Dictionary mapping stage names to elapsed times.
        """
        return {k: v for k, v in self.timings.items() if not k.endswith("_start")}

    def print_report(self) -> None:
        """Print timing report to logger."""
        report = self.report()
        if not report:
            logger.info("No timing data available")
            return

        total = sum(report.values())
        logger.info("=== Pipeline Timings ===")
        for stage, timing in report.items():
            pct = (timing / total * 100) if total > 0 else 0
            logger.info(f"{stage}: {timing:.4f}s ({pct:.1f}%)")
        logger.info(f"Total: {total:.4f}s")
