"""Pipeline profiling utilities for measuring wall-clock time per stage.

This module provides lightweight profiling tools for the ML pipeline,
specifically designed for tracking execution time of pipeline stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class StageTiming:
    """Timing data for a single pipeline stage.

    Attributes:
        name: Stage identifier (e.g., "pose_extraction", "analysis").
        wall_time_s: Wall-clock time in seconds.
        call_count: Number of times this stage was called (merged if duplicate names).
    """

    name: str
    wall_time_s: float
    call_count: int = 1


class PipelineProfiler:
    """Collects wall-clock timings for pipeline stages.

    This profiler tracks execution time for named stages and can generate
    summary reports. It supports merging multiple calls to the same stage
    by summing times and incrementing call counts.

    Example:
        profiler = PipelineProfiler()
        with profiler:
            profiler.record("pose_extraction", 1.23)
            profiler.record("analysis", 0.45)
        print(profiler.summary_table())
    """

    def __init__(self) -> None:
        """Initialize an empty profiler."""
        self._stages: dict[str, StageTiming] = {}
        self._context_start: float | None = None
        self._total_wall_time_s: float = 0.0

    @property
    def stages(self) -> list[StageTiming]:
        """Get list of all recorded stage timings.

        Returns:
            List of StageTiming objects, sorted by first insertion order.
        """
        return list(self._stages.values())

    @property
    def total_wall_time_s(self) -> float:
        """Get total wall-clock time across all stages.

        Returns:
            Sum of all stage times in seconds.
        """
        return self._total_wall_time_s

    def record(self, name: str, wall_time_s: float) -> None:
        """Record timing for a stage.

        If a stage with the same name exists, times are summed and
        call_count is incremented (merging behavior).

        Args:
            name: Stage identifier.
            wall_time_s: Wall-clock time in seconds (must be non-negative).
        """
        if wall_time_s < 0:
            raise ValueError(f"wall_time_s must be non-negative, got {wall_time_s}")

        if name in self._stages:
            existing = self._stages[name]
            existing.wall_time_s += wall_time_s
            existing.call_count += 1
        else:
            self._stages[name] = StageTiming(name=name, wall_time_s=wall_time_s)

        # Update total if not in context manager mode
        if self._context_start is None:
            self._total_wall_time_s = sum(stage.wall_time_s for stage in self.stages)

    def reset(self) -> None:
        """Clear all recorded timings and reset state."""
        self._stages.clear()
        self._total_wall_time_s = 0.0
        self._context_start = None

    def summary_table(self) -> str:
        """Generate formatted summary table of stage timings.

        Returns:
            Multiline string with columns: Stage, Time (s), %, Calls.
            Includes TOTAL row at the bottom.
        """
        if not self._stages:
            return "No profiling data available"

        lines = []
        lines.append(f"{'Stage':<30} {'Time (s)':>12} {'%':>8} {'Calls':>8}")
        lines.append("-" * 62)

        total = self.total_wall_time_s
        for stage in self.stages:
            pct = (stage.wall_time_s / total * 100) if total > 0 else 0
            lines.append(
                f"{stage.name:<30} {stage.wall_time_s:>12.4f} {pct:>7.1f}% {stage.call_count:>8d}"
            )

        lines.append("-" * 62)
        lines.append(f"{'TOTAL':<30} {total:>12.4f} {100.0:>7.1f}% {'':>8}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export profiler data as JSON-serializable dictionary.

        Returns:
            Dict with 'total_wall_time_s' key and 'stages' list.
            Each stage dict has: name, wall_time_s, call_count.
        """
        return {
            "total_wall_time_s": self.total_wall_time_s,
            "stages": [
                {
                    "name": stage.name,
                    "wall_time_s": stage.wall_time_s,
                    "call_count": stage.call_count,
                }
                for stage in self.stages
            ],
        }

    def __enter__(self) -> PipelineProfiler:
        """Enter context manager, tracking total time.

        Returns:
            Self for use in 'with' statement.
        """
        self._context_start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager, finalizing total time.

        Args:
            *args: Exception info if any (ignored).
        """
        if self._context_start is not None:
            self._total_wall_time_s = time.perf_counter() - self._context_start
            self._context_start = None


def profile_stage(stage_name: str | None = None, profiler_attr: str = "_profiler") -> Callable:
    """Decorator to time a method and record to a PipelineProfiler.

    The profiler is found on the instance at the attribute specified by
    profiler_attr (default: "_profiler"). If no profiler is attached,
    the method runs normally without timing.

    Args:
        stage_name: Name for the stage in reports. If None, uses method name.
        profiler_attr: Attribute name on 'self' where profiler is stored.

    Returns:
        Decorator function.

    Example:
        class MyPipeline:
            def __init__(self):
                self._profiler = PipelineProfiler()

            @profile_stage("extract")
            def extract_poses(self, video):
                # ... extraction logic ...
                return poses
    """

    def decorator(func: Callable) -> Callable:
        # Use method name if stage_name not provided
        name = stage_name if stage_name is not None else func.__name__

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Get profiler from instance
            profiler = getattr(self, profiler_attr, None)

            # If no profiler, run normally
            if not isinstance(profiler, PipelineProfiler):
                return func(self, *args, **kwargs)

            # Time the function call
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            wall_time_s = time.perf_counter() - start

            # Record to profiler
            profiler.record(name, wall_time_s)

            return result

        return wrapper

    return decorator
