# Pipeline Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Инструментировать `AnalysisPipeline.analyze()` и `worker.process_video_task()` с cProfile, получить таблицу wall-time по стадиям, найти настоящий bottleneck.

**Architecture:** Декоратор `@profile_stage` оборачивает каждую стадию pipeline. `PipelineProfiler` собирает timing-данные и выводит таблицу. CLI-скрипт `profile_pipeline.py` запускает полный profiling на реальном видео.

**Tech Stack:** Python `cProfile` + `pstats`, `time.perf_counter` для wall-clock, JSON-экспорт

---

### Task 1: Create `profile_stage` decorator

**Files:**

- Create: `ml/skating_ml/utils/profiling.py`
- Test: `ml/tests/utils/test_profiling.py`

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/utils/test_profiling.py
"""Tests for pipeline profiling utilities."""

import time

import numpy as np
import pytest

from skating_ml.utils.profiling import PipelineProfiler, profile_stage


class FakePipeline:
    """Fake pipeline with profiled stages."""

    def __init__(self, profiler: PipelineProfiler) -> None:
        self.profiler = profiler

    @profile_stage("stage_1")
    def fast_stage(self, x: int) -> int:
        return x * 2

    @profile_stage("stage_2")
    def slow_stage(self, n: int) -> float:
        total = 0.0
        for _ in range(n):
            total += np.sin(1.0)
        return total


def test_profiler_records_stages():
    """PipelineProfiler records stage names and wall times."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    pipe.fast_stage(10)
    pipe.slow_stage(1000)

    stages = profiler.stages
    assert len(stages) == 2
    assert stages[0].name == "stage_1"
    assert stages[0].wall_time_s > 0
    assert stages[1].name == "stage_2"
    assert stages[1].wall_time_s > 0


def test_profiler_total_time():
    """Total time >= sum of stage times."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    pipe.fast_stage(10)
    pipe.slow_stage(1000)

    assert profiler.total_wall_time_s >= sum(s.wall_time_s for s in profiler.stages)


def test_profiler_summary_table():
    """Summary table is a string with stage names."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    pipe.fast_stage(10)
    pipe.slow_stage(1000)

    table = profiler.summary_table()
    assert "stage_1" in table
    assert "stage_2" in table
    assert "TOTAL" in table


def test_profiler_to_dict():
    """to_dict returns JSON-serializable data."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    pipe.fast_stage(10)

    import json

    d = profiler.to_dict()
    serialized = json.dumps(d)
    assert "stages" in serialized
    assert "stage_1" in serialized


def test_profiler_reset():
    """reset() clears all recorded stages."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    pipe.fast_stage(10)
    assert len(profiler.stages) == 1

    profiler.reset()
    assert len(profiler.stages) == 0


def test_profiler_context_manager():
    """PipelineProfiler can be used as context manager."""
    profiler = PipelineProfiler()
    pipe = FakePipeline(profiler)

    with profiler:
        pipe.fast_stage(10)
        pipe.slow_stage(100)

    assert len(profiler.stages) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && .venv/bin/python -m pytest tests/utils/test_profiling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skating_ml.utils.profiling'`

- [ ] **Step 3: Write minimal implementation**

```python
# ml/skating_ml/utils/profiling.py
"""Pipeline profiling utilities.

Lightweight timing decorator for measuring wall-clock time per pipeline stage.
No external dependencies — uses time.perf_counter only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


@dataclass
class StageTiming:
    """Timing data for a single pipeline stage."""

    name: str
    wall_time_s: float
    call_count: int = 1


class PipelineProfiler:
    """Collects wall-clock timing for pipeline stages.

    Usage::

        profiler = PipelineProfiler()

        # Decorator usage (bind profiler to instance methods)
        class MyPipeline:
            def __init__(self, profiler: PipelineProfiler):
                self.profiler = profiler

            @profile_stage("extract_poses", profiler_attr="profiler")
            def extract(self, path):
                ...

        # Or context manager
        with profiler:
            pipeline.analyze(video)
        print(profiler.summary_table())
    """

    def __init__(self) -> None:
        self._stages: list[StageTiming] = []
        self._start_time: float = 0.0
        self._running = False

    @property
    def stages(self) -> list[StageTiming]:
        return list(self._stages)

    @property
    def total_wall_time_s(self) -> float:
        if not self._running and self._start_time == 0.0:
            return sum(s.wall_time_s for s in self._stages)
        return time.perf_counter() - self._start_time

    def record(self, name: str, wall_time_s: float) -> None:
        """Manually record a stage timing."""
        # Merge if same stage name already recorded
        for stage in self._stages:
            if stage.name == name:
                stage.wall_time_s += wall_time_s
                stage.call_count += 1
                return
        self._stages.append(StageTiming(name=name, wall_time_s=wall_time_s))

    def reset(self) -> None:
        """Clear all recorded stages."""
        self._stages.clear()
        self._start_time = 0.0
        self._running = False

    def summary_table(self) -> str:
        """Return a formatted table of all recorded stages."""
        if not self._stages:
            return "No stages recorded."

        total = sum(s.wall_time_s for s in self._stages)

        lines = []
        lines.append(f"{'Stage':<40} {'Time (s)':>10} {'%':>7} {'Calls':>6}")
        lines.append("-" * 67)

        for stage in self._stages:
            pct = (stage.wall_time_s / total * 100) if total > 0 else 0
            lines.append(
                f"{stage.name:<40} {stage.wall_time_s:>10.4f} {pct:>6.1f}% {stage.call_count:>6}"
            )

        lines.append("-" * 67)
        lines.append(f"{'TOTAL':<40} {total:>10.4f} {'100.0%':>7}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict of all stages."""
        total = sum(s.wall_time_s for s in self._stages)
        return {
            "total_wall_time_s": total,
            "stages": [
                {
                    "name": s.name,
                    "wall_time_s": round(s.wall_time_s, 6),
                    "pct": round(s.wall_time_s / total * 100, 1) if total > 0 else 0,
                    "call_count": s.call_count,
                }
                for s in self._stages
            ],
        }

    def __enter__(self) -> PipelineProfiler:
        self._start_time = time.perf_counter()
        self._running = True
        return self

    def __exit__(self, *args: Any) -> None:
        self._running = False


def profile_stage(
    stage_name: str,
    profiler_attr: str = "_profiler",
) -> Callable[[Callable], Callable]:
    """Decorator that times a method and records to the pipeline profiler.

    The profiler instance is expected to be on the object at ``profiler_attr``.
    If not found, the call runs normally without recording.

    Args:
        stage_name: Name for this stage in the profiler output.
        profiler_attr: Attribute name on the instance to find PipelineProfiler.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            profiler: PipelineProfiler | None = getattr(self, profiler_attr, None)
            if profiler is None:
                return func(self, *args, **kwargs)

            start = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                wall_time = time.perf_counter() - start
                profiler.record(stage_name, wall_time)

        return wrapper

    return decorator
```

Also add the export to `ml/skating_ml/utils/__init__.py` — check what's already there and add `profiling` if needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && .venv/bin/python -m pytest tests/utils/test_profiling.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/utils/profiling.py ml/tests/utils/test_profiling.py
git commit -m "feat(ml): add PipelineProfiler and @profile_stage decorator"
```

---

### Task 2: Instrument `AnalysisPipeline.analyze()` with profiler

**Files:**

- Modify: `ml/skating_ml/pipeline.py:41-321` (AnalysisPipeline class)
- Modify: `ml/skating_ml/pipeline.py:159-321` (analyze method)

- [ ] **Step 1: Read pipeline.py to confirm current state**

Run: `head -100 ml/skating_ml/pipeline.py`
Verify the file structure matches the plan context.

- [ ] **Step 2: Add profiler to AnalysisPipeline.__init__**

Add `_profiler` attribute and optional `profiler` parameter to `__init__`:

```python
from .utils.profiling import PipelineProfiler

# In __init__:
def __init__(
    self,
    ...,
    profiler: PipelineProfiler | None = None,
) -> None:
    ...
    self._profiler = profiler or PipelineProfiler()
```

- [ ] **Step 3: Instrument `analyze()` stages**

Wrap each stage with `self._profiler.record()`. The stages in `analyze()` (lines 159-321):

1. `video_meta` — `get_video_meta(video_path)`
2. `extract_and_track` — `self._extract_and_track(video_path, meta)`
3. `normalize` — `self._get_normalizer().normalize(compensated_h36m)`
4. `smooth` — `self._get_smoother(meta.fps).smooth(normalized)`
5. `3d_lift` — `self._get_pose_3d_extractor().extract_sequence(smoothed)`
6. `blade_detection` — per-frame blade edge detection
7. `phase_detection` — `self._get_phase_detector().detect_phases()`
8. `metrics` — `analyzer.analyze(smoothed, phases, meta.fps)`
9. `dtw_alignment` — `aligner.compute_distance()`
10. `physics` — `physics_engine.fit_jump_trajectory()`
11. `recommendations` — `recommender.recommend()`

Wrap each stage with manual `start = time.perf_counter()` / `self._profiler.record()` calls. Do NOT use the decorator here — manual calls give more control over the try/except blocks already in the code.

Example for one stage:
```python
import time

# Before: just the call
compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)

# After: wrapped with timing
t0 = time.perf_counter()
compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)
self._profiler.record("extract_and_track", time.perf_counter() - t0)
```

Apply the same pattern to all stages. The `3d_lift` and `blade_detection` stages are inside a try/except — wrap the whole block:

```python
t0 = time.perf_counter()
try:
    poses_3d = self._get_pose_3d_extractor().extract_sequence(smoothed)
    # ... blade detection ...
except Exception:
    pass
self._profiler.record("3d_lift_and_blade", time.perf_counter() - t0)
```

- [ ] **Step 4: Add `profiler` property to AnalysisReport**

The `AnalysisReport` should carry the profiler output so the worker can log it. Check `ml/skating_ml/types.py` for `AnalysisReport` and add an optional `profiling: dict | None = None` field.

- [ ] **Step 5: Verify no regressions — run existing pipeline tests**

Run: `cd ml && .venv/bin/python -m pytest tests/ -v -k "not gpu and not numba" 2>&1 | tail -20`
Expected: All existing tests pass (profiler defaults to no-op recording).

- [ ] **Step 6: Commit**

```bash
git add ml/skating_ml/pipeline.py ml/skating_ml/types.py
git commit -m "perf(ml): instrument AnalysisPipeline stages with PipelineProfiler"
```

---

### Task 3: Create `profile_pipeline.py` CLI script

**Files:**

- Create: `ml/scripts/profile_pipeline.py`

- [ ] **Step 1: Write the profiling script**

```python
#!/usr/bin/env python3
"""Profile the full ML pipeline on a real video.

Usage:
    cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 [--element waltz_jump] [--json output.json]

Outputs a table of wall-clock times per pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skating_ml.pipeline import AnalysisPipeline
from skating_ml.utils.profiling import PipelineProfiler


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ML pipeline on a video")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--element", type=str, default=None, help="Element type (e.g., waltz_jump)")
    parser.add_argument("--json", type=Path, default=None, help="Output JSON to file")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable temporal smoothing")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"Video: {args.video}")
    print(f"Element: {args.element or 'poses only'}")
    print()

    profiler = PipelineProfiler()
    pipeline = AnalysisPipeline(
        profiler=profiler,
        enable_smoothing=not args.no_smoothing,
    )

    with profiler:
        report = pipeline.analyze(
            video_path=args.video,
            element_type=args.element,
        )

    # Print summary
    print(profiler.summary_table())
    print()

    # Print additional info
    print(f"Element: {report.element_type}")
    print(f"Frames: {report.phases.end if report.phases else 'N/A'}")
    print(f"Score: {report.overall_score}/10")
    print(f"Metrics: {len(report.metrics)}")

    # Export JSON if requested
    if args.json:
        data = profiler.to_dict()
        data["video"] = str(args.video)
        data["element"] = args.element
        data["score"] = report.overall_score
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Test with a real video**

Find a test video in `data/` or use any short mp4:

```bash
find /home/michael/Github/skating-biomechanics-ml/data -name "*.mp4" -o -name "*.webm" | head -5
```

Then run:
```bash
cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 --element waltz_jump
```

Expected: Table with wall-clock times per stage. The biggest time should be `extract_and_track` (RTMPose inference).

- [ ] **Step 3: Commit**

```bash
git add ml/scripts/profile_pipeline.py
git commit -m "feat(ml): add profile_pipeline.py CLI for pipeline benchmarking"
```

---

### Task 4: Run profiling, collect real data, report

**Files:**

- Create: `docs/research/PIPELINE_PROFILING_2026-04-18.md`

- [ ] **Step 1: Run profiling on a real video**

Run the profiler on the longest available video for meaningful results:

```bash
cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 --element waltz_jump --json /tmp/profiling_results.json
```

- [ ] **Step 2: Run profiling with smoothing disabled**

```bash
cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 --element waltz_jump --no-smoothing --json /tmp/profiling_no_smooth.json
```

- [ ] **Step 3: Analyze results and write report**

Write `docs/research/PIPELINE_PROFILING_2026-04-18.md` with:

1. Table of stage timings (actual measured numbers, NO estimates)
2. Pie chart of time distribution (optional, ASCII is fine)
3. Identification of the actual bottleneck
4. Comparison: with vs without smoothing
5. Recommendations for where optimization effort should go

Key questions the report must answer:
- What % of total time is RTMPose extraction?
- What % is 3D lifting?
- What % is smoothing (with Numba)?
- What % is all CPU-only stages combined (normalize, metrics, phase detection, recommendations)?
- Is Numba JIT cold-start visible in the first run?

- [ ] **Step 4: Commit**

```bash
git add docs/research/PIPELINE_PROFILING_2026-04-18.md
git commit -m "docs: add pipeline profiling results with measured stage timings"
```
