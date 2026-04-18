#!/usr/bin/env python3
"""Profile the full ML pipeline on a real video.

Usage:
    cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 [--element waltz_jump] [--json output.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skating_ml.pipeline import AnalysisPipeline
from skating_ml.utils.profiling import PipelineProfiler


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ML pipeline on a video")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--element", type=str, default=None, help="Element type")
    parser.add_argument("--json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable smoothing")
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

    print(profiler.summary_table())
    print()
    print(f"Element: {report.element_type}")
    print(f"Frames: {report.phases.end if report.phases else 'N/A'}")
    print(f"Score: {report.overall_score}/10")
    print(f"Metrics: {len(report.metrics)}")

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
