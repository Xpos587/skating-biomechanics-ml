#!/usr/bin/env python3
"""Profile the full ML pipeline on a real video.

Usage:
    cd ml && .venv/bin/python scripts/profile_pipeline.py /path/to/video.mp4 [--element waltz_jump] [--json output.json] [--deep]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Deep profile: measure RTMO model init vs per-frame inference",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"Video: {args.video}")
    print(f"Element: {args.element or 'poses only'}")
    print()

    deep_data = None

    if args.deep:
        # Deep profiling: patch PoseExtractor.tracker property to time model init
        from skating_ml.pose_estimation.pose_extractor import PoseExtractor

        original_tracker_fget = PoseExtractor.tracker.fget  # type: ignore[attr-defined]
        tracker_init_time_s = 0.0

        def timed_tracker_fget(self):
            nonlocal tracker_init_time_s
            t0 = time.perf_counter()
            tracker = original_tracker_fget(self)
            tracker_init_time_s = time.perf_counter() - t0
            return tracker

        PoseExtractor.tracker = property(timed_tracker_fget)  # type: ignore[assignment]

    profiler = PipelineProfiler()
    pipeline = AnalysisPipeline(
        profiler=profiler,
        enable_smoothing=not args.no_smoothing,
    )

    try:
        with profiler:
            report = pipeline.analyze(
                video_path=args.video,
                element_type=args.element,
            )
    finally:
        if args.deep:
            PoseExtractor.tracker = property(original_tracker_fget)  # type: ignore[assignment]

    print(profiler.summary_table())

    if args.deep:
        # Collect sub-stage data from profiler
        stages = {s.name: s.wall_time_s for s in profiler.stages}
        rtmo_inference = stages.get("rtmo_inference_loop", 0)
        extractor_init = stages.get("extractor_init", 0)
        gap_fill = stages.get("gap_filling", 0)
        spatial_ref = stages.get("spatial_reference", 0)

        total_extract = rtmo_inference + extractor_init + gap_fill + spatial_ref

        print(f"\n{'=' * 60}")
        print("DEEP PROFILE: extract_and_track breakdown")
        print(f"{'=' * 60}")
        print(
            f"  extractor_init (lazy load):       {extractor_init:>10.3f}s  ({extractor_init / total_extract * 100:>5.1f}%)"
        )
        print(
            f"  RTMO model download + ONNX init:  {tracker_init_time_s:>10.3f}s  ({tracker_init_time_s / total_extract * 100:>5.1f}%)"
        )
        print(
            f"  rtmo_inference_loop:              {rtmo_inference:>10.3f}s  ({rtmo_inference / total_extract * 100:>5.1f}%)"
        )
        print(
            f"  gap_filling:                      {gap_fill:>10.3f}s  ({gap_fill / total_extract * 100:>5.1f}%)"
        )
        print(
            f"  spatial_reference:                {spatial_ref:>10.3f}s  ({spatial_ref / total_extract * 100:>5.1f}%)"
        )
        print(f"  {'─' * 46}")
        print(f"  total extract_and_track:          {total_extract:>10.3f}s")

        # Compute avg per-frame
        pure_inference = rtmo_inference - tracker_init_time_s
        from skating_ml.utils.video import get_video_meta

        video_meta = get_video_meta(args.video)
        num_frames = video_meta.num_frames
        avg_per_frame_ms = (pure_inference / max(num_frames, 1)) * 1000
        print(f"\n  Per-frame inference: {avg_per_frame_ms:.1f}ms/frame ({num_frames} frames)")
        print(f"{'=' * 60}")

        deep_data = {
            "extractor_init_s": round(extractor_init, 3),
            "rtmo_model_init_s": round(tracker_init_time_s, 3),
            "rtmo_inference_loop_s": round(rtmo_inference, 3),
            "pure_inference_s": round(max(pure_inference, 0), 3),
            "gap_filling_s": round(gap_fill, 3),
            "spatial_reference_s": round(spatial_ref, 3),
            "num_frames": num_frames,
            "avg_per_frame_ms": round(avg_per_frame_ms, 1),
        }

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
        if deep_data:
            data["deep_profile"] = deep_data
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
