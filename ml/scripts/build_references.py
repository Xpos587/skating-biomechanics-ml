#!/usr/bin/env python3
"""CLI script to build reference data from expert videos.

Usage:
    python scripts/build_references.py --video expert.mp4 --element three_turn \\
        --takeoff 1.0 --peak 1.2 --landing 1.4 --start 0.5 --end 2.0
"""

import argparse
import traceback
from pathlib import Path

from skating_ml.references import ReferenceBuilder, ReferenceStore

from skating_ml.pose_estimation import RTMPoseExtractor
from skating_ml.pose_estimation.normalizer import PoseNormalizer
from skating_ml.types import ElementPhase
from skating_ml.utils.video import get_video_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference data from expert skating videos")
    parser.add_argument("video", type=Path, help="Path to expert video file")
    parser.add_argument(
        "--element",
        type=str,
        required=True,
        choices=["three_turn", "waltz_jump", "toe_loop", "flip"],
        help="Type of skating element",
    )
    parser.add_argument(
        "--takeoff",
        type=float,
        required=True,
        help="Takeoff timestamp in seconds",
    )
    parser.add_argument(
        "--peak",
        type=float,
        required=True,
        help="Peak height timestamp in seconds",
    )
    parser.add_argument(
        "--landing",
        type=float,
        required=True,
        help="Landing timestamp in seconds",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start timestamp in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End timestamp in seconds (default: video end)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/references"),
        help="Output directory for reference files (default: data/references)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="expert",
        help="Reference name (default: expert)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate video file
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Initialize components
    pose_extractor = RTMPoseExtractor(mode="balanced")
    normalizer = PoseNormalizer(target_spine_length=0.4)
    builder = ReferenceBuilder(pose_extractor, normalizer)

    # Get video FPS
    meta = get_video_meta(args.video)
    fps = args.fps or meta.fps

    if fps <= 0:
        print(f"Error: Invalid FPS: {fps}")
        return 1

    # Convert timestamps to frame indices
    start_frame = int(args.start * fps)
    takeoff_frame = int(args.takeoff * fps)
    peak_frame = int(args.peak * fps)
    landing_frame = int(args.landing * fps)
    end_frame = int(args.end * fps) if args.end else meta.num_frames - 1

    # Create phase boundaries
    phases = ElementPhase(
        name=args.element,
        start=start_frame,
        takeoff=takeoff_frame,
        peak=peak_frame,
        landing=landing_frame,
        end=end_frame,
    )

    print(f"Building reference for: {args.element}")
    print(f"Video: {args.video} ({meta.width}x{meta.height} @ {fps:.2f} fps)")
    print(
        f"Phases: start={start_frame}, takeoff={takeoff_frame}, "
        f"peak={peak_frame}, landing={landing_frame}, end={end_frame}"
    )

    # Build reference
    try:
        ref = builder.build_from_video(args.video, args.element, phases)
        print(f"Extracted {ref.poses.shape[0]} frames")
        print(f"Pose shape: {ref.poses.shape}")

        # Save to store
        store = ReferenceStore(args.output_dir)
        store.set_builder(builder)
        output_path = store.add(ref)

        print(f"Saved reference to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error building reference: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
