#!/usr/bin/env python3
"""Dual-video comparison tool for figure skating training.

Compares athlete video against reference (professional) video with
configurable overlays: axis/tilt, joint angles, timer, skeleton.

Implements Kinovea-style comparison workflow.

Usage:
    python scripts/compare_videos.py athlete.mp4 reference.mp4 \\
        --element waltz_jump --mode side-by-side \\
        --overlays axis,angles,timer,skeleton \\
        --resize 1280 --output comparison.mp4
"""

import argparse
from pathlib import Path

from scripts.normalize_video import is_normalized, normalize_video
from src.visualization.comparison import ComparisonConfig, ComparisonMode, ComparisonRenderer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare athlete video against reference with overlays",
    )
    parser.add_argument("athlete", type=Path, help="Athlete video path")
    parser.add_argument("reference", type=Path, help="Reference/expert video path")
    parser.add_argument(
        "--element",
        type=str,
        default="three_turn",
        help="Element type (default: three_turn)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["side-by-side", "overlay"],
        default="side-by-side",
        help="Comparison mode (default: side-by-side)",
    )
    parser.add_argument(
        "--overlays",
        type=str,
        default="skeleton,axis,angles,timer",
        help="Comma-separated overlays: skeleton,axis,angles,timer (default: all)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=1280,
        help="Target width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--reference-color",
        type=str,
        default="magenta",
        help="Reference skeleton color in overlay mode (default: magenta)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path (default: <athlete>_compare.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS (0 = auto from athlete video)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = all, useful for quick testing)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start from this frame number (0 = beginning)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use libx265 compression (smaller file, slower)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Compression quality for libx265 (lower=better, default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for YOLO inference: '0' for GPU, 'cpu' for CPU (default: 0)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction of poses (ignore cached .npz)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="No-op: auto-normalization is now always enabled (kept for backward compat)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.athlete.exists():
        print(f"Error: Athlete video not found: {args.athlete}")
        return 1
    if not args.reference.exists():
        print(f"Error: Reference video not found: {args.reference}")
        return 1

    # Default output path
    if args.output is None:
        args.output = args.athlete.with_name(f"{args.athlete.stem}_compare{args.athlete.suffix}")

    # Parse overlay list
    overlays = [o.strip() for o in args.overlays.split(",")]
    valid = {"skeleton", "axis", "angles", "timer"}
    for o in overlays:
        if o not in valid:
            print(f"Warning: Unknown overlay '{o}', skipping. Valid: {valid}")

    # Parse mode
    mode = ComparisonMode(args.mode)

    # Color map
    color_map = {
        "magenta": (255, 0, 255),
        "green": (0, 255, 0),
        "cyan": (255, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
    }
    ref_color = color_map.get(args.reference_color, (255, 0, 255))

    # Auto-normalize: enabled by default, or explicitly via --normalize flag.
    # Skips videos that are already in normalized format.
    athlete_path = args.athlete
    reference_path = args.reference
    for label, vid in [("Athlete", args.athlete), ("Reference", args.reference)]:
        if not is_normalized(vid, args.resize):
            print(f"[{label}] Normalizing {vid}...", flush=True)
            norm = normalize_video(vid, max_width=args.resize, target_fps=30)
            if label == "Athlete":
                athlete_path = norm
            else:
                reference_path = norm

    config = ComparisonConfig(
        mode=mode,
        overlays=overlays,
        resize_width=args.resize,
        reference_color=ref_color,
        fps=args.fps,
        compress=args.compress,
        crf=args.crf,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        device=args.device,
        no_cache=args.no_cache,
    )

    renderer = ComparisonRenderer(config)
    renderer.process(athlete_path, reference_path, args.output, args.element)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
