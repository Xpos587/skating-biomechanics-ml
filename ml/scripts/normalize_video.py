#!/usr/bin/env python3
"""Normalize video to optimal format for pose estimation pipeline.

Converts any input video (4K HEVC, 60fps, etc.) to a standardized format
that is fast to decode and process:
- H.264 with libx264
- Max width 1280px (preserving aspect ratio)
- 30 fps (sufficient for biomechanics)
- yuv420p 8-bit pixel format
- CRF 23, fast preset (good quality/size balance)

This is a ONE-TIME preprocessing step. After normalization, all subsequent
operations (pose extraction, comparison, analysis) run ~20x faster.

Usage:
    python scripts/normalize_video.py input.mp4
    python scripts/normalize_video.py input.mp4 --output normalized.mp4
    python scripts/normalize_video.py input.mp4 --width 960 --fps 24 --crf 20
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import av


def normalize_video(
    input_path: Path,
    output_path: Path | None = None,
    max_width: int = 1280,
    target_fps: float = 30,
    crf: int = 23,
    preset: str = "fast",
    start_sec: float = 0,
    duration_sec: float = 0,
) -> Path:
    """Normalize video to optimal format for pose estimation.

    Args:
        input_path: Path to input video.
        output_path: Path for output (default: <input>_normalized.mp4).
        max_width: Maximum width in pixels (height scaled proportionally).
        target_fps: Target FPS (0 = keep original).
        crf: CRF quality (0-51, lower = better, 23 is default).
        preset: x264 preset (ultrafast/fast/medium/slow).
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds (0 = all).

    Returns:
        Path to normalized video.
    """
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_normalized{input_path.suffix}")

    input_container = av.open(str(input_path))
    in_stream = input_container.streams.video[0]

    # Calculate output dimensions (even numbers)
    orig_w = in_stream.codec_context.width
    orig_h = in_stream.codec_context.height
    out_w = max_width
    out_h = int(orig_h * max_width / orig_w)
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    # Determine output FPS
    if target_fps > 0:
        out_fps = Fraction(target_fps).limit_denominator(1000)
    else:
        out_fps = in_stream.average_rate

    print(f"Normalizing: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {orig_w}x{orig_h} -> {out_w}x{out_h}")
    print(f"  FPS: {float(in_stream.average_rate):.1f} -> {float(out_fps):.1f}")
    print(f"  CRF: {crf}, Preset: {preset}")
    print("Processing...", flush=True)

    output_container = av.open(str(output_path), "w")
    out_stream = output_container.add_stream("libx264", rate=out_fps)
    out_stream.width = out_w
    out_stream.height = out_h
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"preset": preset, "crf": str(crf)}

    # Seek to start
    if start_sec > 0:
        input_container.seek(int(start_sec * av.time_base.den / av.time_base.num))

    frame_count = 0
    max_frames = int(duration_sec * float(out_fps)) if duration_sec > 0 else 0

    for packet in input_container.demux(in_stream):
        for raw_frame in packet.decode():
            # Resize
            frame = raw_frame.reformat(out_w, out_h, format="yuv420p")

            for out_packet in out_stream.encode(frame):
                output_container.mux(out_packet)

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                break

        if max_frames > 0 and frame_count >= max_frames:
            break

    # Flush
    for out_packet in out_stream.encode():
        output_container.mux(out_packet)

    output_container.close()
    input_container.close()

    # Report result
    out_size = output_path.stat().st_size / (1024 * 1024)
    in_size = input_path.stat().st_size / (1024 * 1024)
    ratio = out_size / in_size if in_size > 0 else 0
    print(f"  Input:  {in_size:.1f} MB")
    print(f"  Output: {out_size:.1f} MB ({ratio:.0%} of original)")
    print(f"  Frames: {frame_count}")
    print(f"Done: {output_path}")

    return output_path


def is_normalized(path: Path, max_width: int = 1280, target_fps: float = 30) -> bool:
    """Check if video is already in normalized format.

    Returns True if video width <= max_width and fps <= target_fps + 5.
    """
    if not path.exists():
        return False

    container = av.open(str(path))
    stream = container.streams.video[0]
    width = stream.codec_context.width
    fps = float(stream.average_rate)
    codec = stream.codec_context.name
    container.close()

    return width <= max_width and fps <= target_fps + 5 and codec == "h264"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize video for fast pose estimation processing",
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output path (default: <input>_normalized.mp4)"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Max width in pixels (default: 1280)"
    )
    parser.add_argument("--fps", type=float, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--crf", type=int, default=23, help="CRF quality 0-51 (default: 23)")
    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["ultrafast", "fast", "medium", "slow"],
        help="x264 preset (default: fast)",
    )
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = all)")
    parser.add_argument("--check", action="store_true", help="Only check if already normalized")
    args = parser.parse_args()

    if args.check:
        if is_normalized(args.input, args.width, args.fps):
            print(f"Already normalized: {args.input}")
            return 0
        else:
            print(f"Not normalized: {args.input}")
            return 1

    normalize_video(
        args.input,
        args.output,
        max_width=args.width,
        target_fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        start_sec=args.start,
        duration_sec=args.duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
