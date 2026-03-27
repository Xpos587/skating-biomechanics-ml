#!/usr/bin/env python3
"""Enhanced visualization with skeleton, kinematics, and subtitles.

Implements the layered HUD architecture from Gemini Deep Research:
- Layer 0 (Raw): Video + skeleton only
- Layer 1 (Kinematics): + velocity vectors + trails
- Layer 2 (Technical): + edge indicators + joint angles
- Layer 3 (Coaching): + subtitles + GOE scores

Usage:
    python scripts/visualize_with_skeleton.py video.mp4 --layer 3
    python scripts/visualize_with_skeleton.py video.mp4 --layer 1 --output video_kinematics.mp4
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from skating_biomechanics_ml.pose_2d import BlazePoseExtractor, PoseNormalizer
from skating_biomechanics_ml.types import BKey
from skating_biomechanics_ml.utils import get_skating_optimized_config
from skating_biomechanics_ml.utils.smoothing import PoseSmoother
from skating_biomechanics_ml.utils import (
    draw_debug_hud,
    draw_edge_indicators,
    draw_skeleton,
    draw_subtitle_cyrillic,
    draw_trails,
    draw_velocity_vectors,
)
from skating_biomechanics_ml.utils.subtitles import SubtitleParser
from skating_biomechanics_ml.utils.video import get_video_meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhanced skating visualization with layered HUD"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--layer",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="HUD layer: 0=Raw, 1=Kinematics, 2=Technical, 3=Coaching (default: 3)",
    )
    parser.add_argument("--output", type=Path, help="Output video path")
    parser.add_argument(
        "--poses", type=Path, help="Pre-computed poses .npz file (optional)"
    )
    parser.add_argument(
        "--segments", type=Path, help="Segmentation JSON file (optional)"
    )
    parser.add_argument(
        "--subtitles", type=Path, help="VTT subtitle file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of frames for motion trail (default: 20)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=30,
        help="Font size for subtitles (default: 30)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    print(f"Processing: {args.video}")
    print(f"HUD Layer: {args.layer} ({_get_layer_name(args.layer)})")

    # Load video metadata
    meta = get_video_meta(args.video)
    cap = cv2.VideoCapture(str(args.video))

    # Load or extract poses
    if args.poses and args.poses.exists():
        print(f"Loading poses from: {args.poses}")
        poses_data = np.load(args.poses)
        poses = poses_data["poses"]
        # Assume sequential frame mapping when loading from file
        pose_frame_indices = np.arange(len(poses))
        # For loading: assume poses are already in correct format
        poses_viz = poses[:, :, :2] if poses.shape[2] == 3 else poses
    else:
        print("Extracting poses frame-by-frame for perfect sync...")
        extractor = BlazePoseExtractor(
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            num_poses=1,
        )

        # Extract poses for ALL frames to maintain sync
        # Use extract_frames to iterate with frame indices
        from skating_biomechanics_ml.utils.video import extract_frames

        poses_list = []
        frame_indices = []
        for frame_idx, frame in enumerate(extract_frames(args.video)):
            timestamp_ms = int(frame_idx * 1000 / meta.fps)
            kp = extractor.extract_frame(frame, timestamp_ms)
            if kp is not None:
                poses_list.append(kp)
                frame_indices.append(frame_idx)

        poses_raw = np.stack(poses_list)
        print(f"Extracted {len(poses_raw)} poses from {meta.num_frames} frames")
        print(f"Frame indices: {frame_indices[0]} to {frame_indices[-1]}")

        # Store frame indices for sync during visualization
        pose_frame_indices = np.array(frame_indices)

        # Convert to normalized for smoothing (more stable)
        poses_xy = poses_raw[:, :, :2]
        poses_norm_raw = poses_xy.copy()
        poses_norm_raw[:, :, 0] /= meta.width
        poses_norm_raw[:, :, 1] /= meta.height

        # Smooth in normalized space
        print("Smoothing poses (in normalized space)...")
        config = get_skating_optimized_config(fps=meta.fps)
        smoother = PoseSmoother(config=config, freq=meta.fps)
        poses_smoothed_norm = smoother.smooth(poses_norm_raw)

        # Clamp to [0,1] to prevent out-of-bounds
        poses_smoothed_norm = np.clip(poses_smoothed_norm, 0.0, 1.0)

        # Convert smoothed poses back to pixels for skeleton drawing
        poses_smoothed_px = poses_smoothed_norm.copy()
        poses_smoothed_px[:, :, 0] *= meta.width
        poses_smoothed_px[:, :, 1] *= meta.height

        # Add back confidence (keep original)
        poses = np.zeros((len(poses_smoothed_px), 33, 3), dtype=np.float32)
        poses[:, :, :2] = poses_smoothed_px
        poses[:, :, 2] = poses_raw[:, :, 2]

        # For visualization: use normalized smoothed poses
        poses_viz = poses_smoothed_norm

        raw_jitter = np.abs(np.diff(poses_norm_raw[:, :, 0], axis=0)).mean()
        smooth_jitter = np.abs(np.diff(poses_smoothed_norm[:, :, 0], axis=0)).mean()
        print(f"Jitter reduction: {(1 - smooth_jitter/raw_jitter)*100:.1f}%")

    # Load segments
    segments = []
    if args.segments and args.segments.exists():
        data = json.loads(args.segments.read_text())
        segments = data.get("segments", [])
        print(f"Loaded {len(segments)} segments")

    # Load subtitles
    subtitle_events = []
    subtitle_path = args.subtitles or args.video.with_suffix(".vtt")
    if subtitle_path.exists():
        print(f"Loading subtitles from: {subtitle_path}")
        parser = SubtitleParser()
        subtitle_events = parser.parse_vtt(subtitle_path)
        print(f"Loaded {len(subtitle_events)} subtitle events")

    # Setup output
    output_path = args.output or args.video.parent / f"{args.video.stem}_layer{args.layer}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, meta.fps, (meta.width, meta.height))

    # Trail history (for layer 1+)
    trail_history = deque(maxlen=args.trail_length)

    # Process frames
    frame_idx = 0
    pose_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find the pose that corresponds to this frame
        # pose_frame_indices stores which video frame each pose came from
        current_pose_idx = None
        while pose_idx < len(pose_frame_indices):
            if pose_frame_indices[pose_idx] == frame_idx:
                current_pose_idx = pose_idx
                pose_idx += 1
                break
            elif pose_frame_indices[pose_idx] < frame_idx:
                # This pose is from an earlier frame, skip it
                pose_idx += 1
            else:
                # This pose is from a future frame, no pose for current frame
                break

        # Layer 0: Skeleton (use pixel coords for direct overlay)
        if args.layer >= 0 and current_pose_idx is not None:
            frame = draw_skeleton(frame, poses[current_pose_idx], meta.height, meta.width)

        # Layer 1: Kinematics (use normalized coords)
        if args.layer >= 1 and current_pose_idx is not None:
            frame = draw_velocity_vectors(
                frame, poses_viz, current_pose_idx, meta.fps, meta.height, meta.width
            )
            trail_history.append(poses_viz[current_pose_idx].copy())
            # Draw trail for left ankle (free leg in jumps)
            frame = draw_trails(
                frame, trail_history, BKey.LEFT_ANKLE, meta.height, meta.width
            )

        # Layer 2: Technical (edge indicators - use normalized coords)
        if args.layer >= 2 and current_pose_idx is not None:
            frame = draw_edge_indicators(
                frame, poses_viz, current_pose_idx, meta.height, meta.width
            )

        # Layer 3: Coaching (subtitles)
        if args.layer >= 3 and subtitle_events:
            current_time = frame_idx / meta.fps
            for event in subtitle_events:
                if event.start_time <= current_time <= (
                    event.end_time or event.start_time + 5
                ):
                    # Build subtitle text
                    text_parts = []
                    if event.name != "unknown":
                        text_parts.append(event.name.upper())
                    if event.instructions:
                        text_parts.extend(event.instructions)
                    if text_parts:
                        subtitle_text = " | ".join(text_parts)
                        frame = draw_subtitle_cyrillic(
                            frame,
                            subtitle_text,
                            (50, meta.height - 50),
                            font_size=args.font_size,
                        )
                    break

        # Draw HUD (all layers)
        active_segment = _get_active_segment(segments, frame_idx)
        kinematics = _compute_kinematics(poses_viz, current_pose_idx if current_pose_idx is not None else 0, meta.fps)
        frame = draw_debug_hud(
            frame,
            active_segment,
            kinematics,
            frame_idx,
            meta.num_frames,
            meta.fps,
            meta.height,
            meta.width,
        )

        writer.write(frame)
        frame_idx += 1

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{meta.num_frames} frames")

    cap.release()
    writer.release()

    print(f"\nSaved to: {output_path}")
    return 0


def _get_layer_name(layer: int) -> str:
    """Get human-readable layer name."""
    names = {
        0: "Raw - Skeleton only",
        1: "Kinematics - Velocity + Trails",
        2: "Technical - Edge indicators + Angles",
        3: "Coaching - Subtitles + Full HUD",
    }
    return names.get(layer, "Unknown")


def _get_active_segment(segments: list, frame_idx: int) -> dict:
    """Find segment containing this frame."""
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if start <= frame_idx <= end:
            return {
                "type": seg.get("element_type", "unknown"),
                "start": start,
                "end": end,
                "confidence": seg.get("confidence", 0.0),
            }
    return {}


def _compute_kinematics(
    poses: np.ndarray, frame_idx: int, fps: float
) -> dict[str, float]:
    """Compute kinematics for this frame.

    Args:
        poses: Full pose sequence (num_frames, 33, 2).
        frame_idx: Current frame index.
        fps: Frame rate.

    Returns:
        Dict with kinematic metrics.
    """
    if frame_idx == 0 or frame_idx >= len(poses) - 1:
        return {}

    # Compute velocity using central difference
    velocity = (poses[frame_idx + 1] - poses[frame_idx - 1]) * fps / 2

    # Hip velocity (center of mass)
    hip_v = np.linalg.norm(velocity[BKey.LEFT_HIP])

    # Knee angles
    from skating_biomechanics_ml.utils.geometry import angle_3pt  # noqa: PLC0415

    left_knee = angle_3pt(
        poses[frame_idx, BKey.LEFT_HIP],
        poses[frame_idx, BKey.LEFT_KNEE],
        poses[frame_idx, BKey.LEFT_ANKLE],
    )

    right_knee = angle_3pt(
        poses[frame_idx, BKey.RIGHT_HIP],
        poses[frame_idx, BKey.RIGHT_KNEE],
        poses[frame_idx, BKey.RIGHT_ANKLE],
    )

    return {
        "hip_velocity": hip_v,
        "left_knee": left_knee,
        "right_knee": right_knee,
    }


if __name__ == "__main__":
    raise SystemExit(main())
