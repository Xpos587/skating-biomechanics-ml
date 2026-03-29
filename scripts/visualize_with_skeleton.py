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

from src.blazepose_extractor import BlazePoseExtractor
# 2D BladeEdgeDetector deprecated - use BladeEdgeDetector3D for H3.6M format
# from src.blade_edge_detector import BladeEdgeDetector
from src.blade_edge_detector_3d import BladeEdgeDetector3D, DetectionConfig
from src.pose_estimation import BKey, H36M_SKELETON_EDGES
from src.smoothing import PoseSmoother, get_skating_optimized_config
from src.spatial_reference import SpatialReferenceDetector
from src.subtitles import SubtitleParser
from src.video import get_video_meta
from src.visualization import (
    draw_3d_trajectory,
    draw_axis_indicator,
    draw_blade_state_3d_hud,
    draw_debug_hud,
    draw_ice_trace,
    draw_motion_direction_arrow,
    draw_skeleton,
    draw_skeleton_3d_pip,
    draw_spatial_axes,
    draw_subtitle_cyrillic,
    draw_trails,
    draw_velocity_vectors,
)


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
    parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="Enable 3D pose visualization with depth color coding",
    )
    parser.add_argument(
        "--model-3d",
        type=Path,
        help="Path to 3D pose model (motionagformer-s-ap3d.pth.tr)",
    )
    parser.add_argument(
        "--blade-3d",
        action="store_true",
        help="Enable 3D blade detection (edge zones, motion direction, ice trace)",
    )
    parser.add_argument(
        "--3d-scale",
        dest="d_3d_scale",
        type=float,
        default=0.6,
        help="3D skeleton scale in PIP window (smaller = larger, default: 0.6)",
    )
    parser.add_argument(
        "--no-3d-autoscale",
        action="store_true",
        help="Disable auto-scaling of 3D skeleton to fill PIP window",
    )
    parser.add_argument(
        "--floor-mode",
        action="store_true",
        help="Floor mode: analysis without ice skates (disables blade detection)",
    )
    parser.add_argument(
        "--com-trajectory",
        action="store_true",
        help="Enable Center of Mass trajectory line (yellow line)",
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
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output with libx265 (smaller file, slower)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Compression quality for libx265 (lower=better, default: 30)",
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
        from src.video import extract_frames

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

        # Blade edge detection now requires 3D poses (--blade-3d flag)
        # Skip for floor mode or when 3D not enabled
        if args.floor_mode or not args.blade_3d:
            if not args.floor_mode and not args.blade_3d:
                print("Note: Blade edge detection requires --blade-3d flag for H3.6M format")
            blade_states_left = [None] * len(poses_viz)
            blade_states_right = [None] * len(poses_viz)
        else:
            # Will be handled after 3D pose extraction below
            blade_states_left = None
            blade_states_right = None

        # Show breakdown (only if blade states were computed)
        if not args.floor_mode and blade_states_left and blade_states_left[0] is not None:
            from collections import Counter
            left_breakdown = Counter(s.blade_type.name for s in blade_states_left)
            right_breakdown = Counter(s.blade_type.name for s in blade_states_right)
            print(f"Blade states detected: {len(blade_states_left)} left, {len(blade_states_right)} right")
            print(f"  Left breakdown: {dict(left_breakdown)}")
            print(f"  Right breakdown: {dict(right_breakdown)}")

    # Initialize blade states for loaded poses
    if args.poses and args.poses.exists():
        if args.floor_mode or not args.blade_3d:
            if not args.floor_mode and not args.blade_3d:
                print("Note: Blade edge detection requires --blade-3d flag for H3.6M format")
            blade_states_left = [None] * len(poses_viz)
            blade_states_right = [None] * len(poses_viz)
        else:
            # Will be handled after 3D pose extraction below
            blade_states_left = None
            blade_states_right = None

    # Initialize 3D pose extraction if requested
    poses_3d = None
    # --blade-3d requires 3D poses, so auto-enable --3d if not set
    if args.blade_3d and not args.use_3d:
        print("Note: --blade-3d requires 3D poses, auto-enabling --3d")
        args.use_3d = True

    if args.use_3d:
        if args.model_3d and args.model_3d.exists():
            print(f"Loading 3D model: {args.model_3d}")
            from src.pose_3d import AthletePose3DExtractor
            from src.pose_estimation import blazepose_to_h36m

            # Convert BlazePose to H3.6M format
            poses_h36m = blazepose_to_h36m(poses_viz)

            # Extract 3D poses
            extractor = AthletePose3DExtractor(
                model_path=args.model_3d,
                model_type="motionagformer-s"
            )
            poses_3d = extractor.extract_sequence(poses_h36m)
            print(f"3D poses extracted: {poses_3d.shape}")
        else:
            print("Using biomechanics-based 3D estimation...")
            from src.pose_estimation import blazepose_to_h36m
            from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator

            # Convert BlazePose to H3.6M format
            poses_h36m = blazepose_to_h36m(poses_viz)

            # Use simple biomechanics estimator
            estimator = Biomechanics3DEstimator()
            poses_3d = estimator.estimate_3d(poses_h36m)
            print(f"3D poses estimated: {poses_3d.shape}")

    # Calculate fixed auto-scale parameters from reference frame (median frame)
    # This prevents jitter from per-frame scaling
    pip_scale = None
    pip_offset = None
    if poses_3d is not None and not args.no_3d_autoscale:
        from src.visualization import project_3d_to_2d

        # Use median frame as reference (middle of video)
        ref_frame_idx = len(poses_3d) // 2
        ref_pose_3d = poses_3d[ref_frame_idx]

        # Center reference pose at origin (same as draw_skeleton_3d_pip)
        center_3d = ref_pose_3d.mean(axis=0)
        ref_pose_3d_centered = ref_pose_3d - center_3d

        # Project reference pose
        ref_pose_2d = project_3d_to_2d(
            ref_pose_3d_centered[np.newaxis, ...],
            width=meta.width // 2,
            height=meta.height // 2,
            camera_distance=args.d_3d_scale,
        )[0]

        # Calculate scale from reference pose
        x_coords = ref_pose_2d[:, 0]
        y_coords = ref_pose_2d[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range > 1e-6 and y_range > 1e-6:
            scale_x = 1.0 / x_range
            scale_y = 1.0 / y_range
            pip_scale = min(scale_x, scale_y) * 0.9  # 10% padding

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            pip_offset = (x_center, y_center)

            print(f"3D PIP auto-scale: scale={pip_scale:.2f}, offset=({x_center:.2f}, {y_center:.2f})")

    # Initialize 3D blade detector if requested
    blade_detector_3d = None
    blade_states_3d_left = []
    blade_states_3d_right = []
    if args.blade_3d and poses_3d is not None:
        print("Initializing 3D blade detector...")
        blade_detector_3d = BladeEdgeDetector3D(fps=meta.fps)

        # Process 3D poses to detect blade states
        for i, pose_3d in enumerate(poses_3d):
            # Detect left foot blade state
            state_left = blade_detector_3d.detect_frame(pose_3d, i, foot="left")
            blade_states_3d_left.append(state_left)

            # Detect right foot blade state
            state_right = blade_detector_3d.detect_frame(pose_3d, i, foot="right")
            blade_states_3d_right.append(state_right)

        print(f"3D blade states: {len(blade_states_3d_left)} left, {len(blade_states_3d_right)} right")

        # Update blade_states_left/right for use in visualization
        blade_states_left = blade_states_3d_left
        blade_states_right = blade_states_3d_right

        # Show breakdown
        from collections import Counter
        left_breakdown = Counter(s.blade_type.name for s in blade_states_3d_left)
        right_breakdown = Counter(s.blade_type.name for s in blade_states_3d_right)
        print(f"  Left breakdown: {dict(left_breakdown)}")
        print(f"  Right breakdown: {dict(right_breakdown)}")

    # Initialize spatial reference detector (always)
    print("Initializing spatial reference detector...")
    spatial_detector = SpatialReferenceDetector(
        hough_threshold=80,
        hough_min_line_length=100,
        hough_max_line_gap=10,
    )

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

    # If compression requested, write to temp file first
    if args.compress:
        import tempfile
        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = Path(temp_output.name)
        temp_output.close()
        write_path = temp_path
        print(f"Writing to temp file: {temp_path}")
    else:
        write_path = output_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(write_path), fourcc, meta.fps, (meta.width, meta.height))

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
            # Draw 3D skeleton in PIP window if enabled
            if args.use_3d and poses_3d is not None and current_pose_idx < len(poses_3d):
                frame = draw_skeleton_3d_pip(
                    frame,
                    poses_3d[current_pose_idx],
                    meta.width,
                    meta.height,
                )
            else:
                # Draw 2D skeleton (33 keypoints) when 3D is disabled
                frame = draw_skeleton(frame, poses[current_pose_idx], meta.height, meta.width)

        # Layer 1: Kinematics (use normalized coords)
        if args.layer >= 1 and current_pose_idx is not None:
            # Note: Velocity vectors and trails require H3.6M 17kp format
            # Skip for BlazePose 33kp format (which we're using here)
            if poses_viz.shape[1] == 17:  # H3.6M format
                frame = draw_velocity_vectors(
                    frame, poses_viz, current_pose_idx, meta.fps, meta.height, meta.width
                )
            trail_history.append(poses_viz[current_pose_idx].copy())
            # Draw trail for left ankle (free leg in jumps)
            if poses_viz.shape[1] == 33:  # BlazePose format
                trail_key = BKey.LEFT_ANKLE
            else:  # H3.6M format
                from src.types import H36Key
                trail_key = H36Key.LFOOT
            frame = draw_trails(
                frame, trail_history, trail_key, meta.height, meta.width
            )

            # Draw 3D CoM trajectory if enabled
            if args.use_3d and poses_3d is not None and current_pose_idx < len(poses_3d) and args.com_trajectory:
                from src.visualization import draw_3d_trajectory
                from src.analysis import PhysicsEngine

                # Calculate CoM for the trajectory up to current frame
                engine = PhysicsEngine(body_mass=60.0)
                com_trajectory = engine.calculate_center_of_mass(poses_3d[:current_pose_idx + 1])

                # Draw CoM trajectory
                if len(com_trajectory) > 1:
                    frame = draw_3d_trajectory(
                        frame, com_trajectory, meta.height, meta.width, camera_z=args.d_3d_scale
                    )

        # Layer 2: Technical (blade states shown in HUD via draw_debug_hud)
        if args.layer >= 2 and current_pose_idx is not None:
            # Draw axis alignment indicator (trunk tilt)
            frame = draw_axis_indicator(
                frame, poses_viz[current_pose_idx], meta.height, meta.width
            )

            # 3D blade HUD disabled - too many UNKNOWN states without proper 3D model
            # TODO: Re-enable after MotionAGFormer model is integrated

            # Ice trace disabled - projection doesn't work with meter-scale 3D positions
            # TODO: Re-enable after fixing 3D-to-2D projection with proper scaling
            # if args.blade_3d and blade_detector_3d and current_pose_idx > 10:
            #     left_trace = blade_detector_3d.get_ice_trace("left")
            #     if len(left_trace.points) > 2:
            #         frame = draw_ice_trace(frame, left_trace, meta.height, meta.width)

        # Draw spatial axes (all layers >= 1)
        if args.layer >= 1:
            # Estimate camera pose from current frame
            camera_pose = spatial_detector.estimate_pose(frame)

            # Log roll values periodically (every 100 frames)
            if frame_idx % 100 == 0 and camera_pose.confidence > 0:
                print(f"  Frame {frame_idx}: Roll={camera_pose.roll:.2f}°, Conf={camera_pose.confidence:.2f}, Source={camera_pose.source}")

            # XYZ axes disabled - not useful for skating analysis
            # frame = draw_spatial_axes(frame, camera_pose)

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

        # Get blade states for current frame (prefer 3D if available)
        if blade_states_3d_left and current_pose_idx is not None and current_pose_idx < len(blade_states_3d_left):
            blade_left = blade_states_3d_left[current_pose_idx]
            blade_right = blade_states_3d_right[current_pose_idx] if current_pose_idx < len(blade_states_3d_right) else None
        else:
            blade_left = blade_states_left[current_pose_idx] if current_pose_idx is not None and current_pose_idx < len(blade_states_left) else None
            blade_right = blade_states_right[current_pose_idx] if current_pose_idx is not None and current_pose_idx < len(blade_states_right) else None

        frame = draw_debug_hud(
            frame,
            active_segment,
            kinematics,
            frame_idx,
            meta.num_frames,
            meta.fps,
            meta.height,
            meta.width,
            blade_state_left=blade_left,
            blade_state_right=blade_right,
        )

        writer.write(frame)
        frame_idx += 1

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{meta.num_frames} frames")

    cap.release()
    writer.release()

    # Compress with ffmpeg if requested
    if args.compress and write_path != output_path:
        print(f"\nCompressing with libx265 (CRF={args.crf})...")
        import subprocess

        compress_cmd = [
            "ffmpeg",
            "-i", str(write_path),
            "-c:v", "libx265",
            "-crf", str(args.crf),
            "-preset", "medium",
            "-tune", "animation",
            "-c:a", "copy",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(compress_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Remove temp file
            write_path.unlink()
            original_size = write_path.stat().st_size if write_path.exists() else 0
            compressed_size = output_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            print(f"Compression: {original_size // (1024*1024)}MB -> {compressed_size // (1024*1024)}MB ({ratio:.0f}% reduction)")
        else:
            print(f"FFmpeg error: {result.stderr}")
            print(f"Temp file saved to: {write_path}")
            return 1

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
    from src.geometry import angle_3pt  # noqa: PLC0415

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
