#!/usr/bin/env python3
"""Enhanced visualization with skeleton, kinematics, and subtitles.

Implements the layered HUD architecture:
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.detection.blade_edge_detector_3d import BladeEdgeDetector3D
from src.detection.spatial_reference import SpatialReferenceDetector
from src.pose_estimation import H36Key, H36MExtractor
from src.types import BladeState3D
from src.utils.geometry import angle_3pt
from src.utils.smoothing import PoseSmoother, get_skating_optimized_config
from src.utils.subtitles import SubtitleParser
from src.utils.video import get_video_meta
from src.visualization import (
    LayerContext,
    TrailLayer,
    VelocityLayer,
    draw_blade_indicator_hud,
    draw_skeleton,
    draw_skeleton_3d_pip,
    project_3d_to_2d,
    render_cyrillic_text,
    render_layers,
)
from src.visualization.core.text import draw_text_box


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhanced skating visualization with layered HUD")
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
    parser.add_argument("--poses", type=Path, help="Pre-computed poses .npz file (optional)")
    parser.add_argument("--segments", type=Path, help="Segmentation JSON file (optional)")
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
        "--select-person",
        action="store_true",
        help="Interactive person selection: preview first seconds, choose target",
    )
    parser.add_argument(
        "--person-click",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Click point to select target person (pixel coordinates)",
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
        pose_frame_indices = np.arange(len(poses))
        poses_viz = poses[:, :, :2] if poses.shape[2] == 3 else poses
    else:
        print("Extracting poses with tracking (YOLO26-Pose + OC-SORT)...")
        extractor = H36MExtractor(
            model_size="s",  # small model — better accuracy for distant/small skaters
            conf_threshold=0.1,  # low threshold — detect distant skaters
            output_format="normalized",
            crop_enhance=True,  # ROI crop pass for small/distant skaters
        )

        # Person selection
        from src.types import PersonClick as _PersonClick

        person_click = None
        if args.person_click:
            person_click = _PersonClick(x=args.person_click[0], y=args.person_click[1])
            print(f"Using click point: ({person_click.x}, {person_click.y})")
        elif args.select_person:
            persons = extractor.preview_persons(args.video)
            if not persons:
                print("No persons detected in the first seconds.")
                return 1
            if len(persons) == 1:
                print(f"Only 1 person detected (track #{persons[0]['track_id']}). Auto-selecting.")
                mid_hip = persons[0]["mid_hip"]
                person_click = _PersonClick(
                    x=int(mid_hip[0] * meta.width),
                    y=int(mid_hip[1] * meta.height),
                )
            else:
                print(f"\nDetected {len(persons)} persons:\n")
                for i, p in enumerate(persons, 1):
                    x1, y1, x2, y2 = p["bbox"]
                    print(
                        f"  #{i}: track_id={p['track_id']}, "
                        f"bbox=({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f}), "
                        f"hits={p['hits']}, first_frame={p['first_frame']}"
                    )
                print()
                try:
                    choice = int(input(f"Select person [1-{len(persons)}]: "))
                except (ValueError, EOFError):
                    print("Cancelled.")
                    return 1
                if choice < 1 or choice > len(persons):
                    print(f"Invalid choice: {choice}")
                    return 1
                mid_hip = persons[choice - 1]["mid_hip"]
                person_click = _PersonClick(
                    x=int(mid_hip[0] * meta.width),
                    y=int(mid_hip[1] * meta.height),
                )
                print(f"Selected person #{choice} (track_id={persons[choice - 1]['track_id']})")

        extraction = extractor.extract_video_tracked(args.video, person_click=person_click)

        # Gap filling: linear interpolation for ALL gaps (no split!)
        # Visualization requires 1:1 frame correspondence — splitting
        # the array at long gaps breaks frame sync.
        raw_poses = extraction.poses.copy()
        valid_mask = extraction.valid_mask()
        valid_indices = np.where(valid_mask)[0]
        print(
            f"Raw: {len(valid_indices)}/{len(raw_poses)} valid frames, "
            f"first detection at frame {extraction.first_detection_frame}"
        )

        for kp in range(17):
            for ch in range(2):  # x, y only (not confidence)
                vals = raw_poses[:, kp, ch]
                if len(valid_indices) >= 2:
                    raw_poses[:, kp, ch] = np.interp(
                        np.arange(len(vals)), valid_indices, vals[valid_indices]
                    )
                elif len(valid_indices) == 1:
                    raw_poses[:, kp, ch] = vals[valid_indices[0]]

        # Zero confidence for interpolated frames
        interp_mask = ~valid_mask
        raw_poses[interp_mask, :, 2] = 0.0

        poses_norm_raw = raw_poses[:, :, :2]
        confs = raw_poses[:, :, 2]

        print("Smoothing poses (in normalized space)...")
        config = get_skating_optimized_config(fps=meta.fps)
        smoother = PoseSmoother(config=config, freq=meta.fps)
        poses_smoothed_norm = smoother.smooth(poses_norm_raw)
        poses_smoothed_norm = np.clip(poses_smoothed_norm, 0.0, 1.0)

        poses_smoothed_px = poses_smoothed_norm.copy()
        poses_smoothed_px[:, :, 0] *= meta.width
        poses_smoothed_px[:, :, 1] *= meta.height

        poses = np.zeros((len(poses_smoothed_px), 17, 3), dtype=np.float32)
        poses[:, :, :2] = poses_smoothed_px
        poses[:, :, 2] = confs

        pose_frame_indices = np.arange(len(poses))
        poses_viz = poses_smoothed_norm

        n_valid = int(np.sum(~np.isnan(extraction.poses[:, 0, 0])))
        print(f"Extracted {n_valid}/{len(poses)} valid poses (tracked, gap-filled)")

        valid_raw = ~np.isnan(poses_norm_raw[:, 0, 0])
        if np.sum(valid_raw) > 2:
            raw_jitter = np.abs(np.diff(poses_norm_raw[valid_raw][:, :, 0], axis=0)).mean()
            smooth_jitter = np.abs(np.diff(poses_smoothed_norm[:, :, 0], axis=0)).mean()
            if raw_jitter > 0:
                print(f"Jitter reduction: {(1 - smooth_jitter / raw_jitter) * 100:.1f}%")

    # Initialize blade states
    blade_states_left = [None] * len(poses_viz)
    blade_states_right = [None] * len(poses_viz)

    # Initialize 3D pose extraction if requested
    poses_3d = None
    if args.blade_3d and not args.use_3d:
        print("Note: --blade-3d requires 3D poses, auto-enabling --3d")
        args.use_3d = True

    if args.use_3d:
        if args.model_3d and args.model_3d.exists():
            print(f"Loading 3D model: {args.model_3d}")
            from src.pose_3d import AthletePose3DExtractor

            extractor = AthletePose3DExtractor(
                model_path=args.model_3d, model_type="motionagformer-s"
            )
            poses_3d = extractor.extract_sequence(poses_viz)
            print(f"3D poses extracted: {poses_3d.shape}")
        else:
            print("Using biomechanics-based 3D estimation...")
            from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator

            estimator = Biomechanics3DEstimator()
            poses_3d = estimator.estimate_3d(poses_viz)
            print(f"3D poses estimated: {poses_3d.shape}")

    # Calculate fixed auto-scale parameters from reference frame (median frame)
    pip_scale = None
    if poses_3d is not None and not args.no_3d_autoscale:
        ref_frame_idx = len(poses_3d) // 2
        ref_pose_3d = poses_3d[ref_frame_idx]

        center_3d = ref_pose_3d.mean(axis=0)
        ref_pose_3d_centered = ref_pose_3d - center_3d

        ref_pose_2d = project_3d_to_2d(
            ref_pose_3d_centered[np.newaxis, ...],
            width=meta.width // 2,
            height=meta.height // 2,
            camera_distance=args.d_3d_scale,
        )[0]

        x_coords = ref_pose_2d[:, 0]
        y_coords = ref_pose_2d[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range > 1e-6 and y_range > 1e-6:
            scale_x = 1.0 / x_range
            scale_y = 1.0 / y_range
            pip_scale = min(scale_x, scale_y) * 0.9

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            print(
                f"3D PIP auto-scale: scale={pip_scale:.2f}, offset=({x_center:.2f}, {y_center:.2f})"
            )

    # Initialize 3D blade detector if requested
    blade_detector_3d = None
    blade_states_3d_left: list = []
    blade_states_3d_right: list = []
    if args.blade_3d and poses_3d is not None:
        print("Initializing 3D blade detector...")
        blade_detector_3d = BladeEdgeDetector3D(fps=meta.fps)

        for i, pose_3d in enumerate(poses_3d):
            state_left = blade_detector_3d.detect_frame(pose_3d, i, foot="left")
            blade_states_3d_left.append(state_left)

            state_right = blade_detector_3d.detect_frame(pose_3d, i, foot="right")
            blade_states_3d_right.append(state_right)

        print(
            f"3D blade states: {len(blade_states_3d_left)} left, {len(blade_states_3d_right)} right"
        )

        blade_states_left = blade_states_3d_left
        blade_states_right = blade_states_3d_right

        left_breakdown = Counter(s.blade_type.name for s in blade_states_3d_left)
        right_breakdown = Counter(s.blade_type.name for s in blade_states_3d_right)
        print(f"  Left breakdown: {dict(left_breakdown)}")
        print(f"  Right breakdown: {dict(right_breakdown)}")

    # Initialize spatial reference detector
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

    if args.compress:
        import tempfile

        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115
        temp_path = Path(temp_output.name)
        temp_output.close()
        write_path = temp_path
        print(f"Writing to temp file: {temp_path}")
    else:
        write_path = output_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(write_path), fourcc, meta.fps, (meta.width, meta.height))

    # Build layers based on requested level
    layers: list = []
    if args.layer >= 1:
        layers.append(VelocityLayer(scale=5.0, max_length=50))
        layers.append(TrailLayer(length=args.trail_length, joint=H36Key.LFOOT))

    # Pre-create PhysicsEngine for CoM trajectory (avoid per-frame allocation)
    physics_engine = None
    if args.use_3d and args.com_trajectory:
        from src.analysis import PhysicsEngine

        physics_engine = PhysicsEngine(body_mass=60.0)

    # Process frames
    frame_idx = 0
    pose_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find the pose that corresponds to this frame
        current_pose_idx = None
        while pose_idx < len(pose_frame_indices):
            if pose_frame_indices[pose_idx] == frame_idx:
                current_pose_idx = pose_idx
                pose_idx += 1
                break
            elif pose_frame_indices[pose_idx] < frame_idx:
                pose_idx += 1
            else:
                break

        # Build layer context
        context = LayerContext(
            frame_width=meta.width,
            frame_height=meta.height,
            fps=meta.fps,
            frame_idx=frame_idx,
            total_frames=meta.num_frames,
            normalized=True,
        )

        # Layer 0: Skeleton
        if args.layer >= 0 and current_pose_idx is not None:
            if args.use_3d and poses_3d is not None and current_pose_idx < len(poses_3d):
                frame = draw_skeleton_3d_pip(
                    frame,
                    poses_3d[current_pose_idx],
                    meta.width,
                    meta.height,
                )
            else:
                frame = draw_skeleton(frame, poses[current_pose_idx], meta.height, meta.width)

            context.pose_2d = poses_viz[current_pose_idx]
            if poses_3d is not None and current_pose_idx < len(poses_3d):
                context.pose_3d = poses_3d[current_pose_idx]

        # Layers 1+: velocity, trails (rendered via layer system)
        if args.layer >= 1 and current_pose_idx is not None:
            frame = render_layers(frame, layers, context)

            # Draw 3D CoM trajectory if enabled
            if (
                physics_engine is not None
                and poses_3d is not None
                and current_pose_idx < len(poses_3d)
            ):
                com_trajectory = physics_engine.calculate_center_of_mass(
                    poses_3d[: current_pose_idx + 1]
                )

                if len(com_trajectory) > 1:
                    frame = _draw_3d_trajectory(
                        frame, com_trajectory, meta.height, meta.width, camera_z=args.d_3d_scale
                    )

        # Layer 2: trunk tilt indicator
        if args.layer >= 2 and current_pose_idx is not None:
            frame = _draw_axis_indicator(
                frame, poses_viz[current_pose_idx], meta.height, meta.width
            )

        # Spatial reference detection (all layers >= 1)
        if args.layer >= 1:
            camera_pose = spatial_detector.estimate_pose(frame)
            if frame_idx % 100 == 0 and camera_pose.confidence > 0:
                print(
                    f"  Frame {frame_idx}: Roll={camera_pose.roll:.2f}°, Conf={camera_pose.confidence:.2f}, Source={camera_pose.source}"
                )

        # Layer 3: Coaching (subtitles)
        if args.layer >= 3 and subtitle_events:
            current_time = frame_idx / meta.fps
            for event in subtitle_events:
                if event.start_time <= current_time <= (event.end_time or event.start_time + 5):
                    text_parts = []
                    if event.name != "unknown":
                        text_parts.append(event.name.upper())
                    if event.instructions:
                        text_parts.extend(event.instructions)
                    if text_parts:
                        subtitle_text = " | ".join(text_parts)
                        frame = render_cyrillic_text(
                            frame,
                            subtitle_text,
                            (50, meta.height - 50),
                            font_size=args.font_size,
                        )
                    break

        # Draw HUD (all layers) — element info + frame counter + kinematics + blade state
        active_segment = _get_active_segment(segments, frame_idx)
        kinematics = _compute_kinematics(
            poses_viz, current_pose_idx if current_pose_idx is not None else 0, meta.fps
        )

        blade_left = _get_blade_state(blade_states_left, current_pose_idx)
        blade_right = _get_blade_state(blade_states_right, current_pose_idx)

        frame = _draw_hud(
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
            "-i",
            str(write_path),
            "-c:v",
            "libx265",
            "-crf",
            str(args.crf),
            "-preset",
            "medium",
            "-tune",
            "animation",
            "-c:a",
            "copy",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(compress_cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            write_path.unlink()
            original_size = write_path.stat().st_size if write_path.exists() else 0
            compressed_size = output_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            print(
                f"Compression: {original_size // (1024 * 1024)}MB -> {compressed_size // (1024 * 1024)}MB ({ratio:.0f}% reduction)"
            )
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


def _get_blade_state(states: list, idx: int | None) -> BladeState3D | None:
    """Safely get blade state for a frame index."""
    if idx is not None and idx < len(states):
        return states[idx]
    return None


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


def _compute_kinematics(poses: np.ndarray, frame_idx: int, fps: float) -> dict[str, float]:
    """Compute kinematics for this frame."""
    if frame_idx == 0 or frame_idx >= len(poses) - 1:
        return {}

    velocity = (poses[frame_idx + 1] - poses[frame_idx - 1]) * fps / 2
    hip_v = np.linalg.norm(velocity[H36Key.LHIP])

    left_knee = angle_3pt(
        poses[frame_idx, H36Key.LHIP],
        poses[frame_idx, H36Key.LKNEE],
        poses[frame_idx, H36Key.LFOOT],
    )

    right_knee = angle_3pt(
        poses[frame_idx, H36Key.RHIP],
        poses[frame_idx, H36Key.RKNEE],
        poses[frame_idx, H36Key.RFOOT],
    )

    return {
        "hip_velocity": hip_v,
        "left_knee": left_knee,
        "right_knee": right_knee,
    }


def _draw_hud(
    frame: np.ndarray,
    element_info: dict,
    kinematics: dict,
    frame_idx: int,
    total_frames: int,
    fps: float,
    height: int,
    width: int,
    blade_state_left: BladeState3D | None = None,
    blade_state_right: BladeState3D | None = None,
) -> np.ndarray:
    """Draw comprehensive debug HUD.

    Layout:
    - Top-left: Element info (name, boundaries, confidence)
    - Top-left (below element): Blade edge indicators
    - Top-right: Frame counter, timestamp
    - Bottom-left: Kinematics (velocities, angles)
    """
    # Top-left: Element info
    if element_info:
        elem_type = element_info.get("type", "unknown")
        start = element_info.get("start", 0)
        end = element_info.get("end", 0)
        conf = element_info.get("confidence", 0.0)
        elem_text = f"{elem_type} [{start}:{end}] conf={conf:.2f}"
        draw_text_box(frame, elem_text, (10, 30))

    # Top-right: Frame counter
    time_sec = frame_idx / fps
    frame_text = f"Frame: {frame_idx}/{total_frames} | {time_sec:.2f}s"
    (fw, _fh), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    draw_text_box(frame, frame_text, (width - fw - 20, 30))

    # Bottom-left: Kinematics
    y_offset = height - 30
    for key, value in kinematics.items():
        text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        draw_text_box(frame, text, (10, y_offset))
        y_offset -= 25

    # Blade edge indicators (below element info)
    if blade_state_left is not None:
        draw_blade_indicator_hud(frame, blade_state_left, position=(10, 55))
    if blade_state_right is not None:
        draw_blade_indicator_hud(frame, blade_state_right, position=(80, 55))

    return frame


def _draw_axis_indicator(
    frame: np.ndarray, pose: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Draw trunk alignment indicator (hip-to-shoulder angle)."""
    mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
    mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2

    # Convert to pixel coords
    hip_px = (int(mid_hip[0] * width), int(mid_hip[1] * height))
    shoulder_px = (int(mid_shoulder[0] * width), int(mid_shoulder[1] * height))

    # Calculate angle from vertical
    dx = shoulder_px[0] - hip_px[0]
    dy = shoulder_px[1] - hip_px[1]
    angle = np.degrees(np.arctan2(dx, -dy))

    # Draw line from hip to shoulder
    cv2.line(frame, hip_px, shoulder_px, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw angle text
    angle_text = f"Trunk: {angle:+.1f}°"
    cv2.putText(
        frame,
        angle_text,
        (shoulder_px[0] + 10, shoulder_px[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return frame


def _draw_3d_trajectory(
    frame: np.ndarray,
    trajectory: np.ndarray,
    height: int,
    width: int,
    camera_z: float = 3.0,
) -> np.ndarray:
    """Draw 3D CoM trajectory projected onto frame."""
    # Project 3D trajectory to 2D
    traj_2d = project_3d_to_2d(
        trajectory,
        width=width,
        height=height,
        camera_distance=camera_z,
    )

    # Draw trajectory line (yellow)
    if len(traj_2d) > 1:
        pts = [(int(p[0]), int(p[1])) for p in traj_2d]
        for i in range(len(pts) - 1):
            alpha = (i + 1) / len(pts)
            color = (0, int(255 * alpha), int(255 * alpha))  # Fade from dark to yellow
            cv2.line(frame, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    raise SystemExit(main())
