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
from tqdm import tqdm

from src.detection.blade_edge_detector_3d import BladeEdgeDetector3D
from src.detection.spatial_reference import SpatialReferenceDetector
from src.pose_estimation import H36Key
from src.types import BladeState3D
from src.utils.geometry import detect_visible_side, estimate_floor_angle
from src.utils.subtitles import SubtitleParser
from src.utils.video import get_video_meta
from src.utils.video_writer import H264Writer
from src.visualization import (
    draw_blade_indicator_hud,
    project_3d_to_2d,
    put_text,
)
from src.visualization.core.text import draw_text_box, draw_text_outlined


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
        "--3d-blend-threshold",
        dest="blend_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for blending raw vs 3D-corrected 2D poses (default: 0.5)",
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
        "--export",
        action="store_true",
        help="Export poses (NPY) and biomechanics data (CSV) alongside output video",
    )
    parser.add_argument("--poses", type=Path, help="Pre-computed poses .npz file (optional)")
    parser.add_argument("--segments", type=Path, help="Segmentation JSON file (optional)")
    parser.add_argument(
        "--element",
        type=str,
        default=None,
        help="Element type for AI coach overlay (e.g., salchow, waltz_jump, toe_loop)",
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
    parser.add_argument(
        "--pose-backend",
        choices=["rtmlib", "yolo"],
        default="rtmlib",
        help="2D pose estimation backend (default: rtmlib)",
    )
    parser.add_argument(
        "--tracking",
        choices=["auto", "sports2d", "deepsort"],
        default="auto",
        help="Tracking mode: auto (rtmlib built-in), sports2d (rtmlib Sports2D), deepsort (external DeepSORT)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip video rendering entirely (pose extraction only)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-frame timing breakdown",
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

    # --- Person selection (before pose extraction) ---
    from src.types import PersonClick as _PersonClick

    person_click = None
    if args.poses and args.poses.exists():
        pass  # Legacy path below — no person selection needed
    elif args.person_click:
        person_click = _PersonClick(x=args.person_click[0], y=args.person_click[1])
        print(f"Using click point: ({person_click.x}, {person_click.y})")
    elif args.select_person:
        # Need an extractor just for the preview
        from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor

        _preview_extractor = RTMPoseExtractor(
            output_format="normalized",
            conf_threshold=0.3,
            det_frequency=1,
            device="cuda",
            tracking_mode=args.tracking,
        )
        persons, preview_path = _preview_extractor.preview_persons(args.video)
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
            if preview_path:
                import subprocess as _sp

                _sp.run(["xdg-open", preview_path], check=False)

            print(f"\nОбнаружено {len(persons)} человек:")
            for i, p in enumerate(persons):
                print(f"  #{i + 1}: track_id={p['track_id']}, hits={p['hits']}")
            if preview_path:
                print(f"  Превью: {preview_path}")
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

    # --- Unified pose preparation ---
    if args.poses and args.poses.exists():
        # Legacy path: load pre-computed poses from .npz
        print(f"Loading poses from: {args.poses}")
        poses_data = np.load(args.poses)
        raw_poses = poses_data["poses"]
        poses_viz = raw_poses[:, :, :2] if raw_poses.shape[2] == 3 else raw_poses
        poses = raw_poses.copy()
        poses[:, :, 0] *= meta.width
        poses[:, :, 1] *= meta.height
        poses_3d = None
        raw_foot_kps = None
        confs = np.ones_like(poses_viz[:, :, 0])
        prepared = None
    else:
        from src.visualization.pipeline import prepare_poses

        model_3d = args.model_3d
        if model_3d is None:
            default_model = Path("data/models/motionagformer-s-ap3d.onnx")
            if default_model.exists():
                model_3d = default_model
            else:
                print("Warning: 3D model not found.")
                model_3d = None

        prepared = prepare_poses(
            args.video,
            person_click=person_click,
            frame_skip=1,
            tracking=args.tracking,
            use_corrective_lens=True,
            model_3d_path=model_3d,
            blend_threshold=args.blend_threshold,
            device="auto",
        )

        poses_viz = prepared.poses_norm
        poses = prepared.poses_px
        poses_3d = prepared.poses_3d
        raw_foot_kps = prepared.foot_kps
        confs = prepared.confs
        meta = prepared.meta

    print(f"Poses ready: {len(poses_viz)} frames")
    if poses_3d is not None:
        print(f"3D poses: {poses_3d.shape}")

    # Initialize blade states
    blade_states_left = [None] * len(poses_viz)
    blade_states_right = [None] * len(poses_viz)

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

    # Construct visualization pipeline
    from src.visualization.pipeline import VizPipeline

    pipe = VizPipeline(
        meta=meta,
        poses_norm=poses_viz,
        poses_px=poses,
        foot_kps=raw_foot_kps,
        poses_3d=poses_3d,
        layer=args.layer,
        confs=confs,
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

    if args.no_render:
        print("Pose extraction complete. Skipping rendering (--no-render).")
        return 0

    out_w, out_h = meta.width, meta.height

    codec = "libx265" if args.compress else "libx264"
    preset = "medium" if args.compress else "fast"
    writer = H264Writer(
        output_path, out_w, out_h, meta.fps, codec=codec, preset=preset, crf=args.crf
    )

    # Pre-create PhysicsEngine for CoM trajectory (avoid per-frame allocation)
    physics_engine = None
    if args.use_3d and args.com_trajectory:
        from src.analysis import PhysicsEngine

        physics_engine = PhysicsEngine(body_mass=60.0)

    # Process frames
    import time as _time

    frame_idx = 0
    pose_idx = 0
    pbar = tqdm(total=meta.num_frames, desc="Rendering", unit="frame", ncols=100)
    _pt: dict[str, float] = {}
    _pc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _t0 = _time.perf_counter() if args.profile else 0.0

        draw_h, draw_w = meta.height, meta.width

        # Find the pose that corresponds to this frame
        current_pose_idx, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)

        _t_skel = _time.perf_counter() if args.profile else _t0

        # Render skeleton + layers via VizPipeline
        frame, _context = pipe.render_frame(frame, frame_idx, current_pose_idx)

        _t_layers = _time.perf_counter() if args.profile else _t0

        # Draw 3D CoM trajectory if enabled (CLI-specific feature)
        if (
            physics_engine is not None
            and poses_3d is not None
            and current_pose_idx is not None
            and current_pose_idx < len(poses_3d)
        ):
            com_trajectory = physics_engine.calculate_center_of_mass(
                poses_3d[: current_pose_idx + 1]
            )

            if len(com_trajectory) > 1:
                frame = _draw_3d_trajectory(frame, com_trajectory, draw_h, draw_w, camera_z=3.0)

        # Spatial reference detection (every 30 frames — camera doesn't change fast)
        if args.layer >= 1 and frame_idx % 30 == 0:
            camera_pose = spatial_detector.estimate_pose(frame)
            if camera_pose.confidence > 0:
                print(
                    f"  Frame {frame_idx}: Roll={camera_pose.roll:.2f}°, Conf={camera_pose.confidence:.2f}, Source={camera_pose.source}"
                )

        _t_coach = _time.perf_counter() if args.profile else _t0

        # Layer 3: Subtitles
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
                        put_text(
                            frame,
                            subtitle_text,
                            (50, meta.height - 50),
                            font_size=args.font_size,
                            color=(255, 255, 255),
                            bg_color=(0, 0, 0),
                            bg_alpha=0.6,
                        )
                    break

        # Detect visible side and floor angle (Task 6)
        visible_side = None
        floor_angle = 0.0
        if current_pose_idx is not None:
            # Collect foot positions for floor angle
            try:
                l_foot = poses[current_pose_idx][H36Key.LFOOT, :2]
                r_foot = poses[current_pose_idx][H36Key.RFOOT, :2]
                if not (np.isnan(l_foot).any() or np.isnan(r_foot).any()):
                    floor_angle = estimate_floor_angle(np.array([l_foot, r_foot]))
            except (ValueError, IndexError):
                pass

            # Export data collection via VizPipeline
            if args.export:
                pipe.collect_export_data(frame_idx, current_pose_idx, floor_angle=floor_angle)

            # Detect visible side from HALPE26 foot keypoints
            if raw_foot_kps is not None and current_pose_idx < len(raw_foot_kps):
                fk = raw_foot_kps[current_pose_idx]
                if fk is not None and len(fk) >= 6:
                    visible_side = detect_visible_side(fk.reshape(1, 6, 3))

        # Draw HUD (all layers) — element info + frame counter + blade state
        active_segment = _get_active_segment(segments, frame_idx)

        blade_left = _get_blade_state(blade_states_left, current_pose_idx)
        blade_right = _get_blade_state(blade_states_right, current_pose_idx)

        _t_hud = _time.perf_counter() if args.profile else _t0

        frame = _draw_hud(
            frame,
            active_segment,
            draw_h,
            draw_w,
            blade_state_left=blade_left,
            blade_state_right=blade_right,
            visible_side=visible_side,
            floor_angle=floor_angle,
        )

        # Draw frame counter via VizPipeline
        frame = pipe.draw_frame_counter(frame, frame_idx)

        _t_write = _time.perf_counter() if args.profile else _t0

        writer.write(frame)

        if args.profile and _t0:
            _t1 = _time.perf_counter()
            _pc += 1
            for k, t0, t1 in [
                ("read+resize", _t0, _t_skel),
                ("skeleton", _t_skel, _t_layers),
                ("layers", _t_layers, _t_coach),
                ("subtitles+detect", _t_coach, _t_hud),
                ("coach+hud", _t_hud, _t_write),
                ("write", _t_write, _t1),
            ]:
                _pt[k] = _pt.get(k, 0.0) + (t1 - t0)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    if args.profile and _pc > 0:
        print("\n--- Per-frame timing (avg ms) ---")
        for k in sorted(_pt):
            print(f"  {k:20s}: {_pt[k] / _pc * 1000:.1f} ms")
        total = sum(_pt.values())
        print(f"  {'TOTAL':20s}: {total / _pc * 1000:.1f} ms")
    cap.release()
    writer.close()
    print(f"\nSaved to: {output_path}")

    # Export NPY + CSV via VizPipeline
    if args.export:
        export_result = pipe.save_exports(output_path)
        if export_result["poses_path"]:
            print(f"Poses saved: {export_result['poses_path']}")
        if export_result["csv_path"]:
            print(f"Biomechanics saved: {export_result['csv_path']}")

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


def _draw_hud(
    frame: np.ndarray,
    element_info: dict,
    height: int,
    width: int,
    blade_state_left: BladeState3D | None = None,
    blade_state_right: BladeState3D | None = None,
    visible_side: str | None = None,
    floor_angle: float = 0.0,
) -> np.ndarray:
    """Draw minimal HUD.

    Layout:
    - Top-left: Element info (name, boundaries, confidence)
    - Top-left (below element): Blade edge indicators
    - Bottom-left: Visible side + floor angle
    - Bottom-left: Frame counter (drawn by VizPipeline.draw_frame_counter)

    Note: Frame counter is now handled by VizPipeline.draw_frame_counter().
    """
    # Top-left: Element info
    if element_info:
        elem_type = element_info.get("type", "unknown")
        start = element_info.get("start", 0)
        end = element_info.get("end", 0)
        conf = element_info.get("confidence", 0.0)
        elem_text = f"{elem_type} [{start}:{end}] conf={conf:.2f}"
        draw_text_box(frame, elem_text, (10, 30))

    # Blade edge indicators (below element info)
    if blade_state_left is not None:
        draw_blade_indicator_hud(frame, blade_state_left, position=(10, 55))
    if blade_state_right is not None:
        draw_blade_indicator_hud(frame, blade_state_right, position=(80, 55))

    # Bottom-left: Visible side + floor angle (frame counter drawn by VizPipeline)
    side_str = visible_side or "N/A"
    bottom_line = f"Side: {side_str}  Floor: {floor_angle:.1f} deg"
    info_y = height - 40
    draw_text_outlined(
        frame,
        bottom_line,
        (10, info_y),
        font_scale=0.45,
        thickness=1,
    )

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
