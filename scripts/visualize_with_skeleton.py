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
import contextlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.analysis.angles import compute_joint_angles
from src.detection.blade_edge_detector_3d import BladeEdgeDetector3D
from src.detection.spatial_reference import SpatialReferenceDetector
from src.pose_estimation import H36Key, H36MExtractor
from src.types import BladeState3D
from src.utils.geometry import detect_visible_side, estimate_floor_angle
from src.utils.subtitles import SubtitleParser
from src.utils.video import get_video_meta
from src.visualization import (
    JointAngleLayer,
    LayerContext,
    TrailLayer,
    VelocityLayer,
    VerticalAxisLayer,
    draw_blade_indicator_hud,
    draw_skeleton,
    project_3d_to_2d,
    render_cyrillic_text,
    render_layers,
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
        "--render-scale",
        type=float,
        default=1.0,
        help="Downscale factor for rendering (0.5 = half resolution, 4x faster). Default: 1.0",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip video rendering entirely (pose extraction only)",
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
    raw_foot_kps = None
    confs: np.ndarray | None = None
    if args.poses and args.poses.exists():
        print(f"Loading poses from: {args.poses}")
        poses_data = np.load(args.poses)
        poses = poses_data["poses"]
        pose_frame_indices = np.arange(len(poses))
        poses_viz = poses[:, :, :2] if poses.shape[2] == 3 else poses
    else:
        backend_label = (
            "rtmlib BodyWithFeet" if args.pose_backend == "rtmlib" else "YOLO26-Pose + OC-SORT"
        )
        print(f"Extracting poses with tracking ({backend_label})...")
        if args.pose_backend == "rtmlib":
            from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor

            extractor = RTMPoseExtractor(
                output_format="normalized",
                conf_threshold=0.3,
                det_frequency=1,  # detect every frame — GPU is fast enough
                device="cuda",
                tracking_mode=args.tracking,
            )
        else:
            extractor = H36MExtractor(
                model_size="s",  # small model — better accuracy for distant/small skaters
                conf_threshold=0.1,  # low threshold — detect distant skaters
                output_format="normalized",
                crop_enhance=False,  # ROI crop — enable when CUDA available (slow on CPU)
            )

        # Person selection
        from src.types import PersonClick as _PersonClick

        person_click = None
        if args.person_click:
            person_click = _PersonClick(x=args.person_click[0], y=args.person_click[1])
            print(f"Using click point: ({person_click.x}, {person_click.y})")
        elif args.select_person:
            persons, preview_path = extractor.preview_persons(args.video)
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
                # Show visual preview if available
                if preview_path:
                    import subprocess as _sp

                    _sp.run(["xdg-open", preview_path], check=False)

                print(f"\nОбнаружено {len(persons)} человек. Смотри превью.")
                if preview_path:
                    print(f"  {preview_path}")
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

        raw_poses = extraction.poses  # (N, 17, 3) — 1:1 with video frames
        raw_foot_kps = extraction.foot_keypoints  # (N, 6, 3) normalized
        n_valid = int(extraction.valid_mask().sum())
        print(f"Raw: {n_valid}/{len(raw_poses)} valid frames")

        poses_norm = raw_poses[:, :, :2].copy()
        confs = raw_poses[:, :, 2].copy()

        poses_viz = poses_norm
        poses = raw_poses.copy()
        poses[:, :, 0] *= meta.width
        poses[:, :, 1] *= meta.height

        pose_frame_indices = np.arange(len(poses))
        print(f"Poses ready: {len(poses)} frames")

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

    # 3D corrective lens: lift → constrain → project → blend
    poses_viz_corrected = None
    if args.use_3d and poses_3d is not None:
        from src.pose_3d import CorrectiveLens

        lens = CorrectiveLens(model_path=args.model_3d if args.model_3d else None)
        poses_viz_corrected, _ = lens.correct_sequence(
            poses_2d_norm=poses_viz,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            confidences=confs,
            blend_threshold=args.blend_threshold,
        )
        # Clip to valid range
        poses_viz_corrected = np.clip(poses_viz_corrected, 0.0, 1.0)
        print(f"3D-corrected poses: {poses_viz_corrected.shape}")

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

    if args.no_render:
        print("Pose extraction complete. Skipping rendering (--no-render).")
        return 0

    # Apply render scale for faster processing
    render_scale = args.render_scale
    out_w = int(meta.width * render_scale)
    out_h = int(meta.height * render_scale)
    if render_scale != 1.0:
        print(f"Render scale: {render_scale} ({out_w}x{out_h})")

    if args.compress:
        import subprocess

        compress_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{out_w}x{out_h}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(meta.fps),
            "-i",
            "-",
            "-c:v",
            "libx265",
            "-crf",
            str(args.crf),
            "-preset",
            "medium",
            "-tune",
            "animation",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        writer = subprocess.Popen(
            compress_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Writing directly to H265 (CRF={args.crf})...")
    else:
        write_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(write_path), fourcc, meta.fps, (out_w, out_h))

    # Build layers based on requested level
    layers: list = []
    if args.layer >= 1:
        layers.append(VelocityLayer(scale=3.0, max_length=30, color_mode="solid"))
        layers.append(
            TrailLayer(length=args.trail_length, joint=H36Key.LFOOT, width=1, color=(200, 80, 80))
        )
    if args.layer >= 2:
        layers.append(JointAngleLayer(show_degree_labels=False))
        layers.append(VerticalAxisLayer())

    # Pre-create PhysicsEngine for CoM trajectory (avoid per-frame allocation)
    physics_engine = None
    if args.use_3d and args.com_trajectory:
        from src.analysis import PhysicsEngine

        physics_engine = PhysicsEngine(body_mass=60.0)

    # Process frames
    frame_idx = 0
    pose_idx = 0
    pbar = tqdm(total=meta.num_frames, desc="Rendering", unit="frame", ncols=100)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale for rendering if requested
        if render_scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            draw_h, draw_w = out_h, out_w
        else:
            draw_h, draw_w = meta.height, meta.width

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

        # Build layer context (use original dimensions for normalized coords)
        context = LayerContext(
            frame_width=draw_w,
            frame_height=draw_h,
            fps=meta.fps,
            frame_idx=frame_idx,
            total_frames=meta.num_frames,
            normalized=True,
        )

        # Layer 0: Skeleton — always use raw rtmlib (no 3D jitter)
        if args.layer >= 0 and current_pose_idx is not None:
            foot_kp = raw_foot_kps[current_pose_idx] if raw_foot_kps is not None else None
            frame = draw_skeleton(
                frame,
                poses[current_pose_idx],
                draw_h,
                draw_w,
                line_width=1,
                joint_radius=3,
                foot_keypoints=foot_kp,
            )
            context.pose_2d = poses_viz[current_pose_idx]
            if poses_3d is not None and current_pose_idx < len(poses_3d):
                context.pose_3d = poses_3d[current_pose_idx]

            # Compute angles for angle panel (Task 7: 26 biomechanics angles)
            if args.layer >= 2:
                joint_angles = compute_joint_angles(poses_viz[current_pose_idx])
                context.custom_data["angles"] = joint_angles

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
                    frame = _draw_3d_trajectory(frame, com_trajectory, draw_h, draw_w, camera_z=3.0)

        # Spatial reference detection (every 30 frames — camera doesn't change fast)
        if args.layer >= 1 and frame_idx % 30 == 0:
            camera_pose = spatial_detector.estimate_pose(frame)
            if camera_pose.confidence > 0:
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

            # Detect visible side from HALPE26 foot keypoints
            if raw_foot_kps is not None and current_pose_idx < len(raw_foot_kps):
                fk = raw_foot_kps[current_pose_idx]
                if fk is not None and len(fk) >= 6:
                    visible_side = detect_visible_side(fk.reshape(1, 6, 3))

        # Draw HUD (all layers) — element info + frame counter + blade state
        active_segment = _get_active_segment(segments, frame_idx)

        blade_left = _get_blade_state(blade_states_left, current_pose_idx)
        blade_right = _get_blade_state(blade_states_right, current_pose_idx)

        frame = _draw_hud(
            frame,
            active_segment,
            frame_idx,
            meta.num_frames,
            meta.fps,
            draw_h,
            draw_w,
            blade_state_left=blade_left,
            blade_state_right=blade_right,
            visible_side=visible_side,
            floor_angle=floor_angle,
        )

        if args.compress:
            try:
                writer.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                break
        else:
            writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if args.compress:
        with contextlib.suppress(BrokenPipeError, OSError):
            writer.stdin.close()
        writer.wait()
        if writer.returncode != 0:
            print(f"FFmpeg error: {writer.stderr.read().decode()}")
            return 1
        print(f"Saved to: {output_path}")
    else:
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
    frame_idx: int,
    total_frames: int,
    fps: float,
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
    - Top-right: Frame counter, timestamp
    """
    # Top-left: Element info
    if element_info:
        elem_type = element_info.get("type", "unknown")
        start = element_info.get("start", 0)
        end = element_info.get("end", 0)
        conf = element_info.get("confidence", 0.0)
        elem_text = f"{elem_type} [{start}:{end}] conf={conf:.2f}"
        draw_text_box(frame, elem_text, (10, 30))

    # Top-right: Frame counter + timestamp
    time_sec = frame_idx / fps
    minutes = int(time_sec) // 60
    seconds = time_sec % 60
    ms = int((seconds % 1) * 100)
    frame_text = f"{frame_idx}/{total_frames}  {minutes:02d}:{int(seconds):02d}.{ms:02d}"
    (fw, _fh), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    draw_text_box(frame, frame_text, (width - fw - 20, 10), font_scale=0.5)

    # Blade edge indicators (below element info)
    if blade_state_left is not None:
        draw_blade_indicator_hud(frame, blade_state_left, position=(10, 55))
    if blade_state_right is not None:
        draw_blade_indicator_hud(frame, blade_state_right, position=(80, 55))

    # Bottom-left: Visible side + floor angle (Task 6)
    info_y = height - 40
    side_str = visible_side or "N/A"
    draw_text_outlined(
        frame,
        f"Side: {side_str}  Floor: {floor_angle:.1f} deg",
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
