"""Pure helper functions for the Gradio UI.

All functions are stateless and testable without a running Gradio server.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.device import DeviceConfig
from src.types import PersonClick
from src.utils.video import get_video_meta
from src.utils.video_writer import H264Writer

if TYPE_CHECKING:
    from numpy.typing import NDArray


def match_click_to_person(
    persons: list[dict],
    x: float,
    y: float,
) -> dict | None:
    """Match a normalized click coordinate to the closest person bbox."""
    if not persons:
        return None

    best: dict | None = None
    best_dist = float("inf")

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            mx, my = p["mid_hip"]
            dist = (x - mx) ** 2 + (y - my) ** 2
            if dist < best_dist:
                best_dist = dist
                best = p

    return best


def render_person_preview(
    frame: NDArray[np.uint8],
    persons: list[dict],
    selected_idx: int | None = None,
) -> NDArray[np.uint8]:
    """Draw numbered bounding boxes for each detected person."""
    if not persons:
        return frame

    annotated = frame.copy()
    h, w = frame.shape[:2]

    colors = [
        (255, 165, 0),  # Blue (OpenCV BGR)
        (0, 200, 200),  # Yellow
        (200, 100, 0),  # Cyan
        (200, 0, 200),  # Magenta
        (0, 180, 255),  # Orange
    ]

    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["bbox"]
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        if selected_idx is not None and i == selected_idx:
            color = (0, 255, 0)  # Green for selected
            thickness = 3
        else:
            color = colors[i % len(colors)]
            thickness = 2

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thickness)

        label = f"#{i + 1} (hits: {p['hits']})"
        cv2.rectangle(annotated, (px1, py1 - 28), (px1 + len(label) * 10 + 10, py1), color, -1)
        cv2.putText(
            annotated,
            label,
            (px1 + 5, py1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def persons_to_choices(persons: list[dict]) -> list[str]:
    """Convert person list to Gradio Radio choices."""
    return [
        f"Person #{i + 1} ({p['hits']} hits, track {p['track_id']})" for i, p in enumerate(persons)
    ]


def choice_to_person_click(
    choice: str,
    persons: list[dict],
    width: int,
    height: int,
) -> PersonClick:
    """Convert a Gradio Radio selection to a PersonClick."""
    idx = int(choice.split("#")[1].split(" ", maxsplit=1)[0]) - 1
    mid_hip = persons[idx]["mid_hip"]
    return PersonClick(
        x=int(mid_hip[0] * width),
        y=int(mid_hip[1] * height),
    )


def process_video_pipeline(
    video_path: str | Path,
    person_click: PersonClick | None,
    frame_skip: int,
    layer: int,
    tracking: str,
    use_3d: bool,
    render_scale: float,
    blade_3d: bool,
    export: bool,
    output_path: str | Path,
    progress_cb=None,
) -> dict:
    """Run the full visualization pipeline (mirrors visualize_with_skeleton.py)."""
    from src.analysis.angles import compute_joint_angles
    from src.pose_estimation import H36Key
    from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
    from src.utils.smoothing import PoseSmoother, get_skating_optimized_config
    from src.visualization import (
        JointAngleLayer,
        LayerContext,
        TrailLayer,
        VelocityLayer,
        VerticalAxisLayer,
        draw_skeleton,
        render_layers,
    )

    video_path = Path(video_path) if isinstance(video_path, str) else video_path
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    meta = get_video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if progress_cb:
        progress_cb(0.0, "Extracting poses...")

    extractor = RTMPoseExtractor(
        output_format="normalized",
        conf_threshold=0.3,
        det_frequency=frame_skip,
        frame_skip=frame_skip,
        device=DeviceConfig.default().device,
        tracking_mode=tracking,
    )
    extraction = extractor.extract_video_tracked(
        str(video_path), person_click=person_click, progress_cb=progress_cb
    )

    raw_poses = extraction.poses
    raw_foot_kps = extraction.foot_keypoints

    # Map pose indices to real video frame indices
    # extractor returns poses only for processed frames (every frame_skip-th)
    # but frame_indices are sequential [0..N-1], not actual video frame numbers
    n_poses = len(raw_poses)
    if n_poses > 0 and n_poses < meta.num_frames:
        # Real frame indices when frame_skip > 1
        pose_frame_indices = np.arange(n_poses) * frame_skip
        pose_frame_indices = np.clip(pose_frame_indices, 0, meta.num_frames - 1)
    else:
        pose_frame_indices = extraction.frame_indices
    n_valid = int(extraction.valid_mask().sum())

    poses_norm = raw_poses[:, :, :2].copy()
    confs = raw_poses[:, :, 2].copy()
    poses = raw_poses.copy()
    poses[:, :, 0] *= meta.width
    poses[:, :, 1] *= meta.height

    if len(poses_norm) > 2:
        smooth_config = get_skating_optimized_config(meta.fps)
        smoother = PoseSmoother(smooth_config, freq=meta.fps)
        poses_norm = smoother.smooth(poses_norm)

    poses_viz = poses_norm

    poses_3d = None
    if use_3d:
        from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator

        estimator = Biomechanics3DEstimator()
        poses_3d = estimator.estimate_3d(poses_viz)

    # Always estimate 3D poses for 3D viewer (lightweight Biomechanics3DEstimator)
    if poses_3d is None:
        from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator

        estimator = Biomechanics3DEstimator()
        poses_3d = estimator.estimate_3d(poses_viz)

    if progress_cb:
        progress_cb(0.3, "Poses extracted. Rendering...")

    out_w = int(meta.width * render_scale)
    out_h = int(meta.height * render_scale)
    writer = H264Writer(output_path, out_w, out_h, meta.fps)

    layers: list = []
    if layer >= 1:
        layers.append(VelocityLayer(scale=3.0, max_length=30, color_mode="solid"))
        layers.append(TrailLayer(length=20, joint=H36Key.LFOOT, width=1, color=(200, 80, 80)))
        layers.append(JointAngleLayer())
    if layer >= 2:
        layers.append(VerticalAxisLayer())

    export_frames: list[int] = []
    export_timestamps: list[float] = []
    export_floor_angles: list[float] = []
    export_joint_angles: list[dict[str, float]] = []
    export_poses_list: list[np.ndarray] = []

    frame_idx = 0
    pose_idx = 0
    total = meta.num_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if render_scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            draw_h, draw_w = out_h, out_w
        else:
            draw_h, draw_w = meta.height, meta.width

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

        context = LayerContext(
            frame_width=draw_w,
            frame_height=draw_h,
            fps=meta.fps,
            frame_idx=frame_idx,
            total_frames=total,
            normalized=True,
        )

        if layer >= 0 and current_pose_idx is not None:
            foot_kp = raw_foot_kps[current_pose_idx] if raw_foot_kps is not None else None
            # Scale pixel poses to match downscaled frame
            skel_pose = poses[current_pose_idx]
            skel_foot_kp = foot_kp
            if render_scale != 1.0:
                skel_pose = skel_pose * render_scale
                skel_foot_kp = foot_kp * render_scale if foot_kp is not None else None
            frame = draw_skeleton(
                frame,  # type: ignore[arg-type]
                skel_pose,
                draw_h,
                draw_w,
                line_width=1,
                joint_radius=3,
                foot_keypoints=skel_foot_kp,
            )
            context.pose_2d = poses_viz[current_pose_idx]
            if poses_3d is not None and current_pose_idx < len(poses_3d):
                context.pose_3d = poses_3d[current_pose_idx]

        if layer >= 1 and current_pose_idx is not None:
            frame = render_layers(frame, layers, context)  # type: ignore[arg-type]

        minutes = int(frame_idx / meta.fps) // 60
        seconds = int(frame_idx / meta.fps) % 60
        ms = int((frame_idx / meta.fps - int(frame_idx / meta.fps)) * 100)
        frame_text = f"{frame_idx}/{total}  {minutes:02d}:{seconds:02d}.{ms:02d}"
        from src.visualization.core.text import draw_text_box

        draw_text_box(frame, frame_text, (draw_w - 220, 10), font_scale=0.5)  # type: ignore[arg-type]

        if export and current_pose_idx is not None:
            export_frames.append(frame_idx)
            export_timestamps.append(round(frame_idx / meta.fps, 3))
            export_floor_angles.append(0.0)
            ja = compute_joint_angles(poses_viz[current_pose_idx])
            export_joint_angles.append(ja)
            export_poses_list.append(poses[current_pose_idx].copy())

        writer.write(frame)
        frame_idx += 1

        if progress_cb and frame_idx % 50 == 0:
            progress_cb(0.3 + 0.65 * frame_idx / total, f"Rendering frame {frame_idx}/{total}")

    cap.release()
    writer.close()

    if progress_cb:
        progress_cb(0.95, "Saving exports...")

    poses_path = None
    csv_path = None
    if export and export_poses_list:
        out_dir = output_path.parent  # type: ignore[attr-defined]
        stem = output_path.stem  # type: ignore[attr-defined]

        poses_path = out_dir / f"{stem}_poses.npy"
        np.save(str(poses_path), np.array(export_poses_list))

        csv_path = out_dir / f"{stem}_biomechanics.csv"
        angle_keys = [
            "R Ankle",
            "L Ankle",
            "R Knee",
            "L Knee",
            "R Hip",
            "L Hip",
            "R Shoulder",
            "L Shoulder",
            "R Elbow",
            "L Elbow",
            "R Wrist",
            "L Wrist",
        ]
        header = [*["frame", "timestamp_s", "floor_angle_deg"], *angle_keys]
        with csv_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(header)
            for idx in range(len(export_frames)):
                ja = export_joint_angles[idx]
                row = [
                    export_frames[idx],
                    export_timestamps[idx],
                    export_floor_angles[idx],
                    *(round(ja.get(k, float("nan")), 1) for k in angle_keys),
                ]
                w.writerow(row)

    return {
        "video_path": str(output_path),
        "poses_path": str(poses_path) if poses_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "poses_3d": poses_3d,
        "stats": {
            "total_frames": total,
            "valid_frames": n_valid,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
        },
    }
