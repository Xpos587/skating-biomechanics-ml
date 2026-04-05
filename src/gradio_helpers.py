"""Pure helper functions for the Gradio UI.

All functions are stateless and testable without a running Gradio server.
"""

from __future__ import annotations

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
    blade_3d: bool,
    export: bool,
    output_path: str | Path,
    progress_cb=None,
) -> dict:
    """Run the full visualization pipeline (mirrors visualize_with_skeleton.py)."""
    from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
    from src.utils.smoothing import PoseSmoother, get_skating_optimized_config

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

    # Interpolate NaN frames in-place (frame_skip leaves gaps)
    # Must keep array length = num_frames for 1:1 frame mapping
    nan_mask = np.isnan(raw_poses[:, 0, 0])
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        valid_indices = np.where(~nan_mask)[0]
        for kp in range(raw_poses.shape[1]):
            for dim in range(raw_poses.shape[2]):
                raw_poses[:, kp, dim] = np.interp(
                    np.arange(len(raw_poses)),
                    valid_indices,
                    raw_poses[valid_indices, kp, dim],
                )
        if raw_foot_kps is not None:
            foot_nan = np.isnan(raw_foot_kps[:, 0, 0])
            if foot_nan.any() and (~foot_nan).sum() >= 2:
                foot_valid = np.where(~foot_nan)[0]
                for kp in range(raw_foot_kps.shape[1]):
                    for dim in range(raw_foot_kps.shape[2]):
                        raw_foot_kps[:, kp, dim] = np.interp(
                            np.arange(len(raw_foot_kps)),
                            foot_valid,
                            raw_foot_kps[foot_valid, kp, dim],
                        )

    # All frames are now filled — frame indices map 1:1
    n_poses = len(raw_poses)
    pose_frame_indices = extraction.frame_indices
    n_valid = int((~nan_mask).sum())

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
        from src.pose_3d.onnx_extractor import ONNXPoseExtractor

        onnx_model = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "models"
            / "motionagformer-s-ap3d.onnx"
        )
        if onnx_model.exists():
            cfg = DeviceConfig.default()
            extractor = ONNXPoseExtractor(onnx_model, device=cfg.device)
            poses_3d = extractor.estimate_3d(poses_viz)
        else:
            from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator

            estimator = Biomechanics3DEstimator()
            poses_3d = estimator.estimate_3d(poses_viz)

    if progress_cb:
        progress_cb(0.3, "Poses extracted. Rendering...")

    # --- Build pipeline ---
    from src.visualization.pipeline import VizPipeline

    pipe = VizPipeline(
        meta=meta,
        poses_norm=poses_norm,
        poses_px=poses,
        foot_kps=raw_foot_kps,
        poses_3d=poses_3d,
        layer=layer,
        confs=confs,
        frame_indices=pose_frame_indices,
    )

    writer = H264Writer(output_path, meta.width, meta.height, meta.fps)

    # --- Render loop ---
    frame_idx = 0
    pose_idx = 0
    total = meta.num_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_pose_idx, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)
        frame, _ = pipe.render_frame(frame, frame_idx, current_pose_idx)

        pipe.draw_frame_counter(frame, frame_idx)

        if export:
            pipe.collect_export_data(frame_idx, current_pose_idx)

        writer.write(frame)
        frame_idx += 1

        if progress_cb and frame_idx % 50 == 0:
            progress_cb(0.3 + 0.65 * frame_idx / total, f"Rendering frame {frame_idx}/{total}")

    cap.release()
    writer.close()

    if progress_cb:
        progress_cb(0.95, "Saving exports...")

    export_result = (
        pipe.save_exports(output_path) if export else {"poses_path": None, "csv_path": None}
    )

    return {
        "video_path": str(output_path),
        "poses_path": export_result["poses_path"],
        "csv_path": export_result["csv_path"],
        "poses_3d": poses_3d,
        "stats": {
            "total_frames": total,
            "valid_frames": n_valid,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
        },
    }
