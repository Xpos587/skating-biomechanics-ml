"""Unified visualization pipeline for CLI and Gradio frontends.

Encapsulates shared logic: pose normalization, layer construction,
per-frame rendering, and data export. Callers provide video I/O.
"""

from __future__ import annotations

import csv as _csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.utils.video import get_video_meta
from src.visualization import (
    LayerContext,
    VerticalAxisLayer,
    draw_skeleton,
    render_layers,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.visualization.layers.base import Frame


@dataclass
class VizPipeline:
    """Shared visualization pipeline state and rendering logic.

    Callers (CLI, Gradio) construct this with prepared pose data,
    then call ``render_frame()`` in their video loop.

    Args:
        meta: Video metadata object with ``width``, ``height``, ``fps``, ``num_frames``.
        poses_norm: Normalized [0,1] poses (N, 17, 2).
        poses_px: Pixel-coordinate poses (N, 17, 2 or 17, 3). Used for skeleton drawing.
        foot_kps: Foot keypoints (N, 6, 3) normalized, or None.
        poses_3d: 3D poses (N, 17, 3) or None.
        layer: HUD layer level (0-3).
        confs: Per-keypoint confidence (N, 17) or None.
        frame_indices: Frame index mapping (N,). Defaults to ``np.arange(N)``.
    """

    meta: object
    poses_norm: NDArray[np.float32]
    poses_px: NDArray[np.float32] | None = None
    foot_kps: NDArray[np.float32] | None = None
    poses_3d: NDArray[np.float32] | None = None
    layer: int = 0
    confs: NDArray[np.float32] | None = None
    frame_indices: NDArray[np.intp] | None = None

    # Internal state
    layers: list = field(default_factory=list, init=False)
    export_frames: list[int] = field(default_factory=list, init=False)
    export_timestamps: list[float] = field(default_factory=list, init=False)
    export_floor_angles: list[float] = field(default_factory=list, init=False)
    export_joint_angles: list[dict] = field(default_factory=list, init=False)
    export_poses: list[NDArray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        n = len(self.poses_norm)
        if self.frame_indices is None:
            self.frame_indices = np.arange(n)
        if self.poses_px is None:
            w, h = self.meta.width, self.meta.height
            self.poses_px = self.poses_norm.copy()
            self.poses_px[:, :, 0] *= w
            self.poses_px[:, :, 1] *= h
        self.build_layers()

    def build_layers(self) -> None:
        """Construct visualization layers based on ``self.layer`` level."""
        self.layers = []
        if self.layer >= 2:
            self.layers.append(VerticalAxisLayer())

    def add_ml_layers(self, ml_layers: list) -> None:
        """Add ML-generated layers to the pipeline."""
        self.layers.extend(ml_layers)

    def render_frame(
        self,
        frame: Frame,
        frame_idx: int,
        pose_idx: int | None,
    ) -> tuple[Frame, LayerContext]:
        """Render one frame with skeleton, layers.

        Args:
            frame: BGR image to draw on (modified in place).
            frame_idx: Current video frame index.
            pose_idx: Index into poses_norm/poses_px, or None if no pose.

        Returns:
            Tuple of (modified frame, LayerContext).
        """
        w, h = self.meta.width, self.meta.height
        total = self.meta.num_frames

        context = LayerContext(
            frame_width=w,
            frame_height=h,
            fps=self.meta.fps,
            frame_idx=frame_idx,
            total_frames=total,
            normalized=True,
        )

        # Skeleton (always drawn when pose available)
        if pose_idx is not None and pose_idx < len(self.poses_norm):
            skel_pose = self.poses_px[pose_idx].copy()
            foot_kp = self.foot_kps[pose_idx].copy() if self.foot_kps is not None else None
            frame = draw_skeleton(
                frame,
                skel_pose,
                h,
                w,
                line_width=1,
                joint_radius=3,
                foot_keypoints=foot_kp,
            )
            context.pose_2d = self.poses_norm[pose_idx]
            if self.poses_3d is not None and pose_idx < len(self.poses_3d):
                context.pose_3d = self.poses_3d[pose_idx]

        # Layers 1+: velocity, trails, joint angles, axis
        if self.layer >= 1 and pose_idx is not None:
            frame = render_layers(frame, self.layers, context)

        return frame, context

    def draw_frame_counter(
        self,
        frame: Frame,
        frame_idx: int,
    ) -> Frame:
        """Draw frame counter and timestamp at bottom-left."""
        from src.visualization.core.text import draw_text_outlined

        w, h = self.meta.width, self.meta.height
        total = self.meta.num_frames
        fps = self.meta.fps

        time_sec = frame_idx / fps
        minutes = int(time_sec) // 60
        seconds = time_sec % 60
        ms = int((seconds % 1) * 100)
        frame_text = f"{frame_idx}/{total}  {minutes:02d}:{int(seconds):02d}.{ms:02d}"

        info_y = h - 40
        draw_text_outlined(frame, frame_text, (10, info_y - 25), font_scale=0.45, thickness=1)
        return frame

    def collect_export_data(
        self,
        frame_idx: int,
        pose_idx: int | None,
        floor_angle: float = 0.0,
    ) -> None:
        """Collect data for NPY + CSV export."""
        if pose_idx is None:
            return
        from src.analysis.angles import compute_joint_angles

        self.export_frames.append(frame_idx)
        self.export_timestamps.append(round(frame_idx / self.meta.fps, 3))
        self.export_floor_angles.append(round(floor_angle, 2))
        ja = compute_joint_angles(self.poses_norm[pose_idx])
        self.export_joint_angles.append(ja)
        if self.poses_px is not None:
            self.export_poses.append(self.poses_px[pose_idx].copy())

    def save_exports(self, output_path: Path) -> dict[str, str | None]:
        """Save NPY poses and CSV biomechanics data alongside output video.

        Returns:
            Dict with ``poses_path`` and ``csv_path`` keys (None if no data).
        """
        if not self.export_poses:
            return {"poses_path": None, "csv_path": None}

        out_dir = output_path.parent
        stem = output_path.stem

        poses_path = out_dir / f"{stem}_poses.npy"
        np.save(str(poses_path), np.array(self.export_poses))

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
        header = ["frame", "timestamp_s", "floor_angle_deg", *angle_keys]
        with csv_path.open("w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(header)
            for idx in range(len(self.export_frames)):
                ja = self.export_joint_angles[idx]
                row = [
                    self.export_frames[idx],
                    self.export_timestamps[idx],
                    self.export_floor_angles[idx],
                    *(round(ja.get(k, float("nan")), 1) for k in angle_keys),
                ]
                writer.writerow(row)

        return {"poses_path": str(poses_path), "csv_path": str(csv_path)}

    def find_pose_idx(self, frame_idx: int, pose_idx: int) -> tuple[int | None, int]:
        """Advance pose_idx to match frame_idx.

        Returns:
            Tuple of (current_pose_idx_or_None, next_pose_idx).
        """
        while pose_idx < len(self.frame_indices):
            if self.frame_indices[pose_idx] == frame_idx:
                return pose_idx, pose_idx + 1
            elif self.frame_indices[pose_idx] < frame_idx:
                pose_idx += 1
            else:
                break
        return None, pose_idx


# ------------------------------------------------------------------
# Unified pose preparation
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_3D_CANDIDATES = [
    _PROJECT_ROOT / "data" / "models" / "motionagformer-s-ap3d.onnx",
    Path("data/models/motionagformer-s-ap3d.onnx"),
]

# Module-level imports for testability (mock.patch targets module-level names).
from src.pose_3d import CorrectiveLens  # noqa: E402
from src.pose_3d.onnx_extractor import ONNXPoseExtractor  # noqa: E402
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor  # noqa: E402


@dataclass
class PreparedPoses:
    """Output of the unified pose preparation pipeline.

    Constructed by ``prepare_poses()`` and consumed by ``VizPipeline``.
    """

    poses_norm: NDArray[np.float32]  # (N, 17, 2) corrected normalized [0,1]
    poses_px: NDArray[np.float32]  # (N, 17, 3) pixel coords (x, y, confidence)
    poses_3d: NDArray[np.float32] | None  # (N, 17, 3) 3D poses for GLB export
    foot_kps: NDArray[np.float32] | None  # (N, 6, 3) foot keypoints or None
    confs: NDArray[np.float32]  # (N, 17) per-keypoint confidence
    frame_indices: NDArray[np.intp]  # (N,) frame index mapping
    meta: object  # video metadata (width, height, fps, num_frames)
    n_valid: int  # valid (non-interpolated) frames
    n_total: int  # total video frames


def _resolve_model_3d(path: Path | str | None = None) -> Path | None:
    """Find the 3D pose model.

    Args:
        path: Explicit path, or None to auto-detect.

    Returns:
        Path to model file, or None if not found.
    """
    if path is not None:
        p = Path(path)
        return p if p.exists() else None
    for candidate in _DEFAULT_MODEL_3D_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def prepare_poses(
    video_path: Path | str,
    person_click: object | None = None,
    *,
    frame_skip: int = 1,
    tracking: str = "auto",
    use_corrective_lens: bool = False,
    model_3d_path: Path | str | None = None,
    blend_threshold: float = 0.5,
    device: str = "auto",
    progress_cb=None,
) -> PreparedPoses:
    """Unified pose preparation pipeline.

    Extract 2D poses -> fill gaps -> smooth -> 3D lift + CorrectiveLens.

    Both CLI and Gradio call this function. Any improvement here
    automatically applies to all frontends.

    Args:
        video_path: Path to input video.
        person_click: PersonClick to select target person, or None.
        frame_skip: Process every Nth frame (default 1 = every frame).
        tracking: Tracking mode ("auto", "sports2d", "deepsort").
        use_corrective_lens: Apply 3D-corrected 2D overlay (default True).
        model_3d_path: Path to 3D model, or None to auto-detect.
        blend_threshold: CorrectiveLens confidence blend threshold.
        device: Device string ("auto", "cuda", "cpu").
        progress_cb: Optional callback ``(progress_0_to_1, message)``.

    Returns:
        PreparedPoses with all data needed for VizPipeline construction.
    """
    from src.device import DeviceConfig

    video_path = Path(video_path)
    meta = get_video_meta(video_path)
    cfg = DeviceConfig(device=device)

    if progress_cb:
        progress_cb(0.0, "Extracting poses...")

    # --- Step 1: Extract 2D poses ---
    extractor = RTMPoseExtractor(
        output_format="normalized",
        conf_threshold=0.3,
        det_frequency=max(1, frame_skip),
        frame_skip=frame_skip,
        device=cfg.device,
        tracking_mode=tracking,
    )
    extraction = extractor.extract_video_tracked(
        str(video_path),
        person_click=person_click,
        progress_cb=progress_cb,
    )

    raw_poses = extraction.poses  # (N, 17, 3) — may have NaN from frame_skip
    raw_foot_kps = extraction.foot_keypoints
    frame_indices = extraction.frame_indices

    nan_mask = np.isnan(raw_poses[:, 0, 0])
    n_valid = int((~nan_mask).sum())

    # --- Step 2: Fill NaN gaps (linear interp, preserves array length) ---
    # GapFiller splits at long gaps, breaking 1:1 frame mapping.
    # Use plain np.interp — safe for all gap sizes, keeps N frames.
    if nan_mask.any() and n_valid >= 2:
        valid_indices = np.where(~nan_mask)[0]
        n_frames = len(raw_poses)
        for kp in range(raw_poses.shape[1]):
            for dim in range(raw_poses.shape[2]):
                raw_poses[:, kp, dim] = np.interp(
                    np.arange(n_frames),
                    valid_indices,
                    raw_poses[valid_indices, kp, dim],
                )
        logger.info(
            "Filled %d NaN frame(s) via linear interpolation (%d valid)",
            int(nan_mask.sum()),
            n_valid,
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

    poses_norm = raw_poses[:, :, :2].copy()
    confs = raw_poses[:, :, 2].copy()

    if progress_cb:
        progress_cb(0.4, "3D pose estimation...")

    # --- Step 4: 3D lift + CorrectiveLens ---
    poses_3d = None
    model_path = _resolve_model_3d(model_3d_path)

    if model_path is not None and use_corrective_lens:
        lens = CorrectiveLens(model_path=model_path, device=cfg.device)
        poses_norm_corrected, poses_3d = lens.correct_sequence(
            poses_2d_norm=poses_norm,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            confidences=confs,
            blend_threshold=blend_threshold,
        )
        poses_norm = np.clip(poses_norm_corrected, 0.0, 1.0)
        logger.info("CorrectiveLens applied (blend_threshold=%.2f)", blend_threshold)
    elif model_path is not None:
        onnx = ONNXPoseExtractor(model_path, device=cfg.device)
        poses_3d = onnx.estimate_3d(poses_norm)
        logger.info("3D poses estimated (no CorrectiveLens)")
    else:
        logger.warning("No 3D model found. Skeleton will use raw 2D poses without correction.")

    # --- Step 5: Build pixel coordinates from FINAL poses_norm ---
    poses_px = np.zeros((*poses_norm.shape[:2], 3), dtype=np.float32)
    poses_px[:, :, 0] = poses_norm[:, :, 0] * meta.width
    poses_px[:, :, 1] = poses_norm[:, :, 1] * meta.height
    poses_px[:, :, 2] = confs

    if progress_cb:
        progress_cb(0.6, "Poses ready.")

    return PreparedPoses(
        poses_norm=poses_norm,
        poses_px=poses_px,
        poses_3d=poses_3d,
        foot_kps=raw_foot_kps,
        confs=confs,
        frame_indices=frame_indices,
        meta=meta,
        n_valid=n_valid,
        n_total=meta.num_frames,
    )
