"""Dual-video comparison renderer.

Provides side-by-side and overlay comparison of athlete vs reference videos
with configurable overlay layers (axis, angles, timer, skeleton, etc.).

Based on Kinovea-style sports video analysis workflow.

Optimizations:
- FFmpeg libx264 pipe for fast encoding
- Pre-allocated output buffers (no np.hstack per frame)
- Cached sorted layers and reused LayerContext
- Streaming decode (constant memory)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.pose_estimation import RTMPoseExtractor
from src.utils.smoothing import PoseSmoother, get_skating_optimized_config
from src.utils.video import get_video_meta
from src.visualization import draw_skeleton
from src.visualization.config import COLOR_MAGENTA
from src.visualization.layers.base import LayerContext
from src.visualization.layers.joint_angle_layer import JointAngleLayer
from src.visualization.layers.skeleton_layer import SkeletonLayer
from src.visualization.layers.timer_layer import TimerLayer
from src.visualization.layers.vertical_axis_layer import VerticalAxisLayer

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ComparisonMode(Enum):
    """Comparison rendering mode."""

    SIDE_BY_SIDE = "side-by-side"
    OVERLAY = "overlay"


@dataclass
class ComparisonConfig:
    """Configuration for dual-video comparison rendering."""

    mode: ComparisonMode = ComparisonMode.SIDE_BY_SIDE
    overlays: list[str] = field(default_factory=lambda: ["skeleton", "axis", "angles", "timer"])
    resize_width: int = 1280
    reference_color: tuple[int, int, int] = COLOR_MAGENTA
    reference_alpha: float = 0.4
    divider_width: int = 4
    fps: float = 0.0
    compress: bool = False
    crf: int = 30
    max_frames: int = 0
    start_frame: int = 0
    device: str = "0"
    no_cache: bool = False


def _build_layers(overlays: list[str]) -> list:
    """Build visualization layers from overlay name list."""
    layers = []
    layer_map = {
        "skeleton": SkeletonLayer,
        "axis": VerticalAxisLayer,
        "angles": JointAngleLayer,
        "timer": TimerLayer,
    }
    for name in overlays:
        cls = layer_map.get(name)
        if cls:
            layers.append(cls())
    return layers


class ComparisonRenderer:
    """Renders dual-video comparison with overlays."""

    def __init__(self, config: ComparisonConfig | None = None):
        self.config = config or ComparisonConfig()
        self.layers = _build_layers(self.config.overlays)
        self._sorted_layers = sorted(self.layers, key=lambda ly: ly.z_index)

    def _create_extractor(self, device: str) -> RTMPoseExtractor:
        """Create RTMPoseExtractor with GPU fallback to CPU."""
        dev = "cuda" if device not in ("cpu", "") else "cpu"
        try:
            print(f"  Trying device: {device} (GPU)...", flush=True)
            extractor = RTMPoseExtractor(conf_threshold=0.3, device=dev)
            return extractor
        except Exception as exc:
            logger.warning("GPU failed: %s", exc)
            print(f"  WARNING: GPU failed ({exc}). Falling back to CPU.", flush=True)
            return RTMPoseExtractor(conf_threshold=0.3, device="cpu")

    # -- Pose caching --------------------------------------------------------

    @staticmethod
    def _pose_cache_path(video_path: Path) -> Path:
        """Return cache path for extracted poses (next to video)."""
        return video_path.with_name(f"{video_path.stem}_poses.npz")

    def _save_pose_cache(self, video_path: Path, poses: list[np.ndarray]) -> None:
        """Save extracted poses to .npz cache."""
        if not poses:
            return
        cache_path = self._pose_cache_path(video_path)
        arr = np.stack(poses)
        np.savez_compressed(cache_path, poses=arr)
        print(f"  Cached {len(poses)} poses -> {cache_path}", flush=True)

    def _load_pose_cache(self, video_path: Path, expected_frames: int) -> list[np.ndarray] | None:
        """Load poses from cache if valid.

        Returns None if cache missing, stale, or no_cache=True.
        """
        if self.config.no_cache:
            return None
        cache_path = self._pose_cache_path(video_path)
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path)
            poses_arr = data["poses"]
            # Allow ±10% frame tolerance (resize/normalization may differ)
            if abs(len(poses_arr) - expected_frames) > max(expected_frames * 0.1, 5):
                print(
                    f"  Cache stale ({len(poses_arr)} vs {expected_frames} frames), re-extracting",
                    flush=True,
                )
                return None
            poses = [poses_arr[i] for i in range(len(poses_arr))]
            print(
                f"  Loaded {len(poses)} poses from cache: {cache_path}",
                flush=True,
            )
            return poses
        except Exception:
            return None

    def process(
        self,
        athlete_video: Path,
        reference_video: Path,
        output_path: Path,
        _element_type: str = "three_turn",
    ) -> None:
        """Process two videos and generate comparison output."""
        athlete_meta = get_video_meta(athlete_video)
        reference_meta = get_video_meta(reference_video)

        fps = self.config.fps or athlete_meta.fps
        target_w = self.config.resize_width

        # Calculate resize dimensions (even)
        a_h = int(athlete_meta.height * target_w / athlete_meta.width)
        a_h = a_h if a_h % 2 == 0 else a_h - 1
        r_h = int(reference_meta.height * target_w / reference_meta.width)
        r_h = r_h if r_h % 2 == 0 else r_h - 1

        # Output dimensions
        if self.config.mode == ComparisonMode.SIDE_BY_SIDE:
            out_w = target_w * 2 + self.config.divider_width
            out_h = max(a_h, r_h)
        else:
            out_w = target_w
            out_h = a_h

        max_frames = self.config.max_frames or int(
            max(athlete_meta.num_frames, reference_meta.num_frames)
        )

        print(
            f"Athlete: {athlete_meta.width}x{athlete_meta.height} -> {target_w}x{a_h}",
            flush=True,
        )
        print(
            f"Reference: {reference_meta.width}x{reference_meta.height} -> {target_w}x{r_h}",
            flush=True,
        )
        print(f"Output: {out_w}x{out_h} @ {fps:.1f}fps", flush=True)
        print(
            f"Mode: {self.config.mode.value}, Overlays: {self.config.overlays}",
            flush=True,
        )
        print(f"Processing up to {max_frames} frames...", flush=True)

        # Stage 1: Extract poses (with caching)
        device = self.config.device
        extractor = self._create_extractor(device)

        # Try cache first for athlete
        print("Extracting athlete poses...", flush=True)
        athlete_poses = self._load_pose_cache(athlete_video, max_frames)
        if athlete_poses is None:
            athlete_poses = self._extract_poses_streaming(
                athlete_video,
                extractor,
                target_w,
                a_h,
                max_frames=max_frames,
                start_frame=self.config.start_frame,
            )
            self._save_pose_cache(athlete_video, athlete_poses)
        print(f"  Got {len(athlete_poses)} athlete poses", flush=True)

        # Try cache first for reference
        print("Extracting reference poses...", flush=True)
        ref_poses = self._load_pose_cache(reference_video, max_frames)
        if ref_poses is None:
            ref_poses = self._extract_poses_streaming(
                reference_video,
                extractor,
                target_w,
                r_h,
                max_frames=max_frames,
                start_frame=self.config.start_frame,
            )
            self._save_pose_cache(reference_video, ref_poses)
        print(f"  Got {len(ref_poses)} reference poses", flush=True)

        # Handle empty poses
        if len(athlete_poses) == 0 and len(ref_poses) == 0:
            print(
                "WARNING: No poses detected in either video. "
                "Outputting raw video without overlays.",
                flush=True,
            )
        elif len(athlete_poses) == 0:
            print(
                "WARNING: No poses detected in athlete video. "
                "Only reference skeleton will be drawn.",
                flush=True,
            )
        elif len(ref_poses) == 0:
            print(
                "WARNING: No poses detected in reference video. "
                "Only athlete overlays will be drawn.",
                flush=True,
            )

        # Stage 2: Smooth poses (only if we have enough)
        smooth_config = get_skating_optimized_config(fps)
        smoother = PoseSmoother(smooth_config, freq=fps)
        if len(athlete_poses) > 2:
            athlete_poses = smoother.smooth(np.stack(athlete_poses))
        elif len(athlete_poses) > 0:
            athlete_poses = np.stack(athlete_poses)
            print(
                "  Athlete: too few poses to smooth, using raw",
                flush=True,
            )
        if len(ref_poses) > 2:
            ref_poses = smoother.smooth(np.stack(ref_poses))
        elif len(ref_poses) > 0:
            ref_poses = np.stack(ref_poses)
            print(
                "  Reference: too few poses to smooth, using raw",
                flush=True,
            )

        # Determine actual render frame count
        render_frames = min(
            len(athlete_poses) if len(athlete_poses) > 0 else max_frames,
            len(ref_poses) if len(ref_poses) > 0 else max_frames,
            max_frames,
        )
        if render_frames <= 0:
            print("ERROR: No frames to render. Aborting.", flush=True)
            return

        print(f"Rendering {render_frames} frames...", flush=True)

        # Stage 3: Render (streaming) -- re-open captures for frame decoding
        encoder = self._create_encoder(str(output_path), out_w, out_h, fps)

        cap_a = cv2.VideoCapture(str(athlete_video))
        cap_r = cv2.VideoCapture(str(reference_video))

        if cap_a is None or not cap_a.isOpened():
            print(f"ERROR: Cannot open athlete video: {athlete_video}", flush=True)
            encoder.stdin.close()
            encoder.wait()
            return
        if cap_r is None or not cap_r.isOpened():
            print(
                f"ERROR: Cannot open reference video: {reference_video}",
                flush=True,
            )
            cap_a.release()
            encoder.stdin.close()
            encoder.wait()
            return

        # Seek to start frame
        if self.config.start_frame > 0:
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)

        # Pre-allocate buffers for side-by-side
        if self.config.mode == ComparisonMode.SIDE_BY_SIDE:
            out_buf = np.empty((out_h, out_w, 3), dtype=np.uint8)
            divider = np.full((out_h, self.config.divider_width, 3), 128, dtype=np.uint8)
            pad_a = np.zeros((out_h - a_h, target_w, 3), dtype=np.uint8) if a_h < out_h else None
            pad_r = np.zeros((out_h - r_h, target_w, 3), dtype=np.uint8) if r_h < out_h else None

        # Reuse LayerContext
        ctx = LayerContext(
            frame_width=target_w,
            frame_height=a_h,
            fps=fps,
            frame_idx=0,
            total_frames=render_frames,
            pose_2d=None,
            normalized=True,
        )

        frame_a_last: np.ndarray | None = None
        frame_r_last: np.ndarray | None = None

        for frame_idx in range(render_frames):
            ret_a, frame_a = cap_a.read()
            ret_r, frame_r = cap_r.read()

            if not ret_a and not ret_r:
                print(
                    f"  Both videos ended at frame {frame_idx}, stopping.",
                    flush=True,
                )
                break

            # Use last frame if video ended
            if not ret_a and frame_a_last is not None:
                frame_a = frame_a_last
            if not ret_r and frame_r_last is not None:
                frame_r = frame_r_last
            if ret_a:
                frame_a_last = frame_a.copy()
            if ret_r:
                frame_r_last = frame_r.copy()

            # Resize
            frame_a = cv2.resize(frame_a, (target_w, a_h))
            frame_r = cv2.resize(frame_r, (target_w, r_h))

            # Get poses (use last pose if beyond available poses)
            pose_a = (
                (athlete_poses[frame_idx] if frame_idx < len(athlete_poses) else athlete_poses[-1])
                if len(athlete_poses) > 0
                else None
            )

            pose_r = (
                (ref_poses[frame_idx] if frame_idx < len(ref_poses) else ref_poses[-1])
                if len(ref_poses) > 0
                else None
            )

            # Render skeleton + overlays on athlete frame
            ctx.frame_idx = frame_idx
            ctx.pose_2d = pose_a
            ctx.pose_3d = None
            if pose_a is not None:
                draw_skeleton(frame_a, pose_a, a_h, target_w)
            if pose_a is not None and self._sorted_layers:
                for layer in self._sorted_layers:
                    if layer.is_visible():
                        frame_a = layer.render(frame_a, ctx)

            # Render skeleton + overlays on reference frame
            ctx.pose_2d = pose_r
            if pose_r is not None:
                draw_skeleton(frame_r, pose_r, r_h, target_w)
            if pose_r is not None and self._sorted_layers:
                for layer in self._sorted_layers:
                    if layer.is_visible():
                        frame_r = layer.render(frame_r, ctx)

            # Compose output
            if self.config.mode == ComparisonMode.SIDE_BY_SIDE:
                # Assemble side-by-side buffer
                if pad_a is not None:
                    out_buf[:a_h, :target_w] = frame_a
                    out_buf[a_h:, :target_w] = pad_a
                else:
                    out_buf[:, :target_w] = frame_a

                out_buf[:, target_w : target_w + self.config.divider_width] = divider

                if pad_r is not None:
                    out_buf[:r_h, target_w + self.config.divider_width :] = frame_r
                    out_buf[r_h:, target_w + self.config.divider_width :] = pad_r
                else:
                    out_buf[:, target_w + self.config.divider_width :] = frame_r

                encoder.stdin.write(out_buf.data)
            else:
                # Overlay mode: blend reference skeleton onto athlete frame
                overlay = frame_a.copy()
                if pose_r is not None:
                    draw_skeleton(overlay, pose_r, a_h, target_w)
                frame_a = cv2.addWeighted(overlay, self.config.reference_alpha, frame_a, 1.0, 0)
                encoder.stdin.write(frame_a.data)

            if frame_idx % 200 == 0:
                print(f"  Frame {frame_idx}/{render_frames}", flush=True)

        cap_a.release()
        cap_r.release()
        encoder.stdin.close()
        encoder.wait()
        print(f"Done! Output: {output_path}", flush=True)

    def _create_encoder(
        self, output_path: str, width: int, height: int, fps: float
    ) -> subprocess.Popen:
        """Create FFmpeg encoder process (libx264 ultrafast)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(self.config.crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _extract_poses_streaming(
        self,
        video_path: Path,
        extractor: RTMPoseExtractor,
        target_w: int,
        target_h: int,
        max_frames: int = 0,
        start_frame: int = 0,
    ) -> list[np.ndarray]:
        """Extract poses from video using RTMPoseExtractor."""
        result = extractor.extract_video_tracked(str(video_path))
        poses_3d = result.poses  # (N, 17, 3) normalized

        if max_frames > 0:
            poses_3d = poses_3d[:max_frames]
        if start_frame > 0:
            poses_3d = poses_3d[start_frame:]

        # Convert to list of (17, 2) arrays
        poses: list[np.ndarray] = []
        for i in range(len(poses_3d)):
            pose_2d = poses_3d[i, :, :2]
            poses.append(pose_2d)

        return poses
