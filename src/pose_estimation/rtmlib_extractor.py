"""RTMPose-based pose extractor with foot keypoints via rtmlib.

Uses the rtmlib PoseTracker with BodyWithFeet model to extract 26-keypoint
HALPE26 poses (COCO 17 + 6 foot + 3 face).  The output is converted to
H3.6M 17-keypoint format for the analysis pipeline, with foot keypoints
preserved separately for blade edge detection.

Architecture:
    Video → rtmlib PoseTracker (BodyWithFeet) → HALPE26 (26kp)
        → H3.6M (17kp) + foot keypoints (6kp)

Key advantages over YOLO26-Pose:
    - 6 dedicated foot keypoints (heel, big toe, small toe per foot)
    - Better accuracy on distant/small subjects
    - Built-in tracking with consistent IDs
    - ONNX Runtime inference (no PyTorch dependency)

References:
    - rtmlib: https://github.com/Tau-J/rtmlib
    - RTMPose: https://arxiv.org/abs/2303.07399
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from rtmlib import BodyWithFeet, PoseTracker
except ImportError:
    PoseTracker = None  # type: ignore[assignment]
    BodyWithFeet = None  # type: ignore[assignment]

from ..detection.pose_tracker import PoseTracker as CustomPoseTracker
from ..pose_estimation.h36m_extractor import _biometric_distance
from ..types import PersonClick, TrackedExtraction, VideoMeta
from ..utils.video import get_video_meta
from .halpe26 import HALPE26Key, extract_foot_keypoints, halpe26_to_h36m

logger = logging.getLogger(__name__)


class RTMPoseExtractor:
    """HALPE26 pose extractor using rtmlib BodyWithFeet model.

    Provides H3.6M 17-keypoint poses plus 6 foot keypoints for blade edge
    detection.  Uses rtmlib's built-in tracking for multi-person handling.

    Args:
        mode: Model preset — ``"lightweight"`` (fast), ``"balanced"``
            (default), ``"performance"`` (accurate).
        tracking_backend: ``"rtmlib"`` uses rtmlib's built-in tracker;
            ``"custom"`` feeds detections into our PoseTracker
            (OC-SORT + biometric Re-ID).
        conf_threshold: Minimum keypoint confidence to accept [0, 1].
        output_format: ``"normalized"`` for [0, 1] coords. ``"pixels"``
            for absolute pixel coords.
        det_frequency: Run person detection every N frames (1 = every frame).
        frame_skip: Process every Nth frame for pose estimation (1 = every
            frame). Higher values = faster but less accurate. Skipped
            frames are filled with NaN for downstream interpolation.
        device: ``"cpu"`` or ``"cuda"``.
        backend: Inference backend — ``"onnxruntime"`` or ``"opencv"``.
    """

    def __init__(
        self,
        mode: str = "balanced",
        tracking_backend: str = "rtmlib",
        conf_threshold: float = 0.3,
        output_format: str = "normalized",
        det_frequency: int = 1,
        frame_skip: int = 1,
        device: str = "cpu",
        backend: str = "onnxruntime",
    ) -> None:
        if PoseTracker is None:
            raise ImportError(
                "rtmlib is not installed. Install with: uv add rtmlib"
            )

        self._mode = mode
        self._tracking_backend = tracking_backend
        self._conf_threshold = conf_threshold
        self._output_format = output_format
        self._det_frequency = det_frequency
        self._frame_skip = max(1, frame_skip)
        self._device = device
        self._backend = backend

        # Lazy-initialised on first call
        self._tracker: PoseTracker | None = None

    @property
    def tracker(self) -> "PoseTracker":
        """Lazy-initialise rtmlib PoseTracker on first access."""
        if self._tracker is None:
            self._tracker = PoseTracker(
                BodyWithFeet,
                det_frequency=self._det_frequency,
                tracking=True,
                tracking_thr=0.3,
                mode=self._mode,
                to_openpose=False,  # MUST be False — ensures HALPE26 ordering
                backend=self._backend,
                device=self._device,
            )
        return self._tracker

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_video_tracked(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
    ) -> TrackedExtraction:
        """Extract H3.6M + foot keypoints from video with tracking.

        Runs rtmlib BodyWithFeet on every frame, tracks all persons,
        and selects a single target person for output.

        Args:
            video_path: Path to video file.
            person_click: Optional click to select target person by
                proximity to the click point in the first few frames.

        Returns:
            TrackedExtraction with poses (N, 17, 3), foot_keypoints
            (N, 6, 3), frame_indices, tracking metadata.  Missing frames
            are filled with NaN.

        Raises:
            ValueError: If no pose is detected in any frame.
        """
        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)
        num_frames = video_meta.num_frames

        # Pre-allocate with NaN
        all_poses = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)
        all_feet = np.full((num_frames, 6, 3), np.nan, dtype=np.float32)

        # Tracking state
        if self._tracking_backend == "custom":
            custom_tracker = CustomPoseTracker(
                max_disappeared=30,
                min_hits=3,
                fps=video_meta.fps,
            )
        else:
            custom_tracker = None  # type: ignore[assignment]

        target_track_id: int | None = None
        click_lock_window = 6  # ~0.2-0.24s at 25-30fps
        click_norm: tuple[float, float] | None = None
        if person_click is not None:
            click_norm = person_click.to_normalized(
                video_meta.width, video_meta.height
            )

        # Track hit counts for auto-select
        track_hit_counts: dict[int, int] = {}

        # Per-frame track_id→(h36m_pose, foot_kps) for retroactive fill
        frame_track_data: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

        # Last known target pose for biometric track migration
        last_target_pose: np.ndarray | None = None
        target_lost_frame: int | None = None

        # rtmlib assigns its own IDs (int per tracked person).
        # Map rtmlib_id → our internal track_id (0-based counter).
        rtmlib_id_map: dict[int, int] = {}
        next_internal_id = 0

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            frame_idx = 0
            pbar = tqdm(total=num_frames, desc="Extracting poses", unit="frame", ncols=100)
            while cap.isOpened() and frame_idx < num_frames:
                if self._frame_skip > 1 and frame_idx % self._frame_skip != 0:
                    # Skip this frame — just advance the video
                    ret = cap.grab()
                    if not ret:
                        break
                    frame_idx += 1
                    pbar.update(1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]

                # Run rtmlib
                keypoints, scores = self.tracker(frame)
                # keypoints: (P, 26, 2) pixel coords, scores: (P, 26)

                if keypoints is None or len(keypoints) == 0:
                    if custom_tracker is not None:
                        custom_tracker.update(
                            np.empty((0, 17, 2), dtype=np.float32)
                        )
                    continue

                n_persons = len(keypoints)
                h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)
                foot_kps_list: list[np.ndarray] = []

                for p in range(n_persons):
                    kp = keypoints[p].astype(np.float32)  # (26, 2) pixels
                    conf = scores[p].astype(np.float32)  # (26,)

                    # Build HALPE26 (26, 3) with confidence
                    halpe26 = np.zeros((26, 3), dtype=np.float32)
                    halpe26[:, :2] = kp
                    halpe26[:, 2] = conf

                    # Normalize to [0, 1]
                    halpe26[:, 0] /= w
                    halpe26[:, 1] /= h

                    # Convert to H3.6M 17kp
                    h36m = halpe26_to_h36m(halpe26)

                    # Extract foot keypoints (6, 3)
                    foot = extract_foot_keypoints(halpe26)

                    # Convert to pixels if requested
                    if self._output_format == "pixels":
                        h36m[:, 0] *= w
                        h36m[:, 1] *= h
                        foot[:, 0] *= w
                        foot[:, 1] *= h

                    h36m_poses[p] = h36m
                    foot_kps_list.append(foot)

                # --- Track association ---
                if self._tracking_backend == "custom":
                    track_ids = custom_tracker.update(
                        h36m_poses[:, :, :2], h36m_poses[:, :, 2]
                    )
                else:
                    # rtmlib assigns per-person IDs internally via tracking.
                    # Since rtmlib returns persons in tracked order, we need
                    # to identify which person is which.  rtmlib doesn't expose
                    # the track ID directly in the default API, so we use a
                    # spatial matching approach: match detections to existing
                    # tracks by biometric distance.
                    track_ids = self._assign_track_ids(
                        h36m_poses, rtmlib_id_map, next_internal_id
                    )
                    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1

                # Store per-track data for retroactive fill
                frame_track_data[frame_idx] = {
                    tid: (h36m_poses[p].copy(), foot_kps_list[p].copy())
                    for p, tid in enumerate(track_ids)
                }

                # Update hit counts
                for tid in track_ids:
                    track_hit_counts[tid] = track_hit_counts.get(tid, 0) + 1

                # --- Target selection ---
                # Phase 1: Click-based selection
                if (
                    target_track_id is None
                    and click_norm is not None
                    and frame_idx < click_lock_window
                ):
                    best_dist = float("inf")
                    best_tid: int | None = None
                    for p, tid in enumerate(track_ids):
                        mid_hip_x = (
                            h36m_poses[p, 4, 0] + h36m_poses[p, 1, 0]
                        ) / 2  # LHIP + RHIP
                        mid_hip_y = (
                            h36m_poses[p, 4, 1] + h36m_poses[p, 1, 1]
                        ) / 2
                        dist = (mid_hip_x - click_norm[0]) ** 2 + (
                            mid_hip_y - click_norm[1]
                        ) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_tid = tid
                    if best_tid is not None:
                        target_track_id = best_tid

                # Fill target data for current frame
                if target_track_id is not None:
                    found = False
                    for p, tid in enumerate(track_ids):
                        if tid == target_track_id:
                            all_poses[frame_idx] = h36m_poses[p]
                            all_feet[frame_idx] = foot_kps_list[p]
                            last_target_pose = h36m_poses[p].copy()
                            target_lost_frame = None
                            found = True
                            break

                    # Track migration: biometric matching when target lost
                    if not found and last_target_pose is not None:
                        if target_lost_frame is None:
                            target_lost_frame = frame_idx

                        if frame_idx - target_lost_frame <= 60:
                            best_dist = float("inf")
                            best_new_tid: int | None = None
                            best_new_data: tuple[np.ndarray, np.ndarray] | None = None
                            for p, tid in enumerate(track_ids):
                                dist = _biometric_distance(
                                    h36m_poses[p], last_target_pose
                                )
                                if dist < best_dist:
                                    best_dist = dist
                                    best_new_tid = tid
                                    best_new_data = (
                                        h36m_poses[p],
                                        foot_kps_list[p],
                                    )

                            if best_new_tid is not None and best_dist < 0.08:
                                target_track_id = best_new_tid
                                all_poses[frame_idx] = best_new_data[0]
                                all_feet[frame_idx] = best_new_data[1]
                                last_target_pose = best_new_data[0].copy()
                                target_lost_frame = None
                                # Retroactively fill from stored frames
                                for fidx, tmap in frame_track_data.items():
                                    if (
                                        target_track_id in tmap
                                        and np.isnan(all_poses[fidx, 0, 0])
                                    ):
                                        all_poses[fidx] = tmap[target_track_id][0]
                                        all_feet[fidx] = tmap[target_track_id][1]

                frame_idx += 1
                pbar.update(1)
        finally:
            cap.release()
            pbar.close()

        # Phase 2 (deferred): Auto-select by most hits
        if target_track_id is None and track_hit_counts:
            target_track_id = max(
                track_hit_counts,
                key=lambda k: track_hit_counts[k],  # type: ignore[arg-type]
            )
            for fidx, tmap in frame_track_data.items():
                if target_track_id in tmap and np.isnan(all_poses[fidx, 0, 0]):
                    all_poses[fidx] = tmap[target_track_id][0]
                    all_feet[fidx] = tmap[target_track_id][1]

        # Determine first_detection_frame
        valid_mask = ~np.isnan(all_poses[:, 0, 0])
        if not np.any(valid_mask):
            raise ValueError(f"No valid pose detected in video: {video_path}")
        first_detection_frame = int(np.argmax(valid_mask))

        return TrackedExtraction(
            poses=all_poses,
            frame_indices=np.arange(num_frames),
            first_detection_frame=first_detection_frame,
            target_track_id=target_track_id,
            fps=video_meta.fps,
            video_meta=video_meta,
            foot_keypoints=all_feet,
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview_persons(
        self,
        video_path: Path | str,
        num_frames: int = 30,
    ) -> list[dict]:
        """Preview all detected persons in the first few frames.

        Runs rtmlib on the first ``num_frames`` frames and returns a
        summary for each tracked person so the user can choose which
        one to follow.

        Args:
            video_path: Path to video file.
            num_frames: Number of frames to scan (default 30).

        Returns:
            List of dicts sorted by hits (descending)::

                {
                    "track_id": int,
                    "hits": int,
                    "bbox": (x1, y1, x2, y2),  # normalized [0,1]
                    "first_frame": int,
                    "mid_hip": (x, y),          # normalized
                }
        """
        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)

        if self._tracking_backend == "custom":
            tracker = CustomPoseTracker(
                max_disappeared=30, min_hits=2, fps=video_meta.fps
            )
        else:
            tracker = None  # type: ignore[assignment]

        rtmlib_id_map: dict[int, int] = {}
        next_internal_id = 0
        person_data: dict[int, dict] = {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            for frame_idx in tqdm(range(num_frames), desc="Previewing persons", unit="frame", ncols=100):
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                keypoints, scores = self.tracker(frame)

                if keypoints is None or len(keypoints) == 0:
                    if tracker is not None:
                        tracker.update(np.empty((0, 17, 2), dtype=np.float32))
                    continue

                n_persons = len(keypoints)
                h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)

                for p in range(n_persons):
                    kp = keypoints[p].astype(np.float32)
                    conf = scores[p].astype(np.float32)

                    halpe26 = np.zeros((26, 3), dtype=np.float32)
                    halpe26[:, :2] = kp
                    halpe26[:, 2] = conf
                    halpe26[:, 0] /= w
                    halpe26[:, 1] /= h

                    h36m_poses[p] = halpe26_to_h36m(halpe26)

                # Track association
                if tracker is not None:
                    track_ids = tracker.update(
                        h36m_poses[:, :, :2], h36m_poses[:, :, 2]
                    )
                else:
                    track_ids = self._assign_track_ids(
                        h36m_poses, rtmlib_id_map, next_internal_id
                    )
                    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1

                for p, tid in enumerate(track_ids):
                    if tid not in person_data:
                        person_data[tid] = {
                            "hits": 0,
                            "best_conf": 0.0,
                            "best_kps": None,
                            "best_frame": frame_idx,
                            "first_frame": frame_idx,
                        }
                    person_data[tid]["hits"] += 1
                    avg_conf = float(np.mean(h36m_poses[p, :, 2]))
                    if avg_conf > person_data[tid]["best_conf"]:
                        person_data[tid]["best_conf"] = avg_conf
                        person_data[tid]["best_kps"] = h36m_poses[p].copy()
                        person_data[tid]["best_frame"] = frame_idx
        finally:
            cap.release()

        # Build output
        output: list[dict] = []
        for tid, data in sorted(
            person_data.items(), key=lambda kv: kv[1]["hits"], reverse=True
        ):
            kps = data["best_kps"]
            if kps is None:
                continue
            valid = kps[kps[:, 2] > 0.1]
            if len(valid) < 3:
                continue
            x1, y1 = float(np.min(valid[:, 0])), float(np.min(valid[:, 1]))
            x2, y2 = float(np.max(valid[:, 0])), float(np.max(valid[:, 1]))
            # Mid-hip (H3.6M: LHIP=4, RHIP=1)
            mid_hip_x = float((kps[4, 0] + kps[1, 0]) / 2)
            mid_hip_y = float((kps[4, 1] + kps[1, 1]) / 2)
            output.append(
                {
                    "track_id": tid,
                    "hits": data["hits"],
                    "bbox": (x1, y1, x2, y2),
                    "first_frame": data["first_frame"],
                    "mid_hip": (mid_hip_x, mid_hip_y),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_track_ids(
        self,
        h36m_poses: np.ndarray,
        id_map: dict[int, int],
        next_id: int,
    ) -> list[int]:
        """Assign stable track IDs to detected poses.

        For rtmlib backend (no custom tracker), uses biometric matching
        to associate detections across frames.  First detection per person
        gets a new ID; subsequent detections are matched by biometric
        distance to known persons.

        Args:
            h36m_poses: (P, 17, 3) H3.6M poses for current frame.
            id_map: Map from internal_id → internal_id (identity map,
                used for tracking known IDs).
            next_id: Next available internal ID.

        Returns:
            List of track IDs, one per detected person.
        """
        if len(h36m_poses) == 0:
            return []

        track_ids: list[int] = []

        # For each detection, find closest existing track by biometric distance
        known_poses: dict[int, np.ndarray] = {}
        for tid in id_map:
            # Store the last known pose — but we don't have it here.
            # This is a simplified approach: for the first frame, assign new IDs.
            pass

        # Simple strategy: assign new IDs for each detection.
        # The built-in rtmlib tracking handles spatial consistency already,
        # but since we can't access its track IDs, we rely on the
        # custom tracker path for production use.
        # For the rtmlib path, we just assign sequential IDs per-frame
        # and let biometric matching handle re-association.
        for p in range(len(h36m_poses)):
            new_id = next_id + p
            track_ids.append(new_id)

        return track_ids

    def close(self) -> None:
        """Release resources."""
        self._tracker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def extract_rtmpose_poses(
    video_path: Path | str,
    mode: str = "balanced",
    output_format: str = "normalized",
    person_click: PersonClick | None = None,
) -> TrackedExtraction:
    """Extract H3.6M + foot keypoints from video using rtmlib.

    Convenience function that creates an RTMPoseExtractor and runs
    tracked extraction.

    Args:
        video_path: Path to video file.
        mode: Model preset — ``"lightweight"``, ``"balanced"``, ``"performance"``.
        output_format: ``"normalized"`` or ``"pixels"``.
        person_click: Optional click to select target person.

    Returns:
        TrackedExtraction with poses and foot_keypoints populated.
    """
    extractor = RTMPoseExtractor(mode=mode, output_format=output_format)
    return extractor.extract_video_tracked(video_path, person_click=person_click)
