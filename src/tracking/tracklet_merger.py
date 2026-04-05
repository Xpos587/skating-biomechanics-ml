"""Post-hoc tracklet merging for occlusion recovery.

After frame-by-frame tracking, builds Tracklet objects and merges
split tracklets using 3D skeletal identity similarity.

Reference:
    - AFLink: Appearance-Free Link (StrongSORT)
    - Hawk-Eye SkeleTRACK
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.tracking.skeletal_identity import (
    SkeletalIdentityExtractor,
    identity_similarity,
)

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    """A contiguous sequence of frames with the same track ID."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    poses: dict[int, np.ndarray] = field(default_factory=dict)
    foot_keypoints: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def start_frame(self) -> int:
        return min(self.frames) if self.frames else -1

    @property
    def end_frame(self) -> int:
        return max(self.frames) if self.frames else -1

    @property
    def length(self) -> int:
        return len(self.frames)

    def centroid_at(self, frame: int) -> tuple[float, float]:
        pose = self.poses.get(frame)
        if pose is None:
            return (float("nan"), float("nan"))
        return (float(np.nanmean(pose[:, 0])), float(np.nanmean(pose[:, 1])))

    def last_centroid(self) -> tuple[float, float]:
        return self.centroid_at(self.end_frame)

    def first_centroid(self) -> tuple[float, float]:
        return self.centroid_at(self.start_frame)

    def get_poses_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 17, 3), dtype=np.float32)
        sorted_frames = sorted(self.frames)
        return np.stack([self.poses[f] for f in sorted_frames], axis=0)


def build_tracklets(
    frame_track_data: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> list[Tracklet]:
    """Build Tracklet objects from frame_track_data.

    Args:
        frame_track_data: {frame_idx: {track_id: (pose (17,3), foot_kps (6,3))}}

    Returns:
        List of Tracklet objects, one per unique track_id.
    """
    track_data: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for frame_idx, tid_map in frame_track_data.items():
        for tid, (pose, foot_kps) in tid_map.items():
            track_data.setdefault(tid, []).append((frame_idx, pose, foot_kps))

    tracklets: list[Tracklet] = []
    for tid, entries in track_data.items():
        entries.sort(key=lambda x: x[0])
        tracklets.append(
            Tracklet(
                track_id=tid,
                frames=[e[0] for e in entries],
                poses={e[0]: e[1] for e in entries},
                foot_keypoints={e[0]: e[2] for e in entries},
            )
        )
    return tracklets


class TrackletMerger:
    """Merge split tracklets using 3D skeletal identity + spatial proximity.

    Finds the best matching tracklet for a target after an occlusion gap.

    Args:
        identity_extractor: For 3D bone length profiles. If None, only
            spatial/temporal scoring is used.
        bone_weight: Weight for bone length cosine similarity.
        spatial_weight: Weight for spatial proximity.
        temporal_weight: Weight for temporal gap penalty.
        similarity_threshold: Minimum combined score to accept a match.
        max_gap_frames: Maximum occlusion gap in frames (30fps * 30s = 900).
    """

    def __init__(
        self,
        identity_extractor: SkeletalIdentityExtractor | None = None,
        bone_weight: float = 0.6,
        spatial_weight: float = 0.3,
        temporal_weight: float = 0.1,
        similarity_threshold: float = 0.85,
        max_gap_frames: int = 900,
    ) -> None:
        self._identity_extractor = identity_extractor
        self.bone_weight = bone_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.similarity_threshold = similarity_threshold
        self.max_gap_frames = max_gap_frames

    def find_best_match(
        self,
        target: Tracklet,
        candidates: list[Tracklet],
    ) -> Tracklet | None:
        """Find best matching tracklet after an occlusion gap.

        Args:
            target: Pre-occlusion tracklet.
            candidates: Tracklets to consider (must start after target ends).

        Returns:
            Best matching Tracklet, or None.
        """
        if not candidates or target.length < 10:
            return None

        valid = [c for c in candidates if c.start_frame > target.end_frame]
        if not valid:
            return None

        target_profile = self._compute_profile(target)
        if target_profile is None:
            return self._spatial_only_match(target, valid)

        best_score = -1.0
        best_match: Tracklet | None = None

        for candidate in valid:
            gap = candidate.start_frame - target.end_frame
            if gap > self.max_gap_frames:
                continue
            score = self._compute_match_score(target, candidate, target_profile, gap)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= self.similarity_threshold and best_match is not None:
            logger.info(
                "Tracklet merge: track %d → %d (score=%.3f, gap=%d frames)",
                target.track_id,
                best_match.track_id,
                best_score,
                best_match.start_frame - target.end_frame,  # type: ignore[optional-attr]
            )
            return best_match
        return None

    def merge(self, target: Tracklet, match: Tracklet) -> Tracklet:
        """Merge two tracklets into one (keeps target's track_id)."""
        return Tracklet(
            track_id=target.track_id,
            frames=target.frames + match.frames,
            poses={**target.poses, **match.poses},
            foot_keypoints={**target.foot_keypoints, **match.foot_keypoints},
        )

    def _compute_match_score(
        self,
        target: Tracklet,
        candidate: Tracklet,
        target_profile: np.ndarray,
        gap: int,
    ) -> float:
        candidate_profile = self._compute_profile(candidate)
        if candidate_profile is None:
            return 0.0
        bone_sim = max(0.0, identity_similarity(target_profile, candidate_profile))

        end_pos = np.array(target.last_centroid())
        start_pos = np.array(candidate.first_centroid())
        if np.any(np.isnan(end_pos)) or np.any(np.isnan(start_pos)):
            spatial_sim = 0.0
        else:
            spatial_sim = max(0.0, 1.0 - np.linalg.norm(end_pos - start_pos) / 0.3)

        temporal_sim = max(0.0, 1.0 - gap / self.max_gap_frames)

        return float(
            self.bone_weight * bone_sim
            + self.spatial_weight * spatial_sim
            + self.temporal_weight * temporal_sim
        )

    def _spatial_only_match(
        self,
        target: Tracklet,
        candidates: list[Tracklet],
    ) -> Tracklet | None:
        best_dist = float("inf")
        best: Tracklet | None = None
        end_pos = np.array(target.last_centroid())
        if np.any(np.isnan(end_pos)):
            return None
        for c in candidates:
            gap = c.start_frame - target.end_frame
            if gap > self.max_gap_frames:
                continue
            start_pos = np.array(c.first_centroid())
            if np.any(np.isnan(start_pos)):
                continue
            dist = np.linalg.norm(end_pos - start_pos)
            if dist < best_dist:
                best_dist = dist
                best = c
        if best is not None and best_dist < 0.2:
            return best
        return None

    def _compute_profile(self, tracklet: Tracklet) -> np.ndarray | None:
        if self._identity_extractor is None:
            return None
        return self._identity_extractor.extract_profile(tracklet.get_poses_array())
