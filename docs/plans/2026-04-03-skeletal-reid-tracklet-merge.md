# Robust Person Re-ID: 3D Skeletal Identity + Tracklet Merging

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable person re-identification after full occlusion (5-30s) using 3D bone length biometrics and post-hoc tracklet merging.

**Architecture:** After frame-by-frame tracking, build Tracklet objects from per-frame data. Compute 3D skeletal identity profiles (bone lengths via MotionAGFormer-S lifting). Find and merge split tracklets using cosine similarity of bone length vectors + spatial proximity. Replace centroid-only anti-steal with skeletal anomaly detection.

**Tech Stack:** numpy, scipy, MotionAGFormer-S (existing 3D lifter), Biomechanics3DEstimator (fallback)

---

## Context for Implementers

This project is an AI figure skating coach. RTMPose detects multiple people per frame, a tracker assigns IDs, and the user selects one person to analyze. **Problem:** when the target is fully occluded (another skater passes in front), the tracker assigns the target's ID to the occluder, and after occlusion the target gets a new ID. The skeleton never returns to the correct person.

**Key files you'll modify:**
- `src/tracking/` — new modules for skeletal identity and tracklet merging
- `src/pose_estimation/rtmlib_extractor.py` — integration point (anti-steal + post-hoc merge)

**Existing infrastructure you'll reuse:**
- `src/pose_3d/athletepose_extractor.py` — `AthletePose3DExtractor.extract_sequence(poses_2d: (N,17,2)) -> (N,17,3)` in meters. Model at `data/models/motionagformer-s-ap3d.pth.tr`.
- `src/pose_3d/biomechanics_estimator.py` — `Biomechanics3DEstimator.estimate_3d(poses_2d, body_height=1.7) -> (N,17,3)` (fallback, no GPU).
- `src/types.py` — `H36Key` enum (HIP_CENTER=0, RHIP=1, ..., RWRIST=16).
- `src/tracking/sports2d.py` — existing tracker with `update(keypoints, scores) -> list[int]`.

**Data flow:**
1. Main loop: RTMPose → H3.6M conversion → Sports2D/DeepSORT tracking → anti-steal check → store in `frame_track_data[frame_idx][track_id] = (pose, foot_kps)`
2. Post-hoc: build Tracklets from `frame_track_data` → compute 3D bone profiles → find matching tracklet → fill NaN gaps

---

## Files

### New
| File | Purpose |
|------|---------|
| `src/tracking/skeletal_identity.py` | 3D bone length extraction + identity profile + cosine similarity |
| `src/tracking/tracklet_merger.py` | Tracklet dataclass + builder + TrackletMerger |
| `tests/tracking/test_skeletal_identity.py` | 7 unit tests |
| `tests/tracking/test_tracklet_merger.py` | 6 unit tests |

### Modified
| File | Change |
|------|--------|
| `src/tracking/__init__.py` | Add exports for new classes |
| `src/pose_estimation/rtmlib_extractor.py` | Skeletal anomaly in anti-steal + post-hoc tracklet merge after main loop |

---

## Task 1: SkeletalIdentityExtractor

**Files:**
- Create: `src/tracking/skeletal_identity.py`
- Create: `tests/tracking/test_skeletal_identity.py`

- [ ] **Step 1: Write `src/tracking/skeletal_identity.py`**

```python
"""3D skeletal identity extraction for person re-identification.

Uses 3D bone lengths (invariant to clothing, lighting, viewing angle)
as biometric identity signatures. Lifts 2D H3.6M poses to 3D via
MotionAGFormer-S, then computes Euclidean distances between connected
joints to form a bone length vector.

Reference:
    - Hawk-Eye SkeleTRACK: 29-keypoint skeletal tracking
    - AFLink: Appearance-Free Link for tracklet association (StrongSORT)
"""

import logging
from pathlib import Path

import numpy as np

from src.types import H36Key

logger = logging.getLogger(__name__)

# 12 discriminative bone pairs for identity (indices into H3.6M 17-keypoint format)
BONE_PAIRS: list[tuple[int, int]] = [
    (H36Key.RHIP, H36Key.RKNEE),          # 0: femur R
    (H36Key.RKNEE, H36Key.RFOOT),         # 1: tibia R
    (H36Key.LHIP, H36Key.LKNEE),          # 2: femur L
    (H36Key.LKNEE, H36Key.LFOOT),         # 3: tibia L
    (H36Key.RSHOULDER, H36Key.RELBOW),    # 4: humerus R
    (H36Key.RELBOW, H36Key.RWRIST),       # 5: ulna R
    (H36Key.LSHOULDER, H36Key.LELBOW),    # 6: humerus L
    (H36Key.LELBOW, H36Key.LWRIST),       # 7: ulna L
    (H36Key.HIP_CENTER, H36Key.THORAX),   # 8: spine lower
    (H36Key.THORAX, H36Key.NECK),         # 9: spine upper
    (H36Key.LSHOULDER, H36Key.RSHOULDER), # 10: shoulder width
    (H36Key.LHIP, H36Key.RHIP),           # 11: pelvis width
]

NUM_BONES = len(BONE_PAIRS)  # 12

# Indices of spine bones for scale normalization
SPINE_INDICES = [8, 9]


def compute_bone_lengths_3d(poses_3d: np.ndarray) -> np.ndarray:
    """Compute 3D Euclidean bone lengths from lifted poses.

    Args:
        poses_3d: (N, 17, 3) poses in meters.

    Returns:
        (N, NUM_BONES) bone lengths. NaN where keypoints are missing.
    """
    n = len(poses_3d)
    bones = np.zeros((n, NUM_BONES), dtype=np.float32)
    for i, (j1, j2) in enumerate(BONE_PAIRS):
        diff = poses_3d[:, j1, :] - poses_3d[:, j2, :]
        bones[:, i] = np.linalg.norm(diff, axis=1)
    return bones


def compute_identity_profile(bones: np.ndarray) -> np.ndarray:
    """Compute a stable, scale-invariant identity profile.

    Applies temporal median filtering and normalizes by spine length.

    Args:
        bones: (N, NUM_BONES) bone lengths.

    Returns:
        (NUM_BONES,) normalized identity vector.
    """
    median_bones = np.nanmedian(bones, axis=0)
    spine = median_bones[SPINE_INDICES[0]] + median_bones[SPINE_INDICES[1]]
    if spine > 1e-6:
        median_bones = median_bones / spine
    return median_bones.astype(np.float32)


def identity_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Cosine similarity between two identity profiles.

    Returns:
        1.0 for identical, ~0.0 for orthogonal, -1.0 for opposite.
    """
    norm_a = np.linalg.norm(profile_a)
    norm_b = np.linalg.norm(profile_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(profile_a, profile_b) / (norm_a * norm_b))


class SkeletalIdentityExtractor:
    """Extract 3D skeletal identity profiles from 2D pose sequences.

    Uses MotionAGFormer-S for 2D→3D lifting when available,
    falls back to Biomechanics3DEstimator.

    Args:
        model_path: Path to MotionAGFormer-S checkpoint (.pth.tr).
            If None, uses Biomechanics3DEstimator fallback.
        device: Device for inference ("auto", "cuda", "cpu").
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        self._extractor = None
        self._use_ml = False
        if model_path is not None:
            try:
                from src.pose_3d.athletepose_extractor import (
                    AthletePose3DExtractor,
                )

                self._extractor = AthletePose3DExtractor(
                    model_path=model_path, device=device,
                )
                self._use_ml = True
                logger.info("SkeletalIdentity: using MotionAGFormer-S")
            except Exception as e:
                logger.warning("MotionAGFormer unavailable: %s", e)

    def extract_profile(
        self, poses_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Extract identity profile from 2D poses.

        Args:
            poses_2d: (N, 17, 2 or 3) H3.6M poses (normalized [0,1]).

        Returns:
            (NUM_BONES,) identity profile, or None if insufficient data.
        """
        if len(poses_2d) < 10:
            return None

        poses_xy = poses_2d[:, :, :2]
        poses_3d = self._lift_to_3d(poses_xy)
        if poses_3d is None:
            return None

        bones = compute_bone_lengths_3d(poses_3d)
        valid = np.any(~np.isnan(bones), axis=1)
        if valid.sum() < 5:
            return None
        return compute_identity_profile(bones[valid])

    def _lift_to_3d(self, poses_2d: np.ndarray) -> np.ndarray | None:
        """Lift 2D poses to 3D."""
        if self._use_ml and self._extractor is not None:
            try:
                return self._extractor.extract_sequence(poses_2d)
            except Exception as e:
                logger.warning("3D lifting failed: %s", e)
        return self._estimate_3d_simple(poses_2d)

    def _estimate_3d_simple(
        self, poses_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Fallback: biomechanics-based 3D estimation."""
        try:
            from src.pose_3d.biomechanics_estimator import (
                Biomechanics3DEstimator,
            )

            estimator = Biomechanics3DEstimator()
            return estimator.estimate_3d(poses_2d)
        except Exception as e:
            logger.warning("Biomechanics estimator failed: %s", e)
            return None


def compute_2d_skeletal_ratios(pose: np.ndarray) -> np.ndarray:
    """Compute scale-invariant 2D skeletal ratios for anomaly detection.

    Used in the online anti-steal check. These ratios are
    perspective-distorted but stable within the same viewing angle.
    A sudden discontinuity indicates a different person.

    Args:
        pose: (17, 2 or 3) H3.6M pose.

    Returns:
        (5,) ratio vector.
    """
    xy = pose[:, :2]
    sw = np.linalg.norm(xy[H36Key.LSHOULDER] - xy[H36Key.RSHOULDER])
    pw = np.linalg.norm(xy[H36Key.LHIP] - xy[H36Key.RHIP])
    th = np.linalg.norm(xy[H36Key.HIP_CENTER] - xy[H36Key.NECK])
    fl = np.linalg.norm(xy[H36Key.LHIP] - xy[H36Key.LKNEE])
    fr = np.linalg.norm(xy[H36Key.RHIP] - xy[H36Key.RKNEE])
    denom = max(pw, th, 1e-6)
    return np.array([sw / denom, pw / denom, th / denom, fl / denom, fr / denom])
```

- [ ] **Step 2: Write `tests/tracking/test_skeletal_identity.py`**

```python
"""Tests for 3D skeletal identity extraction."""

import numpy as np
import pytest

from src.tracking.skeletal_identity import (
    NUM_BONES,
    compute_2d_skeletal_ratios,
    compute_bone_lengths_3d,
    compute_identity_profile,
    identity_similarity,
)
from src.types import H36Key


def _make_3d_pose(bone_scale: float = 1.0) -> np.ndarray:
    """Create synthetic 3D H3.6M pose with known bone lengths."""
    pose = np.zeros((17, 3), dtype=np.float32)
    s = bone_scale
    pose[H36Key.HIP_CENTER] = [0, 0, 0]
    pose[H36Key.RHIP] = [-0.05, 0, 0]
    pose[H36Key.LHIP] = [0.05, 0, 0]
    pose[H36Key.RKNEE] = [-0.05, -0.20 * s, 0]
    pose[H36Key.LKNEE] = [0.05, -0.20 * s, 0]
    pose[H36Key.RFOOT] = [-0.05, -0.40 * s, 0]
    pose[H36Key.LFOOT] = [0.05, -0.40 * s, 0]
    pose[H36Key.SPINE] = [0, -0.15 * s, 0]
    pose[H36Key.THORAX] = [0, -0.25 * s, 0]
    pose[H36Key.NECK] = [0, -0.30 * s, 0]
    pose[H36Key.HEAD] = [0, -0.35 * s, 0]
    pose[H36Key.LSHOULDER] = [0.08, -0.25 * s, 0]
    pose[H36Key.RSHOULDER] = [-0.08, -0.25 * s, 0]
    pose[H36Key.LELBOW] = [0.12, -0.15 * s, 0]
    pose[H36Key.RELBOW] = [-0.12, -0.15 * s, 0]
    pose[H36Key.LWRIST] = [0.14, -0.05 * s, 0]
    pose[H36Key.RWRIST] = [-0.14, -0.05 * s, 0]
    return pose


class TestComputeBoneLengths3D:
    def test_returns_correct_shape(self):
        poses = np.array([_make_3d_pose()] * 10)
        bones = compute_bone_lengths_3d(poses)
        assert bones.shape == (10, NUM_BONES)

    def test_femur_length(self):
        poses = np.array([_make_3d_pose(bone_scale=1.0)])
        bones = compute_bone_lengths_3d(poses)
        # RHIP(−0.05,0,0) → RKNEE(−0.05,−0.2,0) = 0.2
        assert abs(bones[0, 0] - 0.2) < 1e-4

    def test_different_scale(self):
        poses_a = np.array([_make_3d_pose(1.0)] * 5)
        poses_b = np.array([_make_3d_pose(1.3)] * 5)
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        assert not np.allclose(ba[0], bb[0])


class TestIdentityProfile:
    def test_shape(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert profile.shape == (NUM_BONES,)

    def test_deterministic(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        p1 = compute_identity_profile(bones)
        p2 = compute_identity_profile(bones)
        assert np.allclose(p1, p2)


class TestIdentitySimilarity:
    def test_identical(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert abs(identity_similarity(profile, profile) - 1.0) < 1e-5

    def test_same_proportions_different_scale(self):
        """Same proportions but different size → cosine sim ≈ 1.0."""
        poses_a = np.array([_make_3d_pose(1.0)] * 20)
        poses_b = np.array([_make_3d_pose(1.3)] * 20)
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        pa = compute_identity_profile(ba)
        pb = compute_identity_profile(bb)
        assert identity_similarity(pa, pb) > 0.99

    def test_different_proportions(self):
        poses_a = np.array([_make_3d_pose(1.0)] * 20)
        poses_b = np.array([_make_3d_pose(1.0)] * 20)
        # Double femurs/tibias on person B
        poses_b[:, H36Key.RKNEE, 1] *= 2.0
        poses_b[:, H36Key.LKNEE, 1] *= 2.0
        poses_b[:, H36Key.RFOOT, 1] *= 2.0
        poses_b[:, H36Key.LFOOT, 1] *= 2.0
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        pa = compute_identity_profile(ba)
        pb = compute_identity_profile(bb)
        assert identity_similarity(pa, pb) < 0.95


class TestNaNHandling:
    def test_nan_keypoints(self):
        poses = np.array([_make_3d_pose()] * 10)
        poses[3, H36Key.RFOOT, :] = np.nan
        bones = compute_bone_lengths_3d(poses)
        assert np.isnan(bones[3, 1])

    def test_profile_ignores_nan_frames(self):
        poses = np.array([_make_3d_pose()] * 20)
        poses[5, :, :] = np.nan
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert not np.any(np.isnan(profile))


class Test2dSkeletalRatios:
    def test_returns_five_ratios(self):
        pose = _make_3d_pose()[:, :2]
        ratios = compute_2d_skeletal_ratios(pose)
        assert ratios.shape == (5,)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/tracking/test_skeletal_identity.py -x --tb=short -v`
Expected: 9 passed

- [ ] **Step 4: Commit**

```bash
git add src/tracking/skeletal_identity.py tests/tracking/test_skeletal_identity.py
git commit -m "feat(tracking): 3D skeletal identity extraction for person re-ID"
```

---

## Task 2: TrackletBuilder + TrackletMerger

**Files:**
- Create: `src/tracking/tracklet_merger.py`
- Create: `tests/tracking/test_tracklet_merger.py`

- [ ] **Step 1: Write `src/tracking/tracklet_merger.py`**

```python
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
        tracklets.append(Tracklet(
            track_id=tid,
            frames=[e[0] for e in entries],
            poses={e[0]: e[1] for e in entries},
            foot_keypoints={e[0]: e[2] for e in entries},
        ))
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
            # No bone data — fall back to spatial only
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

        if best_score >= self.similarity_threshold:
            logger.info(
                "Tracklet merge: track %d → %d (score=%.3f, gap=%d frames)",
                target.track_id, best_match.track_id, best_score,
                best_match.start_frame - target.end_frame,
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

        return (
            self.bone_weight * bone_sim
            + self.spatial_weight * spatial_sim
            + self.temporal_weight * temporal_sim
        )

    def _spatial_only_match(
        self, target: Tracklet, candidates: list[Tracklet],
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
```

- [ ] **Step 2: Write `tests/tracking/test_tracklet_merger.py`**

```python
"""Tests for tracklet building and merging."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.tracking.skeletal_identity import SkeletalIdentityExtractor
from src.tracking.tracklet_merger import (
    Tracklet,
    TrackletMerger,
    build_tracklets,
)


def _make_pose(cx: float, cy: float) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[:, 0] = cx
    pose[:, 1] = cy
    pose[:, 2] = 0.8
    return pose


def _make_foot() -> np.ndarray:
    return np.zeros((6, 3), dtype=np.float32)


class TestTracklet:
    def test_properties(self):
        t = Tracklet(track_id=0, frames=[10, 11, 12])
        assert t.start_frame == 10
        assert t.end_frame == 12
        assert t.length == 3

    def test_empty(self):
        t = Tracklet(track_id=0)
        assert t.start_frame == -1
        assert t.end_frame == -1
        assert t.length == 0

    def test_centroid(self):
        pose = _make_pose(0.5, 0.3)
        t = Tracklet(track_id=0, frames=[5], poses={5: pose})
        cx, cy = t.centroid_at(5)
        assert abs(cx - 0.5) < 1e-5

    def test_missing_centroid(self):
        t = Tracklet(track_id=0, frames=[5])
        cx, cy = t.centroid_at(5)
        assert np.isnan(cx)


class TestBuildTracklets:
    def test_single_track(self):
        fd = {0: {0: (_make_pose(0.3, 0.5), _make_foot())},
              1: {0: (_make_pose(0.31, 0.51), _make_foot())}}
        ts = build_tracklets(fd)
        assert len(ts) == 1
        assert ts[0].track_id == 0
        assert ts[0].length == 2

    def test_multiple_tracks(self):
        fd = {0: {0: (_make_pose(0.3, 0.5), _make_foot()),
                   1: (_make_pose(0.7, 0.5), _make_foot())},
              1: {0: (_make_pose(0.31, 0.51), _make_foot()),
                   1: (_make_pose(0.69, 0.49), _make_foot())}}
        ts = build_tracklets(fd)
        assert len(ts) == 2

    def test_empty(self):
        assert build_tracklets({}) == []


class TestTrackletMerger:
    def test_spatial_only_near_wins(self):
        merger = TrackletMerger(identity_extractor=None, similarity_threshold=0.5)
        target = Tracklet(
            track_id=0, frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        near = Tracklet(
            track_id=1, frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.32, 0.52) for i in range(10)},
        )
        far = Tracklet(
            track_id=2, frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.8, 0.8) for i in range(10)},
        )
        match = merger.find_best_match(target, [near, far])
        assert match is not None
        assert match.track_id == 1

    def test_spatial_only_too_far(self):
        merger = TrackletMerger(identity_extractor=None, similarity_threshold=0.9)
        target = Tracklet(
            track_id=0, frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        far = Tracklet(
            track_id=1, frames=list(range(900, 910)),
            poses={900 + i: _make_pose(0.8, 0.8) for i in range(10)},
        )
        assert merger.find_best_match(target, [far]) is None

    def test_bone_similarity_drives_matching(self):
        extractor = MagicMock(spec=SkeletalIdentityExtractor)
        # profile_a and profile_c are identical; profile_b is different
        pa = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        pb = np.array([0.5, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5])
        profiles = [pa, pb]  # order: first call for wrong, second for correct
        extractor.extract_profile.side_effect = lambda p: profiles.pop(0) / np.linalg.norm(profiles[-1])

        merger = TrackletMerger(
            identity_extractor=extractor,
            similarity_threshold=0.5,
            bone_weight=1.0, spatial_weight=0.0, temporal_weight=0.0,
        )
        target = Tracklet(
            track_id=0, frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        # Near but wrong person
        near_wrong = Tracklet(
            track_id=1, frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.31, 0.51) for i in range(10)},
        )
        # Far but same person
        far_correct = Tracklet(
            track_id=2, frames=list(range(20, 30)),
            poses={20 + i: _make_pose(0.6, 0.4) for i in range(10)},
        )
        match = merger.find_best_match(target, [near_wrong, far_correct])
        assert match is not None
        assert match.track_id == 2

    def test_merge_combines_frames(self):
        t1 = Tracklet(track_id=0, frames=[1, 2],
                       poses={1: _make_pose(0.3, 0.5), 2: _make_pose(0.31, 0.51)},
                       foot_keypoints={1: _make_foot(), 2: _make_foot()})
        t2 = Tracklet(track_id=1, frames=[5, 6],
                       poses={5: _make_pose(0.32, 0.52), 6: _make_pose(0.33, 0.53)},
                       foot_keypoints={5: _make_foot(), 6: _make_foot()})
        merged = TrackletMerger().merge(t1, t2)
        assert merged.track_id == 0
        assert merged.frames == [1, 2, 5, 6]
        assert len(merged.poses) == 4

    def test_no_match_short_target(self):
        merger = TrackletMerger(identity_extractor=None)
        target = Tracklet(track_id=0, frames=[5])  # too short
        assert merger.find_best_match(target, []) is None
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/tracking/test_tracklet_merger.py -x --tb=short -v`
Expected: 10 passed

- [ ] **Step 4: Commit**

```bash
git add src/tracking/tracklet_merger.py tests/tracking/test_tracklet_merger.py
git commit -m "feat(tracking): post-hoc tracklet merging for occlusion recovery"
```

---

## Task 3: Integration into RTMPoseExtractor

**Files:**
- Modify: `src/tracking/__init__.py`
- Modify: `src/pose_estimation/rtmlib_extractor.py`

- [ ] **Step 1: Update `src/tracking/__init__.py`**

Append exports:

```python
from .skeletal_identity import (
    SkeletalIdentityExtractor,
    compute_bone_lengths_3d,
    compute_identity_profile,
    compute_2d_skeletal_ratios,
    identity_similarity,
)
from .tracklet_merger import (
    Tracklet,
    TrackletMerger,
    build_tracklets,
)

__all__ = [
    "Sports2DTracker",
    "DeepSORTTracker",
    "SkeletalIdentityExtractor",
    "TrackletMerger",
    "Tracklet",
    "build_tracklets",
    "compute_bone_lengths_3d",
    "compute_identity_profile",
    "compute_2d_skeletal_ratios",
    "identity_similarity",
]
```

- [ ] **Step 2: Modify `src/pose_estimation/rtmlib_extractor.py` — add import**

Add near the top (after existing tracking imports, around line 41):

```python
from ..tracking.skeletal_identity import compute_2d_skeletal_ratios
from ..tracking.tracklet_merger import TrackletMerger, build_tracklets
```

- [ ] **Step 3: Modify `src/pose_estimation/rtmlib_extractor.py` — add state variable**

Find the line with `last_target_pose: np.ndarray | None = None` (around line 181) and add below it:

```python
last_target_ratios: np.ndarray | None = None
```

- [ ] **Step 4: Modify `src/pose_estimation/rtmlib_extractor.py` — improve anti-steal**

Find the anti-steal block (around lines 315-330). Replace the existing centroid-jump-only check with combined centroid + skeletal anomaly detection. The current code looks like:

```python
if last_target_pose is not None:
    cur_cx = np.nanmean(h36m_poses[p, :, 0])
    cur_cy = np.nanmean(h36m_poses[p, :, 1])
    prev_cx = np.nanmean(last_target_pose[:, 0])
    prev_cy = np.nanmean(last_target_pose[:, 1])
    jump = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
    if jump > 0.15:
        stolen = True
        break
```

Replace with:

```python
if last_target_pose is not None:
    cur_cx = np.nanmean(h36m_poses[p, :, 0])
    cur_cy = np.nanmean(h36m_poses[p, :, 1])
    prev_cx = np.nanmean(last_target_pose[:, 0])
    prev_cy = np.nanmean(last_target_pose[:, 1])
    jump = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)

    # Skeletal anomaly: sudden change in body proportions
    skeletal_anomaly = False
    if last_target_ratios is not None:
        curr_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
        ratio_change = float(np.linalg.norm(curr_ratios - last_target_ratios))
        skeletal_anomaly = ratio_change > 0.25

    if jump > 0.15 or skeletal_anomaly:
        stolen = True
        break
```

- [ ] **Step 5: Modify `src/pose_estimation/rtmlib_extractor.py` — update last_target_ratios**

Find the line where `last_target_pose = h36m_poses[p].copy()` (inside the `if tid == target_track_id` block, after the anti-steal check). Add below it:

```python
last_target_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
```

Do the same in the biometric migration block where `last_target_pose = best_new_data[0].copy()` — add below:

```python
last_target_ratios = compute_2d_skeletal_ratios(best_new_data[0])
```

- [ ] **Step 6: Modify `src/pose_estimation/rtmlib_extractor.py` — add post-hoc tracklet merging**

Find the line `# Determine first_detection_frame` (around line 370). Insert the post-hoc merging block BEFORE it:

```python
        # --- Post-hoc tracklet merging for occlusion recovery ---
        valid_mask_pre = ~np.isnan(all_poses[:, 0, 0])
        if not valid_mask_pre.all() and frame_track_data:
            from ..tracking.skeletal_identity import SkeletalIdentityExtractor
            from pathlib import Path

            model_3d = Path("data/models/motionagformer-s-ap3d.pth.tr")
            identity_ext = None
            if model_3d.exists():
                identity_ext = SkeletalIdentityExtractor(
                    model_path=model_3d, device="auto",
                )

            merger = TrackletMerger(
                identity_extractor=identity_ext,
                similarity_threshold=0.80,
            )
            tracklets = build_tracklets(frame_track_data)

            # Find target's tracklet
            target_tracklet = None
            for t in tracklets:
                if t.track_id == target_track_id:
                    target_tracklet = t
                    break

            if target_tracklet is not None:
                valid_frames = np.where(valid_mask_pre)[0]
                if len(valid_frames) > 0:
                    last_valid = int(valid_frames[-1])
                    if last_valid < num_frames - 1:
                        # Trailing NaN gap — try to find match
                        candidates = [t for t in tracklets if t.track_id != target_track_id]
                        match = merger.find_best_match(target_tracklet, candidates)
                        if match is not None:
                            for f in match.frames:
                                if f < num_frames and np.isnan(all_poses[f, 0, 0]):
                                    all_poses[f] = match.poses.get(f, all_poses[f])
                                    all_feet[f] = match.foot_keypoints.get(f, all_feet[f])
                            logger.info(
                                "Post-hoc merge: filled %d frames from track %d",
                                sum(1 for f in match.frames if f < num_frames and np.isnan(all_poses[f, 0, 0])),
                                match.track_id,
                            )
```

Note: this uses a local import inside the function body so the 3D model is only loaded when actually needed (post-hoc, after the main loop). This avoids impacting frame-by-frame performance.

- [ ] **Step 7: Run existing tracking tests**

Run: `uv run pytest tests/tracking/ -x --tb=short -q`
Expected: all existing tests still pass (no regressions)

- [ ] **Step 8: Run pose estimation tests**

Run: `uv run pytest tests/pose_estimation/ -x --tb=short -q`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add src/tracking/__init__.py src/pose_estimation/rtmlib_extractor.py
git commit -m "feat(tracking): skeletal anti-steal + post-hoc tracklet merge for occlusion recovery"
```

---

## Task 4: Manual Verification on VOLODYA.MOV

- [ ] **Step 1: Render with Sports2D tracking**

Run:
```bash
uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/VOLODYA.MOV \
    --tracking sports2d --person-click 450 941 --layer 1 \
    --compress --crf 18 --output /tmp/volodya_merged.mp4
```

Expected: skeleton stays on Volodya through the occlusion, returns after.

- [ ] **Step 2: Verify in video player**

Check:
1. Skeleton stays on target before occlusion
2. Skeleton disappears during occlusion (NaN gap) — this is correct, better than wrong person
3. Skeleton returns to Volodya after occluder leaves
4. No skeleton on wrong person at any point

- [ ] **Step 3: Commit if successful, or debug**

If skeleton returns correctly: commit any remaining uncommitted changes.
If not: check logs for `Tracklet merge:` and `Post-hoc merge:` messages, adjust thresholds.

---

## Risks

1. **3D lifting accuracy**: Biomechanics3DEstimator fallback produces poor 3D (no real depth). MotionAGFormer-S is needed for reliable bone lengths. Verify model loads correctly on this system.

2. **Same body proportions**: If two skaters have very similar bone lengths, the merger may match the wrong person. Mitigation: spatial proximity weighting + temporal gap penalty help disambiguate.

3. **Performance**: 3D lifting adds ~10-15ms per tracklet. For 4 tracklets × 200 frames each, that's ~10 seconds total. Acceptable for offline analysis.

4. **Pose estimator confusion during occlusion**: RTMPose may detect the occluder's pose as the target's pose (overlapping bounding box). The anti-steal mechanism should catch this via skeletal ratio change.
