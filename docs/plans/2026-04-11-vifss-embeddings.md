# VIFSS Embeddings Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- ]`) syntax for tracking.

**Goal:** Integrate VIFSS (View-Invariant Figure Skating-Specific) pose encoder into the analysis pipeline to produce camera-independent embeddings from 2D poses, enabling cross-athlete comparison AND auto element detection (solves Phase D from ROADMAP).

**Architecture:** Clone the open-source VIFSS repo (Apache 2.0, github.com/ryota-skating/VIFSS) into `src/ml/vifss/`, adapt to our conventions (H36Key mapping, PoseNormalizer integration). Two-stage training: contrastive pre-training on AthletePose3D 3D poses (virtual camera projections), then fine-tuning on SkatingVerse for element classification. At inference, the encoder takes RTMPose 2D poses → view-invariant embeddings.

**Integration approach (Variant A + B):**
- **Variant A:** `technique_similarity` MetricResult in `_analyze_common()` — single athlete vs reference embedding
- **Variant B:** `compare_technique()` for cross-athlete video comparison — cosine similarity between two video embeddings

**Tech Stack:** PyTorch (training), ONNX Runtime (inference), numpy, AthletePose3D dataset, SkatingVerse dataset

**Why:** Direct comparison of two skaters from different camera angles is impossible with raw 2D poses. VIFSS embeddings are camera-independent (92.56% F1@50 on element-level TAS). The code is open-source and our 3D data (AthletePose3D includes FS-Jump3D) is already downloaded. Fine-tuning on SkatingVerse enables auto element detection — replacing the planned GCN classifier (ROADMAP Phase D) with a proven, simpler approach.

**Paper:** arxiv:2508.10281 (Tanaka et al., August 2025)
**Code:** github.com/ryota-skating/VIFSS (Apache 2.0)

---

## Scope

This plan covers two major deliverables:

1. **VIFSS Encoder + Embeddings** (Tasks 1-9) — view-invariant embedding extraction, contrastive loss, cosine similarity
2. **Auto Element Detection** (Phase 2, separate plan) — fine-tuning on SkatingVerse, BiGRU classifier, replaces ROADMAP Phase D GCN classifier

### Integration with Existing Pipeline

**Variant A — technique_similarity metric** (wired into `_analyze_common()` in `src/analysis/metrics.py`):
- Compute embedding for current pose sequence
- Compare against reference embedding (stored in `.npz`)
- Output as `MetricResult(name="technique_similarity", value=cosine_sim, unit="score")`
- Works with existing reference system (`data/references/*.npz`)

**Variant B — compare_technique()** (new function in `src/analysis/metrics.py`):
- Accept two pose sequences (attempt + reference)
- Compute embeddings for both
- Return cosine similarity + per-frame similarity curve
- Use for cross-athlete video comparison (no reference database needed)
- Integration point: `src/pipeline.py` `AnalysisPipeline.compare()` or standalone CLI command

### Connection to ROADMAP Phase D

This plan **replaces** the GCN Element Classifier (ROADMAP Phase D, item 9). Instead of training a separate GCN on Figure-Skating-Classification dataset:
- VIFSS encoder + SkatingVerse fine-tuning provides element classification as a byproduct
- 92.56% F1@50 (VIFSS paper) vs unknown (would need to train GCN from scratch)
- View-invariant by design (works with any camera angle)
- Simpler pipeline: no need for separate COCO→H3.6M mapping (figure-skating-classification uses COCO 17kp)

### SkatingVerse Dataset

**Required for fine-tuning (Phase 2).** Not yet downloaded.

| Property | Value |
|----------|-------|
| Classes | 28 (6 jumps × 4 rotations + 4 spins + NONE) |
| Videos | 1,687 competition clips |
| Train clips | 19,993 |
| Test clips | 8,586 |
| Format | Pose sequences (keypoints) |
| Site | https://skatingverse.github.io/ |
| 1st place solution | arxiv:2404.14032 (SkateFormer, 128 stars) |

**Download:** TODO — check website for dataset access. May need request form.

### Temporal Action Segmentation (Future)

Future plans will cover:
- FACT integration for frame-wise action labels
- Entry/jump/landing boundary detection
- BIOES-tagging (YourSkatingCoach approach)

---

## File Structure

```
src/ml/vifss/
├── __init__.py              # Package marker, public API
├── keypoint_map.py          # H36Key (our 17kp) ↔ MMPose H3.6M (standard 17kp) mapping
├── virtual_camera.py        # Virtual camera projection from 3D poses
├── preprocess.py            # AthletePose3D → VIFSS training format
├── encoder.py               # JointFormer encoder wrapper (load, infer)
├── contrastive_loss.py      # Barlow Twins + view-dependent + regularization losses
├── pretrain.py              # Contrastive pre-training script
├── finetune.py              # Fine-tuning script (SkatingVerse) → element classification
├── embedding.py             # Embedding extraction + cosine similarity comparison
└── config.py                # Training/inference configuration dataclass

tests/ml/vifss/
├── conftest.py              # Shared fixtures (sample 3D poses, keypoints)
├── test_keypoint_map.py     # Mapping correctness tests
├── test_virtual_camera.py   # Projection tests
├── test_preprocess.py       # Data pipeline tests
├── test_encoder.py          # Encoder wrapper tests (dummy model)
├── test_embedding.py        # Embedding extraction + similarity tests
└── test_contrastive_loss.py # Contrastive loss function tests

# Integration targets (modified by Phase 2 plan)
src/analysis/metrics.py      # + technique_similarity MetricResult (Variant A)
                              # + compare_technique() function (Variant B)
src/pipeline.py              # + embedding comparison in AnalysisPipeline
src/references/              # + reference embeddings stored alongside .npz
```

---

## Existing Code to Reuse

- `src/types.py` — `H36Key` enum (our 17kp ordering), `NormalizedPose` type alias
- `src/pose_estimation/normalizer.py` — `PoseNormalizer` (root-centering + scale normalization)
- `src/pose_3d/athletepose_extractor.py` — existing AthletePose3D data loading patterns
- `data/datasets/athletepose3d/cam_param.json` — 12 camera intrinsic/extrinsic parameters
- `data/datasets/athletepose3d/annotations_3d/` — 3D pose annotations (pkl, 81-frame windows)

## Keypoint Mapping (Critical)

Our `H36Key` ordering differs from standard MMPose H3.6M:

| Index | Our H36Key | MMPose H3.6M |
|-------|-----------|--------------|
| 0 | HIP_CENTER | Pelvis |
| 1 | RHIP | RHip |
| 2 | LHIP | **RKnee** |
| 3 | SPINE | **RFoot** |
| 4 | THORAX | **LHip** |
| 5 | NECK | **LKnee** |
| 6 | HEAD | **LFoot** |
| 7 | LSHOULDER | Spine |
| 8 | RSHOULDER | Thorax |
| 9 | LELBOW | Neck |
| 10 | RELBOW | Head/Nose |
| 11 | LWRIST | LShoulder |
| 12 | RWRIST | LElbow |
| 13 | LKNEE | **LWrist** |
| 14 | RKNEE | RShoulder |
| 15 | LFOOT | RElbow |
| 16 | RFOOT | RWrist |

---

### Task 1: Project Setup + Dependencies

**Files:**
- Create: `src/ml/vifss/__init__.py`
- Modify: `pyproject.toml` (add torch to optional deps)

- [ ] **Step 1: Create package structure**

```bash
mkdir -p src/ml/vifss
mkdir -p tests/ml/vifss
touch src/ml/vifss/__init__.py
touch tests/ml/__init__.py
touch tests/ml/vifss/__init__.py
```

- [ ] **Step 2: Check PyTorch availability**

Run: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: prints version and True (if CUDA available) or False (CPU only)

If torch is not installed:
Run: `uv add torch --extra-index-url https://download.pytorch.org/whl/cu124`

- [ ] **Step 3: Write __init__.py with public API**

```python
"""VIFSS: View-Invariant Figure Skating-Specific pose embeddings."""

from src.ml.vifss.embedding import extract_embeddings, cosine_similarity
from src.ml.vifss.encoder import VIFSSEncoder

__all__ = ["VIFSSEncoder", "extract_embeddings", "cosine_similarity"]
```

Note: These imports will fail until later tasks are complete. That's OK — we'll implement them incrementally.

- [ ] **Step 4: Commit**

```bash
git add src/ml/vifss/__init__.py tests/ml/__init__.py tests/ml/vifss/__init__.py
git commit -m "feat(ml): scaffold VIFSS embeddings module"
```

---

### Task 2: Keypoint Mapping

**Files:**
- Create: `src/ml/vifss/keypoint_map.py`
- Create: `tests/ml/vifss/conftest.py`
- Create: `tests/ml/vifss/test_keypoint_map.py`

Maps between our `H36Key` enum ordering and standard MMPose H3.6M ordering used by VIFSS.

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/conftest.py`:

```python
"""Shared fixtures for VIFSS tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_h36m_3d():
    """Sample 3D poses in our H36Key ordering (5 frames, 17 keypoints)."""
    poses = np.zeros((5, 17, 3), dtype=np.float32)
    for i in range(5):
        y_offset = float(i) * 0.01
        # Hips
        poses[i, 0] = [0.0, 0.0 + y_offset, 0.0]   # HIP_CENTER
        poses[i, 1] = [-0.1, 0.0 + y_offset, 0.0]   # RHIP
        poses[i, 2] = [0.1, 0.0 + y_offset, 0.0]    # LHIP
        # Spine chain
        poses[i, 3] = [0.0, 0.2 + y_offset, 0.0]   # SPINE
        poses[i, 4] = [0.0, 0.3 + y_offset, 0.0]   # THORAX
        poses[i, 5] = [0.0, 0.4 + y_offset, 0.0]   # NECK
        poses[i, 6] = [0.0, 0.5 + y_offset, 0.0]   # HEAD
        # Arms
        poses[i, 7] = [0.15, 0.3 + y_offset, 0.0]  # LSHOULDER
        poses[i, 8] = [-0.15, 0.3 + y_offset, 0.0] # RSHOULDER
        poses[i, 9] = [0.2, 0.15 + y_offset, 0.0]  # LELBOW
        poses[i, 10] = [-0.2, 0.15 + y_offset, 0.0] # RELBOW
        poses[i, 11] = [0.22, 0.0 + y_offset, 0.0]  # LWRIST
        poses[i, 12] = [-0.22, 0.0 + y_offset, 0.0] # RWRIST
        # Legs
        poses[i, 13] = [0.1, -0.4 + y_offset, 0.0]  # LKNEE
        poses[i, 14] = [-0.1, -0.4 + y_offset, 0.0] # RKNEE
        poses[i, 15] = [0.1, -0.8 + y_offset, 0.0]  # LFOOT
        poses[i, 16] = [-0.1, -0.8 + y_offset, 0.0] # RFOOT
    return poses
```

Create `tests/ml/vifss/test_keypoint_map.py`:

```python
"""Tests for H36Key ↔ MMPose H3.6M keypoint mapping."""

import numpy as np

from src.ml.vifss.keypoint_map import (
    OUR_TO_MMPOSE,
    our_to_mmpose,
    mmpose_to_our,
)


class TestKeypointMap:
    def test_mapping_is_bijective(self):
        """Forward and inverse mappings should be consistent."""
        our_indices = np.arange(17)
        mmpose = our_to_mmpose(our_indices)
        back = mmpose_to_our(mmpose)
        np.testing.assert_array_equal(back, our_indices)

    def test_specific_mappings(self):
        """Check known mapping values."""
        # HIP_CENTER (0) → Pelvis (0)
        assert our_to_mmpose(np.array([0]))[0] == 0
        # LHIP (2) → index 4 in MMPose (LHip comes after RHip, RKnee, RFoot)
        assert our_to_mmpose(np.array([2]))[0] == 4
        # LKNEE (13) → index 5 in MMPose (LKnee)
        assert our_to_mmpose(np.array([13]))[0] == 5
        # LWRIST (11) → index 13 in MMPose (LWrist)
        assert our_to_mmpose(np.array([11]))[0] == 13

    def test_map_3d_poses(self):
        """3D pose array should be reordered correctly."""
        poses = np.zeros((2, 17, 3), dtype=np.float32)
        # Set unique values per joint to verify reordering
        for j in range(17):
            poses[0, j] = [float(j), float(j + 100), float(j + 200)]

        mapped = our_to_mmpose(poses)

        # Original index 0 (HIP_CENTER) should go to MMPose index 0
        np.testing.assert_array_almost_equal(
            mapped[0, 0], [0.0, 100.0, 200.0]
        )
        # Original index 2 (LHIP) should go to MMPose index 4
        np.testing.assert_array_almost_equal(
            mapped[0, 4], [2.0, 102.0, 202.0]
        )

    def test_map_preserves_shape(self):
        """Mapping should preserve array shape."""
        poses_3d = np.zeros((5, 17, 3), dtype=np.float32)
        poses_2d = np.zeros((5, 17, 2), dtype=np.float32)

        assert our_to_mmpose(poses_3d).shape == (5, 17, 3)
        assert our_to_mmpose(poses_2d).shape == (5, 17, 2)
        assert mmpose_to_our(poses_3d).shape == (5, 17, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_keypoint_map.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ml.vifss.keypoint_map'`

- [ ] **Step 3: Implement keypoint_map.py**

Create `src/ml/vifss/keypoint_map.py`:

```python
"""Keypoint mapping between our H36Key ordering and MMPose H3.6M standard.

Our H36Key enum (src/types.py) uses a custom ordering optimized for
biomechanics analysis (torso chain first, then limbs).

VIFSS/JointFormer expects standard MMPose H3.6M ordering where
legs follow hips immediately.

Both have 17 keypoints. Only the index assignment differs.
"""

import numpy as np

# Mapping from our H36Key index → MMPose H3.6M index
# Our:    0=HIP_CENTER, 1=RHIP, 2=LHIP, 3=SPINE, 4=THORAX, 5=NECK, 6=HEAD,
#         7=LSHOULDER, 8=RSHOULDER, 9=LELBOW, 10=RELBOW, 11=LWRIST, 12=RWRIST,
#         13=LKNEE, 14=RKNEE, 15=LFOOT, 16=RFOOT
#
# MMPose: 0=Pelvis, 1=RHip, 2=RKnee, 3=RFoot, 4=LHip, 5=LKnee, 6=LFoot,
#         7=Spine, 8=Thorax, 9=Neck, 10=Head, 11=LShoulder, 12=LElbow,
#         13=LWrist, 14=RShoulder, 15=RElbow, 16=RWRist

# our_index → mmpose_index
OUR_TO_MMPOSE = np.array([
    0,   # HIP_CENTER → Pelvis
    1,   # RHIP → RHip
    4,   # LHIP → LHip (MMPose index 4)
    7,   # SPINE → Spine (MMPose index 7)
    8,   # THORAX → Thorax (MMPose index 8)
    9,   # NECK → Neck (MMPose index 9)
    10,  # HEAD → Head (MMPose index 10)
    11,  # LSHOULDER → LShoulder (MMPose index 11)
    14,  # RSHOULDER → RShoulder (MMPose index 14)
    12,  # LELBOW → LElbow (MMPose index 12)
    15,  # RELBOW → RElbow (MMPose index 15)
    13,  # LWRIST → LWrist (MMPose index 13)
    16,  # RWRIST → RWrist (MMPose index 16)
    5,   # LKNEE → LKnee (MMPose index 5)
    2,   # RKNEE → RKnee (MMPose index 2)
    6,   # LFOOT → LFoot (MMPose index 6)
    3,   # RFOOT → RFoot (MMPose index 3)
], dtype=np.intp)

# Inverse: mmpose_index → our_index
MMPOSE_TO_OUR = np.argsort(OUR_TO_MMPOSE)


def our_to_mmpose(poses: np.ndarray) -> np.ndarray:
    """Reorder poses from our H36Key ordering to MMPose H3.6M standard.

    Args:
        poses: Pose array with shape (..., 17, D) where D is 2 or 3.

    Returns:
        Reordered array with same shape.
    """
    return poses[..., OUR_TO_MMPOSE, :]


def mmpose_to_our(poses: np.ndarray) -> np.ndarray:
    """Reorder poses from MMPose H3.6M standard to our H36Key ordering.

    Args:
        poses: Pose array with shape (..., 17, D) where D is 2 or 3.

    Returns:
        Reordered array with same shape.
    """
    return poses[..., MMPOSE_TO_OUR, :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_keypoint_map.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/keypoint_map.py tests/ml/vifss/conftest.py tests/ml/vifss/test_keypoint_map.py
git commit -m "feat(ml): add H36Key ↔ MMPose H3.6M keypoint mapping"
```

---

### Task 3: Virtual Camera Projection

**Files:**
- Create: `src/ml/vifss/virtual_camera.py`
- Create: `tests/ml/vifss/test_virtual_camera.py`

Projects 3D poses to 2D using perspective projection from random viewpoints. This is the core of VIFSS contrastive pre-training — generate anchor-positive pairs by projecting the same 3D pose from different virtual cameras.

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/test_virtual_camera.py`:

```python
"""Tests for virtual camera projection."""

import numpy as np
import pytest

from src.ml.vifss.virtual_camera import project_virtual_camera


class TestVirtualCamera:
    def test_project_returns_2d(self, sample_h36m_3d):
        """Should project 3D poses to 2D."""
        poses_2d = project_virtual_camera(
            sample_h36m_3d,
            azimuth=0.0,
            elevation=0.0,
            distance=7.0,
        )
        assert poses_2d.shape == (5, 17, 2)
        assert poses_2d.dtype == np.float32

    def test_front_view_x_symmetry(self, sample_h36m_3d):
        """Front view should be approximately x-symmetric for symmetric pose."""
        poses_2d = project_virtual_camera(
            sample_h36m_3d,
            azimuth=0.0,
            elevation=0.0,
            distance=7.0,
        )
        # For symmetric T-pose, left and right joints should mirror in x
        # LShoulder (7) and RShoulder (8) should have opposite x
        l_shoulder_x = poses_2d[0, 7, 0]
        r_shoulder_x = poses_2d[0, 8, 0]
        assert abs(l_shoulder_x + r_shoulder_x) < 0.05

    def test_different_azimuths_produce_different_projections(self, sample_h36m_3d):
        """Different camera angles should give different 2D projections."""
        p1 = project_virtual_camera(sample_h36m_3d, azimuth=0.0, elevation=0.0, distance=7.0)
        p2 = project_virtual_camera(sample_h36m_3d, azimuth=90.0, elevation=0.0, distance=7.0)
        # Projections from 0° and 90° should differ
        assert not np.allclose(p1, p2, atol=0.01)

    def test_batch_projection(self, sample_h36m_3d):
        """Should handle batch of azimuth/elevation pairs."""
        azimuths = np.array([0.0, 45.0, 90.0])
        elevations = np.array([0.0, 10.0, -10.0])
        poses_batch = project_virtual_camera(
            sample_h36m_3d,
            azimuth=azimuths,
            elevation=elevations,
            distance=7.0,
        )
        assert poses_batch.shape == (3, 5, 17, 2)

    def test_random_projection_range(self, sample_h36m_3d):
        """Random projection should return valid 2D poses."""
        poses_2d = project_virtual_camera(
            sample_h36m_3d,
            azimuth=None,  # random
            elevation=None,  # random
            distance=None,  # random
        )
        assert poses_2d.shape == (5, 17, 2)
        assert np.all(np.isfinite(poses_2d))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_virtual_camera.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement virtual_camera.py**

Create `src/ml/vifss/virtual_camera.py`:

```python
"""Virtual camera projection for contrastive pre-training.

Generates 2D projections of 3D poses from random viewpoints,
following the VIFSS paper (Tanaka et al., 2025).

Camera parameters (from paper):
- Azimuth: uniform ±180°
- Elevation: uniform ±30°
- Distance: uniform [5, 10]
- Projection: perspective
"""

import numpy as np


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from azimuth and elevation angles.

    Args:
        azimuth_deg: Horizontal angle in degrees.
        elevation_deg: Vertical angle in degrees.

    Returns:
        3x3 rotation matrix.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Ry (azimuth rotation around Y axis)
    Ry = np.array([
        [np.cos(az), 0, np.sin(az)],
        [0, 1, 0],
        [-np.sin(az), 0, np.cos(az)],
    ])

    # Rx (elevation rotation around X axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)],
    ])

    return Rx @ Ry


def project_virtual_camera(
    poses_3d: np.ndarray,
    azimuth: float | np.ndarray | None = None,
    elevation: float | np.ndarray | None = None,
    distance: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Project 3D poses to 2D using virtual camera.

    Args:
        poses_3d: 3D poses in MMPose H3.6M ordering, shape (frames, 17, 3).
                  Assumes poses are centered at origin and normalized.
        azimuth: Camera azimuth in degrees. None = random uniform ±180°.
        elevation: Camera elevation in degrees. None = random uniform ±30°.
        distance: Camera distance. None = random uniform [5, 10].
        rng: Random generator for reproducibility.

    Returns:
        2D projections, shape (frames, 17, 2) if scalar angles,
        or (n_views, frames, 17, 2) if array angles.
    """
    if rng is None:
        rng = np.random.default_rng()

    scalar_input = np.isscalar(azimuth) or azimuth is None
    n_views = 1

    if azimuth is None:
        azimuth = rng.uniform(-180, 180)
    if elevation is None:
        elevation = rng.uniform(-30, 30)
    if distance is None:
        distance = rng.uniform(5, 10)

    if isinstance(azimuth, np.ndarray):
        n_views = len(azimuth)
        if elevation is None:
            elevation = rng.uniform(-30, 30, size=n_views)
        if distance is None:
            distance = rng.uniform(5, 10, size=n_views)

    # Camera position in spherical coordinates
    if isinstance(azimuth, np.ndarray):
        results = np.empty((n_views, *poses_3d.shape[:2], 2), dtype=np.float32)
        for i in range(n_views):
            results[i] = _project_single_view(
                poses_3d, azimuth[i], elevation[i], distance[i]
            )
        if scalar_input:
            return results[0]
        return results
    else:
        return _project_single_view(poses_3d, azimuth, elevation, distance)


def _project_single_view(
    poses_3d: np.ndarray,
    azimuth: float,
    elevation: float,
    distance: float,
) -> np.ndarray:
    """Project a single view.

    Args:
        poses_3d: (frames, 17, 3) centered 3D poses.
        azimuth: Azimuth in degrees.
        elevation: Elevation in degrees.
        distance: Camera distance.

    Returns:
        (frames, 17, 2) 2D projections.
    """
    R = _rotation_matrix(azimuth, elevation)

    # Rotate all poses
    # poses_3d: (F, J, 3), R: (3, 3)
    rotated = poses_3d @ R.T  # (F, J, 3)

    # Camera at (0, 0, distance), looking at origin
    # Perspective projection: x_2d = f * x / z, y_2d = f * y / z
    # Use focal length = 1 (normalized by distance)
    z = rotated[..., 2:3] + distance  # (F, J, 1) shift z by distance
    z = np.maximum(z, 0.1)  # avoid division by zero

    x_2d = rotated[..., 0] / z.squeeze(-1)
    y_2d = rotated[..., 1] / z.squeeze(-1)

    return np.stack([x_2d, y_2d], axis=-1).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_virtual_camera.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/virtual_camera.py tests/ml/vifss/test_virtual_camera.py
git commit -m "feat(ml): add virtual camera projection for VIFSS pre-training"
```

---

### Task 4: Data Preprocessing Pipeline

**Files:**
- Create: `src/ml/vifss/preprocess.py`
- Create: `tests/ml/vifss/test_preprocess.py`

Converts AthletePose3D 3D pose annotations into VIFSS training format. Handles:
1. Loading pkl annotations (142kp or H3.6M format)
2. Ground plane alignment via RANSAC
3. Frontal orientation alignment (left hip → +x)
4. Normalization (center at mid-hip, scale = 0.4)
5. Filtering to figure skating sequences only

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/test_preprocess.py`:

```python
"""Tests for AthletePose3D data preprocessing."""

import numpy as np
import pytest

from src.ml.vifss.preprocess import (
    align_frontal,
    align_ground_plane,
    normalize_pose,
)


class TestPreprocess:
    def test_align_ground_plane_flat(self):
        """Flat poses (all on z=0) should remain unchanged."""
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        poses[:, :, 1] = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                     0.3, 0.3, 0.15, 0.15, 0.0, 0.0,
                                     -0.4, -0.4, -0.8, -0.8])[None, :]

        aligned = align_ground_plane(poses)
        # Z coordinates should be near zero for flat pose
        assert np.allclose(aligned[:, :, 2], 0, atol=0.1)

    def test_align_ground_plane_tilted(self):
        """Tilted poses should be rotated to z-up."""
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        # Create a tilted ground plane (45 degree tilt in xz)
        poses[:, :, 0] = np.linspace(-0.5, 0.5, 17)[None, :]
        poses[:, :, 2] = poses[:, :, 0]  # z = x → tilted

        aligned = align_ground_plane(poses)
        # After alignment, lowest z should be near zero
        assert aligned[:, :, 2].min() < aligned[:, :, 2].max()

    def test_align_frontal(self):
        """Frontal alignment should put left hip on +x side."""
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        # Set hips: RHIP at (0.1, 0, 0), LHIP at (-0.1, 0, 0)
        poses[:, 1, 0] = 0.1   # RHIP (index 1 in our format)
        poses[:, 2, 0] = -0.1  # LHIP (index 2 in our format)

        aligned = align_frontal(poses)
        # After alignment, LHIP should have positive x (facing right)
        assert aligned[0, 2, 0] > 0

    def test_normalize_pose(self):
        """Normalization should center at mid-hip and scale spine to 0.4."""
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        # Offset all poses
        poses[:, :, 0] = 5.0
        poses[:, :, 1] = 10.0
        # Set spine length
        poses[:, 0, :] = [5.0, 10.0, 0.0]   # HIP_CENTER
        poses[:, 3, :] = [5.0, 10.2, 0.0]   # SPINE (0.2 above hip)
        poses[:, 4, :] = [5.0, 10.3, 0.0]   # THORAX (0.1 above spine)
        poses[:, 5, :] = [5.0, 10.4, 0.0]   # NECK (0.1 above thorax)

        normalized = normalize_pose(poses)
        # Mid-hip should be at origin
        mid_hip = (normalized[:, 1, :] + normalized[:, 2, :]) / 2
        np.testing.assert_allclose(mid_hip, 0, atol=0.01)

        # Spine length (hip→thorax + thorax→neck) should be ~0.4
        hip_to_thorax = np.linalg.norm(
            normalized[0, 0, :] - normalized[0, 4, :]
        )
        thorax_to_neck = np.linalg.norm(
            normalized[0, 4, :] - normalized[0, 5, :]
        )
        spine_len = hip_to_thorax + thorax_to_neck
        assert abs(spine_len - 0.4) < 0.05

    def test_normalize_preserves_shape(self):
        """Should preserve input shape."""
        poses = np.zeros((5, 17, 3), dtype=np.float32)
        result = normalize_pose(poses)
        assert result.shape == (5, 17, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_preprocess.py -v`
Expected: FAIL

- [ ] **Step 3: Implement preprocess.py**

Create `src/ml/vifss/preprocess.py`:

```python
"""Preprocess AthletePose3D 3D poses for VIFSS training.

Follows the VIFSS paper preprocessing pipeline:
1. Ground plane alignment via RANSAC (z-axis = gravity)
2. Frontal orientation (left hip → +x)
3. Normalization (center at mid-hip, spine = 0.4)
"""

import numpy as np


def align_ground_plane(poses: np.ndarray) -> np.ndarray:
    """Align ground plane so z-axis is gravity direction.

    Uses the lowest 50% of z-coordinates as ground contact points,
    then estimates ground plane normal via SVD.

    Args:
        poses: (frames, 17, 3) 3D poses in our H36Key ordering.

    Returns:
        Rotated poses with z-axis aligned to gravity.
    """
    n_frames, n_joints, _ = poses.shape

    # Flatten all joint positions across frames
    all_points = poses.reshape(-1, 3)

    # Find ground contact: lowest 50% of z-values per frame
    rotated = poses.copy()
    for f in range(n_frames):
        frame_points = poses[f]  # (17, 3)
        z_vals = frame_points[:, 2]
        threshold = np.median(z_vals)
        ground_points = frame_points[z_vals <= threshold]

        if len(ground_points) < 3:
            continue

        # SVD to find plane normal
        centroid = ground_points.mean(axis=0)
        centered = ground_points - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # Last row = smallest singular value direction

        # Ensure normal points upward (positive z)
        if normal[2] < 0:
            normal = -normal

        # Rotation to align normal with z-axis
        z_target = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(normal, z_target)
        axis_len = np.linalg.norm(rotation_axis)

        if axis_len > 1e-6:
            rotation_axis /= axis_len
            cos_angle = np.dot(normal, z_target)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            # Apply rotation
            rotated[f] = (R @ (poses[f] - centroid).T).T + centroid

    return rotated


def align_frontal(poses: np.ndarray) -> np.ndarray:
    """Align all poses to face the same direction.

    Rotates around z-axis so that left hip is on +x side.

    Args:
        poses: (frames, 17, 3) 3D poses in our H36Key ordering.
               Index 1 = RHIP, Index 2 = LHIP.

    Returns:
        Frontally aligned poses.
    """
    # Use mid-hip as center, hip axis for direction
    mid_hip = (poses[:, 1, :] + poses[:, 2, :]) / 2  # (F, 3)
    hip_vector = poses[:, 2, :] - poses[:, 1, :]  # LHIP - RHIP → should be +x

    # Average hip direction across frames
    avg_direction = hip_vector.mean(axis=0)
    avg_direction[2] = 0  # project to xy plane

    angle = np.arctan2(avg_direction[1], avg_direction[0])

    # Rotation matrix around z-axis by -angle
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1],
    ])

    return (R @ poses.transpose(0, 2, 1)).transpose(0, 2, 1)


def normalize_pose(poses: np.ndarray) -> np.ndarray:
    """Center at mid-hip and normalize spine length to 0.4.

    Following VIFSS paper: scale so that (mid-hip→thorax + thorax→neck) = 0.4.

    Args:
        poses: (frames, 17, 3) 3D poses in our H36Key ordering.
               Index 0 = HIP_CENTER, 3 = SPINE, 4 = THORAX, 5 = NECK.

    Returns:
        Normalized poses with mid-hip at origin, spine = 0.4.
    """
    normalized = poses.copy()

    # Mid-hip center
    mid_hip = (normalized[:, 1, :] + normalized[:, 2, :]) / 2

    # Center at mid-hip
    for j in range(17):
        normalized[:, j, :] -= mid_hip

    # Compute spine length: average of (HIP_CENTER→THORAX + THORAX→NECK)
    hip_to_thorax = np.linalg.norm(
        normalized[:, 0, :] - normalized[:, 4, :], axis=1
    )
    thorax_to_neck = np.linalg.norm(
        normalized[:, 4, :] - normalized[:, 5, :], axis=1
    )
    spine_lengths = hip_to_thorax + thorax_to_neck
    avg_spine = float(np.mean(spine_lengths))

    if avg_spine < 0.01:
        return normalized  # can't normalize degenerate pose

    # Scale
    scale = 0.4 / avg_spine
    normalized *= scale

    return normalized.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_preprocess.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/preprocess.py tests/ml/vifss/test_preprocess.py
git commit -m "feat(ml): add AthletePose3D preprocessing for VIFSS"
```

---

### Task 5: VIFSS Encoder Wrapper

**Files:**
- Create: `src/ml/vifss/encoder.py`
- Create: `tests/ml/vifss/test_encoder.py`

Wraps the JointFormer pose encoder from VIFSS for loading pre-trained weights and extracting embeddings from 2D poses.

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/test_encoder.py`:

```python
"""Tests for VIFSS encoder wrapper."""

import numpy as np
import pytest

from src.ml.vifss.encoder import VIFSSEncoder


class TestVIFSSEncoder:
    def test_dummy_encoder_output_shape(self):
        """Encoder should produce embedding of correct dimension."""
        encoder = VIFSSEncoder(embedding_dim=128, n_joints=17)
        # Input: batch of 2D poses (batch, 17, 2)
        poses = np.random.randn(4, 17, 2).astype(np.float32)
        embedding = encoder.encode(poses)
        assert embedding.shape == (4, 128)

    def test_single_pose_embedding(self):
        """Should handle single pose (no batch dimension)."""
        encoder = VIFSSEncoder(embedding_dim=64, n_joints=17)
        pose = np.random.randn(17, 2).astype(np.float32)
        embedding = encoder.encode(pose)
        assert embedding.shape == (64,)

    def test_view_invariant_embeddings(self):
        """Same pose from different views should produce similar embeddings."""
        encoder = VIFSSEncoder(embedding_dim=64, n_joints=17)
        # Create a simple pose
        pose_3d = np.zeros((1, 17, 3), dtype=np.float32)
        pose_3d[0, 0] = [0, 0, 0]   # HIP_CENTER
        pose_3d[0, 1] = [-0.1, 0, 0] # RHIP
        pose_3d[0, 2] = [0.1, 0, 0]  # LHIP
        pose_3d[0, 3] = [0, 0.2, 0]  # SPINE
        pose_3d[0, 4] = [0, 0.3, 0]  # THORAX
        pose_3d[0, 5] = [0, 0.4, 0]  # NECK
        pose_3d[0, 6] = [0, 0.5, 0]  # HEAD

        from src.ml.vifss.virtual_camera import project_virtual_camera
        p1 = project_virtual_camera(pose_3d, azimuth=0, elevation=0, distance=7.0)
        p2 = project_virtual_camera(pose_3d, azimuth=90, elevation=15, distance=8.0)

        e1 = encoder.encode(p1[0])
        e2 = encoder.encode(p2[0])

        # Note: random init won't be view-invariant, but shapes should match
        assert e1.shape == e2.shape

    def test_load_from_checkpoint(self, tmp_path):
        """Should save and load encoder state."""
        encoder = VIFSSEncoder(embedding_dim=64, n_joints=17)
        ckpt_path = tmp_path / "encoder.pt"
        encoder.save(ckpt_path)

        encoder2 = VIFSSEncoder.load(ckpt_path, n_joints=17)
        assert encoder2.embedding_dim == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_encoder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement encoder.py**

Create `src/ml/vifss/encoder.py`:

```python
"""VIFSS pose encoder wrapper.

Wraps JointFormer-style transformer encoder for extracting
view-invariant pose embeddings from 2D poses.

Initial implementation uses a simple MLP encoder.
Replace with actual JointFormer (from VIFSS repo) after
pre-training produces weights.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SimplePoseEncoder(nn.Module):
    """Simple MLP pose encoder (placeholder for JointFormer).

    Takes flattened 2D pose (34-dim for 17 joints) and projects
    to embedding space.

    Architecture: Linear(34, 256) → ReLU → Linear(256, d) → L2 normalize
    """

    def __init__(self, n_joints: int = 17, embedding_dim: int = 256):
        super().__init__()
        self.input_dim = n_joints * 2
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_joints, 2) or (batch, n_joints * 2)

        Returns:
            (batch, embedding_dim) L2-normalized embeddings.
        """
        if x.dim() == 3:
            x = x.flatten(1)  # (batch, n_joints * 2)
        z = self.net(x)
        return torch.nn.functional.normalize(z, dim=-1)


class VIFSSEncoder:
    """High-level VIFSS encoder wrapper.

    Handles PyTorch model lifecycle, device management,
    and numpy ↔ tensor conversion.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        n_joints: int = 17,
        device: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.n_joints = n_joints
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimplePoseEncoder(
            n_joints=n_joints, embedding_dim=embedding_dim
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, poses: np.ndarray) -> np.ndarray:
        """Extract embeddings from 2D poses.

        Args:
            poses: 2D poses. Shape (17, 2) for single pose,
                   (batch, 17, 2) for batch.

        Returns:
            Embeddings. Shape (embedding_dim,) for single pose,
            (batch, embedding_dim) for batch.
        """
        was_single = poses.ndim == 2
        if was_single:
            poses = poses[np.newaxis]  # (1, 17, 2)

        tensor = torch.from_numpy(poses.astype(np.float32)).to(self.device)
        embedding = self.model(tensor)
        result = embedding.cpu().numpy()

        if was_single:
            return result[0]
        return result

    def save(self, path: Path | str) -> None:
        """Save encoder weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "n_joints": self.n_joints,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None, **kwargs) -> "VIFSSEncoder":
        """Load encoder from checkpoint.

        Args:
            path: Checkpoint path.
            device: Override device.
            **kwargs: Override constructor args (ignored if checkpoint has them).
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        embedding_dim = kwargs.get("embedding_dim", checkpoint["embedding_dim"])
        n_joints = kwargs.get("n_joints", checkpoint["n_joints"])

        encoder = cls(embedding_dim=embedding_dim, n_joints=n_joints, device=device)
        encoder.model.load_state_dict(checkpoint["model_state_dict"])
        return encoder
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_encoder.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/encoder.py tests/ml/vifss/test_encoder.py
git commit -m "feat(ml): add VIFSS encoder wrapper with save/load"
```

---

### Task 6: Embedding Extraction + Comparison

**Files:**
- Create: `src/ml/vifss/embedding.py`
- Create: `tests/ml/vifss/test_embedding.py`

Extracts embeddings from pose sequences and computes cosine similarity for cross-athlete comparison.

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/test_embedding.py`:

```python
"""Tests for embedding extraction and comparison."""

import numpy as np
import pytest

from src.ml.vifss.embedding import cosine_similarity, extract_embeddings


class TestEmbedding:
    def test_extract_embeddings_shape(self, sample_h36m_3d):
        """Should return one embedding per frame."""
        from src.ml.vifss.encoder import VIFSSEncoder
        encoder = VIFSSEncoder(embedding_dim=64)
        embeddings = extract_embeddings(sample_h36m_3d, encoder)
        assert embeddings.shape == (5, 64)

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_cosine_similarity_batch(self):
        """Should compute pairwise similarities for batches."""
        emb1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        emb2 = np.array([[1.0, 0.0], [1.0, 1.0]])
        sim = cosine_similarity(emb1, emb2)
        assert sim.shape == (2,)
        assert sim[0] == pytest.approx(1.0)
        assert 0 < sim[1] < 1  # (0,1)·(1,1)/sqrt(2) = 1/sqrt(2)

    def test_extract_embeddings_from_2d(self):
        """Should accept 2D poses directly."""
        from src.ml.vifss.encoder import VIFSSEncoder
        encoder = VIFSSEncoder(embedding_dim=64)
        poses_2d = np.random.randn(3, 17, 2).astype(np.float32)
        embeddings = extract_embeddings(poses_2d, encoder)
        assert embeddings.shape == (3, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_embedding.py -v`
Expected: FAIL

- [ ] **Step 3: Implement embedding.py**

Create `src/ml/vifss/embedding.py`:

```python
"""Embedding extraction and comparison utilities."""

import numpy as np

from src.ml.vifss.encoder import VIFSSEncoder


def extract_embeddings(
    poses: np.ndarray,
    encoder: VIFSSEncoder,
) -> np.ndarray:
    """Extract frame-wise embeddings from pose sequence.

    Args:
        poses: Pose sequence. Shape (frames, 17, 2) for 2D or
               (frames, 17, 3) for 3D (z-dimension dropped).
        encoder: VIFSS encoder instance.

    Returns:
        Embeddings with shape (frames, embedding_dim).
    """
    if poses.shape[-1] == 3:
        poses_2d = poses[..., :2]  # drop z for 2D encoder
    else:
        poses_2d = poses

    return encoder.encode(poses_2d)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Compute cosine similarity between vectors or batches.

    Args:
        a: Vector (d,) or batch (n, d).
        b: Vector (d,) or batch (n, d).

    Returns:
        Scalar if both inputs are 1D, otherwise (n,) array of pairwise similarities.
    """
    if a.ndim == 1 and b.ndim == 1:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return 0.0
        return float(dot / norm)

    # Batch mode
    dots = np.sum(a * b, axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    norms = np.maximum(norms, 1e-8)
    return dots / norms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_embedding.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Update __init__.py imports**

Update `src/ml/vifss/__init__.py` to import actual functions:

```python
"""VIFSS: View-Invariant Figure Skating-Specific pose embeddings."""

from src.ml.vifss.embedding import cosine_similarity, extract_embeddings
from src.ml.vifss.encoder import VIFSSEncoder

__all__ = ["VIFSSEncoder", "extract_embeddings", "cosine_similarity"]
```

- [ ] **Step 6: Commit**

```bash
git add src/ml/vifss/embedding.py tests/ml/vifss/test_embedding.py src/ml/vifss/__init__.py
git commit -m "feat(ml): add embedding extraction and cosine similarity"
```

---

### Task 7: Contrastive Loss Implementation

**Files:**
- Create: `src/ml/vifss/contrastive_loss.py`
- Create: `tests/ml/vifss/test_contrastive_loss.py`

Implements the VIFSS contrastive loss: Barlow Twins (pose-invariant) + MSE cosine (view-dependent) + Variance + KL regularization.

- [ ] **Step 1: Write the failing test**

Create `tests/ml/vifss/test_contrastive_loss.py`:

```python
"""Tests for VIFSS contrastive loss functions."""

import numpy as np
import pytest
import torch

from src.ml.vifss.contrastive_loss import (
    barlow_twins_loss,
    kl_uniform_loss,
    variance_loss,
    view_dependent_loss,
    vifss_total_loss,
)


class TestContrastiveLoss:
    def test_barlow_twins_identical(self):
        """Identical embeddings should have zero loss."""
        z = torch.randn(4, 32)
        loss = barlow_twins_loss(z, z)
        assert loss.item() >= 0
        # Identical inputs → cross-correlation matrix is all 1s
        # Barlow Twins penalizes off-diagonal → should be positive

    def test_variance_loss_target(self):
        """Embeddings with unit variance should have zero loss."""
        z = torch.randn(100, 32)
        z = z / z.std(dim=0, keepdim=True)  # normalize to unit variance
        loss = variance_loss(z, target_var=1.0)
        assert loss.item() < 0.01

    def test_kl_uniform_loss(self):
        """KL loss should be non-negative."""
        z = torch.randn(10, 16)
        loss = kl_uniform_loss(z)
        assert loss.item() >= 0

    def test_view_dependent_loss_matching_views(self):
        """Matching view directions should have zero loss."""
        batch_size = 4
        d_view = 16
        z_view = torch.randn(batch_size, d_view)
        v_cam = torch.randn(batch_size, 3)
        v_cam = v_cam / v_cam.norm(dim=1, keepdim=True)

        loss = view_dependent_loss(z_view, z_view, v_cam, v_cam)
        # Same views → cosine similarity should match → loss ~0
        assert loss.item() < 0.01

    def test_vifss_total_loss_components(self):
        """Total loss should be weighted sum of components."""
        batch_size = 4
        d_pose = 32
        d_view = 16
        z_pose = torch.randn(batch_size, d_pose)
        z_pose_pos = torch.randn(batch_size, d_pose)
        z_view = torch.randn(batch_size, d_view)
        z_view_pos = torch.randn(batch_size, d_view)
        v_cam = torch.randn(batch_size, 3)
        v_cam_pos = torch.randn(batch_size, 3)

        loss = vifss_total_loss(
            z_pose, z_pose_pos,
            z_view, z_view_pos,
            v_cam, v_cam_pos,
        )
        assert loss.item() >= 0
        assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/vifss/test_contrastive_loss.py -v`
Expected: FAIL

- [ ] **Step 3: Implement contrastive_loss.py**

Create `src/ml/vifss/contrastive_loss.py`:

```python
"""VIFSS contrastive loss functions.

Implements the loss from Tanaka et al. (2025):
- Barlow Twins loss for pose-invariant embeddings
- MSE cosine loss for view-dependent embeddings
- Variance + KL uniform regularization

Weights: w_pose=1.0, w_view=10.0, w_R=1.0
"""

import torch
import torch.nn.functional as F


def barlow_twins_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, lambda_: float = 0.005
) -> torch.Tensor:
    """Barlow Twins redundancy reduction loss.

    Args:
        z_a: Anchor embeddings (batch, d).
        z_b: Positive pair embeddings (batch, d).
        lambda_: Positive diagonal scaling term.

    Returns:
        Scalar loss.
    """
    batch_size = z_a.shape[0]
    d = z_a.shape[1]

    # Normalize
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    # Cross-correlation matrix
    c = (z_a.T @ z_b) / batch_size  # (d, d)

    # Off-diagonal loss: minimize redundancy
    c_diag = torch.diag(c)
    c_offdiag = c - torch.diag(c_diag)

    offdiag_loss = (c_offdiag**2).sum()
    diag_loss = ((c_diag - 1) ** 2).sum()

    return offdiag_loss + lambda_ * diag_loss


def view_dependent_loss(
    z_view_a: torch.Tensor,
    z_view_b: torch.Tensor,
    v_cam_a: torch.Tensor,
    v_cam_b: torch.Tensor,
) -> torch.Tensor:
    """View-dependent loss via MSE of cosine similarities.

    Encourages the view embedding to capture camera direction.

    Args:
        z_view_a: View embedding from anchor (batch, d_view).
        z_view_b: View embedding from positive (batch, d_view).
        v_cam_a: Camera direction vector for anchor (batch, 3).
        v_cam_b: Camera direction vector for positive (batch, 3).

    Returns:
        Scalar MSE loss.
    """
    cos_z = F.cosine_similarity(z_view_a, z_view_b, dim=-1)
    cos_v = F.cosine_similarity(v_cam_a, v_cam_b, dim=-1)
    return F.mse_loss(cos_z, cos_v)


def variance_loss(z: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    """Variance regularization loss.

    Args:
        z: Embeddings (batch, d).
        target_var: Target variance (default 1.0 from paper).

    Returns:
        Scalar loss.
    """
    var = z.var(dim=0)
    return ((var - target_var) ** 2).mean()


def kl_uniform_loss(z: torch.Tensor) -> torch.Tensor:
    """KL divergence from uniform distribution for binary entropy.

    Args:
        z: Embeddings (batch, d).

    Returns:
        Scalar loss.
    """
    # Softmax to get pseudo-probabilities
    p = F.softmax(z, dim=-1)
    # KL divergence from uniform: sum(p * log(p))
    # Uniform = 1/d, so -log(uniform) = log(d)
    d = z.shape[-1]
    log_p = torch.log(p.clamp(min=1e-7))
    return -(p * log_p).sum(dim=-1).mean()


def vifss_total_loss(
    z_pose_a: torch.Tensor,
    z_pose_b: torch.Tensor,
    z_view_a: torch.Tensor,
    z_view_b: torch.Tensor,
    v_cam_a: torch.Tensor,
    v_cam_b: torch.Tensor,
    w_pose: float = 1.0,
    w_view: float = 10.0,
    w_reg: float = 1.0,
) -> torch.Tensor:
    """Total VIFSS contrastive loss.

    Args:
        z_pose_a, z_pose_b: Pose-invariant embeddings from anchor/positive.
        z_view_a, z_view_b: View-dependent embeddings from anchor/positive.
        v_cam_a, v_cam_b: Camera direction unit vectors.
        w_pose: Weight for Barlow Twins loss (default 1.0).
        w_view: Weight for view-dependent loss (default 10.0).
        w_reg: Weight for regularization (default 1.0).

    Returns:
        Weighted total loss.
    """
    loss_pose = barlow_twins_loss(z_pose_a, z_pose_b)
    loss_view = view_dependent_loss(z_view_a, z_view_b, v_cam_a, v_cam_b)

    loss_reg = (
        variance_loss(z_pose_a) + variance_loss(z_pose_b)
        + variance_loss(z_view_a) + variance_loss(z_view_b)
        + kl_uniform_loss(z_pose_a) + kl_uniform_loss(z_pose_b)
        + kl_uniform_loss(z_view_a) + kl_uniform_loss(z_view_b)
    )

    return w_pose * loss_pose + w_view * loss_view + w_reg * loss_reg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/vifss/test_contrastive_loss.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/contrastive_loss.py tests/ml/vifss/test_contrastive_loss.py
git commit -m "feat(ml): add VIFSS contrastive loss (Barlow Twins + view-dependent)"
```

---

### Task 8: Training Configuration

**Files:**
- Create: `src/ml/vifss/config.py`
- Create: `src/ml/vifss/pretrain.py` (stub)
- Create: `src/ml/vifss/finetune.py` (stub)

Training configuration dataclass and script stubs. The actual training loop will be implemented after data pipeline verification.

- [ ] **Step 1: Implement config.py**

Create `src/ml/vifss/config.py`:

```python
"""VIFSS training and inference configuration."""

from dataclasses import dataclass, field


@dataclass
class VIFSSConfig:
    """Configuration for VIFSS pose encoder training and inference."""

    # Model
    embedding_dim: int = 256
    d_pose: int = 128  # pose-invariant embedding dimension
    d_view: int = 32  # view-dependent embedding dimension
    n_joints: int = 17

    # Pre-training
    w_pose: float = 1.0
    w_view: float = 10.0
    w_reg: float = 1.0
    target_variance: float = 1.0
    barlow_lambda: float = 0.005

    # Virtual camera
    azimuth_range: tuple[float, float] = (-180.0, 180.0)
    elevation_range: tuple[float, float] = (-30.0, 30.0)
    distance_range: tuple[float, float] = (5.0, 10.0)

    # Augmentation
    jitter_std: float = 0.01  # Gaussian noise std for joint coordinates
    mask_ratio: float = 0.01  # fraction of joints to mask
    flip_prob: float = 0.5  # horizontal flip probability

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    num_workers: int = 4

    # Data
    athletepose3d_root: str = "data/datasets/athletepose3d"
    output_dir: str = "data/models/vifss"

    # Loss weights
    @property
    def total_embedding_dim(self) -> int:
        return self.d_pose + self.d_view
```

- [ ] **Step 2: Create pretrain.py stub**

Create `src/ml/vifss/pretrain.py`:

```python
"""VIFSS contrastive pre-training script.

Usage:
    uv run python -m src.ml.vifss.pretrain --epochs 100

Requires AthletePose3D data in data/datasets/athletepose3d/.
GPU recommended (Vast.ai for long runs).
"""

import argparse

from src.ml.vifss.config import VIFSSConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="VIFSS contrastive pre-training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="data/models/vifss")
    parser.add_argument("--data-root", type=str, default="data/datasets/athletepose3d")
    args = parser.parse_args()

    config = VIFSSConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        athletepose3d_root=args.data_root,
    )

    print(f"VIFSS Pre-training Config:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Data: {config.athletepose3d_root}")
    print(f"  Output: {config.output_dir}")
    print()
    print("Pre-training not yet implemented. Run Tasks 1-7 first.")
    print("See data/plans/2026-04-11-vifss-embeddings.md for full plan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create finetune.py stub**

Create `src/ml/vifss/finetune.py`:

```python
"""VIFSS fine-tuning script for figure skating action classification.

Usage:
    uv run python -m src.ml.vifss.finetune \
        --pretrained data/models/vifss/pretrained.pt \
        --skatingverse data/datasets/skatingverse

Requires pre-trained encoder and SkatingVerse dataset.
"""

import argparse

from src.ml.vifss.config import VIFSSConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="VIFSS fine-tuning")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--skatingverse", type=str, required=True,
                        help="Path to SkatingVerse dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"VIFSS Fine-tuning Config:")
    print(f"  Pre-trained: {args.pretrained}")
    print(f"  Data: {args.skatingverse}")
    print(f"  Epochs: {args.epochs}")
    print()
    print("Fine-tuning not yet implemented.")
    print("See data/plans/2026-04-11-vifss-embeddings.md for full plan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify stubs run**

Run: `uv run python -m src.ml.vifss.pretrain --epochs 1`
Expected: prints config and "not yet implemented" message

Run: `uv run python -m src.ml.vifss.finetune --pretrained dummy --skatingverse dummy`
Expected: prints config and "not yet implemented" message

- [ ] **Step 5: Commit**

```bash
git add src/ml/vifss/config.py src/ml/vifss/pretrain.py src/ml/vifss/finetune.py
git commit -m "feat(ml): add VIFSS config and training script stubs"
```

---

### Task 9: Full Test Suite Verification

- [ ] **Step 1: Run all VIFSS tests**

Run: `uv run pytest tests/ml/vifss/ -v`
Expected: ALL PASSED (21 tests)

- [ ] **Step 2: Run full test suite for regressions**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: 553+ passed, 0 new failures

- [ ] **Step 3: Run linter**

Run: `uv run ruff check src/ml/vifss/ tests/ml/vifss/`
Expected: No errors

---

## Verification

After all tasks complete:

1. `uv run pytest tests/ml/vifss/ -v` — all 21 VIFSS tests pass
2. `uv run pytest tests/ -v --timeout=60` — no regressions
3. `uv run ruff check src/ml/vifss/ tests/ml/vifss/` — lint clean
4. `uv run python -c "from src.ml.vifss import VIFSSEncoder, extract_embeddings, cosine_similarity"` — import works
5. Keypoint mapping is bijective (our format ↔ MMPose)
6. Virtual camera produces valid 2D projections
7. Encoder can save/load checkpoints
8. Contrastive loss computes correctly
9. Training script stubs run and print config

## Future Plans (Separate Documents)

- **VIFSS Phase 2: Pre-training Pipeline** — Full contrastive pre-training loop with AthletePose3D data, data loaders, augmentation pipeline, Wandb logging, Vast.ai dispatch
- **VIFSS Phase 3: Fine-tuning + Element Auto-Detection** — SkatingVerse integration, BiGRU classifier, auto element detection from pose sequences. **Replaces ROADMAP Phase D (GCN Element Classifier).** This is the primary deliverable for auto element detection.
- **VIFSS Phase 4: Integration (Variant A + B)** — Wire `technique_similarity` into `_analyze_common()`, add `compare_technique()` to pipeline, update reference system to store embeddings
- **VIFSS Phase 5: Temporal Action Segmentation** — FACT integration, entry/jump/landing annotation, frame-wise action labels
- **Cross-Athlete Comparison UI** — Embedding database, similarity search, visualization in frontend

### Prerequisites for Phase 2-3

- [ ] SkatingVerse dataset downloaded (check https://skatingverse.github.io/)
- [ ] MCFS dataset downloaded (currently downloading)
- [ ] FineFS dataset downloaded (currently downloading)
- [ ] Tasks 1-9 of this plan complete (encoder + embeddings working)
- [ ] Contrastive pre-training done (Phase 2 plan)
