# RTMPose Fine-Tune Dataset Preparation (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert AthletePose3D pre-computed COCO 17kp annotations + 3D mocap foot keypoints into COCO JSON (HALPE26 26kp) for RTMPose fine-tuning.

**Architecture:** Load `_coco.npy` (17kp, 1920x1088) as base → project 6 foot keypoints from 3D mocap → add 3 face duplicate keypoints → scale to 1280x768 → COCO JSON. Use pre-extracted `pose_2d/` images.

**Tech Stack:** Python, numpy, opencv, json, tqdm. No external ML deps needed for dataset prep.

---

## Key Discovery: Pre-computed 2D Projections Exist

AthletePose3D already provides `_coco.npy` files (17kp, video resolution 1920x1088) for all 5154 sequences. We only need to add 9 keypoints:
- 6 foot keypoints (indices 17-22): project from 3D mocap
- 3 face duplicate keypoints (indices 23-25): copy from existing face points

**Critical projection formula** (from AthletePose3D `utils/demo.py`):
```python
rot_mat = np.array(cam['extrinsic_matrix'])
rot_mat[1:, :] *= -1  # Flip Y and Z rows — THIS IS THE KEY

camera_position = np.array(cam['xyz'])
translated = p_world - camera_position
kpts_camera = rot_mat @ translated

u = fu * (kpts_camera[0] / kpts_camera[2]) + cu
v = fv * (kpts_camera[1] / kpts_camera[2]) + cv
```

Verified: matches `_coco.npy` within 0-15px for all body keypoints.

---

## Data Inventory

| Source | Format | Size | Use |
|--------|--------|------|-----|
| `videos/*/_coco.npy` | (N, 17, 2) float64 | 5154 files | Base 17kp annotations |
| `videos/*/.npy` | (N, 142, 3) float64 | 5154 files | 3D mocap (source for foot kps) |
| `videos/*/.json` | metadata | 5154 files | cam name, fps, keypoint names |
| `pose_2d/annotations/*.json` | COCO JSON | 3 files | 17kp at 1280x768 |
| `pose_2d/{split}/*.jpg` | 1280x768 images | ~130k files | Training images |
| `cam_param.json` | K, R, t, dist | 24 cameras | Projection |

---

## File Structure

```
src/datasets/
├── __init__.py                       # Package init
├── projector.py                      # 3D→2D foot keypoint projection
├── coco_builder.py                   # COCO JSON builder (HALPE26 26kp)
└── merge_coco.py                     # Merge _coco.npy + projected foot kps

scripts/
├── validate_projection.py            # Visual validation tool
└── prepare_athletepose3d.py          # Batch converter entry point

tests/
├── test_projection.py                # Camera projection tests
└── test_coco_builder.py              # COCO JSON format tests

data/datasets/athletepose3d/
├── coco_annotations/                 # OUTPUT: final COCO JSON for mmpose
│   ├── athletepose3d_train.json
│   └── athletepose3d_val.json
├── pose_2d/                          # Pre-extracted images (1280x768)
│   ├── annotations/
│   └── {train,valid,test}_set/
└── videos/                           # Raw data (unchanged)
```

---

## Keypoint Mapping: 142kp → HALPE26 26kp

| HALPE26 Idx | Name | Source | Visibility |
|:---:|---|---|:---:|
| 0-16 | COCO 17kp | `_coco.npy` directly | From _coco (2.0 or 0.0) |
| 17 | left_heel | AP3D idx 49 (LHEL), project from 3D | 1.0 |
| 18 | left_big_toe | AP3D idx 26 (L_Toe), project from 3D | 1.0 |
| 19 | left_small_toe | AP3D idx 10 (L_5th MTP), project from 3D | 1.0 |
| 20 | right_heel | AP3D idx 112 (RHEL), project from 3D | 1.0 |
| 21 | right_big_toe | AP3D idx 93 (R_Toe), project from 3D | 1.0 |
| 22 | right_small_toe | AP3D idx 77 (R_5th_MTP), project from 3D | 1.0 |
| 23 | left_eye_inner | Copy from HALPE26 idx 1 (left_eye) | 0.3 |
| 24 | right_eye_inner | Copy from HALPE26 idx 2 (right_eye) | 0.3 |
| 25 | mouth | Copy from HALPE26 idx 0 (nose) | 0.3 |

---

### Task 1: Foot Keypoint Projector

**Files:**
- Create: `src/datasets/__init__.py`
- Create: `src/datasets/projector.py`
- Test: `tests/test_projection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_projection.py
"""Tests for foot keypoint 3D→2D projection."""
import json
import numpy as np
import pytest

DATA_ROOT = "data/datasets/athletepose3d"


@pytest.fixture
def cam_params():
    with open(f"{DATA_ROOT}/cam_param.json") as f:
        return json.load(f)


@pytest.fixture
def sample_3d():
    return np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1.npy")[0]


@pytest.fixture
def sample_coco():
    return np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1_coco.npy")


class TestFootProjection:
    def test_project_foot_point_in_frame(self, cam_params, sample_3d):
        """Projected LHEL should be within video bounds (1920x1088)."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        p_world = sample_3d[49]  # LHEL
        x, y = project_point(p_world, cam)

        assert 0 <= x <= 1920, f"x={x} out of bounds"
        assert 0 <= y <= 1088, f"y={y} out of bounds"

    def test_foot_points_near_ankle(self, cam_params, sample_3d, sample_coco):
        """LHEL should be within 50px of COCO LANK."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        # Use frame where keypoints are visible (frame 50)
        lank_x, lank_y = sample_coco[50, 15]
        lhel_x, lhel_y = project_point(sample_3d[50, 49], cam)

        dist = np.sqrt((lhel_x - lank_x) ** 2 + (lhel_y - lank_y) ** 2)
        assert dist < 50, f"LHEL to LANK: {dist:.0f}px > 50px"

    def test_projection_matches_coco_body(self, cam_params, sample_3d, sample_coco):
        """Our projection should match _coco.npy within 20px for LANK."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        our_x, our_y = project_point(sample_3d[50, 33], cam)  # LANK = AP3D idx 33
        coco_x, coco_y = sample_coco[50, 15]  # COCO kp15 = left_ankle

        dist = np.sqrt((our_x - coco_x) ** 2 + (our_y - coco_y) ** 2)
        assert dist < 20, f"LANK projection dist={dist:.0f}px > 20px"

    def test_project_foot_frame_returns_6x2(self, cam_params, sample_3d):
        """project_foot_frame returns (6, 2) for foot keypoints."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        result = project_foot_frame(sample_3d, cam)

        assert result.shape == (6, 2), f"Expected (6, 2), got {result.shape}"

    def test_both_cameras_produce_valid_coords(self, cam_params, sample_3d):
        """Camera 1 and Camera 6 both produce in-frame foot coords."""
        from src.datasets.projector import project_foot_frame

        for cam_name in ["fs_camera_1", "fs_camera_6"]:
            cam = cam_params[cam_name]
            pts = project_foot_frame(sample_3d, cam)
            assert not np.any(np.isnan(pts)), f"NaN in {cam_name}"
            # At least some foot points should be in-frame
            valid = (pts[:, 0] >= 0) & (pts[:, 0] <= 1920) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1088)
            assert valid.any(), f"No valid foot points for {cam_name}"

    def test_nan_3d_returns_nan(self, cam_params):
        """NaN input should return NaN output."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        x, y = project_point(np.array([np.nan, 0.0, 0.0]), cam)
        assert np.isnan(x)
        assert np.isnan(y)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_projection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/datasets/__init__.py` and `src/datasets/projector.py`**

```python
# src/datasets/__init__.py
```

```python
# src/datasets/projector.py
"""Foot keypoint 3D→2D camera projection for AthletePose3D dataset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# AthletePose3D 142kp indices for HALPE26 foot keypoints (17-22)
FOOT_AP3D_INDICES = np.array([49, 26, 10, 112, 93, 77], dtype=np.intp)
# LHEL, L_Toe, L_5th_MTP, RHEL, R_Toe, R_5th_MTP


def _load_camera(cam: dict) -> tuple[NDArray, NDArray, NDArray, float, float, float, float]:
    """Parse camera dict into projection components."""
    K = np.array(cam["affine_intrinsics_matrix"], dtype=np.float64)
    rot_mat = np.array(cam["extrinsic_matrix"], dtype=np.float64)
    rot_mat[1:, :] *= -1  # Critical: flip Y and Z rows (AthletePose3D convention)
    t = np.array(cam["xyz"], dtype=np.float64)
    return K, rot_mat, t, K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def project_point(
    p_world: NDArray[np.float64],
    cam: dict,
) -> tuple[float, float]:
    """Project a single 3D world point to 2D pixel coordinates.

    Uses AthletePose3D convention: rot_mat[1:, :] *= -1 before projection.
    """
    if np.isnan(p_world).any():
        return (np.nan, np.nan)

    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    translated = p_world - t
    kpts_camera = rot_mat @ translated

    if kpts_camera[2] <= 0:
        return (np.nan, np.nan)

    u = fu * (kpts_camera[0] / kpts_camera[2]) + cu
    v = fv * (kpts_camera[1] / kpts_camera[2]) + cv
    return (float(u), float(v))


def project_foot_frame(
    keypoints_3d: NDArray[np.float64],
    cam: dict,
) -> NDArray[np.float32]:
    """Project foot keypoints from 3D to 2D.

    Args:
        keypoints_3d: (142, 3) array of 3D world coordinates.
        cam: Camera dict from cam_param.json.

    Returns:
        (6, 2) projected 2D coordinates for HALPE26 foot keypoints 17-22.
    """
    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    pts = np.zeros((6, 2), dtype=np.float32)

    for i, ap3d_idx in enumerate(FOOT_AP3D_INDICES):
        p = keypoints_3d[ap3d_idx]
        if np.isnan(p).any():
            pts[i] = [np.nan, np.nan]
            continue

        translated = p - t
        kc = rot_mat @ translated
        if kc[2] <= 0:
            pts[i] = [np.nan, np.nan]
            continue

        pts[i] = [fu * kc[0] / kc[2] + cu, fv * kc[1] / kc[2] + cv]

    return pts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_projection.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/datasets/__init__.py src/datasets/projector.py tests/test_projection.py
git commit -m "feat(datasets): add foot keypoint 3D→2D projection for AthletePose3D"
```

---

### Task 2: COCO Builder + Merger

**Files:**
- Create: `src/datasets/coco_builder.py`
- Test: `tests/test_coco_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coco_builder.py
"""Tests for COCO JSON generation with HALPE26 26kp format."""
import json
import numpy as np
import pytest

from src.datasets.coco_builder import (
    HALPE26_KEYPOINT_NAMES,
    HALPE26_SKELETON,
    DEFAULT_CATEGORY,
    merge_coco_foot_keypoints,
    format_keypoints,
    build_coco_json,
)


class TestCocoBuilder:
    def test_keypoint_names_count(self):
        assert len(HALPE26_KEYPOINT_NAMES) == 26

    def test_category_is_person(self):
        assert DEFAULT_CATEGORY["name"] == "person"
        assert len(DEFAULT_CATEGORY["keypoint_names"]) == 26
        assert DEFAULT_CATEGORY["skeleton"] is not None

    def test_skeleton_edges(self):
        """Skeleton should connect foot keypoints to ankles."""
        skel = HALPE26_SKELETON
        # left_heel(17) should connect to left_ankle(15)
        assert [15, 17] in skel or [17, 15] in skel
        # right_heel(20) should connect to right_ankle(16)
        assert [16, 20] in skel or [20, 16] in skel

    def test_merge_produces_26kp(self):
        """Merging 17kp COCO + 6 foot points + 3 face dupes = 26kp."""
        coco_kp = np.random.rand(17, 2) * 1000
        foot_kp = np.random.rand(6, 2) * 1000
        merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        assert merged.shape == (26, 2)
        assert vis.shape == (26,)
        assert np.all(vis[:17] > 0)  # COCO 17kp from _coco.npy are valid
        assert np.all(vis[17:23] > 0)  # foot keypoints
        assert np.all(vis[23:] > 0)  # face dupes

    def test_merge_preserves_coco_coords(self):
        """COCO 17kp coordinates should be unchanged."""
        coco_kp = np.array([[100.0, 200.0], [150.0, 250.0]] + [[0.0, 0.0]] * 15)
        foot_kp = np.random.rand(6, 2) * 1000
        merged, _ = merge_coco_foot_keypoints(coco_kp, foot_kp)

        np.testing.assert_array_almost_equal(merged[0], [100.0, 200.0])
        np.testing.assert_array_almost_equal(merged[1], [150.0, 250.0])

    def test_face_dupes_copy_from_existing(self):
        """Face dupes (23-25) should copy from idx 0, 1, 2."""
        coco_kp = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]] + [[0.0, 0.0]] * 14)
        foot_kp = np.random.rand(6, 2) * 1000
        merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        # idx 23 (left_eye_inner) = copy of idx 1 (left_eye)
        np.testing.assert_array_almost_equal(merged[23], [110.0, 210.0])
        # idx 24 (right_eye_inner) = copy of idx 2 (right_eye)
        np.testing.assert_array_almost_equal(merged[24], [120.0, 220.0])
        # idx 25 (mouth) = copy of idx 0 (nose)
        np.testing.assert_array_almost_equal(merged[25], [100.0, 200.0])
        # Face dupes should have low visibility
        assert vis[23] == 0.3
        assert vis[24] == 0.3
        assert vis[25] == 0.3

    def test_nan_foot_sets_zero_vis(self):
        """NaN foot keypoints should get zero visibility."""
        coco_kp = np.ones((17, 2))
        foot_kp = np.array([[np.nan, np.nan], [1.0, 2.0], [3.0, 4.0],
                            [5.0, 6.0], [np.nan, np.nan], [7.0, 8.0]])
        merged, vis = merge_coco_foot_keypoints(coco_kp, foot_kp)

        assert vis[17] == 0.0  # NaN foot
        assert vis[18] == 1.0  # valid foot
        assert vis[21] == 0.0  # NaN foot
        assert vis[22] == 1.0  # valid foot

    def test_format_keypoints_flat(self):
        """format_keypoints should produce flat [x,y,v,x,y,v,...] list."""
        pts = np.random.rand(26, 2)
        vis = np.ones(26)
        kp = format_keypoints(pts, vis)

        assert isinstance(kp, list)
        assert len(kp) == 26 * 3
        assert kp[2] == 1.0  # first visibility

    def test_build_coco_json_structure(self):
        images = [{"file_name": "test.jpg", "id": 1, "width": 1280, "height": 768}]
        annotations = [{"image_id": 1, "id": 1, "keypoints": [0.0] * 78, "num_keypoints": 26, "bbox": [0, 0, 100, 100]}]
        result = build_coco_json(images, annotations)

        assert "images" in result
        assert "annotations" in result
        assert "categories" in result
        assert result["categories"][0]["name"] == "person"
        assert len(result["categories"][0]["keypoint_names"]) == 26
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coco_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/datasets/coco_builder.py`**

```python
# src/datasets/coco_builder.py
"""Build COCO-style HALPE26 26kp annotation JSON for mmpose training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


HALPE26_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "left_big_toe", "left_small_toe",
    "right_heel", "right_big_toe", "right_small_toe",
    "left_eye_inner", "right_eye_inner", "mouth",
]

HALPE26_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],                   # head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],           # upper body
    [5, 11], [6, 12], [11, 12],                         # torso
    [11, 13], [13, 15], [12, 14], [14, 16],             # legs
    [15, 17], [16, 20], [17, 18], [18, 19],             # left foot
    [20, 21], [21, 22],                                  # right foot
    [23, 24], [24, 25],                                  # face dupes
]

DEFAULT_CATEGORY = {
    "supercategory": "person",
    "id": 1,
    "name": "person",
    "keypoint_names": HALPE26_KEYPOINT_NAMES,
    "skeleton": HALPE26_SKELETON,
}

# Face dupe mapping: HALPE26 idx 23-25 copy from these indices
_FACE_DUPE_SOURCE = {23: 1, 24: 2, 25: 0}


def merge_coco_foot_keypoints(
    coco_2d: NDArray[np.float64],
    foot_2d: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Merge COCO 17kp + foot 6kp + face 3dupes into HALPE26 26kp.

    Args:
        coco_2d: (17, 2) COCO keypoints from _coco.npy (may contain NaN).
        foot_2d: (6, 2) projected foot keypoints (may contain NaN).

    Returns:
        pts: (26, 2) merged HALPE26 coordinates.
        vis: (26,) visibility per keypoint.
    """
    pts = np.zeros((26, 2), dtype=np.float32)
    vis = np.zeros(26, dtype=np.float32)

    # COCO 17kp (indices 0-16): visibility based on NaN check
    for i in range(17):
        if not np.isnan(coco_2d[i]).any():
            pts[i] = coco_2d[i]
            vis[i] = 2.0  # COCO standard: visible
        else:
            vis[i] = 0.0

    # Foot 6kp (indices 17-22)
    for i in range(6):
        if not np.isnan(foot_2d[i]).any():
            pts[17 + i] = foot_2d[i]
            vis[17 + i] = 2.0
        else:
            vis[17 + i] = 0.0

    # Face duplicates (indices 23-25): copy from existing, low visibility
    for dup_idx, src_idx in _FACE_DUPE_SOURCE.items():
        pts[dup_idx] = pts[src_idx]
        vis[dup_idx] = 0.3 if vis[src_idx] > 0 else 0.0

    return pts, vis


def format_keypoints(
    pts: NDArray[np.float32],
    vis: NDArray[np.float32],
) -> list[float]:
    """Flatten keypoints to COCO format: [x1, y1, v1, x2, y2, v2, ...]."""
    kp = []
    for i in range(26):
        kp.extend([float(pts[i, 0]), float(pts[i, 1]), float(vis[i])])
    return kp


def build_coco_json(
    images: list[dict],
    annotations: list[dict],
) -> dict:
    """Build a COCO-style annotation dict."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": [DEFAULT_CATEGORY],
    }


def save_coco_json(data: dict, output_path: str) -> None:
    """Save COCO JSON to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_coco_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/datasets/coco_builder.py tests/test_coco_builder.py
git commit -m "feat(datasets): add COCO JSON builder with HALPE26 merge logic"
```

---

### Task 3: Validation Visualizer

**Files:**
- Create: `scripts/validate_projection.py`

- [ ] **Step 1: Write validation script**

```python
#!/usr/bin/env python3
"""Visual validation of _coco.npy + projected foot keypoints.

Usage:
    uv run python scripts/validate_projection.py
    uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50
    uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50 --save /tmp/halpe26_check.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.projector import project_foot_frame
from src.datasets.coco_builder import merge_coco_foot_keypoints

DATA_ROOT = Path("data/datasets/athletepose3d")

HALPE26_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (16, 20), (17, 18), (18, 19),
    (20, 21), (21, 22),
    (23, 24), (24, 25),
]

HALPE26_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb", "L_wri", "R_wri",
    "L_hip", "R_hip", "L_knee", "R_knee", "L_ank", "R_ank",
    "L_heel", "L_bigtoe", "L_smtoe", "R_heel", "R_bigtoe", "R_smtoe",
    "L_eye_in", "R_eye_in", "mouth",
]

# Green for COCO 17, blue for foot 6, magenta for face dupes
KP_COLORS = (
    [(0, 255, 0)] * 17 + [(255, 100, 0)] * 6 + [(255, 0, 255)] * 3
)


def find_sequence(split: str, sequence: str, camera: int) -> Path | None:
    base = DATA_ROOT / "videos" / split
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        npy = d / f"{sequence}_cam_{camera}.npy"
        coco_npy = d / f"{sequence}_cam_{camera}_coco.npy"
        mp4 = d / f"{sequence}_cam_{camera}.mp4"
        json_f = d / f"{sequence}_cam_{camera}.json"
        if npy.exists() and coco_npy.exists() and mp4.exists() and json_f.exists():
            return npy
    return None


def draw_halpe26(frame: np.ndarray, pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    overlay = frame.copy()

    for a, b in HALPE26_SKELETON:
        if vis[a] > 0.1 and vis[b] > 0.1:
            cv2.line(overlay, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), (180, 180, 180), 1, cv2.LINE_AA)

    for i in range(26):
        if vis[i] < 0.1:
            continue
        color = KP_COLORS[i]
        pt = (int(pts[i, 0]), int(pts[i, 1]))
        radius = 4 if i < 17 else 6
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)

    # Label foot points
    for i in range(17, 23):
        if vis[i] > 0.1:
            pt = (int(pts[i, 0]), int(pts[i, 1]))
            cv2.putText(overlay, HALPE26_NAMES[i][:7], (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HALPE26 keypoint projection")
    parser.add_argument("--sequence", default="Axel_10")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--split", default="train_set")
    parser.add_argument("--save", help="Save output image")
    args = parser.parse_args()

    npy_path = find_sequence(args.split, args.sequence, args.camera)
    if npy_path is None:
        print(f"Error: sequence not found")
        return 1

    kp3d = np.load(npy_path)
    coco_kps = np.load(npy_path.parent / f"{npy_path.stem}_coco.npy")

    if args.frame >= len(kp3d):
        print(f"Error: frame {args.frame} out of range (0-{len(kp3d)-1})")
        return 1

    # Get camera from JSON metadata
    json_path = npy_path.with_suffix(".json")
    with open(json_path) as f:
        meta = json.load(f)

    with open(DATA_ROOT / "cam_param.json") as f:
        cam_params = json.load(f)

    cam_key = meta["cam"]
    cam = cam_params[cam_key]

    # Project foot keypoints
    foot_2d = project_foot_frame(kp3d[args.frame], cam)
    coco_2d = coco_kps[args.frame]

    # Merge
    pts, vis = merge_coco_foot_keypoints(coco_2d, foot_2d)

    # Draw
    mp4_path = npy_path.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(mp4_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read frame")
        return 1

    overlay = draw_halpe26(frame, pts, vis)

    # Stats
    valid = (vis > 0.1).sum()
    foot_valid = (vis[17:23] > 0.1).sum()
    print(f"Sequence: {args.sequence} | Camera: {cam_key} | Frame: {args.frame}/{len(kp3d)-1}")
    print(f"Valid keypoints: {valid}/26 (foot: {foot_valid}/6)")
    print(f"COCO 17kp from _coco.npy, foot 6kp projected from 3D mocap")

    if args.save:
        cv2.imwrite(args.save, overlay)
        print(f"Saved to: {args.save}")
    else:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{args.sequence} cam_{args.camera} frame {args.frame}")
            plt.axis("off")
            plt.show()
        except ImportError:
            cv2.imwrite("/tmp/validation.jpg", overlay)
            print("Saved to /tmp/validation.jpg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run validation**

Run: `uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50 --save /tmp/halpe26_check.jpg`

Expected: Image with green (COCO 17) + orange (foot 6) + magenta (face 3) keypoints.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_projection.py
git commit -m "feat(datasets): add HALPE26 projection validation visualizer"
```

---

### Task 4: Batch Dataset Converter

**Files:**
- Create: `scripts/prepare_athletepose3d.py`

- [ ] **Step 1: Write batch conversion script**

```python
#!/usr/bin/env python3
"""Convert AthletePose3D to HALPE26 COCO JSON for RTMPose fine-tuning.

Merges pre-computed _coco.npy (17kp) with projected foot keypoints (6kp)
and face duplicates (3kp) to produce HALPE26 26kp annotations.

Usage:
    uv run python scripts/prepare_athletepose3d.py
    uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3
    uv run python scripts/prepare_athletepose3d.py --max-sequences 10 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.projector import project_foot_frame
from src.datasets.coco_builder import (
    merge_coco_foot_keypoints,
    format_keypoints,
    build_coco_json,
    save_coco_json,
)

DATA_ROOT = Path("data/datasets/athletepose3d")
OUTPUT_DIR = DATA_ROOT / "coco_annotations"


def discover_sequences(split: str) -> list[tuple[Path, Path, Path, Path]]:
    """Find all (npy, coco_npy, json, mp4) tuples in a split."""
    base = DATA_ROOT / "videos" / split
    if not base.exists():
        return []
    sequences = []
    for npy_path in sorted(base.glob("**/*.npy")):
        if "_coco" in npy_path.stem or "_h36m" in npy_path.stem:
            continue
        coco_path = npy_path.parent / f"{npy_path.stem}_coco.npy"
        json_path = npy_path.with_suffix(".json")
        mp4_path = npy_path.with_suffix(".mp4")
        if coco_path.exists() and json_path.exists() and mp4_path.exists():
            sequences.append((npy_path, coco_path, json_path, mp4_path))
    return sequences


def process_sequence(
    kp3d: NDArray,
    coco_kps: NDArray,
    cam: dict,
    sample_rate: int,
    image_width: int,
    image_height: int,
    video_stem: str,
    image_id_offset: int,
    ann_id_offset: int,
    scale_x: float,
    scale_y: float,
) -> tuple[list[dict], list[dict], int, int]:
    """Process a single video sequence.

    Returns:
        (images, annotations, next_image_id, next_ann_id)
    """
    n_frames = len(kp3d)
    images = []
    annotations = []
    img_id = image_id_offset
    ann_id = ann_id_offset

    for frame_idx in range(0, n_frames, sample_rate):
        coco_2d = coco_kps[frame_idx]
        foot_2d = project_foot_frame(kp3d[frame_idx], cam)
        pts, vis = merge_coco_foot_keypoints(coco_2d, foot_2d)

        # Count visible keypoints (COCO 17 only for bbox calculation)
        valid_coco = vis[:17] > 0.1
        if valid_coco.sum() < 5:
            continue

        # Scale to target image resolution
        pts_scaled = pts.copy()
        pts_scaled[:, 0] *= scale_x
        pts_scaled[:, 1] *= scale_y

        # Compute bbox from visible COCO keypoints
        valid_pts = pts_scaled[valid_coco]
        x_min = max(0, float(valid_pts[:, 0].min()) - 20)
        y_min = max(0, float(valid_pts[:, 1].min()) - 20)
        x_max = min(image_width, float(valid_pts[:, 0].max()) + 20)
        y_max = min(image_height, float(valid_pts[:, 1].max()) + 20)
        w, h = x_max - x_min, y_max - y_min

        if w <= 0 or h <= 0:
            continue

        n_visible = int((vis > 0.1).sum())
        if n_visible < 10:
            continue

        img_id += 1
        ann_id += 1

        images.append({
            "file_name": f"{video_stem}/frame_{frame_idx:06d}.jpg",
            "id": img_id,
            "width": image_width,
            "height": image_height,
        })

        annotations.append({
            "image_id": img_id,
            "id": ann_id,
            "keypoints": format_keypoints(pts_scaled, vis),
            "num_keypoints": n_visible,
            "bbox": [x_min, y_min, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    return images, annotations, img_id, ann_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AthletePose3D to HALPE26 COCO JSON")
    parser.add_argument("--split", default="train_set", choices=["train_set", "valid_set", "test_set"])
    parser.add_argument("--sample-rate", type=int, default=3, help="Extract every Nth frame")
    parser.add_argument("--max-sequences", type=int, default=0, help="Max sequences (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-width", type=int, default=1920, help="Target image width")
    parser.add_argument("--target-height", type=int, default=1088, help="Target image height")
    args = parser.parse_args()

    with open(DATA_ROOT / "cam_param.json") as f:
        cam_params = json.load(f)

    sequences = discover_sequences(args.split)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    print(f"Found {len(sequences)} sequences in {args.split}")
    print(f"Sample rate: {args.sample_rate}, target: {args.target_width}x{args.target_height}")
    print()

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    img_id = 0
    ann_id = 0

    for npy_path, coco_path, json_path, mp4_path in tqdm(sequences, desc=f"Processing {args.split}"):
        kp3d = np.load(npy_path)
        coco_kps = np.load(coco_path)

        if kp3d.shape[0] != coco_kps.shape[0]:
            tqdm.write(f"  SKIP {npy_path.stem}: frame count mismatch")
            continue

        # Get camera from JSON metadata
        with open(json_path) as f:
            meta = json.load(f)
        cam_key = meta["cam"]
        if cam_key not in cam_params:
            tqdm.write(f"  SKIP {npy_path.stem}: camera {cam_key} not found")
            continue

        cam = cam_params[cam_key]
        scale_x = args.target_width / meta.get("video_width", 1920)
        scale_y = args.target_height / meta.get("video_height", 1088)

        images, annotations, img_id, ann_id = process_sequence(
            kp3d, coco_kps, cam, args.sample_rate,
            args.target_width, args.target_height,
            npy_path.stem, img_id, ann_id,
            scale_x, scale_y,
        )

        all_images.extend(images)
        all_annotations.extend(annotations)

    print(f"\nTotal images: {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0

    output_path = OUTPUT_DIR / f"athletepose3d_{args.split}.json"
    coco_data = build_coco_json(all_images, all_annotations)
    save_coco_json(coco_data, str(output_path))
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry run (5 sequences)**

Run: `uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3 --max-sequences 5 --dry-run`

Expected: Count of images/annotations without writing.

- [ ] **Step 3: Small test run (5 sequences)**

Run: `uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3 --max-sequences 5`

Expected: COCO JSON saved to `data/datasets/athletepose3d/coco_annotations/`.

- [ ] **Step 4: Visual spot-check**

Run: `uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50 --save /tmp/halpe26_final.jpg`

- [ ] **Step 5: Full conversion (train + val)**

Run: `uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3`
Run: `uv run python scripts/prepare_athletepose3d.py --split valid_set --sample-rate 3`

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_athletepose3d.py
git commit -m "feat(datasets): add batch AthletePose3D→HALPE26 COCO JSON converter"
```

---

### Task 5: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/test_projection.py tests/test_coco_builder.py -v`

Expected: ALL PASS

- [ ] **Step 2: Fix any issues**

If tests fail, investigate and fix.

- [ ] **Step 3: Commit if any fixes needed**

---

## Deliverables

| File | Description |
|------|-------------|
| `src/datasets/__init__.py` | Package init |
| `src/datasets/projector.py` | 3D→2D foot keypoint projection |
| `src/datasets/coco_builder.py` | COCO JSON builder (HALPE26 26kp merge) |
| `tests/test_projection.py` | Projection tests |
| `tests/test_coco_builder.py` | COCO JSON format + merge tests |
| `scripts/validate_projection.py` | Visual validation tool |
| `scripts/prepare_athletepose3d.py` | Batch converter entry point |
| `data/datasets/athletepose3d/coco_annotations/` | Output COCO JSON files |

## Next Steps (After This Plan)

1. Extract frames from videos at training resolution (or use pose_2d images)
2. Set up mmpose training environment on the dedicated GPU machine
3. Write RTMPose fine-tune config for HALPE26 26kp
4. Fine-tune RTMPose on athletepose3d COCO JSON
5. Export fine-tuned model to ONNX for rtmlib inference
