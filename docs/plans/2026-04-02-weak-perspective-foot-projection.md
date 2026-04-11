# Weak-Perspective Foot Projection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-perspective foot keypoint projection with localized weak-perspective projection anchored at the ankle. Drop small toe keypoints (invisible in skate boots) and only project 4 keypoints (heel + big toe per foot).

**Architecture:** `project_foot_frame()` projects the ankle with full perspective (matches `_coco.npy`), then uses the ankle's camera-space depth `Z_ankle` as the reference depth for foot points on that side. Small toe slots (indices 2, 5) are always NaN → v=0 in COCO JSON (no gradient from KLDiscretLoss).

**Tech Stack:** Python, numpy. No new dependencies.

---

## Background

Full perspective projection divides each point's X/Y by its individual depth Z. When the foot points toward the camera, the heel's Z is much larger than the ankle's Z, causing the heel to project above the ankle in 2D (mathematically correct, visually wrong). 34-36% of heel projections fail validation.

**Solution — two changes:**

1. **Weak-perspective** uses a single reference depth `Z_ankle` for all foot points:
   ```
   foot_2d = ankle_2d + (f * delta_cam_xy / Z_ankle)
   ```
   where `delta_cam_xy = rot_mat @ (foot_3d - ankle_3d)` in camera space.

2. **Drop small toes** (HALPE26 indices 19, 22). In skate boots, the 5th metatarsal head is invisible. Only project heel + big toe = 4 keypoints. Small toe slots return NaN → merge_coco_foot_keypoints sets v=0 → KLDiscretLoss zeroes gradient.

**Ankle AP3D indices** (verified 0px distance to `_coco.npy`):
- L_ankle = 24, R_ankle = 59

---

## Keypoint Mapping (Updated)

| HALPE26 Idx | Name | Projection | Visibility |
|:---:|---|---|:---:|
| 0-16 | COCO 17kp | `_coco.npy` directly | From _coco (2.0 or 0.0) |
| 17 | left_heel | AP3D idx 49, weak-perspective | 2.0 (validated) or 0.0 |
| 18 | left_big_toe | AP3D idx 26, weak-perspective | 2.0 (validated) or 0.0 |
| 19 | left_small_toe | **Always NaN** | **Always 0.0** |
| 20 | right_heel | AP3D idx 112, weak-perspective | 2.0 (validated) or 0.0 |
| 21 | right_big_toe | AP3D idx 93, weak-perspective | 2.0 (validated) or 0.0 |
| 22 | right_small_toe | **Always NaN** | **Always 0.0** |
| 23-25 | face dupes | Copy from 0, 1, 2 | 0.3 or 0.0 |

---

## File Structure

Only ONE file is modified:
- `src/datasets/projector.py` — rewrite `project_foot_frame()` internals

Tests updated:
- `tests/test_projection.py` — add weak-perspective-specific tests, update existing validation tests

Output regenerated (no code changes to scripts):
- `data/datasets/athletepose3d/coco_annotations/*.json` — re-run `prepare_athletepose3d.py`
- `/tmp/halpe26_labeled_*.jpg` — re-run `batch_validate_labels.py`

---

### Task 1: Write Failing Tests for Weak-Perspective Behavior

**Files:**
- Modify: `tests/test_projection.py`

- [ ] **Step 1: Add `TestWeakPerspectiveProjection` class after `TestFootProjection`**

```python
class TestWeakPerspectiveProjection:
    """Tests for localized weak-perspective foot projection."""

    def test_ankle_ap3d_indices_defined(self):
        """ANKLE_AP3D_INDICES constant should exist with L_ankle=24, R_ankle=59."""
        from src.datasets.projector import ANKLE_AP3D_INDICES

        assert list(ANKLE_AP3D_INDICES) == [24, 59]

    def test_small_toes_always_nan(self, cam_params, sample_3d):
        """Small toe slots (indices 2, 5) should always be NaN."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        assert np.isnan(foot_2d[2]).all(), "L_small_toe should always be NaN"
        assert np.isnan(foot_2d[5]).all(), "R_small_toe should always be NaN"

    def test_weak_perspective_preserves_foot_geometry(self, cam_params, sample_3d):
        """Foot points should maintain correct relative positions (heel below ankle)."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        coco_kps = np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1_coco.npy")
        l_ankle_y = coco_kps[50, 15, 1]  # left_ankle Y
        r_ankle_y = coco_kps[50, 16, 1]  # right_ankle Y

        # Left heel (index 0) should not be above left ankle
        if not np.isnan(foot_2d[0, 1]):
            assert foot_2d[0, 1] >= l_ankle_y - 5, (
                f"L_heel y={foot_2d[0, 1]:.0f} above L_ankle y={l_ankle_y:.0f}"
            )

        # Right heel (index 3) should not be above right ankle
        if not np.isnan(foot_2d[3, 1]):
            assert foot_2d[3, 1] >= r_ankle_y - 5, (
                f"R_heel y={foot_2d[3, 1]:.0f} above R_ankle y={r_ankle_y:.0f}"
            )

    def test_weak_perspective_foot_near_ankle(self, cam_params, sample_3d, sample_coco):
        """Weak-perspective foot points should be within reasonable distance of ankle."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        l_ankle = sample_coco[50, 15]  # L_ankle from _coco.npy

        # Check only projected points (indices 0, 1 — skip index 2 which is always NaN)
        for i in [0, 1]:
            if not np.isnan(foot_2d[i, 0]):
                dist = np.linalg.norm(foot_2d[i] - l_ankle)
                assert dist < 100, f"Left foot index {i}: {dist:.0f}px from ankle"

    def test_nan_ankle_invalidates_foot_side(self, cam_params):
        """NaN ankle 3D position should invalidate all foot points on that side."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        kp3d = np.zeros((142, 3), dtype=np.float64)

        # Set valid positions for everything EXCEPT left ankle (index 24)
        kp3d[:] = [1000.0, 500.0, 2000.0]
        kp3d[24] = [np.nan, np.nan, np.nan]  # L_ankle is NaN

        # Set foot markers to valid positions (LHEL=49, L_Toe=26)
        kp3d[49] = [1000.0, 300.0, 2000.0]  # LHEL below ankle
        kp3d[26] = [1100.0, 300.0, 2000.0]  # L_Toe

        # Right ankle and foot are valid
        kp3d[59] = [2000.0, 500.0, 2000.0]  # R_ankle
        kp3d[112] = [2000.0, 300.0, 2000.0]  # RHEL
        kp3d[93] = [2100.0, 300.0, 2000.0]   # R_Toe

        foot_2d = project_foot_frame(kp3d, cam)

        # Left foot (indices 0-2) should all be NaN (including small toe)
        for i in range(3):
            assert np.isnan(foot_2d[i]).all(), f"Left foot index {i} should be NaN"

        # Right heel and big toe (indices 3, 4) should be valid
        assert not np.isnan(foot_2d[3]).any(), "R_heel should be valid"
        assert not np.isnan(foot_2d[4]).any(), "R_bigtoe should be valid"

        # Right small toe (index 5) is always NaN
        assert np.isnan(foot_2d[5]).all(), "R_small_toe should always be NaN"
```

- [ ] **Step 2: Run tests to verify new ones fail**

Run: `uv run pytest tests/test_projection.py::TestWeakPerspectiveProjection -v`
Expected: FAIL — `ANKLE_AP3D_INDICES` doesn't exist yet; weak-perspective behavior not implemented.

---

### Task 2: Implement Weak-Perspective Projection

**Files:**
- Modify: `src/datasets/projector.py`

- [ ] **Step 1: Add `ANKLE_AP3D_INDICES` constant**

After the existing `FOOT_AP3D_INDICES` constant, add:

```python
# AthletePose3D 142kp indices for ankle keypoints (used as weak-perspective anchors)
ANKLE_AP3D_INDICES = np.array([24, 59], dtype=np.intp)
# L_ankle, R_ankle — verified 0px distance to _coco.npy
```

- [ ] **Step 2: Rewrite `project_foot_frame` to use weak-perspective + skip small toes**

Replace the entire `project_foot_frame` function body with:

```python
def project_foot_frame(
    keypoints_3d: NDArray[np.float64],
    cam: dict,
) -> NDArray[np.float32]:
    """Project foot keypoints from 3D to 2D using localized weak-perspective.

    Instead of full perspective (individual Z per point), uses the ankle's
    camera-space depth Z_ankle as reference depth for foot points on
    that side. This preserves parallel foot geometry and eliminates the
    "heel above ankle" paradox caused by perspective depth distortion.

    Only projects heel + big toe per foot (4 keypoints total). Small toe
    slots (indices 2, 5) are always NaN because the 5th metatarsal is
    invisible inside a skate boot.

    Algorithm per foot side:
        1. Project ankle with full perspective → (ankle_u, ankle_v, Z_ankle)
        2. For each foot point (heel, big toe):
           delta_cam = rot_mat @ (foot_3d - ankle_3d)
           offset_2d = (fu * delta_cam[0] / Z_ankle, fv * delta_cam[1] / Z_ankle)
           foot_2d = (ankle_u + offset_u, ankle_v + offset_v)

    Args:
        keypoints_3d: (N, 3) array of 3D world coordinates (N >= 113 for all foot kps).
        cam: Camera dict from cam_param.json.

    Returns:
        (6, 2) projected 2D coordinates for HALPE26 foot keypoints 17-22.
        Indices 2 (L_small_toe) and 5 (R_small_toe) are always NaN.
    """
    K, rot_mat, t, fu, fv, cu, cv = _load_camera(cam)
    pts = np.full((6, 2), np.nan, dtype=np.float32)

    # Small toe slots are always NaN (invisible in skate boots)
    # FOOT_AP3D_INDICES layout: [LHEL=49, L_Toe=26, L_5thMTP=10, RHEL=112, R_Toe=93, R_5thMTP=77]
    # Only project indices 0(LHEL), 1(L_Toe), 3(RHEL), 4(R_Toe)
    projected_pairs = [(0, 0, 24), (1, 1, 24), (3, 3, 59), (4, 4, 59)]
    # (foot_2d_idx, FOOT_AP3D_idx, ankle_ap3d_idx)

    for foot_idx, ap3d_idx, ankle_idx in projected_pairs:
        ankle_3d = keypoints_3d[ankle_idx]

        if ankle_idx >= len(keypoints_3d) or np.isnan(ankle_3d).any():
            continue

        ankle_translated = ankle_3d - t
        ankle_cam = rot_mat @ ankle_translated

        if ankle_cam[2] <= 0:
            continue

        ankle_u = fu * (ankle_cam[0] / ankle_cam[2]) + cu
        ankle_v = fv * (ankle_cam[1] / ankle_cam[2]) + cv
        z_ankle = ankle_cam[2]

        if ap3d_idx >= len(keypoints_3d):
            continue

        foot_3d = keypoints_3d[ap3d_idx]
        if np.isnan(foot_3d).any():
            continue

        # Camera-space offset from ankle
        foot_translated = foot_3d - t
        foot_cam = rot_mat @ foot_translated
        delta_cam = foot_cam - ankle_cam

        # Weak perspective: use Z_ankle for all foot points
        pts[foot_idx] = [
            ankle_u + fu * delta_cam[0] / z_ankle,
            ankle_v + fv * delta_cam[1] / z_ankle,
        ]

    return pts
```

- [ ] **Step 3: Run new tests to verify they pass**

Run: `uv run pytest tests/test_projection.py::TestWeakPerspectiveProjection -v`
Expected: ALL PASS

- [ ] **Step 4: Run full projection test suite to verify no regressions**

Run: `uv run pytest tests/test_projection.py::TestFootProjection -v`
Expected: ALL PASS — existing tests use `project_point()` (unchanged) or check shape/NaN (still valid).

- [ ] **Step 5: Run validation tests**

Run: `uv run pytest tests/test_projection.py::TestValidateFootProjection -v`
Expected: ALL PASS — `validate_foot_projection` skips NaN points (small toes are NaN → skipped).

---

### Task 3: Verify Statistics Improvement

**Files:** None (inline script)

- [ ] **Step 1: Run projection statistics on sample**

```bash
uv run python -c "
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from src.datasets.projector import project_foot_frame, validate_foot_projection

DATA_ROOT = Path('data/datasets/athletepose3d')
with open(DATA_ROOT / 'cam_param.json') as f:
    cam_params = json.load(f)

seqs = sorted((DATA_ROOT / 'videos/train_set').glob('**/*.npy'))[:100]
bad_counts = {i: 0 for i in range(6)}
total = 0
for npy_path in seqs:
    if '_coco' in npy_path.stem:
        continue
    coco_path = npy_path.parent / f'{npy_path.stem}_coco.npy'
    json_path = npy_path.with_suffix('.json')
    if not coco_path.exists() or not json_path.exists():
        continue
    kp3d = np.load(npy_path)
    coco_kps = np.load(coco_path)
    with open(json_path) as f:
        meta = json.load(f)
    cam_key = meta.get('cam')
    if cam_key not in cam_params:
        continue
    cam = cam_params[cam_key]
    for fi in range(0, len(kp3d), 10):
        foot_2d = project_foot_frame(kp3d[fi], cam)
        coco_2d = coco_kps[fi]
        validate_foot_projection(foot_2d, coco_2d)
        for i in range(6):
            if np.isnan(foot_2d[i, 0]):
                bad_counts[i] += 1
        total += 1

names = ['L_heel', 'L_bigtoe', 'L_smtoe(always)', 'R_heel', 'R_bigtoe', 'R_smtoe(always)']
print(f'Sampled {total} frames from {len(seqs)} sequences')
print('Bad rates (weak-perspective + validation):')
for i, name in enumerate(names):
    rate = bad_counts[i] / total * 100 if total > 0 else 0
    print(f'  {name}: {rate:.1f}% ({bad_counts[i]}/{total})')
"
```

Expected: Small toes 100% (always NaN). Heel/big_toe bad rates much lower than before (was 34.6% L_heel, 36.2% R_heel).

---

### Task 4: Regenerate COCO JSON

**Files:** None (re-run existing script)

- [ ] **Step 1: Re-run batch converter for train set**

```bash
uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3
```

Expected: COCO JSON saved. Small toe keypoints (indices 19, 22) should have v=0 in all annotations.

- [ ] **Step 2: Re-run batch converter for valid set**

```bash
uv run python scripts/prepare_athletepose3d.py --split valid_set --sample-rate 3
```

---

### Task 5: Regenerate Validation Images

**Files:** None (re-run existing script)

- [ ] **Step 1: Clean old validation images**

```bash
rm -f /tmp/halpe26_labeled_*.jpg
```

- [ ] **Step 2: Re-generate labeled images**

```bash
uv run python scripts/batch_validate_labels.py
```

Expected: No small toe markers visible. Heels and big toes should be correctly positioned below/at ankle level.

---

### Task 6: Commit

- [ ] **Step 1: Review changes**

```bash
git diff src/datasets/projector.py tests/test_projection.py
```

- [ ] **Step 2: Commit**

```bash
git add src/datasets/projector.py tests/test_projection.py
git commit -m "feat(datasets): weak-perspective foot projection, drop small toes

Replace full-perspective projection with localized weak-perspective
anchored at ankle depth Z_ankle. Drop small toe keypoints (indices 19, 22)
as they're invisible in skate boots. Only project heel + big toe = 4
foot keypoints. Small toe slots always NaN -> v=0 in COCO JSON ->
no gradient from KLDiscretLoss during training."
```

---

## Self-Review

**1. Spec coverage:**
- Weak-perspective projection: Task 2
- ANKLE_AP3D_INDICES constant: Task 1 + Task 2
- Small toes always NaN: Task 1 test + Task 2 implementation
- NaN ankle handling: Task 1 test + Task 2 implementation
- Backward compatibility (function signature unchanged): Task 2 preserves `(keypoints_3d, cam) -> (6, 2)` interface
- COCO JSON regeneration: Task 4
- Validation images: Task 5

**2. Placeholder scan:** No TBDs, no vague instructions. All code is complete.

**3. Type consistency:** `project_foot_frame` signature unchanged. Return type `(6, 2) NDArray[np.float32]` unchanged. `validate_foot_projection` interface unchanged. `merge_coco_foot_keypoints` handles NaN→v=0 automatically. All calling code needs zero modifications.

**4. Impact analysis:**
- `project_point()`: untouched
- `validate_foot_projection()`: untouched (skips NaN points)
- `merge_coco_foot_keypoints()`: untouched (NaN → v=0)
- `prepare_athletepose3d.py`: no changes needed
- `batch_validate_labels.py`: no changes needed (small toes are NaN, not drawn)
- Existing tests: unaffected (shape checks, NaN checks, validation logic all still valid)
