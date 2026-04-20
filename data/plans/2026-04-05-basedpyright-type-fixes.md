# Fix 166 basedpyright Type Errors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 166 basedpyright `--level error` type errors so CI passes cleanly.

**Architecture:** Group fixes by error category and file. Each task targets a specific error pattern with a consistent fix strategy. Tasks are ordered from highest-impact/lowest-risk to most complex. No behavior changes — pure type annotation fixes.

**Tech Stack:** basedpyright, typing, numpy.typing

---

## Error Categories & Strategy

| # | Category | Errors | Strategy |
|---|----------|--------|----------|
| 1 | Exclude 3rd-party model code | ~12 | Exclude `src/models/` from basedpyright |
| 2 | Missing type stubs (dtw, ultralytics, timm) | ~10 | Add `# type: ignore[import-untyped]` per import |
| 3 | Missing relative imports (.video, .normalizer, .types) | ~8 | Add `# type: ignore[import-untyped]` per import |
| 4 | `astype` on float | 11 | Cast with `int()` / `float()` instead of `.astype()` |
| 5 | Possibly unbound variables | 10 | Initialize before branches |
| 6 | Optional member access (None checks) | ~20 | Add `assert` / `if ... is None: return` guards |
| 7 | numpy scalar→python type (floating→float, signedinteger→int) | ~15 | Add explicit `float()` / `int()` casts |
| 8 | NDArray dtype mismatch (float64 vs float32) | 5 | Add `.astype(np.float32)` |
| 9 | OpenCV type aliases (MatLike vs Frame) | ~8 | Use `NDArray` type alias or `cast()` |
| 10 | Argument type mismatches (misc) | ~25 | Per-file fixes with explicit casts |
| 11 | Video writer (PyAV) types | ~8 | Add `# type: ignore` for PyAV dynamic attrs |
| 12 | Reference builder import resolution | ~8 | Fix relative imports or add ignores |

---

### Task 1: Exclude third-party model code from basedpyright

**Files:**
- Modify: `pyproject.toml` (basedpyright exclude)

The `src/models/` directory contains MotionAGFormer and TCPFormer — third-party research code with many type issues (DropPath, torchprofile, __getitem__). Exclude it entirely.

- [ ] **Step 1: Update basedpyright exclude in pyproject.toml**

In `[tool.basedpyright]`, update the `exclude` list:

```toml
exclude = [
    "Sports2D/",
    "TCPFormer/",
    "src/models/",
]
```

- [ ] **Step 2: Verify error count drops**

Run: `uv run basedpyright --level error src/ 2>&1 | tail -1`
Expected: significantly fewer errors (should drop from 166 to ~140)

- [ ] **Step 3: Commit**

```bash
git checkout -b feature/basedpyright-fixes
git add pyproject.toml
git commit -m "$(cat <<'EOF'
fix(types): exclude third-party model code from basedpyright
EOF
)"
```

---

### Task 2: Fix dtw-python type stubs (aligner.py, motion_dtw.py)

**Files:**
- Modify: `src/alignment/aligner.py`
- Modify: `src/alignment/motion_dtw.py`

The `dtw-python` library has no type stubs. Its `DTW` class exposes `index1`, `index2`, `distance` dynamically. Fix with `# type: ignore` and explicit casts.

**Errors:** 16 total (attribute access + wrong arg types for `dtw()`)

- [ ] **Step 1: Fix aligner.py**

At the top of the file, find the dtw import and add:
```python
from dtw import *  # type: ignore[import-untyped]
```
Or if it's `from dtw import dtw`, change to:
```python
from dtw import dtw  # type: ignore[import-untyped]
```

For the DTW result access lines (~line 61, 64, 95), the `index1`/`index2`/`distance` attributes are dynamic. Add `# type: ignore[attr-defined]` to those lines, or cast the result:

At lines where `alignment.index1`, `alignment.index2`, `alignment.distance` are used, add inline ignores:
```python
path1 = alignment.index1  # type: ignore[attr-defined]
path2 = alignment.index2  # type: ignore[attr-defined]
```

For the `dtw()` call at line 185 — the library accepts string kwargs like `keep_internals="TRUE"` which basedpyright sees as wrong type. Add `# type: ignore[call-arg]`:
```python
alignment = dtw(..., keep_internals=True, ...)  # change "TRUE" to True
```

Actually, the simplest fix: change the string args to actual booleans. The dtw-python library accepts both. Change `"TRUE"` to `True`, `"FALSE"` to `False` everywhere.

- [ ] **Step 2: Fix motion_dtw.py**

Same pattern. Fix import, attribute access, and string→bool for dtw() call.

- [ ] **Step 3: Verify**

Run: `uv run basedpyright --level error src/alignment/ 2>&1 | tail -1`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
git add src/alignment/
git commit -m "$(cat <<'EOF'
fix(types): fix dtw-python type errors in alignment module
EOF
)"
```

---

### Task 3: Fix missing import stubs across codebase

**Files:**
- Modify: `src/detection/person_detector.py`
- Modify: `src/pose_estimation/h36m_extractor.py`
- Modify: `src/pose_estimation/yolo_extractor.py`
- Modify: `src/types.py`
- Modify: `src/references/reference_store.py`

Libraries without type stubs: `ultralytics` (YOLO export), and internal relative imports that basedpyright can't resolve.

**Errors:** ~8

- [ ] **Step 1: Fix ultralytics imports**

In `person_detector.py`, `h36m_extractor.py`, `yolo_extractor.py` — find `from ultralytics import YOLO` and add:
```python
from ultralytics import YOLO  # type: ignore[import-untyped]
```

- [ ] **Step 2: Fix relative import resolution errors**

These are files that use relative imports like `from .video import VideoMeta` where basedpyright can't resolve them (works at runtime due to PYTHONPATH). Add `# type: ignore[import-untyped]` to each:

- `src/types.py:18` — `from .video import VideoMeta`
- `src/references/reference_store.py:12` — `from .types import ...`

For `src/references/reference_builder.py` — the relative imports `.normalizer` and `.pose_estimation` are cross-module (they go outside `src/references/`). These need `# type: ignore[import-untyped]`:
```python
from .normalizer import ...  # type: ignore[import-untyped]
from .pose_estimation import ...  # type: ignore[import-untyped]
```

Wait — `reference_builder.py` imports from `..normalizer` and `..pose_estimation` (parent package). Check the actual import paths and add appropriate ignores.

- [ ] **Step 3: Verify**

Run: `uv run basedpyright --level error src/detection/person_detector.py src/pose_estimation/h36m_extractor.py src/pose_estimation/yolo_extractor.py src/types.py src/references/ 2>&1 | tail -1`
Expected: fewer errors in these files

- [ ] **Step 4: Commit**

```bash
git add src/detection/person_detector.py src/pose_estimation/ src/types.py src/references/
git commit -m "$(cat <<'EOF'
fix(types): add type ignore for untyped third-party imports
EOF
)"
```

---

### Task 4: Fix `.astype` on float values (drawer.py, velocity_layer.py)

**Files:**
- Modify: `src/visualization/skeleton/drawer.py` (lines 100, 101, 118, 240, 241, 258, 354, 355, 365)
- Modify: `src/visualization/layers/velocity_layer.py` (lines 184, 246)

**Pattern:** Code does `int(value.astype(int))` where `value` could be a python `float`, not `np.ndarray`. Fix: use `int(round(value))` or `int(value)` directly.

**Errors:** 11

- [ ] **Step 1: Fix drawer.py**

At each error line, the pattern is `int(some_coord.astype(int))`. The `some_coord` is already extracted as a scalar from an array. Replace with `int(round(float(some_coord)))` or just `int(some_coord)` depending on context.

Read the actual code at each line, then replace:
- `int(x.astype(int))` → `int(x)` (astyped values from numpy indexing are already numeric)
- If it's `int(value[0].astype(int))` → `int(value[0])`

- [ ] **Step 2: Fix velocity_layer.py**

Same pattern at lines 184, 246.

- [ ] **Step 3: Verify**

Run: `uv run basedpyright --level error src/visualization/skeleton/drawer.py src/visualization/layers/velocity_layer.py 2>&1 | tail -1`
Expected: 0 errors in these files

- [ ] **Step 4: Commit**

```bash
git add src/visualization/skeleton/drawer.py src/visualization/layers/velocity_layer.py
git commit -m "$(cat <<'EOF'
fix(types): replace .astype(int) on scalars with int() cast
EOF
)"
```

---

### Task 5: Fix possibly unbound variables

**Files:**
- Modify: `src/visualization/comparison.py` (out_buf, pad_a, pad_r, divider)
- Modify: `src/cli.py` (output)
- Modify: `src/pose_estimation/rtmlib_extractor.py` (pbar)

**Pattern:** Variables assigned inside `if/elif` branches but used after without guaranteed assignment.

**Errors:** 10

- [ ] **Step 1: Fix comparison.py**

Initialize variables before the conditional block:
```python
out_buf = None
pad_a = None
pad_r = None
divider = None
```
Then add `assert out_buf is not None` before use, or restructure.

- [ ] **Step 2: Fix cli.py line 179**

Add `output = None` before the conditional that assigns it.

- [ ] **Step 3: Fix rtmlib_extractor.py line 430**

Add `pbar = None` before the conditional assignment.

- [ ] **Step 4: Verify**

Run: `uv run basedpyright --level error src/visualization/comparison.py src/cli.py src/pose_estimation/rtmlib_extractor.py 2>&1 | tail -1`
Expected: these specific errors resolved

- [ ] **Step 5: Commit**

```bash
git add src/visualization/comparison.py src/cli.py src/pose_estimation/rtmlib_extractor.py
git commit -m "$(cat <<'EOF'
fix(types): initialize variables before conditional branches
EOF
)"
```

---

### Task 6: Fix numpy scalar → python type mismatches

**Files:**
- Modify: `src/analysis/phase_detector.py` (signedinteger→int, floating→float)
- Modify: `src/detection/blade_edge_detector_3d.py` (floating→float)
- Modify: `src/detection/pose_tracker.py` (floating→float, ndarray|None)
- Modify: `src/detection/spatial_reference.py` (floating→float, astype on None)
- Modify: `src/tracking/tracklet_merger.py` (floating→float, optional attrs)
- Modify: `src/analysis/element_defs.py` (float→int for rotations)

**Pattern:** `np.argmax()` returns `signedinteger`, `np.mean()` returns `floating` — these don't match `int`/`float` parameter types. Fix: wrap with `int()` / `float()`.

**Errors:** ~25

- [ ] **Step 1: Fix phase_detector.py**

Where numpy integers are passed to `ElementPhase(start=..., takeoff=..., peak=..., landing=..., end=...)`, wrap each with `int()`:
```python
start=int(np.argmax(...)),
```
For `confidence` parameter: `confidence=float(...)`.

- [ ] **Step 2: Fix blade_edge_detector_3d.py line 146**

`velocity_mag` parameter: wrap with `float(velocity_mag)`.

- [ ] **Step 3: Fix pose_tracker.py**

- Lines 218, 242: `P` is `ndarray | None` — add `assert P is not None` before the call
- Line 226: `predictions` is `list[Unknown]` — convert to ndarray with `np.array(predictions)`
- Line 399: return type `dict[str, floating]` → `dict[str, float]` — wrap values with `float()`

- [ ] **Step 4: Fix spatial_reference.py**

- Line 214: return `float()` wrapper
- Line 509: add None guard before `.astype()`

- [ ] **Step 5: Fix tracklet_merger.py**

- Lines 166, 168: add None guards for optional track attributes
- Line 203: return `float()` wrapper

- [ ] **Step 6: Fix element_defs.py line 199**

`rotations` expects `int` but gets `float` — add `int()` cast.

- [ ] **Step 7: Verify**

Run: `uv run basedpyright --level error src/analysis/phase_detector.py src/analysis/element_defs.py src/detection/ src/tracking/ 2>&1 | tail -1`
Expected: errors resolved in these files

- [ ] **Step 8: Commit**

```bash
git add src/analysis/phase_detector.py src/analysis/element_defs.py src/detection/ src/tracking/
git commit -m "$(cat <<'EOF'
fix(types): wrap numpy scalars with int()/float() for type safety
EOF
)"
```

---

### Task 7: Fix NDArray dtype mismatches and optional access in pose_3d/

**Files:**
- Modify: `src/pose_3d/kinematic_constraints.py` (float64→float32)
- Modify: `src/pose_3d/athletepose_extractor.py` (None checks, missing imports)
- Modify: `src/pose_3d/tcpformer_extractor.py` (None checks, missing imports)
- Modify: `src/pose_3d/onnx_extractor.py` (SparseTensor indexing)

**Errors:** ~18

- [ ] **Step 1: Fix kinematic_constraints.py line 112**

Add `.astype(np.float32)` to the return value or assignment.

- [ ] **Step 2: Fix athletepose_extractor.py**

- Line 91: `model_path` is `Path | None` — add guard `if model_path is None: raise ValueError(...)`
- Line 105: same guard covers this
- Lines 112, 122: `import tcpformer` / `import motionagformer` — add `# type: ignore[import-untyped]`
- Lines 177-185: model is `Module | None` — add `assert model is not None` after load
- Line 232: `estimate_3d` on None — same guard covers this
- Line 185: return type — add `assert model is not None; return model`

- [ ] **Step 3: Fix tcpformer_extractor.py**

Same pattern as athletepose_extractor.py:
- Line 73: import ignore
- Lines 111-116: None guards for model
- Line 132: None guard for estimate_3d

- [ ] **Step 4: Fix onnx_extractor.py line 115**

SparseTensor indexing — add `# type: ignore[index]`.

- [ ] **Step 5: Verify**

Run: `uv run basedpyright --level error src/pose_3d/ 2>&1 | tail -1`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/pose_3d/
git commit -m "$(cat <<'EOF'
fix(types): fix None guards and dtype casts in pose_3d module
EOF
)"
```

---

### Task 8: Fix OpenCV type issues in visualization/

**Files:**
- Modify: `src/visualization/comparison.py` (MatLike vs Frame)
- Modify: `src/visualization/core/geometry.py` (Position2D type mismatches)
- Modify: `src/visualization/core/text.py` (FreeTypeFont, tuple sizes)
- Modify: `src/visualization/layers/joint_angle_layer.py` (ndarray arg types)
- Modify: `src/visualization/layers/trail_layer.py` (Point type)
- Modify: `src/visualization/layers/velocity_layer.py` (remaining fixes)
- Modify: `src/gradio_helpers.py` (Path vs str, MatLike vs Frame)

**Errors:** ~25

- [ ] **Step 1: Add a shared type alias for frames**

In `src/visualization/core/overlay.py` or a types file, ensure there's a type alias:
```python
from typing import Any
import numpy as np
from numpy.typing import NDArray

# OpenCV uses both Frame and MatLike interchangeably
FrameLike = NDArray[Any]  # type: ignore[misc]
```

Or simpler: add `# type: ignore[arg-type]` at each call site where MatLike/Frame mismatch occurs.

- [ ] **Step 2: Fix comparison.py**

For draw_skeleton/render_layers/draw_text_box calls with MatLike frames:
Add `# type: ignore[arg-type]` at lines 373, 382, 410.

- [ ] **Step 3: Fix gradio_helpers.py**

- Lines 147-148: Change type annotations from `str` to `Path`
- Line 150: Wrap with `Path(path)`
- Lines 258, 271, 279: Add `# type: ignore[arg-type]` for Frame mismatches
- Lines 304-305: These access `.parent`/`.stem` on a variable typed as `str` — the variable is actually a Path. Fix the type annotation.

- [ ] **Step 4: Fix visualization/core/ files**

- `geometry.py:234` — add `# type: ignore[arg-type]` for Position2D mismatch
- `geometry.py:236` — add explicit cast
- `text.py:116` — add `# type: ignore[assignment]` for FreeTypeFont
- `text.py:183` — cast return to `tuple[int, int]`
- `text.py:460` — cast size to `tuple[int, int]`

- [ ] **Step 5: Fix visualization/layers/ files**

- `joint_angle_layer.py:222` — add `# type: ignore[arg-type]` for ndarray arguments
- `trail_layer.py:164` — add `# type: ignore[arg-type]` for Point arguments
- `velocity_layer.py:183-264` — wrap with `float()`, add type ignores

- [ ] **Step 6: Verify**

Run: `uv run basedpyright --level error src/visualization/ src/gradio_helpers.py 2>&1 | tail -1`
Expected: 0 errors

- [ ] **Step 7: Commit**

```bash
git add src/visualization/ src/gradio_helpers.py
git commit -m "$(cat <<'EOF'
fix(types): fix OpenCV and visualization type errors
EOF
)"
```

---

### Task 9: Fix pipeline.py and reference_builder.py

**Files:**
- Modify: `src/pipeline.py`
- Modify: `src/references/reference_builder.py`

**Errors:** ~16

- [ ] **Step 1: Fix pipeline.py**

- Line 104: `extract_video_tracked` on None — add None guard for `_pose_2d_extractor`
- Line 145: `CameraPose` not on `SpatialReferenceDetector` — add `# type: ignore[attr-defined]` or fix the import
- Line 277: `PhysicsEngine` unknown import — fix the import path
- Line 314: `list[ElementPhase]` vs `ElementPhase` — check if it should be `phases=[phase]` or type should accept list
- Lines 317, 319, 320: None args — add `or 0.0` / `or {}` defaults
- Line 378: cannot assign `_pose_2d_extractor` — this is likely a class attribute that needs to be declared
- Line 508: `ElementPhase` not iterable — check if iterating over wrong variable

- [ ] **Step 2: Fix reference_builder.py**

- Lines 24-25: cross-module relative imports — add `# type: ignore[import-untyped]`
- Line 73, 148: missing args for constructor — fix the call
- Line 94: `.name` on `str` — fix type (it's actually a Path)
- Lines 102-106: None attrs on video_meta — add None guard

- [ ] **Step 3: Verify**

Run: `uv run basedpyright --level error src/pipeline.py src/references/ 2>&1 | tail -1`
Expected: 0 errors

- [ ] **Step 4: Commit**

```bash
git add src/pipeline.py src/references/
git commit -m "$(cat <<'EOF'
fix(types): fix pipeline and reference_builder type errors
EOF
)"
```

---

### Task 10: Fix remaining errors (video_writer, rtmlib, misc)

**Files:**
- Modify: `src/utils/video_writer.py`
- Modify: `src/utils/geometry.py`
- Modify: `src/pose_estimation/rtmlib_extractor.py`
- Modify: `src/pose_estimation/h36m_extractor.py`
- Modify: `src/pose_estimation/person_selector.py`
- Modify: `src/tracking/deepsort_tracker.py`

**Errors:** ~20

- [ ] **Step 1: Fix video_writer.py**

PyAV streams have dynamic attributes. Add `# type: ignore[attr-defined]` for `.width`, `.height`, `.pix_fmt` assignments and `.encode()` calls.

- [ ] **Step 2: Fix geometry.py lines 132, 144**

Add `.astype(np.float32)` to return values.

- [ ] **Step 3: Fix rtmlib_extractor.py remaining errors**

- Lines 96, 99: `Variable not allowed in type expression` — these use runtime values in type hints. Fix by using `ClassVar` or string annotations.
- Lines 102-103: `Object of type "None" cannot be called` — add None guard for `BodyWithFeet`
- Line 297: `update` on None — add None guard for tracker
- Lines 410-413: subscript on None — add None guard for detection results
- Line 815: return type mismatch — fix return statement

- [ ] **Step 4: Fix h36m_extractor.py remaining errors**

- Line 323, 326: `Variable not allowed in type expression` — use string annotations
- Lines 330, 334: `Object of type "None" cannot be called` — add None guards
- Line 388: `.orig_shape` on `object` — cast or add type ignore
- Line 607: `.copy()` on None — add None guard

- [ ] **Step 5: Fix person_selector.py line 101**

`set_window_title` on None — add None guard for cv2 window.

- [ ] **Step 6: Fix deepsort_tracker.py lines 137, 139**

`update_tracks` on None — add None guard for tracker.

- [ ] **Step 7: Verify full codebase**

Run: `uv run basedpyright --level error src/ 2>&1 | tail -1`
Expected: `0 errors`

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/ -v -m "not slow" --tb=short -q 2>&1 | tail -5`
Expected: tests still pass

- [ ] **Step 9: Commit**

```bash
git add src/utils/ src/pose_estimation/ src/tracking/
git commit -m "$(cat <<'EOF'
fix(types): fix remaining type errors in utils, pose, tracking
EOF
)"
```

---

### Task 11: Push and verify CI

- [ ] **Step 1: Final basedpyright check**

Run: `uv run basedpyright --level error src/ 2>&1 | tail -1`
Expected: `0 errors, X warnings, 0 notes`

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v -m "not slow" --tb=short -q 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 3: Merge to master and push**

```bash
git checkout master && git merge feature/basedpyright-fixes && git branch -d feature/basedpyright-fixes
git push origin master
```

- [ ] **Step 4: Verify CI passes**

Run: `gh run list --limit 1`
Expected: `completed success`

---

## Verification

After all tasks:
1. `uv run basedpyright --level error src/` — 0 errors
2. `uv run pytest tests/ -v -m "not slow" --tb=short` — all pass
3. `gh run list --limit 1` — CI green
