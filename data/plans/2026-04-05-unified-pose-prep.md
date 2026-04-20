# Unified Pose Preparation Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract duplicated pose preparation logic from CLI (`visualize_with_skeleton.py`) and Gradio (`gradio_helpers.py`) into a single `prepare_poses()` function, making it impossible for the two frontends to diverge on pose quality.

**Architecture:** A `PreparedPoses` dataclass and `prepare_poses()` function in `src/visualization/pipeline.py` encapsulate the full pipeline: extract 2D → fill gaps → smooth → 3D lift + CorrectiveLens. Both CLI and Gradio call this function, then render with the existing `VizPipeline`.

**Tech Stack:** RTMPoseExtractor, CorrectiveLens (wraps AthletePose3DExtractor with ONNX auto-detect), GapFiller, PoseSmoother, VizPipeline

---

## Current Divergences (Why This Matters)

| Feature | CLI (`visualize_with_skeleton.py`) | Gradio (`gradio_helpers.py`) | Problem |
|---------|-----|--------|---------|
| Frame skip | 1 (every frame) | 4 (slider default) | Gradio has NaN gaps |
| NaN handling | None (no gaps) | Naive `np.interp` | Bad interpolation |
| Smoothing | **None** | PoseSmoother | CLI doesn't smooth |
| 3D lifting | AthletePose3DExtractor (PyTorch) | ONNXPoseExtractor (ONNX) | Different code paths |
| CorrectiveLens | Runs but **output UNUSED** | Missing entirely | Neither frontend benefits |
| Model path | `--model-3d` flag | Hardcoded relative path | Inconsistent resolution |

**Critical bug found:** CLI line 341 constructs `VizPipeline(poses_norm=poses_viz)` with the RAW (uncorrected) poses, not `poses_viz_corrected`. The CorrectiveLens output is computed at line 290 but discarded.

After this plan: ONE function, ONE code path, ALL features applied correctly.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/visualization/pipeline.py` | MODIFY | Add `PreparedPoses`, `prepare_poses()`, `_resolve_model_3d()` |
| `src/gradio_helpers.py` | MODIFY | Replace ~120 lines of inline pose prep with `prepare_poses()` call |
| `scripts/visualize_with_skeleton.py` | MODIFY | Replace ~175 lines of inline pose prep with `prepare_poses()` call |
| `tests/visualization/test_pipeline.py` | MODIFY | Add tests for `prepare_poses()` with mocked extractors |

---

### Task 1: Add `PreparedPoses` dataclass and `prepare_poses()` function

**Files:**
- Modify: `src/visualization/pipeline.py`
- Modify: `tests/visualization/test_pipeline.py`

- [ ] **Step 1: Write failing tests for `prepare_poses()`**

Add to `tests/visualization/test_pipeline.py`:

```python
"""Tests for unified pose preparation pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


def _fake_meta(w=640, h=480, fps=30, num_frames=10):
    return SimpleNamespace(width=w, height=h, fps=fps, num_frames=num_frames)


def _fake_extraction(n=10):
    """Create a fake TrackedExtraction result."""
    extraction = mock.MagicMock()
    extraction.poses = np.random.rand(n, 17, 3).astype(np.float32)
    extraction.foot_keypoints = None
    extraction.frame_indices = np.arange(n)
    extraction.valid_mask.return_value = np.ones(n, dtype=bool)
    return extraction


class TestPreparePoses:
    def test_returns_prepared_poses(self):
        """prepare_poses returns PreparedPoses with correct shapes."""
        from src.visualization.pipeline import PreparedPoses, prepare_poses

        with (
            mock.patch("src.visualization.pipeline.get_video_meta", return_value=_fake_meta()),
            mock.patch("src.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("src.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("src.visualization.pipeline._resolve_model_3d", return_value=Path("model.onnx")),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()
            MockLens.return_value.correct_sequence.return_value = (
                np.random.rand(10, 17, 2).astype(np.float32) * 0.5 + 0.25,
                np.random.rand(10, 17, 3).astype(np.float32),
            )

            result = prepare_poses(Path("test.mp4"))

        assert isinstance(result, PreparedPoses)
        assert result.poses_norm.shape == (10, 17, 2)
        assert result.poses_px.shape == (10, 17, 3)
        assert result.poses_3d is not None
        assert result.poses_3d.shape == (10, 17, 3)
        assert result.n_valid == 10
        assert result.n_total == 10

    def test_no_corrective_lens_when_disabled(self):
        """When use_corrective_lens=False, CorrectiveLens is not called."""
        from src.visualization.pipeline import prepare_poses

        with (
            mock.patch("src.visualization.pipeline.get_video_meta", return_value=_fake_meta()),
            mock.patch("src.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("src.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("src.visualization.pipeline.ONNXPoseExtractor") as MockONNX,
            mock.patch("src.visualization.pipeline._resolve_model_3d", return_value=Path("model.onnx")),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()
            MockONNX.return_value.estimate_3d.return_value = np.random.rand(10, 17, 3).astype(np.float32)

            result = prepare_poses(Path("test.mp4"), use_corrective_lens=False)

        MockLens.assert_not_called()
        assert result.poses_3d is not None

    def test_no_3d_when_model_missing(self):
        """When model not found, poses_3d is None but poses_norm is still valid."""
        from src.visualization.pipeline import prepare_poses

        with (
            mock.patch("src.visualization.pipeline.get_video_meta", return_value=_fake_meta()),
            mock.patch("src.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("src.visualization.pipeline._resolve_model_3d", return_value=None),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()

            result = prepare_poses(Path("test.mp4"))

        assert result.poses_3d is None
        assert result.poses_norm.shape == (10, 17, 2)

    def test_gap_filling_when_frame_skip(self):
        """NaN frames from frame_skip are filled."""
        from src.visualization.pipeline import prepare_poses

        extraction = _fake_extraction(20)
        # Simulate frame_skip=4: only frames 0, 4, 8, 12, 16 have poses
        raw = np.full((20, 17, 3), np.nan, dtype=np.float32)
        for i in [0, 4, 8, 12, 16]:
            raw[i] = np.random.rand(17, 3).astype(np.float32)
        extraction.poses = raw
        extraction.valid_mask.return_value = np.array([i in [0, 4, 8, 12, 16] for i in range(20)])

        with (
            mock.patch("src.visualization.pipeline.get_video_meta", return_value=_fake_meta(num_frames=20)),
            mock.patch("src.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("src.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("src.visualization.pipeline._resolve_model_3d", return_value=Path("model.onnx")),
        ):
            MockExt.return_value.extract_video_tracked.return_value = extraction
            MockLens.return_value.correct_sequence.return_value = (
                np.random.rand(20, 17, 2).astype(np.float32) * 0.5 + 0.25,
                np.random.rand(20, 17, 3).astype(np.float32),
            )

            result = prepare_poses(Path("test.mp4"), frame_skip=4)

        # No NaN in output
        assert not np.isnan(result.poses_norm).any()
        assert result.n_valid == 5  # Only 5 frames were originally valid

    def test_smooth_disabled(self):
        """When smooth=False, PoseSmoother is not called."""
        from src.visualization.pipeline import prepare_poses

        with (
            mock.patch("src.visualization.pipeline.get_video_meta", return_value=_fake_meta()),
            mock.patch("src.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("src.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("src.visualization.pipeline.PoseSmoother") as MockSmooth,
            mock.patch("src.visualization.pipeline._resolve_model_3d", return_value=Path("model.onnx")),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()
            MockLens.return_value.correct_sequence.return_value = (
                np.random.rand(10, 17, 2).astype(np.float32) * 0.5 + 0.25,
                np.random.rand(10, 17, 3).astype(np.float32),
            )

            prepare_poses(Path("test.mp4"), smooth=False)

        MockSmooth.assert_not_called()


class TestResolveModel3d:
    def test_explicit_path_returned(self, tmp_path):
        from src.visualization.pipeline import _resolve_model_3d

        model = tmp_path / "model.onnx"
        model.touch()
        result = _resolve_model_3d(model)
        assert result == model

    def test_none_when_not_found(self, tmp_path):
        from src.visualization.pipeline import _resolve_model_3d

        result = _resolve_model_3d(tmp_path / "nonexistent.onnx")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/visualization/test_pipeline.py -v --no-header 2>&1 | tail -20`
Expected: FAIL — `ImportError: cannot import name 'PreparedPoses' from 'src.visualization.pipeline'`

- [ ] **Step 3: Implement `PreparedPoses`, `prepare_poses()`, and `_resolve_model_3d()`**

Add to `src/visualization/pipeline.py` (after the existing `VizPipeline` class):

```python
# ------------------------------------------------------------------
# Unified pose preparation
# ------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

# ... (existing imports stay) ...

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_3D_CANDIDATES = [
    _PROJECT_ROOT / "data" / "models" / "motionagformer-s-ap3d.onnx",
    _PROJECT_ROOT / "data" / "models" / "motionagformer-s-ap3d.pth.tr",
    Path("data/models/motionagformer-s-ap3d.onnx"),
    Path("data/models/motionagformer-s-ap3d.pth.tr"),
]


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
    use_corrective_lens: bool = True,
    model_3d_path: Path | str | None = None,
    blend_threshold: float = 0.5,
    smooth: bool = True,
    device: str = "auto",
    progress_cb: Callable[[float, str], None] | None = None,
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
        smooth: Apply One-Euro Filter smoothing (default True).
        device: Device string ("auto", "cuda", "cpu").
        progress_cb: Optional callback ``(progress_0_to_1, message)``.

    Returns:
        PreparedPoses with all data needed for VizPipeline construction.
    """
    from src.device import DeviceConfig
    from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
    from src.utils.gap_filling import GapFiller
    from src.utils.smoothing import PoseSmoother, get_skating_optimized_config

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

    # --- Step 2: Fill NaN gaps ---
    if nan_mask.any() and n_valid >= 2:
        filler = GapFiller(fps=meta.fps)
        filled, report = filler.fill_gaps(raw_poses, ~nan_mask)
        raw_poses = filled
        if report.gaps:
            logger.info(
                "Filled %d gap(s): %s", len(report.gaps), report.strategy_used
            )
        if raw_foot_kps is not None:
            foot_nan = np.isnan(raw_foot_kps[:, 0, 0])
            if foot_nan.any() and (~foot_nan).sum() >= 2:
                raw_foot_kps, _ = filler.fill_gaps(raw_foot_kps, ~foot_nan)

    poses_norm = raw_poses[:, :, :2].copy()
    confs = raw_poses[:, :, 2].copy()

    # --- Step 3: Smooth ---
    if smooth and len(poses_norm) > 2:
        smooth_config = get_skating_optimized_config(meta.fps)
        smoother = PoseSmoother(smooth_config, freq=meta.fps)
        poses_norm = smoother.smooth(poses_norm)

    if progress_cb:
        progress_cb(0.4, "3D pose estimation...")

    # --- Step 4: 3D lift + CorrectiveLens ---
    poses_3d = None
    model_path = _resolve_model_3d(model_3d_path)

    if model_path is not None and use_corrective_lens:
        from src.pose_3d import CorrectiveLens

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
        from src.pose_3d.onnx_extractor import ONNXPoseExtractor

        onnx = ONNXPoseExtractor(model_path, device=cfg.device)
        poses_3d = onnx.estimate_3d(poses_norm)
        logger.info("3D poses estimated (no CorrectiveLens)")
    else:
        logger.warning(
            "No 3D model found. Skeleton will use raw 2D poses without correction."
        )

    # --- Step 5: Build pixel coordinates ---
    poses_px = raw_poses.copy()
    poses_px[:, :, 0] *= meta.width
    poses_px[:, :, 1] *= meta.height

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
```

**Important:** The `get_video_meta` import must already exist at the top of `pipeline.py`. If not, add:

```python
from src.utils.video import get_video_meta
```

Also add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/visualization/test_pipeline.py -v --no-header 2>&1 | tail -20`
Expected: All 7 tests PASS (5 prepare_poses + 2 resolve_model_3d)

- [ ] **Step 5: Run existing pipeline tests for regressions**

Run: `uv run pytest tests/visualization/ -v --no-header 2>&1 | tail -20`
Expected: All existing tests still PASS (no regressions to VizPipeline)

- [ ] **Step 6: Commit**

```bash
git add src/visualization/pipeline.py tests/visualization/test_pipeline.py
git commit -m "feat(viz): add unified prepare_poses() pipeline for CLI and Gradio"
```

---

### Task 2: Wire Gradio to use `prepare_poses()`

**Files:**
- Modify: `src/gradio_helpers.py`

- [ ] **Step 1: Rewrite `process_video_pipeline()` to use `prepare_poses()`**

Replace the entire body of `process_video_pipeline()` (lines 119-287) with:

```python
def process_video_pipeline(
    video_path: str | Path,
    person_click: PersonClick | None,
    frame_skip: int,
    layer: int,
    tracking: str,
    blade_3d: bool,
    export: bool,
    output_path: str | Path,
    progress_cb=None,
) -> dict:
    """Run the full visualization pipeline (mirrors visualize_with_skeleton.py)."""
    from src.visualization.pipeline import VizPipeline, prepare_poses

    video_path = Path(video_path) if isinstance(video_path, str) else video_path
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    # --- Unified pose preparation ---
    prepared = prepare_poses(
        video_path,
        person_click=person_click,
        frame_skip=frame_skip,
        tracking=tracking,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb(0.6, "Rendering...")

    # --- Build rendering pipeline ---
    pipe = VizPipeline(
        meta=prepared.meta,
        poses_norm=prepared.poses_norm,
        poses_px=prepared.poses_px,
        foot_kps=prepared.foot_kps,
        poses_3d=prepared.poses_3d,
        layer=layer,
        confs=prepared.confs,
        frame_indices=prepared.frame_indices,
    )

    meta = prepared.meta
    cap = cv2.VideoCapture(str(video_path))
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
            progress_cb(0.6 + 0.3 * frame_idx / total, f"Rendering frame {frame_idx}/{total}")

    cap.release()
    writer.close()

    if progress_cb:
        progress_cb(0.95, "Saving exports...")

    export_result = (
        pipe.save_exports(output_path) if export else {"poses_path": None, "csv_path": None}
    )

    # Generate animated GLB for 3D viewer
    glb_path = None
    if prepared.poses_3d is not None:
        from src.visualization.export_3d_animated import poses_to_animated_glb

        glb_path = poses_to_animated_glb(prepared.poses_3d, fps=meta.fps)

    return {
        "video_path": str(output_path),
        "poses_path": export_result["poses_path"],
        "csv_path": export_result["csv_path"],
        "glb_path": glb_path,
        "poses_3d": prepared.poses_3d,
        "stats": {
            "total_frames": total,
            "valid_frames": prepared.n_valid,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
        },
    }
```

**Key changes from current Gradio code:**
1. Pose extraction, NaN interpolation, smoothing, 3D lifting — all replaced by `prepare_poses()` call
2. CorrectiveLens is now ON (was missing before — this fixes the regression)
3. GapFiller replaces naive `np.interp`
4. Removed unused imports (`DeviceConfig`, `PoseSmoother`, `get_skating_optimized_config`)

- [ ] **Step 2: Remove unused imports from gradio_helpers.py**

The following imports are no longer needed at the top of `gradio_helpers.py`:
- `from src.device import DeviceConfig` (used inside `prepare_poses()` now)
- No other imports to remove — `cv2`, `numpy`, `Path`, `PersonClick`, `get_video_meta`, `H264Writer` are still used

Actually, `get_video_meta` is no longer used directly in `gradio_helpers.py` (it's used inside `prepare_poses()`). Check if it's used elsewhere in the file — if not, remove it.

```python
# Remove this import if no longer used:
from src.utils.video import get_video_meta
```

- [ ] **Step 3: Run existing Gradio tests**

Run: `uv run pytest tests/ -k "gradio" -v --no-header 2>&1 | tail -15`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/gradio_helpers.py
git commit -m "refactor(gradio): use unified prepare_poses() pipeline"
```

---

### Task 3: Wire CLI to use `prepare_poses()`

**Files:**
- Modify: `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Replace inline pose preparation with `prepare_poses()` call**

In `scripts/visualize_with_skeleton.py`, replace lines 172-300 (from `# Load or extract poses` through `poses_viz_corrected = np.clip(...)`) with:

```python
    # --- Unified pose preparation ---
    from src.visualization.pipeline import prepare_poses

    model_3d = args.model_3d
    if model_3d is None:
        default_model = Path("data/models/motionagformer-s-ap3d.onnx")
        if default_model.exists():
            model_3d = default_model
        else:
            print("Warning: 3D model not found. Download motionagformer-s-ap3d to data/models/")
            model_3d = None

    prepared = prepare_poses(
        args.video,
        person_click=person_click,
        frame_skip=1,  # CLI always processes every frame
        tracking=args.tracking,
        use_corrective_lens=True,
        model_3d_path=model_3d,
        blend_threshold=args.blend_threshold,
        smooth=True,
        device="auto",
    )

    poses_viz = prepared.poses_norm
    poses = prepared.poses_px
    poses_3d = prepared.poses_3d
    raw_foot_kps = prepared.foot_kps
    confs = prepared.confs
    meta = prepared.meta

    print(f"Poses ready: {len(poses_viz)} frames ({prepared.n_valid} valid)")
    if poses_3d is not None:
        print(f"3D poses: {poses_3d.shape}")
```

**Key changes from current CLI code:**
1. CorrectiveLens output is now ACTUALLY USED (fixes the bug where it was computed but discarded)
2. Smoothing is now ON (was missing)
3. Model path resolution is centralized
4. ~130 lines of inline pose prep replaced by a single function call

**What stays in CLI (NOT moved to prepare_poses):**
- Argument parsing (lines 40-157) — CLI-specific
- Person selection (lines 202-246) — CLI-specific terminal UI
- Blade detection (lines 302-327) — CLI-only optional feature
- Spatial reference detection (lines 329-335) — CLI-only
- Subtitles (lines 357-363) — CLI-only
- HUD drawing (lines 484-501) — CLI-specific overlay
- CoM trajectory (lines 414-427) — CLI-only optional feature
- Profiling (lines 510-532) — CLI-only
- Render loop and export — uses VizPipeline (already shared)

- [ ] **Step 2: Update VizPipeline construction to use `prepared` data**

Find the VizPipeline construction (around line 340) and update it:

```python
    # Before:
    pipe = VizPipeline(
        meta=meta,
        poses_norm=poses_viz,
        poses_px=poses,
        foot_kps=raw_foot_kps,
        poses_3d=poses_3d,
        layer=args.layer,
    )

    # After (no change needed — same variables, just sourced from prepare_poses()):
    pipe = VizPipeline(
        meta=meta,
        poses_norm=poses_viz,
        poses_px=poses,
        foot_kps=raw_foot_kps,
        poses_3d=poses_3d,
        layer=args.layer,
        confs=confs,
    )
```

The only difference is adding `confs=confs` (which is now available from `prepare_poses()`).

- [ ] **Step 3: Remove now-unused imports**

The following imports from the top of `visualize_with_skeleton.py` are no longer needed:
- `from src.pose_3d import CorrectiveLens` — used inside `prepare_poses()` now
- `from src.pose_3d import AthletePose3DExtractor` — used inside `prepare_poses()` now (via CorrectiveLens)

Keep:
- `from src.detection.blade_edge_detector_3d import BladeEdgeDetector3D` — still used for blade detection
- `from src.detection.spatial_reference import SpatialReferenceDetector` — still used
- `from src.pose_estimation import H36Key, H36MExtractor` — H36Key still used in HUD
- `from src.types import BladeState3D` — still used
- All other imports — still used

- [ ] **Step 4: Verify by running on a test video**

Run: `uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/VOLODYA.MOV --layer 3 --output /tmp/unified_test.mp4 2>&1 | head -20`
Expected: Processes successfully, CorrectiveLens and smoothing are applied

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_with_skeleton.py
git commit -m "refactor(cli): use unified prepare_poses() pipeline"
```

---

### Task 4: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v --no-header 2>&1 | tail -30`
Expected: All tests PASS (no regressions)

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/visualization/pipeline.py src/gradio_helpers.py scripts/visualize_with_skeleton.py 2>&1`
Expected: No issues

- [ ] **Step 3: Manual verification — Gradio**

Run: `uv run python scripts/gradio_app.py`
Expected: App launches without errors. Upload video → detect person → process → output video has CorrectiveLens-corrected skeleton

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task |
|---|---|
| Unified pose preparation | Task 1 — `prepare_poses()` function |
| Used by CLI | Task 3 — `visualize_with_skeleton.py` wired |
| Used by Gradio | Task 2 — `gradio_helpers.py` wired |
| CorrectiveLens always on | Task 1 — `use_corrective_lens=True` default |
| CorrectiveLens output used | Task 3 — CLI now uses `prepared.poses_norm` (corrected) |
| Smoothing always on | Task 1 — `smooth=True` default |
| Gap filling (not naive interp) | Task 1 — GapFiller replaces `np.interp` |
| GLB export preserved | Task 2 — still generates GLB from `prepared.poses_3d` |
| 3D model auto-detection | Task 1 — `_resolve_model_3d()` checks 4 candidates |
| Graceful fallback without model | Task 1 — `poses_3d=None`, raw 2D poses used |
| Frame skip configurable | Task 1 — `frame_skip` parameter, default 1 |

### 2. Placeholder Scan

No TBD/TODO/implement-later found. All code blocks contain complete implementations.

### 3. Type Consistency

- `PreparedPoses.poses_norm`: `NDArray[np.float32]` — matches `VizPipeline` input
- `PreparedPoses.poses_px`: `NDArray[np.float32]` — matches `VizPipeline` input
- `PreparedPoses.meta`: `object` — matches `VizPipeline.meta`
- `prepare_poses()` returns `PreparedPoses` — consumed by both CLI and Gradio
- `VizPipeline` constructor unchanged — backward compatible

### 4. Risk Assessment

- **Low risk:** `prepare_poses()` is additive — existing VizPipeline is unchanged
- **Medium risk:** CorrectiveLens is now ON in Gradio (was missing). This changes output quality — but that's the intended fix
- **Low risk:** CLI now smooths poses (was not smoothing). Quality should improve
- **Low risk:** GapFiller replaces naive `np.interp`. GapFiller is already tested
