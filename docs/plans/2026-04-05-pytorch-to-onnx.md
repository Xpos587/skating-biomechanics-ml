# PyTorch → ONNX Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove PyTorch from runtime dependencies by converting MotionAGFormer-S and TCPFormer to ONNX, using ONNX Runtime for all inference.

**Architecture:** Convert `.pth.tr` checkpoints → `.onnx` via `torch.onnx.export()` (opset 17). Create new ONNX-based extractors that use `onnxruntime.InferenceSession`. Keep PyTorch as optional dev-dependency for future fine-tuning and re-export.

**Tech Stack:** PyTorch 2.11 (export only), ONNX Runtime GPU, ONNX opset 17

**Savings:** ~1.2GB venv size, remove `torch>=2.5.0` and `timm` from runtime deps.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/export_models_to_onnx.py` | Create | One-time conversion script: `.pth.tr` → `.onnx` |
| `src/pose_3d/onnx_extractor.py` | Create | ONNX Runtime-based 3D pose extractor |
| `src/pose_3d/athletepose_extractor.py` | Modify | Delegate to ONNX extractor when `.onnx` available |
| `src/pose_3d/tcpformer_extractor.py` | Modify | Delegate to ONNX extractor when `.onnx` available |
| `pyproject.toml` | Modify | Move `torch`, `timm` to optional `[dev]` deps |
| `data/models/*.onnx` | Create | Converted model artifacts |
| `tests/pose_3d/test_onnx_extractor.py` | Create | Tests for ONNX extractor |

---

## Task 1: Export Script for MotionAGFormer-S

**Files:**
- Create: `scripts/export_models_to_onnx.py`
- Read: `src/models/motionagformer/MotionAGFormer.py`
- Read: `src/pose_3d/athletepose_extractor.py` (for loading logic)

This script runs ONCE to produce ONNX files. PyTorch is only needed at export time.

- [ ] **Step 1: Create export script**

```python
#!/usr/bin/env python3
"""Convert PyTorch pose estimation models to ONNX format.

Usage:
    uv run python scripts/export_models_to_onnx.py --model motionagformer-s
    uv run python scripts/export_models_to_onnx.py --model tcpformer
    uv run python scripts/export_models_to_onnx.py --all
"""
import argparse
from pathlib import Path

import torch
import numpy as np


def export_motionagformer_s(checkpoint_path: str, output_path: str) -> None:
    """Export MotionAGFormer-S to ONNX."""
    from src.models.motionagformer.MotionAGFormer import MotionAGFormer

    model = MotionAGFormer(
        n_layers=4,
        dim_in=3,
        dim_feat=64,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        num_heads=4,
        num_joints=17,
        n_frames=81,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        hierarchical=False,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
    )

    # Load checkpoint (handle different formats)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Strip "module." prefix (DataParallel)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    # Dummy input: [1, 81, 17, 3] (batch, frames, joints, channels)
    dummy_input = torch.randn(1, 81, 17, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["poses_2d"],
        output_names=["poses_3d"],
        dynamic_axes={
            "poses_2d": {0: "batch"},
            "poses_3d": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"Exported MotionAGFormer-S → {output_path}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"poses_2d": dummy_input.numpy()})
    print(f"  Verification: output shape = {result[0].shape}, ONNX OK")


def export_tcpformer(checkpoint_path: str, output_path: str) -> None:
    """Export TCPFormer to ONNX."""
    from src.models.tcpformer.TCPFormer import MemoryInducedTransformer

    model = MemoryInducedTransformer(
        n_layers=16,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        num_heads=4,
        num_joints=17,
        n_frames=81,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        hierarchical=False,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 81, 17, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["poses_2d"],
        output_names=["poses_3d"],
        dynamic_axes={
            "poses_2d": {0: "batch"},
            "poses_3d": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"Exported TCPFormer → {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"poses_2d": dummy_input.numpy()})
    print(f"  Verification: output shape = {result[0].shape}, ONNX OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--model", choices=["motionagformer-s", "tcpformer"], help="Model to export")
    parser.add_argument("--all", action="store_true", help="Export all models")
    args = parser.parse_args()

    models_dir = Path("data/models")

    if args.all or args.model == "motionagformer-s":
        src = models_dir / "motionagformer-s-ap3d.pth.tr"
        dst = models_dir / "motionagformer-s-ap3d.onnx"
        if src.exists():
            export_motionagformer_s(str(src), str(dst))
        else:
            print(f"Skip: {src} not found")

    if args.all or args.model == "tcpformer":
        src = models_dir / "TCPFormer_ap3d_81.pth.tr"
        dst = models_dir / "TCPFormer_ap3d_81.onnx"
        if src.exists():
            export_tcpformer(str(src), str(dst))
        else:
            print(f"Skip: {src} not found")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run export for MotionAGFormer-S**

```bash
uv run python scripts/export_models_to_onnx.py --model motionagformer-s
```

Expected: `data/models/motionagformer-s-ap3d.onnx` created (~25MB), verification passes.

- [ ] **Step 3: Run export for TCPFormer**

```bash
uv run python scripts/export_models_to_onnx.py --model tcpformer
```

Expected: `data/models/TCPFormer_ap3d_81.onnx` created (~200MB), verification passes.

- [ ] **Step 4: Commit export script and ONNX models**

```bash
echo "data/models/*.onnx" >> data/models/.gitignore
git add scripts/export_models_to_onnx.py
git commit -m "feat: add PyTorch→ONNX model export script"
```

Note: ONNX files are large, add to `.gitignore`. Document in README how to re-export.

---

## Task 2: ONNX Extractor

**Files:**
- Create: `src/pose_3d/onnx_extractor.py`
- Create: `tests/pose_3d/test_onnx_extractor.py`

This is the runtime replacement for `AthletePose3DExtractor` and `TCPFormerExtractor` that uses ONNX Runtime instead of PyTorch.

- [ ] **Step 1: Write failing tests**

```python
# tests/pose_3d/test_onnx_extractor.py
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def onnx_model_path():
    p = Path("data/models/motionagformer-s-ap3d.onnx")
    if not p.exists():
        pytest.skip("ONNX model not exported yet")
    return p


def test_onnx_extractor_init(onnx_model_path):
    from src.pose_3d.onnx_extractor import ONNXPoseExtractor
    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    assert ext.temporal_window == 81


def test_onnx_extractor_single_window(onnx_model_path):
    from src.pose_3d.onnx_extractor import ONNXPoseExtractor
    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input: (81, 17, 2) normalized 2D poses
    poses_2d = np.random.rand(81, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (81, 17, 3)
    # Z coordinates should be reasonable (not all zeros, not huge)
    assert not np.allclose(result[:, :, 2], 0)
    assert np.nanmax(np.abs(result)) < 10


def test_onnx_extractor_long_sequence(onnx_model_path):
    from src.pose_3d.onnx_extractor import ONNXPoseExtractor
    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input longer than 81 frames — should be windowed
    poses_2d = np.random.rand(200, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (200, 17, 3)


def test_onnx_extractor_short_sequence(onnx_model_path):
    from src.pose_3d.onnx_extractor import ONNXPoseExtractor
    ext = ONNXPoseExtractor(onnx_model_path, device="cpu")
    # Input shorter than 81 frames — should be padded
    poses_2d = np.random.rand(30, 17, 2).astype(np.float32) * 0.5 + 0.25
    result = ext.estimate_3d(poses_2d)
    assert result.shape == (30, 17, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/pose_3d/test_onnx_extractor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pose_3d.onnx_extractor'`

- [ ] **Step 3: Implement ONNX extractor**

```python
# src/pose_3d/onnx_extractor.py
"""ONNX Runtime-based 3D pose estimation.

Drop-in replacement for AthletePose3DExtractor / TCPFormerExtractor
that uses ONNX Runtime instead of PyTorch.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ONNXPoseExtractor:
    """3D pose estimation using ONNX Runtime.

    Loads a converted MotionAGFormer or TCPFormer ONNX model and
    runs inference via onnxruntime — no PyTorch dependency needed.

    Args:
        model_path: Path to .onnx model file.
        device: ``"cpu"`` or ``"cuda"`` (falls back to CPU).
        temporal_window: Number of frames per inference window (default 81).
    """

    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str,
        device: str = "cpu",
        temporal_window: int = 81,
    ) -> None:
        import onnxruntime as ort

        self.temporal_window = temporal_window
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active = self.session.get_providers()[0]
        logger.info(f"ONNX 3D pose: {model_path.name}, provider={active}")

    def estimate_3d(self, poses_2d: NDArray[np.float32]) -> NDArray[np.float32]:
        """Estimate 3D poses from 2D input.

        Args:
            poses_2d: (N, 17, 2) normalized coordinates [0, 1].

        Returns:
            (N, 17, 3) with estimated z-coordinates.
        """
        n_frames = len(poses_2d)
        w = self.temporal_window

        if n_frames <= w:
            return self._infer_window(poses_2d)[:n_frames]

        # Sliding window with stride = w // 2
        stride = w // 2
        results = np.zeros((n_frames, 17, 3), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        start = 0
        while start < n_frames:
            end = min(start + w, n_frames)
            window = poses_2d[start:end]
            out = self._infer_window(window)
            results[start:end] += out[:end - start]
            counts[start:end] += 1
            if end == n_frames:
                break
            start += stride

        # Average overlapping regions
        counts = np.maximum(counts, 1)[:, np.newaxis, np.newaxis]
        results /= counts
        return results

    def _infer_window(self, poses_2d: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run inference on a single window (pad if needed). (N, 17, 2) → (N, 17, 3)."""
        n = len(poses_2d)
        w = self.temporal_window

        # Pad to window size if needed
        if n < w:
            pad_count = w - n
            # Repeat last frame for padding
            padded = np.concatenate([poses_2d, np.tile(poses_2d[-1:], (pad_count, 1, 1))], axis=0)
        else:
            padded = poses_2d

        # Add confidence channel (=1.0) and batch dim: (1, w, 17, 3)
        conf = np.ones((w, 17, 1), dtype=np.float32)
        inp = np.concatenate([padded, conf], axis=2)[np.newaxis]

        result = self.session.run(None, {self.input_name: inp})[0]
        # result: (1, w, 17, 3) → (w, 17, 3)
        return result[0]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/pose_3d/test_onnx_extractor.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pose_3d/onnx_extractor.py tests/pose_3d/test_onnx_extractor.py
git commit -m "feat: ONNX Runtime 3D pose extractor (no PyTorch dependency)"
```

---

## Task 3: Wire ONNX Extractor into Existing Code

**Files:**
- Modify: `src/pose_3d/athletepose_extractor.py`
- Modify: `src/pose_3d/tcpformer_extractor.py`

Both extractors should auto-detect `.onnx` files and delegate to `ONNXPoseExtractor` when available, avoiding PyTorch import entirely.

- [ ] **Step 1: Add ONNX auto-detection to AthletePose3DExtractor**

In `src/pose_3d/athletepose_extractor.py`, add at the top of `__init__` (before any torch import):

```python
def __init__(self, ...):
    # ... existing param setup ...

    # Auto-detect ONNX model
    onnx_path = Path(model_path).with_suffix(".onnx") if model_path else None
    if onnx_path and onnx_path.exists():
        from src.pose_3d.onnx_extractor import ONNXPoseExtractor
        self._onnx = ONNXPoseExtractor(onnx_path, device=resolved_device)
        self._model = None
        logger.info(f"Using ONNX model: {onnx_path.name}")
        return

    # Fallback to PyTorch (only if ONNX not available)
    import torch
    # ... rest of existing __init__ ...
```

Add in `estimate_3d` method:

```python
def estimate_3d(self, poses_2d):
    if hasattr(self, '_onnx') and self._onnx is not None:
        return self._onnx.estimate_3d(poses_2d)
    # ... existing PyTorch path ...
```

- [ ] **Step 2: Same for TCPFormerExtractor**

Apply the identical ONNX auto-detection pattern to `src/pose_3d/tcpformer_extractor.py`.

- [ ] **Step 3: Run all existing tests**

```bash
uv run pytest tests/pose_3d/ -v
```

Expected: All existing tests still pass (they use Biomechanics3DEstimator or mock the models).

- [ ] **Step 4: Verify end-to-end with CLI**

```bash
uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/Waltz.mp4 --layer 0 --3d --output /tmp/test_onnx.mp4
```

Expected: Video produced successfully, 3D overlay rendered.

- [ ] **Step 5: Commit**

```bash
git add src/pose_3d/athletepose_extractor.py src/pose_3d/tcpformer_extractor.py
git commit -m "feat: auto-detect ONNX models in 3D extractors, skip PyTorch when available"
```

---

## Task 4: Move PyTorch to Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move torch and timm to optional dev group**

In `pyproject.toml`, change:

```toml
# Before (runtime deps):
"torch>=2.5.0",
"timm>=1.0.26",

# After (optional dev deps):
[project.optional-dependencies]
dev = [
    "torch>=2.5.0",
    "timm>=1.0.26",
]
```

Keep `onnxruntime-gpu>=1.24.4` in runtime deps (already there).

- [ ] **Step 2: Verify imports don't crash without torch**

```bash
uv run python -c "
# These should work without torch import:
from src.pose_3d.onnx_extractor import ONNXPoseExtractor
from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator
from src.gradio_helpers import process_video_pipeline
print('All imports OK without PyTorch')
"
```

Expected: All imports succeed, no `torch` imported.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: Same pass count as before (tests that need torch should be skipped gracefully).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "refactor: move torch/timm to optional dev dependency"
```

---

## Task 5: Verify Gradio Pipeline Works Without PyTorch

**Files:**
- Modify: `src/gradio_helpers.py` (ensure no torch import)

- [ ] **Step 1: Verify gradio_helpers has no torch dependency**

```bash
grep -n "torch" src/gradio_helpers.py
```

Expected: No matches. If any found, replace with ONNX path.

- [ ] **Step 2: Run full Gradio pipeline test**

```bash
uv run python scripts/gradio_app.py &
sleep 3
# Upload video and process via UI or API test
curl -s http://127.0.0.1:7860 | head -5
kill %1
```

Expected: Gradio starts, no torch import errors.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete PyTorch→ONNX migration, Gradio runs without torch"
```

---

## Summary

| Task | What | PyTorch Needed? |
|------|------|----------------|
| 1 | Export `.pth.tr` → `.onnx` | Yes (one-time) |
| 2 | `ONNXPoseExtractor` | No |
| 3 | Wire into existing extractors | No |
| 4 | Move torch to dev-dep | No |
| 5 | Verify Gradio works | No |

**After migration:**
- Runtime deps: `onnxruntime-gpu` (no `torch`, no `timm`)
- `uv sync` installs ~200MB less
- Dev: `uv sync --extra dev` to get torch back for fine-tuning/re-export
- All inference goes through ONNX Runtime (CPU + CUDA compat)
