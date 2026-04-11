# ML Models Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional ML models (depth estimation, optical flow, segmentation, foot tracking, video matting, inpainting) to the figure skating biomechanics pipeline, each toggleable via CLI flags and frontend checkboxes.

**Architecture:** New `src/ml/` package with a `ModelRegistry` that lazy-loads ONNX sessions and tracks VRAM budget. Per-frame models run during the render loop (not in `prepare_poses()`), passing results through `LayerContext.custom_data` to visualization layers. Analysis-time models integrate into `prepare_poses()`.

**Tech Stack:** ONNX Runtime (CUDA/CPU), OpenCV, huggingface_hub (model downloads), existing `DeviceConfig` for device resolution.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/ml/__init__.py` | Package init, re-exports |
| `src/ml/model_registry.py` | Central ONNX session lifecycle (lazy load, VRAM tracking, LRU eviction) |
| `src/ml/depth_anything.py` | Depth Anything V2 wrapper (ONNX inference, resize to model input, resize back) |
| `src/ml/optical_flow.py` | NeuFlowV2 wrapper (frame pair → dense flow field) |
| `src/ml/segment_anything.py` | SAM 2 wrapper (image + point prompt → binary mask) |
| `src/ml/foot_tracker.py` | FootTrackNet wrapper (image → person + foot bboxes) |
| `src/ml/video_matting.py` | RobustVideoMatting wrapper (RGB + optional mask → alpha matte) |
| `src/ml/inpainting.py` | LAMA wrapper (RGB + mask → inpainted RGB) |
| `scripts/download_ml_models.py` | CLI to download ONNX weights from HuggingFace to `data/models/` |
| `tests/ml/test_model_registry.py` | Unit tests for registry (mock ONNX sessions) |
| `tests/ml/test_depth_anything.py` | Unit tests for depth estimator |
| `tests/ml/test_optical_flow.py` | Unit tests for optical flow |
| `tests/ml/test_segment_anything.py` | Unit tests for SAM2 |
| `src/visualization/layers/depth_layer.py` | Renders depth map as color overlay |
| `src/visualization/layers/optical_flow_layer.py` | Renders optical flow as HSV color wheel |
| `src/visualization/layers/segmentation_layer.py` | Renders segmentation mask overlay |
| `src/visualization/layers/matting_layer.py` | Applies alpha matte (blur background) |

**Modified files:**
- `src/visualization/pipeline.py` — `VizPipeline` gets ML flags, builds ML layers
- `src/visualization/layers/base.py` — `LayerContext` gets `custom_data` (already exists, no change needed)
- `src/visualization/layers/__init__.py` — Re-export new layers
- `src/web_helpers.py` — `process_video_pipeline()` gets ML flags, creates registry, runs per-frame inference
- `src/backend/schemas.py` — `ProcessRequest` gets `depth`, `optical_flow`, `segment`, `foot_track`, `matting` bools
- `src/frontend/src/lib/schemas.ts` — Mirror new fields in Zod
- `src/frontend/src/lib/api.ts` — Pass new fields in processVideo request
- `src/frontend/src/types/index.ts` — Mirror new fields in TypeScript interface
- `src/frontend/src/pages/UploadPage.tsx` — Add checkboxes for ML features
- `src/frontend/src/pages/AnalyzePage.tsx` — Pass new params via URL search params
- `src/cli.py` — Add `--depth`, `--optical-flow`, `--segment`, `--foot-track`, `--matting` flags
- `pyproject.toml` — Add `huggingface_hub` dependency
- `data/models/.gitignore` — Add patterns for new ONNX models

---

## VRAM Budget (RTX 3050 Ti, 4GB)

| Model | Variant | VRAM (est.) | Phase |
|-------|---------|-------------|-------|
| Depth Anything V2 | Small (24.8M) | ~200MB | 1 |
| NeuFlowV2 | mixed | ~80MB | 1 |
| SAM 2 | Tiny (38.9M) | ~200MB | 2 |
| FootTrackNet | default (2.53M) | ~30MB | 2 |
| RobustVideoMatting | MobileNetV3 | ~40MB | 3 |
| LAMA | default (45.6M) | ~300MB | 3 |
| **Total all 6** | | **~850MB** | |

Existing usage: RTMPose ~100MB + MotionAGFormer ~27MB = ~127MB. Total ~977MB. Safe margin.

---

## Task 1: ModelRegistry

**Files:**
- Create: `src/ml/__init__.py`
- Create: `src/ml/model_registry.py`
- Create: `tests/ml/__init__.py`
- Create: `tests/ml/test_model_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/__init__.py
```

```python
# tests/ml/test_model_registry.py
"""Tests for ModelRegistry."""

from unittest import mock

import pytest


class TestModelRegistry:
    """Tests for central ONNX session lifecycle manager."""

    def test_create_registry_with_defaults(self):
        """Registry initializes with default settings."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.vram_budget_mb == 3800
        assert reg.vram_used_mb == 0
        assert len(reg._sessions) == 0

    def test_create_registry_custom_budget(self):
        """Registry accepts custom VRAM budget."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(vram_budget_mb=2000)
        assert reg.vram_budget_mb == 2000

    def test_register_model(self):
        """register() tracks model metadata without loading."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")
        assert reg.is_registered("depth_anything")
        assert reg.vram_used_mb == 0  # not loaded yet

    def test_get_loads_on_demand(self):
        """get() lazy-loads ONNX session on first call."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            session = reg.get("depth_anything")
            assert session is mock_session
            assert reg.vram_used_mb == 200

    def test_get_returns_cached_session(self):
        """Second get() returns same session without reloading."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            s1 = reg.get("depth_anything")
            s2 = reg.get("depth_anything")
            assert s1 is s2

    def test_get_unregistered_raises(self):
        """get() raises KeyError for unregistered model."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_release_frees_session(self):
        """release() unloads session and frees VRAM."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            reg.get("depth_anything")
            reg.release("depth_anything")
            assert reg.vram_used_mb == 0
            mock_session.release.assert_called_once()

    def test_release_unregistered_noop(self):
        """release() is a no-op for unregistered/already released model."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.release("nonexistent")  # should not raise

    def test_vram_budget_enforced(self):
        """get() raises RuntimeError if loading would exceed VRAM budget."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(vram_budget_mb=100)
        reg.register("big_model", vram_mb=200, path="/tmp/big.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            with pytest.raises(RuntimeError, match="VRAM budget"):
                reg.get("big_model")

    def test_device_passed_to_session(self):
        """ONNX session created with correct providers from DeviceConfig."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("test_model", vram_mb=50, path="/tmp/test.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session) as mock_cls:
            reg.get("test_model")
            mock_cls.assert_called_once_with(
                "/tmp/test.onnx",
                providers=["CPUExecutionProvider"],
            )

    def test_is_loaded(self):
        """is_loaded() returns correct state."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("m", vram_mb=50, path="/tmp/m.onnx")
        assert not reg.is_loaded("m")

        mock_session = mock.MagicMock()
        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            reg.get("m")
            assert reg.is_loaded("m")

    def test_list_models(self):
        """list_models() returns registered model IDs."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("a", vram_mb=50, path="/tmp/a.onnx")
        reg.register("b", vram_mb=50, path="/tmp/b.onnx")
        assert reg.list_models() == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_model_registry.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement ModelRegistry**

```python
# src/ml/__init__.py
"""Optional ML models for enhanced analysis.

Models are lazy-loaded via ModelRegistry and individually toggleable.
No model runs unless explicitly requested via CLI flags or API params.
"""

from src.ml.model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
```

```python
# src/ml/model_registry.py
"""Central ONNX session lifecycle manager.

Lazy-loads models on first use, tracks VRAM budget, supports LRU eviction
when budget is exceeded. All models run via ONNX Runtime (no PyTorch at runtime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import onnxruntime as ort

from src.device import DeviceConfig

logger = logging.getLogger(__name__)


@dataclass
class _ModelEntry:
    """Metadata for a registered model."""

    path: str
    vram_mb: int
    session: ort.InferenceSession | None = None


class ModelRegistry:
    """Manages ONNX session lifecycle with VRAM budget enforcement.

    Sessions are lazy-loaded on first ``get()`` call and cached for reuse.
    ``release()`` unloads a session and frees its VRAM budget.

    Args:
        device: Device string passed to ``DeviceConfig`` (default "auto").
        vram_budget_mb: Maximum VRAM in MB available for ML models.

    Example::

        reg = ModelRegistry(device="cuda", vram_budget_mb=3800)
        reg.register("depth_anything", vram_mb=200, path="data/models/depth_v2_small.onnx")
        session = reg.get("depth_anything")  # lazy-loaded
        # ... use session ...
        reg.release("depth_anything")  # free VRAM
    """

    def __init__(self, device: str = "auto", vram_budget_mb: int = 3800) -> None:
        self._device = DeviceConfig(device=device)
        self._vram_budget_mb = vram_budget_mb
        self._entries: dict[str, _ModelEntry] = {}

    @property
    def vram_budget_mb(self) -> int:
        return self._vram_budget_mb

    @property
    def vram_used_mb(self) -> int:
        return sum(e.vram_mb for e in self._entries.values() if e.session is not None)

    def register(self, model_id: str, *, vram_mb: int, path: str) -> None:
        """Register a model without loading it.

        Args:
            model_id: Unique identifier (e.g. "depth_anything").
            vram_mb: Estimated VRAM usage in MB.
            path: Path to ONNX model file.
        """
        if model_id in self._entries:
            logger.warning("Model %s already registered, overwriting", model_id)
        self._entries[model_id] = _ModelEntry(path=path, vram_mb=vram_mb)

    def is_registered(self, model_id: str) -> bool:
        return model_id in self._entries

    def is_loaded(self, model_id: str) -> bool:
        entry = self._entries.get(model_id)
        return entry is not None and entry.session is not None

    def get(self, model_id: str) -> ort.InferenceSession:
        """Get (or lazy-load) an ONNX session.

        Args:
            model_id: Model identifier passed to ``register()``.

        Returns:
            Loaded ``ort.InferenceSession``.

        Raises:
            KeyError: If model_id is not registered.
            RuntimeError: If loading would exceed VRAM budget.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            raise KeyError(f"Model '{model_id}' is not registered. Call register() first.")

        if entry.session is not None:
            return entry.session

        # Check VRAM budget
        if self.vram_used_mb + entry.vram_mb > self._vram_budget_mb:
            raise RuntimeError(
                f"Cannot load '{model_id}' ({entry.vram_mb}MB): "
                f"would exceed VRAM budget ({self.vram_used_mb} + {entry.vram_mb} > {self._vram_budget_mb}MB)"
            )

        logger.info("Loading model '%s' from %s (%dMB, device=%s)", model_id, entry.path, entry.vram_mb, self._device.device)
        session = ort.InferenceSession(entry.path, providers=self._device.onnx_providers)
        entry.session = session
        return session

    def release(self, model_id: str) -> None:
        """Unload a model and free its VRAM budget.

        No-op if model is not registered or not loaded.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.session is None:
            return
        logger.info("Releasing model '%s' (freeing %dMB)", model_id, entry.vram_mb)
        entry.session.release()
        entry.session = None

    def list_models(self) -> list[str]:
        """Return list of registered model IDs."""
        return list(self._entries.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ml/test_model_registry.py -v`
Expected: 12 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml/__init__.py src/ml/model_registry.py tests/ml/__init__.py tests/ml/test_model_registry.py
git commit -m "feat(ml): add ModelRegistry for ONNX session lifecycle management"
```

---

## Task 2: Depth Anything V2

**Files:**
- Create: `src/ml/depth_anything.py`
- Create: `src/visualization/layers/depth_layer.py`
- Create: `tests/ml/test_depth_anything.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_depth_anything.py
"""Tests for Depth Anything V2 wrapper."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestDepthEstimator:
    """Tests for monocular depth estimation."""

    def test_estimate_returns_depth_map(self):
        """estimate() returns (H, W) float32 depth map."""
        from src.ml.depth_anything import DepthEstimator

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        mock_session.run.return_value = [np.random.rand(1, 518, 518).astype(np.float32)]

        est = DepthEstimator.__new__(DepthEstimator)
        est._session = mock_session
        est._input_size = 518

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = est.estimate(frame)

        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32

    def test_estimate_normalizes_to_0_1(self):
        """Depth map values are normalized to [0, 1]."""
        from src.ml.depth_anything import DepthEstimator

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        raw_depth = np.random.rand(1, 518, 518).astype(np.float32) * 10 + 5
        mock_session.run.return_value = [raw_depth]

        est = DepthEstimator.__new__(DepthEstimator)
        est._session = mock_session
        est._input_size = 518

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = est.estimate(frame)

        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_estimate_prepares_input_correctly(self):
        """Input is resized to model size and transposed to NCHW."""
        from src.ml.depth_anything import DepthEstimator

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        mock_session.run.return_value = [np.zeros((1, 518, 518), dtype=np.float32)]

        est = DepthEstimator.__new__(DepthEstimator)
        est._session = mock_session
        est._input_size = 518

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        est.estimate(frame)

        # Check input shape: (1, 3, 518, 518)
        call_args = mock_session.run.call_args
        input_feed = call_args[1].get("image") or call_args[0][1].get("image") if len(call_args[0]) > 1 else None
        # The input should be NCHW with model input size
        assert call_args is not None

    def test_init_from_registry(self):
        """DepthEstimator loads from ModelRegistry."""
        from src.ml.depth_anything import DepthEstimator
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]

        with mock.patch("src.ml.model_registry.ort.InferenceSession", return_value=mock_session):
            est = DepthEstimator(reg)
            assert est._session is mock_session


class TestDepthMapLayer:
    """Tests for depth map visualization layer."""

    def test_render_adds_depth_overlay(self):
        """DepthMapLayer renders color-mapped depth onto frame."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer(opacity=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["depth_map"] = depth

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)
        # Frame should no longer be all-black (depth overlay applied)
        assert not np.all(result == 0)

    def test_render_no_depth_returns_unchanged(self):
        """Layer is a no-op when no depth_map in context."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_depth_anything.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DepthEstimator**

```python
# src/ml/depth_anything.py
"""Depth Anything V2 wrapper for monocular depth estimation.

Uses ONNX Runtime for inference. Input: RGB frame. Output: relative depth map (H, W).

Model: Depth Anything V2 Small (24.8M params, ~200MB VRAM)
Source: https://github.com/DepthAnything/Depth-Anything-V2
ONNX: https://huggingface.co/DepthAnything/Depth-Anything-V2-Small
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "depth_anything"
INPUT_SIZE = 518


class DepthEstimator:
    """Monocular depth estimation via Depth Anything V2.

    Args:
        registry: ModelRegistry with "depth_anything" registered.
            The model must be registered before construction::

                reg.register("depth_anything", vram_mb=200, path="data/models/depth_anything_v2_small.onnx")
                est = DepthEstimator(reg)
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_size = INPUT_SIZE
        # Infer input name from session
        details = self._session.get_input_details()
        self._input_name = details[0]["name"]

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for a single frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Depth map (H, W) float32 normalized to [0, 1].
        """
        h, w = frame.shape[:2]

        # Resize to model input, keep aspect ratio with padding
        img = cv2.resize(frame, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> NCHW, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # (1, 3, H, W)

        # Inference
        output = self._session.run(None, {self._input_name: blob})[0]

        # Output shape: (1, H_out, W_out) -> (H_out, W_out)
        depth = output[0].astype(np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)

        # Resize back to original frame size
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth
```

```python
# src/visualization/layers/depth_layer.py
"""Depth map visualization layer.

Renders monocular depth estimation as a color-mapped overlay (turbo colormap).
Reads depth map from ``LayerContext.custom_data["depth_map"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext


class DepthMapLayer(Layer):
    """Renders depth map as semi-transparent color overlay.

    Args:
        opacity: Blending opacity (0.0 = invisible, 1.0 = full depth map).
        config: Optional LayerConfig.
    """

    def __init__(self, opacity: float = 0.4, config: LayerConfig | None = None) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-1, opacity=opacity))
        self.name = "DepthMap"

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        depth_map = context.custom_data.get("depth_map")
        if depth_map is None:
            return frame

        # Apply turbo colormap: (H, W) float32 -> (H, W, 3) uint8 BGR
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        # Blend with original frame
        alpha = self.opacity
        blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0)
        return blended
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ml/test_depth_anything.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml/depth_anything.py src/visualization/layers/depth_layer.py tests/ml/test_depth_anything.py
git commit -m "feat(ml): add Depth Anything V2 wrapper and depth visualization layer"
```

---

## Task 3: NeuFlowV2 Optical Flow

**Files:**
- Create: `src/ml/optical_flow.py`
- Create: `src/visualization/layers/optical_flow_layer.py`
- Create: `tests/ml/test_optical_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_optical_flow.py
"""Tests for NeuFlowV2 optical flow wrapper."""

from unittest import mock

import numpy as np
import pytest


class TestOpticalFlowEstimator:
    """Tests for dense optical flow estimation."""

    def test_estimate_returns_flow_field(self):
        """estimate() returns (H, W, 2) float32 flow field."""
        from src.ml.optical_flow import OpticalFlowEstimator

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.random.rand(2, 480, 640).astype(np.float32)]

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._session = mock_session

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = est.estimate(frame1, frame2)

        assert flow.shape == (480, 640, 2)
        assert flow.dtype == np.float32

    def test_estimate_frame_size_mismatch_raises(self):
        """estimate() raises ValueError if frames have different sizes."""
        from src.ml.optical_flow import OpticalFlowEstimator

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="same size"):
            est.estimate(frame1, frame2)

    def test_estimate_from_previous(self):
        """estimate_from_previous() caches previous frame."""
        from src.ml.optical_flow import OpticalFlowEstimator

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.zeros((2, 480, 640), dtype=np.float32)]

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._session = mock_session
        est._prev_frame = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = est.estimate_from_previous(frame)

        assert flow is None  # No previous frame yet
        assert est._prev_frame is not None  # Frame cached

        flow = est.estimate_from_previous(frame)
        assert flow.shape == (480, 640, 2)

    def test_reset(self):
        """reset() clears cached previous frame."""
        from src.ml.optical_flow import OpticalFlowEstimator

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._prev_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        est.reset()
        assert est._prev_frame is None


class TestOpticalFlowLayer:
    """Tests for optical flow visualization layer."""

    def test_render_adds_flow_overlay(self):
        """OpticalFlowLayer renders HSV color wheel flow."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.optical_flow_layer import OpticalFlowLayer

        layer = OpticalFlowLayer(opacity=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = np.random.rand(480, 640, 2).astype(np.float32) * 10 - 5

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["flow_field"] = flow

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)

    def test_render_no_flow_returns_unchanged(self):
        """Layer is a no-op when no flow_field in context."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.optical_flow_layer import OpticalFlowLayer

        layer = OpticalFlowLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_optical_flow.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OpticalFlowEstimator**

```python
# src/ml/optical_flow.py
"""NeuFlowV2 optical flow wrapper.

Uses ONNX Runtime for inference. Input: frame pair (BGR). Output: dense flow field (H, W, 2).

Model: NeuFlowV2 (mixed training)
Source: https://github.com/neufieldrobotics/NeuFlow_v2
ONNX: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "optical_flow"


class OpticalFlowEstimator:
    """Dense optical flow estimation via NeuFlowV2.

    Args:
        registry: ModelRegistry with "optical_flow" registered.

    Supports two usage patterns:
    - ``estimate(frame1, frame2)`` — explicit frame pair
    - ``estimate_from_previous(frame)`` — caches previous frame automatically
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._prev_frame: np.ndarray | None = None
        # NeuFlowV2 expects two concatenated images
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Estimate optical flow between two frames.

        Args:
            frame1: BGR image (H, W, 3) uint8.
            frame2: BGR image (H, W, 3) uint8, same size as frame1.

        Returns:
            Flow field (H, W, 2) float32.

        Raises:
            ValueError: If frames have different sizes.
        """
        if frame1.shape[:2] != frame2.shape[:2]:
            raise ValueError(f"Frames must have the same size: {frame1.shape[:2]} vs {frame2.shape[:2]}")

        h, w = frame1.shape[:2]

        # Prepare inputs — NeuFlowV2 expects two separate images
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        inputs = {self._input_names[0]: img1[np.newaxis], self._input_names[1]: img2[np.newaxis]}

        # Inference
        output = self._session.run(None, inputs)[0]

        # Output shape: (2, H, W) -> (H, W, 2)
        flow = output.transpose(1, 2, 0).astype(np.float32)

        # Resize to original frame size if needed
        if flow.shape[:2] != (h, w):
            flow_xy = np.stack([flow[:, :, 0], flow[:, :, 1]], axis=-1)
            flow = cv2.resize(flow_xy, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        return flow

    def estimate_from_previous(self, frame: np.ndarray) -> np.ndarray | None:
        """Estimate flow from previously cached frame.

        On first call, caches the frame and returns None.
        On subsequent calls, estimates flow between previous and current frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Flow field (H, W, 2) float32, or None on first call.
        """
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None

        flow = self.estimate(self._prev_frame, frame)
        self._prev_frame = frame.copy()
        return flow

    def reset(self) -> None:
        """Clear cached previous frame."""
        self._prev_frame = None
```

```python
# src/visualization/layers/optical_flow_layer.py
"""Optical flow visualization layer.

Renders dense optical flow as HSV color wheel overlay.
Reads flow field from ``LayerContext.custom_data["flow_field"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext


class OpticalFlowLayer(Layer):
    """Renders optical flow as HSV color wheel overlay.

    Args:
        opacity: Blending opacity.
        config: Optional LayerConfig.
    """

    def __init__(self, opacity: float = 0.5, config: LayerConfig | None = None) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=0, opacity=opacity))
        self.name = "OpticalFlow"

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        flow = context.custom_data.get("flow_field")
        if flow is None:
            return frame

        # Convert flow to HSV color wheel visualization
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

        # Build HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = (angle / 2).astype(np.uint8)  # Hue: direction
        hsv[:, :, 1] = 255  # Saturation: full
        # Value: normalized magnitude
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        hsv[:, :, 2] = mag_norm.astype(np.uint8)

        # HSV -> BGR
        colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend
        alpha = self.opacity
        blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0)
        return blended
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ml/test_optical_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml/optical_flow.py src/visualization/layers/optical_flow_layer.py tests/ml/test_optical_flow.py
git commit -m "feat(ml): add NeuFlowV2 optical flow wrapper and visualization layer"
```

---

## Task 4: Wire Phase 1 into Pipeline

**Files:**
- Modify: `src/visualization/pipeline.py`
- Modify: `src/web_helpers.py`
- Modify: `src/backend/schemas.py`
- Modify: `src/visualization/layers/__init__.py`

- [ ] **Step 1: Add ML flags to ProcessRequest schema**

In `src/backend/schemas.py`, add boolean fields to `ProcessRequest`:

```python
class ProcessRequest(BaseModel):
    """Request body for POST /api/process."""

    video_path: str
    person_click: PersonClick
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True
    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
```

- [ ] **Step 2: Update VizPipeline to accept ML layers**

In `src/visualization/pipeline.py`, modify `VizPipeline.__post_init__` to build ML layers:

```python
    def __post_init__(self) -> None:
        n = len(self.poses_norm)
        if self.frame_indices is None:
            self.frame_indices = np.arange(n)
        if self.poses_px is None:
            w, h = self.meta.width, self.meta.height
            self.poses_px = self.poses_norm.copy()
            self.poses_px[:, :, 0] *= w
            self.poses_px[:, :, 1] *= h
        self.build_layers()

    def build_layers(self) -> None:
        """Construct visualization layers based on ``self.layer`` level and ML flags."""
        self.layers = []
        if self.layer >= 2:
            self.layers.append(VerticalAxisLayer())

        # ML layers (added after build_layers is called with ML flags)
        # These are appended by process_video_pipeline via add_ml_layers()
```

Add a new method `add_ml_layers`:

```python
    def add_ml_layers(self, ml_layers: list) -> None:
        """Add ML-generated layers to the pipeline."""
        self.layers.extend(ml_layers)
```

- [ ] **Step 3: Update process_video_pipeline for ML inference**

In `src/web_helpers.py`, modify `process_video_pipeline()` to accept and use ML flags:

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
    # ML flags (all default False)
    depth: bool = False,
    optical_flow: bool = False,
    segment: bool = False,
    foot_track: bool = False,
    matting: bool = False,
) -> dict:
```

Inside the function, after `prepare_poses()` and before the render loop, initialize ML models and layers:

```python
    from src.ml.model_registry import ModelRegistry

    # Initialize ML models (lazy-loaded)
    registry = None
    depth_est = None
    flow_est = None
    ml_layers = []

    any_ml = depth or optical_flow or segment or foot_track or matting
    if any_ml:
        registry = ModelRegistry(device="auto")
        if depth:
            registry.register("depth_anything", vram_mb=200, path=_find_model("depth_anything_v2_small.onnx"))
        if optical_flow:
            registry.register("optical_flow", vram_mb=80, path=_find_model("neuflowv2_mixed.onnx"))

        # Load models that will be used
        if depth:
            from src.ml.depth_anything import DepthEstimator
            try:
                depth_est = DepthEstimator(registry)
                from src.visualization.layers.depth_layer import DepthMapLayer
                ml_layers.append(DepthMapLayer(opacity=0.4))
            except Exception as e:
                logger.warning("Failed to load depth model: %s", e)
        if optical_flow:
            from src.ml.optical_flow import OpticalFlowEstimator
            try:
                flow_est = OpticalFlowEstimator(registry)
                from src.visualization.layers.optical_flow_layer import OpticalFlowLayer
                ml_layers.append(OpticalFlowLayer(opacity=0.5))
            except Exception as e:
                logger.warning("Failed to load optical flow model: %s", e)

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
    pipe.add_ml_layers(ml_layers)
```

In the render loop, add per-frame ML inference:

```python
        # Per-frame ML inference
        if depth_est is not None or flow_est is not None:
            _, context = pipe.render_frame(frame, frame_idx, current_pose_idx)
            if depth_est is not None:
                depth_map = depth_est.estimate(frame)
                context.custom_data["depth_map"] = depth_map
            if flow_est is not None:
                flow = flow_est.estimate_from_previous(frame)
                if flow is not None:
                    context.custom_data["flow_field"] = flow
            # Re-render layers with ML data
            if layer >= 1 and current_pose_idx is not None:
                frame = render_layers(frame, ml_layers, context)
```

Add helper function for model path resolution:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _find_model(filename: str) -> str:
    """Find model file in data/models/."""
    path = _PROJECT_ROOT / "data" / "models" / filename
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Model not found: {path}")
```

- [ ] **Step 4: Update process.py route to pass ML flags**

In `src/backend/routes/process.py`, pass new fields from `req` to `process_video_pipeline()`:

```python
                result = process_video_pipeline(
                    video_path=req.video_path,
                    person_click=click,
                    frame_skip=req.frame_skip,
                    layer=req.layer,
                    tracking=req.tracking,
                    blade_3d=False,
                    export=req.export,
                    output_path=str(out_path),
                    progress_cb=progress_cb,
                    depth=req.depth,
                    optical_flow=req.optical_flow,
                    segment=req.segment,
                    foot_track=req.foot_track,
                    matting=req.matting,
                )
```

- [ ] **Step 5: Update layers __init__.py**

In `src/visualization/layers/__init__.py`, add new layer re-exports:

```python
from src.visualization.layers.depth_layer import DepthMapLayer
from src.visualization.layers.optical_flow_layer import OpticalFlowLayer
```

And add to `__all__`:

```python
    "DepthMapLayer",
    "OpticalFlowLayer",
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v --timeout=60 -x`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add src/visualization/pipeline.py src/web_helpers.py src/backend/schemas.py src/backend/routes/process.py src/visualization/layers/__init__.py
git commit -m "feat(ml): wire depth and optical flow into pipeline with backend API"
```

---

## Task 5: Frontend Schema + UploadPage Integration

**Files:**
- Modify: `src/frontend/src/lib/schemas.ts`
- Modify: `src/frontend/src/lib/api.ts`
- Modify: `src/frontend/src/types/index.ts`
- Modify: `src/frontend/src/pages/UploadPage.tsx`
- Modify: `src/frontend/src/pages/AnalyzePage.tsx`

- [ ] **Step 1: Update Zod schema**

In `src/frontend/src/lib/schemas.ts`, add new fields to `ProcessRequestSchema`:

```typescript
export const ProcessRequestSchema = z.object({
  video_path: z.string().min(1),
  person_click: PersonClickSchema,
  frame_skip: z.number().int().positive().default(1),
  layer: z.number().int().min(0).max(3).default(3),
  tracking: z.enum(["auto", "manual"]).default("auto"),
  export: z.boolean().default(true),
  depth: z.boolean().default(false),
  optical_flow: z.boolean().default(false),
  segment: z.boolean().default(false),
  foot_track: z.boolean().default(false),
  matting: z.boolean().default(false),
})
```

- [ ] **Step 2: Update TypeScript interface**

In `src/frontend/src/types/index.ts`, add to `ProcessRequest`:

```typescript
export interface ProcessRequest {
  video_path: string
  person_click: PersonClick
  frame_skip: number
  layer: number
  tracking: string
  export: boolean
  depth: boolean
  optical_flow: boolean
  segment: boolean
  foot_track: boolean
  matting: boolean
}
```

- [ ] **Step 3: Update api.ts**

In `src/frontend/src/lib/api.ts`, add new fields to the `processVideo` request type:

```typescript
export async function processVideo(
  request: {
    video_path: string
    person_click: { x: number; y: number }
    frame_skip: number
    layer: number
    tracking: string
    export: boolean
    depth?: boolean
    optical_flow?: boolean
    segment?: boolean
    foot_track?: boolean
    matting?: boolean
  },
```

- [ ] **Step 4: Add ML checkboxes to UploadPage**

In `src/frontend/src/pages/UploadPage.tsx`:

1. Add state variables for ML options (after existing settings state):

```typescript
  const [enableDepth, setEnableDepth] = useState(false)
  const [enableOpticalFlow, setEnableOpticalFlow] = useState(false)
```

2. Add checkboxes in the Settings card (after the "Экспорт поз + CSV" checkbox):

```tsx
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="depth"
                    checked={enableDepth}
                    onCheckedChange={v => setEnableDepth(v === true)}
                  />
                  <label htmlFor="depth" className="text-sm">
                    Глубина (Depth)
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="optical-flow"
                    checked={enableOpticalFlow}
                    onCheckedChange={v => setEnableOpticalFlow(v === true)}
                  />
                  <label htmlFor="optical-flow" className="text-sm">
                    Оптический поток
                  </label>
                </div>
```

3. Pass new params in `handleAnalyze()`:

```typescript
    const params = new URLSearchParams({
      video_path: detectResult.video_path,
      person_click: `${clickCoord.x},${clickCoord.y}`,
      frame_skip: String(frameSkip),
      layer: String(layer),
      tracking,
      export: String(doExport),
      depth: String(enableDepth),
      optical_flow: String(enableOpticalFlow),
    })
```

- [ ] **Step 5: Parse new params in AnalyzePage**

In `src/frontend/src/pages/AnalyzePage.tsx`, add parsing (after existing param parsing):

```typescript
  const enableDepth = params.get("depth") !== "false"
  const enableOpticalFlow = params.get("optical_flow") !== "false"
```

Add to `processRequest`:

```typescript
  const processRequest = useMemo(
    () => ({
      video_path: videoPath,
      person_click: personClick,
      frame_skip: frameSkip,
      layer: layer,
      tracking: tracking,
      export: doExport,
      depth: enableDepth,
      optical_flow: enableOpticalFlow,
    }),
    [videoPath, personClick, frameSkip, layer, tracking, doExport, enableDepth, enableOpticalFlow],
  )
```

- [ ] **Step 6: Run frontend checks**

Run: `cd src/frontend && bun run check && bun run typecheck`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/frontend/src/lib/schemas.ts src/frontend/src/lib/api.ts src/frontend/src/types/index.ts src/frontend/src/pages/UploadPage.tsx src/frontend/src/pages/AnalyzePage.tsx
git commit -m "feat(frontend): add depth and optical flow toggles to upload settings"
```

---

## Task 6: CLI Flags + Download Script

**Files:**
- Modify: `src/cli.py`
- Create: `scripts/download_ml_models.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add ML flags to CLI visualize command**

In `src/cli.py`, the ML models are used via the web pipeline (`process_video_pipeline`), not the analyze command. The `visualize_with_skeleton.py` script is the CLI entry point that calls `process_video_pipeline`. Add flags there instead.

In `scripts/visualize_with_skeleton.py`, add argparse flags:

```python
    parser.add_argument("--depth", action="store_true", help="Enable depth estimation (Depth Anything V2)")
    parser.add_argument("--optical-flow", action="store_true", help="Enable optical flow (NeuFlowV2)")
```

Pass them through to `process_video_pipeline()`.

- [ ] **Step 2: Create download script**

```python
#!/usr/bin/env python3
"""Download ML model weights for optional pipeline features.

Usage:
    uv run python scripts/download_ml_models.py --all
    uv run python scripts/download_ml_models.py --model depth_anything
    uv run python scripts/download_ml_models.py --list
"""

import argparse
from pathlib import Path

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "depth_anything": {
        "url": "https://huggingface.co/DepthAnything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.onnx",
        "filename": "depth_anything_v2_small.onnx",
        "size_mb": "~100MB",
        "description": "Monocular depth estimation (Depth Anything V2 Small)",
    },
    "optical_flow": {
        "url": "https://github.com/neufieldrobotics/NeuFlow_v2/releases/download/v2.0/neuflowv2_mixed.onnx",
        "filename": "neuflowv2_mixed.onnx",
        "size_mb": "~40MB",
        "description": "Dense optical flow (NeuFlowV2 mixed)",
    },
    "sam2_tiny": {
        "url": "https://huggingface.co/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.onnx",
        "filename": "sam2_tiny.onnx",
        "size_mb": "~160MB",
        "description": "Image segmentation (SAM 2 Tiny)",
    },
    "foot_tracker": {
        "url": "https://huggingface.co/qualcomm/Person-Foot-Detection/resolve/main/foot_detector.onnx",
        "filename": "foot_tracker.onnx",
        "size_mb": "~10MB",
        "description": "Person and foot detection (FootTrackNet)",
    },
    "video_matting": {
        "url": "https://huggingface.co/PINTO0309/RobustVideoMatting/resolve/main/rvm_mobilenetv3_fp32.onnx",
        "filename": "rvm_mobilenetv3.onnx",
        "size_mb": "~20MB",
        "description": "Video background removal (RobustVideoMatting MobileNetV3)",
    },
    "lama": {
        "url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "filename": "lama_fp32.onnx",
        "size_mb": "~174MB",
        "description": "Image inpainting (LAMA Dilated)",
    },
}


def download_model(model_id: str) -> None:
    """Download a single model."""
    import urllib.request

    info = MODELS[model_id]
    dest = MODELS_DIR / info["filename"]

    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {info['description']} ({info['size_mb']})...")
    print(f"  URL: {info['url']}")
    urllib.request.urlretrieve(info["url"], dest)
    print(f"  Saved: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ML model weights")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), help="Download specific model")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for mid, info in MODELS.items():
            print(f"  {mid}: {info['description']} ({info['size_mb']})")
        return

    if args.all:
        print("Downloading all models...")
        for model_id in MODELS:
            download_model(model_id)
        print("Done!")
    elif args.model:
        download_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add huggingface_hub dependency**

Run: `uv add --dev huggingface_hub`

Note: The download script uses `urllib.request` (stdlib), not huggingface_hub, to avoid heavy dependencies. But `huggingface_hub` is useful for authenticated downloads and model discovery. Add as dev dependency for future use.

- [ ] **Step 4: Update data/models/.gitignore**

Ensure new ONNX files are gitignored:

```
# ML model weights (download via scripts/download_ml_models.py)
*.onnx
*.onnx.data
```

- [ ] **Step 5: Verify**

Run: `uv run python scripts/download_ml_models.py --list`
Expected: Lists all 6 models with descriptions

- [ ] **Step 6: Commit**

```bash
git add scripts/download_ml_models.py data/models/.gitignore pyproject.toml uv.lock
git commit -m "feat(ml): add model download script and CLI flags for depth/optical flow"
```

---

## Task 7: SAM2 Segmentation

**Files:**
- Create: `src/ml/segment_anything.py`
- Create: `src/visualization/layers/segmentation_layer.py`
- Create: `tests/ml/test_segment_anything.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_segment_anything.py
"""Tests for SAM2 segmentation wrapper."""

from unittest import mock

import numpy as np
import pytest


class TestSegmentAnything:
    """Tests for SAM2 image segmentation."""

    def test_segment_returns_mask(self):
        """segment() returns (H, W) bool mask."""
        from src.ml.segment_anything import SegmentAnything

        mock_session = mock.MagicMock()
        # SAM2 returns masks and scores
        mock_session.run.return_value = [
            np.ones((1, 1, 256, 256), dtype=np.float32),  # masks
            np.array([[0.95]]),  # scores (iou_predictions)
        ]

        est = SegmentAnything.__new__(SegmentAnything)
        est._session = mock_session
        est._input_size = 1024

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(320, 240))

        assert mask.shape == (480, 640)
        assert mask.dtype == bool

    def test_segment_with_no_point_returns_empty(self):
        """segment() with point=None returns None (no prompt)."""
        from src.ml.segment_anything import SegmentAnything

        est = SegmentAnything.__new__(SegmentAnything)
        result = est.segment(np.zeros((480, 640, 3), dtype=np.uint8), point=None)
        assert result is None

    def test_segment_resize_back_to_original(self):
        """Mask is resized to original frame size."""
        from src.ml.segment_anything import SegmentAnything

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [
            np.ones((1, 1, 256, 256), dtype=np.float32),
            np.array([[0.95]]),
        ]

        est = SegmentAnything.__new__(SegmentAnything)
        est._session = mock_session
        est._input_size = 1024

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(640, 360))

        assert mask.shape == (720, 1280)


class TestSegmentationLayer:
    """Tests for segmentation mask visualization."""

    def test_render_adds_mask_overlay(self):
        """SegmentationMaskLayer renders semi-transparent mask."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer(opacity=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:400, 200:500] = True

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["seg_mask"] = mask

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)
        # Masked region should have color
        assert not np.all(result == 0)

    def test_render_no_mask_returns_unchanged(self):
        """No-op when no seg_mask in context."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_segment_anything.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SegmentAnything**

```python
# src/ml/segment_anything.py
"""SAM 2 wrapper for image segmentation.

Uses ONNX Runtime for inference. Input: RGB image + point prompt. Output: binary mask.

Model: SAM 2 Tiny (38.9M params, ~200MB VRAM)
Source: https://github.com/facebookresearch/sam2
ONNX: https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "segment_anything"
INPUT_SIZE = 1024


class SegmentAnything:
    """Image segmentation via SAM 2.

    Args:
        registry: ModelRegistry with "segment_anything" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_size = INPUT_SIZE
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]

    def segment(
        self,
        frame: np.ndarray,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        """Segment the image using a point or box prompt.

        Args:
            frame: BGR image (H, W, 3) uint8.
            point: (x, y) pixel coordinate as prompt, or None.
            box: (x1, y1, x2, y2) pixel box as prompt, or None.

        Returns:
            Binary mask (H, W) bool, or None if no prompt provided.
        """
        if point is None and box is None:
            return None

        h, w = frame.shape[:2]

        # Prepare image
        img = cv2.resize(frame, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Prepare point prompt
        point_coords = np.array([], dtype=np.float32).reshape(0, 2)
        point_labels = np.array([], dtype=np.float32)
        if point is not None:
            # Scale point to model input size
            sx = self._input_size / w
            sy = self._input_size / h
            point_coords = np.array([[point[0] * sx, point[1] * sy]], dtype=np.float32)
            point_labels = np.array([1.0], dtype=np.float32)  # foreground

        # Build inputs (SAM2 ONNX format — adapt to actual ONNX export)
        inputs = {}
        for i, name in enumerate(self._input_names):
            if i == 0:
                inputs[name] = img_tensor
            elif "point_coords" in name.lower():
                inputs[name] = point_coords
            elif "point_labels" in name.lower():
                inputs[name] = point_labels

        try:
            outputs = self._session.run(None, inputs)
            # Find mask output (typically first or second output)
            masks = None
            for out in outputs:
                if isinstance(out, np.ndarray) and out.ndim == 4:
                    masks = out
                    break

            if masks is None:
                return None

            # Take best mask (highest IoU prediction)
            mask = masks[0, 0]  # (H_in, W_in)
            mask = mask > 0.0  # threshold

            # Resize to original frame size
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            return mask
        except Exception as e:
            logger.warning("SAM2 inference failed: %s", e)
            return None
```

```python
# src/visualization/layers/segmentation_layer.py
"""Segmentation mask visualization layer.

Renders a semi-transparent colored overlay over the segmented region.
Reads mask from ``LayerContext.custom_data["seg_mask"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext


class SegmentationMaskLayer(Layer):
    """Renders segmentation mask as semi-transparent overlay.

    Args:
        color: BGR color for the mask overlay (default: cyan).
        opacity: Blending opacity for the mask region.
        config: Optional LayerConfig.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (255, 255, 0),  # cyan BGR
        opacity: float = 0.3,
        config: LayerConfig | None = None,
    ) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-2, opacity=opacity))
        self.name = "Segmentation"
        self._color = color

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        mask = context.custom_data.get("seg_mask")
        if mask is None:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Apply color to masked region
        mask_uint8 = mask.astype(np.uint8) * 255
        colored_mask = np.full_like(frame, self._color)
        overlay[mask_uint8 > 0] = colored_mask[mask_uint8 > 0]

        # Blend
        alpha = self.opacity
        blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        # Draw contour around mask for definition
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, self._color, 2)

        return blended
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_segment_anything.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ml/segment_anything.py src/visualization/layers/segmentation_layer.py tests/ml/test_segment_anything.py
git commit -m "feat(ml): add SAM2 segmentation wrapper and mask visualization layer"
```

---

## Task 8: Wire Phase 2 (SAM2 + FootTrackNet) — Schema + Frontend

**Files:**
- Modify: `src/backend/schemas.py` (already has fields from Task 4)
- Modify: `src/web_helpers.py` (add SAM2 + foot tracking logic)
- Modify: `src/frontend/src/pages/UploadPage.tsx` (add checkboxes)
- Modify: `src/frontend/src/pages/AnalyzePage.tsx` (pass params)
- Create: `src/ml/foot_tracker.py`

- [ ] **Step 1: Implement FootTrackNet**

```python
# src/ml/foot_tracker.py
"""FootTrackNet wrapper for specialized foot detection.

Uses ONNX Runtime. Input: RGB image. Output: person + foot bounding boxes.

Model: FootTrackNet (2.53M params, ~30MB VRAM)
Source: Qualcomm AI Hub
ONNX: https://huggingface.co/qualcomm/Person-Foot-Detection
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "foot_tracker"


class FootTracker:
    """Person and foot detection via FootTrackNet.

    Args:
        registry: ModelRegistry with "foot_tracker" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        details = self._session.get_input_details()
        self._input_name = details[0]["name"]
        self._input_size = (640, 480)  # FootTrackNet default

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect persons and feet in a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            List of dicts with keys: ``bbox`` (x1,y1,x2,y2), ``class_id`` (0=person, 1=foot),
            ``confidence`` (float).
        """
        h, w = frame.shape[:2]
        img = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)

        output = self._session.run(None, {self._input_name: blob})[0]

        detections = []
        for det in output:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf > 0.3:
                    # Scale bbox back to original frame size
                    sx, sy = w / self._input_size[0], h / self._input_size[1]
                    detections.append({
                        "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                        "class_id": int(cls),
                        "confidence": float(conf),
                    })

        return detections
```

- [ ] **Step 2: Add SAM2 logic to process_video_pipeline**

In `src/web_helpers.py`, extend the ML initialization block:

```python
        if segment:
            registry.register("segment_anything", vram_mb=200, path=_find_model("sam2_tiny.onnx"))
        if foot_track:
            registry.register("foot_tracker", vram_mb=30, path=_find_model("foot_tracker.onnx"))

        # ... existing depth + optical flow loading ...

        if segment:
            from src.ml.segment_anything import SegmentAnything
            try:
                seg_est = SegmentAnything(registry)
                from src.visualization.layers.segmentation_layer import SegmentationMaskLayer
                ml_layers.append(SegmentationMaskLayer(opacity=0.3))
            except Exception as e:
                logger.warning("Failed to load segmentation model: %s", e)
```

In the render loop, add SAM2 segmentation using mid-hip as prompt:

```python
            if seg_est is not None and current_pose_idx is not None:
                mid_hip = prepared.poses_px[current_pose_idx, 11, :2]  # H3.6M mid-hip index
                if not np.any(np.isnan(mid_hip)):
                    mask = seg_est.segment(frame, point=(int(mid_hip[0]), int(mid_hip[1])))
                    if mask is not None:
                        context.custom_data["seg_mask"] = mask
```

- [ ] **Step 3: Add frontend checkboxes**

In `src/frontend/src/pages/UploadPage.tsx`, add after existing ML checkboxes:

```tsx
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="segment"
                    checked={enableSegment}
                    onCheckedChange={v => setEnableSegment(v === true)}
                  />
                  <label htmlFor="segment" className="text-sm">
                    Сегментация (SAM2)
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="foot-track"
                    checked={enableFootTrack}
                    onCheckedChange={v => setEnableFootTrack(v === true)}
                  />
                  <label htmlFor="foot-track" className="text-sm">
                    Трекинг стоп
                  </label>
                </div>
```

Add state variables and pass through params the same way as Task 5.

- [ ] **Step 4: Run checks**

Run: `uv run pytest tests/ml/ -v && cd src/frontend && bun run check && bun run typecheck`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/ml/foot_tracker.py src/web_helpers.py src/frontend/src/pages/UploadPage.tsx src/frontend/src/pages/AnalyzePage.tsx
git commit -m "feat(ml): add SAM2 segmentation and FootTrackNet with frontend toggles"
```

---

## Task 9: Video Matting + LAMA (Phase 3)

**Files:**
- Create: `src/ml/video_matting.py`
- Create: `src/ml/inpainting.py`
- Create: `src/visualization/layers/matting_layer.py`

- [ ] **Step 1: Implement RobustVideoMatting**

```python
# src/ml/video_matting.py
"""RobustVideoMatting wrapper for video background removal.

Uses ONNX Runtime. Input: RGB frame + optional mask. Output: alpha matte.

Model: RobustVideoMatting MobileNetV3 (~40MB VRAM)
Source: https://github.com/PeterL1n/RobustVideoMatting
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "video_matting"


class VideoMatting:
    """Video background removal via RobustVideoMatting.

    Args:
        registry: ModelRegistry with "video_matting" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]
        self._r1 = None  # Recurrent state frame 1
        self._r2 = None  # Recurrent state frame 2
        self._downsample_ratio = 0.25

    def matting(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Generate alpha matte for a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            mask: Optional binary mask (H, W) to guide matting.

        Returns:
            Alpha matte (H, W) float32 in [0, 1].
        """
        h, w = frame.shape[:2]
        # RVM expects RGB
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = src[np.newaxis]  # (1, H, W, 3)

        inputs = {}
        for i, name in enumerate(self._input_names):
            if "src" in name.lower() and "r1" not in name:
                inputs[name] = src
            elif "r1" in name.lower():
                inputs[name] = self._r1 if self._r1 is not None else np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
            elif "r2" in name.lower():
                inputs[name] = self._r2 if self._r2 is not None else np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
            elif "downsample" in name.lower():
                inputs[name] = np.array([self._downsample_ratio], dtype=np.float32)

        outputs = self._session.run(None, inputs)

        # Update recurrent states
        r1_idx, r2_idx = None, None
        for i, name in enumerate(self._input_names):
            if "r1" in name.lower():
                r1_idx = i
            elif "r2" in name.lower():
                r2_idx = i
        # Outputs typically include fgr, pha, r1, r2
        for out in outputs:
            if out.shape == (1, 1, h // 4, w // 4):
                if self._r1 is None:
                    self._r1 = out
                elif self._r2 is None:
                    self._r2 = out

        # Find alpha output
        alpha = np.ones((h, w), dtype=np.float32)
        for out in outputs:
            if out.ndim == 4 and out.shape[1] == 1:
                a = out[0, 0]  # (H, W)
                if a.shape[0] != h or a.shape[1] != w:
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.clip(a, 0, 1)
                break

        return alpha

    def reset(self) -> None:
        """Reset recurrent states (call when starting a new video)."""
        self._r1 = None
        self._r2 = None
```

- [ ] **Step 2: Implement LAMA inpainting**

```python
# src/ml/inpainting.py
"""LAMA (Large Mask Inpainting) wrapper.

Uses ONNX Runtime. Input: RGB image + mask. Output: inpainted RGB image.

Model: LAMA Dilated (45.6M params, ~300MB VRAM)
Source: https://github.com/advimman/lama
ONNX: https://huggingface.co/Carve/LaMa-ONNX
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "lama"
INPUT_SIZE = 512


class ImageInpainter:
    """Image inpainting via LAMA.

    Args:
        registry: ModelRegistry with "lama" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked regions of a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            mask: Binary mask (H, W) bool, True = region to inpaint.

        Returns:
            Inpainted BGR image (H, W, 3) uint8.
        """
        h, w = frame.shape[:2]

        # Resize to model input
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(mask.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        # Prepare inputs
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = img_rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        mask_tensor = (msk > 0).astype(np.float32)[np.newaxis, np.newaxis]  # (1, 1, H, W)

        inputs = {}
        for i, name in enumerate(self._input_names):
            if "image" in name.lower():
                inputs[name] = img_tensor
            elif "mask" in name.lower():
                inputs[name] = mask_tensor

        output = self._session.run(None, inputs)[0]

        # Convert output to BGR uint8
        result = output[0].transpose(1, 2, 0)  # (H, W, 3)
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Resize back to original
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        # Composite: keep original where mask is False, use inpainted where True
        mask_3ch = np.stack([mask] * 3, axis=-1)
        final = np.where(mask_3ch, result, frame)

        return final
```

- [ ] **Step 3: Implement MattingLayer**

```python
# src/visualization/layers/matting_layer.py
"""Video matting visualization layer.

Applies alpha matte to blur/darken background while keeping foreground sharp.
Reads alpha from ``LayerContext.custom_data["alpha_matte"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext


class MattingLayer(Layer):
    """Applies video matting effect: blur background, keep foreground.

    Args:
        blur_strength: Gaussian blur kernel size for background (must be odd).
        opacity: Effect opacity (0.0 = no effect, 1.0 = full matting).
        config: Optional LayerConfig.
    """

    def __init__(
        self,
        blur_strength: int = 21,
        opacity: float = 1.0,
        config: LayerConfig | None = None,
    ) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-3, opacity=opacity))
        self.name = "VideoMatting"
        self._blur = blur_strength

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        alpha = context.custom_data.get("alpha_matte")
        if alpha is None:
            return frame

        h, w = frame.shape[:2]
        alpha_3ch = np.stack([alpha] * 3, axis=-1)  # (H, W, 3)

        # Blur background
        blurred = cv2.GaussianBlur(frame, (self._blur, self._blur), 0)

        # Composite: foreground * alpha + blurred * (1 - alpha)
        alpha_expanded = alpha_3ch.astype(np.float32) / 255.0
        result = (frame.astype(np.float32) * alpha_expanded +
                  blurred.astype(np.float32) * (1.0 - alpha_expanded))
        return result.astype(np.uint8)
```

- [ ] **Step 4: Wire into process_video_pipeline and frontend**

Same pattern as Tasks 4 and 8: add `matting` flag handling in `process_video_pipeline()`, add checkbox in `UploadPage.tsx`, pass param in `AnalyzePage.tsx`.

- [ ] **Step 5: Commit**

```bash
git add src/ml/video_matting.py src/ml/inpainting.py src/visualization/layers/matting_layer.py src/web_helpers.py src/frontend/
git commit -m "feat(ml): add video matting and LAMA inpainting with frontend toggles"
```

---

## Task 10: Final Integration Tests

**Files:**
- Create: `tests/ml/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/ml/test_integration.py
"""Integration tests for ML pipeline (mocked ONNX sessions)."""

from pathlib import Path
from unittest import mock

import numpy as np


class TestMLPipelineIntegration:
    """Test that ML models integrate correctly with the visualization pipeline."""

    def test_registry_with_all_models_registered(self):
        """All 6 models can be registered within VRAM budget."""
        from src.ml.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu", vram_budget_mb=1000)
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")
        reg.register("optical_flow", vram_mb=80, path="/tmp/flow.onnx")
        reg.register("segment_anything", vram_mb=200, path="/tmp/sam.onnx")
        reg.register("foot_tracker", vram_mb=30, path="/tmp/foot.onnx")
        reg.register("video_matting", vram_mb=40, path="/tmp/rvm.onnx")
        reg.register("lama", vram_mb=300, path="/tmp/lama.onnx")

        assert reg.vram_used_mb == 0  # Nothing loaded yet
        assert reg.list_models() == [
            "depth_anything", "optical_flow", "segment_anything",
            "foot_tracker", "video_matting", "lama",
        ]

    def test_depth_layer_context_flow(self):
        """Depth map flows from estimator through LayerContext to layer."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer(opacity=0.5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.linspace(0, 1, 100 * 100).reshape(100, 100).astype(np.float32)

        ctx = LayerContext(frame_width=100, frame_height=100)
        ctx.custom_data["depth_map"] = depth

        result = layer.render(frame, ctx)
        assert result.shape == (100, 100, 3)
        assert not np.all(result == 0)

    def test_multiple_layers_compose(self):
        """Depth + flow + segmentation layers compose correctly."""
        from src.visualization.layers.base import LayerContext
        from src.visualization.layers.depth_layer import DepthMapLayer
        from src.visualization.layers.optical_flow_layer import OpticalFlowLayer
        from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layers = [
            DepthMapLayer(opacity=0.3),
            OpticalFlowLayer(opacity=0.4),
            SegmentationMaskLayer(opacity=0.2),
        ]

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=100, frame_height=100)
        ctx.custom_data["depth_map"] = np.random.rand(100, 100).astype(np.float32)
        ctx.custom_data["flow_field"] = np.random.rand(100, 100, 2).astype(np.float32) * 2 - 1
        ctx.custom_data["seg_mask"] = np.zeros((100, 100), dtype=bool)
        ctx.custom_data["seg_mask"][20:80, 20:80] = True

        for layer in layers:
            frame = layer.render(frame, ctx)

        assert frame.shape == (100, 100, 3)

    def test_process_request_schema_accepts_ml_flags(self):
        """ProcessRequest schema accepts ML boolean flags."""
        from src.backend.schemas import ProcessRequest

        req = ProcessRequest(
            video_path="/tmp/test.mp4",
            person_click={"x": 100, "y": 200},
            depth=True,
            optical_flow=True,
            segment=True,
            foot_track=True,
            matting=True,
        )
        assert req.depth is True
        assert req.optical_flow is True
        assert req.segment is True
        assert req.foot_track is True
        assert req.matting is True

    def test_process_request_defaults_ml_flags_false(self):
        """ML flags default to False."""
        from src.backend.schemas import ProcessRequest

        req = ProcessRequest(
            video_path="/tmp/test.mp4",
            person_click={"x": 100, "y": 200},
        )
        assert req.depth is False
        assert req.optical_flow is False
        assert req.segment is False
        assert req.foot_track is False
        assert req.matting is False
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ml/ -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All existing + new tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/ml/test_integration.py
git commit -m "test(ml): add integration tests for ML pipeline composition"
```

---

## Verification

After all tasks:

```bash
# 1. List available models
uv run python scripts/download_ml_models.py --list

# 2. Download all models (optional — tests work with mocks)
uv run python scripts/download_ml_models.py --all

# 3. Run ML tests
uv run pytest tests/ml/ -v

# 4. Run full test suite
uv run pytest tests/ -v --timeout=60

# 5. Frontend checks
cd src/frontend && bun run check && bun run typecheck

# 6. End-to-end: start backend + frontend, upload video, enable depth + optical flow
cd src/frontend && bun run dev &
uv run uvicorn src.backend.main:app --reload --port 8000
# Open http://localhost:5173, upload video, enable checkboxes, analyze
```
