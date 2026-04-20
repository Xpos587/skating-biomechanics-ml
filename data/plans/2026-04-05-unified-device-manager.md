# Unified Device Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single `DeviceConfig` module that provides authoritative GPU/CPU device resolution, and propagate it to every component so `device="cuda"` is the default everywhere (with graceful CPU fallback).

**Architecture:** A lightweight `DeviceConfig` dataclass in `src/device.py` resolves the target device once at startup. Every extractor, pipeline, script, and CLI reads from it instead of accepting ad-hoc `device` strings. The module provides `resolve_device()` for one-line resolution and `get_onnx_providers()` for ONNX Runtime compatibility.

**Tech Stack:** stdlib dataclass, `torch.cuda.is_available()` (optional), `onnxruntime` (optional)

---

## Current State (Why This Matters)

| Component | Current default | Problem |
|-----------|----------------|---------|
| `RTMPoseExtractor` | `"cpu"` | GPU never used by default |
| `ONNXPoseExtractor` | `"cpu"` | GPU never used by default |
| `AthletePose3DExtractor` | `"auto"` | Inconsistent with others |
| `TCPFormerExtractor` | `"auto"` | Inconsistent with others |
| `H36MExtractor` | `"0"` | Magic string for GPU index |
| `ComparisonConfig` | `"0"` | Magic string |
| `AnalysisPipeline` | `use_gpu=True` | Boolean, not propagated to extractors |
| `gradio_app.py` | `"cuda"` hardcoded | Duplicated fallback logic |
| `gradio_helpers.py` | `"cuda"` hardcoded | Duplicated fallback logic |
| `compare_models.py` | `"cuda"` hardcoded | No fallback |
| `SkeletalIdentityExtractor` | `"auto"` | Inconsistent |

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/device.py` | **CREATE** | `DeviceConfig` dataclass + `resolve_device()` + `get_onnx_providers()` |
| `src/pipeline.py` | MODIFY | Use `DeviceConfig`, propagate to all lazy-loaded components |
| `src/pose_estimation/rtmlib_extractor.py` | MODIFY | Accept `DeviceConfig` or string, default to `DeviceConfig.default()` |
| `src/pose_3d/onnx_extractor.py` | MODIFY | Use `get_onnx_providers()` from DeviceConfig |
| `src/pose_3d/athletepose_extractor.py` | MODIFY | Use `DeviceConfig` for PyTorch device resolution |
| `src/pose_3d/tcpformer_extractor.py` | MODIFY | Use `DeviceConfig` for PyTorch device resolution |
| `src/pose_estimation/h36m_extractor.py` | MODIFY | Accept `DeviceConfig` or string |
| `src/visualization/comparison.py` | MODIFY | Use `DeviceConfig` instead of `"0"` |
| `src/tracking/skeletal_identity.py` | MODIFY | Use `DeviceConfig` |
| `src/gradio_helpers.py` | MODIFY | Use `DeviceConfig`, remove duplicated fallback |
| `scripts/gradio_app.py` | MODIFY | Use `DeviceConfig`, remove duplicated fallback |
| `scripts/compare_models.py` | MODIFY | Use `DeviceConfig` |
| `src/cli.py` | MODIFY | Pass `DeviceConfig` to pipeline |
| `tests/test_device.py` | **CREATE** | Unit tests for `DeviceConfig` |
| `src/__init__.py` | MODIFY | Re-export `DeviceConfig` |

---

## Task 1: Create `src/device.py` with tests

**Files:**
- Create: `src/device.py`
- Create: `tests/test_device.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_device.py
"""Tests for unified device configuration."""

import os
from unittest import mock

import pytest


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_default_is_cuda(self):
        """Default DeviceConfig should resolve to cuda when available."""
        from src.device import DeviceConfig

        cfg = DeviceConfig()
        # On systems with CUDA, default should be cuda
        # On systems without, should fall back to cpu
        assert cfg.device in ("cuda", "cpu")

    def test_explicit_cpu(self):
        """Explicit CPU request always returns cpu."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert cfg.device == "cpu"
        assert cfg.is_cpu

    def test_explicit_cuda_fallback(self):
        """Explicit CUDA request falls back to cpu when unavailable."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            cfg = DeviceConfig(device="cuda")
            assert cfg.device == "cpu"

    def test_auto_with_cuda_available(self):
        """Auto resolves to cuda when torch.cuda.is_available()."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cuda"
            assert cfg.is_cuda

    def test_auto_without_cuda(self):
        """Auto resolves to cpu when CUDA unavailable."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cpu"
            assert cfg.is_cpu

    def test_gpu_index_resolved(self):
        """GPU index '0' resolves to 'cuda'."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            cfg = DeviceConfig(device="0")
            assert cfg.device == "cuda"

    def test_gpu_index_fallback(self):
        """GPU index '0' falls back to cpu when CUDA unavailable."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            cfg = DeviceConfig(device="0")
            assert cfg.device == "cpu"

    def test_onnx_providers_cpu(self):
        """ONNX providers for CPU-only."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert cfg.onnx_providers == ["CPUExecutionProvider"]

    def test_onnx_providers_cuda(self):
        """ONNX providers for CUDA."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            cfg = DeviceConfig(device="cuda")
            assert cfg.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_torch_device_cuda(self):
        """torch_device property returns correct torch device."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = mock.Mock()
            cfg = DeviceConfig(device="cuda")
            _ = cfg.torch_device
            mock_torch.device.assert_called_with("cuda")

    def test_torch_device_cpu(self):
        """torch_device property returns cpu device."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        # Should work even without torch installed
        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.device.return_value = mock.Mock()
            _ = cfg.torch_device
            mock_torch.device.assert_called_with("cpu")

    def test_default_class_method(self):
        """DeviceConfig.default() returns auto-resolved config."""
        from src.device import DeviceConfig

        cfg = DeviceConfig.default()
        assert isinstance(cfg, DeviceConfig)
        assert cfg.device in ("cuda", "cpu")

    def test_from_str_cpu(self):
        """DeviceConfig.from_str('cpu') creates CPU config."""
        from src.device import DeviceConfig

        cfg = DeviceConfig.from_str("cpu")
        assert cfg.device == "cpu"

    def test_from_str_auto(self):
        """DeviceConfig.from_str('auto') creates auto-resolved config."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            cfg = DeviceConfig.from_str("auto")
            assert cfg.device == "cuda"

    def test_from_str_gpu_index(self):
        """DeviceConfig.from_str('0') resolves GPU index."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            cfg = DeviceConfig.from_str("0")
            assert cfg.device == "cuda"

    def test_repr(self):
        """repr shows device string."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert "cpu" in repr(cfg)

    def test_equality(self):
        """Two configs with same device are equal."""
        from src.device import DeviceConfig

        a = DeviceConfig(device="cpu")
        b = DeviceConfig(device="cpu")
        assert a == b

    def test_inequality(self):
        """Two configs with different devices are not equal."""
        from src.device import DeviceConfig

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            a = DeviceConfig(device="cpu")
            b = DeviceConfig(device="cuda")
            assert a != b

    def test_no_torch_falls_back_to_cpu(self):
        """When torch is not importable, auto falls back to cpu."""
        from src.device import DeviceConfig

        with mock.patch.dict("sys.modules", {"torch": None}):
            # Force reimport to trigger the torch-unavailable path
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cpu"

    def test_env_override_skating_device(self):
        """SKATING_DEVICE env var overrides default."""
        from src.device import DeviceConfig

        with mock.patch.dict(os.environ, {"SKATING_DEVICE": "cpu"}):
            cfg = DeviceConfig()
            assert cfg.device == "cpu"


class TestResolveDevice:
    """Tests for resolve_device convenience function."""

    def test_resolve_cpu(self):
        """resolve_device('cpu') returns 'cpu'."""
        from src.device import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_resolve_auto_with_cuda(self):
        """resolve_device('auto') returns 'cuda' when available."""
        from src.device import resolve_device

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            assert resolve_device("auto") == "cuda"

    def test_resolve_auto_without_cuda(self):
        """resolve_device('auto') returns 'cpu' when unavailable."""
        from src.device import resolve_device

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            assert resolve_device("auto") == "cpu"

    def test_resolve_gpu_index(self):
        """resolve_device('0') returns 'cuda' when available."""
        from src.device import resolve_device

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            assert resolve_device("0") == "cuda"


class TestGetOnnxProviders:
    """Tests for get_onnx_providers convenience function."""

    def test_cpu_providers(self):
        """CPU providers list."""
        from src.device import get_onnx_providers

        assert get_onnx_providers("cpu") == ["CPUExecutionProvider"]

    def test_cuda_providers(self):
        """CUDA providers list with CPU fallback."""
        from src.device import get_onnx_providers

        assert get_onnx_providers("cuda") == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_device.py -v --no-header 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.device'`

- [ ] **Step 3: Write `src/device.py` implementation**

```python
# src/device.py
"""Unified device configuration for GPU/CPU management.

Single source of truth for device resolution across all pipeline components.
Default: CUDA when available, CPU otherwise.

Usage:
    from src.device import DeviceConfig

    # Simple — auto-detect best device
    cfg = DeviceConfig.default()

    # Explicit
    cfg = DeviceConfig(device="cpu")

    # Pass to components
    extractor = RTMPoseExtractor(device=cfg.device)
    session = ort.InferenceSession(path, providers=cfg.onnx_providers)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

DeviceName = Literal["cuda", "cpu"]

# Try importing torch — graceful fallback when not installed
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _cuda_available() -> bool:
    """Check if CUDA is available (works even without torch installed)."""
    if not _HAS_TORCH:
        return False
    try:
        return torch.cuda.is_available()  # type: ignore[union-attr]
    except Exception:
        return False


def _resolve_device_name(device: str) -> DeviceName:
    """Normalize device string to 'cuda' or 'cpu'.

    Accepts: "cuda", "cpu", "auto", "0", "1", etc.
    GPU index strings ("0", "1") resolve to "cuda" when available.
    "auto" resolves to "cuda" when available, else "cpu".
    """
    # Environment variable override takes priority
    env = os.environ.get("SKATING_DEVICE", "").lower().strip()
    if env:
        if env == "cpu":
            return "cpu"
        if env == "cuda":
            if _cuda_available():
                return "cuda"
            logger.warning("SKATING_DEVICE=cuda but CUDA unavailable, falling back to CPU")
            return "cpu"

    device = device.lower().strip()

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if _cuda_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    if device == "auto":
        return "cuda" if _cuda_available() else "cpu"
    # GPU index ("0", "1", etc.)
    if device.isdigit():
        if _cuda_available():
            return "cuda"
        logger.warning(f"GPU index '{device}' requested but CUDA unavailable, falling back to CPU")
        return "cpu"

    logger.warning(f"Unknown device '{device}', falling back to CPU")
    return "cpu"


@dataclass(frozen=True)
class DeviceConfig:
    """Immutable device configuration.

    Resolves device once at creation time. Use DeviceConfig.default()
    for auto-detection or DeviceConfig(device="cpu") for explicit control.

    Attributes:
        device: Resolved device name ("cuda" or "cpu").
    """

    device: DeviceName = "cuda"

    def __init__(self, device: str = "auto") -> None:
        """Create device config.

        Args:
            device: "cuda", "cpu", "auto", or GPU index ("0", "1").
                Defaults to "auto" (CUDA when available).
        """
        resolved = _resolve_device_name(device)
        object.__setattr__(self, "device", resolved)

    @classmethod
    def default(cls) -> DeviceConfig:
        """Create config with auto-detected device (GPU preferred)."""
        return cls(device="auto")

    @classmethod
    def from_str(cls, s: str) -> DeviceConfig:
        """Alias for constructor — accepts string device spec."""
        return cls(device=s)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def onnx_providers(self) -> list[str]:
        """ONNX Runtime execution providers for this device."""
        if self.is_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def torch_device(self):
        """torch.device object for PyTorch models."""
        import torch as _torch

        return _torch.device(self.device)

    def __repr__(self) -> str:
        return f"DeviceConfig(device={self.device!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeviceConfig):
            return self.device == other.device
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.device)


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def resolve_device(device: str = "auto") -> DeviceName:
    """Resolve device string to 'cuda' or 'cpu'. Convenience wrapper."""
    return _resolve_device_name(device)


def get_onnx_providers(device: str = "auto") -> list[str]:
    """Get ONNX Runtime providers for device. Convenience wrapper."""
    cfg = DeviceConfig(device=device)
    return cfg.onnx_providers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_device.py -v --no-header 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/device.py tests/test_device.py
git commit -m "feat(device): add unified DeviceConfig module with GPU-first defaults"
```

---

## Task 2: Wire DeviceConfig into `AnalysisPipeline`

**Files:**
- Modify: `src/pipeline.py`

- [ ] **Step 1: Update AnalysisPipeline constructor to use DeviceConfig**

In `src/pipeline.py`, add the import and replace `use_gpu: bool` with `device: str | DeviceConfig`:

```python
# At top of file, add to imports:
from .device import DeviceConfig

# In AnalysisPipeline.__init__, replace:
#   use_gpu: bool = True,
# with:
#   device: str | DeviceConfig = "auto",
#
# In the body, replace:
#   self._use_gpu = use_gpu
# with:
#   self._device_config = DeviceConfig(device) if isinstance(device, str) else device
```

The full `__init__` signature becomes:

```python
def __init__(
    self,
    reference_store: "ReferenceStore | None" = None,
    device: str | DeviceConfig = "auto",
    enable_smoothing: bool = True,
    smoothing_config: "OneEuroFilterConfig | None" = None,
    person_click: PersonClick | None = None,
    reestimate_camera: bool = False,
    pose_backend: str = "rtmlib",
) -> None:
```

- [ ] **Step 2: Propagate device to lazy-loaded extractors**

In `_get_pose_2d_extractor()`, pass `device=self._device_config.device`:

```python
def _get_pose_2d_extractor(self):
    if self._pose_2d_extractor is None:
        if self._pose_backend == "rtmlib":
            from .pose_estimation.rtmlib_extractor import RTMPoseExtractor

            self._pose_2d_extractor = RTMPoseExtractor(
                output_format="normalized",
                device=self._device_config.device,
            )
        else:
            from .pose_estimation import H36MExtractor

            self._pose_2d_extractor = H36MExtractor(
                output_format="normalized",
                device=self._device_config.device,
            )
    return self._pose_2d_extractor
```

In `_get_pose_3d_extractor()`, pass `device=self._device_config.device`:

```python
def _get_pose_3d_extractor(self):
    if self._pose_3d_extractor is None:
        from .pose_3d import AthletePose3DExtractor

        model_path = "data/models/motionagformer-s-ap3d.pth.tr"
        self._pose_3d_extractor = AthletePose3DExtractor(
            model_path=Path(model_path) if Path(model_path).exists() else None,
            use_simple=True,
            device=self._device_config.device,
        )
    return self._pose_3d_extractor
```

- [ ] **Step 3: Verify existing pipeline tests still pass**

Run: `uv run pytest tests/ -k "pipeline" -v --no-header 2>&1 | tail -20`
Expected: All pipeline tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/pipeline.py
git commit -m "feat(pipeline): wire DeviceConfig into AnalysisPipeline"
```

---

## Task 3: Update `RTMPoseExtractor` default to GPU

**Files:**
- Modify: `src/pose_estimation/rtmlib_extractor.py`

- [ ] **Step 1: Change default device from "cpu" to "auto"**

In `src/pose_estimation/rtmlib_extractor.py`, line 83, change:

```python
# Before:
device: str = "cpu",
# After:
device: str = "auto",
```

- [ ] **Step 2: Add device resolution in constructor body**

After `self._device = device` (line 96), add:

```python
# Resolve device via DeviceConfig for consistent GPU-first behavior
if device == "auto":
    from ..device import DeviceConfig

    self._device = DeviceConfig(device="auto").device
```

- [ ] **Step 3: Verify tests still pass**

Run: `uv run pytest tests/pose_estimation/ -v --no-header 2>&1 | tail -20`
Expected: All pose estimation tests PASS (mocks already handle device)

- [ ] **Step 4: Commit**

```bash
git add src/pose_estimation/rtmlib_extractor.py
git commit -m "feat(pose): change RTMPoseExtractor default device to auto (GPU-first)"
```

---

## Task 4: Update `ONNXPoseExtractor` to use DeviceConfig

**Files:**
- Modify: `src/pose_3d/onnx_extractor.py`

- [ ] **Step 1: Replace inline provider logic with DeviceConfig**

In `src/pose_3d/onnx_extractor.py`, change the constructor:

```python
# Before (lines 39-54):
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

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    self.session = ort.InferenceSession(str(model_path), providers=providers)
    self.input_name = self.session.get_inputs()[0].name

# After:
def __init__(
    self,
    model_path: Path | str,
    device: str = "auto",
    temporal_window: int = 81,
) -> None:
    import onnxruntime as ort

    from ..device import DeviceConfig

    self.temporal_window = temporal_window
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    cfg = DeviceConfig(device=device)
    self.session = ort.InferenceSession(str(model_path), providers=cfg.onnx_providers)
    self.input_name = self.session.get_inputs()[0].name
```

- [ ] **Step 2: Update test files that pass device="cpu"**

In `tests/pose_3d/test_onnx_extractor.py`, the tests that pass `device="cpu"` will still work since `DeviceConfig(device="cpu")` resolves to `"cpu"`.

- [ ] **Step 3: Run ONNX extractor tests**

Run: `uv run pytest tests/pose_3d/test_onnx_extractor.py -v --no-header 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/pose_3d/onnx_extractor.py
git commit -m "feat(pose_3d): use DeviceConfig for ONNX Runtime provider resolution"
```

---

## Task 5: Update `AthletePose3DExtractor` to use DeviceConfig

**Files:**
- Modify: `src/pose_3d/athletepose_extractor.py`

- [ ] **Step 1: Replace inline auto-detect with DeviceConfig**

In `src/pose_3d/athletepose_extractor.py`, lines 37-75, change:

```python
# Before (lines 70-74):
if device == "auto":
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    self.device = torch.device(device)

# After:
from ..device import DeviceConfig

cfg = DeviceConfig(device=device)
self.device = torch.device(cfg.device)
```

Also update the ONNX sub-extractor creation (line 62):

```python
# Before:
self._onnx = ONNXPoseExtractor(onnx_path, device="cpu")

# After — ONNX sub-extractor inherits the parent's device:
self._onnx = ONNXPoseExtractor(onnx_path, device=device)
```

Note: Remove the `from ..device import DeviceConfig` from inside `__init__` and add it at the top of the file if preferred, or keep it local to avoid import cycles. Since `device.py` imports nothing from `pose_3d`, there's no cycle risk — add at top level.

- [ ] **Step 2: Run athletepose tests**

Run: `uv run pytest tests/pose_3d/ -v --no-header 2>&1 | tail -20`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/pose_3d/athletepose_extractor.py
git commit -m "feat(pose_3d): use DeviceConfig in AthletePose3DExtractor"
```

---

## Task 6: Update `TCPFormerExtractor` to use DeviceConfig

**Files:**
- Modify: `src/pose_3d/tcpformer_extractor.py`

- [ ] **Step 1: Replace inline auto-detect with DeviceConfig**

In `src/pose_3d/tcpformer_extractor.py`, lines 44-47:

```python
# Before:
if device == "auto":
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    self.device = torch.device(device)

# After:
from ..device import DeviceConfig

cfg = DeviceConfig(device=device)
self.device = torch.device(cfg.device)
```

Add the import at the top of the file (no cycle risk).

- [ ] **Step 2: Commit**

```bash
git add src/pose_3d/tcpformer_extractor.py
git commit -m "feat(pose_3d): use DeviceConfig in TCPFormerExtractor"
```

---

## Task 7: Update `H36MExtractor` to accept device strings

**Files:**
- Modify: `src/pose_estimation/h36m_extractor.py`

- [ ] **Step 1: Change default device from "0" to "auto"**

In `src/pose_estimation/h36m_extractor.py`, line 280:

```python
# Before:
device: str = "0",
# After:
device: str = "auto",
```

This file stores `device` but delegates to Ultralytics YOLO which handles its own device resolution. The string `"auto"` is fine for YOLO.

- [ ] **Step 2: Commit**

```bash
git add src/pose_estimation/h36m_extractor.py
git commit -m "feat(pose): change H36MExtractor default device to auto"
```

---

## Task 8: Update `ComparisonConfig` and `ComparisonRenderer`

**Files:**
- Modify: `src/visualization/comparison.py`

- [ ] **Step 1: Replace magic "0" with DeviceConfig**

In `src/visualization/comparison.py`:

```python
# Add import at top:
from ..device import DeviceConfig

# In ComparisonConfig (line 65):
# Before:
device: str = "0"
# After:
device: str = "auto"

# In ComparisonRenderer._create_extractor (lines 93-103):
# Before:
def _create_extractor(self, device: str) -> RTMPoseExtractor:
    """Create RTMPoseExtractor with GPU fallback to CPU."""
    dev = "cuda" if device not in ("cpu", "") else "cpu"
    try:
        print(f"  Trying device: {device} (GPU)...", flush=True)
        extractor = RTMPoseExtractor(conf_threshold=0.3, device=dev)
        return extractor
    except Exception as exc:
        logger.warning("GPU failed: %s", exc)
        print(f"  WARNING: GPU failed ({exc}). Falling back to CPU.", flush=True)
        return RTMPoseExtractor(conf_threshold=0.3, device="cpu")

# After:
def _create_extractor(self, device: str = "auto") -> RTMPoseExtractor:
    """Create RTMPoseExtractor using DeviceConfig."""
    cfg = DeviceConfig(device=device)
    return RTMPoseExtractor(conf_threshold=0.3, device=cfg.device)
```

This removes the duplicated try/except fallback logic — `DeviceConfig` handles it.

- [ ] **Step 2: Commit**

```bash
git add src/visualization/comparison.py
git commit -m "feat(viz): use DeviceConfig in ComparisonRenderer"
```

---

## Task 9: Update `SkeletalIdentityExtractor`

**Files:**
- Modify: `src/tracking/skeletal_identity.py`

- [ ] **Step 1: Pass DeviceConfig to sub-extractor**

In `src/tracking/skeletal_identity.py`, line 104-119:

```python
# Before:
def __init__(
    self,
    model_path: Path | str | None = None,
    device: str = "auto",
) -> None:
    ...
    self._extractor = AthletePose3DExtractor(
        model_path=model_path,
        device=device,
    )

# After — no change needed to constructor signature since
# AthletePose3DExtractor now uses DeviceConfig internally.
# But change the default to be explicit:
# (device="auto" is already correct, and AthletePose3DExtractor
#  will handle resolution via DeviceConfig after Task 5)
```

No code change needed here — the downstream `AthletePose3DExtractor` already handles device resolution via `DeviceConfig` after Task 5. Verify:

- [ ] **Step 2: Verify tests pass**

Run: `uv run pytest tests/tracking/ -v --no-header 2>&1 | tail -15`
Expected: All PASS

---

## Task 10: Update scripts to use DeviceConfig

**Files:**
- Modify: `scripts/gradio_app.py`
- Modify: `scripts/compare_models.py`
- Modify: `src/gradio_helpers.py`

- [ ] **Step 1: Update `gradio_app.py`**

Replace the try/except CUDA fallback block (lines 38-54):

```python
# Before:
def _create_extractor(tracking: str):
    try:
        return RTMPoseExtractor(
            mode="balanced",
            tracking_backend="rtmlib",
            tracking_mode=tracking,
            conf_threshold=0.3,
            output_format="normalized",
            device="cuda",
        )
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU")
        return RTMPoseExtractor(
            mode="balanced",
            tracking_backend="rtmlib",
            tracking_mode=tracking,
            conf_threshold=0.3,
            output_format="normalized",
            device="cpu",
        )

# After:
from src.device import DeviceConfig

def _create_extractor(tracking: str):
    cfg = DeviceConfig.default()
    return RTMPoseExtractor(
        mode="balanced",
        tracking_backend="rtmlib",
        tracking_mode=tracking,
        conf_threshold=0.3,
        output_format="normalized",
        device=cfg.device,
    )
```

- [ ] **Step 2: Update `gradio_helpers.py`**

Replace lines 156-163:

```python
# Before:
extractor = RTMPoseExtractor(
    output_format="normalized",
    conf_threshold=0.3,
    det_frequency=frame_skip,
    frame_skip=frame_skip,
    device="cuda",
    tracking_mode=tracking,
)

# After:
from .device import DeviceConfig

cfg = DeviceConfig.default()
extractor = RTMPoseExtractor(
    output_format="normalized",
    conf_threshold=0.3,
    det_frequency=frame_skip,
    frame_skip=frame_skip,
    device=cfg.device,
    tracking_mode=tracking,
)
```

- [ ] **Step 3: Update `compare_models.py`**

Replace lines 53-63:

```python
# Before:
def _create_extractor(backend: str, conf_threshold: float = 0.3):
    if backend == "rtmlib":
        from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
        return RTMPoseExtractor(
            output_format="normalized",
            conf_threshold=conf_threshold,
            det_frequency=1,
            device="cuda",
        )

# After:
from src.device import DeviceConfig

def _create_extractor(backend: str, conf_threshold: float = 0.3):
    cfg = DeviceConfig.default()
    if backend == "rtmlib":
        from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
        return RTMPoseExtractor(
            output_format="normalized",
            conf_threshold=conf_threshold,
            det_frequency=1,
            device=cfg.device,
        )
```

- [ ] **Step 4: Commit**

```bash
git add scripts/gradio_app.py scripts/compare_models.py src/gradio_helpers.py
git commit -m "feat(scripts): use DeviceConfig in gradio and comparison scripts"
```

---

## Task 11: Update CLI device argument

**Files:**
- Modify: `src/cli.py`

- [ ] **Step 1: Update CLI compare --device default**

In `src/cli.py`, line 691-696:

```python
# Before:
compare_parser.add_argument(
    "--device",
    type=str,
    default="0",
    help="Устройство: '0' GPU, 'cpu' CPU (default: 0)",
)

# After:
compare_parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Устройство: 'auto' (default), 'cuda', 'cpu', или GPU index '0'",
)
```

- [ ] **Step 2: Commit**

```bash
git add src/cli.py
git commit -m "feat(cli): update device argument default to auto"
```

---

## Task 12: Re-export DeviceConfig from `src/__init__.py`

**Files:**
- Modify: `src/__init__.py`

- [ ] **Step 1: Add DeviceConfig to package exports**

In `src/__init__.py`:

```python
from .device import DeviceConfig

__all__ = ["DeviceConfig"]
```

- [ ] **Step 2: Commit**

```bash
git add src/__init__.py
git commit -m "feat: export DeviceConfig from src package"
```

---

## Task 13: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v --no-header 2>&1 | tail -40`
Expected: All tests PASS (same count as before, ~279+)

- [ ] **Step 2: Run type checker**

Run: `uv run basedpyright src/device.py 2>&1`
Expected: 0 errors

- [ ] **Step 3: Run linter**

Run: `uv run ruff check src/device.py 2>&1`
Expected: No issues

---

## Summary

After all tasks:

| Component | Before | After |
|-----------|--------|-------|
| `RTMPoseExtractor` | `"cpu"` | `"auto"` → CUDA-first via `DeviceConfig` |
| `ONNXPoseExtractor` | `"cpu"` | `"auto"` → CUDA-first via `DeviceConfig` |
| `AthletePose3DExtractor` | `"auto"` (inline) | `"auto"` via `DeviceConfig` |
| `TCPFormerExtractor` | `"auto"` (inline) | `"auto"` via `DeviceConfig` |
| `H36MExtractor` | `"0"` | `"auto"` |
| `ComparisonConfig` | `"0"` | `"auto"` via `DeviceConfig` |
| `AnalysisPipeline` | `use_gpu: bool` | `device: str | DeviceConfig` |
| `gradio_app.py` | `"cuda"` + try/except | `DeviceConfig.default()` |
| `gradio_helpers.py` | `"cuda"` | `DeviceConfig.default()` |
| `compare_models.py` | `"cuda"` | `DeviceConfig.default()` |
| CLI `--device` | `"0"` | `"auto"` |

**One module, one resolution, GPU by default.**
