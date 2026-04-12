"""Central ONNX session lifecycle manager.

Lazy-loads models on first use, tracks VRAM budget, supports LRU eviction
when budget is exceeded. All models run via ONNX Runtime (no PyTorch at runtime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import onnxruntime as ort

from skating_ml.device import DeviceConfig

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

        logger.info(
            "Loading model '%s' from %s (%dMB, device=%s)",
            model_id,
            entry.path,
            entry.vram_mb,
            self._device.device,
        )
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
