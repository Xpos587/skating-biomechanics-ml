# src/device.py
"""Unified device configuration for GPU/CPU management.

Single source of truth for device resolution across all pipeline components.
Uses onnxruntime for CUDA detection (no torch dependency).
Default: CUDA when available, CPU otherwise.

Usage:
    from skating_ml.device import DeviceConfig

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

# Suppress onnxruntime "Some nodes not assigned to preferred EP" warnings
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # 3 = ORT_LOGGING_LEVEL_ERROR

logger = logging.getLogger(__name__)

DeviceName = Literal["cuda", "cpu"]


def _cuda_available() -> bool:
    """Check if CUDA is available via onnxruntime providers."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _resolve_device_name(device: str) -> DeviceName:
    """Normalize device string to 'cuda' or 'cpu'.

    Accepts: "cuda", "cpu", "auto", "0", "1", etc.
    GPU index strings ("0", "1") resolve to "cuda" when available.
    "auto" resolves to "cuda" when available, else "cpu".
    """
    device = device.lower().strip()

    # Environment variable override takes priority
    env = os.environ.get("SKATING_DEVICE", "").lower().strip()
    if env:
        device = env

    # Direct device specifications
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

    # Unknown device
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
