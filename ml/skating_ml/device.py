# src/device.py
"""Unified device configuration for GPU/CPU management.

Single source of truth for device resolution across all pipeline components.
Uses onnxruntime for CUDA detection (no torch dependency).
Default: CUDA when available, CPU otherwise.

Usage:
    from skating_ml.device import DeviceConfig, MultiGPUConfig

    # Simple — auto-detect best device
    cfg = DeviceConfig.default()

    # Explicit
    cfg = DeviceConfig(device="cpu")

    # Multi-GPU configuration
    multi_cfg = MultiGPUConfig()  # Auto-detect all GPUs
    multi_cfg = MultiGPUConfig(gpu_ids=[0, 1])  # Specific GPUs

    # Pass to components
    extractor = PoseExtractor(device=cfg.device)
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


# ------------------------------------------------------------------
# Multi-GPU Configuration
# ------------------------------------------------------------------


def _get_gpu_count() -> int:
    """Get number of available GPUs using NVML.

    Returns:
        Number of GPUs, or 0 if query fails.
    """
    try:
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        nvmlInit()
        count = nvmlDeviceGetCount()
        nvmlShutdown()
        return count
    except Exception:
        return 0


def _get_gpu_memory_mb(gpu_id: int) -> int:
    """Get total GPU memory in MB using NVML.

    Args:
        gpu_id: GPU device ID.

    Returns:
        Total memory in MB, or 0 if query fails.
    """
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_id)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        # Convert bytes to MB
        return memory_info.total // (1024 * 1024)
    except Exception:
        return 0


@dataclass
class GPUInfo:
    """Information about a single GPU.

    Attributes:
        device_id: GPU device ID (0-based index).
        total_memory_mb: Total GPU memory in MB.
        memory_reserve_mb: Memory to reserve in MB.
    """

    device_id: int
    total_memory_mb: int
    memory_reserve_mb: int = 512

    @property
    def available_memory_mb(self) -> int:
        """Available memory after reserve."""
        return max(0, self.total_memory_mb - self.memory_reserve_mb)


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU processing.

    Auto-detects available GPUs via nvidia-smi and provides
    device assignment for parallel workers.

    Attributes:
        gpu_ids: List of GPU IDs to use (None = all available).
        memory_reserve_mb: Memory to reserve per GPU in MB.
        enabled_gpus: List of GPUInfo for enabled GPUs.

    Example:
        # Auto-detect all GPUs
        config = MultiGPUConfig()

        # Use specific GPUs
        config = MultiGPUConfig(gpu_ids=[0, 2])

        # Get device for worker
        device = config.get_device_for_worker(worker_id=0)
    """

    gpu_ids: list[int] | None = None
    memory_reserve_mb: int = 512
    enabled_gpus: list[GPUInfo] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Detect GPUs and populate enabled_gpus."""
        # Detect available GPUs using NVML
        all_gpus: list[GPUInfo] = []

        gpu_count = _get_gpu_count()
        for i in range(gpu_count):
            memory_mb = _get_gpu_memory_mb(i)
            all_gpus.append(
                GPUInfo(
                    device_id=i,
                    total_memory_mb=memory_mb,
                    memory_reserve_mb=self.memory_reserve_mb,
                )
            )

        # Filter by gpu_ids if specified
        if self.gpu_ids is not None:
            enabled = [gpu for gpu in all_gpus if gpu.device_id in self.gpu_ids]
        else:
            enabled = all_gpus

        object.__setattr__(self, "enabled_gpus", enabled)

        if not enabled:
            logger.warning("No GPUs detected, falling back to CPU-only mode")

    def get_device_for_worker(self, worker_id: int) -> str:
        """Get device string for a specific worker.

        Distributes workers across available GPUs in round-robin fashion.

        Args:
            worker_id: Worker index (0-based).

        Returns:
            "cuda" if GPUs available, "cpu" otherwise.
        """
        if not self.enabled_gpus:
            return "cpu"

        # Round-robin assignment
        gpu_idx = worker_id % len(self.enabled_gpus)
        gpu_info = self.enabled_gpus[gpu_idx]

        # Set CUDA device for this worker
        gpu_device_id = gpu_info.device_id

        # Return device string (caller handles CUDA_VISIBLE_DEVICES)
        return "cuda"

    @property
    def num_gpus(self) -> int:
        """Number of enabled GPUs."""
        return len(self.enabled_gpus)

    @property
    def has_gpu(self) -> bool:
        """Whether at least one GPU is available."""
        return self.num_gpus > 0
