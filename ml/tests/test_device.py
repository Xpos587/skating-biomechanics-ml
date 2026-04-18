# tests/test_device.py
"""Tests for unified device configuration."""

import os
from unittest import mock


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_default_is_cuda(self):
        """Default DeviceConfig should resolve to cuda when available."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig()
        # On systems with CUDA, default should be cuda
        # On systems without, should fall back to cpu
        assert cfg.device in ("cuda", "cpu")

    def test_explicit_cpu(self):
        """Explicit CPU request always returns cpu."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert cfg.device == "cpu"
        assert cfg.is_cpu

    def test_explicit_cuda_fallback(self):
        """Explicit CUDA request falls back to cpu when unavailable."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=False):
            cfg = DeviceConfig(device="cuda")
            assert cfg.device == "cpu"

    def test_auto_with_cuda_available(self):
        """Auto resolves to cuda when CUDA is available."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cuda"
            assert cfg.is_cuda

    def test_auto_without_cuda(self):
        """Auto resolves to cpu when CUDA unavailable."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=False):
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cpu"
            assert cfg.is_cpu

    def test_gpu_index_resolved(self):
        """GPU index '0' resolves to 'cuda'."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            cfg = DeviceConfig(device="0")
            assert cfg.device == "cuda"

    def test_gpu_index_fallback(self):
        """GPU index '0' falls back to cpu when CUDA unavailable."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=False):
            cfg = DeviceConfig(device="0")
            assert cfg.device == "cpu"

    def test_onnx_providers_cpu(self):
        """ONNX providers for CPU-only."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert cfg.onnx_providers == ["CPUExecutionProvider"]

    def test_onnx_providers_cuda(self):
        """ONNX providers for CUDA."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            cfg = DeviceConfig(device="cuda")
            assert cfg.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_default_class_method(self):
        """DeviceConfig.default() returns auto-resolved config."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig.default()
        assert isinstance(cfg, DeviceConfig)
        assert cfg.device in ("cuda", "cpu")

    def test_from_str_cpu(self):
        """DeviceConfig.from_str('cpu') creates CPU config."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig.from_str("cpu")
        assert cfg.device == "cpu"

    def test_from_str_auto(self):
        """DeviceConfig.from_str('auto') creates auto-resolved config."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            cfg = DeviceConfig.from_str("auto")
            assert cfg.device == "cuda"

    def test_from_str_gpu_index(self):
        """DeviceConfig.from_str('0') resolves GPU index."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            cfg = DeviceConfig.from_str("0")
            assert cfg.device == "cuda"

    def test_repr(self):
        """repr shows device string."""
        from skating_ml.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        assert "cpu" in repr(cfg)

    def test_equality(self):
        """Two configs with same device are equal."""
        from skating_ml.device import DeviceConfig

        a = DeviceConfig(device="cpu")
        b = DeviceConfig(device="cpu")
        assert a == b

    def test_inequality(self):
        """Two configs with different devices are not equal."""
        from skating_ml.device import DeviceConfig

        a = DeviceConfig(device="cpu")
        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            b = DeviceConfig(device="cuda")
        assert a != b

    def test_no_cuda_falls_back_to_cpu(self):
        """When CUDA is unavailable, auto falls back to cpu."""
        from skating_ml.device import DeviceConfig

        with mock.patch("skating_ml.device._cuda_available", return_value=False):
            cfg = DeviceConfig(device="auto")
            assert cfg.device == "cpu"

    def test_env_override_skating_device(self):
        """SKATING_DEVICE env var overrides default."""
        from skating_ml.device import DeviceConfig

        with mock.patch.dict(os.environ, {"SKATING_DEVICE": "cpu"}):
            cfg = DeviceConfig()
            assert cfg.device == "cpu"


class TestResolveDevice:
    """Tests for resolve_device convenience function."""

    def test_resolve_cpu(self):
        """resolve_device('cpu') returns 'cpu'."""
        from skating_ml.device import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_resolve_auto_with_cuda(self):
        """resolve_device('auto') returns 'cuda' when available."""
        from skating_ml.device import resolve_device

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_resolve_auto_without_cuda(self):
        """resolve_device('auto') returns 'cpu' when unavailable."""
        from skating_ml.device import resolve_device

        with mock.patch("skating_ml.device._cuda_available", return_value=False):
            assert resolve_device("auto") == "cpu"

    def test_resolve_gpu_index(self):
        """resolve_device('0') returns 'cuda' when available."""
        from skating_ml.device import resolve_device

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            assert resolve_device("0") == "cuda"


class TestGetOnnxProviders:
    """Tests for get_onnx_providers convenience function."""

    def test_cpu_providers(self):
        """CPU providers list."""
        from skating_ml.device import get_onnx_providers

        assert get_onnx_providers("cpu") == ["CPUExecutionProvider"]

    def test_cuda_providers(self):
        """CUDA providers list with CPU fallback."""
        from skating_ml.device import get_onnx_providers

        with mock.patch("skating_ml.device._cuda_available", return_value=True):
            assert get_onnx_providers("cuda") == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]


class TestMultiGPUConfig:
    """Tests for MultiGPUConfig."""

    def test_init_auto_detect(self):
        """Config should auto-detect available GPUs."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig()
        assert config.num_gpus >= 0
        assert isinstance(config.enabled_gpus, list)

    def test_init_specific_gpu_ids(self):
        """Specific GPU IDs should filter enabled GPUs."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig(gpu_ids=[0])
        # Should only enable GPU 0 if available
        assert config.num_gpus <= 1

    def test_init_empty_gpu_ids(self):
        """Empty GPU IDs list should result in no GPUs."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig(gpu_ids=[])
        assert config.num_gpus == 0
        assert not config.has_gpu

    def test_memory_reserve(self):
        """Memory reserve should be reflected in GPUInfo."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig(memory_reserve_mb=1024)
        for gpu in config.enabled_gpus:
            assert gpu.memory_reserve_mb == 1024

    def test_get_device_for_worker_no_gpu(self):
        """Should return 'cpu' when no GPUs available."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig(gpu_ids=[])
        device = config.get_device_for_worker(0)
        assert device == "cpu"

    def test_get_device_for_worker_with_gpu(self):
        """Should return 'cuda' when GPUs available."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig()
        if config.num_gpus > 0:
            device = config.get_device_for_worker(0)
            assert device == "cuda"

    def test_has_gpu_property(self):
        """has_gpu property should reflect GPU availability."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig()
        assert config.has_gpu == (config.num_gpus > 0)

    def test_round_robin_assignment(self):
        """Workers should be assigned in round-robin fashion."""
        from skating_ml.device import MultiGPUConfig

        config = MultiGPUConfig()
        if config.num_gpus >= 2:
            # Both should return cuda (actual GPU ID handled by CUDA_VISIBLE_DEVICES)
            device0 = config.get_device_for_worker(0)
            device4 = config.get_device_for_worker(4)
            assert device0 == "cuda"
            assert device4 == "cuda"


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_available_memory(self):
        """Available memory should subtract reserve."""
        from skating_ml.device import GPUInfo

        gpu = GPUInfo(device_id=0, total_memory_mb=8192, memory_reserve_mb=512)
        assert gpu.available_memory_mb == 7680

    def test_available_memory_clamped_at_zero(self):
        """Available memory should not go negative."""
        from skating_ml.device import GPUInfo

        gpu = GPUInfo(device_id=0, total_memory_mb=256, memory_reserve_mb=512)
        assert gpu.available_memory_mb == 0

    def test_gpu_info_properties(self):
        """GPUInfo properties should be accessible."""
        from skating_ml.device import GPUInfo

        gpu = GPUInfo(device_id=1, total_memory_mb=4096, memory_reserve_mb=256)
        assert gpu.device_id == 1
        assert gpu.total_memory_mb == 4096
        assert gpu.memory_reserve_mb == 256
