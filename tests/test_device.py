# tests/test_device.py
"""Tests for unified device configuration."""

import os
from unittest import mock


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
            # Patch torch in the property's local scope
            with mock.patch.dict("sys.modules", {"torch": mock_torch}):
                _ = cfg.torch_device
            mock_torch.device.assert_called_with("cuda")

    def test_torch_device_cpu(self):
        """torch_device property returns cpu device."""
        from src.device import DeviceConfig

        cfg = DeviceConfig(device="cpu")
        # Should work even without torch installed
        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.device.return_value = mock.Mock()
            # Patch torch in the property's local scope
            with mock.patch.dict("sys.modules", {"torch": mock_torch}):
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

        with mock.patch("src.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            assert get_onnx_providers("cuda") == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
