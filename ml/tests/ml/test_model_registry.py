"""Tests for ModelRegistry."""

from unittest import mock

import pytest


class TestModelRegistry:
    """Tests for central ONNX session lifecycle manager."""

    def test_create_registry_with_defaults(self):
        """Registry initializes with default settings."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.vram_budget_mb == 3800
        assert reg.vram_used_mb == 0
        assert len(reg._entries) == 0

    def test_create_registry_custom_budget(self):
        """Registry accepts custom VRAM budget."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry(vram_budget_mb=2000)
        assert reg.vram_budget_mb == 2000

    def test_register_model(self):
        """register() tracks model metadata without loading."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")
        assert reg.is_registered("depth_anything")
        assert reg.vram_used_mb == 0  # not loaded yet

    def test_get_loads_on_demand(self):
        """get() lazy-loads ONNX session on first call."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            session = reg.get("depth_anything")
            assert session is mock_session
            assert reg.vram_used_mb == 200

    def test_get_returns_cached_session(self):
        """Second get() returns same session without reloading."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            s1 = reg.get("depth_anything")
            s2 = reg.get("depth_anything")
            assert s1 is s2

    def test_get_unregistered_raises(self):
        """get() raises KeyError for unregistered model."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_release_frees_session(self):
        """release() unloads session and frees VRAM."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            reg.get("depth_anything")
            reg.release("depth_anything")
            assert reg.vram_used_mb == 0
            mock_session.release.assert_called_once()

    def test_release_unregistered_noop(self):
        """release() is a no-op for unregistered/already released model."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.release("nonexistent")  # should not raise

    def test_vram_budget_enforced(self):
        """get() raises RuntimeError if loading would exceed VRAM budget."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry(vram_budget_mb=100)
        reg.register("big_model", vram_mb=200, path="/tmp/big.onnx")

        mock_session = mock.MagicMock()
        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            with pytest.raises(RuntimeError, match="VRAM budget"):
                reg.get("big_model")

    def test_device_passed_to_session(self):
        """ONNX session created with correct providers from DeviceConfig."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("test_model", vram_mb=50, path="/tmp/test.onnx")

        mock_session = mock.MagicMock()
        with mock.patch(
            "skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session
        ) as mock_cls:
            reg.get("test_model")
            mock_cls.assert_called_once_with(
                "/tmp/test.onnx",
                providers=["CPUExecutionProvider"],
            )

    def test_is_loaded(self):
        """is_loaded() returns correct state."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("m", vram_mb=50, path="/tmp/m.onnx")
        assert not reg.is_loaded("m")

        mock_session = mock.MagicMock()
        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            reg.get("m")
            assert reg.is_loaded("m")

    def test_list_models(self):
        """list_models() returns registered model IDs."""
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry()
        reg.register("a", vram_mb=50, path="/tmp/a.onnx")
        reg.register("b", vram_mb=50, path="/tmp/b.onnx")
        assert reg.list_models() == ["a", "b"]
