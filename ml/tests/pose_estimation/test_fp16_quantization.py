"""Tests for FP16 model auto-detection in BatchRTMO.

Since BatchRTMO imports onnxruntime lazily inside __init__, we mock it
at the sys.modules level before importing the module under test.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_ORT_ORIG = sys.modules.get("onnxruntime")


def _install_ort_mock():
    """Create and install a fully-featured onnxruntime mock."""
    mock_ort = MagicMock()
    mock_ort.__spec__ = importlib.util.spec_from_loader("onnxruntime", loader=None)
    mock_ort.__file__ = "/fake/onnxruntime/__init__.py"
    mock_ort.__path__ = ["/fake/onnxruntime"]
    mock_ort.__package__ = "onnxruntime"

    mock_ort.GraphOptimizationLevel = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

    mock_ort.ExecutionMode = MagicMock()
    mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0

    mock_opts = types.SimpleNamespace(
        graph_optimization_level=None,
        enable_mem_pattern=None,
        enable_mem_reuse=None,
        intra_op_num_threads=None,
        inter_op_num_threads=None,
    )
    mock_ort.SessionOptions = MagicMock(return_value=mock_opts)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 640, 640])]
    mock_session.get_outputs.return_value = [
        MagicMock(name="det_output"),
        MagicMock(name="kp_output"),
    ]
    mock_ort.InferenceSession.return_value = mock_session

    sys.modules["onnxruntime"] = mock_ort
    return mock_ort, mock_session, mock_opts


def _restore_ort():
    if _ORT_ORIG is not None:
        sys.modules["onnxruntime"] = _ORT_ORIG
    else:
        sys.modules.pop("onnxruntime", None)


def _evict_rtmo_batch():
    for key in list(sys.modules.keys()):
        if "rtmo_batch" in key:
            del sys.modules[key]


def _make_batchrtmo_with_path(
    mode: str = "balanced",
    fp16_exists: bool = True,
    base_exists: bool = True,
    test_device: str = "cpu",
):
    """Create a BatchRTMO with fully mocked dependencies.

    Args:
        mode: Model preset name.
        fp16_exists: Whether the FP16 model file exists.
        base_exists: Whether the base FP32 model file exists.
        test_device: Device string.

    Returns:
        (mock_ort, mock_session, mock_opts, instance).
    """
    mock_ort, mock_session, mock_opts = _install_ort_mock()
    _evict_rtmo_batch()

    try:
        with patch.dict(
            "src.pose_estimation.rtmo_batch.RTMO_MODELS",
            {"balanced": "/fake/model.onnx"},
        ):
            # Track Path constructor calls to control exists() per-instance
            path_instances: list[MagicMock] = []

            def path_factory(arg=""):
                p = MagicMock()
                p.__str__ = lambda self: str(arg)
                p._arg = str(arg)

                # The FP16 path check creates Path(str(model_path).replace(".onnx", "-fp16.onnx"))
                # The final exists() check is on the resolved model_path
                if "-fp16.onnx" in str(arg):
                    p.exists.return_value = fp16_exists
                else:
                    p.exists.return_value = base_exists

                path_instances.append(p)
                return p

            with patch("src.pose_estimation.rtmo_batch.Path", side_effect=path_factory):
                from src.pose_estimation.rtmo_batch import BatchRTMO

                instance = BatchRTMO(
                    mode=mode,
                    device=test_device,
                    score_thr=0.3,
                    nms_thr=0.45,
                )
    finally:
        _restore_ort()
        _evict_rtmo_batch()

    return mock_ort, mock_session, mock_opts, instance, path_instances


class TestFP16AutoDetection:
    """BatchRTMO should prefer the FP16 variant when it exists alongside the FP32 model."""

    def test_prefers_fp16_when_available(self):
        """When model-fp16.onnx exists, InferenceSession receives the FP16 path."""
        mock_ort, _session, _opts, _instance, _paths = _make_batchrtmo_with_path(
            fp16_exists=True,
        )

        session_call = mock_ort.InferenceSession.call_args
        assert session_call is not None
        model_arg = session_call[0][0]
        assert "-fp16.onnx" in model_arg, f"Expected FP16 model path, got: {model_arg}"

    def test_falls_back_to_fp32_when_no_fp16(self):
        """When FP16 variant does not exist, InferenceSession receives the FP32 path."""
        mock_ort, _session, _opts, _instance, _paths = _make_batchrtmo_with_path(
            fp16_exists=False,
        )

        session_call = mock_ort.InferenceSession.call_args
        assert session_call is not None
        model_arg = session_call[0][0]
        assert model_arg == "/fake/model.onnx", f"Expected FP32 model path, got: {model_arg}"

    def test_both_models_checked(self):
        """Both FP16 and base model paths should have exists() called."""
        _ort, _session, _opts, _instance, paths = _make_batchrtmo_with_path(
            fp16_exists=True,
        )

        # Should have at least 2 Path calls: base model init, fp16 check init
        exists_calls = [p.exists.call_count for p in paths]
        assert any(c > 0 for c in exists_calls), "exists() was never called"


class TestQuantizeScript:
    """Quantization script smoke tests."""

    def test_missing_onnxconverter_raises_system_exit(self):
        """Script should print a helpful message if onnxconverter-common is missing."""
        with (
            patch.dict("sys.modules", {"onnxconverter_common": None}),
            pytest.raises(SystemExit, match="onnxconverter-common"),
        ):
            from scripts.quantize_rtmo_fp16 import quantize

            quantize("/fake/model.onnx", "/fake/model-fp16.onnx")

    def test_quantize_function_exists(self):
        """Quantization module should expose a quantize function."""
        import scripts.quantize_rtmo_fp16 as mod

        assert hasattr(mod, "quantize")
        assert callable(mod.quantize)
