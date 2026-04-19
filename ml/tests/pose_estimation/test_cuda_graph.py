"""Tests for CUDA Graph capture in BatchRTMO.

ONNX Runtime exposes CUDA Graph via CUDAExecutionProvider options:
    ('CUDAExecutionProvider', {'enable_cuda_graph': True})

This must be set at session creation time. The _enable_cuda_graph()
method re-creates the session with this option enabled.
"""

from __future__ import annotations

import importlib
import sys
import types as _stdlib_types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Track original onnxruntime to restore after tests
_ORT_ORIG = sys.modules.get("onnxruntime")


def _install_ort_mock():
    """Create and install a fully-featured onnxruntime mock.

    Returns (mock_ort, mock_session, mock_opts).
    """
    mock_ort = MagicMock()
    mock_ort.__spec__ = importlib.util.spec_from_loader("onnxruntime", loader=None)
    mock_ort.__file__ = "/fake/onnxruntime/__init__.py"
    mock_ort.__path__ = ["/fake/onnxruntime"]
    mock_ort.__package__ = "onnxruntime"

    mock_ort.GraphOptimizationLevel = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

    mock_ort.ExecutionMode = MagicMock()
    mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0

    # SessionOptions -- attribute setting must persist
    mock_opts = _stdlib_types.SimpleNamespace(
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
    """Restore original onnxruntime in sys.modules."""
    if _ORT_ORIG is not None:
        sys.modules["onnxruntime"] = _ORT_ORIG
    else:
        sys.modules.pop("onnxruntime", None)


def _evict_rtmo_batch():
    """Remove cached rtmo_batch module so re-import picks up mocks."""
    for key in list(sys.modules.keys()):
        if "rtmo_batch" in key:
            del sys.modules[key]


def _fresh_batchrtmo(test_device: str = "cpu"):
    """Create a BatchRTMO with fully mocked dependencies.

    Returns (mock_ort, mock_session, mock_opts, instance).
    The onnxruntime mock is **restored** after this call.
    """
    mock_ort, mock_session, mock_opts = _install_ort_mock()
    _evict_rtmo_batch()

    try:
        with (
            patch.dict(
                "src.pose_estimation.rtmo_batch.RTMO_MODELS", {"balanced": "/fake/model.onnx"}
            ),
            patch("src.pose_estimation.rtmo_batch.Path") as mock_path_cls,
        ):
            mock_path = MagicMock()
            mock_path.__str__ = lambda self: "/fake/model.onnx"
            mock_path.exists.return_value = True
            mock_path_cls.return_value = mock_path

            from src.pose_estimation.rtmo_batch import BatchRTMO

            instance = BatchRTMO(
                mode="balanced",
                device=test_device,
                score_thr=0.3,
                nms_thr=0.45,
            )
    finally:
        _restore_ort()
        _evict_rtmo_batch()

    return mock_ort, mock_session, mock_opts, instance


def _fresh_batchrtmo_cuda_with_mock():
    """Create a CUDA BatchRTMO and return it with the mock still active.

    Caller must call _restore_ort() and _evict_rtmo_batch() when done.

    Returns (mock_ort, mock_session, mock_opts, instance).
    """
    mock_ort, mock_session, mock_opts = _install_ort_mock()
    _evict_rtmo_batch()

    with (
        patch.dict("src.pose_estimation.rtmo_batch.RTMO_MODELS", {"balanced": "/fake/model.onnx"}),
        patch("src.pose_estimation.rtmo_batch.Path") as mock_path_cls,
    ):
        mock_path = MagicMock()
        mock_path.__str__ = lambda self: "/fake/model.onnx"
        mock_path.exists.return_value = True
        mock_path_cls.return_value = mock_path

        from src.pose_estimation.rtmo_batch import BatchRTMO

        instance = BatchRTMO(
            mode="balanced",
            device="cuda",
            score_thr=0.3,
            nms_thr=0.45,
        )

    return mock_ort, mock_session, mock_opts, instance


class TestCUDAGraphMethod:
    """BatchRTMO should have _enable_cuda_graph method."""

    def test_batch_rtmo_has_cuda_graph_method(self):
        """BatchRTMO should have _enable_cuda_graph method."""
        _, _, _, instance = _fresh_batchrtmo("cpu")
        assert hasattr(instance, "_enable_cuda_graph")

    def test_cuda_graph_ignored_on_cpu(self):
        """_enable_cuda_graph should be a no-op on CPU device."""
        mock_ort, _session, _, instance = _fresh_batchrtmo("cpu")

        # Re-install mock so _enable_cuda_graph sees it
        _install_ort_mock()
        _evict_rtmo_batch()

        try:
            call_count_before = mock_ort.InferenceSession.call_count
            instance._enable_cuda_graph(batch_size=8)
            call_count_after = mock_ort.InferenceSession.call_count
        finally:
            _restore_ort()
            _evict_rtmo_batch()

        # Should NOT re-create the session on CPU
        assert call_count_after == call_count_before

    def test_cuda_graph_recreates_session_with_provider_options(self):
        """_enable_cuda_graph should re-create session with enable_cuda_graph=True on CUDA."""
        mock_ort, _session, _, instance = _fresh_batchrtmo_cuda_with_mock()

        try:
            # Reset call tracking after __init__
            mock_ort.InferenceSession.reset_mock()

            instance._enable_cuda_graph(batch_size=8)

            # Should have re-created the session
            assert mock_ort.InferenceSession.call_count == 1

            # Check providers include enable_cuda_graph option
            call_kwargs = mock_ort.InferenceSession.call_args.kwargs
            providers = call_kwargs["providers"]
            assert len(providers) >= 1

            # First provider should be CUDAExecutionProvider with options dict
            cuda_provider = providers[0]
            assert cuda_provider[0] == "CUDAExecutionProvider"
            assert "enable_cuda_graph" in cuda_provider[1]
            assert cuda_provider[1]["enable_cuda_graph"] is True
        finally:
            _restore_ort()
            _evict_rtmo_batch()

    def test_cuda_graph_preserves_session_options(self):
        """_enable_cuda_graph should preserve existing SessionOptions optimizations."""
        mock_ort, _, _, instance = _fresh_batchrtmo_cuda_with_mock()

        try:
            mock_ort.InferenceSession.reset_mock()

            instance._enable_cuda_graph(batch_size=4)

            call_kwargs = mock_ort.InferenceSession.call_args.kwargs
            assert "sess_options" in call_kwargs
        finally:
            _restore_ort()
            _evict_rtmo_batch()

    def test_cuda_graph_warmup_runs(self):
        """_enable_cuda_graph should warm up with the given batch_size."""
        mock_ort, mock_session, _, instance = _fresh_batchrtmo_cuda_with_mock()

        try:
            mock_ort.InferenceSession.reset_mock()
            # The new mock session after _enable_cuda_graph re-creation
            mock_ort.InferenceSession.return_value = mock_session
            mock_session.run.reset_mock()

            instance._enable_cuda_graph(batch_size=16)

            # Warm-up should run with batch_size=16
            warmup_call = mock_session.run.call_args
            assert warmup_call is not None
            feeds = warmup_call.args[1]
            input_values = list(feeds.values())
            dummy = input_values[0]
            assert dummy.shape == (16, 3, 640, 640)
        finally:
            _restore_ort()
            _evict_rtmo_batch()


@pytest.mark.cuda
def test_batch_rtmo_cuda_graph():
    """CUDA Graph should capture and replay correctly."""
    pytest.skip("Requires GPU -- manual verification only")
