"""Tests for ONNX SessionOptions optimization in BatchRTMO.

Since BatchRTMO imports onnxruntime lazily inside __init__, we mock it
at the sys.modules level before importing the module under test.
"""

from __future__ import annotations

import importlib
import sys
import types
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

    # SessionOptions — attribute setting must persist
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


class TestSessionOptions:
    """BatchRTMO should create InferenceSession with optimized SessionOptions."""

    def test_creates_session_with_sess_options(self):
        """InferenceSession must be called with a sess_options argument."""
        mock_ort, _session, _opts, _ = _fresh_batchrtmo()

        call_args = mock_ort.InferenceSession.call_args
        assert call_args is not None, "InferenceSession was never called"
        assert "sess_options" in call_args.kwargs, (
            "InferenceSession called without sess_options kwarg"
        )

    def test_session_options_graph_optimization(self):
        """SessionOptions must have graph_optimization_level = ORT_ENABLE_ALL."""
        _ort, _session, opts, _ = _fresh_batchrtmo()

        assert opts.graph_optimization_level == 99  # ORT_ENABLE_ALL mock value

    def test_session_options_memory_reuse(self):
        """SessionOptions must have memory reuse and mem_pattern enabled."""
        _ort, _session, opts, _ = _fresh_batchrtmo()

        assert opts.enable_mem_pattern is True
        assert opts.enable_mem_reuse is True


class TestWarmup:
    """BatchRTMO should perform a warm-up inference after session creation."""

    def test_warmup_runs_after_session_creation(self):
        """Session.run must be called during __init__ for warm-up."""
        _ort, mock_session, _opts, _ = _fresh_batchrtmo()

        assert mock_session.run.call_count >= 1, "Warm-up inference was not performed"

    def test_warmup_uses_correct_input_shape(self):
        """Warm-up input should be (1, 3, 640, 640) float32 zeros."""
        _ort, mock_session, _opts, _ = _fresh_batchrtmo()

        warmup_call = mock_session.run.call_args
        assert warmup_call is not None

        # session.run(output_names, {input_name: dummy_tensor})
        assert len(warmup_call.args) >= 2, f"Expected 2+ positional args, got {warmup_call.args}"
        feeds = warmup_call.args[1]
        assert isinstance(feeds, dict), f"Expected dict for feeds, got {type(feeds)}"

        input_values = list(feeds.values())
        assert len(input_values) == 1
        dummy = input_values[0]
        assert dummy.shape == (1, 3, 640, 640)
        assert dummy.dtype == np.float32
        assert np.all(dummy == 0.0)
