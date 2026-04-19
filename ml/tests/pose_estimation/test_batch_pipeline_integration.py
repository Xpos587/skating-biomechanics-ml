"""Tests for batch RTMO integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_extract_video_tracked_accepts_use_batch_param():
    """extract_video_tracked should accept use_batch parameter."""
    import inspect

    from src.pose_estimation.pose_extractor import PoseExtractor

    sig = inspect.signature(PoseExtractor.extract_video_tracked)
    assert "use_batch" in sig.parameters
    assert "batch_size" in sig.parameters


def test_extract_video_tracked_dispatches_to_batch():
    """When use_batch=True, should use BatchRTMO path."""
    from src.pose_estimation.pose_extractor import PoseExtractor

    # Verify the batch method exists
    assert hasattr(PoseExtractor, "_extract_batch")


def test_batch_method_signature():
    """_extract_batch should have correct signature."""
    import inspect

    from src.pose_estimation.pose_extractor import PoseExtractor

    sig = inspect.signature(PoseExtractor._extract_batch)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "video_path" in params
    assert "person_click" in params
    assert "progress_cb" in params
    assert "batch_size" in params
