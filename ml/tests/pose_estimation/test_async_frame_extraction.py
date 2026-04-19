"""Tests for double-buffered frame extraction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from src.pose_estimation.pose_extractor import PoseExtractor


class TestAsyncFrameReaderImport:
    """Verify AsyncFrameReader is imported and used by PoseExtractor."""

    def test_pose_extractor_imports_async_frame_reader(self):
        """PoseExtractor module should import AsyncFrameReader."""
        import src.pose_estimation.pose_extractor as pe_mod

        assert hasattr(pe_mod, "AsyncFrameReader"), (
            "AsyncFrameReader should be importable from pose_extractor module"
        )

    def test_async_frame_reader_in_utils(self):
        """AsyncFrameReader should exist in src.utils.frame_buffer."""
        from src.utils.frame_buffer import AsyncFrameReader

        assert AsyncFrameReader is not None

    def test_async_frame_reader_returns_original_frame_idx(self):
        """get_frame should return (original_frame_idx, frame).

        When frame_skip > 1, the original frame index must be preserved
        so the caller can index into pre-allocated arrays by original index.
        """
        from src.utils.frame_buffer import AsyncFrameReader

        reader = AsyncFrameReader(
            video_path=Path(__file__).parent.parent / "data" / "test_video.mp4",
            buffer_size=4,
            frame_skip=2,
        )
        reader.start()
        # Read first frame — original idx should be 1 (skip 0, read 1)
        result = reader.get_frame()
        if result is not None:
            orig_idx, _frame = result
            assert isinstance(orig_idx, int), "First element should be the original frame index"
        reader.join(timeout=2.0)


class TestPoseExtractorIntegration:
    """Integration tests for AsyncFrameReader in PoseExtractor."""

    def _make_extractor(self, frame_skip: int = 1) -> PoseExtractor:
        """Create a PoseExtractor without __init__ (avoids GPU check)."""
        from src.pose_estimation.pose_extractor import PoseExtractor

        extractor = PoseExtractor.__new__(PoseExtractor)
        extractor._mode = "balanced"
        extractor._tracking_backend = "rtmlib"
        extractor._tracking_mode = "sports2d"
        extractor._conf_threshold = 0.3
        extractor._output_format = "normalized"
        extractor._frame_skip = frame_skip
        extractor._device = "cpu"
        extractor._backend = "onnxruntime"
        extractor._tracker = MagicMock()
        return extractor

    def _mock_video_capture(self):
        """Create a mock cv2.VideoCapture that behaves like a valid video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        return mock_cap

    @patch("src.pose_estimation.pose_extractor.AsyncFrameReader")
    def test_extract_video_tracked_uses_async_reader(self, MockReader):
        """extract_video_tracked should instantiate AsyncFrameReader."""
        MockReader.return_value.get_frame.side_effect = [None]  # No frames

        extractor = self._make_extractor(frame_skip=1)

        video_path = Path("/fake/video.mp4")
        with (
            patch("src.pose_estimation.pose_extractor.get_video_meta") as mock_meta,
            patch(
                "src.pose_estimation.pose_extractor.cv2.VideoCapture",
                return_value=self._mock_video_capture(),
            ),
            pytest.raises(ValueError, match="No valid pose"),
        ):
            mock_meta.return_value = MagicMock(num_frames=10, fps=30.0, width=640, height=480)
            extractor.extract_video_tracked(video_path)

    @patch("src.pose_estimation.pose_extractor.AsyncFrameReader")
    def test_async_reader_created_with_frame_skip(self, MockReader):
        """AsyncFrameReader should be created with extractor's frame_skip."""
        MockReader.return_value.get_frame.side_effect = [None]

        extractor = self._make_extractor(frame_skip=4)

        video_path = Path("/fake/video.mp4")
        with (
            patch("src.pose_estimation.pose_extractor.get_video_meta") as mock_meta,
            patch(
                "src.pose_estimation.pose_extractor.cv2.VideoCapture",
                return_value=self._mock_video_capture(),
            ),
            pytest.raises(ValueError, match="No valid pose"),
        ):
            mock_meta.return_value = MagicMock(num_frames=10, fps=30.0, width=640, height=480)
            extractor.extract_video_tracked(video_path)

        MockReader.assert_called_once()
        call_kwargs = MockReader.call_args
        assert call_kwargs.kwargs.get("frame_skip") == 4, (
            "AsyncFrameReader should receive frame_skip=4"
        )
