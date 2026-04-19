"""Producer-consumer frame buffer for overlapping video decode with GPU inference.

Background thread decodes frames into a bounded queue so that GPU inference
on frame N overlaps with CPU video decode of frame N+1.
"""

from __future__ import annotations

import threading
from queue import Queue
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class AsyncFrameReader:
    """Background thread decodes frames into a bounded queue.

    Args:
        video_path: Path to video file.
        buffer_size: Max frames to buffer ahead (default 16).
        frame_skip: Process every Nth frame (default 1).

    Example::

        reader = AsyncFrameReader("video.mp4", buffer_size=16, frame_skip=1)
        reader.start()
        while (result := reader.get_frame()) is not None:
            frame_idx, frame = result
            # ... process frame ...
        reader.join()
    """

    _SENTINEL = object()

    def __init__(
        self,
        video_path: str | Path,
        buffer_size: int = 16,
        frame_skip: int = 1,
    ) -> None:
        self._path = str(video_path)
        self._buffer_size = buffer_size
        self._frame_skip = max(1, frame_skip)
        self._queue: Queue[tuple[int, np.ndarray] | object] = Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background decode thread."""
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._queue.put(self._SENTINEL)
            return

        original_idx = 0
        skip_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if skip_counter < self._frame_skip - 1:
                skip_counter += 1
                original_idx += 1
                continue
            skip_counter = 0
            self._queue.put((original_idx, frame))
            original_idx += 1

        cap.release()
        self._queue.put(self._SENTINEL)

    def get_frame(self) -> tuple[int, np.ndarray] | None:
        """Get next frame. Returns (original_frame_idx, frame) or None when exhausted."""
        item = self._queue.get()
        if item is self._SENTINEL:
            return None
        return item  # type: ignore[return-value]

    def join(self, timeout: float = 5.0) -> None:
        """Wait for background thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
