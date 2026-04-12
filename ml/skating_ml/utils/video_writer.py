"""PyAV-based H.264 video writer.

Replaces cv2.VideoWriter (mp4v) and ffmpeg subprocess calls with a single
consistent interface that produces browser-compatible H.264 / yuv420p output.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import av
import cv2

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class H264Writer:
    """Write BGR frames to an H.264 mp4 file via PyAV."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        preset: str = "fast",
        crf: int = 23,
    ) -> None:
        self._container = av.open(str(path), "w")
        rate = Fraction(fps).limit_denominator(1000)
        self._stream = self._container.add_stream(codec, rate=rate)
        self._stream.width = width  # type: ignore[attr-defined]
        self._stream.height = height  # type: ignore[attr-defined]
        self._stream.pix_fmt = "yuv420p"  # type: ignore[attr-defined]
        self._stream.options = {"preset": preset, "crf": str(crf)}  # type: ignore[attr-defined]

    def write(self, frame: np.ndarray) -> None:
        """Write a BGR frame (numpy array)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")  # type: ignore[arg-type]
        for packet in self._stream.encode(av_frame):  # type: ignore[attr-defined]
            self._container.mux(packet)

    def close(self) -> None:
        """Flush remaining packets and close the file."""
        for packet in self._stream.encode():  # type: ignore[attr-defined]
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
