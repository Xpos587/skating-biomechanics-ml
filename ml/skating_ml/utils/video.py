"""Video processing utilities using OpenCV."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ..types import BoundingBox, VideoMeta


def open_video(path: Path) -> cv2.VideoCapture:
    """Open video file for reading.

    Args:
        path: Path to video file.

    Returns:
        cv2.VideoCapture instance. Caller must call .release().

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If video cannot be opened.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    return cap


def get_video_meta(path: Path) -> VideoMeta:
    """Extract video metadata without loading frames.

    Args:
        path: Path to video file.

    Returns:
        VideoMeta with fps, width, height, num_frames.
    """
    cap = open_video(path)

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return VideoMeta(
            path=path,
            fps=float(fps),
            width=width,
            height=height,
            num_frames=num_frames,
        )
    finally:
        cap.release()


def extract_frames(
    path: Path,
    max_frames: int = 0,
    start_frame: int = 0,
) -> Iterator[np.ndarray]:
    """Extract frames from video as generator.

    Args:
        path: Path to video file.
        max_frames: Maximum number of frames to extract (0 = all frames).
        start_frame: Starting frame index.

    Yields:
        np.ndarray: Frame as BGR image (height, width, 3).

    Raises:
        RuntimeError: If frame cannot be read.
    """
    cap = open_video(path)

    try:
        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        count = 0
        while True:
            if max_frames > 0 and count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            yield frame
            count += 1

    finally:
        cap.release()


def select_person_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding: float = 0.2,
) -> np.ndarray:
    """Extract crop around detected person with padding.

    Args:
        frame: Full frame (height, width, 3).
        bbox: BoundingBox of detected person.
        padding: Padding ratio (0.2 = 20% extra on each side).

    Returns:
        Cropped frame (crop_height, crop_width, 3).
    """
    h, w = frame.shape[:2]

    # Calculate crop dimensions with padding
    crop_w = int(bbox.width * (1 + 2 * padding))
    crop_h = int(bbox.height * (1 + 2 * padding))

    # Calculate top-left corner
    x1 = int(bbox.center_x - crop_w / 2)
    y1 = int(bbox.center_y - crop_h / 2)

    # Clamp to frame boundaries
    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Handle edge cases
    if crop_w <= 0 or crop_h <= 0:
        return frame

    if x2 > w or y2 > h:
        # Fallback to bbox without padding if out of bounds
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

    return frame[y1:y2, x1:x2]
