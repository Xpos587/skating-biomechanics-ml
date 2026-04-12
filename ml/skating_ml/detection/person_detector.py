"""Person detection using YOLOv11.

This module provides YOLOv11-based person detection for figure skating videos.
Uses Ultralytics library for model loading and inference.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO  # type: ignore[import-untyped]

from ..types import BoundingBox
from ..utils.video import extract_frames


class PersonDetector:
    """Person detector using YOLOv11.

    Detects people in video frames using YOLOv11n (nano) model for
    real-time performance on RTX 3050 Ti.

    Model is downloaded automatically on first use from Ultralytics hub.
    """

    def __init__(self, model_size: str = "n", confidence: float = 0.5) -> None:
        """Initialize person detector.

        Args:
            model_size: YOLOv11 model size ('n' for nano, 's' for small, 'm' for medium).
                       'n' is recommended for RTX 3050 Ti (~2.6M params, ~1.5ms/frame).
            confidence: Minimum confidence threshold for detections [0, 1].
        """
        self._model_size = model_size
        self._confidence = confidence
        self._model: YOLO | None = None
        self._target_class = 0  # COCO 'person' class

    @property
    def model(self) -> YOLO:
        """Lazy-load YOLO model on first access."""
        if self._model is None:
            model_name = f"yolov8{self._model_size}.pt"
            self._model = YOLO(model_name)
        return self._model

    def detect_frame(self, frame: np.ndarray) -> BoundingBox | None:
        """Detect person in a single frame.

        Args:
            frame: Input frame (height, width, 3) as BGR image.

        Returns:
            BoundingBox of highest confidence person detection, or None if no person found.
        """
        results = self.model(
            frame,
            classes=[self._target_class],
            conf=self._confidence,
            verbose=False,
        )

        if not results or not results[0].boxes:
            return None

        # Get highest confidence detection
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax()) if len(boxes.conf) > 0 else 0

        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        conf = float(boxes.conf[best_idx])

        return BoundingBox(
            x1=float(xyxy[0]),
            y1=float(xyxy[1]),
            x2=float(xyxy[2]),
            y2=float(xyxy[3]),
            confidence=conf,
        )

    def detect_video(self, video_path: Path) -> list[BoundingBox]:
        """Detect person in all frames of a video.

        Args:
            video_path: Path to video file.

        Returns:
            List of BoundingBox per frame. May contain None for frames
            where no person was detected.

        Note:
            For production use, consider using tracking (ByteTrack) instead
            of per-frame detection to maintain consistent person ID.
        """
        bboxes: list[BoundingBox | None] = []

        for frame in extract_frames(video_path):
            bbox = self.detect_frame(frame)
            bboxes.append(bbox)

        # Filter out None values (frames with no detection)
        return [b for b in bboxes if b is not None]

    def detect_first_frame(self, video_path: Path) -> BoundingBox | None:
        """Detect person in the first frame only.

        Args:
            video_path: Path to video file.

        Returns:
            BoundingBox from first frame, or None if no person found.

        Note:
            This is efficient for videos with a single person, as BlazePose
            will handle tracking internally. Use this instead of detect_video()
            for single-person videos.
        """
        from .video import extract_frames  # type: ignore[import]

        for frame in extract_frames(video_path, max_frames=1):
            return self.detect_frame(frame)

        return None
