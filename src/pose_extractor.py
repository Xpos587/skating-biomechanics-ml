"""2D pose estimation using Ultralytics YOLO-Pose.

This module extracts keypoints from video frames using YOLO-Pose,
providing human pose information for figure skating analysis.

Note: YOLO-Pose provides 17 COCO-format keypoints. For 33 BlazePose-format
keypoints with detailed foot geometry, MediaPipe would be needed but requires
additional model file downloads in newer versions.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from .types import BoundingBox, FrameKeypoints
from .video import extract_frames

if TYPE_CHECKING:
    pass


class PoseExtractor:
    """2D pose extractor using Ultralytics YOLO-Pose.

    Extracts 17 keypoints per frame using YOLOv11-pose model.
    Provides sufficient detail for basic biomechanics analysis.

    Model options:
    - yolo11n-pose: Nano (fastest, ~2.6M params)
    - yolo11s-pose: Small (balanced, ~9M params)
    - yolo11m-pose: Medium (more accurate, ~21M params)
    """

    def __init__(
        self,
        model_size: str = "s",
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize pose extractor.

        Args:
            model_size: YOLO model size ('n', 's', or 'm').
            min_confidence: Minimum confidence for keypoint detection [0, 1].
        """
        self._model_size = model_size
        self._min_confidence = min_confidence
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Lazy-load YOLO-Pose model on first access."""
        if self._model is None:
            model_name = f"yolov8{self._model_size}-pose.pt"
            self._model = YOLO(model_name)
        return self._model

    def extract_frame(self, frame: np.ndarray) -> FrameKeypoints | None:
        """Extract keypoints from a single frame.

        Args:
            frame: Input frame (height, width, 3) as BGR image.

        Returns:
            FrameKeypoints (17, 3) with x, y, confidence, or None if no person detected.
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self._min_confidence,
        )

        if not results or not results[0].keypoints:
            return None

        # Get keypoints from first detected person
        keypoints_data = results[0].keypoints.data.cpu().numpy()

        if len(keypoints_data) == 0:
            return None

        # Ultralytics returns (num_persons, 17, 3) - take first person
        keypoints = keypoints_data[0]  # (17, 3) with x, y, confidence

        return keypoints.astype(np.float32)

    def extract_video(
        self,
        video_path: Path,
        crop: BoundingBox | None = None,
    ) -> FrameKeypoints:
        """Extract keypoints from all frames of a video.

        Args:
            video_path: Path to video file.
            crop: Optional bounding box to crop frames before pose estimation.

        Returns:
            FrameKeypoints (num_frames, 17, 3) with x, y, confidence.
        """
        keypoints_list: list[np.ndarray | None] = []

        for frame in extract_frames(video_path):
            # Apply crop if provided
            if crop is not None:
                from .video import select_person_crop

                frame = select_person_crop(frame, crop)

            kp = self.extract_frame(frame)
            keypoints_list.append(kp)

        # Filter out None values and stack
        valid_keypoints = [kp for kp in keypoints_list if kp is not None]

        if not valid_keypoints:
            raise ValueError("No valid pose detected in video")

        return np.stack(valid_keypoints)

    def extract_with_bbox(
        self,
        video_path: Path,
        detector: "PersonDetector",  # type: ignore[valid-type]
    ) -> FrameKeypoints:
        """Extract poses using initial YOLO detection.

        YOLO-Pose handles detection internally, so this is
        equivalent to extract_video().

        Args:
            video_path: Path to video file.
            detector: PersonDetector instance (unused, kept for API compatibility).

        Returns:
            FrameKeypoints (num_frames, 17, 3) with x, y, confidence.
        """
        # YOLO-Pose does its own detection
        return self.extract_video(video_path)
