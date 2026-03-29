"""2D pose estimation using MediaPipe BlazePose.

This module extracts keypoints from video frames using MediaPipe BlazePose,
providing detailed 33-keypoint human pose with foot geometry for figure skating analysis.

BlazePose provides:
- 33 keypoints (vs 17 in COCO/YOLO-Pose)
- Detailed foot points (heel, foot index) for edge detection
- Hand keypoints (thumb, index, pinky) for arm position analysis

MediaPipe 10+ Task API:
- Uses PoseLandmarker from mp.tasks.vision
- Requires separate .task model file download
- Supports IMAGE, VIDEO, and LIVE_STREAM modes
"""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .types import BoundingBox, FrameKeypoints
from .video import extract_frames

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None  # type: ignore[assignment]
    python = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision import pose_landmarker


# Default model paths
DEFAULT_MODEL_PATH = Path("data/models/pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


class BlazePoseExtractor:
    """2D pose extractor using MediaPipe BlazePose with 10+ Task API.

    Extracts 33 keypoints per frame using PoseLandmarker.
    Provides detailed foot geometry for edge detection in figure skating.

    Model variants:
    - lite: Fastest, ~3MB, lower accuracy
    - full: Balanced, ~10MB, good accuracy
    - heavy: Most accurate, ~20MB, slower

    The heavy model is downloaded by default for best accuracy.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        """Initialize BlazePose extractor with MediaPipe 10+ Task API.

        Args:
            model_path: Path to .task model file. Defaults to data/models/pose_landmarker_heavy.task.
            min_detection_confidence: Minimum confidence for person detection [0, 1].
            min_presence_confidence: Minimum confidence for pose presence [0, 1].
            num_poses: Maximum number of poses to detect.

        Raises:
            ImportError: If MediaPipe is not installed.
            FileNotFoundError: If model file doesn't exist.
        """
        if mp is None or vision is None:
            raise ImportError("MediaPipe is not installed. Install with: uv add mediapipe")

        # Determine model path
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download with:\n"
                f"  mkdir -p data/models\n"
                f"  wget {MODEL_URL} -O {model_path}"
            )

        self._model_path = model_path
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._num_poses = num_poses
        self._landmarker: "pose_landmarker.PoseLandmarker" | None = None

    @property
    def landmarker(self) -> "pose_landmarker.PoseLandmarker":
        """Lazy-load PoseLandmarker on first access."""
        if self._landmarker is None:
            # Create BaseOptions with model path
            base_options = python.BaseOptions(model_asset_path=str(self._model_path))

            # Configure PoseLandmarker options
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=self._min_detection_confidence,
                min_pose_presence_confidence=self._min_presence_confidence,
                num_poses=self._num_poses,
                output_segmentation_masks=False,
            )

            # Create PoseLandmarker
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

        return self._landmarker

    def extract_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> FrameKeypoints | None:
        """Extract keypoints from a single frame.

        Args:
            frame: Input frame (height, width, 3) as BGR image.
            timestamp_ms: Timestamp in milliseconds for video mode.

        Returns:
            FrameKeypoints (33, 3) with x, y, confidence or None if no person detected.
            Coordinates are in pixels (not normalized).
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Check if any pose detected
        if not result.pose_landmarks:
            return None

        # Get first person's landmarks (list of 33 NormalizedLandmark)
        landmarks = result.pose_landmarks[0]

        h, w = frame.shape[:2]

        # Convert to numpy array (33, 3)
        keypoints = np.zeros((33, 3), dtype=np.float32)

        for i, landmark in enumerate(landmarks):
            # x, y are normalized [0, 1], convert to pixel coords
            keypoints[i, 0] = landmark.x * w
            keypoints[i, 1] = landmark.y * h
            # Use presence as confidence score
            keypoints[i, 2] = landmark.presence if landmark.presence > 0 else 0.0

        return keypoints

    def extract_video(
        self,
        video_path: Path,
        crop: BoundingBox | None = None,
        fps: float | None = None,
    ) -> FrameKeypoints:
        """Extract keypoints from all frames of a video.

        Args:
            video_path: Path to video file.
            crop: Optional bounding box to crop frames before pose estimation.
            fps: Video FPS for timestamp calculation. If None, will be detected.

        Returns:
            FrameKeypoints (num_frames, 33, 3) with x, y, confidence.
        """
        # Get FPS if not provided
        if fps is None:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        keypoints_list: list[np.ndarray] = []
        frame_idx = 0

        for frame in extract_frames(video_path):
            # Apply crop if provided
            if crop is not None:
                frame = self._apply_crop(frame, crop)

            # Calculate timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            kp = self.extract_frame(frame, timestamp_ms)
            if kp is not None:
                keypoints_list.append(kp)

            frame_idx += 1

        if not keypoints_list:
            raise ValueError("No valid pose detected in video")

        return np.stack(keypoints_list)

    def extract_with_bbox(
        self,
        video_path: Path,
        detector: "PersonDetector",  # type: ignore[valid-type]
    ) -> FrameKeypoints:
        """Extract poses using initial YOLO detection.

        Args:
            video_path: Path to video file.
            detector: PersonDetector instance for initial crop.

        Returns:
            FrameKeypoints (num_frames, 33, 3) with x, y, confidence.
        """
        # Get first frame detection for crop
        bbox = detector.detect_first_frame(video_path)

        if bbox is None:
            # No detection, fall back to full video
            return self.extract_video(video_path)

        # Extract with crop
        return self.extract_video(video_path, crop=bbox)

    def _apply_crop(self, frame: np.ndarray, crop: BoundingBox) -> np.ndarray:
        """Apply bounding box crop to frame.

        Args:
            frame: Input frame.
            crop: BoundingBox to crop to.

        Returns:
            Cropped frame.
        """
        h, w = frame.shape[:2]

        # Convert normalized coords if needed
        if crop.x1 < 1 and crop.x2 < 1:
            # Already normalized
            x1 = int(crop.x1 * w)
            y1 = int(crop.y1 * h)
            x2 = int(crop.x2 * w)
            y2 = int(crop.y2 * h)
        else:
            x1, y1, x2, y2 = int(crop.x1), int(crop.y1), int(crop.x2), int(crop.y2)

        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        return frame[y1:y2, x1:x2].copy()

    def close(self) -> None:
        """Close the landmarker and release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# For backwards compatibility, alias PoseExtractor to BlazePoseExtractor
PoseExtractor = BlazePoseExtractor
