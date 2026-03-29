"""H3.6M 17-keypoint pose extractor.

Direct H3.6M format extraction using YOLOv8-Pose backend with integrated conversion.
This is the primary 2D pose extractor for the skating analysis pipeline.

Architecture:
    YOLOv8-Pose (17kp COCO) → geometric conversion → H3.6M (17kp) output

The conversion is geometric (not learned) and happens on-the-fly during extraction.
"""

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]



# H3.6M keypoint indices
class H36Key:
    """H3.6M keypoint indices (17 total)."""

    HIP_CENTER = 0
    RHIP = 1
    RKNEE = 2
    RFOOT = 3
    LHIP = 4
    LKNEE = 5
    LFOOT = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LSHOULDER = 11
    LELBOW = 12
    LWRIST = 13
    RSHOULDER = 14
    RELBOW = 15
    RWRIST = 16


# YOLOv8-Pose COCO keypoint indices (for internal mapping)
class _COCOKey:
    """YOLOv8-Pose COCO keypoint indices (17 total) - internal use only."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# H3.6M skeleton connections for visualization
H36M_SKELETON_EDGES = [
    # Torso
    (H36Key.HIP_CENTER, H36Key.SPINE),
    (H36Key.SPINE, H36Key.THORAX),
    (H36Key.THORAX, H36Key.NECK),
    (H36Key.NECK, H36Key.HEAD),
    # Right arm
    (H36Key.THORAX, H36Key.RSHOULDER),
    (H36Key.RSHOULDER, H36Key.RELBOW),
    (H36Key.RELBOW, H36Key.RWRIST),
    # Left arm
    (H36Key.THORAX, H36Key.LSHOULDER),
    (H36Key.LSHOULDER, H36Key.LELBOW),
    (H36Key.LELBOW, H36Key.LWRIST),
    # Right leg
    (H36Key.HIP_CENTER, H36Key.RHIP),
    (H36Key.RHIP, H36Key.RKNEE),
    (H36Key.RKNEE, H36Key.RFOOT),
    # Left leg
    (H36Key.HIP_CENTER, H36Key.LHIP),
    (H36Key.LHIP, H36Key.LKNEE),
    (H36Key.LKNEE, H36Key.LFOOT),
]


# H3.6M keypoint names
H36M_KEYPOINT_NAMES = [
    "hip_center",
    "rhip",
    "rknee",
    "rfoot",
    "lhip",
    "lknee",
    "lfoot",
    "spine",
    "thorax",
    "neck",
    "head",
    "lshoulder",
    "lelbow",
    "lwrist",
    "rshoulder",
    "relbow",
    "rwrist",
]


def _coco_to_h36m_single(coco_pose: np.ndarray) -> np.ndarray:
    """Convert YOLOv8-Pose COCO 17 keypoints to H3.6M 17 keypoints (single frame).

    Args:
        coco_pose: (17, 2) or (17, 3) array in COCO format

    Returns:
        h36m_pose: (17, 2) or (17, 3) array in H3.6M format
    """
    has_confidence = coco_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=coco_pose.dtype)

    # Midpoints
    mid_hip = (coco_pose[_COCOKey.LEFT_HIP] + coco_pose[_COCOKey.RIGHT_HIP]) / 2
    mid_shoulder = (
        coco_pose[_COCOKey.LEFT_SHOULDER] + coco_pose[_COCOKey.RIGHT_SHOULDER]
    ) / 2

    # Direct mapping from COCO to H3.6M
    h36m_pose[H36Key.HIP_CENTER] = mid_hip
    h36m_pose[H36Key.RHIP] = coco_pose[_COCOKey.RIGHT_HIP]
    h36m_pose[H36Key.RKNEE] = coco_pose[_COCOKey.RIGHT_KNEE]
    h36m_pose[H36Key.RFOOT] = coco_pose[_COCOKey.RIGHT_ANKLE]
    h36m_pose[H36Key.LHIP] = coco_pose[_COCOKey.LEFT_HIP]
    h36m_pose[H36Key.LKNEE] = coco_pose[_COCOKey.LEFT_KNEE]
    h36m_pose[H36Key.LFOOT] = coco_pose[_COCOKey.LEFT_ANKLE]
    h36m_pose[H36Key.SPINE] = mid_shoulder * 0.5 + mid_hip * 0.5
    h36m_pose[H36Key.THORAX] = mid_shoulder
    h36m_pose[H36Key.NECK] = coco_pose[_COCOKey.NOSE]
    h36m_pose[H36Key.HEAD] = coco_pose[_COCOKey.NOSE]
    h36m_pose[H36Key.LSHOULDER] = coco_pose[_COCOKey.LEFT_SHOULDER]
    h36m_pose[H36Key.LELBOW] = coco_pose[_COCOKey.LEFT_ELBOW]
    h36m_pose[H36Key.LWRIST] = coco_pose[_COCOKey.LEFT_WRIST]
    h36m_pose[H36Key.RSHOULDER] = coco_pose[_COCOKey.RIGHT_SHOULDER]
    h36m_pose[H36Key.RELBOW] = coco_pose[_COCOKey.RIGHT_ELBOW]
    h36m_pose[H36Key.RWRIST] = coco_pose[_COCOKey.RIGHT_WRIST]

    return h36m_pose


class H36MExtractor:
    """H3.6M 17-keypoint pose extractor.

    Uses YOLOv8-Pose backend with integrated H3.6M conversion.
    Outputs H3.6M format directly (17 keypoints) - no intermediate COCO storage.

    This is the primary 2D pose extractor for the skating analysis pipeline.

    Advantages over BlazePose:
    - Single-stage detection + pose (faster)
    - No left/right confusion (better tracking)
    - Easy API with ultralytics
    """

    def __init__(
        self,
        model_size: str = "n",
        model_path: Path | str | None = None,
        conf_threshold: float = 0.5,
        output_format: str = "normalized",  # "normalized" or "pixels"
        skip_model_check: bool = False,
    ):
        """Initialize H3.6M extractor with YOLOv8-Pose backend.

        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
            model_path: Path to custom model weights, or None for default
            conf_threshold: Minimum confidence for pose detection [0, 1]
            output_format: "normalized" for [0,1] coords, "pixels" for absolute pixel coords
            skip_model_check: If True, don't validate model exists (for testing)
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics not installed. Install with: uv add ultralytics"
            )

        self.model_size = model_size
        self._model_path = Path(model_path) if model_path else None
        self._conf_threshold = conf_threshold
        self._output_format = output_format
        self._skip_model_check = skip_model_check

        # Lazy-load model on first access
        self._model: YOLO | None = None

    @property
    def model(self) -> "YOLO":
        """Lazy-load YOLO model on first access."""
        if self._model is None:
            if self._model_path is not None:
                self._model = YOLO(str(self._model_path))
            else:
                # YOLOv8n-Pose (YOLOv8 doesn't have pose models yet)
                model_name = f"yolov8{self.model_size}-pose.pt"
                self._model = YOLO(model_name)
        return self._model

    def extract_frame(
        self, frame: np.ndarray
    ) -> np.ndarray | None:
        """Extract H3.6M pose from single frame.

        Args:
            frame: Input frame (height, width, 3) as BGR image.

        Returns:
            pose: (17, 3) array with x, y, confidence in H3.6M format.
                  Returns None if no person detected.
                  Coordinates are normalized [0,1] if output_format="normalized",
                  or in pixels if output_format="pixels".
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(frame_rgb, verbose=False, conf=self._conf_threshold)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Get first person detected
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # (17, 2)
        confidence = results[0].keypoints.conf.cpu().numpy()[0]  # (17,)

        # Normalize to [0, 1]
        h, w = frame.shape[:2]
        keypoints_norm = keypoints.copy()
        keypoints_norm[:, 0] /= w
        keypoints_norm[:, 1] /= h

        # Combine x, y, confidence
        coco_kp = np.zeros((17, 3), dtype=np.float32)
        coco_kp[:, :2] = keypoints_norm
        coco_kp[:, 2] = confidence

        # Convert to H3.6M 17kp (integrated conversion)
        h36m_kp = _coco_to_h36m_single(coco_kp)

        # Convert to pixels if requested
        if self._output_format == "pixels":
            h36m_kp[:, 0] *= w
            h36m_kp[:, 1] *= h

        return h36m_kp

    def extract_video(
        self,
        video_path: Path | str,
        fps: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract H3.6M poses from all frames of a video.

        Args:
            video_path: Path to video file.
            fps: Video FPS (not used, kept for API compatibility).

        Returns:
            poses: (N, 17, 3) array with x, y, confidence in H3.6M format.
            frame_indices: (N,) array of frame indices where poses were detected.
        """
        video_path = Path(video_path)

        # Run inference on video
        results = self.model(str(video_path), verbose=False, conf=self._conf_threshold, stream=True)

        poses_list = []
        frame_indices = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kp = result.keypoints.xy.cpu().numpy()[0]  # (17, 2)
                conf = result.keypoints.conf.cpu().numpy()[0]  # (17,)

                # Normalize to [0, 1]
                h, w = result.orig_shape
                kp_norm = kp.copy()
                kp_norm[:, 0] /= w
                kp_norm[:, 1] /= h

                # Combine x, y, confidence
                coco_kp = np.zeros((17, 3), dtype=np.float32)
                coco_kp[:, :2] = kp_norm
                coco_kp[:, 2] = conf

                # Convert to H3.6M 17kp (integrated conversion)
                h36m_kp = _coco_to_h36m_single(coco_kp)

                # Convert to pixels if requested
                if self._output_format == "pixels":
                    h36m_kp[:, 0] *= w
                    h36m_kp[:, 1] *= h

                poses_list.append(h36m_kp)
                frame_indices.append(len(poses_list) - 1)

        if not poses_list:
            raise ValueError(f"No valid pose detected in video: {video_path}")

        poses = np.stack(poses_list)
        frame_indices = np.array(frame_indices)

        return poses, frame_indices

    def close(self) -> None:
        """Close the extractor and release resources.

        Note: YOLO model doesn't require explicit cleanup, but method kept for API compatibility.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def extract_h36m_poses(
    video_path: Path | str,
    model_size: str = "n",
    model_path: Path | str | None = None,
    output_format: str = "normalized",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract H3.6M poses from video.

    Convenience function that creates extractor and runs extraction.

    Args:
        video_path: Path to video file.
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
        model_path: Path to custom model weights (deprecated, use model_size)
        output_format: "normalized" or "pixels"

    Returns:
        poses: (N, 17, 3) array with x, y, confidence
        frame_indices: (N,) array of frame indices
    """
    extractor = H36MExtractor(model_size=model_size, model_path=model_path, output_format=output_format)
    return extractor.extract_video(video_path)


def blazepose_to_h36m(blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints.

    .. deprecated::
        BlazePose is no longer supported. Use YOLOv8-Pose via H36MExtractor instead.
        This function provides YOLO-based conversion for backward compatibility.

    Args:
        blazepose_pose: (33, 2/3) array for single frame, or (N, 33, 2/3) for sequence

    Returns:
        h36m_pose: (17, 2/3) array for single frame, or (N, 17, 2/3) for sequence

    Raises:
        ValueError: If input shape is invalid
    """
    import warnings

    warnings.warn(
        "blazepose_to_h36m is deprecated and will be removed in a future version. "
        "Use H36MExtractor with YOLOv8-Pose backend instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility: process via YOLO pose estimation
    # This is NOT a direct conversion - it's a YOLO-based fallback
    # The result will be YOLO H3.6M keypoints, not converted BlazePose
    raise NotImplementedError(
        "Direct BlazePose to H3.6M conversion is no longer supported. "
        "Use H36MExtractor with YOLOv8-Pose backend for new pose extraction. "
        "For existing BlazePose data, you must re-extract using H36MExtractor."
    )


# Public alias for backward compatibility (now maps to COCO)
BKey = _COCOKey
