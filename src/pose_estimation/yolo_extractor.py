"""YOLO26-Pose extractor for figure skating analysis.

YOLO26-Pose provides fast, accurate 2D pose estimation with 17 keypoints (COCO format).
Advantages over BlazePose:
- Single-stage detection + pose (faster)
- No left/right confusion (better tracking)
- Easy API with ultralytics
"""

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except ImportError:
    YOLO = None  # type: ignore[assignment]


class YOLOPoseExtractor:
    """YOLO26-Pose wrapper for figure skating pose extraction.

    Advantages over BlazePose:
    - Single-stage detection + pose (faster)
    - No left/right confusion (better tracking)
    - Easy API with ultralytics

    Trade-offs:
    - 17 keypoints vs 33 (less detailed)
    - Fewer foot/hand keypoints
    """

    COCO_KEYPOINTS: ClassVar[list[str]] = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self, model_size: str = "n", model_path: Path | str | None = None):
        """Initialize YOLO26-Pose model.

        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
            model_path: Path to custom model weights, or None for default
        """
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Install with: uv add ultralytics")

        if model_path is not None:
            self.model = YOLO(str(model_path))
        else:
            model_name = f"yolo26{model_size}-pose.pt"
            self.model = YOLO(model_name)

        self.model_size = model_size

    def extract_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Extract pose from single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            pose: (17, 3) array with x, y, confidence coordinates in normalized [0,1]
                  Returns None if no person detected.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(frame_rgb, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Get first person detected
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # (17, 2)
        confidence = results[0].keypoints.conf.cpu().numpy()[0]  # (17,)

        # Normalize to [0, 1]
        h, w = frame.shape[:2]
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        # Combine x, y, confidence
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[:, :2] = keypoints
        pose[:, 2] = confidence

        return pose

    def extract_video(
        self,
        video_path: Path | str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract poses from video.

        Args:
            video_path: Path to video file

        Returns:
            poses: (N, 17, 3) array with x, y, confidence in H3.6M format
            frame_indices: (N,) array of frame indices
        """
        video_path = Path(video_path)

        # Run inference on video
        results = self.model(str(video_path), verbose=False, stream=True)

        poses = []
        frame_indices = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kp = result.keypoints.xy.cpu().numpy()[0]  # (17, 2)
                conf = result.keypoints.conf.cpu().numpy()[0]  # (17,)

                # Normalize
                h, w = result.orig_shape
                kp_norm = kp.copy()
                kp_norm[:, 0] /= w
                kp_norm[:, 1] /= h

                # Combine
                pose = np.zeros((17, 3), dtype=np.float32)
                pose[:, :2] = kp_norm
                pose[:, 2] = conf

                poses.append(pose)
                frame_indices.append(result.keypoints.data.get("frame_idx", len(poses) - 1))

        if len(poses) == 0:
            raise ValueError(f"No valid pose detected in video: {video_path}")

        return np.array(poses), np.array(frame_indices)

    def map_to_h36m_format(self, keypoints: np.ndarray) -> np.ndarray:
        """Map YOLO 17 keypoints to H3.6M format.

        Args:
            keypoints: (17, 2) or (17, 3) YOLO keypoints

        Returns:
            h36m_kp: (17, 2) or (17, 3) keypoints in H3.6M format
        """
        has_conf = keypoints.shape[1] == 3
        n_channels = 3 if has_conf else 2

        h36m_kp = np.zeros((17, n_channels), dtype=keypoints.dtype)

        # Map YOLO COCO to H3.6M indices
        # YOLO: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        # H3.6M: hip_center, rhip, rknee, rfoot, lhip, lknee, lfoot, spine, thorax, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist

        mapping = {
            # YOLO -> H3.6M
            0: 10,  # nose -> head
            5: 11,  # left_shoulder -> lshoulder
            6: 14,  # right_shoulder -> rshoulder
            7: 12,  # left_elbow -> lelbow
            8: 15,  # right_elbow -> relbow
            9: 13,  # left_wrist -> lwrist
            10: 16,  # right_wrist -> rwrist
            11: 4,  # left_hip -> lhip
            12: 1,  # right_hip -> rhip
            13: 5,  # left_knee -> lknee
            14: 2,  # right_knee -> rknee
            15: 6,  # left_ankle -> lfoot
            16: 3,  # right_ankle -> rfoot
        }

        # Compute hip_center (midpoint of hips)
        left_hip = keypoints[11, :2] if has_conf else keypoints[11]
        right_hip = keypoints[12, :2] if has_conf else keypoints[12]
        hip_center = (left_hip + right_hip) / 2

        # Map keypoints
        for yolo_idx, h36m_idx in mapping.items():
            h36m_kp[h36m_idx] = keypoints[yolo_idx]

        # Add computed keypoints
        h36m_kp[0] = hip_center  # hip_center

        # Add spine and thorax (computed)
        h36m_kp[7] = hip_center * 0.5 + (keypoints[5, :2] + keypoints[6, :2]) / 2 * 0.5  # spine
        h36m_kp[8] = (keypoints[5, :2] + keypoints[6, :2]) / 2  # thorax (mid-shoulder)

        return h36m_kp
