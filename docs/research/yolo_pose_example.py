"""
YOLOv11-Pose Usage Example for Figure Skating Biomechanics

This shows how to use YOLOv11-Pose as an alternative to BlazePose
for 2D pose estimation in the skating analysis pipeline.
"""

import numpy as np
from ultralytics import YOLO


class YOLOPoseExtractor:
    """
    YOLOv11-Pose wrapper for figure skating pose extraction.

    Advantages over BlazePose:
    - Single-stage detection + pose (faster)
    - No left/right confusion (better tracking)
    - Easy API with ultralytics

    Trade-offs:
    - 17 keypoints vs 33 (less detailed)
    - Fewer foot/hand keypoints
    """

    # YOLOv11-Pose 17 keypoints (COCO format)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    def __init__(self, model_size: str = "n"):
        """
        Initialize YOLOv11-Pose model.

        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium),
                       'l' (large), 'x' (xlarge)
        """
        model_name = f"yolov11{model_size}-pose.pt"
        self.model = YOLO(model_name)
        self.model_size = model_size

    def extract_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract pose from single frame.

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            keypoints: (17, 2) array of x, y coordinates
            confidence: (17,) array of confidence scores
        """
        results = self.model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None

        # Get first person detected
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # (17, 2)
        confidence = results[0].keypoints.conf.cpu().numpy()[0]  # (17,)

        return keypoints, confidence

    def extract_video(self, video_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract poses from entire video.

        Args:
            video_path: Path to video file

        Returns:
            poses: (N, 17, 2) array of keypoints
            confidences: (N, 17) array of confidence scores
        """
        results = self.model(video_path, verbose=False, stream=True)

        poses = []
        confidences = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kp = result.keypoints.xy.cpu().numpy()[0]  # (17, 2)
                conf = result.keypoints.conf.cpu().numpy()[0]  # (17,)
                poses.append(kp)
                confidences.append(conf)

        if len(poses) == 0:
            return None, None

        return np.array(poses), np.array(confidences)

    def map_to_blazepose_format(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Map YOLO 17 keypoints to BlazePose 33 format.

        This allows drop-in replacement with existing pipeline.

        Args:
            keypoints: (17, 2) YOLO keypoints

        Returns:
            blazepose_kp: (33, 2) BlazePose-format keypoints
        """
        # Initialize 33-keypoint array with NaN
        blazepose_kp = np.full((33, 2), np.nan)

        # Map YOLO keypoints to BlazePose indices
        mapping = {
            # YOLO -> BlazePose
            0: 0,  # nose -> nose
            1: 2,  # left_eye -> left_eye
            2: 5,  # right_eye -> right_eye
            3: 7,  # left_ear -> left_ear
            4: 8,  # right_ear -> right_ear
            5: 11,  # left_shoulder -> left_shoulder
            6: 12,  # right_shoulder -> right_shoulder
            7: 13,  # left_elbow -> left_elbow
            8: 14,  # right_elbow -> right_elbow
            9: 15,  # left_wrist -> left_wrist
            10: 16,  # right_wrist -> right_wrist
            11: 23,  # left_hip -> left_hip
            12: 24,  # right_hip -> right_hip
            13: 25,  # left_knee -> left_knee
            14: 26,  # right_knee -> right_knee
            15: 27,  # left_ankle -> left_ankle
            16: 28,  # right_ankle -> right_ankle
        }

        for yolo_idx, blazepose_idx in mapping.items():
            blazepose_kp[blazepose_idx] = keypoints[yolo_idx]

        # Estimate missing keypoints
        # Mid-hip (index 0)
        if not np.isnan(keypoints[11]).any() and not np.isnan(keypoints[12]).any():
            blazepose_kp[0] = (keypoints[11] + keypoints[12]) / 2

        # Mid-shoulder (index 1)
        if not np.isnan(keypoints[5]).any() and not np.isnan(keypoints[6]).any():
            blazepose_kp[1] = (keypoints[5] + keypoints[6]) / 2

        return blazepose_kp


# Integration example with existing pipeline
def replace_blazepose_with_yolo():
    """
    Example of how to integrate YOLOv11-Pose into existing pipeline.
    """

    # Initialize YOLO pose extractor
    yolo_extractor = YOLOPoseExtractor(model_size="n")  # Use nano for speed

    # Process video
    video_path = "data/test_video.mp4"
    poses, confidences = yolo_extractor.extract_video(video_path)

    if poses is not None:
        print(f"Extracted {len(poses)} poses from video")
        print(f"Pose shape: {poses.shape}")  # (N, 17, 2)
        print(f"Confidence shape: {confidences.shape}")  # (N, 17)

        # Map to BlazePose format for compatibility
        first_pose_blazepose = yolo_extractor.map_to_blazepose_format(poses[0])
        print(f"Mapped pose shape: {first_pose_blazepose.shape}")  # (33, 2)

        return poses, confidences
    else:
        print("No poses detected")
        return None, None


# Performance comparison
def benchmark():
    """
    Compare YOLOv11-Pose vs BlazePose performance.
    """
    import time

    # Test with dummy image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Benchmark YOLO
    yolo_extractor = YOLOPoseExtractor(model_size="n")

    start = time.time()
    for _ in range(100):
        yolo_extractor.extract_frame(test_image)
    yolo_time = (time.time() - start) / 100

    print(f"YOLOv11n-Pose: {yolo_time * 1000:.1f} ms/frame")
    print(f"Estimated FPS: {1 / yolo_time:.1f}")


if __name__ == "__main__":
    # Test extraction
    poses, confidences = replace_blazepose_with_yolo()

    # Run benchmark
    benchmark()
