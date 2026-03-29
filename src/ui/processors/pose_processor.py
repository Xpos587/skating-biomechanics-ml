"""Процессор извлечения поз из видео.

Pose extraction processor with H3.6M 17-keypoint format as primary output.
Uses H36MExtractor with integrated BlazePose-to-H3.6M conversion.
"""

from pathlib import Path

import numpy as np
import streamlit as st

from src.blade_edge_detector_3d import BladeEdgeDetector3D
from src.pose_3d.biomechanics_estimator import Biomechanics3DEstimator
from src.pose_3d.athletepose_extractor import AthletePose3DExtractor
from src.pose_estimation import H36MExtractor
from src.smoothing import PoseSmoother, get_skating_optimized_config
from src.ui.core.events import EventBus
from src.ui.types import ProcessedPoses
from src.video import extract_frames, get_video_meta


class PoseProcessor:
    """Извлечение поз из видео с H3.6M формат.

    Extract poses from video with H3.6M 17-keypoint format as primary output.
    """

    def __init__(self, events: EventBus | None = None) -> None:
        """Инициализация процессора поз.

        Args:
            events: Шина событий для оповещений.
        """
        self._events = events
        # H3.6M format extractor (17 keypoints directly)
        self._extractor = H36MExtractor(
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            num_poses=1,
            output_format="normalized",  # Normalized [0,1] for smoothing
        )

    def process(
        self,
        video_path: Path,
        enable_3d: bool = False,
        blade_3d: bool = False,
    ) -> ProcessedPoses:
        """Извлечь позы из видео.

        Args:
            video_path: Путь к видео файлу.
            enable_3d: Включить 3D оценку поз.
            blade_3d: Включить 3D детекцию ребра конька.

        Returns:
            ProcessedPoses с извлечёнными позами и метаданными.
        """
        # Load video metadata
        meta = get_video_meta(video_path)

        # Progress indicator
        progress_bar = st.progress(0, text="Извлечение поз...")
        status_text = st.empty()

        # Extract poses in H3.6M format directly (17 keypoints)
        poses_h36m, frame_indices = self._extractor.extract_video(video_path)

        status_text.text(
            f"Извлечено {len(poses_h36m)} поз из {meta.num_frames} кадров"
        )

        # Smooth in normalized space
        status_text.text("Сглаживание поз...")
        config = get_skating_optimized_config(fps=meta.fps)
        smoother = PoseSmoother(config=config, freq=meta.fps)

        poses_xy = poses_h36m[:, :, :2]  # (N, 17, 2)
        confidences = poses_h36m[:, :, 2:3]  # (N, 17, 1)

        poses_smoothed_norm = smoother.smooth(poses_xy)
        poses_smoothed_norm = np.clip(poses_smoothed_norm, 0.0, 1.0)

        # Add back confidence
        poses_final = np.zeros((len(poses_smoothed_norm), 17, 3), dtype=np.float32)
        poses_final[:, :, :2] = poses_smoothed_norm
        poses_final[:, :, 2] = confidences[:, :, 0]

        # Prepare result with H3.6M format as primary
        result = ProcessedPoses(
            poses_h36m=poses_final,
            pose_frame_indices=frame_indices,
            metadata={"video_path": str(video_path)},
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            num_frames=meta.num_frames,
        )

        # 3D pose estimation (uses H3.6M format directly)
        poses_3d = None
        if enable_3d:
            status_text.text("3D оценка поз...")
            # Select model path based on model_type
            model_paths = {
                "motionagformer-s": "data/models/motionagformer-s-ap3d.pth.tr",
                "tcpformer": "data/models/TCPFormer_ap3d_81.pth.tr",
            }
            model_path = model_paths.get(settings.model_3d_type, "data/models/motionagformer-s-ap3d.pth.tr")

            estimator = AthletePose3DExtractor(
                model_path=model_path,
                device="auto",
                model_type=settings.model_3d_type,
            )
            poses_3d = estimator.extract_sequence(poses_final)
            result.poses_3d = poses_3d

        # Blade detection (requires 3D)
        blade_states_left = None
        blade_states_right = None
        if blade_3d and poses_3d is not None:
            status_text.text("Детекция ребра конька...")
            detector = BladeEdgeDetector3D(fps=meta.fps)
            blade_states_left = []
            blade_states_right = []
            for i, pose_3d in enumerate(poses_3d):
                state_left = detector.detect_frame(pose_3d, i, foot="left")
                state_right = detector.detect_frame(pose_3d, i, foot="right")
                blade_states_left.append(state_left)
                blade_states_right.append(state_right)
            result.blade_states_left = blade_states_left
            result.blade_states_right = blade_states_right

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Publish event
        if self._events:
            self._events.publish("poses:extracted", result)

        return result
