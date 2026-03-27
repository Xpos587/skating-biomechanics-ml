"""Shared pytest fixtures and configuration."""

import numpy as np
import pytest
from pathlib import Path

from skating_biomechanics_ml.types import BKey


@pytest.fixture
def sample_frame():
    """Create a sample video frame (640x480x3 BGR)."""
    # Create gradient pattern
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        frame[i, :] = i * 255 // 480
    return frame


@pytest.fixture
def sample_keypoints():
    """Create sample BlazePose 33-keypoint pose for testing."""
    # 33 keypoints in simple T-pose configuration
    keypoints = np.zeros((1, 33, 3), dtype=np.float32)

    # Face (nose)
    keypoints[0, BKey.NOSE, :] = [320, 100, 0.9]

    # Shoulders
    keypoints[0, BKey.LEFT_SHOULDER, :] = [280, 200, 0.9]
    keypoints[0, BKey.RIGHT_SHOULDER, :] = [360, 200, 0.9]

    # Elbows
    keypoints[0, BKey.LEFT_ELBOW, :] = [260, 280, 0.8]
    keypoints[0, BKey.RIGHT_ELBOW, :] = [380, 280, 0.8]

    # Wrists
    keypoints[0, BKey.LEFT_WRIST, :] = [250, 350, 0.7]
    keypoints[0, BKey.RIGHT_WRIST, :] = [390, 350, 0.7]

    # Hips
    keypoints[0, BKey.LEFT_HIP, :] = [290, 350, 0.9]
    keypoints[0, BKey.RIGHT_HIP, :] = [350, 350, 0.9]

    # Knees
    keypoints[0, BKey.LEFT_KNEE, :] = [290, 420, 0.8]
    keypoints[0, BKey.RIGHT_KNEE, :] = [350, 420, 0.8]

    # Ankles
    keypoints[0, BKey.LEFT_ANKLE, :] = [290, 470, 0.7]
    keypoints[0, BKey.RIGHT_ANKLE, :] = [350, 470, 0.7]

    # Heels and foot index (for edge detection)
    keypoints[0, BKey.LEFT_HEEL, :] = [280, 470, 0.7]
    keypoints[0, BKey.RIGHT_HEEL, :] = [360, 470, 0.7]
    keypoints[0, BKey.LEFT_FOOT_INDEX, :] = [300, 480, 0.7]
    keypoints[0, BKey.RIGHT_FOOT_INDEX, :] = [340, 480, 0.7]

    return keypoints


@pytest.fixture
def sample_normalized_poses():
    """Create sample normalized poses (T-pose, 3 frames) with 33 BlazePose keypoints."""
    poses = np.zeros((3, 33, 2), dtype=np.float32)

    # Centered at origin, scale = 1
    for i in range(3):
        # Hips at origin
        poses[i, BKey.LEFT_HIP] = [-0.05, 0.0]
        poses[i, BKey.RIGHT_HIP] = [0.05, 0.0]

        # Shoulders above (negative Y = up in image coords)
        poses[i, BKey.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[i, BKey.RIGHT_SHOULDER] = [0.1, -0.3]

        # Elbows
        poses[i, BKey.LEFT_ELBOW] = [-0.15, -0.5]
        poses[i, BKey.RIGHT_ELBOW] = [0.15, -0.5]

        # Wrists
        poses[i, BKey.LEFT_WRIST] = [-0.2, -0.7]
        poses[i, BKey.RIGHT_WRIST] = [0.2, -0.7]

        # Knees
        poses[i, BKey.LEFT_KNEE] = [-0.05, 0.3]
        poses[i, BKey.RIGHT_KNEE] = [0.05, 0.3]

        # Ankles
        poses[i, BKey.LEFT_ANKLE] = [-0.05, 0.6]
        poses[i, BKey.RIGHT_ANKLE] = [0.05, 0.6]

        # Heels and foot index (for edge detection)
        poses[i, BKey.LEFT_HEEL] = [-0.08, 0.6]
        poses[i, BKey.RIGHT_HEEL] = [0.08, 0.6]
        poses[i, BKey.LEFT_FOOT_INDEX] = [-0.02, 0.65]
        poses[i, BKey.RIGHT_FOOT_INDEX] = [0.02, 0.65]

    return poses


@pytest.fixture
def temp_video_file(tmp_path: Path):
    """Create a temporary test video file path."""
    return tmp_path / "test_video.mp4"


@pytest.fixture
def temp_output_dir(tmp_path: Path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
