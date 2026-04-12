"""Shared pytest fixtures and configuration."""

from pathlib import Path

import numpy as np
import pytest

from skating_ml.types import H36Key


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
    """Create sample H3.6M 17-keypoint 3D pose for testing."""
    # 17 keypoints in simple T-pose configuration (x, y, z in meters)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)

    # Root/Hips
    keypoints[0, H36Key.HIP_CENTER, :] = [0.0, 0.0, 0.0]  # Origin
    keypoints[0, H36Key.RHIP, :] = [-0.1, 0.0, 0.0]
    keypoints[0, H36Key.LHIP, :] = [0.1, 0.0, 0.0]
    keypoints[0, H36Key.RKNEE, :] = [-0.1, -0.4, 0.0]
    keypoints[0, H36Key.LKNEE, :] = [0.1, -0.4, 0.0]
    keypoints[0, H36Key.RFOOT, :] = [-0.1, -0.8, 0.0]
    keypoints[0, H36Key.LFOOT, :] = [0.1, -0.8, 0.0]

    # Torso/Spine
    keypoints[0, H36Key.SPINE, :] = [0.0, 0.2, 0.0]
    keypoints[0, H36Key.THORAX, :] = [0.0, 0.3, 0.0]
    keypoints[0, H36Key.NECK, :] = [0.0, 0.4, 0.0]
    keypoints[0, H36Key.HEAD, :] = [0.0, 0.5, 0.0]

    # Arms (from thorax in H3.6M)
    keypoints[0, H36Key.RSHOULDER, :] = [-0.15, 0.3, 0.0]
    keypoints[0, H36Key.RELBOW, :] = [-0.2, 0.15, 0.0]
    keypoints[0, H36Key.RWRIST, :] = [-0.22, 0.0, 0.0]
    keypoints[0, H36Key.LSHOULDER, :] = [0.15, 0.3, 0.0]
    keypoints[0, H36Key.LELBOW, :] = [0.2, 0.15, 0.0]
    keypoints[0, H36Key.LWRIST, :] = [0.22, 0.0, 0.0]

    return keypoints


@pytest.fixture
def sample_normalized_poses():
    """Create sample normalized poses (T-pose, 3 frames) with 33 BlazePose keypoints."""
    poses = np.zeros((3, 33, 2), dtype=np.float32)

    # Centered at origin, scale = 1
    for i in range(3):
        # Hips at origin
        poses[i, H36Key.LEFT_HIP] = [-0.05, 0.0]
        poses[i, H36Key.RIGHT_HIP] = [0.05, 0.0]

        # Shoulders above (negative Y = up in image coords)
        poses[i, H36Key.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[i, H36Key.RIGHT_SHOULDER] = [0.1, -0.3]

        # Elbows
        poses[i, H36Key.LEFT_ELBOW] = [-0.15, -0.5]
        poses[i, H36Key.RIGHT_ELBOW] = [0.15, -0.5]

        # Wrists
        poses[i, H36Key.LEFT_WRIST] = [-0.2, -0.7]
        poses[i, H36Key.RIGHT_WRIST] = [0.2, -0.7]

        # Knees
        poses[i, H36Key.LEFT_KNEE] = [-0.05, 0.3]
        poses[i, H36Key.RIGHT_KNEE] = [0.05, 0.3]

        # Ankles
        poses[i, H36Key.LEFT_ANKLE] = [-0.05, 0.6]
        poses[i, H36Key.RIGHT_ANKLE] = [0.05, 0.6]

        # Heels and foot index (for edge detection)
        poses[i, H36Key.LEFT_HEEL] = [-0.08, 0.6]
        poses[i, H36Key.RIGHT_HEEL] = [0.08, 0.6]
        poses[i, H36Key.LEFT_FOOT_INDEX] = [-0.02, 0.65]
        poses[i, H36Key.RIGHT_FOOT_INDEX] = [0.02, 0.65]

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
