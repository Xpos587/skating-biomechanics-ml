"""Типы данных для UI модуля.

Data classes for type-safe data passing between UI components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LayerSettings:
    """Настройки слоёв визуализации.

    Visualization layer settings for rendering.
    """

    # Basic layers
    skeleton: bool = True
    velocity: bool = True
    trails: bool = True
    edge_indicators: bool = True
    subtitles: bool = True

    # Advanced options
    enable_3d: bool = False
    model_3d_type: str = "motionagformer-s"  # "motionagformer-s" or "tcpformer"
    blade_3d: bool = False
    com_trajectory: bool = False
    floor_mode: bool = False

    # Parameters
    trail_length: int = 20
    d_3d_scale: float = 0.6
    no_3d_autoscale: bool = False

    # Font size
    font_size: int = 30


@dataclass
class ProcessedPoses:
    """Результаты извлечения поз из видео.

    Cached pose extraction results for reuse.

    H3.6M Migration Note:
        This now uses H3.6M 17-keypoint format as the primary format.
        The BlazePose 33kp format has been removed from the pipeline.
        All poses are stored in H3.6M format (normalized [0,1] coordinates).
    """

    poses_h36m: np.ndarray  # (N, 17, 3) H3.6M format, normalized [0,1] + confidence
    poses_3d: np.ndarray | None = None  # (N, 17, 3) 3D poses in meters
    blade_states_left: list | None = None
    blade_states_right: list | None = None
    pose_frame_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Video metadata
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    num_frames: int = 0

    @property
    def poses(self) -> np.ndarray:
        """Alias for poses_h36m (primary format)."""
        return self.poses_h36m

    @property
    def has_3d(self) -> bool:
        """Check if 3D poses are available."""
        return self.poses_3d is not None

    @property
    def has_blade_states(self) -> bool:
        """Check if blade states are available."""
        return (
            self.blade_states_left is not None
            and self.blade_states_right is not None
        )


@dataclass
class VideoSource:
    """Информация о загруженном видео.

    Video source information.
    """

    path: Path
    width: int
    height: int
    fps: float
    num_frames: int
    duration: float  # seconds

    @property
    def aspect_ratio(self) -> float:
        """Ширина / высота."""
        return self.width / self.height if self.height > 0 else 1.0
