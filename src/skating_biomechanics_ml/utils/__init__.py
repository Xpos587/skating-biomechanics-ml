"""Utility modules for video processing, geometry, subtitle parsing, smoothing, visualization, and blade edge detection."""

from skating_biomechanics_ml.utils.blade_edge_detector import (
    BladeEdgeDetector,
    BladeState,
    angle_with_horizontal,
    calculate_ankle_angle,
    calculate_foot_angle,
    calculate_foot_vector,
    calculate_motion_direction,
    calculate_vertical_acceleration,
)
from skating_biomechanics_ml.utils.geometry import (
    angle_3pt,
    distance,
    normalize_poses,
    smooth_signal,
)
from skating_biomechanics_ml.utils.smoothing import (
    OneEuroFilter,
    OneEuroFilterConfig,
    PoseSmoother,
    get_skating_optimized_config,
)
from skating_biomechanics_ml.utils.subtitles import ElementEvent, SubtitleParser
from skating_biomechanics_ml.utils.video import (
    extract_frames,
    get_video_meta,
    open_video,
    select_person_crop,
)
from skating_biomechanics_ml.utils.visualization import (
    draw_debug_hud,
    draw_edge_indicators,
    draw_skeleton,
    draw_subtitle_cyrillic,
    draw_text_box,
    draw_trails,
    draw_velocity_vectors,
)

__all__ = [
    # Video utilities
    "open_video",
    "get_video_meta",
    "extract_frames",
    "select_person_crop",
    # Geometry utilities
    "angle_3pt",
    "distance",
    "normalize_poses",
    "smooth_signal",
    # Smoothing utilities
    "OneEuroFilter",
    "OneEuroFilterConfig",
    "PoseSmoother",
    "get_skating_optimized_config",
    # Blade edge detection
    "BladeEdgeDetector",
    "BladeState",
    "calculate_foot_vector",
    "calculate_foot_angle",
    "calculate_ankle_angle",
    "calculate_vertical_acceleration",
    "calculate_motion_direction",
    "angle_with_horizontal",
    # Subtitle utilities
    "SubtitleParser",
    "ElementEvent",
    # Visualization utilities
    "draw_skeleton",
    "draw_velocity_vectors",
    "draw_trails",
    "draw_edge_indicators",
    "draw_subtitle_cyrillic",
    "draw_debug_hud",
    "draw_text_box",
]
