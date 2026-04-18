"""Utility modules for video processing, geometry, and smoothing."""

from skating_ml.utils.gap_filling import GapFiller, GapReport
from skating_ml.utils.geometry import (
    angle_3pt,
    calculate_center_of_mass,
    calculate_com_trajectory,
    distance,
    get_mid_hip,
    get_mid_shoulder,
    normalize_poses,
    smooth_signal,
)
from skating_ml.utils.profiling import PipelineProfiler, StageTiming, profile_stage
from skating_ml.utils.smoothing import OneEuroFilter, OneEuroFilterConfig, PoseSmoother
from skating_ml.utils.subtitles import ElementEvent, SubtitleParser
from skating_ml.utils.video import VideoMeta, extract_frames, get_video_meta, open_video

__all__ = [
    "ElementEvent",
    "GapFiller",
    "GapReport",
    "OneEuroFilter",
    "OneEuroFilterConfig",
    "PipelineProfiler",
    "PoseSmoother",
    "StageTiming",
    "SubtitleParser",
    "VideoMeta",
    "angle_3pt",
    "calculate_center_of_mass",
    "calculate_com_trajectory",
    "distance",
    "extract_frames",
    "get_mid_hip",
    "get_mid_shoulder",
    "get_video_meta",
    "normalize_poses",
    "open_video",
    "profile_stage",
    "smooth_signal",
]
