"""Shared data types for the skating biomechanics analysis pipeline.

This module defines all core data structures that flow between pipeline stages.
Types are annotated for mypy strict mode compatibility.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from skating_biomechanics_ml.utils.video import VideoMeta


class BKey(IntEnum):
    """BlazePose 33 keypoint indices.

    Based on Google MediaPipe BlazePose topology.
    https://google.github.io/mediapipe/solutions/pose.html
    """

    # Face
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    LEFT_INDEX = 18
    LEFT_THUMB = 19
    RIGHT_INDEX = 20
    RIGHT_PINKY = 21
    RIGHT_THUMB = 22

    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# All 33 BlazePose keypoints
BLAZEPOSE_INDICES = list(BKey)

# Skeleton edges for visualization
BLAZEPOSE_SKELETON_EDGES = [
    # Face
    (BKey.LEFT_EYE_INNER, BKey.LEFT_EYE),
    (BKey.LEFT_EYE, BKey.LEFT_EYE_OUTER),
    (BKey.RIGHT_EYE_INNER, BKey.RIGHT_EYE),
    (BKey.RIGHT_EYE, BKey.RIGHT_EYE_OUTER),
    (BKey.LEFT_EYE_OUTER, BKey.LEFT_EAR),
    (BKey.RIGHT_EYE_OUTER, BKey.RIGHT_EAR),
    (BKey.MOUTH_LEFT, BKey.MOUTH_RIGHT),
    # Upper body
    (BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER),
    (BKey.LEFT_SHOULDER, BKey.LEFT_ELBOW),
    (BKey.LEFT_ELBOW, BKey.LEFT_WRIST),
    (BKey.LEFT_WRIST, BKey.LEFT_THUMB),
    (BKey.LEFT_WRIST, BKey.LEFT_INDEX),
    (BKey.LEFT_WRIST, BKey.LEFT_PINKY),
    (BKey.RIGHT_SHOULDER, BKey.RIGHT_ELBOW),
    (BKey.RIGHT_ELBOW, BKey.RIGHT_WRIST),
    (BKey.RIGHT_WRIST, BKey.RIGHT_THUMB),
    (BKey.RIGHT_WRIST, BKey.RIGHT_INDEX),
    (BKey.RIGHT_WRIST, BKey.RIGHT_PINKY),
    # Lower body
    (BKey.LEFT_SHOULDER, BKey.LEFT_HIP),
    (BKey.RIGHT_SHOULDER, BKey.RIGHT_HIP),
    (BKey.LEFT_HIP, BKey.RIGHT_HIP),
    (BKey.LEFT_HIP, BKey.LEFT_KNEE),
    (BKey.LEFT_KNEE, BKey.LEFT_ANKLE),
    (BKey.LEFT_ANKLE, BKey.LEFT_HEEL),
    (BKey.LEFT_ANKLE, BKey.LEFT_FOOT_INDEX),
    (BKey.RIGHT_HIP, BKey.RIGHT_KNEE),
    (BKey.RIGHT_KNEE, BKey.RIGHT_ANKLE),
    (BKey.RIGHT_ANKLE, BKey.RIGHT_HEEL),
    (BKey.RIGHT_ANKLE, BKey.RIGHT_FOOT_INDEX),
]


FrameKeypoints = NDArray[np.float32]  # (num_frames, 33, 3) with x, y, confidence
NormalizedPose = NDArray[np.float32]  # (num_frames, 33, 2) with x, y in [0,1]
PixelPose = NDArray[np.float32]  # (num_frames, 33, 2) with x, y in pixels
TimeSeries = NDArray[np.float32]  # (num_frames,) time series data

__all__ = [
    "FrameKeypoints",
    "NormalizedPose",
    "PixelPose",
    "TimeSeries",
    "BKey",
    "BLAZEPOSE_INDICES",
    "BLAZEPOSE_SKELETON_EDGES",
    "BoundingBox",
    "VideoMeta",
    "ElementPhase",
    "MetricResult",
    "AnalysisReport",
    "ReferenceData",
    "ElementSegment",
    "SegmentationResult",
    "assert_pose_format",
    "normalize_pixel_poses",
    "pixelize_normalized_poses",
]


def assert_pose_format(
    poses: np.ndarray,
    expected_format: str = "normalized",
    width: int | None = None,
    height: int | None = None,
    context: str = "",
) -> None:
    """Validate pose array format at runtime.

    Helps catch coordinate system bugs early. Call this at function entry
    when pose format matters.

    Args:
        poses: Pose array to validate.
        expected_format: Either "normalized" (coords in [0,1]) or "pixel" (coords in pixels).
        width: Frame width (required for pixel format validation).
        height: Frame height (required for pixel format validation).
        context: Description of where validation is happening (for error messages).

    Raises:
        AssertionError: If poses don't match expected format.
        ValueError: If invalid parameters provided.

    Examples:
        >>> assert_pose_format(poses, "normalized", context="draw_velocity_vectors")
        >>> assert_pose_format(poses_px, "pixel", width=1920, height=1080, context="draw_skeleton")
    """
    if poses.ndim != 3 or poses.shape[1] != 33 or poses.shape[2] not in (2, 3):
        raise AssertionError(
            f"{context}: Invalid pose shape {poses.shape}. Expected (N, 33, 2) or (N, 33, 3)"
        )

    xy = poses[:, :, :2]  # Get x, y coordinates (drop confidence if present)

    if expected_format == "normalized":
        x_min, x_max = xy[:, :, 0].min(), xy[:, :, 0].max()
        y_min, y_max = xy[:, :, 1].min(), xy[:, :, 1].max()

        if x_min < -0.01 or x_max > 1.01 or y_min < -0.01 or y_max > 1.01:
            raise AssertionError(
                f"{context}: Expected normalized coords [0,1], got "
                f"x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]. "
                f"Did you pass pixel coords instead?"
            )

    elif expected_format == "pixel":
        if width is None or height is None:
            raise ValueError(f"{context}: width/height required for pixel format validation")

        x_min, x_max = xy[:, :, 0].min(), xy[:, :, 0].max()
        y_min, y_max = xy[:, :, 1].min(), xy[:, :, 1].max()

        # Pixel coords should be roughly within frame bounds
        # Allow small negative values (padding) and slightly over bounds
        if x_max < width * 0.1 or x_min > width * 0.9:
            raise AssertionError(
                f"{context}: Expected pixel coords for {width}x{height}, got "
                f"x=[{x_min:.1f}, {x_max:.1f}]. Did you pass normalized coords instead?"
            )
        if y_max < height * 0.1 or y_min > height * 0.9:
            raise AssertionError(
                f"{context}: Expected pixel coords for {width}x{height}, got "
                f"y=[{y_min:.1f}, {y_max:.1f}]. Did you pass normalized coords instead?"
            )
    else:
        raise ValueError(f"Invalid expected_format: {expected_format}. Use 'normalized' or 'pixel'")


def normalize_pixel_poses(poses_px: PixelPose, width: int, height: int) -> NormalizedPose:
    """Convert pixel coordinates to normalized [0,1].

    Args:
        poses_px: Poses in pixel coordinates (N, 33, 2) or (N, 33, 3).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Normalized poses (N, 33, 2) with coords in [0,1].

    Example:
        >>> poses_norm = normalize_pixel_poses(poses_px, width=1920, height=1080)
    """
    xy = poses_px[:, :, :2].copy()
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    return xy.astype(np.float32)


def pixelize_normalized_poses(poses_norm: NormalizedPose, width: int, height: int) -> PixelPose:
    """Convert normalized [0,1] coordinates to pixel coordinates.

    Args:
        poses_norm: Normalized poses (N, 33, 2) with coords in [0,1].
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Poses in pixel coordinates (N, 33, 2).

    Example:
        >>> poses_px = pixelize_normalized_poses(poses_norm, width=1920, height=1080)
    """
    xy = poses_norm.copy()
    xy[:, :, 0] *= width
    xy[:, :, 1] *= height
    return xy.astype(np.float32)


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box for person detection.

    Attributes:
        x1, y1: Top-left corner (normalized [0, 1] or pixels).
        x2, y2: Bottom-right corner.
        confidence: Detection confidence score.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        """Box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Box center (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def center_x(self) -> float:
        """Box center x coordinate."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Box center y coordinate."""
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Box area."""
        return self.width * self.height


@dataclass(frozen=True)
class VideoMeta:
    """Video file metadata.

    Attributes:
        path: Path to video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate.
        num_frames: Total number of frames.
    """

    path: Path
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def duration_sec(self) -> float:
        """Video duration in seconds."""
        return self.num_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class ElementPhase:
    """Phase boundaries for a skating element.

    All values are frame indices.

    Attributes:
        name: Element type name.
        start: First frame of the element.
        takeoff: Takeoff frame (0 for steps/turns).
        peak: Peak height frame (turn center for steps).
        landing: Landing frame (0 for steps/turns).
        end: Last frame of the element.
    """

    name: str
    start: int
    takeoff: int
    peak: int
    landing: int
    end: int

    @property
    def airtime_frames(self) -> int:
        """Airtime in frames."""
        return self.landing - self.takeoff

    @property
    def has_takeoff(self) -> bool:
        """Whether element has a takeoff phase."""
        return self.takeoff > 0

    def airtime_sec(self, fps: float) -> float:
        """Calculate airtime in seconds."""
        if self.takeoff == 0 or self.landing == 0:
            return 0.0
        return (self.landing - self.takeoff) / fps


@dataclass
class MetricResult:
    """Result of a single metric computation.

    Attributes:
        name: Metric identifier (e.g., "airtime", "max_height").
        value: Computed value.
        unit: Unit string (e.g., "s", "deg", "norm").
        is_good: Whether value falls within ideal range.
        reference_range: (min_good, max_good) ideal range.
    """

    name: str
    value: float
    unit: str
    is_good: bool
    reference_range: tuple[float, float]


@dataclass
class AnalysisReport:
    """Complete analysis report for a skating element.

    Attributes:
        element_type: Type of skating element analyzed.
        phases: Detected or manual phase boundaries.
        metrics: List of computed metric results.
        dtw_distance: DTW distance to reference (lower = better).
        recommendations: List of Russian recommendation strings.
        overall_score: Overall score [0, 10].
    """

    element_type: str
    phases: ElementPhase
    metrics: list[MetricResult]
    dtw_distance: float
    recommendations: list[str]
    overall_score: float

    def format(self) -> str:
        """Format report as readable Russian text."""
        lines = [
            "=" * 60,
            f"АНАЛИЗ: {self.element_type.upper()}",
            "=" * 60,
            "",
            "--- Фазы элемента ---",
            f"  Начало:     {self.phases.start}",
            f"  Отрыв:      {self.phases.takeoff}",
            f"  Пик:        {self.phases.peak}",
            f"  Приземление: {self.phases.landing}",
            f"  Конец:       {self.phases.end}",
            "",
            "--- Биомеханические метрики ---",
        ]

        for metric in self.metrics:
            status = "✓ ОК" if metric.is_good else "✗ ПЛОХО"
            lines.append(
                f"  {metric.name}: {metric.value:.2f} {metric.unit} [{status}] "
                f"(референс: {metric.reference_range[0]:.2f}-{metric.reference_range[1]:.2f})"
            )

        lines.extend(
            [
                "",
                "--- Сходство с референсом ---",
                f"  DTW-расстояние: {self.dtw_distance:.3f} (0 = идеально)",
                "",
                "--- РЕКОМЕНДАЦИИ ---",
            ]
        )

        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        lines.extend(
            [
                "",
                f"Общий балл: {self.overall_score:.1f} / 10",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


@dataclass(frozen=True)
class ReferenceData:
    """Stored reference data for an element.

    Attributes:
        element_type: Type of skating element.
        name: Reference name (e.g., "expert_waltz_jump").
        poses: Normalized pose sequence (num_frames, 17, 2).
        phases: Phase boundaries for this reference.
        fps: Original video frame rate.
        meta: Video metadata (optional).
        source: Source description (e.g., "YouTube: Expert Skater").
    """

    element_type: str
    name: str
    poses: NormalizedPose
    phases: ElementPhase
    fps: float
    meta: VideoMeta | None = None
    source: str = "unknown"

    def save(self, path: Path) -> None:
        """Save reference to .npz file."""
        np.savez(
            path,
            element_type=self.element_type,
            name=self.name,
            poses=self.poses,
            phases=(
                self.phases.name,
                self.phases.start,
                self.phases.takeoff,
                self.phases.peak,
                self.phases.landing,
                self.phases.end,
            ),
            fps=self.fps,
        )

    @classmethod
    def load(cls, path: Path) -> "ReferenceData":
        """Load reference from .npz file."""
        data = np.load(path)
        phase_name, start, takeoff, peak, landing, end = data["phases"]
        phases = ElementPhase(
            name=str(phase_name),
            start=int(start),
            takeoff=int(takeoff),
            peak=int(peak),
            landing=int(landing),
            end=int(end),
        )
        return cls(
            element_type=str(data["element_type"]),
            name=str(data["name"]),
            poses=data["poses"].astype(np.float32),
            phases=phases,
            fps=float(data["fps"]),
        )


@dataclass(frozen=True)
class ElementSegment:
    """A detected skating element segment from automatic segmentation.

    Attributes:
        element_type: Element identifier (e.g., 'waltz_jump', 'three_turn').
        start: First frame of the element.
        end: Last frame of the element.
        confidence: Detection confidence [0, 1].
        phases: ElementPhase boundaries (takeoff, peak, landing, etc.).
        metadata: Additional info (rotation count, edge type, motion stats).
    """

    element_type: str
    start: int
    end: int
    confidence: float
    phases: ElementPhase | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end - self.start


@dataclass
class SegmentationResult:
    """Result of automatic video segmentation into elements.

    Attributes:
        segments: List of detected element segments (ordered by time).
        video_path: Original video path.
        video_meta: Video metadata.
        method: Segmentation method used.
        confidence: Overall segmentation confidence.
    """

    segments: list[ElementSegment]
    video_path: Path
    video_meta: "VideoMeta"
    method: str
    confidence: float

    def get_timeline(self) -> str:
        """Return human-readable timeline of detected elements."""
        lines: list[str] = []
        lines.append(f"Segmentation: {self.method} (confidence: {self.confidence:.2f})")
        lines.append(f"Video: {self.video_path.name}")
        lines.append(f"Detected {len(self.segments)} elements:")
        lines.append("")

        for i, seg in enumerate(self.segments, 1):
            duration = seg.duration_frames / self.video_meta.fps
            lines.append(
                f"  {i}. {seg.element_type:20s} [{seg.start:4d}:{seg.end:4d}] "
                f"({duration:.2f}s) conf={seg.confidence:.2f}"
            )

        return "\n".join(lines)

    def export_segments_json(self, output_path: Path) -> None:
        """Export segmentation results as JSON for verification/editing."""
        import json

        data = {
            "video_path": str(self.video_path),
            "method": self.method,
            "confidence": self.confidence,
            "video_meta": {
                "fps": self.video_meta.fps,
                "num_frames": self.video_meta.num_frames,
                "duration_sec": self.video_meta.duration_sec,
            },
            "segments": [
                {
                    "element_type": s.element_type,
                    "start": s.start,
                    "end": s.end,
                    "confidence": s.confidence,
                    "duration_frames": s.duration_frames,
                    "duration_sec": s.duration_frames / self.video_meta.fps,
                }
                for s in self.segments
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
