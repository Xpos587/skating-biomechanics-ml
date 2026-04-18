"""Shared data types for the skating biomechanics analysis pipeline.

This module defines all core data structures that flow between pipeline stages.
Types are annotated for mypy strict mode compatibility.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .video import VideoMeta  # type: ignore[import-untyped]


class H36Key(IntEnum):
    """H3.6M 17-keypoint indices (primary 3D format).

    Standard format for 3D human pose estimation.
    Used by AthletePose3D, Pose3DM, and most 3D pose estimators.

    References:
        - H3.6M dataset: Human3.6M (illion actions in 3.6 million frames)
        - Used by: Pose2Sim, OpenSim, Sports2D
    """

    # Root/Hips
    HIP_CENTER = 0  # Midpoint of hips (H3.6M convention)
    RHIP = 1
    RKNEE = 2
    RFOOT = 3
    LHIP = 4
    LKNEE = 5
    LFOOT = 6

    # Torso/Spine
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10

    # Arms
    LSHOULDER = 11
    LELBOW = 12
    LWRIST = 13
    RSHOULDER = 14
    RELBOW = 15
    RWRIST = 16

    # Convenience aliases using full names (LEFT_x, RIGHT_x)
    LEFT_HIP = LHIP
    RIGHT_HIP = RHIP
    LEFT_KNEE = LKNEE
    RIGHT_KNEE = RKNEE
    LEFT_FOOT = LFOOT
    RIGHT_FOOT = RFOOT
    LEFT_SHOULDER = LSHOULDER
    RIGHT_SHOULDER = RSHOULDER
    LEFT_ELBOW = LELBOW
    RIGHT_ELBOW = RELBOW
    LEFT_WRIST = LWRIST
    RIGHT_WRIST = RWRIST

    # Additional facial/hand keypoints (not in H3.6M 17kp, map to nearest)
    # Eyes/Ears/Mouth - map to HEAD
    LEFT_EYE = HEAD
    RIGHT_EYE = HEAD
    LEFT_EYE_INNER = HEAD
    LEFT_EYE_OUTER = HEAD
    RIGHT_EYE_INNER = HEAD
    RIGHT_EYE_OUTER = HEAD
    LEFT_EAR = HEAD
    RIGHT_EAR = HEAD
    MOUTH_LEFT = HEAD
    MOUTH_RIGHT = HEAD

    # Hands (fingers) - map to wrist
    LEFT_THUMB = LWRIST
    LEFT_INDEX = LWRIST
    LEFT_PINKY = LWRIST
    RIGHT_THUMB = RWRIST
    RIGHT_INDEX = RWRIST
    RIGHT_PINKY = RWRIST

    # Detailed foot keypoints - map to foot
    LEFT_ANKLE = LFOOT
    LEFT_HEEL = LFOOT
    LEFT_FOOT_INDEX = LFOOT
    RIGHT_ANKLE = RFOOT
    RIGHT_HEEL = RFOOT
    RIGHT_FOOT_INDEX = RFOOT

    # Nose - map to HEAD
    NOSE = HEAD


class BladeType(Enum):
    """Figure skating blade edge types.

    Based on BDA (Blade Discrimination Algorithm) research.
    Enhanced with 3D pose detection capabilities.
    """

    # Edge types
    INSIDE = "inside"  # inside edge
    OUTSIDE = "outside"  # outside edge
    FLAT = "flat"  # blade flat

    # Blade zone types
    TOE_PICK = "toe_pick"  # front of blade
    ROCKER = "rocker"  # middle arc
    HEEL = "heel"  # rear (rarely used)

    # Fallback
    UNKNOWN = "unknown"  # undetermined


class MotionDirection(Enum):
    """Direction of movement on ice.

    Relative to the skater's forward-facing direction.
    """

    FORWARD = "forward"  # Вперёд
    BACKWARD = "backward"  # Назад
    LEFT = "left"  # Влево (левое плечо ведущее)
    RIGHT = "right"  # Вправо (правое плечо ведущее)
    DIAGONAL_LEFT = "diagonal_left"  # Диагональ влево-вперёд
    DIAGONAL_RIGHT = "diagonal_right"  # Диагональ вправо-вперёд
    ROTATION_LEFT = "rotation_left"  # Вращение влево
    ROTATION_RIGHT = "rotation_right"  # Вращение вправо
    STATIONARY = "stationary"  # in-place (spin, steps in place)


@dataclass
class BladeState3D:
    """Enhanced blade state with 3D pose information.

    Attributes:
        blade_type: Detected blade edge/zone type.
        foot: Which foot (left/right).
        motion_direction: Direction of movement.
        foot_angle: Foot angle relative to motion (degrees).
        ankle_angle: Ankle flexion angle (degrees).
        knee_angle: Knee flexion angle (degrees, for weight distribution).
        vertical_accel: Vertical acceleration (norm units/s²).
        position_3d: 3D position of foot on ice (x, y, z).
        velocity_3d: 3D velocity vector (vx, vy, vz).
        confidence: Detection confidence [0, 1].
        frame_idx: Frame index in video.
    """

    blade_type: BladeType
    foot: str  # "left" or "right"
    motion_direction: MotionDirection
    foot_angle: float
    ankle_angle: float
    knee_angle: float
    vertical_accel: float
    position_3d: tuple[float, float, float]  # (x, y, z) on ice surface
    velocity_3d: tuple[float, float, float]  # (vx, vy, vz)
    confidence: float
    frame_idx: int


@dataclass
class IceTrace:
    """Ice trace - path of blade on ice surface.

    Attributes:
        foot: Which foot (left/right).
        points: List of 3D points (x, y, z) on ice surface.
        timestamps: Corresponding frame timestamps.
        blade_types: Blade type at each point.
    """

    foot: str
    points: list[tuple[float, float, float]]  # (x, y, z) positions
    timestamps: list[float]  # seconds
    blade_types: list[BladeType]  # blade state at each point


# All 17 H3.6M keypoints
H36M_INDICES = list(H36Key)


# Skeleton edges for H3.6M 17-keypoint format (3D-only pipeline)
H36M_SKELETON_EDGES = [
    # Torso/Spine (root → head)
    (H36Key.HIP_CENTER, H36Key.SPINE),
    (H36Key.SPINE, H36Key.THORAX),
    (H36Key.THORAX, H36Key.NECK),
    (H36Key.NECK, H36Key.HEAD),
    # Right leg
    (H36Key.HIP_CENTER, H36Key.RHIP),
    (H36Key.RHIP, H36Key.RKNEE),
    (H36Key.RKNEE, H36Key.RFOOT),
    # Left leg
    (H36Key.HIP_CENTER, H36Key.LHIP),
    (H36Key.LHIP, H36Key.LKNEE),
    (H36Key.LKNEE, H36Key.LFOOT),
    # Right arm
    (H36Key.THORAX, H36Key.RSHOULDER),
    (H36Key.RSHOULDER, H36Key.RELBOW),
    (H36Key.RELBOW, H36Key.RWRIST),
    # Left arm
    (H36Key.THORAX, H36Key.LSHOULDER),
    (H36Key.LSHOULDER, H36Key.LELBOW),
    (H36Key.LELBOW, H36Key.LWRIST),
]


# H3.6M 17-keypoint pose types (primary format for 3D-only pipeline)
Pose3D = NDArray[np.float32]  # (num_frames, 17, 3) with x, y, z in meters
H36MPose2D = NDArray[np.float32]  # (num_frames, 17, 2) with x, y normalized
H36MPose3D = NDArray[np.float32]  # (num_frames, 17, 3) with x, y, z in meters

# Type aliases for convenience
FrameKeypoints = Pose3D  # (num_frames, 17, 3) with x, y, z
NormalizedPose = H36MPose2D  # (num_frames, 17, 2) with x, y in [0,1]
PixelPose = NDArray[np.float32]  # (num_frames, 17, 2) with x, y in pixels
TimeSeries = NDArray[np.float32]  # (num_frames,) time series data

__all__ = [
    "H36M_INDICES",
    "H36M_SKELETON_EDGES",
    "AnalysisReport",
    "BladeType",
    "BoundingBox",
    "ElementPhase",
    "ElementSegment",
    "FrameKeypoints",
    "H36Key",
    "H36MPose2D",
    "H36MPose3D",
    "MetricResult",
    "NormalizedPose",
    "PersonClick",
    "PixelPose",
    "Pose3D",
    "RecommendationRule",
    "ReferenceData",
    "SegmentationResult",
    "TimeSeries",
    "TrackedExtraction",
    "VideoMeta",
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
    if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] not in (2, 3):
        raise AssertionError(
            f"{context}: Invalid pose shape {poses.shape}. Expected (N, 17, 2) or (N, 17, 3)"
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
        poses_px: Poses in pixel coordinates (N, 17, 2) or (N, 17, 3).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Normalized poses (N, 17, 2) with coords in [0,1].

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
        poses_norm: Normalized poses (N, 17, 2) with coords in [0,1].
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Poses in pixel coordinates (N, 17, 2).

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
        blade_summary_left: Left foot blade edge summary (optional).
        blade_summary_right: Right foot blade edge summary (optional).
        physics: Physics metrics (CoM trajectory, jump height, inertia, etc.).
        profiling: Pipeline profiler timing data (optional).
    """

    element_type: str
    phases: ElementPhase
    metrics: list[MetricResult]
    dtw_distance: float
    recommendations: list[str]
    overall_score: float
    blade_summary_left: dict[str, Any] = field(default_factory=dict)
    blade_summary_right: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    profiling: dict[str, Any] | None = None

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
            status = (
                "\u2713 \u041e\u041a" if metric.is_good else "\u2717 \u041f\u041b\u041e\u0425\u041e"
            )
            lines.append(
                f"  {metric.name}: {metric.value:.2f} {metric.unit} [{status}] "
                f"(референс: {metric.reference_range[0]:.2f}-{metric.reference_range[1]:.2f})"
            )

        lines.extend(
            [
                "",
                "--- \u0421\u0445\u043e\u0434\u0441\u0442\u0432\u043e \u0441 \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043e\u043c ---",
                f"  DTW-расстояние: {self.dtw_distance:.3f} (0 = идеально)",
                "",
                "--- РЕКОМЕНДАЦИИ ---",
            ]
        )

        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        # Blade edge information
        if self.blade_summary_left or self.blade_summary_right:
            lines.extend(
                [
                    "",
                    "--- Состояние лезвия ---",
                ]
            )
            if self.blade_summary_left:
                lines.append(
                    f"  Левая нога: {self.blade_summary_left.get('dominant_edge', 'unknown')}"
                )
                if "type_percentages" in self.blade_summary_left:
                    types_str = ", ".join(
                        f"{k}: {v:.1f}%"
                        for k, v in self.blade_summary_left["type_percentages"].items()
                    )
                    lines.append(f"    Распределение: {types_str}")
            if self.blade_summary_right:
                lines.append(
                    f"  Правая нога: {self.blade_summary_right.get('dominant_edge', 'unknown')}"
                )
                if "type_percentages" in self.blade_summary_right:
                    types_str = ", ".join(
                        f"{k}: {v:.1f}%"
                        for k, v in self.blade_summary_right["type_percentages"].items()
                    )
                    lines.append(f"    Распределение: {types_str}")

        # Physics information
        if self.physics:
            lines.extend(
                [
                    "",
                    "--- Физические метрики ---",
                ]
            )
            if "jump_height" in self.physics:
                h = self.physics["jump_height"]
                lines.append(f"  Высота прыжка (CoM): {h:.2f} м")
            if "flight_time" in self.physics:
                t = self.physics["flight_time"]
                lines.append(
                    f"  \u0412\u0440\u0435\u043c\u044f \u043f\u043e\u043b\u0451\u0442\u0430: {t:.2f} \u0441"
                )
            if "takeoff_velocity" in self.physics:
                v = self.physics["takeoff_velocity"]
                lines.append(
                    f"  \u0421\u043a\u043e\u0440\u043e\u0441\u0442\u044c \u043e\u0442\u0440\u044b\u0432\u0430: {v:.2f} \u043c/\u0441"
                )
            if "avg_inertia" in self.physics:
                i = self.physics["avg_inertia"]
                lines.append(f"  Средний момент инерции: {i:.3f} кг·м²")
            if "fit_quality" in self.physics:
                q = self.physics["fit_quality"]
                lines.append(f"  Качество траектории (R²): {q:.2f}")

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

        with output_path.open("w") as f:
            json.dump(data, f, indent=2)


# Recommendation system types


@dataclass(frozen=True)
class PersonClick:
    """A user click on a video frame to select a person.

    Stores pixel coordinates from a click event and provides
    conversion to normalized [0,1] coordinates.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
    """

    x: int
    y: int

    def to_normalized(self, w: int, h: int) -> tuple[float, float]:
        """Convert pixel coordinates to normalized [0,1].

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            (x_norm, y_norm) in [0, 1].
        """
        return (self.x / w, self.y / h)


@dataclass
class TrackedExtraction:
    """Pose extraction result with multi-person tracking support.

    Holds a pose sequence for a single tracked person across video frames.
    Missing frames (gaps) are represented as NaN values in the poses array.

    Attributes:
        poses: Pose array of shape (N, 17, 3) with NaN for missing frames.
        frame_indices: Real frame numbers, shape (N,), monotonically increasing.
        first_detection_frame: Index of first frame with a valid detection
            (pre-roll boundary, used by gap filler).
        target_track_id: The locked-on track ID, or None if tracking not used.
        fps: Video frame rate.
        video_meta: Full video metadata.
        first_frame: First video frame (BGR) cached for spatial reference.
    """

    poses: np.ndarray
    frame_indices: np.ndarray
    first_detection_frame: int
    target_track_id: int | None
    fps: float
    video_meta: VideoMeta
    first_frame: np.ndarray | None = None

    def valid_mask(self) -> np.ndarray:
        """Return boolean mask of frames with valid (non-NaN) poses.

        Returns:
            Boolean array of shape (N,) where True means the pose is valid.
        """
        return ~np.isnan(self.poses[:, 0, 0])


@dataclass(frozen=True)
class RecommendationRule:
    """A single recommendation rule.

    Attributes:
        metric_name: Name of the metric to check.
        condition: Callable that returns True if rule should trigger.
        priority: Lower = more critical (for sorting).
        templates: Mapping from severity to Russian text template.
                  Severity keys: "too_low", "too_high", "default".
                  Template variables: {value}, {unit}, {target_min}, {target_max}
    """

    metric_name: str
    condition: Callable[[float, tuple[float, float]], bool]
    priority: int
    templates: Mapping[str, str]
