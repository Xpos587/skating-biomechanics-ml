"""AI coach overlay panel for element feedback.

Pre-computes analysis results and renders a broadcast-style overlay
showing element name, key metrics, and recommendations in Russian.
"""

from dataclasses import dataclass

from src.analysis.element_defs import ELEMENT_DEFS
from src.types import ElementPhase, MetricResult


# Metric name translations (English key -> Russian display)
METRIC_NAMES_RU: dict[str, str] = {
    "airtime": "Время полёта",
    "max_height": "Высота",
    "landing_knee_angle": "Колено при приземлении",
    "rotation_speed": "Скорость вращения",
    "arm_position_score": "Позиция рук",
    "takeoff_angle": "Угол толчка",
    "trunk_lean": "Наклон тела",
    "knee_angle": "Угол колена",
    "edge_change_smoothness": "Плавность смены ребра",
    "symmetry": "Симметрия",
    "edge_quality": "Качество ребра",
    "pick_quality": "Качество зубца",
    "air_position": "Позиция в воздухе",
    "toe_pick_timing": "Тайминг зубца",
}

QUALITY_SYMBOLS = {
    True: "\u2713",   # Good
    False: "\u2717",  # Bad
}


@dataclass
class CoachOverlayData:
    """Data for rendering an AI coach overlay on video frames.

    Attributes:
        element_name_ru: Element name in Russian.
        metrics: List of (name_ru, formatted_value, is_good) tuples.
        recommendations: Top Russian recommendations.
        landing_frame: Frame where element lands (overlay starts here).
        fps: Video frame rate.
        display_duration: How long to show overlay in seconds.
    """

    element_name_ru: str
    metrics: list[tuple[str, str, bool]]
    recommendations: list[str]
    landing_frame: int
    fps: float
    display_duration: float = 4.0

    def is_visible_at(self, frame_idx: int) -> bool:
        """Check if overlay should be visible at given frame.

        Args:
            frame_idx: Current frame index.

        Returns:
            True if frame is within [landing, landing + duration*fps).
        """
        if frame_idx < self.landing_frame:
            return False
        end_frame = self.landing_frame + int(self.display_duration * self.fps)
        return frame_idx < end_frame


def _format_metric(metric: MetricResult) -> tuple[str, str, bool]:
    """Format a MetricResult into (name_ru, formatted_value, is_good).

    Args:
        metric: Computed metric result.

    Returns:
        Tuple of Russian name, formatted string value, and goodness flag.
    """
    name_ru = METRIC_NAMES_RU.get(metric.name, metric.name)

    if metric.unit == "s":
        value_str = f"{metric.value:.2f}\u0441"
    elif metric.unit == "deg":
        value_str = f"{metric.value:.0f}\u00b0"
    elif metric.unit == "norm":
        value_str = f"{metric.value:.2f}"
    elif metric.unit == "score":
        value_str = f"{metric.value:.2f}"
    else:
        value_str = f"{metric.value:.2f}{metric.unit}"

    return (name_ru, value_str, metric.is_good)


def compute_coach_overlays(
    phases: ElementPhase,
    metrics: list[MetricResult],
    recommendations: list[str],
    element_type: str,
    fps: float,
) -> list[CoachOverlayData]:
    """Pre-compute coach overlay data from analysis results.

    Args:
        phases: Detected element phases.
        metrics: Computed biomechanical metrics.
        recommendations: Russian text recommendations.
        element_type: Element type identifier.
        fps: Video frame rate.

    Returns:
        List of CoachOverlayData (empty if no jump phases detected).
    """
    # Only show for jumps (have takeoff/landing)
    if phases.takeoff == 0 and phases.landing == 0:
        return []

    # Get Russian element name
    element_def = ELEMENT_DEFS.get(element_type)
    element_name_ru = element_def.name_ru if element_def else element_type

    # Format metrics
    formatted_metrics = [_format_metric(m) for m in metrics]

    # Take top 3 recommendations
    top_recs = recommendations[:3]

    return [
        CoachOverlayData(
            element_name_ru=element_name_ru,
            metrics=formatted_metrics,
            recommendations=top_recs,
            landing_frame=phases.landing,
            fps=fps,
        )
    ]
