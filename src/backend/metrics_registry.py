"""
Metric Registry — Single Source of Truth for Biomechanical Metrics.

Defines all available metrics, their display properties, and applicability
to different element types. Shared between backend and frontend.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricDef:
    """Definition of a biomechanical metric.

    Attributes:
        name: Unique metric identifier (snake_case)
        label_ru: Russian display label
        unit: Unit of measurement ("s", "deg", "score", "norm", "ratio", "deg/s")
        format: Python format spec (e.g., ".2f" for 2 decimal places)
        direction: "higher" = higher is better, "lower" = lower is better
        element_types: Tuple of element types this metric applies to
        ideal_range: (min, max) range for elite-level performance
    """

    name: str
    label_ru: str
    unit: Literal["s", "deg", "score", "norm", "ratio", "deg/s"]
    format: str
    direction: Literal["higher", "lower"]
    element_types: tuple[str, ...]
    ideal_range: tuple[float, float]


# Element type groups
JUMP_ELEMENTS = (
    "waltz_jump",
    "toe_loop",
    "flip",
    "salchow",
    "loop",
    "lutz",
    "axel",
)

ALL_ELEMENTS = JUMP_ELEMENTS + ("three_turn",)


# Metric registry
METRIC_REGISTRY: dict[str, MetricDef] = {
    # Jump-specific metrics
    "airtime": MetricDef(
        name="airtime",
        label_ru="Время полёта",
        unit="s",
        format=".2f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.3, 0.7),
    ),
    "max_height": MetricDef(
        name="max_height",
        label_ru="Высота прыжка",
        unit="norm",
        format=".3f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.2, 0.5),
    ),
    "relative_jump_height": MetricDef(
        name="relative_jump_height",
        label_ru="Относительная высота",
        unit="ratio",
        format=".2f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.3, 1.5),
    ),
    "landing_knee_angle": MetricDef(
        name="landing_knee_angle",
        label_ru="Угол колена при приземлении",
        unit="deg",
        format=".0f",
        direction="lower",
        element_types=JUMP_ELEMENTS,
        ideal_range=(90, 130),
    ),
    "landing_knee_stability": MetricDef(
        name="landing_knee_stability",
        label_ru="Стабильность приземления",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.5, 1.0),
    ),
    "landing_trunk_recovery": MetricDef(
        name="landing_trunk_recovery",
        label_ru="Восстановление корпуса",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.5, 1.0),
    ),
    "arm_position_score": MetricDef(
        name="arm_position_score",
        label_ru="Контроль рук",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(0.6, 1.0),
    ),
    "rotation_speed": MetricDef(
        name="rotation_speed",
        label_ru="Скорость вращения",
        unit="deg/s",
        format=".0f",
        direction="higher",
        element_types=JUMP_ELEMENTS,
        ideal_range=(300, 550),
    ),
    # Step-specific metrics
    "knee_angle": MetricDef(
        name="knee_angle",
        label_ru="Угол колена",
        unit="deg",
        format=".0f",
        direction="lower",
        element_types=("three_turn",),
        ideal_range=(100, 140),
    ),
    "trunk_lean": MetricDef(
        name="trunk_lean",
        label_ru="Наклон корпуса",
        unit="deg",
        format=".1f",
        direction="lower",
        element_types=("three_turn",),
        ideal_range=(-15, 20),
    ),
    "edge_change_smoothness": MetricDef(
        name="edge_change_smoothness",
        label_ru="Плавность смены ребра",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=("three_turn",),
        ideal_range=(0.1, 0.5),
    ),
    # Universal metrics
    "symmetry": MetricDef(
        name="symmetry",
        label_ru="Симметрия",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=ALL_ELEMENTS,
        ideal_range=(0.6, 1.0),
    ),
}


def get_metrics_for_element(element_type: str) -> dict[str, MetricDef]:
    """Return metrics applicable to a given element type.

    Args:
        element_type: Element type identifier (e.g., "waltz_jump", "three_turn")

    Returns:
        Dictionary mapping metric names to MetricDef objects

    Raises:
        ValueError: If element_type is not recognized
    """
    if element_type not in ALL_ELEMENTS:
        raise ValueError(
            f"Unknown element type: {element_type}. "
            f"Valid options: {ALL_ELEMENTS}"
        )

    return {
        metric_name: metric_def
        for metric_name, metric_def in METRIC_REGISTRY.items()
        if element_type in metric_def.element_types
    }
