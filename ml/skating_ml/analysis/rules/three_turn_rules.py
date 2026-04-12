"""Recommendation rules for three-turn elements.

These rules generate specific Russian recommendations for common three-turn errors.
"""

from ...types import RecommendationRule


def _is_bad(value: float, ref_range: tuple[float, float]) -> bool:
    """Check if value is outside acceptable range."""
    return not (ref_range[0] <= value <= ref_range[1])


# Three turn rules
THREE_TURN_RULES = [
    RecommendationRule(
        metric_name="trunk_lean",
        condition=_is_bad,
        priority=0,
        templates={
            "too_high": (
                "Чрезмерный наклон корпуса ({value:.1f}° вместо {target_min:.1f}-{target_max:.1f}°). "
                "Держи корпус прямо — это влияет на центр тяжести и точность вращения. "
                "Представь, что от головы до пят натянута струна."
            ),
            "too_low": ("Хорошее положение корпуса! Минимальный наклон."),
            "default": "Следи за положением корпуса.",
        },
    ),
    RecommendationRule(
        metric_name="edge_change_smoothness",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Резкая смена ребра (score {value:.2f}). "
                "Плавно переводи вес с ребра на ребро. Представь, что рисуешь дугу коньком."
            ),
            "too_high": ("Отличная плавность смены ребра!"),
            "default": "Работай над плавностью перехода.",
        },
    ),
    RecommendationRule(
        metric_name="knee_angle",
        condition=_is_bad,
        priority=2,
        templates={
            "too_high": (
                "Слишком прямые ноги ({value:.0f}° вместо {target_min:.0f}-{target_max:.0f}°). "
                "Сгибай колени для лучшего баланса и контроля."
            ),
            "too_low": ("Чрезмерное сгибание коленей. Попробуй более прямые ноги."),
            "default": "Контролируй сгибание коленей.",
        },
    ),
    RecommendationRule(
        metric_name="shoulder_stability",
        condition=_is_bad,
        priority=1,
        templates={
            "too_high": (
                "Плечи слишком подвижны во время поворота. Фиксируй плечи для лучшей оси."
            ),
            "default": "Держи плечи стабильно.",
        },
    ),
]
