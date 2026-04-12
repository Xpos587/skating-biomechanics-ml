"""Recommendation rules for jump elements.

These rules generate specific Russian recommendations for common jump errors.
"""

from ...types import RecommendationRule


def _is_bad(value: float, ref_range: tuple[float, float]) -> bool:
    """Check if value is outside acceptable range."""
    return not (ref_range[0] <= value <= ref_range[1])


# Waltz jump rules
WALTZ_JUMP_RULES = [
    RecommendationRule(
        metric_name="airtime",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": (
                "Недостаточное время полёта ({value:.2f}с вместо {target_min:.2f}-{target_max:.2f}с). "
                "Работай над отталкиванием: укрепляй икроножные мышцы (прыжки на скакалке, "
                "выпады, упражнения на степ-платформе)."
            ),
            "too_high": ("Отличное время полёта! Это выше референса, что хорошо для вращения."),
            "default": "Проверь технику отталкивания.",
        },
    ),
    RecommendationRule(
        metric_name="landing_knee_angle",
        condition=_is_bad,
        priority=1,
        templates={
            "too_high": (
                "Ноги слишком прямые при приземлении (угол {value:.0f}° вместо {target_min:.0f}-{target_max:.0f}°). "
                "Приземляйся с мягким сгибанием колен (целевой угол 100-120°) для амортизации. "
                "Это защитит колени и улучшит баланс."
            ),
            "too_low": (
                "Чрезмерное сгибание коленей при приземлении. Попробуй более контролируемое приземление."
            ),
            "default": "Следи за сгибанием коленей при приземлении.",
        },
    ),
    RecommendationRule(
        metric_name="arm_position_score",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": (
                "Руки 'разлетаются' во время прыжка (score {value:.2f}). "
                "Выпрями руки вперёд и делай тройки с зафиксированными руками. "
                "Руки служат балансиром — если они болтаются, теряется ось вращения."
            ),
            "too_high": ("Отличная позиция рук! Они хорошо зафиксированы."),
            "default": "Контролируй позицию рук.",
        },
    ),
    RecommendationRule(
        metric_name="max_height",
        condition=_is_bad,
        priority=2,
        templates={
            "too_low": (
                "Недостаточная высота прыжка ({value:.2f} вместо {target_min:.2f}-{target_max:.2f}). "
                "Работай над силой отталкивания: приседания, прыжки на двух ногах, "
                "упражнения на взрывную силу."
            ),
            "default": "Следи за высотой прыжка.",
        },
    ),
    RecommendationRule(
        metric_name="rotation_speed",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Недостаточная скорость вращения ({value:.0f}°/с вместо {target_min:.0f}-{target_max:.0f}°/с). "
                "Работай над группировкой в воздухе: руки ближе к телу, плотная группировка."
            ),
            "too_high": ("Отличная скорость вращения! Это выше референса."),
            "default": "Контролируй скорость вращения.",
        },
    ),
]

# Toe loop rules
TOE_LOOP_RULES = [
    RecommendationRule(
        metric_name="airtime",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": (
                "Недостаточное время полёта для перекидного ({value:.2f}с). "
                "Укрепляй толчковую ногу и работай над техникой зубцового захода."
            ),
            "default": "Проверь технику захода на зубец.",
        },
    ),
    RecommendationRule(
        metric_name="landing_knee_angle",
        condition=_is_bad,
        priority=1,
        templates={
            "too_high": ("Жёсткое приземление с прямыми ногами. Амортизируй сгибанием коленей!"),
            "default": "Мягкое приземление — ключ к безопасности.",
        },
    ),
    RecommendationRule(
        metric_name="toe_pick_timing",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": (
                "Слишком долгая подготовка к удару зубцом. Укороти время между заходом и отталкиванием."
            ),
            "too_high": ("Слишком резкий удар зубцом. Постепенно наращивай силу толчка."),
            "default": "Работай над таймингом зубцового удара.",
        },
    ),
]

# Flip rules
FLIP_RULES = [
    RecommendationRule(
        metric_name="airtime",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": (
                "Недостаточное время полёта для флипа ({value:.2f}с). "
                "Работай над маховым движением и силой отталкивания."
            ),
            "default": "Проверь технику махового движения.",
        },
    ),
    RecommendationRule(
        metric_name="pick_quality",
        condition=_is_bad,
        priority=0,
        templates={
            "too_low": ("Нечёткий удар зубцом при заходе на флип. Следи за точностью удара."),
            "default": "Контролируй точность зубцового удара.",
        },
    ),
    RecommendationRule(
        metric_name="air_position",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Расслабленная позиция в воздухе (score {value:.2f}). "
                "Группируй плотнее: руки прижаты к телу, ноги вместе."
            ),
            "default": "Работай над группировкой в воздухе.",
        },
    ),
]
