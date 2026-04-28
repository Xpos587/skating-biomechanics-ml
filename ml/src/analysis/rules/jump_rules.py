"""Recommendation rules for jump elements.

These rules generate specific Russian recommendations for common jump errors.
"""

from ...types import RecommendationRule


def _is_bad(value: float, ref_range: tuple[float, float]) -> bool:
    """Check if value is outside acceptable range."""
    return not (ref_range[0] <= value <= ref_range[1])


# Common rules shared by all jumps — avoids duplication across element lists
_COMMON_JUMP_RULES = [
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
            "too_low": (
                "Колени слишком прямые при приземлении (угол {value:.0f}° вместо {target_min:.0f}-{target_max:.0f}°). "
                "Старайся приземляться на более согнутые колени для амортизации. "
                "Это защитит колени и улучшит баланс."
            ),
            "too_high": (
                "Чрезмерное сгибание коленей при приземлении (угол {value:.0f}° вместо {target_min:.0f}-{target_max:.0f}°). "
                "Выпрями ноги в момент касания льда."
            ),
            "default": "Следи за сгибанием коленей при приземлении.",
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
        metric_name="relative_jump_height",
        condition=_is_bad,
        priority=2,
        templates={
            "too_low": (
                "Недостаточная высота прыжка относительно длины тела ({value:.2f} вместо {target_min:.2f}-{target_max:.2f}). "
                "Работай над силой отталкивания: приседания, прыжки на двух ногах."
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
    RecommendationRule(
        metric_name="landing_com_velocity",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Жёсткое приземление (скорость CoM {value:.2f} norm/s, целевой диапазон {target_min:.2f}-{target_max:.2f}). "
                "Приземляйся мягче, амортизируя сгибанием коленей. "
                "Резкое торможение = плоское лезвие или зубец."
            ),
            "default": "Контролируй приземление.",
        },
    ),
    RecommendationRule(
        metric_name="landing_smoothness",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Нестабильное приземление (smoothness {value:.2f}, целевой {target_min:.2f}-{target_max:.2f}). "
                "Работай над балансом после выезда: удерживай центр тяжести над опорной ногой."
            ),
            "default": "Улучшай стабильность после приземления.",
        },
    ),
    RecommendationRule(
        metric_name="approach_torso_lean",
        condition=_is_bad,
        priority=2,
        templates={
            "too_low": (
                "Слишком сильный наклон назад при заходе ({value:.1f}°). "
                "Для этого прыжка держи торс более вертикально."
            ),
            "too_high": (
                "Слишком сильный наклон вперёд при заходе ({value:.1f}°). Проверь технику захода."
            ),
            "default": "Контролируй наклон торса при заходе.",
        },
    ),
    RecommendationRule(
        metric_name="toe_assist_proxy",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Приземление слишком резкое — возможно, приземляешься на зубец конька. "
                "Старайся касаться льда плавно, через ребро лезвия."
            ),
            "default": "Контролируй качество приземления.",
        },
    ),
    RecommendationRule(
        metric_name="hard_landing",
        condition=_is_bad,
        priority=1,
        templates={
            "too_low": (
                "Жесткое приземление. Работай над амортизацией: сгибай колени и бедра, приземляйся мягко."
            ),
            "default": "Контролируй мягкость приземления.",
        },
    ),
    RecommendationRule(
        metric_name="goe_score",
        condition=_is_bad,
        priority=3,
        templates={
            "too_low": (
                "Оценка качества элемента: {value:.1f}/10 (ниже {target_min:.1f}). "
                "Работай над: высотой, группировкой, приземлением, торсом."
            ),
            "default": "Улучшай общее качество элемента.",
        },
    ),
]


# Waltz jump rules
WALTZ_JUMP_RULES = [
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
    *_COMMON_JUMP_RULES,
]


# Toe loop rules
TOE_LOOP_RULES = [
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
    *_COMMON_JUMP_RULES,
]


# Flip rules
FLIP_RULES = [
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
    *_COMMON_JUMP_RULES,
]


# Salchow rules
SALCHOW_RULES = [*_COMMON_JUMP_RULES]


# Loop rules
LOOP_RULES = [*_COMMON_JUMP_RULES]


# Lutz rules
LUTZ_RULES = [*_COMMON_JUMP_RULES]


# Axel rules
AXEL_RULES = [*_COMMON_JUMP_RULES]
