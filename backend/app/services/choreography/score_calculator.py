"""IJS score calculation: TES, GOE, back-half bonus."""

from __future__ import annotations

from app.services.choreography.elements_db import ELEMENTS


def goe_factor(base_value: float) -> float:
    """GOE scaling factor based on element base value.

    ISU rules:
      BV < 2.0   → factor 0.5
      2.0 ≤ BV < 4.0 → factor 0.7
      BV ≥ 4.0   → factor 1.0
    """
    if base_value < 2.0:
        return 0.5
    if base_value < 4.0:
        return 0.7
    return 1.0


def calculate_goe_total(base_value: float, goe: int) -> float:
    """Calculate element total: base_value + GOE * factor.

    GOE range: -5 to +5.
    """
    clamped = max(-5, min(5, goe))
    return base_value + clamped * goe_factor(base_value)


def calculate_tes(
    elements: list[dict],
    back_half_indices: set[int] | frozenset[int],
) -> float:
    """Calculate Total Element Score.

    Args:
        elements: list of dicts with keys "code" and "goe".
        back_half_indices: set of element indices that qualify for back-half bonus (+10% BV).

    Returns:
        Sum of all element scores including back-half bonus.
    """
    total = 0.0
    for i, el in enumerate(elements):
        elem_def = ELEMENTS.get(el["code"])
        if elem_def is None:
            continue
        bv = elem_def.base_value
        if i in back_half_indices:
            bv *= 1.10  # back-half bonus
        total += calculate_goe_total(bv, el["goe"])
    return total
