"""CSP solver for choreography layout using random search + constraint filtering.

Generates valid ISU-compliant programs by placing elements on a timeline,
optimizing for maximum TES with back-half bonus.
"""

from __future__ import annotations

import random

from backend.app.services.choreography.elements_db import ELEMENTS
from backend.app.services.choreography.rules_engine import validate_layout
from backend.app.services.choreography.score_calculator import calculate_tes


def _parse_combination(combo_str: str) -> list[str]:
    """Parse combination string like '3Lz+2T' into list of codes."""
    return [c.strip() for c in combo_str.split("+")]


def _generate_candidates(
    inventory: dict,
    segment: str,
) -> list[dict]:
    """Generate candidate layouts via constraint-guided random search."""
    jumps = inventory.get("jumps", [])
    spins = inventory.get("spins", [])
    combos = inventory.get("combinations", [])

    jump_pass_options: list[list[str]] = []
    for j in jumps:
        jump_pass_options.append([j])
    for c in combos:
        parsed = _parse_combination(c)
        if parsed:
            jump_pass_options.append(parsed)

    step_options = ["StSq4", "StSq3", "StSq2"]
    choreo_options = ["ChSq1"]

    candidates: list[dict] = []

    for _ in range(500):
        num_passes = min(7, len(jump_pass_options))
        if num_passes < 5:
            continue

        selected_passes = random.sample(jump_pass_options, num_passes)
        all_jump_codes: list[str] = []
        for jp in selected_passes:
            all_jump_codes.extend(jp)

        available_spins = [s for s in spins if s in ELEMENTS]
        if len(available_spins) < 3:
            available_spins = list({"CSp4", "LSp4", "FSp4"} | set(available_spins))
        selected_spins = random.sample(available_spins, min(3, len(available_spins)))

        selected_stsq = random.choice(step_options)
        selected_chsq = choreo_options[0]

        elements: list[dict] = []
        jump_idx = 0
        for jp in selected_passes:
            for j_idx, code in enumerate(jp):
                entry: dict = {"code": code, "goe": 0, "timestamp": 0.0}
                if j_idx == 0:
                    entry["jump_pass_index"] = jump_idx
                    jump_idx += 1
                elements.append(entry)

        for spin in selected_spins:
            elements.append({"code": spin, "goe": 0, "timestamp": 0.0})
        elements.append({"code": selected_stsq, "goe": 0, "timestamp": 0.0})
        elements.append({"code": selected_chsq, "goe": 0, "timestamp": 0.0})

        layout = {
            "discipline": "mens_singles",
            "segment": segment,
            "elements": elements,
        }
        result = validate_layout(layout)
        if not result.is_valid:
            continue

        jump_pass_count = sum(1 for e in elements if "jump_pass_index" in e)
        if segment == "free_skate" and jump_pass_count >= 3:
            back_half = set(range(jump_pass_count - 3, jump_pass_count))
        else:
            back_half = set()

        flat = [{"code": e["code"], "goe": e["goe"]} for e in elements]
        tes = calculate_tes(flat, back_half)

        candidates.append(
            {
                "elements": elements,
                "total_tes": round(tes, 2),
                "back_half_indices": sorted(back_half),
            }
        )

        if len(candidates) >= 50:
            break

    candidates.sort(key=lambda c: c["total_tes"], reverse=True)
    return candidates


def solve_layout(
    inventory: dict,
    music_features: dict,
    discipline: str,
    segment: str,
    num_layouts: int = 3,
) -> list[dict]:
    """Generate valid choreography layouts."""
    candidates = _generate_candidates(inventory, segment)

    duration = music_features.get("duration", 180.0)
    peaks = music_features.get("peaks", [])

    for layout in candidates:
        elements = layout["elements"]
        n = len(elements)
        if n == 0:
            continue

        for i, el in enumerate(elements):
            if peaks:
                target_time = (
                    peaks[i % len(peaks)]
                    if "jump_pass_index" in el
                    else (duration * (i + 1) / (n + 1))
                )
                target_time = min(target_time, duration - 5.0)
            else:
                target_time = duration * (i + 1) / (n + 1)
            el["timestamp"] = round(target_time, 1)
            el["goe"] = 0

    return candidates[:num_layouts]
