"""CSP solver for choreography layout using random search + constraint filtering.

Generates valid ISU-compliant programs by placing elements on a timeline,
optimizing for maximum TES with back-half bonus.
"""

from __future__ import annotations

import random
from itertools import combinations

from app.services.choreography.elements_db import ELEMENTS
from app.services.choreography.rules_engine import validate_layout
from app.services.choreography.score_calculator import calculate_tes


def _parse_combination(combo_str: str) -> list[str]:
    """Parse combination string like '3Lz+2T' into list of codes."""
    return [c.strip() for c in combo_str.split("+")]


def _base_jump(code: str) -> str:
    """Extract base jump type (without rotation number) for Zayak tracking.

    '3Lz' -> 'Lz', '4T' -> 'T', '1Eu' -> 'Eu', '2A' -> 'A'
    """
    return code.lstrip("0123456789")


def _layout_fingerprint(elements: list[dict]) -> frozenset[str]:
    """Fingerprint layout by element codes (order-independent)."""
    return frozenset(e["code"] for e in elements)


def _score_layout(
    elements: list[dict],
    back_half_indices: set[int],
) -> float:
    """Score a layout with given back-half placement."""
    flat = [{"code": e["code"], "goe": e["goe"]} for e in elements]
    return calculate_tes(flat, back_half_indices)


def _generate_back_half_variants(
    base_elements: list[dict],
    segment: str,
    max_variants: int = 5,
) -> list[dict]:
    """Generate layouts with different back-half jump pass placements.

    Takes a base layout and produces variants where different jump passes
    receive the back-half bonus, yielding different TES scores.
    """
    jump_pass_indices = [i for i, e in enumerate(base_elements) if "jump_pass_index" in e]
    n_jp = len(jump_pass_indices)
    if n_jp < 3 or segment != "free_skate":
        return []

    # Generate distinct back-half selections: always last 3, plus other combos
    variants: list[dict] = []
    seen_bh: set[frozenset[int]] = set()

    # All possible ways to pick 3 jump passes for back-half bonus
    for combo in combinations(range(n_jp), 3):
        bh = frozenset(jump_pass_indices[i] for i in combo)
        if bh in seen_bh:
            continue
        seen_bh.add(bh)

        tes = _score_layout(base_elements, bh)
        variants.append(
            {
                "elements": [dict(e) for e in base_elements],
                "total_tes": round(tes, 2),
                "back_half_indices": sorted(bh),
            }
        )
        if len(variants) >= max_variants:
            break

    variants.sort(key=lambda c: c["total_tes"], reverse=True)
    return variants


def _generate_candidates(
    inventory: dict,
    segment: str,
) -> list[dict]:
    """Generate candidate layouts via constraint-guided random search.

    Builds jump passes with Zayak awareness: each base jump type appears
    at most twice, and if twice, at least once in a combination.
    Deduplicates by element fingerprint, keeping highest TES per unique combo.
    """
    jumps = inventory.get("jumps", [])
    spins = inventory.get("spins", [])
    combos = inventory.get("combinations", [])

    # Separate single jumps from combos; exclude 1Eu from singles (half-jump only for combos)
    single_jumps = [j for j in jumps if j != "1Eu"]
    combo_passes: list[list[str]] = []
    for c in combos:
        parsed = _parse_combination(c)
        if parsed:
            combo_passes.append(parsed)

    step_options = ["StSq4", "StSq3", "StSq2"]
    choreo_options = ["ChSq1"]

    # ISU limits
    min_passes = 3 if segment == "short_program" else 5
    max_passes = 7 if segment == "free_skate" else min(7, len(single_jumps) + len(combo_passes))

    # best[fp] = (tes, layout) — keep highest TES per unique element set
    best_by_fp: dict[frozenset[str], tuple[float, dict]] = {}

    for _ in range(500):
        # Shuffle options for variety
        shuffled_singles = list(single_jumps)
        random.shuffle(shuffled_singles)
        shuffled_combos = list(combo_passes)
        random.shuffle(shuffled_combos)

        num_passes = random.randint(min_passes, max(min_passes, max_passes))

        # Build passes with Zayak constraint: each base jump max 2 times
        selected_passes: list[list[str]] = []
        base_jump_counts: dict[str, int] = {}
        base_jump_in_combo: set[str] = set()

        # Collect all base jumps used in combos (for Zayak combo check)
        pool = shuffled_combos + [[j] for j in shuffled_singles]
        random.shuffle(pool)

        for pass_codes in pool:
            if len(selected_passes) >= num_passes:
                break

            # Get base jumps in this pass (excluding 1Eu)
            pass_bases = [_base_jump(c) for c in pass_codes if c != "1Eu"]

            # Check Zayak: each base jump max 2 total
            can_add = True
            for base in pass_bases:
                current = base_jump_counts.get(base, 0)
                if current >= 2:
                    can_add = False
                    break
                # If this would be the 2nd use, it must be in a combo
                if current == 1 and len(pass_codes) == 1:
                    can_add = False
                    break

            if not can_add:
                continue

            selected_passes.append(pass_codes)
            for base in pass_bases:
                base_jump_counts[base] = base_jump_counts.get(base, 0) + 1
            if len(pass_codes) > 1:
                base_jump_in_combo.update(pass_bases)

        if len(selected_passes) < min_passes:
            continue

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

        tes = _score_layout(elements, back_half)

        # Dedup by fingerprint — keep highest TES per unique element set
        fp = _layout_fingerprint(elements)
        if fp not in best_by_fp or tes > best_by_fp[fp][0]:
            best_by_fp[fp] = (
                tes,
                {
                    "elements": elements,
                    "total_tes": round(tes, 2),
                    "back_half_indices": sorted(back_half),
                },
            )

    candidates = [layout for _, layout in best_by_fp.values()]
    candidates.sort(key=lambda c: c["total_tes"], reverse=True)

    # Fallback: if < 10 unique combos, add back-half placement variants
    # Same element set but different jump passes in back-half → different TES
    if len(candidates) < 10:
        extended: list[dict] = list(candidates)
        seen_tes: set[float] = set()
        for c in candidates:
            if len(extended) >= 50:
                break
            variants = _generate_back_half_variants(c["elements"], segment, max_variants=3)
            for v in variants:
                if v["total_tes"] in seen_tes:
                    continue
                seen_tes.add(v["total_tes"])
                extended.append(v)
                if len(extended) >= 50:
                    break

        extended.sort(key=lambda c: c["total_tes"], reverse=True)
        return extended

    return candidates[:50]


def _generate_positions(n: int) -> list[dict]:
    """Generate rink positions for n elements on a 60x30m rink.

    Uses a simple Poisson-disk-like approach: jittered grid to avoid clustering.
    Elements are spread across the full rink surface with padding from edges.
    """
    random.seed()  # ensure different layout each call
    positions: list[dict] = []

    # Grid-based jittered sampling for even distribution
    cols = max(1, int(n**0.5 * 1.5))
    rows = max(1, (n + cols - 1) // cols)
    cell_w = 52.0 / cols  # rink inner area: 4..56 x 3..27
    cell_h = 24.0 / rows

    for i in range(n):
        row = i // cols
        col = i % cols
        x = 4.0 + col * cell_w + random.uniform(cell_w * 0.2, cell_w * 0.8)
        y = 3.0 + row * cell_h + random.uniform(cell_h * 0.2, cell_h * 0.8)
        x = round(min(max(x, 4.0), 56.0), 1)
        y = round(min(max(y, 3.0), 27.0), 1)
        positions.append({"x": x, "y": y})

    random.shuffle(positions)
    return positions


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

        # Assign rink positions
        positions = _generate_positions(n)
        for i, el in enumerate(elements):
            el["position"] = positions[i]

    return candidates[:num_layouts]
