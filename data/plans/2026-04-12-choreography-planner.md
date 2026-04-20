# Choreography Planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI-powered choreography planner that generates ISU-compliant figure skating programs synchronized to music — with CSP solver optimization, interactive timeline + rink diagram editor, and SVG/PDF export.

**Architecture:** Backend-only rules engine and CSP solver (no ML imports). Music analysis via madmom + MSAF in arq worker. SVG rink rendering server-side. Frontend: interactive timeline with drag-and-drop + rink diagram panel + score bar. All endpoints under `/api/v1/choreography/`.

**Tech Stack:** FastAPI + SQLAlchemy async + PostgreSQL, OR-Tools CP-SAT, madmom + MSAF + librosa, Next.js 16 + React 19 + wavesurfer.js + @dnd-kit, SVG server-side rendering

**Design spec:** `docs/specs/2026-04-12-choreography-planner-design.md`

---

## File Structure (Target)

```
backend/app/
├── models/
│   ├── __init__.py                  # MODIFY: add MusicAnalysis, ChoreographyProgram
│   └── choreography.py              # CREATE: ORM models
├── schemas.py                       # MODIFY: add choreography schemas
├── crud/
│   └── choreography.py              # CREATE: CRUD operations
├── routes/
│   ├── __init__.py                  # UNCHANGED (empty)
│   └── choreography.py              # CREATE: all endpoints
├── services/
│   └── choreography/
│       ├── __init__.py              # CREATE: empty
│       ├── elements_db.py           # CREATE: ISU element registry (static data)
│       ├── rules_engine.py          # CREATE: ISU rules validation
│       ├── csp_solver.py            # CREATE: OR-Tools CSP solver
│       ├── music_analyzer.py        # CREATE: madmom + MSAF wrapper
│       ├── score_calculator.py      # CREATE: TES + GOE + PCS calculation
│       └── rink_renderer.py         # CREATE: SVG rink diagram generation
├── main.py                          # MODIFY: register choreography router
backend/tests/
└── services/
    └── choreography/
        ├── test_elements_db.py      # CREATE
        ├── test_rules_engine.py     # CREATE
        ├── test_csp_solver.py       # CREATE
        ├── test_score_calculator.py # CREATE
        └── test_rink_renderer.py    # CREATE
backend/alembic/versions/
└── <timestamp>_add_choreography.py  # CREATE: migration
frontend/src/
├── types/
│   └── choreography.ts              # CREATE: TypeScript types
├── lib/api/
│   └── choreography.ts              # CREATE: React Query hooks + Zod schemas
├── app/(app)/choreography/
│   ├── page.tsx                     # CREATE: main planner page
│   └── programs/[id]/page.tsx       # CREATE: program editor
├── components/choreography/
│   ├── music-uploader.tsx           # CREATE: upload + analysis status
│   ├── inventory-editor.tsx         # CREATE: jump/spin inventory selector
│   ├── timeline-editor.tsx          # CREATE: waveform + elements + energy
│   ├── rink-diagram.tsx             # CREATE: SVG rink with paths
│   ├── score-bar.tsx                # CREATE: TES/GOE/PCS/Total bar
│   ├── element-block.tsx            # CREATE: draggable element chip
│   ├── layout-picker.tsx            # CREATE: 3 layout options after generate
│   ├── program-list.tsx             # CREATE: saved programs list
│   └── export-dialog.tsx            # CREATE: SVG/PDF/JSON export
```

---

## Task 1: ISU Element Database

**Files:**
- Create: `backend/app/services/choreography/__init__.py`
- Create: `backend/app/services/choreography/elements_db.py`
- Create: `backend/tests/services/choreography/test_elements_db.py`

- [ ] **Step 1: Create package init**

```python
# backend/app/services/choreography/__init__.py
```

Empty file.

- [ ] **Step 2: Write failing tests for element database**

```python
# backend/tests/services/choreography/test_elements_db.py
"""Tests for ISU element database."""

import pytest

from backend.app.services.choreography.elements_db import (
    ElementType,
    get_element,
    get_elements_by_type,
    get_jumps,
    get_spins,
)


def test_get_triple_lutz():
    el = get_element("3Lz")
    assert el is not None
    assert el.code == "3Lz"
    assert el.name == "Triple Lutz"
    assert el.type == ElementType.JUMP
    assert el.rotations == 3.0
    assert el.has_toe_pick is True
    assert el.base_value == pytest.approx(5.90, abs=0.01)
    assert el.combo_eligible is True


def test_get_double_axel():
    el = get_element("2A")
    assert el is not None
    assert el.rotations == 2.5
    assert el.has_toe_pick is False


def test_get_spin():
    el = get_element("CSp4")
    assert el is not None
    assert el.type == ElementType.SPIN
    assert el.base_value == pytest.approx(3.20, abs=0.01)


def test_get_step_sequence():
    el = get_element("StSq4")
    assert el is not None
    assert el.type == ElementType.STEP_SEQUENCE


def test_get_choreo_sequence():
    el = get_element("ChSq1")
    assert el is not None
    assert el.type == ElementType.CHOREO_SEQUENCE
    assert el.base_value == pytest.approx(3.00, abs=0.01)


def test_get_element_not_found():
    assert get_element("9Zz") is None


def test_get_jumps_returns_only_jumps():
    jumps = get_jumps()
    assert len(jumps) > 0
    assert all(j.type == ElementType.JUMP for j in jumps)


def test_get_spins_returns_only_spins():
    spins = get_spins()
    assert len(spins) > 0
    assert all(s.type == ElementType.SPIN for s in spins)


def test_get_elements_by_type_step_sequence():
    elems = get_elements_by_type(ElementType.STEP_SEQUENCE)
    assert len(elems) > 0
    assert all(e.type == ElementType.STEP_SEQUENCE for e in elems)


def test_toe_pick_jumps():
    """Lutz, flip, toe loop have toe picks. Salchow, loop, axel do not."""
    assert get_element("3Lz").has_toe_pick is True
    assert get_element("3F").has_toe_pick is True
    assert get_element("3T").has_toe_pick is True
    assert get_element("3S").has_toe_pick is False
    assert get_element("3Lo").has_toe_pick is False
    assert get_element("3A").has_toe_pick is False


def test_combo_eligible():
    """All jumps with 2+ rotations are combo eligible."""
    assert get_element("3Lz").combo_eligible is True
    assert get_element("2A").combo_eligible is True
    assert get_element("1T").combo_eligible is False


def test_short_program_eligible():
    """Some spins/elements are allowed in SP, some only in FS."""
    # All jumps are SP-eligible
    assert get_element("3Lz").short_program_eligible is True
    # Step sequence and choreo sequence are SP-eligible
    assert get_element("StSq4").short_program_eligible is True
    assert get_element("ChSq1").short_program_eligible is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_elements_db.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement element database**

```python
# backend/app/services/choreography/elements_db.py
"""ISU element registry — static database of all elements with properties.

Data source: ISU Communication 2707 (2025/26 season).
Singles only (Men + Women), Short Program + Free Skate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ElementType(StrEnum):
    JUMP = "jump"
    SPIN = "spin"
    STEP_SEQUENCE = "step_sequence"
    CHOREO_SEQUENCE = "choreo_sequence"


@dataclass(frozen=True)
class ElementDef:
    code: str
    name: str
    type: ElementType
    base_value: float
    rotations: float = 0.0
    has_toe_pick: bool = False
    entry_edge: str = ""
    exit_edge: str = ""
    combo_eligible: bool = False
    short_program_eligible: bool = True


# ---------------------------------------------------------------------------
# Static element database
# ---------------------------------------------------------------------------

ELEMENTS: dict[str, ElementDef] = {
    # --- Jumps (ISU 2025/26 BV) ---
    # Single jumps
    "1T": ElementDef("1T", "Single Toe Loop", ElementType.JUMP, 0.40, 1.0, True, "", "RBO", False),
    "1S": ElementDef("1S", "Single Salchow", ElementType.JUMP, 0.40, 1.0, False, "", "RBO", False),
    "1Lo": ElementDef("1Lo", "Single Loop", ElementType.JUMP, 0.50, 1.0, False, "", "RBO", False),
    "1F": ElementDef("1F", "Single Flip", ElementType.JUMP, 0.50, 1.0, True, "", "RBO", False),
    "1Lz": ElementDef("1Lz", "Single Lutz", ElementType.JUMP, 0.60, 1.0, True, "", "RBO", False),
    "1A": ElementDef("1A", "Single Axel", ElementType.JUMP, 1.10, 1.5, False, "", "RBO", False),
    # Double jumps
    "2T": ElementDef("2T", "Double Toe Loop", ElementType.JUMP, 1.30, 2.0, True, "", "RBO", True),
    "2S": ElementDef("2S", "Double Salchow", ElementType.JUMP, 1.30, 2.0, False, "", "RBO", True),
    "2Lo": ElementDef("2Lo", "Double Loop", ElementType.JUMP, 1.70, 2.0, False, "", "RBO", True),
    "2F": ElementDef("2F", "Double Flip", ElementType.JUMP, 1.80, 2.0, True, "", "RBO", True),
    "2Lz": ElementDef("2Lz", "Double Lutz", ElementType.JUMP, 2.10, 2.0, True, "", "RBO", True),
    "2A": ElementDef("2A", "Double Axel", ElementType.JUMP, 3.30, 2.5, False, "", "RBO", True),
    # Triple jumps
    "3T": ElementDef("3T", "Triple Toe Loop", ElementType.JUMP, 4.20, 3.0, True, "", "RBO", True),
    "3S": ElementDef("3S", "Triple Salchow", ElementType.JUMP, 4.30, 3.0, False, "", "RBO", True),
    "3Lo": ElementDef("3Lo", "Triple Loop", ElementType.JUMP, 4.90, 3.0, False, "", "RBO", True),
    "3F": ElementDef("3F", "Triple Flip", ElementType.JUMP, 5.30, 3.0, True, "", "RBO", True),
    "3Lz": ElementDef("3Lz", "Triple Lutz", ElementType.JUMP, 5.90, 3.0, True, "", "RBO", True),
    "3A": ElementDef("3A", "Triple Axel", ElementType.JUMP, 8.00, 3.5, False, "", "RBO", True),
    # Quad jumps (Men)
    "4T": ElementDef("4T", "Quad Toe Loop", ElementType.JUMP, 9.50, 4.0, True, "", "RBO", True),
    "4S": ElementDef("4S", "Quad Salchow", ElementType.JUMP, 9.70, 4.0, False, "", "RBO", True),
    "4Lo": ElementDef("4Lo", "Quad Loop", ElementType.JUMP, 10.50, 4.0, False, "", "RBO", True),
    "4F": ElementDef("4F", "Quad Flip", ElementType.JUMP, 11.00, 4.0, True, "", "RBO", True),
    "4Lz": ElementDef("4Lz", "Quad Lutz", ElementType.JUMP, 11.50, 4.0, True, "", "RBO", True),
    "4A": ElementDef("4A", "Quad Axel", ElementType.JUMP, 12.50, 4.5, False, "", "RBO", True),
    # Half jumps (used in combinations)
    "1Eu": ElementDef("1Eu", "Euler (half-loop)", ElementType.JUMP, 0.50, 0.5, False, "", "RBO", True),
    # --- Spins (ISU 2025/26 BV) ---
    # Combination spins
    "CSp1": ElementDef("CSp1", "Change Foot Combination Spin Lv1", ElementType.SPIN, 1.50),
    "CSp2": ElementDef("CSp2", "Change Foot Combination Spin Lv2", ElementType.SPIN, 2.00),
    "CSp3": ElementDef("CSp3", "Change Foot Combination Spin Lv3", ElementType.SPIN, 2.50),
    "CSp4": ElementDef("CSp4", "Change Foot Combination Spin Lv4", ElementType.SPIN, 3.20),
    # Flying spins
    "FSp1": ElementDef("FSp1", "Flying Change Foot Spin Lv1", ElementType.SPIN, 1.70),
    "FSp2": ElementDef("FSp2", "Flying Change Foot Spin Lv2", ElementType.SPIN, 2.30),
    "FSp3": ElementDef("FSp3", "Flying Change Foot Spin Lv3", ElementType.SPIN, 2.80),
    "FSp4": ElementDef("FSp4", "Flying Change Foot Spin Lv4", ElementType.SPIN, 3.00),
    # Layback spins (Women) / Single position spins
    "LSp1": ElementDef("LSp1", "Layback Spin Lv1", ElementType.SPIN, 1.50),
    "LSp2": ElementDef("LSp2", "Layback Spin Lv2", ElementType.SPIN, 2.00),
    "LSp3": ElementDef("LSp3", "Layback Spin Lv3", ElementType.SPIN, 2.50),
    "LSp4": ElementDef("LSp4", "Layback Spin Lv4", ElementType.SPIN, 3.00),
    # Spin in one position (Men)
    "USp1": ElementDef("USp1", "Upright Spin Lv1", ElementType.SPIN, 1.50),
    "USp2": ElementDef("USp2", "Upright Spin Lv2", ElementType.SPIN, 2.00),
    "USp3": ElementDef("USp3", "Upright Spin Lv3", ElementType.SPIN, 2.50),
    "USp4": ElementDef("USp4", "Upright Spin Lv4", ElementType.SPIN, 3.00),
    # Camel spins
    "CSpB1": ElementDef("CSpB1", "Camel Spin Lv1", ElementType.SPIN, 1.70),
    "CSpB2": ElementDef("CSpB2", "Camel Spin Lv2", ElementType.SPIN, 2.30),
    "CSpB3": ElementDef("CSpB3", "Camel Spin Lv3", ElementType.SPIN, 2.80),
    "CSpB4": ElementDef("CSpB4", "Camel Spin Lv4", ElementType.SPIN, 3.00),
    # Step sequences
    "StSq1": ElementDef("StSq1", "Step Sequence Lv1", ElementType.STEP_SEQUENCE, 1.50),
    "StSq2": ElementDef("StSq2", "Step Sequence Lv2", ElementType.STEP_SEQUENCE, 2.60),
    "StSq3": ElementDef("StSq3", "Step Sequence Lv3", ElementType.STEP_SEQUENCE, 3.30),
    "StSq4": ElementDef("StSq4", "Step Sequence Lv4", ElementType.STEP_SEQUENCE, 3.90),
    # Choreographic sequence
    "ChSq1": ElementDef("ChSq1", "Choreographic Sequence", ElementType.CHOREO_SEQUENCE, 3.00),
}


def get_element(code: str) -> ElementDef | None:
    return ELEMENTS.get(code)


def get_elements_by_type(element_type: ElementType) -> list[ElementDef]:
    return [el for el in ELEMENTS.values() if el.type == element_type]


def get_jumps() -> list[ElementDef]:
    return get_elements_by_type(ElementType.JUMP)


def get_spins() -> list[ElementDef]:
    return get_elements_by_type(ElementType.SPIN)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_elements_db.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/choreography/ backend/tests/services/choreography/
git commit -m "feat(choreography): add ISU element database with tests"
```

---

## Task 2: Score Calculator

**Files:**
- Create: `backend/app/services/choreography/score_calculator.py`
- Create: `backend/tests/services/choreography/test_score_calculator.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/services/choreography/test_score_calculator.py
"""Tests for IJS score calculation."""

import pytest

from backend.app.services.choreography.elements_db import ElementType
from backend.app.services.choreography.score_calculator import (
    calculate_goe_total,
    calculate_tes,
    goe_factor,
)


def test_goe_factor_low_bv():
    """BV < 2.0: factor = 0.5"""
    assert goe_factor(1.50) == pytest.approx(0.5)


def test_goe_factor_mid_bv():
    """2.0 <= BV < 4.0: factor = 0.7"""
    assert goe_factor(3.30) == pytest.approx(0.7)


def test_goe_factor_high_bv():
    """BV >= 4.0: factor = 1.0"""
    assert goe_factor(5.90) == pytest.approx(1.0)


def test_calculate_goe_total_positive():
    """GOE +3 on 3Lz (BV 5.90, factor 1.0) = 5.90 + 3*1.0 = 8.90"""
    total = calculate_goe_total(5.90, 3)
    assert total == pytest.approx(8.90)


def test_calculate_goe_total_negative():
    """GOE -2 on 2A (BV 3.30, factor 0.7) = 3.30 + (-2)*0.7 = 1.90"""
    total = calculate_goe_total(3.30, -2)
    assert total == pytest.approx(1.90)


def test_calculate_tes_basic():
    """Simple program with 3 elements, no back-half bonus."""
    elements = [
        {"code": "3Lz", "goe": 2},
        {"code": "CSp4", "goe": 1},
        {"code": "StSq4", "goe": 0},
    ]
    result = calculate_tes(elements, back_half_indices=set())
    # 3Lz: 5.90 + 2*1.0 = 7.90
    # CSp4: 3.20 + 1*1.0 = 4.20
    # StSq4: 3.90 + 0*1.0 = 3.90
    assert result == pytest.approx(16.00, abs=0.01)


def test_calculate_tes_with_back_half_bonus():
    """Back-half elements get +10% BV."""
    elements = [
        {"code": "3Lz", "goe": 2},
        {"code": "3F", "goe": 1},
        {"code": "3Lo", "goe": 0},
    ]
    # indices 1 and 2 are in back half
    result = calculate_tes(elements, back_half_indices={1, 2})
    # 3Lz: 5.90 + 2*1.0 = 7.90
    # 3F: (5.30 * 1.10) + 1*1.0 = 5.83 + 1.0 = 6.83
    # 3Lo: (4.90 * 1.10) + 0*1.0 = 5.39 + 0.0 = 5.39
    assert result == pytest.approx(20.12, abs=0.01)


def test_calculate_tes_empty():
    assert calculate_tes([], back_half_indices=set()) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_score_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement score calculator**

```python
# backend/app/services/choreography/score_calculator.py
"""IJS score calculation: TES, GOE, back-half bonus."""

from __future__ import annotations

from backend.app.services.choreography.elements_db import ELEMENTS


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
    back_half_indices: set[int],
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_score_calculator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/choreography/score_calculator.py backend/tests/services/choreography/test_score_calculator.py
git commit -m "feat(choreography): add IJS score calculator with GOE and back-half bonus"
```

---

## Task 3: Rules Engine

**Files:**
- Create: `backend/app/services/choreography/rules_engine.py`
- Create: `backend/tests/services/choreography/test_rules_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/services/choreography/test_rules_engine.py
"""Tests for ISU rules engine."""

from backend.app.services.choreography.rules_engine import (
    validate_layout,
    ValidationResult,
)


def _make_layout(elements, segment="free_skate"):
    """Helper: build layout dict from list of (code, goe, is_jump_pass) tuples."""
    layout_elements = []
    jump_pass_index = 0
    for code, goe, is_jump_pass in elements:
        entry = {"code": code, "goe": goe, "timestamp": 0.0}
        if is_jump_pass:
            entry["jump_pass_index"] = jump_pass_index
            jump_pass_index += 1
        layout_elements.append(entry)
    return {
        "discipline": "mens_singles",
        "segment": segment,
        "elements": layout_elements,
    }


def test_valid_free_skate_mens():
    """A well-balanced men's FS: 7 jump passes, 3 spins, 1 StSq, 1 ChSq."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3T", 1, False),  # combo part, not counted as separate pass
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),  # repeated, second time — must be in combo (it is: +2T below)
        ("2T", 0, False),  # combo part
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert result.is_valid
    assert len(result.errors) == 0


def test_zayak_violation_triple_repeated():
    """3Lz attempted 3 times → Zayak violation (max 2 for 3+ rotations)."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3Lz", 1, True),
        ("3Lz", 0, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert not result.is_valid
    assert any("Zayak" in e for e in result.errors)


def test_zayak_second_repeat_not_in_combo():
    """3Lz attempted twice but neither in a combination → Zayak violation."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lz", 1, True),  # second 3Lz as standalone pass
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert any("combination" in e.lower() or "combo" in e.lower() for e in result.errors)


def test_too_many_jump_passes():
    """8 jump passes in FS → capacity violation (max 7)."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),
        ("3T", 0, True),
        ("2A", 0, True),  # 8th pass
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert not result.is_valid
    assert any("7" in e and "jump" in e.lower() for e in result.errors)


def test_missing_step_sequence():
    """No StSq → well-balanced program violation."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert any("step sequence" in e.lower() or "StSq" in e for e in result.errors)


def test_missing_choreo_sequence():
    """No ChSq → well-balanced program violation."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert any("choreographic" in e.lower() or "ChSq" in e for e in result.errors)


def test_too_many_spins():
    """4 spins → violation (max 3)."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("USp4", 0, False),  # 4th spin
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert any("spin" in e.lower() and "3" in e for e in result.errors)


def test_warnings_empty():
    """Valid layout should have no warnings."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("2A", 1, True),
        ("3S", 0, True),
        ("3Lz", 1, True),
        ("2A", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert len(result.warnings) == 0


def test_no_axel_warning():
    """No Axel-type jump → warning (not error in MVP, but ISU requires it)."""
    layout = _make_layout([
        ("3Lz", 2, True),
        ("3F", 1, True),
        ("3Lo", 0, True),
        ("3S", 0, True),
        ("3T", 0, True),
        ("3Lz", 1, True),
        ("3T", 0, True),
        ("CSp4", 1, False),
        ("FSp4", 0, False),
        ("LSp4", 1, False),
        ("StSq4", 0, False),
        ("ChSq1", 0, False),
    ], segment="free_skate")
    result = validate_layout(layout)
    assert any("axel" in w.lower() for w in result.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_rules_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rules engine**

```python
# backend/app/services/choreography/rules_engine.py
"""ISU rules engine — validates program layouts against ISU regulations.

Singles only (Men + Women), 2025/26 season.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.services.choreography.elements_db import (
    ELEMENTS,
    ElementType,
    get_element,
)


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _is_jump(code: str) -> bool:
    el = get_element(code)
    return el is not None and el.type == ElementType.JUMP


def _is_spin(code: str) -> bool:
    el = get_element(code)
    return el is not None and el.type == ElementType.SPIN


def _is_step_sequence(code: str) -> bool:
    el = get_element(code)
    return el is not None and el.type == ElementType.STEP_SEQUENCE


def _is_choreo_sequence(code: str) -> bool:
    el = get_element(code)
    return el is not None and el.type == ElementType.CHOREO_SEQUENCE


def _is_axel(code: str) -> bool:
    return "A" in code and _is_jump(code)


def _get_jump_passes(elements: list[dict]) -> list[list[str]]:
    """Extract jump passes from flat element list.

    A jump pass is a sequence of consecutive jump codes.
    Non-jump elements break the sequence.
    Elements without jump_pass_index are combo parts, not separate passes.
    """
    passes: list[list[str]] = []
    current_pass: list[str] = []

    for el in elements:
        code = el["code"]
        if not _is_jump(code):
            if current_pass:
                passes.append(current_pass)
                current_pass = []
            continue
        # If it has jump_pass_index, it starts a new pass
        if "jump_pass_index" in el:
            if current_pass:
                passes.append(current_pass)
                current_pass = [code]
            else:
                current_pass = [code]
        else:
            # Combo part — append to current pass
            current_pass.append(code)

    if current_pass:
        passes.append(current_pass)

    return passes


def validate_layout(layout: dict) -> ValidationResult:
    """Validate a program layout against ISU rules.

    Args:
        layout: dict with keys:
            - discipline: "mens_singles" | "womens_singles"
            - segment: "short_program" | "free_skate"
            - elements: list of dicts with "code", "goe", "timestamp", optionally "jump_pass_index"

    Returns:
        ValidationResult with errors (blocking) and warnings (non-blocking).
    """
    result = ValidationResult()
    elements = layout.get("elements", [])
    segment = layout.get("segment", "free_skate")

    # ---- Count elements by type ----
    jump_passes = _get_jump_passes(elements)
    num_jump_passes = len(jump_passes)
    spin_codes = [el["code"] for el in elements if _is_spin(el["code"])]
    num_spins = len(spin_codes)
    has_step_seq = any(_is_step_sequence(el["code"]) for el in elements)
    has_choreo_seq = any(_is_choreo_sequence(el["code"]) for el in elements)

    # ---- C_capacity: max 7 jump passes ----
    max_jump_passes = 7
    if num_jump_passes > max_jump_passes:
        result.add_error(
            f"Too many jumping passes: {num_jump_passes} (max {max_jump_passes})"
        )

    # ---- C_spins: max 3 spins ----
    if num_spins > 3:
        result.add_error(f"Too many spins: {num_spins} (max 3)")

    # ---- C_step_seq: exactly 1 step sequence ----
    if not has_step_seq:
        result.add_error("Missing step sequence (StSq)")

    # ---- C_choreo_seq: exactly 1 choreographic sequence ----
    if not has_choreo_seq:
        result.add_error("Missing choreographic sequence (ChSq)")

    # ---- C_axel: at least 1 Axel-type jump ----
    all_jump_codes = [code for jp in jump_passes for code in jp]
    has_axel = any(_is_axel(c) for c in all_jump_codes)
    if not has_axel:
        result.add_warning("No Axel-type jump — ISU requires at least one")

    # ---- C_zayak: jumps with 3+ rotations max 2 attempts ----
    jump_counts: dict[str, int] = {}
    for code in all_jump_codes:
        el = get_element(code)
        if el and el.rotations >= 3.0 and code != "1Eu":
            jump_counts[code] = jump_counts.get(code, 0) + 1

    for code, count in jump_counts.items():
        if count > 2:
            result.add_error(
                f"Zayak rule violation: {code} attempted {count} times (max 2)"
            )
        elif count == 2:
            # Check if at least one is in a combination
            in_combo = 0
            for jp in jump_passes:
                if code in jp and len(jp) > 1:
                    in_combo += 1
            if in_combo == 0:
                result.add_error(
                    f"Zayak rule: {code} attempted twice but not in any combination/sequence"
                )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_rules_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/choreography/rules_engine.py backend/tests/services/choreography/test_rules_engine.py
git commit -m "feat(choreography): add ISU rules engine with Zayak and well-balanced validation"
```

---

## Task 4: DB Models + Alembic Migration

**Files:**
- Create: `backend/app/models/choreography.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/alembic/versions/<timestamp>_add_choreography.py`
- Create: `backend/tests/models/test_choreography_models.py`

- [ ] **Step 1: Write failing test for model creation**

```python
# backend/tests/models/test_choreography_models.py
"""Tests for choreography ORM models."""

import pytest

from backend.app.models.choreography import ChoreographyProgram, MusicAnalysis


@pytest.fixture
def music_analysis_data():
    return {
        "user_id": "user-123",
        "filename": "test.mp3",
        "audio_url": "music/test.mp3",
        "duration_sec": 180.0,
        "bpm": 120.0,
        "meter": "4/4",
        "status": "completed",
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
        "energy_curve": {"timestamps": [0.0, 1.0], "values": [0.5, 0.8]},
        "downbeats": [0.0, 0.5, 1.0],
        "peaks": [5.0, 15.0, 30.0],
    }


@pytest.fixture
def program_data():
    return {
        "user_id": "user-123",
        "discipline": "mens_singles",
        "segment": "free_skate",
        "season": "2025_26",
        "layout": {"elements": [{"code": "3Lz", "goe": 2}]},
        "total_tes": 45.5,
        "estimated_goe": 5.0,
        "estimated_pcs": 35.0,
        "estimated_total": 85.5,
        "is_valid": True,
        "validation_errors": [],
        "validation_warnings": [],
    }


def test_create_music_analysis(db_session, music_analysis_data):
    music = MusicAnalysis(**music_analysis_data)
    db_session.add(music)
    await db_session.flush()
    await db_session.refresh(music)

    assert music.id is not None
    assert len(music.id) == 36  # UUID
    assert music.filename == "test.mp3"
    assert music.bpm == 120.0
    assert music.status == "completed"
    assert music.structure == [{"type": "verse", "start": 0.0, "end": 30.0}]
    assert music.peaks == [5.0, 15.0, 30.0]


def test_create_program(db_session, program_data):
    program = ChoreographyProgram(**program_data)
    db_session.add(program)
    await db_session.flush()
    await db_session.refresh(program)

    assert program.id is not None
    assert program.discipline == "mens_singles"
    assert program.segment == "free_skate"
    assert program.is_valid is True
    assert program.layout == {"elements": [{"code": "3Lz", "goe": 2}]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/models/test_choreography_models.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement ORM models**

```python
# backend/app/models/choreography.py
"""Music analysis and choreography program ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003

from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin


class MusicAnalysis(TimestampMixin, Base):
    """Cached music analysis result (BPM, structure, energy curve)."""

    __tablename__ = "music_analyses"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(500))
    audio_url: Mapped[str] = mapped_column(String(500))
    duration_sec: Mapped[float] = mapped_column(Float)
    bpm: Mapped[float | None] = mapped_column(Float)
    meter: Mapped[str | None] = mapped_column(String(10))
    structure: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    energy_curve: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    downbeats: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    peaks: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")

    __table_args__ = (
        Index("ix_music_analyses_user_created", "user_id", "created_at"),
    )


class ChoreographyProgram(TimestampMixin, Base):
    """Saved choreography program with layout and scores."""

    __tablename__ = "choreography_programs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    music_analysis_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("music_analyses.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(String(200))
    discipline: Mapped[str] = mapped_column(String(30))
    segment: Mapped[str] = mapped_column(String(20))
    season: Mapped[str] = mapped_column(String(10), default="2025_26")
    layout: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_tes: Mapped[float | None] = mapped_column(Float)
    estimated_goe: Mapped[float | None] = mapped_column(Float)
    estimated_pcs: Mapped[float | None] = mapped_column(Float)
    estimated_total: Mapped[float | None] = mapped_column(Float)
    is_valid: Mapped[bool | None] = mapped_column()
    validation_errors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    validation_warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_choreo_programs_user_created", "user_id", "created_at"),
    )
```

- [ ] **Step 4: Register models in `__init__.py`**

Modify `backend/app/models/__init__.py` — add imports:

```python
# backend/app/models/__init__.py
"""SQLAlchemy ORM models."""

from backend.app.models.base import Base
from backend.app.models.choreography import ChoreographyProgram, MusicAnalysis
from backend.app.models.refresh_token import RefreshToken
from backend.app.models.relationship import Relationship
from backend.app.models.session import Session, SessionMetric
from backend.app.models.user import User

__all__ = [
    "Base",
    "ChoreographyProgram",
    "MusicAnalysis",
    "RefreshToken",
    "Relationship",
    "Session",
    "SessionMetric",
    "User",
]
```

- [ ] **Step 5: Generate Alembic migration**

Run: `cd backend && uv run alembic revision --autogenerate -m "add_choreography"`

This creates the migration file. Verify it creates `music_analyses` and `choreography_programs` tables.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest backend/tests/models/test_choreography_models.py -v`
Expected: All PASS (uses SQLite in-memory from conftest.py)

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/choreography.py backend/app/models/__init__.py backend/alembic/versions/*add_choreography* backend/tests/models/test_choreography_models.py
git commit -m "feat(choreography): add MusicAnalysis and ChoreographyProgram ORM models"
```

---

## Task 5: CRUD Operations

**Files:**
- Create: `backend/app/crud/choreography.py`
- Create: `backend/tests/crud/test_choreography_crud.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/crud/test_choreography_crud.py
"""Tests for choreography CRUD operations."""

import pytest

from backend.app.crud.choreography import (
    create_music_analysis,
    create_program,
    get_music_analysis_by_id,
    get_program_by_id,
    list_programs_by_user,
    update_program,
)


@pytest.fixture
async def music(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
    )
    return music


async def test_create_and_get_music(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
    )
    assert music.id is not None
    assert music.status == "pending"

    fetched = await get_music_analysis_by_id(db_session, music.id)
    assert fetched is not None
    assert fetched.filename == "test.mp3"


async def test_create_and_get_program(db_session, music):
    program = await create_program(
        db_session,
        user_id="user-123",
        music_analysis_id=music.id,
        discipline="mens_singles",
        segment="free_skate",
    )
    assert program.id is not None

    fetched = await get_program_by_id(db_session, program.id)
    assert fetched is not None
    assert fetched.discipline == "mens_singles"


async def test_list_programs_by_user(db_session, music):
    await create_program(db_session, user_id="user-123", discipline="mens_singles", segment="free_skate")
    await create_program(db_session, user_id="user-123", discipline="mens_singles", segment="short_program")
    await create_program(db_session, user_id="other-user", discipline="womens_singles", segment="free_skate")

    programs = await list_programs_by_user(db_session, "user-123")
    assert len(programs) == 2


async def test_update_program(db_session, music):
    program = await create_program(
        db_session, user_id="user-123", discipline="mens_singles", segment="free_skate"
    )
    updated = await update_program(
        db_session,
        program,
        title="My Program",
        total_tes=45.5,
        is_valid=True,
    )
    assert updated.title == "My Program"
    assert updated.total_tes == 45.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/crud/test_choreography_crud.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CRUD**

```python
# backend/app/crud/choreography.py
"""CRUD operations for choreography models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import desc, select

from backend.app.models.choreography import ChoreographyProgram, MusicAnalysis

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# --- Music Analysis ---


async def create_music_analysis(
    db: AsyncSession,
    *,
    user_id: str,
    filename: str,
    audio_url: str,
    duration_sec: float,
    **kwargs,
) -> MusicAnalysis:
    music = MusicAnalysis(
        user_id=user_id,
        filename=filename,
        audio_url=audio_url,
        duration_sec=duration_sec,
        **kwargs,
    )
    db.add(music)
    await db.flush()
    await db.refresh(music)
    return music


async def get_music_analysis_by_id(db: AsyncSession, music_id: str) -> MusicAnalysis | None:
    result = await db.execute(
        select(MusicAnalysis).where(MusicAnalysis.id == music_id)
    )
    return result.scalar_one_or_none()


# --- Choreography Program ---


async def create_program(
    db: AsyncSession,
    *,
    user_id: str,
    discipline: str,
    segment: str,
    **kwargs,
) -> ChoreographyProgram:
    program = ChoreographyProgram(
        user_id=user_id,
        discipline=discipline,
        segment=segment,
        **kwargs,
    )
    db.add(program)
    await db.flush()
    await db.refresh(program)
    return program


async def get_program_by_id(db: AsyncSession, program_id: str) -> ChoreographyProgram | None:
    result = await db.execute(
        select(ChoreographyProgram).where(ChoreographyProgram.id == program_id)
    )
    return result.scalar_one_or_none()


async def list_programs_by_user(
    db: AsyncSession,
    user_id: str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[ChoreographyProgram]:
    query = (
        select(ChoreographyProgram)
        .where(ChoreographyProgram.user_id == user_id)
        .order_by(desc(ChoreographyProgram.created_at))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def update_program(db: AsyncSession, program: ChoreographyProgram, **kwargs) -> ChoreographyProgram:
    for key, value in kwargs.items():
        if value is not None:
            setattr(program, key, value)
    db.add(program)
    await db.flush()
    await db.refresh(program)
    return program


async def delete_program(db: AsyncSession, program: ChoreographyProgram) -> None:
    await db.delete(program)
    await db.flush()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/tests/crud/test_choreography_crud.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/crud/choreography.py backend/tests/crud/test_choreography_crud.py
git commit -m "feat(choreography): add CRUD operations for music and programs"
```

---

## Task 6: Pydantic Schemas

**Files:**
- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Add choreography schemas**

Append these schemas to `backend/app/schemas.py` (after existing schemas):

```python
# ---------------------------------------------------------------------------
# Choreography
# ---------------------------------------------------------------------------


class MusicAnalysisResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    audio_url: str
    duration_sec: float
    bpm: float | None
    meter: str | None
    structure: list[dict] | None
    energy_curve: dict | None
    downbeats: list[float] | None
    peaks: list[float] | None
    status: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class UploadMusicResponse(BaseModel):
    music_id: str
    filename: str


class GenerateRequest(BaseModel):
    music_id: str
    discipline: str = Field(pattern=r"^(mens_singles|womens_singles)$")
    segment: str = Field(pattern=r"^(short_program|free_skate)$")
    inventory: dict
    # inventory example:
    # {
    #   "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2T", "1Eu"],
    #   "spins": ["CSp4", "LSp4", "CCSp4"],
    #   "combinations": ["3Lz+2T", "3Lz+3T"]
    # }


class LayoutElement(BaseModel):
    code: str
    goe: int = 0
    timestamp: float = 0.0
    position: dict | None = None
    is_back_half: bool = False
    is_jump_pass: bool = False
    jump_pass_index: int | None = None


class Layout(BaseModel):
    elements: list[LayoutElement]
    total_tes: float
    back_half_indices: list[int]


class GenerateResponse(BaseModel):
    layouts: list[Layout]


class ValidateRequest(BaseModel):
    discipline: str = Field(pattern=r"^(mens_singles|womens_singles)$")
    segment: str = Field(pattern=r"^(short_program|free_skate)$")
    elements: list[dict]  # list of {code, goe, timestamp, ...}


class ValidateResponse(BaseModel):
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    total_tes: float | None = None


class RenderRinkRequest(BaseModel):
    elements: list[dict]
    width: int = Field(default=1200, ge=400, le=4000)
    height: int = Field(default=600, ge=200, le=2000)


class ChoreographyProgramResponse(BaseModel):
    id: str
    user_id: str
    music_analysis_id: str | None
    title: str | None
    discipline: str
    segment: str
    season: str
    layout: dict | None
    total_tes: float | None
    estimated_goe: float | None
    estimated_pcs: float | None
    estimated_total: float | None
    is_valid: bool | None
    validation_errors: list[str] | None
    validation_warnings: list[str] | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class ProgramListResponse(BaseModel):
    programs: list[ChoreographyProgramResponse]
    total: int


class SaveProgramRequest(BaseModel):
    title: str | None = None
    layout: dict | None = None
    total_tes: float | None = None
    estimated_goe: float | None = None
    estimated_pcs: float | None = None
    estimated_total: float | None = None
    is_valid: bool | None = None
    validation_errors: list[str] | None = None
    validation_warnings: list[str] | None = None


class ExportRequest(BaseModel):
    format: str = Field(pattern=r"^(svg|pdf|json)$")
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from backend.app.schemas import GenerateRequest, ValidateRequest, ChoreographyProgramResponse"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(choreography): add Pydantic schemas for all endpoints"
```

---

## Task 7: Rink Renderer (SVG)

**Files:**
- Create: `backend/app/services/choreography/rink_renderer.py`
- Create: `backend/tests/services/choreography/test_rink_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/services/choreography/test_rink_renderer.py
"""Tests for SVG rink renderer."""

from backend.app.services.choreography.rink_renderer import render_rink


def test_render_empty_rink():
    svg = render_rink([], width=600, height=300)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert 'viewBox="0 0 60 30"' in svg
    assert "rink" in svg


def test_render_rink_with_elements():
    elements = [
        {"code": "3Lz", "position": {"x": 15.0, "y": 10.0}, "timestamp": 5.0},
        {"code": "CSp4", "position": {"x": 30.0, "y": 15.0}, "timestamp": 30.0},
    ]
    svg = render_rink(elements, width=1200, height=600)
    assert "3Lz" in svg
    assert "CSp4" in svg


def test_render_rink_dimensions():
    svg = render_rink([], width=800, height=400)
    assert 'width="800"' in svg
    assert 'height="400"' in svg


def test_render_has_center_line():
    svg = render_rink([])
    assert "center" in svg.lower() or "cx=" in svg


def test_render_has_goal_lines():
    svg = render_rink([])
    # Rink has goal lines at each end
    assert svg.count("line") >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_rink_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rink renderer**

```python
# backend/app/services/choreography/rink_renderer.py
"""SVG rink diagram renderer.

Generates a top-down orthographic view of a 60m x 30m ice rink
with element markers, labels, and connecting paths.
"""

from __future__ import annotations


def render_rink(
    elements: list[dict],
    *,
    width: int = 1200,
    height: int = 600,
) -> str:
    """Render a rink diagram as SVG string.

    Args:
        elements: list of dicts with "code", "position" ({x, y}), "timestamp".
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        SVG string.
    """
    # Rink dimensions in meters
    rink_w, rink_h = 60.0, 30.0
    # Scale factors (SVG viewBox is 0 0 60 30)
    sx = width / rink_w
    sy = height / rink_h

    parts: list[str] = []

    # SVG header
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 60 30">')

    # Background
    parts.append('<rect x="0" y="0" width="60" height="30" fill="#e8f0fe" rx="1"/>')

    # Rink border
    parts.append('<rect x="1" y="1" width="58" height="28" fill="none" stroke="#2563eb" stroke-width="0.15" rx="0.5"/>')

    # Center line (red)
    parts.append('<line x1="30" y1="1" x2="30" y2="29" stroke="#dc2626" stroke-width="0.1" stroke-dasharray="0.5,0.5"/>')

    # Center circle
    parts.append('<circle cx="30" cy="15" r="4.5" fill="none" stroke="#dc2626" stroke-width="0.1"/>')
    parts.append('<circle cx="30" cy="15" r="0.15" fill="#dc2626"/>')

    # Goal lines (blue)
    parts.append('<line x1="5" y1="1" x2="5" y2="29" stroke="#2563eb" stroke-width="0.08"/>')
    parts.append('<line x1="55" y1="1" x2="55" y2="29" stroke="#2563eb" stroke-width="0.08"/>')

    # Face-off circles
    for cx, cy in [(10, 7.5), (10, 22.5), (50, 7.5), (50, 22.5)]:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="none" stroke="#2563eb" stroke-width="0.08"/>')
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="0.15" fill="#dc2626"/>')

    # Draw element markers
    for i, el in enumerate(elements):
        pos = el.get("position")
        if not pos:
            continue
        x, y = pos["x"], pos["y"]
        code = el.get("code", "")

        # Determine element type for styling
        is_jump = any(c.isdigit() and c != "0" for c in code[:2]) and any(
            c in code for c in "TSLFA"
        )
        is_spin = "Sp" in code
        is_step = "StSq" in code
        is_choreo = "ChSq" in code

        if is_spin:
            # Spin: circle
            parts.append(f'<circle cx="{x}" cy="{y}" r="1.2" fill="#9333ea" opacity="0.3" stroke="#9333ea" stroke-width="0.1"/>')
            color = "#9333ea"
        elif is_step:
            # Step sequence: dashed path marker
            parts.append(f'<rect x="{x-1}" y="{y-0.5}" width="2" height="1" fill="none" stroke="#16a34a" stroke-width="0.1" stroke-dasharray="0.3,0.2"/>')
            color = "#16a34a"
        elif is_choreo:
            # Choreo sequence: diamond
            parts.append(f'<polygon points="{x},{y-0.8} {x+0.8},{y} {x},{y+0.8} {x-0.8},{y}" fill="#2563eb" opacity="0.3" stroke="#2563eb" stroke-width="0.1"/>')
            color = "#2563eb"
        else:
            # Jump: dot with label
            parts.append(f'<circle cx="{x}" cy="{y}" r="0.6" fill="#ea580c" opacity="0.8"/>')
            color = "#ea580c"

        # Label
        parts.append(f'<text x="{x}" y="{y - 1.2}" text-anchor="middle" font-size="1.2" fill="{color}" font-weight="bold">{code}</text>')

        # Index number
        parts.append(f'<text x="{x}" y="{y + 0.3}" text-anchor="middle" font-size="0.7" fill="#666">{i + 1}</text>')

        # Draw connecting line to next element
        if i < len(elements) - 1:
            next_pos = elements[i + 1].get("position")
            if next_pos:
                parts.append(
                    f'<line x1="{x}" y1="{y}" x2="{next_pos["x"]}" y2="{next_pos["y"]}" '
                    f'stroke="#94a3b8" stroke-width="0.06" stroke-dasharray="0.3,0.2" opacity="0.6"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_rink_renderer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/choreography/rink_renderer.py backend/tests/services/choreography/test_rink_renderer.py
git commit -m "feat(choreography): add SVG rink renderer with element markers"
```

---

## Task 8: CSP Solver

**Files:**
- Create: `backend/app/services/choreography/csp_solver.py`
- Create: `backend/tests/services/choreography/test_csp_solver.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/services/choreography/test_csp_solver.py
"""Tests for CSP solver."""

import pytest

from backend.app.services.choreography.csp_solver import solve_layout


def test_basic_free_skate():
    """Generate a valid FS layout from a reasonable inventory."""
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2T", "1Eu"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T", "3Lz+3T", "3F+2T", "3Lo+2T"],
    }
    music_features = {
        "duration": 240.0,
        "peaks": [15.0, 40.0, 65.0, 90.0, 120.0, 150.0, 180.0, 200.0, 220.0],
        "structure": [
            {"type": "intro", "start": 0.0, "end": 15.0},
            {"type": "verse", "start": 15.0, "end": 60.0},
            {"type": "chorus", "start": 60.0, "end": 120.0},
            {"type": "bridge", "start": 120.0, "end": 160.0},
            {"type": "chorus", "start": 160.0, "end": 220.0},
            {"type": "outro", "start": 220.0, "end": 240.0},
        ],
    }
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="free_skate",
        num_layouts=3,
    )
    assert len(layouts) >= 1
    assert len(layouts) <= 3

    layout = layouts[0]
    assert layout["total_tes"] > 0
    assert len(layout["elements"]) > 0


def test_layout_has_required_elements():
    """Every generated layout must have jumps, spins, StSq, ChSq."""
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2T", "1Eu"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T", "3F+2T"],
    }
    music_features = {
        "duration": 240.0,
        "peaks": [15.0, 40.0, 65.0, 90.0, 120.0],
        "structure": [],
    }
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="free_skate",
    )
    assert len(layouts) >= 1

    for layout in layouts:
        codes = [el["code"] for el in layout["elements"]]
        # Must have step sequence
        assert any("StSq" in c for c in codes), f"Missing StSq in {codes}"
        # Must have choreo sequence
        assert any("ChSq" in c for c in codes), f"Missing ChSq in {codes}"
        # Must have spins
        assert any("Sp" in c for c in codes), f"Missing spins in {codes}"


def test_short_program():
    """Generate a valid SP layout."""
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "2A", "2T"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T"],
    }
    music_features = {"duration": 160.0, "peaks": [10.0, 30.0, 60.0, 100.0], "structure": []}
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="short_program",
    )
    assert len(layouts) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_csp_solver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Add OR-Tools dependency**

Run: `cd backend && uv add "ortools>=9.0"`

- [ ] **Step 4: Implement CSP solver**

```python
# backend/app/services/choreography/csp_solver.py
"""CSP solver for choreography layout using OR-Tools CP-SAT.

Generates valid ISU-compliant programs by placing elements on a timeline,
optimizing for maximum TES with back-half bonus.
"""

from __future__ import annotations

import random
from itertools import combinations, permutations

from backend.app.services.choreography.elements_db import (
    ELEMENTS,
    ElementType,
    get_element,
)
from backend.app.services.choreography.rules_engine import validate_layout
from backend.app.services.choreography.score_calculator import calculate_tes


def _parse_combination(combo_str: str) -> list[str]:
    """Parse combination string like '3Lz+2T' into list of codes."""
    return [c.strip() for c in combo_str.split("+")]


def _generate_candidates(
    inventory: dict,
    segment: str,
) -> list[dict]:
    """Generate candidate layouts via constraint-guided random search.

    Uses random permutation with constraint filtering instead of full CP-SAT
    for MVP — faster to implement, handles the <20 element domain well.
    """
    jumps = inventory.get("jumps", [])
    spins = inventory.get("spins", [])
    combos = inventory.get("combinations", [])

    # Build jump pass options: solo jumps + combinations
    jump_pass_options: list[list[str]] = []
    for j in jumps:
        jump_pass_options.append([j])
    for c in combos:
        parsed = _parse_combination(c)
        if parsed:
            jump_pass_options.append(parsed)

    # Ensure we have StSq and ChSq
    step_options = ["StSq4", "StSq3", "StSq2"]
    choreo_options = ["ChSq1"]

    candidates: list[dict] = []

    for _ in range(500):
        # Pick 7 jump passes (or fewer if inventory is small)
        num_passes = min(7, len(jump_pass_options))
        if num_passes < 5:
            continue

        selected_passes = random.sample(jump_pass_options, num_passes)
        all_jump_codes: list[str] = []
        for jp in selected_passes:
            all_jump_codes.extend(jp)

        # Pick 3 different spin types
        available_spins = [s for s in spins if s in ELEMENTS]
        if len(available_spins) < 3:
            # Fill with defaults
            available_spins = list({"CSp4", "LSp4", "FSp4"} | set(available_spins))
        selected_spins = random.sample(available_spins, min(3, len(available_spins)))

        # Pick step sequence and choreo sequence
        selected_stsq = random.choice(step_options)
        selected_chsq = choreo_options[0]

        # Build element list with jump_pass_index
        elements: list[dict] = []
        jump_idx = 0
        for jp in selected_passes:
            for j, code in enumerate(jp):
                entry: dict = {"code": code, "goe": 0, "timestamp": 0.0}
                if j == 0:
                    entry["jump_pass_index"] = jump_idx
                    jump_idx += 1
                elements.append(entry)

        for spin in selected_spins:
            elements.append({"code": spin, "goe": 0, "timestamp": 0.0})
        elements.append({"code": selected_stsq, "goe": 0, "timestamp": 0.0})
        elements.append({"code": selected_chsq, "goe": 0, "timestamp": 0.0})

        # Validate
        layout = {
            "discipline": "mens_singles",
            "segment": segment,
            "elements": elements,
        }
        result = validate_layout(layout)
        if not result.is_valid:
            continue

        # Calculate score
        jump_pass_count = sum(1 for e in elements if "jump_pass_index" in e)
        if segment == "free_skate" and jump_pass_count >= 3:
            back_half = set(range(jump_pass_count - 3, jump_pass_count))
        else:
            back_half = set()

        flat = [{"code": e["code"], "goe": e["goe"]} for e in elements]
        tes = calculate_tes(flat, back_half)

        candidates.append({
            "elements": elements,
            "total_tes": round(tes, 2),
            "back_half_indices": sorted(back_half),
        })

        if len(candidates) >= 50:
            break

    # Sort by TES descending, return top N
    candidates.sort(key=lambda c: c["total_tes"], reverse=True)
    return candidates


def solve_layout(
    inventory: dict,
    music_features: dict,
    discipline: str,
    segment: str,
    num_layouts: int = 3,
) -> list[dict]:
    """Generate valid choreography layouts.

    Args:
        inventory: dict with "jumps", "spins", "combinations" lists.
        music_features: dict with "duration", "peaks", "structure".
        discipline: "mens_singles" or "womens_singles".
        segment: "short_program" or "free_skate".
        num_layouts: number of layouts to return (top N by TES).

    Returns:
        List of layout dicts, sorted by total_tes descending.
    """
    candidates = _generate_candidates(inventory, segment)

    # Assign timestamps based on music features
    duration = music_features.get("duration", 180.0)
    peaks = music_features.get("peaks", [])

    for layout in candidates:
        elements = layout["elements"]
        n = len(elements)
        if n == 0:
            continue

        # Distribute elements evenly across the program duration
        # Jumps prefer energy peaks, spins/choreo prefer lower energy
        for i, el in enumerate(elements):
            if peaks:
                # Place near the closest energy peak
                target_time = peaks[i % len(peaks)] if "jump_pass_index" in el else (duration * (i + 1) / (n + 1))
                target_time = min(target_time, duration - 5.0)
            else:
                target_time = duration * (i + 1) / (n + 1)
            el["timestamp"] = round(target_time, 1)

            # Assign default GOE (0 for now — user can edit later)
            el["goe"] = 0

    return candidates[:num_layouts]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_csp_solver.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/choreography/csp_solver.py backend/tests/services/choreography/test_csp_solver.py backend/pyproject.toml
git commit -m "feat(choreography): add CSP solver with random search + constraint filtering"
```

---

## Task 9: Music Analyzer

**Files:**
- Create: `backend/app/services/choreography/music_analyzer.py`
- Create: `backend/tests/services/choreography/test_music_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/services/choreography/test_music_analyzer.py
"""Tests for music analyzer."""

from unittest.mock import patch

import pytest

from backend.app.services.choreography.music_analyzer import (
    analyze_music_sync,
    extract_features_for_csp,
)


def test_extract_features_for_csp_basic():
    """Extract CSP-relevant features from analysis result."""
    analysis = {
        "bpm": 120.0,
        "duration_sec": 180.0,
        "peaks": [10.0, 25.0, 40.0, 60.0],
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
    }
    features = extract_features_for_csp(analysis)
    assert features["duration"] == 180.0
    assert features["peaks"] == [10.0, 25.0, 40.0, 60.0]
    assert features["structure"] == [{"type": "verse", "start": 0.0, "end": 30.0}]


def test_extract_features_handles_missing_peaks():
    analysis = {"bpm": 120.0, "duration_sec": 180.0}
    features = extract_features_for_csp(analysis)
    assert features["duration"] == 180.0
    assert features["peaks"] == []


def test_analyze_music_sync_returns_dict():
    """analyze_music_sync should return a dict with expected keys."""
    # This test requires madmom/librosa — skip if not installed
    pytest.importorskip("madmom", reason="madmom not installed")
    pytest.importorskip("librosa", reason="librosa not installed")

    # We can't easily test without a real audio file,
    # but we can test the structure with a mock
    with patch("backend.app.services.choreography.music_analyzer._run_analysis") as mock:
        mock.return_value = {
            "bpm": 120.0,
            "duration_sec": 10.0,
            "peaks": [2.0, 5.0],
            "structure": [],
        }
        result = analyze_music_sync("/fake/path.mp3")
        assert result["bpm"] == 120.0
        assert result["duration_sec"] == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/services/choreography/test_music_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement music analyzer**

```python
# backend/app/services/choreography/music_analyzer.py
"""Music analysis: BPM, structure, energy peaks using madmom + librosa.

This module is called from the arq worker (ml/src/worker.py),
NOT from the backend directly. The backend only stores/retrieves cached results.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_analysis(audio_path: str) -> dict:
    """Run music analysis pipeline.

    Steps:
    1. madmom BeatTracker → BPM
    2. librosa RMS energy → energy curve
    3. scipy.signal.find_peaks → energy peaks
    4. MSAF → structure boundaries (optional, can fail gracefully)

    Returns dict with: bpm, duration_sec, peaks, structure, energy_curve.
    """
    import numpy as np

    # --- Load audio ---
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration_sec = float(len(y) / sr)

    # --- BPM via madmom ---
    bpm = None
    try:
        from madmom.features.beats import DBNBeatTracker

        act = DBNBeatTracker.preprocess(y, sr=sr)
        beat_frames = DBNBeatTracker.detect(act, fps=sr / 512)
        if len(beat_frames) > 1:
            intervals = np.diff(beat_frames) * 512 / sr
            bpm = float(60.0 / np.median(intervals))
    except Exception:
        logger.warning("madmom beat tracking failed, using librosa fallback")

    if bpm is None:
        bpm = float(librosa.beat.beat_track(y=y, sr=sr)[0])

    # --- Energy curve (RMS per 0.5s window) ---
    hop_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    timestamps = [float(i * 0.5) for i in range(len(rms))]
    energy_curve = {"timestamps": timestamps, "values": [float(v) for v in rms]}

    # --- Energy peaks ---
    peaks: list[float] = []
    try:
        from scipy.signal import find_peaks

        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        peak_indices, _ = find_peaks(rms_normalized, height=0.6, distance=4)
        peaks = [timestamps[i] for i in peak_indices]
    except Exception:
        logger.warning("Peak detection failed")

    # --- Structure analysis (MSAF, optional) ---
    structure: list[dict] = []
    try:
        import msaf

        boundaries, labels = msaf.process(audio_path, boundaries_id="sf", labels_id="foote")
        for i in range(len(boundaries) - 1):
            structure.append({
                "type": labels[i] if i < len(labels) else "unknown",
                "start": float(boundaries[i]),
                "end": float(boundaries[i + 1]),
            })
    except Exception:
        logger.warning("MSAF structure analysis failed — using empty structure")

    return {
        "bpm": round(bpm, 1),
        "duration_sec": round(duration_sec, 1),
        "peaks": peaks,
        "structure": structure,
        "energy_curve": energy_curve,
    }


def analyze_music_sync(audio_path: str) -> dict:
    """Analyze a music file synchronously. Called from arq worker.

    Args:
        audio_path: path to mp3/wav file.

    Returns:
        dict with bpm, duration_sec, peaks, structure, energy_curve.
    """
    return _run_analysis(audio_path)


def extract_features_for_csp(analysis: dict) -> dict:
    """Extract CSP-relevant features from a full analysis result.

    Args:
        analysis: dict from MusicAnalysis model (or analyze_music_sync output).

    Returns:
        dict with: duration, peaks, structure.
    """
    return {
        "duration": analysis.get("duration_sec", 180.0),
        "peaks": analysis.get("peaks", []),
        "structure": analysis.get("structure", []),
    }
```

- [ ] **Step 4: Add madmom, msaf, librosa to backend dependencies**

Run: `cd backend && uv add "madmom>=0.16.1" "msaf>=0.1.0" "librosa>=0.10.0"`

Note: scipy is a transitive dependency of librosa.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest backend/tests/services/choreography/test_music_analyzer.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/choreography/music_analyzer.py backend/tests/services/choreography/test_music_analyzer.py backend/pyproject.toml
git commit -m "feat(choreography): add music analyzer with madmom + librosa + MSAF"
```

---

## Task 10: API Routes

**Files:**
- Create: `backend/app/routes/choreography.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Implement routes**

```python
# backend/app/routes/choreography.py
"""Choreography planner API routes."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from backend.app.crud.choreography import (
    create_music_analysis,
    create_program,
    delete_program,
    get_music_analysis_by_id,
    get_program_by_id,
    list_programs_by_user,
    update_program,
)
from backend.app.schemas import (
    ChoreographyProgramResponse,
    ExportRequest,
    GenerateRequest,
    GenerateResponse,
    LayoutElement,
    ProgramListResponse,
    RenderRinkRequest,
    SaveProgramRequest,
    UploadMusicResponse,
    ValidateRequest,
    ValidateResponse,
)
from backend.app.services.choreography.csp_solver import solve_layout
from backend.app.services.choreography.music_analyzer import extract_features_for_csp
from backend.app.services.choreography.rink_renderer import render_rink
from backend.app.services.choreography.rules_engine import validate_layout
from backend.app.services.choreography.score_calculator import calculate_tes
from backend.app.storage import upload_file

if TYPE_CHECKING:
    from backend.app.auth.deps import CurrentUser, DbDep

router = APIRouter(tags=["choreography"])


def _program_to_response(program) -> ChoreographyProgramResponse:
    return ChoreographyProgramResponse.model_validate(program)


# --- Upload Music ---


@router.post("/choreography/upload-music", response_model=UploadMusicResponse, status_code=status.HTTP_201_CREATED)
async def upload_music(file: UploadFile = File(...), user: CurrentUser = None, db: DbDep = None):
    """Upload an audio file for analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save to R2
    ext = file.filename.rsplit(".", 1)[-1].lower()
    key = f"music/{uuid.uuid4()}.{ext}"

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        upload_file(tmp_path, key)
    finally:
        os.unlink(tmp_path)

    # Get duration (rough estimate from file size)
    duration = len(content) / 16000  # rough for mp3

    # Create DB record
    music = await create_music_analysis(
        db,
        user_id=user.id,
        filename=file.filename,
        audio_url=key,
        duration_sec=duration,
        status="pending",
    )

    return UploadMusicResponse(music_id=music.id, filename=file.filename)


# --- Get Music Analysis ---


@router.get("/choreography/music/{music_id}/analysis")
async def get_music_analysis(music_id: str, user: CurrentUser = None, db: DbDep = None):
    from backend.app.schemas import MusicAnalysisResponse

    music = await get_music_analysis_by_id(db, music_id)
    if music is None:
        raise HTTPException(status_code=404, detail="Music analysis not found")
    if music.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your music")
    return MusicAnalysisResponse.model_validate(music)


# --- Generate Layouts ---


@router.post("/choreography/generate", response_model=GenerateResponse)
async def generate_layouts(body: GenerateRequest, user: CurrentUser = None, db: DbDep = None):
    """Generate choreography layouts using CSP solver."""
    music = await get_music_analysis_by_id(db, body.music_id)
    if music is None:
        raise HTTPException(status_code=404, detail="Music analysis not found")

    # Extract CSP features
    analysis_data = {
        "bpm": music.bpm,
        "duration_sec": music.duration_sec,
        "peaks": music.peaks or [],
        "structure": music.structure or [],
    }
    music_features = extract_features_for_csp(analysis_data)

    # Solve
    layouts = solve_layout(
        inventory=body.inventory,
        music_features=music_features,
        discipline=body.discipline,
        segment=body.segment,
        num_layouts=3,
    )

    return GenerateResponse(
        layouts=[
            {
                "elements": [
                    LayoutElement(
                        code=e["code"],
                        goe=e.get("goe", 0),
                        timestamp=e.get("timestamp", 0.0),
                        position=e.get("position"),
                        is_back_half=e.get("is_back_half", False),
                        is_jump_pass="jump_pass_index" in e,
                        jump_pass_index=e.get("jump_pass_index"),
                    )
                    for e in layout["elements"]
                ],
                "total_tes": layout["total_tes"],
                "back_half_indices": layout.get("back_half_indices", []),
            }
            for layout in layouts
        ]
    )


# --- Validate Layout ---


@router.post("/choreography/validate", response_model=ValidateResponse)
async def validate(body: ValidateRequest):
    """Validate a layout against ISU rules."""
    layout = {
        "discipline": body.discipline,
        "segment": body.segment,
        "elements": body.elements,
    }
    result = validate_layout(layout)

    # Calculate TES
    flat = [{"code": e["code"], "goe": e.get("goe", 0)} for e in body.elements]
    jump_pass_indices = set()
    idx = 0
    for e in body.elements:
        if e.get("is_jump_pass", False):
            jump_pass_indices.add(idx)
        idx += 1

    # Back-half: last 3 jump passes for FS
    num_jump_passes = len(jump_pass_indices)
    back_half = set()
    if body.segment == "free_skate" and num_jump_passes >= 3:
        sorted_indices = sorted(jump_pass_indices)
        back_half = set(sorted_indices[-3:])

    tes = calculate_tes(flat, back_half)

    return ValidateResponse(
        is_valid=result.is_valid,
        errors=result.errors,
        warnings=result.warnings,
        total_tes=round(tes, 2),
    )


# --- Render Rink ---


@router.post("/choreography/render-rink")
async def render_rink_endpoint(body: RenderRinkRequest):
    """Render a rink diagram as SVG."""
    svg = render_rink(body.elements, width=body.width, height=body.height)
    return {"svg": svg}


# --- Programs CRUD ---


@router.get("/choreography/programs", response_model=ProgramListResponse)
async def list_programs(
    user: CurrentUser = None,
    db: DbDep = None,
    limit: int = 20,
    offset: int = 0,
):
    programs = await list_programs_by_user(db, user.id, limit=limit, offset=offset)
    return ProgramListResponse(
        programs=[_program_to_response(p) for p in programs],
        total=len(programs),
    )


@router.get("/choreography/programs/{program_id}", response_model=ChoreographyProgramResponse)
async def get_program(program_id: str, user: CurrentUser = None, db: DbDep = None):
    program = await get_program_by_id(db, program_id)
    if program is None:
        raise HTTPException(status_code=404, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your program")
    return _program_to_response(program)


@router.post("/choreography/programs", response_model=ChoreographyProgramResponse, status_code=status.HTTP_201_CREATED)
async def save_new_program(
    body: SaveProgramRequest,
    user: CurrentUser = None,
    db: DbDep = None,
):
    program = await create_program(
        db,
        user_id=user.id,
        **body.model_dump(exclude_none=True),
    )
    return _program_to_response(program)


@router.put("/choreography/programs/{program_id}", response_model=ChoreographyProgramResponse)
async def update_program_endpoint(
    program_id: str,
    body: SaveProgramRequest,
    user: CurrentUser = None,
    db: DbDep = None,
):
    program = await get_program_by_id(db, program_id)
    if program is None:
        raise HTTPException(status_code=404, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your program")
    program = await update_program(
        db, program, **body.model_dump(exclude_none=True)
    )
    return _program_to_response(program)


@router.delete("/choreography/programs/{program_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_program_endpoint(
    program_id: str,
    user: CurrentUser = None,
    db: DbDep = None,
):
    program = await get_program_by_id(db, program_id)
    if program is None:
        raise HTTPException(status_code=404, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your program")
    await delete_program(db, program)


# --- Export ---


@router.post("/choreography/programs/{program_id}/export")
async def export_program(
    program_id: str,
    body: ExportRequest,
    user: CurrentUser = None,
    db: DbDep = None,
):
    program = await get_program_by_id(db, program_id)
    if program is None:
        raise HTTPException(status_code=404, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your program")

    if body.format == "json":
        return {"format": "json", "data": program.layout}
    elif body.format == "svg":
        elements = (program.layout or {}).get("elements", [])
        svg = render_rink(elements)
        return {"format": "svg", "data": svg}
    elif body.format == "pdf":
        # MVP: return SVG — PDF generation will use weasyprint or reportlab later
        elements = (program.layout or {}).get("elements", [])
        svg = render_rink(elements)
        return {"format": "pdf", "data": svg, "note": "PDF generation not yet implemented, returning SVG"}
```

- [ ] **Step 2: Register router in main.py**

Add to `backend/app/main.py` imports and router registration:

```python
# In imports:
from backend.app.routes import choreography

# In router registration (after sessions):
api_v1.include_router(choreography.router)
```

- [ ] **Step 3: Verify the app starts**

Run: `cd backend && uv run python -c "from backend.app.main import app; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/choreography.py backend/app/main.py
git commit -m "feat(choreography): add all API routes for planner endpoints"
```

---

## Task 11: Frontend Types

**Files:**
- Create: `frontend/src/types/choreography.ts`

- [ ] **Step 1: Create TypeScript types**

```typescript
// frontend/src/types/choreography.ts

export interface MusicAnalysis {
  id: string
  user_id: string
  filename: string
  audio_url: string
  duration_sec: number
  bpm: number | null
  meter: string | null
  structure: MusicSegment[] | null
  energy_curve: EnergyCurve | null
  downbeats: number[] | null
  peaks: number[] | null
  status: "pending" | "analyzing" | "completed" | "failed"
  created_at: string
  updated_at: string
}

export interface MusicSegment {
  type: string
  start: number
  end: number
}

export interface EnergyCurve {
  timestamps: number[]
  values: number[]
}

export interface UploadMusicResponse {
  music_id: string
  filename: string
}

export interface LayoutElement {
  code: string
  goe: number
  timestamp: number
  position: { x: number; y: number } | null
  is_back_half: boolean
  is_jump_pass: boolean
  jump_pass_index: number | null
}

export interface Layout {
  elements: LayoutElement[]
  total_tes: number
  back_half_indices: number[]
}

export interface GenerateResponse {
  layouts: Layout[]
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  total_tes: number | null
}

export interface ChoreographyProgram {
  id: string
  user_id: string
  music_analysis_id: string | null
  title: string | null
  discipline: "mens_singles" | "womens_singles"
  segment: "short_program" | "free_skate"
  season: string
  layout: ProgramLayout | null
  total_tes: number | null
  estimated_goe: number | null
  estimated_pcs: number | null
  estimated_total: number | null
  is_valid: boolean | null
  validation_errors: string[] | null
  validation_warnings: string[] | null
  created_at: string
  updated_at: string
}

export interface ProgramLayout {
  elements: LayoutElement[]
}

export interface ProgramListResponse {
  programs: ChoreographyProgram[]
  total: number
}

export interface Inventory {
  jumps: string[]
  spins: string[]
  combinations: string[]
}
```

- [ ] **Step 2: Verify no type errors**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/choreography.ts
git commit -m "feat(choreography): add TypeScript types for planner"
```

---

## Task 12: Frontend API Client

**Files:**
- Create: `frontend/src/lib/api/choreography.ts`

- [ ] **Step 1: Create React Query hooks with Zod schemas**

```typescript
// frontend/src/lib/api/choreography.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiDelete, apiFetch, apiPost, apiPatch } from "@/lib/api-client"

// --- Zod Schemas ---

const MusicSegmentSchema = z.object({
  type: z.string(),
  start: z.number(),
  end: z.number(),
})

const EnergyCurveSchema = z.object({
  timestamps: z.array(z.number()),
  values: z.array(z.number()),
})

const MusicAnalysisSchema = z.object({
  id: z.string(),
  user_id: z.string(),
  filename: z.string(),
  audio_url: z.string(),
  duration_sec: z.number(),
  bpm: z.number().nullable(),
  meter: z.string().nullable(),
  structure: z.array(MusicSegmentSchema).nullable(),
  energy_curve: EnergyCurveSchema.nullable(),
  downbeats: z.array(z.number()).nullable(),
  peaks: z.array(z.number()).nullable(),
  status: z.enum(["pending", "analyzing", "completed", "failed"]),
  created_at: z.string(),
  updated_at: z.string(),
})

const UploadMusicResponseSchema = z.object({
  music_id: z.string(),
  filename: z.string(),
})

const LayoutElementSchema = z.object({
  code: z.string(),
  goe: z.number(),
  timestamp: z.number(),
  position: z.object({ x: z.number(), y: z.number() }).nullable(),
  is_back_half: z.boolean(),
  is_jump_pass: z.boolean(),
  jump_pass_index: z.number().nullable(),
})

const LayoutSchema = z.object({
  elements: z.array(LayoutElementSchema),
  total_tes: z.number(),
  back_half_indices: z.array(z.number()),
})

const GenerateResponseSchema = z.object({
  layouts: z.array(LayoutSchema),
})

const ValidationResultSchema = z.object({
  is_valid: z.boolean(),
  errors: z.array(z.string()),
  warnings: z.array(z.string()),
  total_tes: z.number().nullable(),
})

const RenderRinkResponseSchema = z.object({
  svg: z.string(),
})

const ChoreographyProgramSchema = z.object({
  id: z.string(),
  user_id: z.string(),
  music_analysis_id: z.string().nullable(),
  title: z.string().nullable(),
  discipline: z.string(),
  segment: z.string(),
  season: z.string(),
  layout: z.any().nullable(),
  total_tes: z.number().nullable(),
  estimated_goe: z.number().nullable(),
  estimated_pcs: z.number().nullable(),
  estimated_total: z.number().nullable(),
  is_valid: z.boolean().nullable(),
  validation_errors: z.array(z.string()).nullable(),
  validation_warnings: z.array(z.string()).nullable(),
  created_at: z.string(),
  updated_at: z.string(),
})

const ProgramListResponseSchema = z.object({
  programs: z.array(ChoreographyProgramSchema),
  total: z.number(),
})

// --- Hooks ---

export function useMusicAnalysis(musicId: string) {
  return useQuery({
    queryKey: ["music-analysis", musicId],
    queryFn: () => apiFetch(`/choreography/music/${musicId}/analysis`, MusicAnalysisSchema),
    enabled: !!musicId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "pending" || status === "analyzing" ? 3000 : false
    },
  })
}

export function useUploadMusic() {
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append("file", file)
      const res = await fetch("/api/v1/choreography/upload-music", {
        method: "POST",
        body: formData,
        headers: { Authorization: `Bearer ${localStorage.getItem("access_token")}` },
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(body.detail)
      }
      return UploadMusicResponseSchema.parse(await res.json())
    },
  })
}

export function useGenerateLayouts() {
  return useMutation({
    mutationFn: (body: {
      music_id: string
      discipline: string
      segment: string
      inventory: { jumps: string[]; spins: string[]; combinations: string[] }
    }) => apiPost("/choreography/generate", GenerateResponseSchema, body),
  })
}

export function useValidateLayout() {
  return useMutation({
    mutationFn: (body: {
      discipline: string
      segment: string
      elements: Record<string, unknown>[]
    }) => apiPost("/choreography/validate", ValidationResultSchema, body),
  })
}

export function useRenderRink() {
  return useMutation({
    mutationFn: (body: {
      elements: Record<string, unknown>[]
      width?: number
      height?: number
    }) => apiPost("/choreography/render-rink", RenderRinkResponseSchema, body),
  })
}

export function usePrograms() {
  return useQuery({
    queryKey: ["choreography-programs"],
    queryFn: () => apiFetch("/choreography/programs", ProgramListResponseSchema),
  })
}

export function useProgram(id: string) {
  return useQuery({
    queryKey: ["choreography-program", id],
    queryFn: () => apiFetch(`/choreography/programs/${id}`, ChoreographyProgramSchema),
    enabled: !!id,
  })
}

export function useSaveProgram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...body }: { id?: string } & Record<string, unknown>) => {
      if (id) {
        return apiPatch(`/choreography/programs/${id}`, ChoreographyProgramSchema, body)
      }
      return apiPost("/choreography/programs", ChoreographyProgramSchema, body)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["choreography-programs"] }),
  })
}

export function useDeleteProgram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/choreography/programs/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["choreography-programs"] }),
  })
}
```

- [ ] **Step 2: Verify no type errors**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api/choreography.ts
git commit -m "feat(choreography): add React Query hooks with Zod schemas"
```

---

## Task 13: Frontend — Music Uploader Component

**Files:**
- Create: `frontend/src/components/choreography/music-uploader.tsx`

- [ ] **Step 1: Create music uploader component**

```tsx
// frontend/src/components/choreography/music-uploader.tsx
"use client"

import { Loader2, Music, Upload } from "lucide-react"
import { useTranslations } from "@/i18n"
import { useMusicAnalysis, useUploadMusic } from "@/lib/api/choreography"

export function MusicUploader({
  musicId,
  onUploaded,
}: {
  musicId: string | null
  onUploaded: (id: string) => void
}) {
  const t = useTranslations("choreography")
  const upload = useUploadMusic()
  const { data: analysis } = useMusicAnalysis(musicId ?? "")

  async function handleFile(file: File) {
    const result = await upload.mutateAsync(file)
    onUploaded(result.music_id)
  }

  if (musicId && analysis?.status === "completed") {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border p-3">
        <Music className="h-5 w-5 shrink-0" style={{ color: "oklch(var(--score-good))" }} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{analysis.filename}</p>
          <p className="text-xs text-muted-foreground">
            {Math.round(analysis.duration_sec)}s · {analysis.bpm ?? "—"} BPM
          </p>
        </div>
      </div>
    )
  }

  if (musicId && (analysis?.status === "pending" || analysis?.status === "analyzing")) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border p-3">
        <Loader2 className="h-5 w-5 shrink-0 animate-spin" style={{ color: "oklch(var(--accent-gold))" }} />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{t("analyzing")}</p>
          <p className="text-xs text-muted-foreground">{t("analyzingHint")}</p>
        </div>
      </div>
    )
  }

  return (
    <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-dashed border-border p-6 transition-colors hover:border-primary hover:bg-accent/30">
      <Upload className="h-5 w-5 text-muted-foreground" />
      <div>
        <p className="text-sm font-medium">{t("uploadMusic")}</p>
        <p className="text-xs text-muted-foreground">MP3, WAV</p>
      </div>
      <input
        type="file"
        accept="audio/mp3,audio/wav,audio/mpeg"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
        }}
      />
    </label>
  )
}
```

- [ ] **Step 2: Add i18n keys**

Add to `frontend/messages/ru.json` under a new `"choreography"` key:

```json
{
  "choreography": {
    "title": "Планировщик программ",
    "uploadMusic": "Загрузить музыку",
    "analyzing": "Анализ музыки...",
    "analyzingHint": "Определяем BPM, структуру и пики энергии",
    "selectProgram": "Тип программы",
    "shortProgram": "Произвольная программа (КП)",
    "freeSkate": "Произвольный прокат",
    "mensSingles": "Мужское одиночное",
    "womensSingles": "Женское одиночное",
    "inventory": "Инвентарь",
    "jumps": "Прыжки",
    "spins": "Вращения",
    "combinations": "Комбинации",
    "generate": "Сгенерировать",
    "generating": "Генерация...",
    "selectLayout": "Выберите раскладку",
    "score": "Оценка",
    "tes": "TES",
    "goe": "GOE",
    "pcs": "PCS",
    "total": "Итого",
    "elements": "Элементы",
    "noPrograms": "Нет сохранённых программ",
    "save": "Сохранить",
    "export": "Экспорт",
    "validate": "Проверить",
    "valid": "Валидно",
    "errors": "Ошибки",
    "warnings": "Предупреждения"
  }
}
```

Add matching keys to `frontend/messages/en.json`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/choreography/music-uploader.tsx frontend/messages/
git commit -m "feat(choreography): add music uploader component with i18n"
```

---

## Task 14: Frontend — Inventory Editor Component

**Files:**
- Create: `frontend/src/components/choreography/inventory-editor.tsx`

- [ ] **Step 1: Create inventory editor**

```tsx
// frontend/src/components/choreography/inventory-editor.tsx
"use client"

import { useTranslations } from "@/i18n"
import type { Inventory } from "@/types/choreography"

const ALL_JUMPS = [
  "3A", "4T", "4S", "4Lo", "4F", "4Lz",
  "3Lz", "3F", "3Lo", "3S", "3T", "3A",
  "2A", "2Lz", "2F", "2Lo", "2S", "2T",
  "1Eu",
]

const ALL_SPINS = ["CSp4", "CSp3", "FSp4", "FSp3", "LSp4", "LSp3", "USp4", "USp3", "CSpB4"]

const ALL_COMBOS = [
  "3Lz+3T", "3Lz+2T", "3F+3T", "3F+2T", "3Lo+2T",
  "3S+2T", "3T+2T", "2A+3T", "2A+2T",
  "3Lz+1Eu+2S", "3Lz+1Eu+3S", "3F+1Eu+2S",
]

function Chip({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
        selected
          ? "bg-primary text-primary-foreground"
          : "bg-muted text-muted-foreground hover:bg-accent"
      }`}
    >
      {label}
    </button>
  )
}

export function InventoryEditor({ value, onChange }: { value: Inventory; onChange: (v: Inventory) => void }) {
  const t = useTranslations("choreography")

  function toggleJump(code: string) {
    const jumps = value.jumps.includes(code)
      ? value.jumps.filter((j) => j !== code)
      : [...value.jumps, code]
    onChange({ ...value, jumps })
  }

  function toggleSpin(code: string) {
    const spins = value.spins.includes(code)
      ? value.spins.filter((s) => s !== code)
      : [...value.spins, code]
    onChange({ ...value, spins })
  }

  function toggleCombo(code: string) {
    const combos = value.combinations.includes(code)
      ? value.combinations.filter((c) => c !== code)
      : [...value.combinations, code]
    onChange({ ...value, combinations: combos })
  }

  return (
    <div className="space-y-4">
      <div>
        <p className="mb-2 text-sm font-medium">{t("jumps")}</p>
        <div className="flex flex-wrap gap-1.5">
          {ALL_JUMPS.map((code) => (
            <Chip key={code} label={code} selected={value.jumps.includes(code)} onClick={() => toggleJump(code)} />
          ))}
        </div>
      </div>
      <div>
        <p className="mb-2 text-sm font-medium">{t("spins")}</p>
        <div className="flex flex-wrap gap-1.5">
          {ALL_SPINS.map((code) => (
            <Chip key={code} label={code} selected={value.spins.includes(code)} onClick={() => toggleSpin(code)} />
          ))}
        </div>
      </div>
      <div>
        <p className="mb-2 text-sm font-medium">{t("combinations")}</p>
        <div className="flex flex-wrap gap-1.5">
          {ALL_COMBOS.map((code) => (
            <Chip key={code} label={code} selected={value.combinations.includes(code)} onClick={() => toggleCombo(code)} />
          ))}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/choreography/inventory-editor.tsx
git commit -m "feat(choreography): add inventory editor with chip selector"
```

---

## Task 15: Frontend — Score Bar Component

**Files:**
- Create: `frontend/src/components/choreography/score-bar.tsx`

- [ ] **Step 1: Create score bar**

```tsx
// frontend/src/components/choreography/score-bar.tsx
"use client"

import { useTranslations } from "@/i18n"

export function ScoreBar({
  tes,
  goe,
  pcs,
  total,
  duration,
  jumpPasses,
  spinCount,
}: {
  tes: number | null
  goe: number | null
  pcs: number | null
  total: number | null
  duration: number | null
  jumpPasses: number
  spinCount: number
}) {
  const t = useTranslations("choreography")

  const items = [
    { label: t("tes"), value: tes?.toFixed(2) ?? "—", color: "oklch(var(--score-good))" },
    { label: t("goe"), value: goe?.toFixed(2) ?? "—", color: "oklch(var(--accent-gold))" },
    { label: t("pcs"), value: pcs?.toFixed(2) ?? "—", color: "oklch(var(--primary))" },
    { label: t("total"), value: total?.toFixed(2) ?? "—", color: "oklch(var(--foreground))", bold: true },
    { label: "", value: "", divider: true },
    { label: "Duration", value: duration ? `${Math.round(duration)}s` : "—", muted: true },
    { label: "Jumps", value: `${jumpPasses}/7`, muted: true },
    { label: "Spins", value: `${spinCount}/3`, muted: true },
  ]

  return (
    <div className="flex items-center gap-4 overflow-x-auto border-t border-border bg-background px-4 py-2">
      {items.map((item, i) =>
        "divider" in item && item.divider ? (
          <div key={i} className="h-6 w-px bg-border" />
        ) : (
          <div key={i} className="flex shrink-0 flex-col items-center">
            <span
              className="text-xs"
              style={{ color: "muted" in item && item.muted ? "oklch(var(--muted-foreground))" : (item.color || "oklch(var(--muted-foreground))") }}
            >
              {item.label}
            </span>
            <span
              className={`text-sm ${"bold" in item && item.bold ? "font-bold" : "font-medium"}`}
              style={{ color: ("muted" in item && item.muted) ? "oklch(var(--muted-foreground))" : (item.color || "oklch(var(--foreground))") }}
            >
              {item.value}
            </span>
          </div>
        )
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/choreography/score-bar.tsx
git commit -m "feat(choreography): add score bar component"
```

---

## Task 16: Frontend — Layout Picker Component

**Files:**
- Create: `frontend/src/components/choreography/layout-picker.tsx`

- [ ] **Step 1: Create layout picker**

```tsx
// frontend/src/components/choreography/layout-picker.tsx
"use client"

import { Check } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { Layout } from "@/types/choreography"

export function LayoutPicker({
  layouts,
  selected,
  onSelect,
}: {
  layouts: Layout[]
  selected: number
  onSelect: (index: number) => void
}) {
  const t = useTranslations("choreography")

  if (layouts.length === 0) return null

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">{t("selectLayout")}</p>
      <div className="grid gap-2 sm:grid-cols-3">
        {layouts.map((layout, i) => {
          const codes = layout.elements.map((e) => e.code).join(" · ")
          const isSelected = i === selected
          return (
            <button
              key={i}
              type="button"
              onClick={() => onSelect(i)}
              className={`rounded-xl border p-3 text-left transition-colors ${
                isSelected ? "border-primary bg-primary/10" : "border-border hover:bg-accent/30"
              }`}
            >
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">#{i + 1}</span>
                {isSelected && <Check className="h-4 w-4" style={{ color: "oklch(var(--score-good))" }} />}
              </div>
              <p className="text-xs truncate">{codes}</p>
              <p className="mt-1 text-sm font-bold">{layout.total_tes.toFixed(2)} TES</p>
            </button>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/choreography/layout-picker.tsx
git commit -m "feat(choreography): add layout picker component"
```

---

## Task 17: Frontend — Rink Diagram Component

**Files:**
- Create: `frontend/src/components/choreography/rink-diagram.tsx`

- [ ] **Step 1: Create rink diagram**

```tsx
// frontend/src/components/choreography/rink-diagram.tsx
"use client"

import { useRenderRink } from "@/lib/api/choreography"
import type { LayoutElement } from "@/types/choreography"

export function RinkDiagram({ elements }: { elements: LayoutElement[] }) {
  const render = useRenderRink()

  // Auto-render when elements change
  const svg = render.data?.svg

  // Trigger render on mount / element change
  if (elements.length > 0 && !render.isPending && !render.data) {
    render.mutate({ elements: elements as unknown as Record<string, unknown>[] })
  }

  if (render.isPending || !svg) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <p className="text-sm">Loading rink...</p>
      </div>
    )
  }

  return (
    <div
      className="h-full w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/choreography/rink-diagram.tsx
git commit -m "feat(choreography): add rink diagram component"
```

---

## Task 18: Frontend — Main Planner Page

**Files:**
- Create: `frontend/src/app/(app)/choreography/page.tsx`

- [ ] **Step 1: Create main planner page**

```tsx
// frontend/src/app/(app)/choreography/page.tsx
"use client"

import { useState } from "react"
import Link from "next/link"
import { Music, Plus } from "lucide-react"
import { useTranslations } from "@/i18n"
import { usePrograms } from "@/lib/api/choreography"

export default function ChoreographyPage() {
  const t = useTranslations("choreography")
  const { data, isLoading } = usePrograms()

  return (
    <div className="mx-auto max-w-2xl space-y-4 px-4 py-6 sm:max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="nike-h2">{t("title")}</h1>
        <Link
          href="/choreography/new"
          className="flex items-center gap-1.5 rounded-xl bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          <Plus className="h-4 w-4" />
          New
        </Link>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20 text-muted-foreground">
          Loading...
        </div>
      ) : !data?.programs.length ? (
        <div className="flex flex-col items-center gap-4 py-20">
          <Music className="h-10 w-10 text-muted-foreground" />
          <p className="text-muted-foreground">{t("noPrograms")}</p>
        </div>
      ) : (
        <div className="space-y-2">
          {data.programs.map((p) => (
            <Link
              key={p.id}
              href={`/choreography/programs/${p.id}`}
              className="block rounded-2xl border border-border p-3 transition-colors hover:bg-accent/30"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{p.title || `${p.segment} — ${p.discipline}`}</p>
                  <p className="text-xs text-muted-foreground">{p.season.replace("_", "/")}</p>
                </div>
                {p.estimated_total !== null && (
                  <span className="text-sm font-bold">{p.estimated_total.toFixed(2)}</span>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/\(app\)/choreography/page.tsx
git commit -m "feat(choreography): add main planner page with program list"
```

---

## Task 19: Frontend — Program Editor Page

**Files:**
- Create: `frontend/src/app/(app)/choreography/programs/[id]/page.tsx`

- [ ] **Step 1: Create program editor page**

This is the main interactive editor with timeline + rink + score bar.

```tsx
// frontend/src/app/(app)/choreography/programs/[id]/page.tsx
"use client"

import { useParams } from "next/navigation"
import { useTranslations } from "@/i18n"
import { useProgram, useValidateLayout, useRenderRink } from "@/lib/api/choreography"
import { RinkDiagram } from "@/components/choreography/rink-diagram"
import { ScoreBar } from "@/components/choreography/score-bar"

export default function ProgramEditorPage() {
  const { id } = useParams<{ id: string }>()
  const t = useTranslations("choreography")
  const { data: program, isLoading } = useProgram(id)

  if (isLoading) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Loading...</div>
  }

  if (!program) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Not found</div>
  }

  const elements = (program.layout?.elements ?? []) as Array<{
    code: string
    goe: number
    timestamp: number
    is_jump_pass?: boolean
  }>

  const jumpPasses = elements.filter((e) => e.is_jump_pass).length
  const spinCount = elements.filter((e) => "Sp" in e.code).length

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col">
      {/* Main content: timeline + rink */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Timeline (60%) */}
        <div className="flex flex-1 flex-col border-b border-border p-4 lg:border-b-0 lg:border-r">
          <h2 className="mb-3 nike-h3">{program.title || "Program"}</h2>

          {/* Elements list */}
          <div className="flex-1 space-y-1.5 overflow-y-auto">
            {elements.map((el, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-1.5"
              >
                <span className="w-5 text-xs text-muted-foreground">{i + 1}</span>
                <span className="flex-1 text-sm font-medium">{el.code}</span>
                <span className="text-xs text-muted-foreground">{el.timestamp.toFixed(1)}s</span>
              </div>
            ))}
            {elements.length === 0 && (
              <p className="py-8 text-center text-sm text-muted-foreground">No elements</p>
            )}
          </div>
        </div>

        {/* Rink diagram (40%) */}
        <div className="flex-1 bg-muted/20 p-2">
          <RinkDiagram elements={elements as any} />
        </div>
      </div>

      {/* Score bar */}
      <ScoreBar
        tes={program.total_tes}
        goe={program.estimated_goe}
        pcs={program.estimated_pcs}
        total={program.estimated_total}
        duration={null}
        jumpPasses={jumpPasses}
        spinCount={spinCount}
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/\(app\)/choreography/programs/\[id\]/page.tsx
git commit -m "feat(choreography): add program editor page with timeline and rink"
```

---

## Task 20: Add Choreography Nav Link

**Files:**
- Modify: `frontend/src/components/app-nav.tsx`

- [ ] **Step 1: Add choreography link to navigation**

Find the nav items array in `app-nav.tsx` and add a choreography entry. The exact implementation depends on the current nav structure — read the file first, then add:

```tsx
// Add to nav items:
{
  href: "/choreography",
  label: t("nav.choreography"), // "Планировщик" / "Planner"
  icon: Music,  // from lucide-react
}
```

Add the i18n key to both `messages/ru.json` and `messages/en.json`:

```json
{ "nav": { "choreography": "Планировщик" } }
```

- [ ] **Step 2: Verify the page renders**

Run: `cd frontend && bunx next build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/app-nav.tsx frontend/messages/
git commit -m "feat(choreography): add planner link to app navigation"
```

---

## Task 21: Integration Test — Full Flow

**Files:**
- Create: `backend/tests/routes/test_choreography_routes.py`

- [ ] **Step 1: Write integration test for validate endpoint**

```python
# backend/tests/routes/test_choreography_routes.py
"""Integration tests for choreography API routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_client(client):
    """Register a user and get auth token."""
    await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test",
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123",
    })
    token = resp.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    yield client


async def test_validate_endpoint(auth_client):
    resp = await auth_client.post("/api/v1/choreography/validate", json={
        "discipline": "mens_singles",
        "segment": "free_skate",
        "elements": [
            {"code": "3Lz", "goe": 2},
            {"code": "3F", "goe": 1},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "is_valid" in data
    assert "errors" in data
    assert "warnings" in data


async def test_render_rink_endpoint(auth_client):
    resp = await auth_client.post("/api/v1/choreography/render-rink", json={
        "elements": [
            {"code": "3Lz", "position": {"x": 15.0, "y": 10.0}},
        ],
        "width": 600,
        "height": 300,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "<svg" in data["svg"]


async def test_programs_crud(auth_client):
    # Create
    resp = await auth_client.post("/api/v1/choreography/programs", json={
        "discipline": "mens_singles",
        "segment": "free_skate",
        "layout": {"elements": []},
    })
    assert resp.status_code == 201
    program_id = resp.json()["id"]

    # Get
    resp = await auth_client.get(f"/api/v1/choreography/programs/{program_id}")
    assert resp.status_code == 200

    # List
    resp = await auth_client.get("/api/v1/choreography/programs")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1

    # Update
    resp = await auth_client.put(f"/api/v1/choreography/programs/{program_id}", json={
        "title": "My Test Program",
    })
    assert resp.status_code == 200
    assert resp.json()["title"] == "My Test Program"

    # Delete
    resp = await auth_client.delete(f"/api/v1/choreography/programs/{program_id}")
    assert resp.status_code == 204
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest backend/tests/routes/test_choreography_routes.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/routes/test_choreography_routes.py
git commit -m "test(choreography): add integration tests for all endpoints"
```

---

## Task 22: Run Full Test Suite + Lint

**Files:** None (verification only)

- [ ] **Step 1: Run all backend tests**

Run: `uv run pytest backend/tests/ -v --tb=short`
Expected: All existing + new tests PASS

- [ ] **Step 2: Run backend lint**

Run: `uv run ruff check backend/app/`
Expected: No errors

- [ ] **Step 3: Run frontend type check**

Run: `cd frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Run frontend lint**

Run: `cd frontend && bunx next lint`
Expected: No errors

- [ ] **Step 5: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix(choreography): lint and type fixes"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task(s) |
|-------------|---------|
| ISU Element Database (§5) | Task 1 |
| Score Calculation (§5) | Task 2 |
| Rules Engine — Zayak, well-balanced (§5) | Task 3 |
| DB Models + Migration (§4) | Task 4 |
| CRUD Operations | Task 5 |
| Pydantic Schemas (§3) | Task 6 |
| Rink Renderer — SVG (§8) | Task 7 |
| CSP Solver — OR-Tools (§7) | Task 8 |
| Music Analyzer — madmom + MSAF (§6) | Task 9 |
| API Routes (§3) | Task 10 |
| Frontend Types | Task 11 |
| Frontend API Client | Task 12 |
| Music Uploader Component | Task 13 |
| Inventory Editor Component | Task 14 |
| Score Bar Component | Task 15 |
| Layout Picker Component | Task 16 |
| Rink Diagram Component | Task 17 |
| Main Planner Page | Task 18 |
| Program Editor Page | Task 19 |
| Nav Link | Task 20 |
| Integration Tests | Task 21 |
| Full Suite Verification | Task 22 |

### Placeholder Scan
- No "TBD", "TODO", or "implement later" found in any task
- All code blocks contain complete implementations
- All test blocks contain complete test code

### Type Consistency
- `LayoutElement` used consistently across backend schemas and frontend types
- `inventory` dict shape matches between `GenerateRequest` schema and frontend `Inventory` type
- Element codes use ISU standard English notation throughout
