# AI Coach Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ISU-broadcast-style AI coach overlay that appears after each detected element, showing element name in Russian, key metrics with quality indicators, and top recommendation.

**Architecture:** Pre-compute analysis results (phases + metrics + recommendations) for the entire video, then look up per-frame to render the overlay panel during a display window after each element's landing. New `draw_coach_panel()` function in existing HUD elements, wired into `_draw_hud()` in the visualization script.

**Tech Stack:** OpenCV, Pillow (Cyrillic text), numpy — all already in use

---

## Context

### The Problem

Currently the HUD shows: element type name, frame counter, blade indicators, floor angle. After a jump lands, the viewer sees nothing about how well it was executed. Real skating broadcasts show an element scoreboard — we want an AI coach version with biomechanics feedback in Russian.

### What the Overlay Looks Like

```
┌─────────────────────────────────┐
│ Двойной сальхов                 │
│ Время полёта: 0.37с  ✓ OK       │
│ Высота: 0.12                     │
│ Ось: наклон 25°  ⚠              │
│ Приземление: ✗ не на ту ногу    │
└─────────────────────────────────┘
```

Appears after landing, stays for 4 seconds, then fades.

### Data Flow

```
out_poses.npy (N,17,3) ─→ PhaseDetector.detect_jump_phases()
                         ─→ BiomechanicsAnalyzer.analyze()
                         ─→ Recommender.recommend()
                         ─→ CoachOverlayData (per-frame lookup)
                         ─→ draw_coach_panel() per frame
```

### Files Involved

| File | Role |
|------|------|
| `src/visualization/hud/coach_panel.py` | **Create** — `draw_coach_panel()`, `CoachOverlayData` |
| `scripts/visualize_with_skeleton.py` | **Modify** — pre-compute analysis, wire into `_draw_hud()` |
| `tests/visualization/test_coach_panel.py` | **Create** — tests for panel rendering and data |

### Existing Code to Reuse

- `render_cyrillic_text()` → `src/visualization/core/text.py:218-291` — Pillow-based Cyrillic rendering
- `draw_text_box()` → `src/visualization/core/text.py:294-370` — text with semi-transparent background
- `PhaseDetector` → `src/analysis/phase_detector.py` — already uses parabolic method
- `BiomechanicsAnalyzer` → `src/analysis/metrics.py:40-77` — returns `list[MetricResult]`
- `Recommender` → `src/analysis/recommender.py` — returns `list[str]` (Russian)
- `ELEMENT_DEFS` → `src/analysis/element_defs.py:34-128` — has `name_ru` for Russian element names
- `MetricResult` → `src/types.py:477-492` — has `name`, `value`, `unit`, `is_good`
- HUD config constants → `src/visualization/config.py` — colors, alpha, padding
- `_draw_hud()` → `scripts/visualize_with_skeleton.py:691-747` — main HUD entry point

---

## File Structure

```
src/visualization/hud/coach_panel.py     # NEW: overlay rendering + data structure
tests/visualization/test_coach_panel.py  # NEW: tests
scripts/visualize_with_skeleton.py       # MODIFY: wire into main loop
```

---

### Task 1: CoachOverlayData dataclass and analysis pre-computation

**Files:**
- Create: `src/visualization/hud/coach_panel.py`
- Create: `tests/visualization/test_coach_panel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/visualization/test_coach_panel.py`:

```python
"""Tests for AI coach overlay panel."""

import numpy as np

from src.types import ElementPhase, H36Key, MetricResult
from src.visualization.hud.coach_panel import CoachOverlayData, compute_coach_overlays


class TestCoachOverlayData:
    """Test CoachOverlayData dataclass."""

    def test_creation(self):
        data = CoachOverlayData(
            element_name_ru="двойной сальхов",
            metrics=[
                ("Время полёта", "0.37с", True),
                ("Высота", "0.12", True),
                ("Ось", "наклон 25°", False),
            ],
            recommendations=["Работай над осью"],
            landing_frame=100,
            fps=30.0,
            display_duration=4.0,
        )
        assert data.element_name_ru == "двойной сальхов"
        assert len(data.metrics) == 3
        assert data.landing_frame == 100

    def test_is_visible_at_frame(self):
        data = CoachOverlayData(
            element_name_ru="тест",
            metrics=[("м1", "1.0", True)],
            recommendations=[],
            landing_frame=100,
            fps=30.0,
            display_duration=4.0,
        )
        # Not visible before landing
        assert not data.is_visible_at(99)
        # Visible at landing
        assert data.is_visible_at(100)
        # Visible during display window (100 + 4.0*30 = 220)
        assert data.is_visible_at(219)
        # Not visible after display duration
        assert not data.is_visible_at(221)


class TestComputeCoachOverlays:
    """Test analysis pre-computation."""

    def test_compute_from_jump_phases(self):
        """Should produce overlay data from jump phases and metrics."""
        phases = ElementPhase(
            name="jump",
            start=250,
            takeoff=278,
            peak=283,
            landing=289,
            end=300,
        )
        metrics = [
            MetricResult(name="airtime", value=0.37, unit="s", is_good=True, reference_range=(0.3, 0.7)),
            MetricResult(name="max_height", value=0.12, unit="norm", is_good=False, reference_range=(0.2, 0.5)),
        ]
        recommendations = ["Недостаточная высота прыжка"]

        overlays = compute_coach_overlays(
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            element_type="waltz_jump",
            fps=29.9,
        )

        assert len(overlays) == 1
        assert overlays[0].landing_frame == 289
        assert overlays[0].element_name_ru == "вальсовый прыжок"
        assert len(overlays[0].metrics) == 2
        assert overlays[0].recommendations == recommendations

    def test_no_overlay_for_step(self):
        """Should produce no overlay when phases have no takeoff/landing (steps)."""
        phases = ElementPhase(
            name="three_turn",
            start=10,
            takeoff=0,
            peak=20,
            landing=0,
            end=30,
        )
        overlays = compute_coach_overlays(
            phases=phases,
            metrics=[],
            recommendations=[],
            element_type="three_turn",
            fps=30.0,
        )
        assert len(overlays) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/visualization/test_coach_panel.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement CoachOverlayData and compute_coach_overlays**

Create `src/visualization/hud/coach_panel.py`:

```python
"""AI coach overlay panel for element feedback.

Pre-computes analysis results and renders a broadcast-style overlay
showing element name, key metrics, and recommendations in Russian.
"""

from dataclasses import dataclass

from src.analysis.element_defs import ELEMENT_DEFS
from src.types import ElementPhase, MetricResult


# Metric name translations (English key → Russian display)
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
    True: "✓",   # Good
    False: "✗",  # Bad
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
        value_str = f"{metric.value:.2f}с"
    elif metric.unit == "deg":
        value_str = f"{metric.value:.0f}°"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/visualization/test_coach_panel.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/hud/coach_panel.py tests/visualization/test_coach_panel.py
git commit -m "feat(viz): add CoachOverlayData and compute_coach_overlays

Pre-computes analysis results into per-frame overlay data structure.
Russian metric names, quality symbols, visibility window after landing."
```

---

### Task 2: draw_coach_panel rendering function

**Files:**
- Modify: `src/visualization/hud/coach_panel.py` — add `draw_coach_panel()`
- Modify: `tests/visualization/test_coach_panel.py` — add rendering tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/visualization/test_coach_panel.py`:

```python
class TestDrawCoachPanel:
    """Test coach panel rendering."""

    def test_draw_on_empty_frame(self):
        """Should draw panel without errors on a blank frame."""
        from src.visualization.hud.coach_panel import CoachOverlayData, draw_coach_panel

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        data = CoachOverlayData(
            element_name_ru="вальсовый прыжок",
            metrics=[
                ("Время полёта", "0.37с", True),
                ("Высота", "0.12", False),
            ],
            recommendations=["Недостаточная высота"],
            landing_frame=100,
            fps=30.0,
        )
        result = draw_coach_panel(frame, data)
        assert result.shape == frame.shape
        # Should have drawn something (non-zero pixels)
        assert np.any(result > 0)

    def test_draw_returns_same_frame(self):
        """Should modify frame in place and return it."""
        from src.visualization.hud.coach_panel import CoachOverlayData, draw_coach_panel

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        data = CoachOverlayData(
            element_name_ru="тест",
            metrics=[("м1", "1.0", True)],
            recommendations=[],
            landing_frame=0,
            fps=30.0,
        )
        result = draw_coach_panel(frame, data)
        assert result is frame  # Same object (in-place modification)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/visualization/test_coach_panel.py::TestDrawCoachPanel -v`
Expected: FAIL — `draw_coach_panel` not defined

- [ ] **Step 3: Implement draw_coach_panel**

Add to `src/visualization/hud/coach_panel.py` (after imports, before the dataclass):

```python
import numpy as np
from numpy.typing import NDArray

from src.visualization.config import hud_bg_alpha, hud_bg_color, hud_padding
from src.visualization.core.text import render_cyrillic_text


Frame = NDArray[np.uint8]
Position = tuple[int, int]


def draw_coach_panel(
    frame: Frame,
    data: CoachOverlayData,
    position: Position = (10, 90),
    font_size: int = 24,
    line_height: int = 30,
) -> Frame:
    """Draw AI coach overlay panel on frame.

    Renders a broadcast-style panel with:
    - Element name in Russian (bold title)
    - Key metrics with quality indicators (✓/✗)
    - Top recommendation

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        data: Coach overlay data to display.
        position: (x, y) top-left position for the panel.
        font_size: Font size in pixels.
        line_height: Vertical spacing between lines.

    Returns:
        Frame with panel drawn (modified in place).
    """
    x, y = position

    # 1. Element name (title)
    frame = render_cyrillic_text(
        frame,
        data.element_name_ru,
        (x, y),
        font_size=font_size + 4,
        color=(255, 255, 255),
        background=(0, 0, 0),
        background_alpha=0.7,
    )
    y += line_height + 5

    # 2. Metrics with quality symbols
    for name_ru, value_str, is_good in data.metrics:
        symbol = QUALITY_SYMBOLS[is_good]
        color = (0, 220, 0) if is_good else (0, 0, 220)  # BGR: green=good, red=bad
        text = f"{name_ru}: {value_str}  {symbol}"
        frame = render_cyrillic_text(
            frame,
            text,
            (x + 10, y),
            font_size=font_size,
            color=color,
            background=(0, 0, 0),
            background_alpha=0.5,
        )
        y += line_height

    # 3. Recommendations (if any)
    if data.recommendations:
        y += 5
        for rec in data.recommendations:
            frame = render_cyrillic_text(
                frame,
                f"⚠ {rec}",
                (x + 10, y),
                font_size=font_size - 4,
                color=(0, 180, 255),  # BGR: orange
                background=(0, 0, 0),
                background_alpha=0.5,
            )
            y += line_height - 4

    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/visualization/test_coach_panel.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/hud/coach_panel.py tests/visualization/test_coach_panel.py
git commit -m "feat(viz): add draw_coach_panel rendering function

Broadcast-style overlay with Russian element name, metrics with ✓/✗
quality indicators, and orange recommendation text. Uses render_cyrillic_text
for Pillow-based Cyrillic rendering."
```

---

### Task 3: Wire into visualize_with_skeleton.py

**Files:**
- Modify: `scripts/visualize_with_skeleton.py`

This task wires the coach overlay into the main visualization loop. The key insight: the script already has `--element` flag and access to poses. We pre-compute overlays after pose extraction, then render per-frame.

- [ ] **Step 1: Add import**

Near the top of `scripts/visualize_with_skeleton.py` (after existing imports, around line 44), add:

```python
from src.visualization.hud.coach_panel import CoachOverlayData, compute_coach_overlays, draw_coach_panel
```

- [ ] **Step 2: Add pre-computation after pose extraction**

Find where `out_poses` is assembled (the `poses_all` variable after the main extraction loop). After the pose extraction loop completes and before the rendering loop starts, add the overlay pre-computation. Look for the comment or section where `segments` data is available (used for `_get_active_segment`).

Add this block after pose data is finalized (after line ~430, before the rendering loop starts at ~444):

```python
    # --- Pre-compute AI coach overlays ---
    coach_overlays: list[CoachOverlayData] = []
    if args.element and poses_all is not None:
        try:
            from src.analysis.metrics import BiomechanicsAnalyzer
            from src.analysis.phase_detector import PhaseDetector
            from src.analysis.recommender import Recommender
            from src.analysis.element_defs import get_element_def
            from src.utils.geometry import calculate_com_trajectory

            element_def = get_element_def(args.element)
            if element_def:
                detector = PhaseDetector()
                analyzer = BiomechanicsAnalyzer(element_def)
                recommender = Recommender()

                poses_for_analysis = poses_all[:, :, :2].astype(np.float32)
                phase_result = detector.detect_phases(poses_for_analysis, meta.fps, args.element)
                metrics = analyzer.analyze(poses_for_analysis, phase_result.phases, meta.fps)
                recs = recommender.recommend(metrics, args.element)
                coach_overlays = compute_coach_overlays(
                    phases=phase_result.phases,
                    metrics=metrics,
                    recommendations=recs,
                    element_type=args.element,
                    fps=meta.fps,
                )
        except Exception:
            coach_overlays = []  # Non-critical, skip overlay on error
```

- [ ] **Step 3: Add helper to find active coach overlay**

Add a helper function (near `_get_active_segment`):

```python
def _get_coach_overlay(
    overlays: list[CoachOverlayData], frame_idx: int
) -> CoachOverlayData | None:
    """Find the coach overlay visible at the current frame."""
    for overlay in overlays:
        if overlay.is_visible_at(frame_idx):
            return overlay
    return None
```

- [ ] **Step 4: Render coach panel in the frame loop**

Inside `_draw_hud()`, after the existing HUD elements are drawn (after the blade indicators section, around line 730), add:

```python
    # AI coach overlay (passed via element_info)
    coach_overlay = element_info.get("coach_overlay")
    if coach_overlay is not None:
        frame = draw_coach_panel(frame, coach_overlay, position=(10, 90))
```

And in the main frame loop, where `_draw_hud` is called (around line 582), pass the coach overlay:

Change:
```python
frame = _draw_hud(
    frame, active_segment, frame_idx, meta.num_frames, meta.fps,
    draw_h, draw_w, blade_left, blade_right, visible_side, floor_angle
)
```

To:
```python
# Attach coach overlay to element_info if visible
if active_segment is not None:
    active_segment = dict(active_segment)  # copy to avoid mutation
    active_segment["coach_overlay"] = _get_coach_overlay(coach_overlays, frame_idx)

frame = _draw_hud(
    frame, active_segment, frame_idx, meta.num_frames, meta.fps,
    draw_h, draw_w, blade_left, blade_right, visible_side, floor_angle
)
```

Note: `active_segment` may be `None` in some frames. Check how `_draw_hud` handles it currently and ensure the `coach_overlay` key access is safe.

- [ ] **Step 5: Test manually on the 2S video**

```bash
uv run python scripts/visualize_with_skeleton.py out.mp4 \
    --element salchow --pose-backend rtmlib --layer 2 --output /tmp/coach_overlay_test.mp4
```

Expected: After the jump lands (~f289), a Russian overlay panel appears in the upper-left showing "вальсовый прыжок" or element name, metrics with ✓/✗, and recommendations. It stays for ~4 seconds.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: All tests pass, no regressions.

- [ ] **Step 7: Commit**

```bash
git add scripts/visualize_with_skeleton.py
git commit -m "feat(viz): wire AI coach overlay into visualization pipeline

Pre-computes analysis (phases + metrics + recommendations) after pose
extraction, then renders coach panel after element landing for 4 seconds.
Only active when --element flag is provided."
```

---

## Verification Checklist

- [ ] `test_coach_overlay_data_creation` passes
- [ ] `test_is_visible_at_frame` passes — window logic correct
- [ ] `test_compute_from_jump_phases` passes — Russian names, metric formatting
- [ ] `test_no_overlay_for_step` passes — steps produce no overlay
- [ ] `test_draw_on_empty_frame` passes — no crash on blank frame
- [ ] `test_draw_returns_same_frame` passes — in-place modification
- [ ] Manual test: coach overlay appears after landing on 2S video
- [ ] Full `uv run pytest tests/` passes (394+ tests)
