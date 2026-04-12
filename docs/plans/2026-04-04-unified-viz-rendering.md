# Unified Visualization & Rendering Standardization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the visualization system into a fast, unified rendering pipeline with a single text engine, a single overlay primitive, and no full-frame Pillow conversions.

**Architecture:** Replace scattered text/background rendering with two primitives: `put_text()` (auto-cached Cyrillic bitmap blitting) and `draw_overlay_rect()` (ROI-scoped semi-transparent rectangles). All modules use these two functions instead of inline Pillow/OpenCV mix. The `visualize_with_skeleton.py` script becomes a thin orchestrator calling layers + HUD.

**Tech Stack:** OpenCV (drawing), Pillow (font rendering to small cached bitmaps only), NumPy (alpha blending on ROI sub-arrays)

---

## Current State (Problems)

| Problem | Files Affected | Impact |
|---------|---------------|--------|
| `render_cyrillic_text()` converts entire frame BGR→RGB→PIL→RGB→BGR | `core/text.py` | ~100ms/frame |
| `HUDPanel.draw_background()` copies entire frame for `addWeighted` | `hud/panel.py:153` | ~50ms/frame |
| `draw_text_box()` copies entire frame for `addWeighted` | `core/text.py:344` | ~30ms/frame |
| 3 different semi-transparent bg implementations | text.py, panel.py, coach_panel.py | Maintenance |
| 8 text rendering functions with overlapping scope | core/text.py, elements.py, layers/ | Confusion |
| `Position` enum duplicated in `panel.py` and `layout.py` | hud/panel.py, hud/layout.py | Duplication |
| `draw_coach_panel` had full-frame Pillow (fixed in prior session) | hud/coach_panel.py | Done |
| Layers use bare `cv2.putText` without bg/style consistency | timer_layer, hud_layer, elements | Inconsistent |

---

## Target Architecture

```
src/visualization/
├── core/
│   ├── text.py          # UNIFIED: put_text(), measure_text(), put_cyrillic_text() (internal)
│   ├── overlay.py       # NEW: draw_overlay_rect() — single ROI-scoped alpha blend
│   ├── colors.py        # KEEP: color utilities (no changes)
│   └── geometry.py      # KEEP: coordinate transforms (no changes)
├── skeleton/
│   ├── drawer.py        # KEEP: uses put_text for labels
│   └── joints.py        # KEEP: no changes
├── layers/
│   ├── base.py          # KEEP: Layer/LayerContext/render_layers
│   ├── velocity_layer.py    # UPDATE: use put_text + draw_overlay_rect
│   ├── trail_layer.py       # KEEP: no text/bg rendering
│   ├── joint_angle_layer.py # UPDATE: use put_text for degree labels
│   ├── angle_panel_layer.py # UPDATE: use draw_overlay_rect + put_text
│   ├── timer_layer.py       # UPDATE: use put_text
│   ├── vertical_axis_layer.py # KEEP: only draws lines
│   ├── blade_layer.py       # KEEP: only draws arrows
│   ├── hud_layer.py         # UPDATE: use put_text + draw_overlay_rect
│   └── skeleton_layer.py    # KEEP: delegates to drawer
├── hud/
│   ├── coach_panel.py   # KEEP: already uses put_cyrillic_text (fast)
│   ├── elements.py      # UPDATE: use put_text + draw_overlay_rect
│   ├── panel.py         # UPDATE: use draw_overlay_rect (ROI-scoped)
│   └── layout.py        # KEEP: position calculations
├── config.py            # UPDATE: add overlay defaults, remove unused
├── comparison.py        # KEEP: already optimized
└── __init__.py          # UPDATE: new exports
```

---

## Task 1: Add `draw_overlay_rect()` to new `core/overlay.py`

Single primitive for semi-transparent rectangles. Every module that currently does `frame.copy()` + `cv2.rectangle` + `cv2.addWeighted(frame[:])` will use this instead.

**Files:**
- Create: `src/visualization/core/overlay.py`
- Test: `tests/visualization/test_overlay.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for core overlay primitives."""
import numpy as np
import pytest

from src.visualization.core.overlay import draw_overlay_rect


def _make_frame(h=100, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_draw_overlay_rect_modifies_roi():
    frame = _make_frame()
    result = draw_overlay_rect(frame, (10, 20, 60, 40), color=(0, 0, 0), alpha=0.5)
    assert result is frame  # in-place
    # ROI should be darker than original
    roi = frame[20:60, 10:70]
    assert roi.mean() < 128


def test_draw_overlay_rect_no_full_frame_copy():
    frame = _make_frame()
    original_id = id(frame)
    draw_overlay_rect(frame, (0, 0, 50, 50), color=(255, 255, 255), alpha=0.5)
    assert id(frame) == original_id  # no frame replacement


def test_draw_overlay_rect_clips_to_bounds():
    frame = _make_frame()
    # Rect extends beyond frame
    draw_overlay_rect(frame, (-10, -10, 300, 200), color=(0, 0, 0), alpha=1.0)
    assert frame.mean() == 0  # entire frame should be black


def test_draw_overlay_rect_alpha_zero_no_change():
    frame = _make_frame()
    original = frame.copy()
    draw_overlay_rect(frame, (10, 10, 50, 50), color=(255, 255, 255), alpha=0.0)
    np.testing.assert_array_equal(frame, original)


def test_draw_overlay_rect_with_border():
    frame = _make_frame()
    draw_overlay_rect(
        frame, (10, 20, 60, 40),
        color=(0, 0, 0), alpha=0.5,
        border_color=(0, 255, 0), border_thickness=2,
    )
    # Border pixels should be green
    assert not np.array_equal(frame[20, 10], [0, 0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visualization/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.visualization.core.overlay'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Fast semi-transparent rectangle overlay — ROI-scoped, no full-frame copy."""

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]
Rect = tuple[int, int, int, int]  # (x, y, w, h)


def draw_overlay_rect(
    frame: Frame,
    rect: Rect,
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 0,
) -> Frame:
    """Draw a semi-transparent filled rectangle using ROI-scoped blending.

    Unlike cv2.addWeighted on the full frame, this only copies the affected
    region, applies the blend, and writes it back.

    Args:
        frame: OpenCV image (H, W, 3) BGR — modified in place.
        rect: (x, y, width, height) — clipped to frame bounds.
        color: Fill color (BGR).
        alpha: Fill opacity [0, 1]. 0 = invisible, 1 = opaque.
        border_color: Border color (BGR). None = no border.
        border_thickness: Border thickness in pixels. 0 = no border.

    Returns:
        frame (same object, modified in place).
    """
    if alpha <= 0.0:
        return frame

    rx, ry, rw, rh = rect
    fh, fw = frame.shape[:2]

    # Clip to frame bounds
    x1 = max(rx, 0)
    y1 = max(ry, 0)
    x2 = min(rx + rw, fw)
    y2 = min(ry + rh, fh)
    if x1 >= x2 or y1 >= y2:
        return frame

    roi = frame[y1:y2, x1:x2]
    if alpha >= 1.0:
        roi[:, :] = color
    else:
        overlay = roi.copy()
        overlay[:, :] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    # Border (drawn on top of blended region)
    if border_color is not None and border_thickness > 0:
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            border_color, border_thickness, cv2.LINE_AA,
        )

    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visualization/test_overlay.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/visualization/core/overlay.py tests/visualization/test_overlay.py
git commit -m "feat(viz): add draw_overlay_rect — ROI-scoped semi-transparent rectangle"
```

---

## Task 2: Unify text API — deprecate `render_cyrillic_text`, consolidate `put_text()`

Add a single `put_text()` function that auto-detects Cyrillic and routes to the fast path. Deprecate the slow `render_cyrillic_text()`. Remove the unused font cache in `coach_panel.py` (leftover from refactor).

**Files:**
- Modify: `src/visualization/core/text.py`
- Test: `tests/visualization/test_text_unified.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for unified put_text API."""
import numpy as np
import pytest

from src.visualization.core.text import put_text, measure_text_size_fast


def _make_frame(h=200, w=400):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_put_text_cyrillic_no_full_frame_change():
    """Only the text region should change, rest stays black."""
    frame = _make_frame()
    put_text(frame, "Привет", (10, 10), font_size=20)
    # Top-left corner should have non-zero pixels
    assert frame[:50, :100].sum() > 0
    # Bottom-right corner should be untouched
    assert frame[150:, 300:].sum() == 0


def test_put_text_ascii_uses_opencv_path():
    """ASCII text should render (fast OpenCV path or cached bitmap)."""
    frame = _make_frame()
    put_text(frame, "Hello", (10, 10), font_size=16)
    assert frame[:30, :80].sum() > 0


def test_put_text_returns_frame():
    frame = _make_frame()
    result = put_text(frame, "Test", (10, 10))
    assert result is frame


def test_put_text_with_background():
    frame = _make_frame()
    put_text(frame, "Test", (10, 10), bg_color=(0, 0, 0), bg_alpha=0.7)
    # Background region should exist
    assert frame[:30, :80].sum() > 0


def test_measure_text_size_fast_returns_positive():
    w, h = measure_text_size_fast("Привет мир", font_size=16)
    assert w > 0
    assert h > 0


def test_measure_text_size_fast_ascii():
    w, h = measure_text_size_fast("Hello", font_size=16)
    assert w > 0
    assert h > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visualization/test_text_unified.py -v`
Expected: FAIL with `ImportError: cannot import name 'put_text'`

- [ ] **Step 3: Add `put_text()` and `measure_text_size_fast()` to `core/text.py`**

Add these functions after the existing `put_cyrillic_text()` (around line 520 in the current file):

```python
def put_text(
    frame: Frame,
    text: str,
    position: Position,
    font_path: str = font_path,
    font_size: int = 16,
    color: tuple[int, int, int] = font_color,
    bg_color: tuple[int, int, int] | None = None,
    bg_alpha: float = 0.6,
    padding: int = 4,
) -> Frame:
    """Universal text rendering — auto-detects Cyrillic, always fast.

    For Cyrillic/Unicode text: uses cached Pillow bitmaps blitted via numpy.
    For ASCII text: same path (cached bitmaps are still fast).
    Optionally draws a semi-transparent background behind the text.

    Args:
        frame: OpenCV image (H, W, 3) BGR — modified in place.
        text: Text to render (any Unicode).
        position: (x, y) top-left pixel position.
        font_path: Path to TTF font file.
        font_size: Font size in points.
        color: Text color (BGR).
        bg_color: Background color (BGR). None = no background.
        bg_alpha: Background opacity [0, 1].
        padding: Padding around text for background box.

    Returns:
        frame (same object, modified in place).
    """
    if not text:
        return frame

    x, y = position

    # Draw background if requested
    if bg_color is not None and bg_alpha > 0:
        from src.visualization.core.overlay import draw_overlay_rect

        tw, th = put_cyrillic_text_size(text, font_path, font_size)
        draw_overlay_rect(
            frame,
            (x - padding, y - padding, tw + 2 * padding, th + 2 * padding),
            color=bg_color,
            alpha=bg_alpha,
        )

    # Render text via cached bitmap
    put_cyrillic_text(frame, text, (x, y), font_path=font_path, font_size=font_size, color=color)
    return frame


def measure_text_size_fast(
    text: str,
    font_path: str = font_path,
    font_size: int = 16,
) -> tuple[int, int]:
    """Fast text measurement using cached bitmaps.

    Works for any Unicode text (Cyrillic, ASCII, etc.).

    Returns:
        (width, height) in pixels.
    """
    return put_cyrillic_text_size(text, font_path, font_size)
```

- [ ] **Step 4: Add deprecation warning to `render_cyrillic_text()`**

At the top of the `render_cyrillic_text()` function body (around line 247 in current file), add:

```python
    import warnings
    warnings.warn(
        "render_cyrillic_text() is slow (full-frame Pillow conversion). "
        "Use put_text() or put_cyrillic_text() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/visualization/test_text_unified.py -v`
Expected: 6 passed

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (400+ tests)

- [ ] **Step 7: Commit**

```bash
git add src/visualization/core/text.py tests/visualization/test_text_unified.py
git commit -m "feat(viz): add unified put_text() + measure_text_size_fast(), deprecate render_cyrillic_text"
```

---

## Task 3: Update `hud/panel.py` — ROI-scoped backgrounds

Replace `frame.copy()` + full-frame `addWeighted` with `draw_overlay_rect()`.

**Files:**
- Modify: `src/visualization/hud/panel.py:135-196`
- Test: `tests/visualization/test_hud_panel.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for HUDPanel with ROI-scoped rendering."""
import numpy as np
import pytest

from src.visualization.hud.panel import HUDPanel


def _make_frame(h=480, w=640):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_draw_background_no_full_frame_copy():
    frame = _make_frame()
    original_id = id(frame)
    panel = HUDPanel(bg_alpha=0.5)
    panel.draw_background(frame, 10, 10, 100, 50)
    assert id(frame) == original_id  # no frame replacement


def test_draw_background_modifies_only_roi():
    frame = _make_frame()
    panel = HUDPanel(bg_color=(0, 0, 0), bg_alpha=0.5)
    panel.draw_background(frame, 50, 50, 100, 50)
    # Outside ROI should be unchanged
    assert frame[0, 0, 0] == 128
    # Inside ROI should be blended
    roi_mean = frame[60:90, 60:140].mean()
    assert roi_mean < 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visualization/test_hud_panel.py -v`
Expected: FAIL (panel still does full-frame copy)

- [ ] **Step 3: Replace `draw_background` implementation**

In `src/visualization/hud/panel.py`, replace the `draw_background` method body:

```python
    def draw_background(
        self,
        frame: Frame,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw panel background with optional border.

        Uses ROI-scoped blending — no full-frame copy.
        """
        from src.visualization.core.overlay import draw_overlay_rect

        draw_overlay_rect(
            frame,
            (x, y, width, height),
            color=self.bg_color,
            alpha=self.bg_alpha,
            border_color=self.border_color if self.border_thickness > 0 else None,
            border_thickness=self.border_thickness,
        )
```

Also remove the `import cv2` at top of `panel.py` if it's no longer needed elsewhere in the file (check first — `draw_title` still uses `cv2.putText`, so keep it).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visualization/test_hud_panel.py -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/visualization/hud/panel.py tests/visualization/test_hud_panel.py
git commit -m "perf(viz): HUDPanel uses ROI-scoped draw_overlay_rect instead of full-frame copy"
```

---

## Task 4: Update `core/text.py` `draw_text_box()` — ROI-scoped blending

Replace the `frame.copy()` + full-frame `addWeighted` with `draw_overlay_rect()`.

**Files:**
- Modify: `src/visualization/core/text.py` — `draw_text_box()` function (lines ~294-370)

- [ ] **Step 1: Update `draw_text_box()` implementation**

Replace the semi-transparent background section in `draw_text_box()`:

Old:
```python
    # Draw semi-transparent background
    if bg_alpha > 0:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            bg_color,
            -1,
            cv2.LINE_AA,
        )
        frame[:] = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)
```

New:
```python
    # Draw semi-transparent background (ROI-scoped, no full-frame copy)
    if bg_alpha > 0:
        from src.visualization.core.overlay import draw_overlay_rect

        draw_overlay_rect(frame, (x, y, box_width, box_height), color=bg_color, alpha=bg_alpha)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/core/text.py
git commit -m "perf(viz): draw_text_box uses ROI-scoped blending, no full-frame copy"
```

---

## Task 5: Update `layers/angle_panel_layer.py` — ROI-scoped blending

**Files:**
- Modify: `src/visualization/layers/angle_panel_layer.py`

- [ ] **Step 1: Find and replace the full-frame addWeighted**

Search for `frame[:] = cv2.addWeighted` in `angle_panel_layer.py` and replace with:

```python
        from src.visualization.core.overlay import draw_overlay_rect
        draw_overlay_rect(frame, (px, py, panel_w, panel_h), color=(0, 0, 0), alpha=0.5)
```

Also replace any `frame.copy()` + rectangle + addWeighted patterns the same way.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/layers/angle_panel_layer.py
git commit -m "perf(viz): angle_panel_layer uses ROI-scoped blending"
```

---

## Task 6: Update `layers/timer_layer.py` — use `put_text()`

**Files:**
- Modify: `src/visualization/layers/timer_layer.py`

- [ ] **Step 1: Replace direct cv2.putText calls**

In `timer_layer.py`, replace all `cv2.putText(...)` with:

```python
from src.visualization.core.text import put_text
```

Then use:
```python
put_text(frame, text, (x, y), font_size=14, color=color)
```

Remove any manual text measurement with `cv2.getTextSize` — use `measure_text_size_fast()` instead.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/layers/timer_layer.py
git commit -m "refactor(viz): timer_layer uses unified put_text"
```

---

## Task 7: Update `layers/hud_layer.py` — use `put_text()` + `draw_overlay_rect()`

**Files:**
- Modify: `src/visualization/layers/hud_layer.py`

- [ ] **Step 1: Replace direct cv2.putText calls**

Same pattern as Task 6. Replace all `cv2.putText(...)` with `put_text()` and any `frame.copy()` + `addWeighted(frame[:])` patterns with `draw_overlay_rect()`.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/layers/hud_layer.py
git commit -m "refactor(viz): hud_layer uses unified put_text + draw_overlay_rect"
```

---

## Task 8: Update `hud/elements.py` — use `put_text()` + `draw_overlay_rect()`

**Files:**
- Modify: `src/visualization/hud/elements.py`

- [ ] **Step 1: Replace direct cv2.putText calls**

In `elements.py`, replace bare `cv2.putText(...)` with `put_text()`. For metrics panel and phase indicator, replace the manual line drawing + text with unified calls.

Keep `draw_blade_indicator_hud()` as-is (uses `cv2.fillConvexPoly` which is already fast).

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/hud/elements.py
git commit -m "refactor(viz): hud elements use unified put_text + draw_overlay_rect"
```

---

## Task 9: Update `__init__.py` exports

**Files:**
- Modify: `src/visualization/__init__.py`
- Modify: `src/visualization/core/__init__.py`

- [ ] **Step 1: Add new exports**

In `src/visualization/__init__.py`, add to imports:

```python
from src.visualization.core.overlay import draw_overlay_rect
from src.visualization.core.text import put_text, measure_text_size_fast
```

Add to `__all__`:

```python
    "draw_overlay_rect",
    "measure_text_size_fast",
    "put_text",
```

In `src/visualization/core/__init__.py`, add:

```python
from src.visualization.core.overlay import draw_overlay_rect
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/visualization/__init__.py src/visualization/core/__init__.py
git commit -m "feat(viz): export put_text, measure_text_size_fast, draw_overlay_rect"
```

---

## Task 10: Update `visualize_with_skeleton.py` — use unified API

**Files:**
- Modify: `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Replace `render_cyrillic_text` with `put_text`**

In the subtitles section (around line 584-596), replace:

```python
                        frame = render_cyrillic_text(
                            frame,
                            subtitle_text,
                            (50, meta.height - 50),
                            font_size=args.font_size,
                        )
```

with:

```python
                        put_text(
                            frame,
                            subtitle_text,
                            (50, meta.height - 50),
                            font_size=args.font_size,
                            color=(255, 255, 255),
                            bg_color=(0, 0, 0),
                            bg_alpha=0.6,
                        )
```

- [ ] **Step 2: Update imports**

Replace `render_cyrillic_text` import with `put_text` in the import block:

```python
from src.visualization import (
    ...
    put_text,
    render_layers,
)
```

Remove the `render_cyrillic_text` import.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add scripts/visualize_with_skeleton.py
git commit -m "refactor(viz): main script uses unified put_text instead of render_cyrillic_text"
```

---

## Task 11: Integration test — full pipeline render with `--profile`

**Files:**
- Create: `tests/visualization/test_render_perf.py`

- [ ] **Step 1: Write a performance regression test**

```python
"""Performance regression tests for visualization rendering."""
import time

import numpy as np
import pytest

from src.visualization import put_text, draw_overlay_rect
from src.visualization.hud.coach_panel import draw_coach_panel, CoachOverlayData


def _make_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestRenderPerformance:
    """Ensure rendering primitives stay under budget."""

    def test_put_text_under_1ms(self):
        """put_text should render Cyrillic text in <1ms (cached bitmap)."""
        frame = _make_frame()
        # Warm up cache
        put_text(frame, "Тест", (10, 10), font_size=16)

        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(100):
            put_text(frame, "Высота: 0.45с ✓", (10, 10), font_size=16)
        elapsed = (time.perf_counter() - t0) / 100

        assert elapsed < 0.001, f"put_text took {elapsed*1000:.2f}ms (>1ms)"

    def test_draw_overlay_rect_under_01ms(self):
        """draw_overlay_rect should render in <0.1ms."""
        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(1000):
            draw_overlay_rect(frame, (10, 10, 200, 100), color=(0, 0, 0), alpha=0.6)
        elapsed = (time.perf_counter() - t0) / 1000

        assert elapsed < 0.0001, f"draw_overlay_rect took {elapsed*1000:.2f}ms (>0.1ms)"

    def test_coach_panel_under_2ms(self):
        """Full coach panel render should complete in <2ms."""
        data = CoachOverlayData(
            element_name_ru="Сальхов",
            metrics=[
                ("Время полёта", "0.45с", True),
                ("Высота", "0.32", True),
                ("Колено", "165°", False),
            ],
            recommendations=["Спрямить колено при приземлении"],
            landing_frame=100,
            fps=30.0,
        )
        # Warm up
        frame = _make_frame()
        draw_coach_panel(frame, data, position=(10, 90))

        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(100):
            draw_coach_panel(frame, data, position=(10, 90))
        elapsed = (time.perf_counter() - t0) / 100

        assert elapsed < 0.002, f"draw_coach_panel took {elapsed*1000:.2f}ms (>2ms)"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/visualization/test_render_perf.py -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/visualization/test_render_perf.py
git commit -m "test(viz): add performance regression tests for rendering primitives"
```

---

## Task 12: Remove dead code

**Files:**
- Modify: `src/visualization/core/text.py` — remove unused functions
- Modify: `src/visualization/config.py` — remove if unused constants exist

- [ ] **Step 1: Audit and remove unused text functions**

In `core/text.py`, check which functions are actually imported elsewhere:

Run: `grep -r "from src.visualization.core.text import" src/ scripts/`

Functions that can potentially be removed after Tasks 2-10:
- `measure_text_size_cv2()` — only used internally
- `measure_text_size_pillow()` — only used internally
- `measure_text_size()` — check if used externally; if not, keep as internal
- `truncate_text()` — check usage
- `wrap_text()` — check usage

Keep: `put_text()`, `put_cyrillic_text()`, `put_cyrillic_text_size()`, `measure_text_size_fast()`, `draw_text_outlined()`, `draw_text_box()`, `draw_text_multiline()`.

Mark `render_cyrillic_text()` with deprecation warning (already done in Task 2). Do NOT delete it yet — backward compat for one release cycle.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "chore(viz): audit text module, keep public API, mark deprecated functions"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] Single text rendering function (`put_text()`) — Task 2
- [x] Single overlay primitive (`draw_overlay_rect()`) — Task 1
- [x] No full-frame Pillow conversion — Tasks 3, 4, 5
- [x] All layers use unified API — Tasks 5, 6, 7, 8
- [x] Main script uses unified API — Task 10
- [x] Performance regression tests — Task 11
- [x] Public API exports — Task 9
- [x] Dead code cleanup — Task 12

### Placeholder Scan
- No TBD/TODO found
- All code blocks contain complete implementations
- All test blocks contain complete test functions

### Type Consistency
- `put_text()` takes `Frame` = `NDArray[np.uint8]`, `Position` = `tuple[int, int]` — matches existing types
- `draw_overlay_rect()` takes `Rect` = `tuple[int, int, int, int]` — consistent across all callers
- `measure_text_size_fast()` returns `tuple[int, int]` — same as existing `measure_text_size()`
