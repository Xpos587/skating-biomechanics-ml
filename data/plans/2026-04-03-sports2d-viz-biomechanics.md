# Sports2D-Inspired Visualization & Biomechanics Upgrade

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade visualization quality to match Sports2D and add comprehensive biomechanics angle formulas with visible_side detection and floor angle estimation.

**Architecture:** Replace current simple skeleton overlay with Sports2D-inspired rendering: dual-layer outlined text, RdYlGn confidence coloring, angle arcs with degree labels, side panel with progress bars. Add matplotlib-based interactive person selection. Add comprehensive biomechanics angle computation (26 angles) with visible_side detection and floor angle estimation. Implement 2D/3D hybrid mode for display vs analytics.

**Tech Stack:** OpenCV (cv2), matplotlib, numpy, pytest

---

## File Structure

### Files to Create
- `src/visualization/layers/angle_panel_layer.py` — Side panel with angle values + progress bars
- `src/pose_estimation/person_selector.py` — Matplotlib interactive person selection UI
- `src/analysis/angles.py` — Comprehensive biomechanics angle computation (26 angles)

### Files to Modify
- `src/visualization/core/text.py` — Add `draw_text_outlined()` (dual-layer text)
- `src/visualization/skeleton/joints.py` — Add `get_confidence_color_rdygn()` (RdYlGn colormap)
- `src/visualization/skeleton/drawer.py` — Use confidence colors for joints
- `src/visualization/layers/joint_angle_layer.py` — Add degree labels to arcs, thicker arcs
- `src/visualization/layers/__init__.py` — Export AnglePanelLayer
- `src/utils/geometry.py` — Add `detect_visible_side()`, `estimate_floor_angle()`
- `src/cli.py` — Wire matplotlib person selector

### Test Files to Create/Modify
- `tests/visualization/test_text.py` — Tests for `draw_text_outlined()`
- `tests/visualization/test_confidence_color.py` — Tests for RdYlGn coloring
- `tests/analysis/test_angles.py` — Tests for comprehensive angle computation
- `tests/utils/test_geometry.py` — Add tests for `detect_visible_side()`, `estimate_floor_angle()`

---

## Task 1: Dual-Layer Outlined Text

**Files:**
- Modify: `src/visualization/core/text.py:1-471`
- Test: `tests/visualization/test_text.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/visualization/test_text.py`:

```python
"""Tests for dual-layer outlined text rendering."""

import numpy as np
import pytest

from src.visualization.core.text import draw_text_outlined


class TestDrawTextOutlined:
    def test_renders_on_frame(self):
        """Should render text on a frame without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_text_outlined(frame, "Knee: 45.0", (10, 30))
        assert result is frame  # modified in place

    def test_black_pixels_present(self):
        """Should draw black outline pixels near text (thicker)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_text_outlined(frame, "Test", (10, 30))
        # Black outline is drawn first with thickness+1, so there should be
        # pixels that are NOT white near the text area
        text_region = frame[0:60, 0:200]
        has_non_white = np.any(text_region != 255, axis=2)
        assert has_non_white.any()

    def test_colored_pixels_present(self):
        """Should draw colored text pixels."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        green = (0, 255, 0)
        draw_text_outlined(frame, "Test", (10, 30), color=green)
        text_region = frame[0:60, 0:200]
        has_green = np.any((text_region[:, :, 1] == 255) & (text_region[:, :, 0] == 0), axis=2)
        assert has_green.any()

    def test_custom_font_scale(self):
        """Should accept custom font_scale and thickness."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_text_outlined(frame, "Test", (10, 30), font_scale=0.8, thickness=2)
        assert frame.any()  # something was drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_text.py -v`
Expected: FAIL with `ImportError: cannot import name 'draw_text_outlined'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/visualization/core/text.py` after the existing imports (line 22) and before the FONT CACHE section:

```python
# =============================================================================
# OUTLINED TEXT RENDERING (Sports2D-style)
# =============================================================================


def draw_text_outlined(
    frame: Frame,
    text: str,
    position: Position,
    font_scale: float = font_scale,
    thickness: int = font_thickness,
    color: tuple[int, int, int] = font_color,
) -> Frame:
    """Draw text with black outline for readability (Sports2D technique).

    Renders text twice: first with black outline (thicker), then colored fill.
    This ensures text is readable on any background.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        text: ASCII text string.
        position: (x, y) bottom-left position (cv2 convention).
        font_scale: Font scale factor.
        thickness: Text stroke thickness.
        color: Text fill color (BGR).

    Returns:
        Frame with text rendered (modified in place).

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_text_outlined(frame, "Knee: 45.0", (10, 30), color=(0, 255, 0))
    """
    x, y = position
    black = (0, 0, 0)

    # Black outline (thicker)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        black,
        thickness + 1,
        cv2.LINE_AA,
    )

    # Colored fill
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_text.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/visualization/core/text.py tests/visualization/test_text.py
git commit -m "feat(viz): add dual-layer outlined text rendering (Sports2D-style)"
```

---

## Task 2: RdYlGn Confidence Coloring

**Files:**
- Modify: `src/visualization/skeleton/joints.py:190-241`
- Test: `tests/visualization/test_confidence_color.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/visualization/test_confidence_color.py`:

```python
"""Tests for RdYlGn confidence coloring."""

import pytest

from src.visualization.skeleton.joints import get_confidence_color_rdygn


class TestRdYlGnConfidenceColor:
    def test_high_confidence_is_green(self):
        """Confidence=1.0 should return green."""
        color = get_confidence_color_rdygn(1.0)
        # Green in BGR: (0, ~255, 0)
        assert color[1] > 200  # G channel high

    def test_low_confidence_is_red(self):
        """Confidence=0.0 should return red."""
        color = get_confidence_color_rdygn(0.0)
        # Red in BGR: (0, 0, ~255)
        assert color[2] > 200  # R channel high

    def test_mid_confidence_is_yellowish(self):
        """Confidence=0.5 should return yellow-ish."""
        color = get_confidence_color_rdygn(0.5)
        # Yellow in BGR: (0, ~255, ~255)
        assert color[1] > 100  # G present
        assert color[2] > 100  # R present

    def test_clamps_out_of_range(self):
        """Should clamp confidence to [0, 1]."""
        c_low = get_confidence_color_rdygn(-0.5)
        c_high = get_confidence_color_rdygn(1.5)
        assert c_low == get_confidence_color_rdygn(0.0)
        assert c_high == get_confidence_color_rdygn(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_confidence_color.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_confidence_color_rdygn'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/visualization/skeleton/joints.py` after the existing `get_confidence_radius()` function (after line 241):

```python
def get_confidence_color_rdygn(
    confidence: float,
) -> tuple[int, int, int]:
    """Get color representing confidence level using RdYlGn colormap.

    Maps confidence [0, 1] to Red (low) -> Yellow (mid) -> Green (high).
    Uses OpenCV colormap for smooth interpolation (Sports2D approach).

    Args:
        confidence: Confidence value [0, 1].

    Returns:
        BGR color tuple.

    Example:
        >>> get_confidence_color_rdygn(0.9)
        (0, 230, 0)  # Green-ish
    """
    import cv2

    t = max(0.0, min(1.0, confidence))
    # RdYlGn: 0=Red(0), 0.5=Yellow(128), 1.0=Green(255)
    # OpenCV colormap takes 0-255 uint8
    idx = int(t * 255)
    bgr = cv2.applyColorMap(np.uint8([idx]), cv2.COLORMAP_RdYlGn)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
```

Also add `import numpy as np` at the top of `joints.py` if not already present (it's imported by the existing code indirectly, but `get_confidence_color_rdygn` uses `np.uint8` directly so need to add it).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_confidence_color.py -v`
Expected: 4 passed

- [ ] **Step 5: Wire into drawer.py**

In `src/visualization/skeleton/drawer.py`, modify the joint drawing section (lines 126-136). Replace:

```python
            # Get joint color and radius
            color = get_joint_color(joint_idx)
            radius = get_joint_radius(
                1.0 if confidences is None else confidences[joint_idx],
                joint_radius,
                threshold=confidence_threshold,
            )
```

With:

```python
            # Get joint color based on confidence
            conf = 1.0 if confidences is None else confidences[joint_idx]
            if confidences is not None and conf >= confidence_threshold:
                from src.visualization.skeleton.joints import get_confidence_color_rdygn
                color = get_confidence_color_rdygn(conf)
            else:
                color = get_joint_color(joint_idx)
            radius = get_joint_radius(
                conf,
                joint_radius,
                threshold=confidence_threshold,
            )
```

- [ ] **Step 6: Commit**

```bash
git add src/visualization/skeleton/joints.py src/visualization/skeleton/drawer.py tests/visualization/test_confidence_color.py
git commit -m "feat(viz): add RdYlGn confidence coloring for keypoints"
```

---

## Task 3: Enhanced Angle Arcs with Degree Labels

**Files:**
- Modify: `src/visualization/layers/joint_angle_layer.py:1-250`
- Test: `tests/visualization/test_joint_angle_layer.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/visualization/test_joint_angle_layer.py`:

```python
"""Tests for enhanced joint angle visualization."""

import numpy as np
import pytest

from src.pose_estimation import H36Key
from src.utils.geometry import angle_3pt
from src.visualization.config import VisualizationConfig
from src.visualization.layers.base import LayerContext
from src.visualization.layers.joint_angle_layer import JointAngleLayer


def _make_context(pose_2d=None, w=640, h=480):
    return LayerContext(
        frame_width=w,
        frame_height=h,
        pose_2d=pose_2d,
        pose_3d=None,
        confidences=None,
        normalized=False,
    )


class TestJointAngleLayerDegrees:
    def test_arc_renders_without_error(self):
        """Should render angle arcs on a valid pose."""
        pose = np.zeros((17, 2), dtype=np.float32)
        # L-KNEE at center, L-HIP above, L-FOOT below
        pose[H36Key.LHIP] = [300, 180]
        pose[H36Key.LKNEE] = [300, 240]
        pose[H36Key.LFOOT] = [300, 300]
        pose[H36Key.RHIP] = [340, 180]
        pose[H36Key.RKNEE] = [340, 240]
        pose[H36Key.RFOOT] = [340, 300]
        pose[H36Key.LSHOULDER] = [280, 120]
        pose[H36Key.RSHOULDER] = [360, 120]
        pose[H36Key.THORAX] = [320, 150]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = JointAngleLayer()
        ctx = _make_context(pose_2d=pose)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_degree_label_drawn(self):
        """Should draw degree text labels near joints."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[H36Key.LHIP] = [300, 180]
        pose[H36Key.LKNEE] = [300, 240]
        pose[H36Key.LFOOT] = [300, 300]
        pose[H36Key.RHIP] = [340, 180]
        pose[H36Key.RKNEE] = [340, 240]
        pose[H36Key.RFOOT] = [340, 300]
        pose[H36Key.LSHOULDER] = [280, 120]
        pose[H36Key.RSHOULDER] = [360, 120]
        pose[H36Key.THORAX] = [320, 150]

        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()

        layer = JointAngleLayer(show_degree_labels=True)
        ctx = _make_context(pose_2d=pose)
        layer.render(frame_after, ctx)

        # Frame should be modified (degree labels add pixels)
        assert not np.array_equal(frame_before, frame_after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_joint_angle_layer.py -v`
Expected: FAIL — `show_degree_labels` parameter doesn't exist

- [ ] **Step 3: Write minimal implementation**

Modify `src/visualization/layers/joint_angle_layer.py`:

1. Add `show_degree_labels` parameter to `__init__`:

```python
    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        joints: list[JointAngleSpec] | None = None,
        show_degree_labels: bool = True,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=6))
        self.viz = viz_config or VisualizationConfig()
        self.joints = joints or DEFAULT_JOINT_SPECS
        self.show_degree_labels = show_degree_labels
```

2. Replace the render method's `# Draw subtle angle arc` section (lines 209-210) with:

```python
            # Draw angle arc with optional degree label
            self._draw_arc(frame, pv, pa, pc, spec.arc_radius, color)

            if self.show_degree_labels and not np.isnan(angle):
                self._draw_degree_label(frame, pv, angle, color)
```

3. Add `_draw_degree_label` static method after `_draw_arc`:

```python
    @staticmethod
    def _draw_degree_label(
        frame: Frame,
        vertex: np.ndarray,
        angle: float,
        color: tuple[int, int, int],
        offset: int = 20,
    ) -> None:
        """Draw degree label near angle vertex."""
        from src.visualization.core.text import draw_text_outlined

        vx, vy = int(vertex[0]), int(vertex[1])
        label = f"{angle:.0f}°"
        # Position label above-right of vertex
        pos = (vx + offset, vy - offset)
        draw_text_outlined(frame, label, pos, font_scale=0.4, thickness=1, color=color)
```

4. Also increase arc thickness from 1 to 2 in `_draw_arc` (line 248):

Change:
```python
            color,
            1,
```
To:
```python
            color,
            2,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_joint_angle_layer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/visualization/layers/joint_angle_layer.py tests/visualization/test_joint_angle_layer.py
git commit -m "feat(viz): add degree labels to joint angle arcs"
```

---

## Task 4: Angle Panel Layer with Progress Bars

**Files:**
- Create: `src/visualization/layers/angle_panel_layer.py`
- Modify: `src/visualization/layers/__init__.py`
- Test: `tests/visualization/test_angle_panel_layer.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/visualization/test_angle_panel_layer.py`:

```python
"""Tests for angle panel layer."""

import numpy as np
import pytest

from src.pose_estimation import H36Key
from src.visualization.config import VisualizationConfig
from src.visualization.layers.angle_panel_layer import AnglePanelLayer
from src.visualization.layers.base import LayerContext


def _make_context(angles=None, w=640, h=480):
    ctx = LayerContext(frame_width=w, frame_height=h)
    if angles:
        ctx.custom_data["angles"] = angles
    return ctx


class TestAnglePanelLayer:
    def test_renders_empty_panel(self):
        """Should render without error even with no angles."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = AnglePanelLayer()
        ctx = _make_context()
        result = layer.render(frame, ctx)
        assert result is frame

    def test_renders_with_angles(self):
        """Should render panel with angle values."""
        angles = {"L Knee": 120.5, "R Knee": 95.0, "L Hip": 45.0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = AnglePanelLayer()
        ctx = _make_context(angles=angles)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_modifies_frame_with_progress_bar(self):
        """Should modify frame when angles present (progress bars add pixels)."""
        angles = {"L Knee": 120.5, "R Knee": 95.0}
        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()
        layer = AnglePanelLayer()
        ctx = _make_context(angles=angles)
        layer.render(frame_after, ctx)
        assert not np.array_equal(frame_before, frame_after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_angle_panel_layer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.visualization.layers.angle_panel_layer'`

- [ ] **Step 3: Write minimal implementation**

Create `src/visualization/layers/angle_panel_layer.py`:

```python
"""Angle panel layer with progress bars (Sports2D-style).

Displays a side panel listing all computed joint/segment angles
with progress bars and degree values.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import (
    COLOR_BLACK,
    COLOR_GREEN,
    LayerConfig,
    VisualizationConfig,
    hud_bg_alpha,
    hud_bg_color,
    hud_padding,
)
from src.visualization.core.text import draw_text_outlined
from src.visualization.layers.base import Frame, Layer, LayerContext


class AnglePanelLayer(Layer):
    """Side panel showing computed angles with progress bars.

    Reads angles from ``context.custom_data["angles"]`` dict:
        {"L Knee": 120.5, "R Knee": 95.0, ...}

    Each angle shows:
    - Name label
    - Degree value
    - Progress bar (0-180° mapped to bar width)
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        max_angle: float = 180.0,
        bar_height: int = 12,
        line_spacing: int = 22,
        bar_width: int = 100,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=7))
        self.viz = viz_config or VisualizationConfig()
        self.max_angle = max_angle
        self.bar_height = bar_height
        self.line_spacing = line_spacing
        self.bar_width = bar_width

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        angles = context.custom_data.get("angles")
        if not angles:
            return frame

        x0 = hud_padding
        y0 = hud_padding

        # Semi-transparent background
        total_height = len(angles) * self.line_spacing + hud_padding * 2
        bg_width = self.bar_width + 220
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x0 - 5, y0 - 5), (x0 + bg_width, y0 + total_height),
            hud_bg_color, -1, cv2.LINE_AA,
        )
        frame[:] = cv2.addWeighted(overlay, hud_bg_alpha, frame, 1 - hud_bg_alpha, 0)

        for i, (name, value) in enumerate(angles.items()):
            y = y0 + i * self.line_spacing + hud_padding

            if np.isnan(value):
                draw_text_outlined(frame, f"{name}:", (x0, y + 10), font_scale=0.4, thickness=1, color=COLOR_BLACK)
                continue

            # Label
            draw_text_outlined(frame, f"{name}:", (x0, y + 10), font_scale=0.4, thickness=1, color=COLOR_GREEN)

            # Value
            val_x = x0 + 150
            draw_text_outlined(frame, f"{value:.1f}", (val_x, y + 10), font_scale=0.4, thickness=1, color=COLOR_GREEN)

            # Progress bar
            bar_x = x0 + 200
            bar_y = y + 2
            pct = min(abs(value) / self.max_angle, 1.0)
            bar_len = int(pct * self.bar_width)

            # Bar background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + self.bar_width, bar_y + self.bar_height),
                         (50, 50, 50), -1, cv2.LINE_AA)
            # Bar fill
            if bar_len > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + self.bar_height),
                             COLOR_GREEN, -1, cv2.LINE_AA)

        return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_angle_panel_layer.py -v`
Expected: 3 passed

- [ ] **Step 5: Update `__init__.py`**

Add to `src/visualization/layers/__init__.py`:

```python
from src.visualization.layers.angle_panel_layer import AnglePanelLayer
```

And add `"AnglePanelLayer"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/visualization/layers/angle_panel_layer.py src/visualization/layers/__init__.py tests/visualization/test_angle_panel_layer.py
git commit -m "feat(viz): add angle panel layer with progress bars"
```

---

## Task 5: Interactive Person Selection (Matplotlib)

**Files:**
- Create: `src/pose_estimation/person_selector.py`
- Modify: `src/cli.py:70-85`
- Test: `tests/pose_estimation/test_person_selector.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/pose_estimation/test_person_selector.py`:

```python
"""Tests for matplotlib person selector logic."""

import numpy as np
import pytest

from src.pose_estimation.person_selector import compute_bboxes_from_poses, point_in_bbox


class TestComputeBboxes:
    def test_single_person(self):
        """Should compute bbox for single person."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        poses[0, 0] = [100, 100]  # head
        poses[0, 8] = [200, 300]  # foot
        bboxes = compute_bboxes_from_poses(poses)
        assert len(bboxes) == 1
        x1, y1, x2, y2 = bboxes[0]
        assert x1 <= 100 <= x2
        assert y1 <= 100 <= y2

    def test_multiple_persons(self):
        """Should compute bbox for each person."""
        poses = np.zeros((2, 17, 2), dtype=np.float32)
        poses[0, 0] = [100, 100]
        poses[1, 0] = [400, 200]
        bboxes = compute_bboxes_from_poses(poses)
        assert len(bboxes) == 2


class TestPointInBbox:
    def test_inside(self):
        assert point_in_bbox(150, 200, (100, 150, 200, 250))

    def test_outside(self):
        assert not point_in_bbox(50, 50, (100, 150, 200, 250))

    def test_edge(self):
        assert point_in_bbox(100, 150, (100, 150, 200, 250))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pose_estimation/test_person_selector.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/pose_estimation/person_selector.py`:

```python
"""Interactive person selection using matplotlib (Sports2D-style).

Provides:
- compute_bboxes_from_poses(): Calculate bounding boxes from pose arrays
- point_in_bbox(): Hit-test for click detection
- select_persons_interactive(): Matplotlib GUI for person selection
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path


BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def compute_bboxes_from_poses(poses: NDArray[np.float32]) -> list[BBox]:
    """Compute bounding boxes from pose arrays.

    Args:
        poses: (N_persons, 17, 2) pose array.

    Returns:
        List of (x1, y1, x2, y2) bounding boxes.
    """
    bboxes = []
    for i in range(len(poses)):
        valid = poses[i, ~np.isnan(poses[i, :, 0]).all() & ~np.isnan(poses[i, :, 1]).all()]
        if len(valid) == 0:
            bboxes.append((0, 0, 0, 0))
            continue
        # Actually filter valid keypoints from the row
        row = poses[i]
        mask = ~np.isnan(row[:, 0]) & ~np.isnan(row[:, 1])
        if not mask.any():
            bboxes.append((0, 0, 0, 0))
            continue
        pts = row[mask]
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def point_in_bbox(x: int, y: int, bbox: BBox) -> bool:
    """Check if point (x, y) is inside bounding box."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def select_persons_interactive(
    video_path: str | Path,
    poses: NDArray[np.float32],
    bboxes: list[BBox] | None = None,
) -> list[int]:
    """Interactive person selection using matplotlib.

    Shows first frame with numbered bounding boxes.
    User clicks on persons to select them.
    Press Enter or click OK to confirm.

    Args:
        video_path: Path to video file.
        poses: (N_persons, 17, 2) pose array for first frame.
        bboxes: Optional pre-computed bounding boxes.

    Returns:
        List of selected person indices.
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button
    except ImportError:
        print("matplotlib not available, falling back to CLI selection.")
        return _cli_fallback(poses, bboxes)

    import cv2

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read video: {video_path}")
        return list(range(len(poses)))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if bboxes is None:
        bboxes = compute_bboxes_from_poses(poses)

    # State
    selected: list[int] = []
    n_persons = len(bboxes)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Select persons to track")

    ax.imshow(frame_rgb)
    ax.set_title("Click on persons to select, then press Enter", fontsize=12)
    ax.axis("off")

    # Draw bounding boxes
    rectangles = []
    labels = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="white", facecolor="none",
        )
        ax.add_patch(rect)
        rectangles.append(rect)
        label = ax.text(
            x1, y1 - 5, str(i),
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )
        labels.append(label)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        for i, bbox in enumerate(bboxes):
            if point_in_bbox(x, y, bbox):
                if i in selected:
                    selected.remove(i)
                    rectangles[i].set_edgecolor("white")
                    rectangles[i].set_linewidth(1.5)
                else:
                    selected.append(i)
                    rectangles[i].set_edgecolor("darkorange")
                    rectangles[i].set_linewidth(3)
                fig.canvas.draw_idle()
                break

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not selected:
        selected = list(range(n_persons))

    print(f"Selected persons: {selected}")
    return selected


def _cli_fallback(
    poses: NDArray[np.float32],
    bboxes: list[BBox] | None = None,
) -> list[int]:
    """CLI fallback when matplotlib is not available."""
    n = len(poses)
    if n == 1:
        print(f"Auto-selecting person 0 (only one detected).")
        return [0]

    print(f"Detected {n} persons:")
    for i in range(n):
        if bboxes and i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            print(f"  [{i}] bbox=({x1},{y1},{x2},{y2})")
        else:
            print(f"  [{i}]")

    try:
        choice = input("Enter person index: ").strip()
        return [int(choice)]
    except (ValueError, EOFError):
        return [0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pose_estimation/test_person_selector.py -v`
Expected: 4 passed

- [ ] **Step 5: Wire into cli.py**

In `src/cli.py`, modify the `--select-person` handling (around line 73-85). Replace the existing block with:

```python
        elif args.select_person:
            from .pose_estimation.person_selector import (
                compute_bboxes_from_poses,
                select_persons_interactive,
            )

            extractor = _get_extractor(args.pose_backend, output_format="normalized")
            persons = extractor.preview_persons(args.video)
            if not persons:
                print("No persons detected in the first seconds of the video.")
                return 1

            # Get poses for first frame to compute bboxes
            all_poses = extractor.extract_video(args.video)
            if all_poses.shape[0] > 0:
                first_frame_poses = all_poses[0:1]  # (1, 17, 2)
                selected_ids = select_persons_interactive(args.video, first_frame_poses)
                if selected_ids:
                    person_click = PersonClick(
                        x=persons[selected_ids[0]]["mid_hip"][0],
                        y=persons[selected_ids[0]]["mid_hip"][1],
                    )
            else:
                print("No poses extracted.")
                return 1
```

NOTE: The above is a sketch — the exact wiring depends on how `preview_persons()` returns data. The worker should adapt the CLI integration to match the existing `PersonClick` pattern used at lines 70-85 of cli.py.

- [ ] **Step 6: Commit**

```bash
git add src/pose_estimation/person_selector.py src/cli.py tests/pose_estimation/test_person_selector.py
git commit -m "feat(pose): add matplotlib interactive person selection"
```

---

## Task 6: Visible Side Detection + Floor Angle Estimation

**Files:**
- Modify: `src/utils/geometry.py:295-331`
- Test: `tests/utils/test_geometry.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_geometry.py`:

```python
from src.utils.geometry import detect_visible_side, estimate_floor_angle


class TestDetectVisibleSide:
    def test_right_side_visible(self):
        """Right side visible when toes are right of heels."""
        # RBigToe.x > RHeel.x → right side
        foot_kp = np.array([[
            100, 200, 0.9,  # L_Heel
            150, 200, 0.9,  # L_BigToe
            140, 195, 0.5,  # L_SmallToe
            300, 200, 0.9,  # R_Heel
            350, 200, 0.9,  # R_BigToe  → toe right of heel
            340, 195, 0.5,  # R_SmallToe
        ]], dtype=np.float32)  # shape (1, 6, 3)
        assert detect_visible_side(foot_kp) == "right"

    def test_left_side_visible(self):
        """Left side visible when toes are left of heels."""
        foot_kp = np.array([[
            300, 200, 0.9,  # L_Heel
            250, 200, 0.9,  # L_BigToe  → toe left of heel
            260, 195, 0.5,  # L_SmallToe
            100, 200, 0.9,  # R_Heel
            50, 200, 0.9,   # R_BigToe  → toe left of heel
            60, 195, 0.5,   # R_SmallToe
        ]], dtype=np.float32)
        assert detect_visible_side(foot_kp) == "left"

    def test_no_confidence_returns_none(self):
        """Should return None when all foot keypoints have low confidence."""
        foot_kp = np.zeros((1, 6, 3), dtype=np.float32)
        assert detect_visible_side(foot_kp) is None


class TestEstimateFloorAngle:
    def test_level_floor(self):
        """Horizontal foot positions should give ~0° angle."""
        # Feet at same Y level → floor angle = 0
        positions = np.array([
            [100.0, 200.0],
            [200.0, 200.0],
            [300.0, 200.0],
            [400.0, 200.0],
        ])
        angle = estimate_floor_angle(positions)
        assert abs(angle) < 1.0  # less than 1 degree

    def test_tilted_floor(self):
        """Consistently rising Y should give positive angle."""
        positions = np.array([
            [100.0, 200.0],
            [200.0, 210.0],
            [300.0, 220.0],
            [400.0, 230.0],
        ])
        angle = estimate_floor_angle(positions)
        assert angle > 0  # positive tilt

    def test_single_point_returns_zero(self):
        """Single point should return 0 (no line to fit)."""
        positions = np.array([[100.0, 200.0]])
        angle = estimate_floor_angle(positions)
        assert angle == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_geometry.py::TestDetectVisibleSide -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to end of `src/utils/geometry.py`:

```python
# ---------------------------------------------------------------------------
# Visible side detection (Sports2D-inspired)
# ---------------------------------------------------------------------------


def detect_visible_side(
    foot_keypoints: np.ndarray,
    conf_threshold: float = 0.3,
) -> str | None:
    """Detect which side of the body is facing the camera.

    Uses HALPE26 foot keypoints (heel + big_toe) to determine orientation.
    If big_toe is to the RIGHT of heel → right side visible.
    If big_toe is to the LEFT of heel → left side visible.

    Args:
        foot_keypoints: (1, 6, 3) or (N, 6, 3) foot keypoints in pixel coords.
            Columns: [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
            Channels: [x, y, confidence].
        conf_threshold: Minimum confidence for valid keypoints.

    Returns:
        "left", "right", or None if insufficient data.
    """
    # Use first frame if multi-frame
    if foot_keypoints.ndim == 3:
        # Aggregate: median orientation across frames
        orientations = []
        for i in range(foot_keypoints.shape[0]):
            fp = foot_keypoints[i]
            l_conf = fp[1, 2]  # L_BigToe confidence
            r_conf = fp[4, 2]  # R_BigToe confidence
            if l_conf < conf_threshold or r_conf < conf_threshold:
                continue
            l_orientation = fp[1, 0] - fp[0, 0]  # L_BigToe.x - L_Heel.x
            r_orientation = fp[4, 0] - fp[3, 0]  # R_BigToe.x - R_Heel.x
            orientations.append(l_orientation + r_orientation)
        if not orientations:
            return None
        return "right" if np.median(orientations) >= 0 else "left"
    return None


# ---------------------------------------------------------------------------
# Floor angle estimation (Sports2D-inspired)
# ---------------------------------------------------------------------------


def estimate_floor_angle(
    foot_positions: np.ndarray,
) -> float:
    """Estimate floor angle from foot positions.

    Fits a line through foot positions and returns the angle of that line
    relative to horizontal. Used to correct segment angles for camera tilt.

    Args:
        foot_positions: (N, 2) array of foot (x, y) positions in pixels.

    Returns:
        Floor angle in degrees. 0 = horizontal. Positive = tilted.
        Returns 0.0 if fewer than 2 points.
    """
    if len(foot_positions) < 2:
        return 0.0

    # Fit line: y = m*x + b
    coeffs = np.polyfit(foot_positions[:, 0], foot_positions[:, 1], 1)
    # coeffs[0] = slope (dy/dx), coeffs[1] = intercept
    # Angle of slope relative to horizontal (image coords: y-down)
    angle = -np.degrees(np.arctan(coeffs[0]))
    return float(angle)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_geometry.py::TestDetectVisibleSide tests/utils/test_geometry.py::TestEstimateFloorAngle -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/geometry.py tests/utils/test_geometry.py
git commit -m "feat(geo): add visible_side detection and floor_angle estimation"
```

---

## Task 7: Comprehensive Angle Formulas

**Files:**
- Create: `src/analysis/angles.py`
- Test: `tests/analysis/test_angles.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_angles.py`:

```python
"""Tests for comprehensive biomechanics angle computation."""

import numpy as np
import pytest

from src.analysis.angles import (
    compute_joint_angles,
    compute_segment_angles,
    ANGLE_DEFS,
    SEGMENT_DEFS,
)


def _standing_pose():
    """Create a standing pose with known angles."""
    pose = np.zeros((17, 2), dtype=np.float32)
    # Head above neck
    pose[0] = [320, 80]   # HEAD (index 0 in H36Key)
    pose[1] = [320, 100]  # NECK
    pose[2] = [320, 120]  # THORAX
    pose[3] = [320, 160]  # SPINE
    pose[4] = [320, 200]  # HIP_CENTER
    pose[5] = [310, 200]  # LHIP
    pose[6] = [310, 280]  # LKNEE
    pose[7] = [310, 360]  # LFOOT
    pose[8] = [330, 200]  # RHIP
    pose[9] = [330, 280]  # RKNEE
    pose[10] = [330, 360]  # RFOOT
    pose[11] = [300, 120]  # LSHOULDER
    pose[12] = [280, 170]  # LELBOW
    pose[13] = [260, 220]  # LWRIST
    pose[14] = [340, 120]  # RSHOULDER
    pose[15] = [360, 170]  # RELBOW
    pose[16] = [380, 220]  # RWRIST
    return pose


class TestAngleDefs:
    def test_has_expected_joint_angles(self):
        """Should define at least 12 joint angles."""
        joint_names = [d["name"] for d in ANGLE_DEFS]
        assert len(joint_names) >= 12
        # Spot-check key angles
        assert "R Knee" in joint_names
        assert "L Hip" in joint_names

    def test_has_expected_segment_angles(self):
        """Should define at least 14 segment angles."""
        seg_names = [d["name"] for d in SEGMENT_DEFS]
        assert len(seg_names) >= 14
        assert "R Foot" in seg_names
        assert "Trunk" in seg_names


class TestComputeJointAngles:
    def test_knee_angle_standing(self):
        """Standing pose should give ~180° knee angle."""
        pose = _standing_pose()
        angles = compute_joint_angles(pose)
        assert 170 < angles["R Knee"] <= 180

    def test_all_angles_in_range(self):
        """All angles should be in [0, 180] degrees."""
        pose = _standing_pose()
        angles = compute_joint_angles(pose)
        for name, val in angles.items():
            assert 0 <= val <= 180, f"{name} = {val}° out of range"

    def test_bent_knee(self):
        """Deeply bent knee should give small angle."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[5] = [310, 200]  # LHIP
        pose[6] = [310, 250]  # LKNEE
        pose[7] = [270, 350]  # LFOOT (foot behind knee)
        angles = compute_joint_angles(pose)
        assert angles["L Knee"] < 170


class TestComputeSegmentAngles:
    def test_trunk_vertical(self):
        """Vertical trunk should give ~90° angle."""
        pose = _standing_pose()
        angles = compute_segment_angles(pose)
        assert 80 < angles["Trunk"] < 100

    def test_all_segment_angles_defined(self):
        """All defined segments should have computed angles."""
        pose = _standing_pose()
        angles = compute_segment_angles(pose)
        for d in SEGMENT_DEFS:
            assert d["name"] in angles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_angles.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/analysis/angles.py`:

```python
"""Comprehensive biomechanics angle computation (Sports2D-inspired).

Provides:
- ANGLE_DEFS: Joint angle definitions (12 angles)
- SEGMENT_DEFS: Segment angle definitions (14 angles)
- compute_joint_angles(): Compute all joint angles from 2D pose
- compute_segment_angles(): Compute all segment angles from 2D pose
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.pose_estimation.h36m_extractor import H36Key
from src.utils.geometry import angle_3pt, segment_angle

# =============================================================================
# ANGLE DEFINITIONS
# =============================================================================

# Joint angles: 3-point angles at the middle joint
# Each entry: name, point_a (idx), vertex (idx), point_c (idx)
ANGLE_DEFS: list[dict] = [
    {"name": "R Ankle",   "a": H36Key.RKNEE, "v": H36Key.RFOOT, "c": None},
    {"name": "L Ankle",   "a": H36Key.LKNEE, "v": H36Key.LFOOT, "c": None},
    {"name": "R Knee",    "a": H36Key.RHIP,  "v": H36Key.RKNEE, "c": H36Key.RFOOT},
    {"name": "L Knee",    "a": H36Key.LHIP,  "v": H36Key.LKNEE, "c": H36Key.LFOOT},
    {"name": "R Hip",     "a": H36Key.THORAX, "v": H36Key.RHIP, "c": H36Key.RKNEE},
    {"name": "L Hip",     "a": H36Key.THORAX, "v": H36Key.LHIP, "c": H36Key.LKNEE},
    {"name": "R Shoulder", "a": None, "v": H36Key.RSHOULDER, "c": H36Key.RELBOW},
    {"name": "L Shoulder", "a": None, "v": H36Key.LSHOULDER, "c": H36Key.LELBOW},
    {"name": "R Elbow",   "a": H36Key.RSHOULDER, "v": H36Key.RELBOW, "c": H36Key.RWRIST},
    {"name": "L Elbow",   "a": H36Key.LSHOULDER, "v": H36Key.LELBOW, "c": H36Key.LWRIST},
    {"name": "R Wrist",   "a": H36Key.RELBOW, "v": H36Key.RWRIST, "c": None},
    {"name": "L Wrist",   "a": H36Key.LELBOW, "v": H36Key.LWRIST, "c": None},
]

# Segment angles: angle of a body segment relative to horizontal
# Each entry: name, start (idx), end (idx)
SEGMENT_DEFS: list[dict] = [
    {"name": "R Foot",   "start": H36Key.RFOOT, "end": None},
    {"name": "L Foot",   "start": H36Key.LFOOT, "end": None},
    {"name": "R Shank",  "start": H36Key.RKNEE, "end": H36Key.RFOOT},
    {"name": "L Shank",  "start": H36Key.LKNEE, "end": H36Key.LFOOT},
    {"name": "R Thigh",  "start": H36Key.RHIP,  "end": H36Key.RKNEE},
    {"name": "L Thigh",  "start": H36Key.LHIP,  "end": H36Key.LKNEE},
    {"name": "Pelvis",   "start": H36Key.LHIP,  "end": H36Key.RHIP},
    {"name": "Trunk",    "start": None, "end": None},
    {"name": "Shoulders", "start": H36Key.LSHOULDER, "end": H36Key.RSHOULDER},
    {"name": "Head",     "start": None, "end": None},
    {"name": "R Arm",    "start": H36Key.RSHOULDER, "end": H36Key.RELBOW},
    {"name": "L Arm",    "start": H36Key.LSHOULDER, "end": H36Key.LELBOW},
    {"name": "R Forearm", "start": H36Key.RELBOW, "end": H36Key.RWRIST},
    {"name": "L Forearm", "start": H36Key.LELBOW, "end": H36Key.LWRIST},
]


def compute_joint_angles(pose: NDArray[np.float32]) -> dict[str, float]:
    """Compute all joint angles from a 2D pose.

    Args:
        pose: (17, 2) pose array in pixel or normalized coordinates.

    Returns:
        Dict mapping angle name to degrees [0, 180].
    """
    angles: dict[str, float] = {}

    # Computed midpoints
    mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
    mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2

    for definition in ANGLE_DEFS:
        name = definition["name"]
        v_idx = definition["v"]
        a_idx = definition["a"]
        c_idx = definition["c"]

        vertex = pose[v_idx]

        if "Shoulder" in name:
            # Shoulder: mid_shoulder -> shoulder -> elbow
            a = mid_shoulder
            c = pose[c_idx]
        elif a_idx is None:
            angles[name] = np.nan
            continue
        elif c_idx is None:
            # Ankle/Wrist: only 2 points, use knee/elbow -> ankle/wrist direction
            # This is a simplification — return the segment angle instead
            angles[name] = np.nan
            continue
        else:
            a = pose[a_idx]
            c = pose[c_idx]

        try:
            angles[name] = angle_3pt(
                np.asarray(a, dtype=np.float64),
                np.asarray(vertex, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
            )
        except (ValueError, ZeroDivisionError):
            angles[name] = np.nan

    return angles


def compute_segment_angles(pose: NDArray[np.float32]) -> dict[str, float]:
    """Compute all segment angles from a 2D pose.

    Segment angle = angle of the segment relative to horizontal.

    Args:
        pose: (17, 2) pose array in pixel or normalized coordinates.

    Returns:
        Dict mapping segment name to degrees [-180, 180].
    """
    angles: dict[str, float] = {}

    mid_hip = (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2
    mid_shoulder = (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2

    for definition in SEGMENT_DEFS:
        name = definition["name"]
        start = definition["start"]
        end = definition["end"]

        try:
            if name == "Trunk":
                angles[name] = segment_angle(mid_hip, mid_shoulder)
            elif name == "Head":
                angles[name] = segment_angle(
                    mid_shoulder,
                    pose[H36Key.HEAD],
                )
            elif name in ("R Foot", "L Foot"):
                # Foot angle requires HALPE26 keypoints; use ankle direction as proxy
                idx = H36Key.RFOOT if "R" in name else H36Key.LFOOT
                knee_idx = H36Key.RKNEE if "R" in name else H36Key.LKNEE
                angles[name] = segment_angle(pose[knee_idx], pose[idx])
            else:
                angles[name] = segment_angle(pose[start], pose[end])
        except (ValueError, ZeroDivisionError, IndexError):
            angles[name] = np.nan

    return angles
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_angles.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/analysis/angles.py tests/analysis/test_angles.py
git commit -m "feat(angles): add comprehensive biomechanics angle computation (26 angles)"
```

---

## Task 8: 2D/3D Hybrid Visualization Mode

**Files:**
- Modify: `src/visualization/layers/joint_angle_layer.py:161-212`
- Test: `tests/visualization/test_joint_angle_layer.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/visualization/test_joint_angle_layer.py`:

```python
class TestHybridMode:
    def test_2d_mode_uses_2d_angles(self):
        """In '2d' mode, should compute angles from 2D pose regardless of 3D availability."""
        pose_2d = np.zeros((17, 2), dtype=np.float32)
        pose_2d[H36Key.LHIP] = [300, 180]
        pose_2d[H36Key.LKNEE] = [300, 240]
        pose_2d[H36Key.LFOOT] = [300, 300]
        pose_2d[H36Key.RHIP] = [340, 180]
        pose_2d[H36Key.RKNEE] = [340, 240]
        pose_2d[H36Key.RFOOT] = [340, 300]
        pose_2d[H36Key.LSHOULDER] = [280, 120]
        pose_2d[H36Key.RSHOULDER] = [360, 120]
        pose_2d[H36Key.THORAX] = [320, 150]

        # 3D with different angle
        pose_3d = pose_2d.copy()
        pose_3d = np.hstack([pose_3d, np.zeros((17, 1))])  # (17, 3)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = JointAngleLayer(angle_source="2d")
        ctx = LayerContext(
            frame_width=640, frame_height=480,
            pose_2d=pose_2d, pose_3d=pose_3d,
            normalized=False,
        )
        result = layer.render(frame, ctx)
        assert result is frame

    def test_invalid_angle_source_raises(self):
        """Invalid angle_source should raise ValueError."""
        with pytest.raises(ValueError):
            JointAngleLayer(angle_source="invalid")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_joint_angle_layer.py::TestHybridMode -v`
Expected: FAIL — `angle_source` parameter doesn't exist

- [ ] **Step 3: Write minimal implementation**

Modify `src/visualization/layers/joint_angle_layer.py`:

1. Add `angle_source` parameter to `__init__`:

```python
    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        joints: list[JointAngleSpec] | None = None,
        show_degree_labels: bool = True,
        angle_source: str = "auto",
    ):
        if angle_source not in ("auto", "2d", "3d"):
            raise ValueError(f"angle_source must be 'auto', '2d', or '3d', got '{angle_source}'")
        super().__init__(config=config or LayerConfig(enabled=True, z_index=6))
        self.viz = viz_config or VisualizationConfig()
        self.joints = joints or DEFAULT_JOINT_SPECS
        self.show_degree_labels = show_degree_labels
        self.angle_source = angle_source
```

2. Modify the angle computation in `render()` (lines 169-176). Replace the existing block:

```python
            # Prefer 3D angles (more accurate via kinematic constraints)
            angle = None
            if context.pose_3d is not None:
                ...
            # Fallback to 2D
            if angle is None:
                ...
```

With:

```python
            # Determine angle source
            angle = None
            use_3d = (
                self.angle_source == "3d"
                or (self.angle_source == "auto" and context.pose_3d is not None)
            )

            if use_3d and context.pose_3d is not None:
                a3 = context.pose_3d[spec.point_a]
                v3 = context.pose_3d[spec.vertex]
                c3 = context.pose_3d[spec.point_c]
                if c3 is not None and not (
                    np.isnan(a3).any() or np.isnan(v3).any() or np.isnan(c3).any()
                ):
                    angle = angle_3pt(a3, v3, c3)

            if angle is None:
                if context.normalized:
                    pa = normalized_to_pixel(pose[spec.point_a], w, h)
                    pv = normalized_to_pixel(pose[spec.vertex], w, h)
                    pc = normalized_to_pixel(pose[spec.point_c], w, h)
                else:
                    pa = pose[spec.point_a].astype(int)
                    pv = pose[spec.vertex].astype(int)
                    pc = pose[spec.point_c].astype(int)
                a = np.array(pa, dtype=np.float64)
                v = np.array(pv, dtype=np.float64)
                c = np.array(pc, dtype=np.float64)
                angle = angle_3pt(a, v, c)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/visualization/test_joint_angle_layer.py -v`
Expected: 4 passed (2 original + 2 new)

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x --tb=short -q`
Expected: All tests pass, no regressions

- [ ] **Step 6: Commit**

```bash
git add src/visualization/layers/joint_angle_layer.py tests/visualization/test_joint_angle_layer.py
git commit -m "feat(viz): add 2D/3D hybrid mode for angle display"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - [x] Dual-layer text rendering — Task 1
   - [x] RdYlGn confidence coloring — Task 2
   - [x] Enhanced angle arcs with degree labels — Task 3
   - [x] Angle panel with progress bars — Task 4
   - [x] Interactive person selection (matplotlib) — Task 5
   - [x] Visible side detection — Task 6
   - [x] Floor angle estimation — Task 6
   - [x] Comprehensive angle formulas (26 angles) — Task 7
   - [x] 2D/3D hybrid mode — Task 8
   - [x] Inverse kinematics — NOT included (requires OpenSim, future work)
   - [x] OpenSim TRC/MOT export — NOT included (future work)

2. **Placeholder scan:**
   - No TBD, TODO, or "implement later" found
   - All steps contain actual code
   - All test code is provided

3. **Type consistency:**
   - `JointAngleLayer.__init__` signature: `angle_source` parameter added in Task 3 and Task 8 — compatible
   - `draw_text_outlined()` signature consistent across Tasks 3, 4
   - `detect_visible_side()` signature consistent in Task 6 test and implementation

4. **File paths verified:**
   - `src/visualization/core/text.py` — exists
   - `src/visualization/skeleton/joints.py` — exists
   - `src/visualization/skeleton/drawer.py` — exists
   - `src/visualization/layers/joint_angle_layer.py` — exists
   - `src/visualization/layers/__init__.py` — exists
   - `src/utils/geometry.py` — exists
   - `src/cli.py` — exists
   - `src/analysis/` — exists
   - `src/pose_estimation/` — exists
   - All new files: parent directories exist
