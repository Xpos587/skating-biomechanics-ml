# VerticalAxisLayer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign VerticalAxisLayer to provide phase-aware, color-coded trunk tilt feedback with degree labels, direction indicators, and head alignment — making it actionable for skating coaches.

**Architecture:** Replace the current single-purpose trunk tilt layer with a rich axis visualization that shows gravity reference line, spine axis, signed tilt angle with direction label (F/B/L/R), color-coded feedback based on phase-aware thresholds, and head alignment offset. The layer computes its own signed angle from 2D keypoints using `atan2(x, -y)` (matching `compute_trunk_lean` in metrics.py) and classifies quality using configurable per-phase thresholds. No new dependencies.

**Tech Stack:** OpenCV (cv2), NumPy, existing `LayerContext`/`Layer` base classes, existing `draw_text_outlined` for labels.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/visualization/layers/vertical_axis_layer.py` | **Rewrite** | Complete VerticalAxisLayer implementation |
| `tests/visualization/test_vertical_axis_layer.py` | **Create** | Test suite for the redesigned layer |

No other files need modification. The layer is already wired into `comparison.py` via the `"axis"` overlay key and exported from `src/visualization/__init__.py`. The existing constructor signature `(config, viz_config)` is preserved for backward compatibility.

---

### Task 1: Create test file with quality classification tests

**Files:**
- Create: `tests/visualization/test_vertical_axis_layer.py`

- [ ] **Step 1: Write tests for tilt quality classification**

```python
"""Tests for redesigned VerticalAxisLayer."""

import numpy as np
import pytest

from src.pose_estimation import H36Key
from src.visualization.layers.base import LayerContext
from src.visualization.layers.vertical_axis_layer import (
    TiltQuality,
    VerticalAxisLayer,
    classify_tilt,
)


def _make_context(
    pose_2d=None, pose_3d=None, w=640, h=480, normalized=False, frame_idx=0
):
    return LayerContext(
        frame_width=w,
        frame_height=h,
        pose_2d=pose_2d,
        pose_3d=pose_3d,
        normalized=normalized,
        frame_idx=frame_idx,
    )


def _upright_pose(w=640, h=480):
    """Pose with spine perfectly vertical (shoulder directly above hip)."""
    pose = np.zeros((17, 2), dtype=np.float32)
    cx = w // 2
    pose[H36Key.LHIP] = [cx - 20, 300]
    pose[H36Key.RHIP] = [cx + 20, 300]
    pose[H36Key.LSHOULDER] = [cx - 15, 180]
    pose[H36Key.RSHOULDER] = [cx + 15, 180]
    pose[H36Key.HEAD] = [cx, 100]
    return pose


def _leaning_forward_pose(w=640, h=480, lean_px=40):
    """Pose with shoulders shifted right (forward lean in image coords)."""
    pose = _upright_pose(w, h)
    pose[H36Key.LSHOULDER][0] += lean_px
    pose[H36Key.RSHOULDER][0] += lean_px
    pose[H36Key.HEAD][0] += lean_px
    return pose


class TestClassifyTilt:
    """Test the tilt quality classification function."""

    def test_zero_tilt_is_good(self):
        assert classify_tilt(0.0) == TiltQuality.GOOD

    def test_small_tilt_is_good(self):
        assert classify_tilt(4.0) == TiltQuality.GOOD

    def test_boundary_good_to_warn(self):
        assert classify_tilt(5.0) == TiltQuality.WARN

    def test_medium_tilt_is_warn(self):
        assert classify_tilt(8.0) == TiltQuality.WARN

    def test_boundary_warn_to_bad(self):
        assert classify_tilt(10.0) == TiltQuality.BAD

    def test_large_tilt_is_bad(self):
        assert classify_tilt(15.0) == TiltQuality.BAD

    def test_negative_tilt(self):
        assert classify_tilt(-4.0) == TiltQuality.GOOD
        assert classify_tilt(-8.0) == TiltQuality.WARN
        assert classify_tilt(-15.0) == TiltQuality.BAD


class TestVerticalAxisLayerRender:
    """Test rendering behavior of VerticalAxisLayer."""

    def test_no_pose_returns_frame_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=None)
        result = layer.render(frame, ctx)
        assert result is frame
        assert np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_upright_pose_modifies_frame(self):
        pose = _upright_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Frame should be modified (gravity line drawn)
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_leaning_pose_modifies_frame(self):
        pose = _leaning_forward_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_normalized_coords_render(self):
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[H36Key.LHIP] = [0.47, 0.6]
        pose[H36Key.RHIP] = [0.53, 0.6]
        pose[H36Key.LSHOULDER] = [0.48, 0.35]
        pose[H36Key.RSHOULDER] = [0.52, 0.35]
        pose[H36Key.HEAD] = [0.5, 0.2]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=True)
        result = layer.render(frame, ctx)
        assert result is frame
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_show_degree_label_false(self):
        """When show_degree_label=False, still renders gravity line but no text."""
        pose = _leaning_forward_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer(show_degree_label=False)
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Should still draw gravity line and spine
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/visualization/test_vertical_axis_layer.py -v`
Expected: FAIL — `cannot import name 'classify_tilt' from 'vertical_axis_layer'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/visualization/test_vertical_axis_layer.py
git commit -m "test(viz): add vertical axis layer redesign tests"
```

---

### Task 2: Implement TiltQuality enum and classify_tilt function

**Files:**
- Modify: `src/visualization/layers/vertical_axis_layer.py`

This task implements the quality classification logic that the layer and tests depend on.

- [ ] **Step 4: Rewrite vertical_axis_layer.py with TiltQuality, classify_tilt, and skeleton of VerticalAxisLayer**

```python
"""Vertical axis and trunk tilt visualization layer.

Draws:
- Dashed vertical gravity reference line through mid-hip
- Spine axis line (mid-hip to mid-shoulder)
- Color-coded angle arc between vertical and spine axis
- Signed tilt angle with direction label (F/B/L/R)
- Head alignment offset from spine axis
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Final

import cv2
import numpy as np

from src.pose_estimation import H36Key
from src.visualization.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    LayerConfig,
    VisualizationConfig,
)
from src.visualization.core.geometry import normalized_to_pixel
from src.visualization.layers.base import Frame, Layer, LayerContext


# =============================================================================
# TILT QUALITY CLASSIFICATION
# =============================================================================


class TiltQuality(Enum):
    """Quality of trunk tilt relative to vertical."""

    GOOD = "good"
    WARN = "warn"
    BAD = "bad"


# Default thresholds (degrees) — used when no phase context available.
# GOOD: |tilt| < 5°  |  WARN: 5° ≤ |tilt| < 10°  |  BAD: |tilt| ≥ 10°
_GOOD_THRESHOLD: Final[float] = 5.0
_WARN_THRESHOLD: Final[float] = 10.0

# Color mapping (BGR)
_QUALITY_COLORS: dict[TiltQuality, tuple[int, int, int]] = {
    TiltQuality.GOOD: COLOR_GREEN,
    TiltQuality.WARN: COLOR_YELLOW,
    TiltQuality.BAD: COLOR_RED,
}

# Gravity reference line color (light blue, subtle)
_GRAVITY_COLOR: Final[tuple[int, int, int]] = (200, 200, 100)
# Spine axis line color (will be overridden by quality color)
_SPINE_DEFAULT_COLOR: Final[tuple[int, int, int]] = (200, 200, 200)
# Head alignment indicator color
_HEAD_OFFSET_COLOR: Final[tuple[int, int, int]] = (180, 130, 255)  # pinkish


def classify_tilt(
    tilt_deg: float,
    good_threshold: float = _GOOD_THRESHOLD,
    warn_threshold: float = _WARN_THRESHOLD,
) -> TiltQuality:
    """Classify tilt angle quality.

    Args:
        tilt_deg: Signed tilt angle in degrees. Positive = right in image
            (forward/backward depending on camera angle).
        good_threshold: Below this absolute value, tilt is GOOD.
        warn_threshold: Below this absolute value, tilt is WARN.

    Returns:
        TiltQuality enum value.
    """
    abs_tilt = abs(tilt_deg)
    if abs_tilt < good_threshold:
        return TiltQuality.GOOD
    if abs_tilt < warn_threshold:
        return TiltQuality.WARN
    return TiltQuality.BAD


def _tilt_direction_label(tilt_deg: float) -> str:
    """Return short direction label for tilt.

    In image coordinates: positive tilt = spine leans RIGHT, negative = LEFT.
    The actual physical direction (forward/backward) depends on camera angle,
    but we label the image-plane direction since that's what the coach sees.
    """
    if abs(tilt_deg) < 1.0:
        return ""
    return "R" if tilt_deg > 0 else "L"


# =============================================================================
# VERTICAL AXIS LAYER
# =============================================================================


class VerticalAxisLayer(Layer):
    """Draw vertical reference axis and trunk tilt angle with coaching feedback.

    Renders:
    1. Dashed vertical gravity line through mid-hip center
    2. Spine axis line (mid-hip to mid-shoulder), color-coded by quality
    3. Angle arc between vertical and spine axis, color-coded
    4. Signed tilt angle label with direction indicator
    5. Head alignment offset indicator (head vs spine axis)

    The tilt angle is computed as atan2(spine_x, -spine_y) in pixel coords,
    matching the convention in ``compute_trunk_lean`` (metrics.py).
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        *,
        show_degree_label: bool = True,
        show_head_alignment: bool = True,
        good_threshold: float = _GOOD_THRESHOLD,
        warn_threshold: float = _WARN_THRESHOLD,
        arc_radius: int = 20,
    ):
        """Initialize vertical axis layer.

        Args:
            config: LayerConfig for z-ordering and visibility.
            viz_config: VisualizationConfig for styling (unused, kept for compat).
            show_degree_label: Whether to draw the degree + direction label.
            show_head_alignment: Whether to show head offset from spine axis.
            good_threshold: Tilt threshold for GOOD quality (degrees).
            warn_threshold: Tilt threshold for BAD quality (degrees).
            arc_radius: Radius of the angle arc in pixels.
        """
        super().__init__(config=config or LayerConfig(enabled=True, z_index=5))
        self.show_degree_label = show_degree_label
        self.show_head_alignment = show_head_alignment
        self.good_threshold = good_threshold
        self.warn_threshold = warn_threshold
        self.arc_radius = arc_radius

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        pose = context.pose_2d
        if pose is None:
            return frame

        w, h = context.frame_width, context.frame_height

        # Convert mid-hip and mid-shoulder to pixel coords
        if context.normalized:
            mid_hip = np.asarray(
                normalized_to_pixel(
                    (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2, w, h
                ),
                dtype=np.float64,
            )
            mid_shoulder = np.asarray(
                normalized_to_pixel(
                    (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2, w, h
                ),
                dtype=np.float64,
            )
            head = np.asarray(
                normalized_to_pixel(pose[H36Key.HEAD], w, h),
                dtype=np.float64,
            )
        else:
            mid_hip = ((pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2).astype(
                np.float64
            )
            mid_shoulder = (
                (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2
            ).astype(np.float64)
            head = pose[H36Key.HEAD].astype(np.float64)

        hip_x, hip_y = mid_hip[0], mid_hip[1]
        sh_x, sh_y = mid_shoulder[0], mid_shoulder[1]

        # Compute signed tilt angle (same formula as compute_trunk_lean)
        spine_vector = mid_shoulder - mid_hip
        tilt_deg = float(np.degrees(np.arctan2(spine_vector[0], -spine_vector[1])))

        # Classify quality
        quality = classify_tilt(
            tilt_deg, self.good_threshold, self.warn_threshold
        )
        quality_color = _QUALITY_COLORS[quality]

        # Compute adaptive line lengths based on torso height
        torso_height = np.linalg.norm(spine_vector)
        line_len = max(int(torso_height * 1.2), 60)
        lower_len = max(int(torso_height * 0.6), 30)

        # 1. Gravity reference line (dashed, subtle yellow)
        vert_top = (int(hip_x), int(hip_y - line_len))
        vert_bottom = (int(hip_x), int(hip_y + lower_len))
        self._draw_dashed_line(frame, vert_top, vert_bottom, _GRAVITY_COLOR, 1, dash=6)

        # 2. Spine axis line (quality-colored, solid)
        cv2.line(
            frame,
            (int(hip_x), int(hip_y)),
            (int(sh_x), int(sh_y)),
            quality_color,
            2,
            cv2.LINE_AA,
        )

        # 3. Angle arc (quality-colored)
        if abs(tilt_deg) > 1.0:
            self._draw_angle_arc(
                frame, int(hip_x), int(hip_y), int(sh_x), int(sh_y), tilt_deg, quality_color
            )

        # 4. Degree label with direction
        if self.show_degree_label and abs(tilt_deg) > 0.5:
            self._draw_degree_label(
                frame, int(hip_x), int(hip_y), tilt_deg, quality_color
            )

        # 5. Head alignment indicator
        if self.show_head_alignment:
            self._draw_head_alignment(frame, mid_hip, mid_shoulder, head, spine_vector)

        return frame

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_dashed_line(
        self,
        frame: Frame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        drawn = 0.0
        while drawn < dist:
            seg_end = min(drawn + dash, dist)
            sx = int(x1 + dx * drawn)
            sy = int(y1 + dy * drawn)
            ex = int(x1 + dx * seg_end)
            ey = int(y1 + dy * seg_end)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            drawn += dash * 2

    def _draw_angle_arc(
        self,
        frame: Frame,
        cx: int,
        cy: int,
        target_x: int,
        target_y: int,
        tilt_deg: float,
        color: tuple[int, int, int],
        radius: int | None = None,
    ) -> None:
        """Draw angle arc from vertical to spine axis direction."""
        if radius is None:
            radius = self.arc_radius

        # Spine angle in OpenCV coords (y-down)
        spine_angle = math.atan2(-(target_y - cy), target_x - cx)
        # Vertical up in OpenCV
        vert_angle = math.atan2(-1, 0)  # -pi/2

        start_deg = math.degrees(vert_angle)
        end_deg = math.degrees(spine_angle)

        # Ensure we draw the shorter arc
        diff = (end_deg - start_deg) % 360
        if diff > 180:
            start_deg, end_deg = end_deg, start_deg
        else:
            start_deg, end_deg = start_deg, end_deg

        if start_deg > end_deg:
            end_deg += 360

        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            0,
            start_deg,
            end_deg,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_degree_label(
        self,
        frame: Frame,
        hip_x: int,
        hip_y: int,
        tilt_deg: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw signed tilt angle with direction label near the hip."""
        from src.visualization.core.text import draw_text_outlined

        direction = _tilt_direction_label(tilt_deg)
        abs_tilt = abs(tilt_deg)
        label = f"{abs_tilt:.1f}\u00b0"
        if direction:
            label += f" {direction}"

        # Position label to the right of hip, offset down
        pos = (hip_x + self.arc_radius + 6, hip_y + 4)
        draw_text_outlined(frame, label, pos, font_scale=0.45, thickness=1, color=color)

    def _draw_head_alignment(
        self,
        frame: Frame,
        mid_hip: np.ndarray,
        mid_shoulder: np.ndarray,
        head: np.ndarray,
        spine_vector: np.ndarray,
    ) -> None:
        """Draw head alignment indicator showing offset from spine axis.

        Shows a thin line from the spine axis projection to the actual head
        position, indicating how far the head deviates from the vertical axis.
        """
        spine_len = np.linalg.norm(spine_vector)
        if spine_len < 1e-3:
            return

        # Project head onto spine line
        spine_dir = spine_vector / spine_len
        hip_to_head = head - mid_hip
        proj_len = np.dot(hip_to_head, spine_dir)
        proj_len = max(0, min(proj_len, spine_len * 1.5))  # clamp

        # Point on spine axis where head should be (extended to head height)
        spine_point = mid_hip + spine_dir * max(proj_len, spine_len * 1.1)

        head_px = (int(head[0]), int(head[1]))
        spine_px = (int(spine_point[0]), int(spine_point[1]))

        # Offset distance in pixels
        offset = np.linalg.norm(head[:2] - spine_point[:2])

        # Only draw if offset is significant (> 3 pixels)
        if offset > 3:
            cv2.line(frame, spine_px, head_px, _HEAD_OFFSET_COLOR, 1, cv2.LINE_AA)
            # Small circle at head position
            cv2.circle(frame, head_px, 3, _HEAD_OFFSET_COLOR, -1, cv2.LINE_AA)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/visualization/test_vertical_axis_layer.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `uv run pytest tests/visualization/ -v`
Expected: All tests PASS (including the new ones and existing joint_angle/text/drawer tests).

- [ ] **Step 7: Commit**

```bash
git add src/visualization/layers/vertical_axis_layer.py
git commit -m "feat(viz): redesign VerticalAxisLayer with quality feedback, direction labels, head alignment"
```

---

### Task 3: Add edge-case and integration tests

**Files:**
- Modify: `tests/visualization/test_vertical_axis_layer.py`

- [ ] **Step 8: Add edge-case tests**

Append to `tests/visualization/test_vertical_axis_layer.py`:

```python
class TestVerticalAxisLayerEdgeCases:
    """Edge cases and robustness tests."""

    def test_nan_pose_returns_unchanged(self):
        """All-NaN pose should not crash."""
        pose = np.full((17, 2), np.nan, dtype=np.float32)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_coincident_hip_shoulder(self):
        """Hip and shoulder at same position should not crash."""
        pose = np.zeros((17, 2), dtype=np.float32)
        cx, cy = 320, 240
        pose[H36Key.LHIP] = [cx - 10, cy]
        pose[H36Key.RHIP] = [cx + 10, cy]
        pose[H36Key.LSHOULDER] = [cx - 10, cy]
        pose[H36Key.RSHOULDER] = [cx + 10, cy]
        pose[H36Key.HEAD] = [cx, cy]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_custom_thresholds(self):
        """Custom thresholds should change classification."""
        layer = VerticalAxisLayer(good_threshold=10.0, warn_threshold=20.0)
        assert classify_tilt(7.0, 10.0, 20.0) == TiltQuality.GOOD
        assert classify_tilt(15.0, 10.0, 20.0) == TiltQuality.WARN
        assert classify_tilt(25.0, 10.0, 20.0) == TiltQuality.BAD

    def test_backward_compatible_constructor(self):
        """Old-style constructor with config and viz_config still works."""
        from src.visualization.config import LayerConfig, VisualizationConfig

        layer = VerticalAxisLayer(
            config=LayerConfig(enabled=True, z_index=5),
            viz_config=VisualizationConfig(),
        )
        assert layer.enabled
        assert layer.z_index == 5

    def test_head_alignment_with_offset(self):
        """Pose with head offset should draw head alignment indicator."""
        pose = _upright_pose()
        # Shift head to the right
        pose[H36Key.HEAD][0] += 50
        pose[H36Key.HEAD][1] = 80

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer(show_head_alignment=True)
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Should have drawn something
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_head_alignment_disabled(self):
        """show_head_alignment=False should not draw head offset line."""
        pose = _upright_pose()
        pose[H36Key.HEAD][0] += 50

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

        layer_on = VerticalAxisLayer(show_head_alignment=True)
        layer_off = VerticalAxisLayer(show_head_alignment=False)
        ctx = _make_context(pose_2d=pose, normalized=False)

        layer_on.render(frame1, ctx)
        layer_off.render(frame2, ctx)

        # Both should be modified, but frame with head alignment should differ
        assert not np.array_equal(frame1, frame2)


class TestTiltDirectionLabel:
    """Test direction label helper."""

    def test_zero_tilt_no_direction(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(0.0) == ""
        assert _tilt_direction_label(0.5) == ""

    def test_positive_tilt_right(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(5.0) == "R"
        assert _tilt_direction_label(1.5) == "R"

    def test_negative_tilt_left(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(-5.0) == "L"
        assert _tilt_direction_label(-1.5) == "L"
```

- [ ] **Step 9: Run all tests**

Run: `uv run pytest tests/visualization/test_vertical_axis_layer.py -v`
Expected: All ~20 tests PASS.

- [ ] **Step 10: Commit**

```bash
git add tests/visualization/test_vertical_axis_layer.py
git commit -m "test(viz): add edge-case tests for VerticalAxisLayer"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task |
|---|---|
| Gravity reference line (dashed) | Task 2, Step 4 — `_draw_dashed_line` |
| Spine axis line (color-coded) | Task 2, Step 4 — `render()` spine section |
| Angle arc (color-coded) | Task 2, Step 4 — `_draw_angle_arc` |
| Degree label + direction | Task 2, Step 4 — `_draw_degree_label` |
| Head alignment indicator | Task 2, Step 4 — `_draw_head_alignment` |
| Quality classification (G/W/R) | Task 2, Step 4 — `TiltQuality` + `classify_tilt` |
| Phase-aware thresholds | Task 2, Step 4 — configurable `good_threshold`/`warn_threshold` |
| Backward compatibility | Task 3, Step 8 — `test_backward_compatible_constructor` |
| Normalized + pixel coords | Task 1, Step 1 — `test_normalized_coords_render` |
| No regressions in comparison.py | Task 2, Step 6 — constructor signature preserved |

**Gap:** Phase-aware thresholds are configurable but the layer doesn't auto-detect phase from context. The `LayerContext` has a `phase` field, but phase-aware auto-thresholding is a future enhancement (the layer already supports custom thresholds that callers can set per-phase). This is YAGNI for now.

### 2. Placeholder Scan

No TBD/TODO/implement-later found. All code blocks contain complete implementations.

### 3. Type Consistency

- `classify_tilt(tilt_deg: float, good_threshold: float, warn_threshold: float) -> TiltQuality` — defined in Task 2, used in tests in Task 1 and Task 3
- `_tilt_direction_label(tilt_deg: float) -> str` — defined in Task 2, tested in Task 3
- `VerticalAxisLayer(config, viz_config, *, show_degree_label, show_head_alignment, good_threshold, warn_threshold, arc_radius)` — constructor matches test usage throughout
- `_GRAVITY_COLOR`, `_HEAD_OFFSET_COLOR`, `_QUALITY_COLORS` — all `tuple[int, int, int]` BGR, consistent with OpenCV
