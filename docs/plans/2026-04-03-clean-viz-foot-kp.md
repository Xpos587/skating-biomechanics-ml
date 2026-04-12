# Clean Skeleton Viz + Foot Keypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match Sports2D's clean skeleton rendering style and add HALPE26 foot keypoint (heel + toe) visualization.

**Architecture:** Update `draw_skeleton()` to use Sports2D's exact color scheme (left=green, right=orange, center=blue bones + RdYlGn keypoints), add optional `foot_keypoints` parameter to draw HALPE26 foot markers and segments. Wire foot keypoints through the viz script. Remove angle panel from default layer stack.

**Tech Stack:** OpenCV (cv2), numpy, pytest

---

## File Structure

### Files to Modify
- `src/visualization/skeleton/drawer.py` — Sports2D bone colors, foot kp drawing
- `scripts/visualize_with_skeleton.py` — wire foot kp, fix visible_side, remove angle panel

### Test Files to Modify
- `tests/visualization/test_drawer.py` (create) — tests for foot kp drawing

---

## Task 1: Sports2D Bone Colors in `draw_skeleton()`

**Files:**
- Modify: `src/visualization/skeleton/drawer.py:92-112` (bone drawing section)

- [ ] **Step 1: Write the failing test**

Create `tests/visualization/test_drawer.py`:

```python
"""Tests for skeleton drawer Sports2D-style updates."""

import numpy as np
import pytest

from src.visualization.skeleton.drawer import draw_skeleton


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestSports2DBoneColors:
    def test_left_bone_is_green(self):
        """Left-side bones should use Sports2D green color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        # L-HIP to L-KNEE is a left-side bone
        pose[5] = [300.0, 200.0]  # LHIP
        pose[6] = [300.0, 300.0]  # LKNEE
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0)
        # Frame should have green pixels (BGR channel 1) in the bone area
        region = frame[200:300, 295:305]
        assert np.any(region[:, :, 1] > 200)  # green channel high

    def test_right_bone_is_orange(self):
        """Right-side bones should use Sports2D orange color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[8] = [340.0, 200.0]  # RHIP
        pose[9] = [340.0, 300.0]  # RKNEE
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0)
        region = frame[200:300, 335:345]
        # Orange in BGR: (0, 128, 255) — B=0, G=128, R=255
        assert np.any(region[:, :, 2] > 200)  # red channel high
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_drawer.py::TestSports2DBoneColors -v`
Expected: FAIL — left bones are blue, right bones are red (not Sports2D colors)

- [ ] **Step 3: Update bone colors to match Sports2D**

In `src/visualization/skeleton/drawer.py`, replace the bone drawing color logic (lines 108-112). Currently:

```python
        # Get bone color and thickness
        color = get_skeleton_color(joint_a)
        thickness = get_bone_thickness(joint_a, line_width)
```

Replace with Sports2D colors. The bone color should be determined by which side the bone belongs to, using the edge `(joint_a, joint_b)`:

```python
        # Sports2D bone colors: left=green, right=orange, center=blue
        color = _get_sports2d_bone_color(joint_a, joint_b)
        thickness = 1  # Sports2D uses thickness=2, but we match their thin style
```

Add the helper function in the UTILITY section (before `_is_valid_point`):

```python
# Sports2D color constants (BGR)
_SPORTS2D_LEFT_COLOR: tuple[int, int, int] = (0, 255, 0)     # Green
_SPORTS2D_RIGHT_COLOR: tuple[int, int, int] = (0, 128, 255)   # Orange
_SPORTS2D_CENTER_COLOR: tuple[int, int, int] = (255, 153, 51)  # Blue


def _get_sports2d_bone_color(joint_a: int, joint_b: int) -> tuple[int, int, int]:
    """Return bone color based on Sports2D side convention.

    Left-only bones (start with L, no R): green
    Right-only bones (start with R, no L): orange
    Center/mixed bones (trunk, head, pelvis): blue
    """
    from src.types import H36Key

    left_joints = {H36Key.LHIP, H36Key.LKNEE, H36Key.LFOOT, H36Key.LSHOULDER, H36Key.LELBOW, H36Key.LWRIST}
    right_joints = {H36Key.RHIP, H36Key.RKNEE, H36Key.RFOOT, H36Key.RSHOULDER, H36Key.RELBOW, H36Key.RWRIST}

    a_left = joint_a in left_joints
    a_right = joint_a in right_joints
    b_left = joint_b in left_joints
    b_right = joint_b in right_joints

    if a_left and not a_right and b_left and not b_right:
        return _SPORTS2D_LEFT_COLOR
    if a_right and not a_left and b_right and not b_left:
        return _SPORTS2D_RIGHT_COLOR
    return _SPORTS2D_CENTER_COLOR
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_drawer.py::TestSports2DBoneColors -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite to check regressions**

Run: `pytest tests/ -x --tb=short -q`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/visualization/skeleton/drawer.py tests/visualization/test_drawer.py
git commit -m "feat(viz): switch to Sports2D bone colors (left=green, right=orange, center=blue)"
```

---

## Task 2: Add Foot Keypoint Drawing to `draw_skeleton()`

**Files:**
- Modify: `src/visualization/skeleton/drawer.py` — add `foot_keypoints` param + drawing logic
- Test: `tests/visualization/test_drawer.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/visualization/test_drawer.py`:

```python
class TestFootKeypoints:
    def test_draws_foot_keypoints(self):
        """Should draw foot keypoints when provided."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[0] = [320, 80]   # HEAD
        pose[5] = [310, 200]  # LHIP
        pose[6] = [310, 280]  # LKNEE
        pose[7] = [310, 360]  # LFOOT

        # 6 foot keypoints: L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe
        foot_kp = np.array([
            [290, 365, 0.9],  # L_Heel
            [330, 370, 0.9],  # L_BigToe
            [310, 372, 0.5],  # L_SmallToe (skipped)
            [340, 365, 0.9],  # R_Heel
            [380, 370, 0.9],  # R_BigToe
            [360, 372, 0.5],  # R_SmallToe (skipped)
        ], dtype=np.float32)

        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()
        draw_skeleton(frame_after, pose, 480, 640, foot_keypoints=foot_kp)

        # Should have drawn foot kp markers (heel + big toe = 4 dots + 2 lines)
        assert not np.array_equal(frame_before, frame_after)

    def test_no_foot_keypoints_no_error(self):
        """Should work fine without foot keypoints."""
        pose = np.zeros((17, 2), dtype=np.float32)
        frame = _frame()
        result = draw_skeleton(frame, pose, 480, 640)
        assert result is frame

    def test_draws_heel_toe_line(self):
        """Should draw line connecting heel to big toe."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[7] = [310, 360]  # LFOOT (near foot kp area)
        foot_kp = np.array([
            [290, 365, 0.9],  # L_Heel
            [330, 370, 0.9],  # L_BigToe
            [310, 372, 0.0],  # L_SmallToe
            [0, 0, 0.0],      # R_Heel (zero conf, skipped)
            [0, 0, 0.0],      # R_BigToe
            [0, 0, 0.0],      # R_SmallToe
        ], dtype=np.float32)

        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, foot_keypoints=foot_kp)

        # Check that pixels exist between heel (290,365) and toe (330,370)
        foot_region = frame[360:375, 285:335]
        has_nonzero = np.any(foot_region > 0)
        assert has_nonzero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/visualization/test_drawer.py::TestFootKeypoints -v`
Expected: FAIL — `draw_skeleton() got an unexpected keyword argument 'foot_keypoints'`

- [ ] **Step 3: Add foot_keypoints parameter and drawing logic**

In `src/visualization/skeleton/drawer.py`:

1. Add `foot_keypoints` parameter to `draw_skeleton()`:

```python
def draw_skeleton(
    frame: Frame,
    pose: Pose2D,
    height: int,
    width: int,
    confidence_threshold: float = 0.5,
    line_width: int = 1,
    joint_radius: int = 3,
    normalized: bool | None = None,
    confidences: np.ndarray | None = None,
    foot_keypoints: np.ndarray | None = None,
) -> Frame:
```

2. At the end of `draw_skeleton()`, before the final `return frame`, add foot keypoint drawing:

```python
    # Draw foot keypoints (HALPE26: heel + big toe + small toe)
    if foot_keypoints is not None:
        _draw_foot_keypoints(frame, foot_keypoints, width, height, confidence_threshold)

    return frame
```

3. Add the `_draw_foot_keypoints()` static function in the UTILITY section:

```python
# Indices into the (6, 3) foot keypoint array
_FOOT_HEEL_L = 0
_FOOT_BIG_TOE_L = 1
_FOOT_SMALL_TOE_L = 2
_FOOT_HEEL_R = 3
_FOOT_BIG_TOE_R = 4
_FOOT_SMALL_TOE_R = 5


def _draw_foot_keypoints(
    frame: Frame,
    foot_keypoints: np.ndarray,
    width: int,
    height: int,
    confidence_threshold: float = 0.5,
    kp_radius: int = 3,
    line_thickness: int = 1,
) -> None:
    """Draw HALPE26 foot keypoints and segments.

    Draws heels and big toes as circles (RdYlGn confidence color).
    Draws heel-to-toe connecting lines.
    Skips small toes (too noisy on ice skates).
    Skips keypoints below confidence threshold.

    Args:
        frame: OpenCV image (H, W, 3).
        foot_keypoints: (6, 3) array — [x, y, conf] in normalized or pixel coords.
        width: Frame width for coordinate conversion.
        height: Frame height for coordinate conversion.
        confidence_threshold: Skip keypoints below this confidence.
        kp_radius: Radius for keypoint circles.
        line_thickness: Thickness for foot segment lines.
    """
    from src.visualization.skeleton.joints import get_confidence_color_rdygn

    # Determine coordinate space
    fk = foot_keypoints.copy()
    if fk[:, :2].max() <= 1.0:
        fk[:, 0] *= width
        fk[:, 1] *= height

    # Pairs to draw: (heel_idx, toe_idx) for left and right
    foot_pairs = [
        (_FOOT_HEEL_L, _FOOT_BIG_TOE_L),
        (_FOOT_HEEL_R, _FOOT_BIG_TOE_R),
    ]

    for heel_idx, toe_idx in foot_pairs:
        heel_conf = fk[heel_idx, 2]
        toe_conf = fk[toe_idx, 2]

        heel_pt = (int(fk[heel_idx, 0]), int(fk[heel_idx, 1]))
        toe_pt = (int(fk[toe_idx, 0]), int(fk[toe_idx, 1]))

        # Check both are within frame
        if not (_is_valid_point(heel_pt, width, height) and _is_valid_point(toe_pt, width, height)):
            continue

        # Draw heel-to-toe segment line (Sports2D: same color as side bone)
        if heel_conf >= confidence_threshold and toe_conf >= confidence_threshold:
            # Determine side color for the line
            if heel_idx == _FOOT_HEEL_L:
                line_color = _SPORTS2D_LEFT_COLOR
            else:
                line_color = _SPORTS2D_RIGHT_COLOR
            cv2.line(frame, heel_pt, toe_pt, line_color, line_thickness, cv2.LINE_AA)

        # Draw heel circle
        if heel_conf >= confidence_threshold:
            color = get_confidence_color_rdygn(heel_conf)
            cv2.circle(frame, heel_pt, kp_radius, color, -1, cv2.LINE_AA)

        # Draw big toe circle
        if toe_conf >= confidence_threshold:
            color = get_confidence_color_rdygn(toe_conf)
            cv2.circle(frame, toe_pt, kp_radius, color, -1, cv2.LINE_AA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/visualization/test_drawer.py::TestFootKeypoints -v`
Expected: 3 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x --tb=short -q`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/visualization/skeleton/drawer.py tests/visualization/test_drawer.py
git commit -m "feat(viz): add HALPE26 foot keypoint drawing (heels + big toes)"
```

---

## Task 3: Wire Foot Keypoints + Fix Visible Side in Viz Script

**Files:**
- Modify: `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Store foot keypoints from extraction**

After line 239 (`extraction = extractor.extract_video_tracked(...)`), add:

```python
        raw_foot_kps = extraction.foot_keypoints  # (N, 6, 3) normalized
```

Also handle the pre-computed poses path. After line 168 (the `if args.poses` block), add:

```python
        raw_foot_kps = None
```

- [ ] **Step 2: Pass foot keypoints to draw_skeleton**

Find the `draw_skeleton()` call (around line 440):

```python
            frame = draw_skeleton(frame, poses[current_pose_idx], draw_h, draw_w)
```

Replace with:

```python
            foot_kp = raw_foot_kps[current_pose_idx] if raw_foot_kps is not None else None
            frame = draw_skeleton(frame, poses[current_pose_idx], draw_h, draw_w, foot_keypoints=foot_kp)
```

- [ ] **Step 3: Fix visible_side detection**

Find the broken visible_side code block (the `try/except` with `extractor._foot_keypoints`). Replace the entire block with:

```python
        # Detect visible side from HALPE26 foot keypoints
        visible_side = None
        floor_angle = 0.0
        if current_pose_idx is not None:
            # Floor angle from H3.6M foot positions
            try:
                l_foot = poses[current_pose_idx][H36Key.LFOOT, :2]
                r_foot = poses[current_pose_idx][H36Key.RFOOT, :2]
                if not (np.isnan(l_foot).any() or np.isnan(r_foot).any()):
                    floor_angle = estimate_floor_angle(np.array([l_foot, r_foot]))
            except (ValueError, IndexError):
                pass

            # Visible side from HALPE26 foot keypoints
            if raw_foot_kps is not None and current_pose_idx < len(raw_foot_kps):
                fk = raw_foot_kps[current_pose_idx]
                if fk is not None and len(fk) >= 6:
                    visible_side = detect_visible_side(fk.reshape(1, 6, 3))
```

- [ ] **Step 4: Remove AnglePanelLayer from default layers**

Find the layer setup section:

```python
    if args.layer >= 2:
        layers.append(JointAngleLayer())
        layers.append(AnglePanelLayer())
        layers.append(VerticalAxisLayer())
```

Replace with:

```python
    if args.layer >= 2:
        layers.append(JointAngleLayer())
        layers.append(VerticalAxisLayer())
```

- [ ] **Step 5: Remove unused AnglePanelLayer import**

Find and remove the `AnglePanelLayer` import at the top of the file.

- [ ] **Step 6: Run the demo**

Run: `uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/VOVA.MOV --layer 3 --compress --crf 25 --output /home/michael/Downloads/VOVA_clean.mp4`

Expected: Video renders successfully with clean skeleton, foot keypoints visible, no angle panel clutter.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -x --tb=short -q`
Expected: All passing

- [ ] **Step 8: Commit**

```bash
git add scripts/visualize_with_skeleton.py
git commit -m "fix(viz): wire foot keypoints, fix visible_side, remove angle panel clutter"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - [x] Sports2D bone colors (left=green, right=orange, center=blue) — Task 1
   - [x] HALPE26 foot keypoints drawn (heel + big toe dots + segment lines) — Task 2
   - [x] Foot keypoints wired from extraction through viz — Task 3
   - [x] visible_side detection fixed (was using wrong attribute) — Task 3
   - [x] Angle panel removed from default layers — Task 3
   - [x] RdYlGn joint coloring preserved (this IS Sports2D's approach) — kept as-is

2. **Placeholder scan:**
   - No TBD, TODO, or incomplete steps
   - All code provided in full

3. **Type consistency:**
   - `draw_skeleton()` new param `foot_keypoints` consistent across Tasks 2 and 3
   - `_SPORTS2D_*` color constants defined before use in Task 1, reused in Task 2
   - `raw_foot_kps` variable consistent between definition and usage

4. **File paths verified:**
   - `src/visualization/skeleton/drawer.py` — exists
   - `scripts/visualize_with_skeleton.py` — exists
   - `tests/visualization/test_drawer.py` — to be created
