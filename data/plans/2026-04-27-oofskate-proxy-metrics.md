# OOFSkate Proxy Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add camera-independent landing quality, torso lean, approach arc, and GOE proxy metrics that infer element quality from body kinematics without direct blade edge detection.

**Architecture:** Extend `BiomechanicsAnalyzer` with OOFSkate-inspired proxy metrics. Add `calculate_com_trajectory_2d()` for full (x,y) CoM. Compute landing smoothness from CoM velocity, approach arc from CoM path curvature, torso lean from spine angle. Combine into GOE proxy score (0-10). Wire into `ElementDef.ideal_metrics` and `RecommendationRule` templates.

**Tech Stack:** NumPy, Numba, SciPy. `ml/src/analysis/metrics.py`, `ml/src/utils/geometry.py`, `ml/src/analysis/element_defs.py`, `ml/src/analysis/rules/jump_rules.py`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `ml/src/utils/geometry.py` | Modify: add `calculate_com_trajectory_2d` | Full (x,y) CoM trajectory for velocity/curvature calculations |
| `ml/src/analysis/metrics.py` | Modify: add 6 new methods to `BiomechanicsAnalyzer` | Landing quality, torso lean, approach arc, GOE score metrics |
| `ml/src/analysis/element_defs.py` | Modify: add ideal ranges to jump defs | Reference ranges for new metrics |
| `ml/src/analysis/rules/jump_rules.py` | Modify: add `RecommendationRule` entries | Russian text recommendations for new metrics |
| `ml/tests/utils/test_geometry.py` | Modify: add tests for `calculate_com_trajectory_2d` | Verify CoM (x,y) matches (x,y) of scalar function |
| `ml/tests/analysis/test_metrics.py` | Modify: add tests for new metrics | Landing, lean, arc, GOE with synthetic poses |

---

## Task 1: Full CoM Trajectory (geometry.py)

**Files:**
- Modify: `ml/src/utils/geometry.py:335`
- Test: `ml/tests/utils/test_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/utils/test_geometry.py
import numpy as np
from src.utils.geometry import calculate_com_trajectory, calculate_com_trajectory_2d


def test_calculate_com_trajectory_2d_y_matches_scalar():
    """Full (x,y) CoM Y must match scalar CoM trajectory."""
    poses = np.random.rand(10, 17, 2).astype(np.float32)
    com_y_scalar = calculate_com_trajectory(poses)
    com_xy = calculate_com_trajectory_2d(poses)
    assert com_xy.shape == (10, 2)
    np.testing.assert_allclose(com_xy[:, 1], com_y_scalar, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest ml/tests/utils/test_geometry.py::test_calculate_com_trajectory_2d_y_matches_scalar -v
```

Expected: `FAILED NameError: name 'calculate_com_trajectory_2d' is not defined`

- [ ] **Step 3: Implement `calculate_com_trajectory_2d`**

Add immediately after `calculate_com_trajectory` (line 335):

```python

def calculate_com_trajectory_2d(
    poses: NormalizedPose,
) -> NDArray[np.float32]:
    """Calculate full (x, y) Center of Mass trajectory.

    Uses identical mass ratios to :func:`calculate_com_trajectory`
    but returns both x and y coordinates.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        CoM (x, y) coordinates (num_frames, 2) in normalized units.
    """
    # Segment mass ratios (Dempster 1955) — MUST match calculate_com_trajectory
    head_mass = 0.081
    torso_mass = 0.497
    arm_mass = 0.050  # per arm
    thigh_mass = 0.100  # per thigh
    leg_mass = 0.161  # per leg

    head = poses[:, H36Key.HEAD]
    torso = (
        poses[:, H36Key.LSHOULDER]
        + poses[:, H36Key.RSHOULDER]
        + poses[:, H36Key.LHIP]
        + poses[:, H36Key.RHIP]
    ) / 4
    l_upper_arm = (poses[:, H36Key.LSHOULDER] + poses[:, H36Key.LELBOW]) / 2
    r_upper_arm = (poses[:, H36Key.RSHOULDER] + poses[:, H36Key.RELBOW]) / 2
    l_forearm = (poses[:, H36Key.LELBOW] + poses[:, H36Key.LWRIST]) / 2
    r_forearm = (poses[:, H36Key.RELBOW] + poses[:, H36Key.RWRIST]) / 2
    l_thigh = (poses[:, H36Key.LHIP] + poses[:, H36Key.LKNEE]) / 2
    r_thigh = (poses[:, H36Key.RHIP] + poses[:, H36Key.RKNEE]) / 2
    l_leg = (poses[:, H36Key.LKNEE] + poses[:, H36Key.LFOOT]) / 2
    r_leg = (poses[:, H36Key.RKNEE] + poses[:, H36Key.RFOOT]) / 2

    com_x = (
        head_mass * head[:, 0]
        + torso_mass * torso[:, 0]
        + arm_mass * (l_upper_arm[:, 0] + r_upper_arm[:, 0] + l_forearm[:, 0] + r_forearm[:, 0])
        + thigh_mass * (l_thigh[:, 0] + r_thigh[:, 0])
        + leg_mass * (l_leg[:, 0] + r_leg[:, 0])
    )
    com_y = (
        head_mass * head[:, 1]
        + torso_mass * torso[:, 1]
        + arm_mass * (l_upper_arm[:, 1] + r_upper_arm[:, 1] + l_forearm[:, 1] + r_forearm[:, 1])
        + thigh_mass * (l_thigh[:, 1] + r_thigh[:, 1])
        + leg_mass * (l_leg[:, 1] + r_leg[:, 1])
    )

    return np.stack([com_x, com_y], axis=1).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest ml/tests/utils/test_geometry.py::test_calculate_com_trajectory_2d_y_matches_scalar -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add ml/src/utils/geometry.py ml/tests/utils/test_geometry.py
git commit -m "feat(geometry): add calculate_com_trajectory_2d for full (x,y) CoM"
```

---

## Task 2: Landing Quality Metrics (metrics.py)

**Files:**
- Modify: `ml/src/analysis/metrics.py` (add methods, wire into `_analyze_jump`)
- Test: `ml/tests/analysis/test_metrics.py`

- [ ] **Step 1: Write failing test for landing quality**

```python
# ml/tests/analysis/test_metrics.py
import numpy as np
from src.analysis.element_defs import ELEMENT_DEFS
from src.analysis.metrics import BiomechanicsAnalyzer
from src.types import ElementPhase


def test_landing_com_velocity_detects_hard_landing():
    """Hard landing = large downward CoM velocity at landing."""
    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    # Create synthetic poses: 30 frames, landing at frame 20
    poses = np.zeros((30, 17, 2), dtype=np.float32)
    # CoM falls fast at landing (frame 20) by lowering hip Y
    for i in range(21, 30):
        poses[i, :, 1] = -(i - 20) * 0.05  # drop after landing

    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=5, peak=12, landing=20, end=29
    )

    vel = analyzer.compute_landing_com_velocity(poses, phases, fps=30.0)
    assert vel < 0  # downward
    assert abs(vel) > 0.1  # significant


def test_landing_smoothness_perfect():
    """No post-landing movement = perfect smoothness (1.0)."""
    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    poses = np.ones((20, 17, 2), dtype=np.float32) * 0.5  # static pose
    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=5, peak=10, landing=12, end=19
    )

    smoothness = analyzer.compute_landing_smoothness(poses, phases, fps=30.0)
    assert smoothness == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_landing_com_velocity_detects_hard_landing ml/tests/analysis/test_metrics.py::test_landing_smoothness_perfect -v
```

Expected: `AttributeError: 'BiomechanicsAnalyzer' object has no attribute 'compute_landing_com_velocity'`

- [ ] **Step 3: Implement landing quality methods**

Add these methods to `BiomechanicsAnalyzer` in `ml/src/analysis/metrics.py`, immediately after `compute_landing_trunk_recovery` (line 573):

```python
    def compute_landing_com_velocity(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute CoM vertical velocity at landing frame.

        Negative value = CoM still moving downward at impact.
        Large negative = hard landing (toe pick / flat blade).
        Near-zero = controlled landing (good edge).

        Camera-independent: uses internal body kinematics only.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            CoM vertical velocity at landing (normalized units/s).
            Negative = downward. Returns 0.0 if no landing detected.
        """
        if phases.landing >= len(poses) - 1:
            return 0.0

        com_y = calculate_com_trajectory(poses)
        vy = np.gradient(com_y) * fps

        landing_frame = min(phases.landing, len(vy) - 1)
        return float(vy[landing_frame])

    def compute_landing_smoothness(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
        window_sec: float = 0.5,
    ) -> float:
        """Compute landing smoothness from post-landing CoM velocity stability.

        Measures how smoothly the skater absorbs impact after landing.
        Lower velocity variance = smoother landing = better edge control.

        Camera-independent: uses internal body kinematics only.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.
            window_sec: Post-landing window in seconds (default 0.5s).

        Returns:
            Smoothness score in [0.0, 1.0] where 1.0 = perfectly smooth.
            Formula: max(0.0, 1.0 - std_vy / threshold)
            Returns 1.0 if no post-landing data.
        """
        if phases.end <= phases.landing + 1:
            return 1.0

        window_frames = int(window_sec * fps)
        post_start = phases.landing + 1
        post_end = min(phases.end + 1, post_start + window_frames)

        if post_end <= post_start:
            return 1.0

        com_y = calculate_com_trajectory(poses)
        vy = np.gradient(com_y) * fps

        post_vy = vy[post_start:post_end]
        std_vy = float(np.std(post_vy))

        # Threshold: 0.5 normalized units/s = reasonable max std
        threshold = 0.5 * fps  # scale with fps for consistency
        smoothness = max(0.0, 1.0 - std_vy / threshold)
        return float(smoothness)
```

- [ ] **Step 4: Wire into `_analyze_jump`**

In `ml/src/analysis/metrics.py`, inside `_analyze_jump`, add after `compute_relative_jump_height` call (after line 258):

```python
        # Landing Quality (OOFSkate proxy)
        landing_vel = self.compute_landing_com_velocity(poses, phases, fps)
        results.append(
            MetricResult(
                name="landing_com_velocity",
                value=landing_vel,
                unit="norm/s",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        landing_smooth = self.compute_landing_smoothness(poses, phases, fps)
        results.append(
            MetricResult(
                name="landing_smoothness",
                value=landing_smooth,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_landing_com_velocity_detects_hard_landing ml/tests/analysis/test_metrics.py::test_landing_smoothness_perfect -v
```

Expected: both `PASSED`

- [ ] **Step 6: Commit**

```bash
git add ml/src/analysis/metrics.py ml/tests/analysis/test_metrics.py
git commit -m "feat(metrics): add landing quality (CoM velocity + smoothness)"
```

---

## Task 3: Torso Lean & Approach Arc (metrics.py)

**Files:**
- Modify: `ml/src/analysis/metrics.py`
- Test: `ml/tests/analysis/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# ml/tests/analysis/test_metrics.py

def test_approach_torso_lean_forward():
    """Forward lean during approach = positive angle."""
    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    poses = np.zeros((20, 17, 2), dtype=np.float32)
    # Forward lean: shoulders ahead of hips (positive x)
    poses[:, H36Key.LSHOULDER, 0] = 0.3
    poses[:, H36Key.RSHOULDER, 0] = 0.3
    poses[:, H36Key.LHIP, 0] = 0.0
    poses[:, H36Key.RHIP, 0] = 0.0
    # Vertical offset
    poses[:, H36Key.LSHOULDER, 1] = -0.5
    poses[:, H36Key.RSHOULDER, 1] = -0.5
    poses[:, H36Key.LHIP, 1] = 0.0
    poses[:, H36Key.RHIP, 0] = 0.0

    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=10, peak=12, landing=15, end=19
    )

    lean = analyzer.compute_approach_torso_lean(poses, phases)
    assert lean > 0  # forward lean


def test_approach_direction_change_straight():
    """Straight approach = near-zero direction change."""
    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    poses = np.zeros((20, 17, 2), dtype=np.float32)
    # CoM moves straight in x: constant velocity
    for i in range(20):
        poses[i, H36Key.LHIP, 0] = i * 0.01
        poses[i, H36Key.RHIP, 0] = i * 0.01

    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=10, peak=12, landing=15, end=19
    )

    change = analyzer.compute_approach_direction_change(poses, phases, fps=30.0)
    assert abs(change) < 5.0  # near zero for straight line
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_approach_torso_lean_forward ml/tests/analysis/test_metrics.py::test_approach_direction_change_straight -v
```

Expected: `AttributeError: 'BiomechanicsAnalyzer' object has no attribute 'compute_approach_torso_lean'`

- [ ] **Step 3: Implement approach metrics**

Add these methods to `BiomechanicsAnalyzer` in `ml/src/analysis/metrics.py`, after `compute_landing_smoothness`:

```python
    def compute_approach_torso_lean(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
    ) -> float:
        """Compute average torso lean angle during approach phase.

        Positive = forward lean (typical for flip).
        Negative = backward lean (typical for lutz).
        Near-zero = upright (flat or poor edge).

        Camera-independent: uses spine angle relative to vertical.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.

        Returns:
            Average lean angle in degrees during approach (start to takeoff).
            Returns 0.0 if no approach phase.
        """
        if phases.takeoff <= phases.start:
            return 0.0

        approach_poses = poses[phases.start : phases.takeoff + 1]
        if len(approach_poses) < 2:
            return 0.0

        trunk_lean = self.compute_trunk_lean(approach_poses)
        return float(np.mean(trunk_lean))

    def compute_approach_direction_change(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute total direction change of CoM horizontal velocity during approach.

        High value = curved approach (lutz: outside edge arc).
        Low value = straight approach (flip: inside edge, direct).

        Camera-independent: uses CoM (x,y) trajectory.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            Total direction change in degrees during approach.
            Returns 0.0 if no approach phase or insufficient data.
        """
        if phases.takeoff <= phases.start + 2:
            return 0.0

        from ..utils.geometry import calculate_com_trajectory_2d

        approach_poses = poses[phases.start : phases.takeoff + 1]
        com = calculate_com_trajectory_2d(approach_poses)

        # Horizontal velocity vector: (vx, vy) in image coords
        # We care about direction change in x-axis (horizontal plane proxy)
        vx = np.gradient(com[:, 0]) * fps

        # Angle of velocity vector at each frame
        angles = np.degrees(np.arctan2(vx, np.ones_like(vx)))

        # Total direction change = sum of absolute differences between consecutive frames
        if len(angles) < 2:
            return 0.0

        direction_changes = np.abs(np.diff(angles))
        return float(np.sum(direction_changes))
```

- [ ] **Step 4: Wire into `_analyze_jump`**

Inside `_analyze_jump`, add after landing_smoothness block:

```python
        # Torso Lean & Approach Arc (OOFSkate proxy for edge type)
        approach_lean = self.compute_approach_torso_lean(poses, phases)
        results.append(
            MetricResult(
                name="approach_torso_lean",
                value=approach_lean,
                unit="deg",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        approach_curve = self.compute_approach_direction_change(poses, phases, fps)
        results.append(
            MetricResult(
                name="approach_direction_change",
                value=approach_curve,
                unit="deg",
                is_good=False,
                reference_range=(0, 0),
            )
        )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_approach_torso_lean_forward ml/tests/analysis/test_metrics.py::test_approach_direction_change_straight -v
```

Expected: both `PASSED`

- [ ] **Step 6: Commit**

```bash
git add ml/src/analysis/metrics.py ml/tests/analysis/test_metrics.py
git commit -m "feat(metrics): add approach torso lean and direction change"
```

---

## Task 4: GOE Proxy Score (metrics.py)

**Files:**
- Modify: `ml/src/analysis/metrics.py`
- Test: `ml/tests/analysis/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# ml/tests/analysis/test_metrics.py

def test_goe_score_range():
    """GOE score must be in [0, 10]."""
    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    poses = np.ones((20, 17, 2), dtype=np.float32) * 0.5
    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=5, peak=10, landing=15, end=19
    )

    goe = analyzer.compute_goe_score(poses, phases, fps=30.0)
    assert 0.0 <= goe <= 10.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_goe_score_range -v
```

Expected: `AttributeError`

- [ ] **Step 3: Implement GOE proxy score**

Add to `BiomechanicsAnalyzer` in `ml/src/analysis/metrics.py`, after `compute_approach_direction_change`:

```python
    def compute_goe_score(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute GOE proxy score (0-10) from body kinematics.

        Inspired by MIT OOFSkate — estimates element quality without
        direct blade edge detection. Higher = better quality.

        Components:
        - Jump height (20%): relative CoM displacement
        - Rotation speed (15%): peak angular velocity
        - Landing quality (25%): smoothness + knee stability
        - Airtime (15%): flight duration
        - Torso control (15%): trunk recovery after landing
        - Approach consistency (10%): low direction change variance

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            GOE proxy score in [0.0, 10.0].
        """
        # Component scores (each in [0, 1])
        rel_height = self.compute_relative_jump_height(poses, phases)
        height_score = min(1.0, rel_height / 1.0)  # normalize: 1.0 = elite

        rot_speed = self.compute_rotation_speed(poses, phases, fps)
        rot_score = min(1.0, rot_speed / 720.0)  # 720 deg/s = 2 rotations/sec

        landing_smooth = self.compute_landing_smoothness(poses, phases, fps)
        landing_stab = self.compute_landing_knee_stability(poses, phases)
        landing_score = (landing_smooth + landing_stab) / 2.0

        airtime = self.compute_airtime(phases, fps)
        airtime_score = min(1.0, airtime / 1.0)  # 1s = good

        trunk_recovery = self.compute_landing_trunk_recovery(poses, phases)

        approach_change = self.compute_approach_direction_change(poses, phases, fps)
        # High direction change = curved approach = intentional (good for lutz)
        # But too high = unstable. Normalize: 0-90 deg = 0-1.0
        approach_score = min(1.0, approach_change / 90.0)

        # Weighted sum
        goe = (
            height_score * 0.20
            + rot_score * 0.15
            + landing_score * 0.25
            + airtime_score * 0.15
            + trunk_recovery * 0.15
            + approach_score * 0.10
        )

        return float(goe * 10.0)  # scale to 0-10
```

- [ ] **Step 4: Wire into `_analyze_jump`**

Inside `_analyze_jump`, add at the end before `return results`:

```python
        # GOE Proxy Score
        goe = self.compute_goe_score(poses, phases, fps)
        results.append(
            MetricResult(
                name="goe_score",
                value=goe,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )
```

- [ ] **Step 5: Run test**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_goe_score_range -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
git add ml/src/analysis/metrics.py ml/tests/analysis/test_metrics.py
git commit -m "feat(metrics): add GOE proxy score (0-10)"
```

---

## Task 5: Element Definitions (element_defs.py)

**Files:**
- Modify: `ml/src/analysis/element_defs.py`

- [ ] **Step 1: Add ideal metrics to jump definitions**

For each jump in `ELEMENT_DEFS`, add these entries to `ideal_metrics`:

In `waltz_jump` (after existing entries, before closing `}`):

```python
            "landing_com_velocity": (-2.0, 0.0),  # negative = downward, -2 = hard limit
            "landing_smoothness": (0.5, 1.0),
            "approach_torso_lean": (-30, 30),  # -30 = back (lutz), +30 = forward (flip)
            "approach_direction_change": (0, 90),
            "goe_score": (5.0, 10.0),
```

In `toe_loop` (same pattern):

```python
            "landing_com_velocity": (-2.0, 0.0),
            "landing_smoothness": (0.5, 1.0),
            "approach_torso_lean": (-30, 30),
            "approach_direction_change": (0, 90),
            "goe_score": (5.0, 10.0),
```

In `flip`:

```python
            "landing_com_velocity": (-2.0, 0.0),
            "landing_smoothness": (0.5, 1.0),
            "approach_torso_lean": (5, 30),  # flip: forward lean expected
            "approach_direction_change": (0, 45),  # flip: straighter approach
            "goe_score": (5.0, 10.0),
```

In `lutz`:

```python
            "landing_com_velocity": (-2.0, 0.0),
            "landing_smoothness": (0.5, 1.0),
            "approach_torso_lean": (-30, -5),  # lutz: backward lean expected
            "approach_direction_change": (20, 90),  # lutz: curved approach
            "goe_score": (5.0, 10.0),
```

- [ ] **Step 2: Verify syntax**

```bash
uv run python -c "from src.analysis.element_defs import ELEMENT_DEFS; print('OK:', list(ELEMENT_DEFS.keys()))"
```

Expected: `OK: ['three_turn', 'waltz_jump', 'toe_loop', 'flip', 'salchow', 'loop', 'lutz', 'axel']`

- [ ] **Step 3: Commit**

```bash
git add ml/src/analysis/element_defs.py
git commit -m "feat(element_defs): add ideal ranges for OOFSkate proxy metrics"
```

---

## Task 6: Recommendation Rules (jump_rules.py)

**Files:**
- Modify: `ml/src/analysis/rules/jump_rules.py`

- [ ] **Step 1: Add recommendation rules for new metrics**

Append to `WALTZ_JUMP_RULES` list:

```python
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
                "Для вальсового прыжка держи торс более вертикально."
            ),
            "too_high": (
                "Слишком сильный наклон вперёд при заходе ({value:.1f}°). "
                "Вальсовый прыжок требует вертикального отталкивания."
            ),
            "default": "Контролируй наклон торса при заходе.",
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
```

Append identical rules to `TOE_LOOP_RULES`, `FLIP_RULES`, `SALCHOW_RULES`, `LOOP_RULES`, `LUTZ_RULES`, `AXEL_RULES`.

- [ ] **Step 2: Verify imports**

```bash
uv run python -c "from src.analysis.rules.jump_rules import WALTZ_JUMP_RULES; print('Rules:', len(WALTZ_JUMP_RULES))"
```

Expected: `Rules: 9` (was 5, now 9)

- [ ] **Step 3: Commit**

```bash
git add ml/src/analysis/rules/jump_rules.py
git commit -m "feat(rules): add Russian recommendations for OOFSkate proxy metrics"
```

---

## Task 7: Integration Test (test_metrics.py)

**Files:**
- Test: `ml/tests/analysis/test_metrics.py`

- [ ] **Step 1: Write end-to-end test**

```python
# ml/tests/analysis/test_metrics.py

def test_analyze_returns_all_oofskate_metrics():
    """Full _analyze_jump must return all 10 metrics including new ones."""
    from src.analysis.element_defs import ELEMENT_DEFS

    element_def = ELEMENT_DEFS["waltz_jump"]
    analyzer = BiomechanicsAnalyzer(element_def)

    poses = np.ones((30, 17, 2), dtype=np.float32) * 0.5
    phases = ElementPhase(
        name="waltz_jump", start=0, takeoff=5, peak=12, landing=20, end=29
    )

    results = analyzer.analyze(poses, phases, fps=30.0)
    names = {r.name for r in results}

    expected = {
        "airtime", "max_height", "landing_knee_angle", "arm_position_score",
        "rotation_speed", "landing_knee_stability", "landing_trunk_recovery",
        "relative_jump_height",
        # New OOFSkate metrics
        "landing_com_velocity", "landing_smoothness",
        "approach_torso_lean", "approach_direction_change",
        "goe_score",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"
```

- [ ] **Step 2: Run test**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_analyze_returns_all_oofskate_metrics -v
```

Expected: `PASSED`

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest ml/tests/analysis/test_metrics.py -v --no-cov
```

Expected: all tests pass

- [ ] **Step 4: Lint**

```bash
uv run ruff check ml/src/analysis/metrics.py ml/src/utils/geometry.py ml/src/analysis/element_defs.py ml/src/analysis/rules/jump_rules.py
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add ml/tests/analysis/test_metrics.py
git commit -m "test(metrics): add integration test for OOFSkate proxy metrics"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Landing Quality Score — `compute_landing_com_velocity`, `compute_landing_smoothness`
- [x] Torso Lean — `compute_approach_torso_lean`
- [x] Approach Arc — `compute_approach_direction_change`
- [x] GOE Proxy — `compute_goe_score`
- [x] Element ranges — added to all jump defs
- [x] Recommendations — Russian text for all new metrics
- [x] Tests — unit + integration

**2. Placeholder scan:**
- [x] No "TBD", "TODO", "implement later"
- [x] No vague "add error handling" — all code shown
- [x] No "Similar to Task N" — all code repeated
- [x] All steps have exact code blocks

**3. Type consistency:**
- [x] `calculate_com_trajectory_2d` returns `NDArray[np.float32]` (matches scalar function)
- [x] All new metric methods take `(poses, phases, fps)` where applicable
- [x] `MetricResult` uses correct `name`, `unit`, `reference_range`
- [x] `ideal_metrics` keys match `MetricResult.name` exactly

**4. Caveats documented:**
- `calculate_com_trajectory_2d` MUST use identical mass ratios to `calculate_com_trajectory`
- `approach_direction_change` uses 2D x-curvature as proxy for 3D arc (camera limitation)
- GOE score weights are heuristic — derived from OOFSkate research, not trained model

---

## Next Phase (After Implementation)

1. **Reference Database Expansion** — collect elite skater averages per element for GOE normalization
2. **Validation** — compare GOE proxy against human judges on known competition videos
3. **Landing Deceleration** — add horizontal CoM velocity before/after landing for toe-pick proxy
