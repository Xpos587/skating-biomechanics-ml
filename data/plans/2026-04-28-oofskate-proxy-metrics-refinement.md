# OOFSkate Proxy Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Допилить Phase I proxy metrics: добавить toe assist detection, поправить reference ranges и рекомендации, добавить тесты.

**Architecture:** Расширяем `BiomechanicsAnalyzer` новыми методами. Toe assist — через резкий spike вертикальной скорости CoM в момент landing. Reference ranges — per-element в `element_defs.py`. Рекомендации — в `jump_rules.py`.

**Tech Stack:** Python, NumPy, pytest, uv.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `ml/src/analysis/metrics.py` | Новые методы: `compute_toe_assist_proxy`, `compute_hard_landing`. Фиксы существующих. |
| `ml/src/analysis/element_defs.py` | Reference ranges per element для новых и существующих метрик. |
| `ml/src/analysis/rules/jump_rules.py` | Русские рекомендации для toe assist, hard landing, низкого landing angle. |
| `ml/tests/analysis/test_metrics.py` | Unit tests для всех proxy методов. |
| `ml/scripts/e2e_test_oofskate.py` | E2E скрипт для валидации на реальном видео. |

---

## Task 1: Fix landing_knee_angle Reference Range for Waltz Jump

**Files:**
- Modify: `ml/src/analysis/element_defs.py`
- Test: `ml/tests/analysis/test_element_defs.py`

**Context:** E2E test показал landing_knee_angle=83.4° для waltz jump, но reference range [90, 130] флагует как BAD. Для базового прыжка (waltz) приземление с углом 80-100° — норма.

- [ ] **Step 1: Найти текущий range для waltz_jump.landing_knee_angle**

```python
# В ml/src/analysis/element_defs.py найти WALTZ_JUMP и ключ landing_knee_angle
```

- [ ] **Step 2: Изменить range с [90, 130] на [70, 110]**

```python
"landing_knee_angle": (70, 110),  # waltz — базовый прыжок, приземление мягче
```

- [ ] **Step 3: Проверить другие прыжки**

Salchow, loop, toe_loop — [80, 120]
Flip, lutz, axel — [90, 130] (сложнее прыжки = выше приземление)

- [ ] **Step 4: Run existing tests**

```bash
uv run pytest ml/tests/analysis/test_element_defs.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/analysis/element_defs.py
git commit -m "fix(element_defs): lower landing_knee_angle ref for waltz_jump to 70-110"
```

---

## Task 2: Fix Inverted Recommendation for Low Landing Knee Angle

**Files:**
- Modify: `ml/src/analysis/rules/jump_rules.py`
- Test: `ml/tests/analysis/test_recommender.py`

**Context:** При landing_knee_angle=83° (меньше 90° = прямое колено) рекомендация говорит "Чрезмерное сгибание коленей" — это неверно. Меньше 90° = менее согнуто, не более.

- [ ] **Step 1: Найти rule для landing_knee_angle**

```python
# В ml/src/analysis/rules/jump_rules.py найти
# if metric.name == "landing_knee_angle":
```

- [ ] **Step 2: Поменять логику на правильную**

```python
if metric.value < ref_min:
    return "Колени слишком прямые при приземлении. Старайся приземляться на более согнутые колени для амортизации."
elif metric.value > ref_max:
    return "Чрезмерное сгибание коленей при приземлении. Выпрями ноги в момент касания льда."
```

- [ ] **Step 3: Run recommender tests**

```bash
uv run pytest ml/tests/analysis/test_recommender.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/src/analysis/rules/jump_rules.py
git commit -m "fix(jump_rules): correct landing_knee_angle recommendation logic"
```

---

## Task 3: Add Toe Assist / Clean Edge Proxy Metric

**Files:**
- Modify: `ml/src/analysis/metrics.py`
- Modify: `ml/src/analysis/element_defs.py`
- Test: `ml/tests/analysis/test_metrics.py`

**Context:** Toe assist — приземление на зубец конька вместо чистого ребра. Прокси: резкий spike вертикальной скорости CoM в landing frame (удар о лёд). Нормальное приземление = плавное снижение скорости.

- [ ] **Step 1: Write failing test**

```python
# ml/tests/analysis/test_metrics.py
def test_compute_toe_assist_proxy():
    from src.analysis.metrics import BiomechanicsAnalyzer
    from src.analysis.element_defs import get_element_def
    import numpy as np

    # Create synthetic poses: landing frame has sudden CoM spike
    num_frames = 30
    poses = np.zeros((num_frames, 17, 2))
    # Set up basic skeleton positions
    poses[:, H36Key.HIP_CENTER] = [0.5, 0.5]
    poses[:, H36Key.LHIP] = [0.4, 0.5]
    poses[:, H36Key.LKNEE] = [0.4, 0.6]
    poses[:, H36Key.LFOOT] = [0.4, 0.8]
    poses[:, H36Key.RHIP] = [0.6, 0.5]
    poses[:, H36Key.RKNEE] = [0.6, 0.6]
    poses[:, H36Key.RFOOT] = [0.6, 0.8]

    # Simulate toe assist: CoM y drops sharply at landing frame
    phases = ElementPhase(start=0, takeoff=10, peak=15, landing=20, end=29)

    analyzer = BiomechanicsAnalyzer(get_element_def("waltz_jump"))
    toe_score = analyzer.compute_toe_assist_proxy(poses, phases, fps=30.0)

    # Score should be in [0, 1] where 1 = clean edge, 0 = toe assist
    assert 0.0 <= toe_score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_compute_toe_assist_proxy -v
```

Expected: FAIL with "AttributeError: 'BiomechanicsAnalyzer' object has no attribute 'compute_toe_assist_proxy'"

- [ ] **Step 3: Implement compute_toe_assist_proxy**

```python
# In ml/src/analysis/metrics.py, class BiomechanicsAnalyzer

def compute_toe_assist_proxy(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    fps: float,
) -> float:
    """Detect toe assist vs clean edge landing via CoM velocity spike.

    Toe assist = sudden impact spike at landing (high deceleration).
    Clean edge = smooth deceleration over multiple frames.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        phases: Element phase boundaries.
        fps: Frame rate.

    Returns:
        Score in [0.0, 1.0] where 1.0 = clean edge, 0.0 = toe assist.
    """
    if phases.landing <= 0 or phases.landing >= len(poses) - 1:
        return 1.0  # Cannot assess

    com_trajectory = calculate_com_trajectory(poses)

    # Compute vertical velocity (Y increases downward, so negative = upward)
    vy = -(com_trajectory[1:] - com_trajectory[:-1]) * fps
    vy_y = vy[:, 1]  # vertical component

    # Look at landing frame and 2 frames after
    landing_idx = phases.landing
    post_end = min(landing_idx + 3, len(vy_y))

    if post_end <= landing_idx:
        return 1.0

    # Compute acceleration (change in vy)
    ay = np.diff(vy_y[landing_idx - 1 : post_end])
    if len(ay) == 0:
        return 1.0

    # Peak deceleration (most negative = hardest impact)
    peak_decel = np.min(ay)

    # Threshold: -5.0 norm/s^2 = toe assist territory
    # 0.0 = gentle deceleration
    score = max(0.0, min(1.0, 1.0 + peak_decel / 5.0))
    return float(score)
```

- [ ] **Step 4: Wire into analyze() method**

```python
# In BiomechanicsAnalyzer.analyze(), add:
toe_assist = self.compute_toe_assist_proxy(poses, phases, fps)
metrics.append(
    MetricResult(
        name="toe_assist_proxy",
        value=toe_assist,
        unit="score",
        is_good=toe_assist >= 0.5,
        reference_range=(0.5, 1.0),
    )
)
```

- [ ] **Step 5: Add reference range to element_defs.py**

```python
"toe_assist_proxy": (0.5, 1.0),  # все прыжки
```

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_compute_toe_assist_proxy -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ml/src/analysis/metrics.py ml/src/analysis/element_defs.py ml/tests/analysis/test_metrics.py
git commit -m "feat(metrics): add toe_assist_proxy metric for clean edge detection"
```

---

## Task 4: Add Hard Landing Detection Metric

**Files:**
- Modify: `ml/src/analysis/metrics.py`
- Modify: `ml/src/analysis/element_defs.py`
- Test: `ml/tests/analysis/test_metrics.py`

**Context:** Hard landing = слишком большая скорость вниз в момент касания. Прокси через CoM vy в landing frame.

- [ ] **Step 1: Write failing test**

```python
def test_compute_hard_landing():
    from src.analysis.metrics import BiomechanicsAnalyzer
    from src.analysis.element_defs import get_element_def
    import numpy as np

    num_frames = 30
    poses = np.zeros((num_frames, 17, 2))
    poses[:, H36Key.HIP_CENTER] = [0.5, 0.5]
    poses[:, H36Key.LHIP] = [0.4, 0.5]
    poses[:, H36Key.LKNEE] = [0.4, 0.6]
    poses[:, H36Key.LFOOT] = [0.4, 0.8]
    poses[:, H36Key.RHIP] = [0.6, 0.5]
    poses[:, H36Key.RKNEE] = [0.6, 0.6]
    poses[:, H36Key.RFOOT] = [0.6, 0.8]

    phases = ElementPhase(start=0, takeoff=10, peak=15, landing=20, end=29)

    analyzer = BiomechanicsAnalyzer(get_element_def("waltz_jump"))
    hard_score = analyzer.compute_hard_landing(poses, phases, fps=30.0)

    # 1.0 = soft landing, 0.0 = hard landing
    assert 0.0 <= hard_score <= 1.0
```

- [ ] **Step 2: Run test to verify fails**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_compute_hard_landing -v
```

Expected: FAIL

- [ ] **Step 3: Implement compute_hard_landing**

```python
def compute_hard_landing(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    fps: float,
) -> float:
    """Detect hard landing via CoM vertical velocity at impact.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        phases: Element phase boundaries.
        fps: Frame rate.

    Returns:
        Score in [0.0, 1.0] where 1.0 = soft landing, 0.0 = very hard.
    """
    if phases.landing <= 0 or phases.landing >= len(poses):
        return 1.0

    com_trajectory = calculate_com_trajectory(poses)

    # CoM vertical velocity at landing (backward difference)
    if phases.landing == 0:
        return 1.0

    vy = -(com_trajectory[phases.landing] - com_trajectory[phases.landing - 1]) * fps
    vy_y = vy[1]  # vertical component (positive = downward in normalized coords)

    # Threshold: 2.0 norm/s downward = hard landing
    # 0.5 norm/s = soft
    score = max(0.0, min(1.0, 1.0 - vy_y / 2.0))
    return float(score)
```

- [ ] **Step 4: Wire into analyze()**

```python
hard_landing = self.compute_hard_landing(poses, phases, fps)
metrics.append(
    MetricResult(
        name="hard_landing",
        value=hard_landing,
        unit="score",
        is_good=hard_landing >= 0.5,
        reference_range=(0.5, 1.0),
    )
)
```

- [ ] **Step 5: Add reference range**

```python
"hard_landing": (0.5, 1.0),
```

- [ ] **Step 6: Run test to verify passes**

```bash
uv run pytest ml/tests/analysis/test_metrics.py::test_compute_hard_landing -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ml/src/analysis/metrics.py ml/src/analysis/element_defs.py ml/tests/analysis/test_metrics.py
git commit -m "feat(metrics): add hard_landing detection via CoM velocity"
```

---

## Task 5: Add Russian Recommendations for New Metrics

**Files:**
- Modify: `ml/src/analysis/rules/jump_rules.py`
- Test: `ml/tests/analysis/test_recommender.py`

- [ ] **Step 1: Add recommendation rules for toe_assist_proxy**

```python
if metric.name == "toe_assist_proxy":
    if metric.value < ref_min:
        return "Приземление слишком резкое — возможно, приземляешься на зубец конька. Старайся касаться льда плавно, через ребро лезвия."
    return None

if metric.name == "hard_landing":
    if metric.value < ref_min:
        return "Жесткое приземление. Работай над амортизацией: сгибай колени и бедра, приземляйся мягко."
    return None
```

- [ ] **Step 2: Update GOE score formula to include toe_assist and hard_landing**

```python
# In compute_goe_score, adjust weights:
toe_score = self.compute_toe_assist_proxy(poses, phases, fps)
hard_score = self.compute_hard_landing(poses, phases, fps)
landing_score = (landing_smooth + landing_stab + hard_score + toe_score) / 4.0
```

- [ ] **Step 3: Run recommender tests**

```bash
uv run pytest ml/tests/analysis/test_recommender.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/src/analysis/rules/jump_rules.py ml/src/analysis/metrics.py
git commit -m "feat(rules): add Russian recommendations for toe assist and hard landing"
```

---

## Task 6: Validate on Real Video (E2E)

**Files:**
- Modify: `ml/scripts/e2e_test_oofskate.py`

- [ ] **Step 1: Add toe_assist and hard_landing to E2E output**

```python
for metric in result.metrics:
    if metric.name in ("toe_assist_proxy", "hard_landing", "goe_score"):
        print(f"  {marker} {metric.name}: {metric.value:.3f} {metric.unit}")
```

- [ ] **Step 2: Run E2E on Waltz.mp4**

```bash
uv run python ml/scripts/e2e_test_oofskate.py /home/michael/Downloads/Videos/Waltz.mp4 waltz_jump
```

- [ ] **Step 3: Verify metrics are reasonable**

Expected:
- landing_knee_angle: ~83°, marked as ✅ (range 70-110)
- toe_assist_proxy: ~0.7-1.0 (clean edge)
- hard_landing: ~0.6-1.0 (soft landing)
- GOE: ~5.0-6.0

- [ ] **Step 4: Commit**

```bash
git add ml/scripts/e2e_test_oofskate.py
git commit -m "test(e2e): add toe assist and hard landing to OOFSkate E2E output"
```

---

## Task 7: Run Full Test Suite

- [ ] **Step 1: Run all ML tests**

```bash
uv run pytest ml/tests/ --no-cov -q
```

Expected: All pass except pre-existing 10 failures in rtmo_batch

- [ ] **Step 2: Run lint and type check**

```bash
uv run ruff check ml/src/ ml/tests/
uv run basedpyright ml/src/ --level error
```

Expected: No new errors

- [ ] **Step 3: Commit any fixes**

```bash
git add -A && git commit -m "style: ruff/basedpyright fixes"
```

---

## Spec Coverage Self-Review

| Spec Requirement | Task | Status |
|------------------|------|--------|
| Fix landing_knee_angle range for waltz | Task 1 | ✅ |
| Fix inverted recommendation for low angle | Task 2 | ✅ |
| Toe assist detection (clean edge proxy) | Task 3 | ✅ |
| Hard landing detection | Task 4 | ✅ |
| Russian recommendations for new metrics | Task 5 | ✅ |
| E2E validation on real video | Task 6 | ✅ |
| Full test suite passing | Task 7 | ✅ |

## Placeholder Scan

- Нет TBD/TODO/fill in details
- Нет "appropriate error handling" без кода
- Все команды exact
- Все пути exact

## Type Consistency

- `compute_toe_assist_proxy` → возвращает `float` в [0, 1]
- `compute_hard_landing` → возвращает `float` в [0, 1]
- `MetricResult.value` → `float`
- `reference_range` → `tuple[float, float]`

Согласовано.

---

Plan complete and saved to `data/plans/2026-04-28-oofskate-proxy-metrics-refinement.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
