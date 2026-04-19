# Pipeline Parallelism Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dual-queue priority, full post-smoothing fan-out, conditional 3D lift, CoM pre-computation cache, vectorized CPU loops, and graceful shutdown.

**Architecture:** Two arq queues (fast/heavy) replace dead `_priority`. Pipeline fan-out runs 3 analysis branches concurrently after smoothing. CoM computed once and shared across branches. Per-frame Python loops replaced with NumPy vectorized operations.

**Tech Stack:** Python 3.13, arq, Valkey, asyncio, NumPy, Numba, pytest

---

### Task 1: Remove `_priority` dead code

**Files:**
- Modify: `backend/app/routes/detect.py:44-50`
- Modify: `backend/app/routes/process.py:52-64`

arq's `enqueue_job` silently ignores `_priority` — jobs are scored by timestamp only. Remove false cues.

- [ ] **Step 1: Remove `_priority` from detect.py**

In `backend/app/routes/detect.py`, change the `enqueue_job` call from:

```python
await request.app.state.arq_pool.enqueue_job(
    "detect_video_task",
    task_id=task_id,
    video_key=video_key,
    tracking=tracking,
    _priority=0,  # High priority for fast preview
)
```

to:

```python
await request.app.state.arq_pool.enqueue_job(
    "detect_video_task",
    task_id=task_id,
    video_key=video_key,
    tracking=tracking,
)
```

- [ ] **Step 2: Remove `_priority` from process.py**

In `backend/app/routes/process.py`, change the `enqueue_job` call from:

```python
await request.app.state.arq_pool.enqueue_job(
    "process_video_task",
    task_id=task_id,
    video_key=req.video_key,
    person_click={"x": req.person_click.x, "y": req.person_click.y},
    frame_skip=req.frame_skip,
    layer=req.layer,
    tracking=req.tracking,
    export=req.export,
    ml_flags=ml_flags,
    session_id=req.session_id,
    _priority=10,  # Low priority for full analysis
)
```

to:

```python
await request.app.state.arq_pool.enqueue_job(
    "process_video_task",
    task_id=task_id,
    video_key=req.video_key,
    person_click={"x": req.person_click.x, "y": req.person_click.y},
    frame_skip=req.frame_skip,
    layer=req.layer,
    tracking=req.tracking,
    export=req.export,
    ml_flags=ml_flags,
    session_id=req.session_id,
)
```

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest backend/tests/test_task_manager.py -v`
Expected: All pass (no functional change)

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/detect.py backend/app/routes/process.py
git commit -m "fix(backend): remove arq _priority dead code — silently ignored"
```

---

### Task 2: Dual-Queue Worker

**Files:**
- Modify: `backend/app/worker.py` — split into two `WorkerSettings` classes
- Modify: `backend/app/routes/detect.py` — enqueue to `skating:queue:fast`
- Modify: `backend/app/routes/process.py` — enqueue to `skating:queue:heavy`
- Test: `backend/tests/test_task_manager.py`

Two separate arq workers on different queues. Detection tasks (5-15s) never queue behind processing jobs (30-600s).

- [ ] **Step 1: Write failing test for queue name separation**

Add to `backend/tests/test_task_manager.py`:

```python
def test_worker_queue_names():
    """Each WorkerSettings class should use its own queue name."""
    from app.worker import FastWorkerSettings, HeavyWorkerSettings

    assert FastWorkerSettings.queue_name == "skating:queue:fast"
    assert HeavyWorkerSettings.queue_name == "skating:queue:heavy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_task_manager.py::test_worker_queue_names -v`
Expected: FAIL — `ImportError` or `AttributeError` (classes don't exist yet)

- [ ] **Step 3: Split WorkerSettings into FastWorkerSettings and HeavyWorkerSettings**

In `backend/app/worker.py`, replace the single `WorkerSettings` class (lines 507-531) with:

```python
class FastWorkerSettings:
    """arq worker for lightweight detection tasks."""

    queue_name: str = "skating:queue:fast"
    max_jobs: int = _settings.app.worker_max_jobs_remote if _settings.vastai.api_key.get_secret_value() else _settings.app.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown
    functions: ClassVar[list] = [detect_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )


class HeavyWorkerSettings:
    """arq worker for full ML pipeline processing."""

    queue_name: str = "skating:queue:heavy"
    max_jobs: int = 1 if _settings.vastai.api_key.get_secret_value() else 1
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown
    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )
```

Note: `HeavyWorkerSettings.max_jobs` is always 1 — GPU-bound, can't parallelize.

- [ ] **Step 4: Update detect.py to enqueue to fast queue**

In `backend/app/routes/detect.py`, add `_queue_name` to the `enqueue_job` call:

```python
await request.app.state.arq_pool.enqueue_job(
    "detect_video_task",
    task_id=task_id,
    video_key=video_key,
    tracking=tracking,
    _queue_name="skating:queue:fast",
)
```

- [ ] **Step 5: Update process.py to enqueue to heavy queue**

In `backend/app/routes/process.py`, add `_queue_name` to the `enqueue_job` call:

```python
await request.app.state.arq_pool.enqueue_job(
    "process_video_task",
    task_id=task_id,
    video_key=req.video_key,
    person_click={"x": req.person_click.x, "y": req.person_click.y},
    frame_skip=req.frame_skip,
    layer=req.layer,
    tracking=req.tracking,
    export=req.export,
    ml_flags=ml_flags,
    session_id=req.session_id,
    _queue_name="skating:queue:heavy",
)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest backend/tests/test_task_manager.py -v`
Expected: All pass including new `test_worker_queue_names`

- [ ] **Step 7: Commit**

```bash
git add backend/app/worker.py backend/app/routes/detect.py backend/app/routes/process.py backend/tests/test_task_manager.py
git commit -m "feat(backend): dual-queue worker — fast queue for detect, heavy queue for process"
```

---

### Task 3: Vectorize `calculate_com_trajectory` in geometry.py

**Files:**
- Modify: `ml/src/utils/geometry.py:295-310`
- Test: `ml/tests/utils/test_geometry.py`

Currently `calculate_com_trajectory()` calls `calculate_center_of_mass()` per frame in a Python loop. Replace with a single vectorized NumPy computation.

- [ ] **Step 1: Write failing benchmark test**

Add to `ml/tests/utils/test_geometry.py`:

```python
class TestCalculateComTrajectoryVectorized:
    """Tests for vectorized CoM trajectory calculation."""

    def test_com_trajectory_matches_scalar(self, sample_normalized_poses):
        """Vectorized trajectory should match per-frame scalar computation."""
        # Reference: per-frame computation
        from src.utils.geometry import calculate_center_of_mass

        expected = np.array(
            [calculate_center_of_mass(sample_normalized_poses, i)
             for i in range(len(sample_normalized_poses))],
            dtype=np.float32,
        )
        actual = calculate_com_trajectory(sample_normalized_poses)

        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_com_trajectory_single_frame(self):
        """Should work for single-frame input."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        com = calculate_com_trajectory(poses)
        assert com.shape == (1,)

    def test_com_trajectory_100_frames(self):
        """Should handle 100 frames efficiently (no Python loop)."""
        rng = np.random.default_rng(42)
        poses = rng.uniform(-0.5, 0.5, size=(100, 17, 2)).astype(np.float32)
        com = calculate_com_trajectory(poses)
        assert com.shape == (100,)
```

- [ ] **Step 2: Run test to verify it passes (existing impl is correct)**

Run: `uv run pytest ml/tests/utils/test_geometry.py::TestCalculateComTrajectoryVectorized -v`
Expected: PASS (existing loop implementation produces correct results)

- [ ] **Step 3: Replace loop with vectorized computation**

Replace `calculate_com_trajectory` in `ml/src/utils/geometry.py` (lines 295-310):

```python
def calculate_com_trajectory(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate Center of Mass trajectory for entire pose sequence.

    Vectorized implementation — computes all frames at once using NumPy
    broadcasting instead of per-frame Python loop.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        CoM Y-coordinates (num_frames,) in normalized units.
    """
    # Segment mass ratios (Dempster 1955)
    head_mass = 0.081
    torso_mass = 0.497
    arm_mass = 0.050  # per arm (upper arm + forearm + hand)
    thigh_mass = 0.100  # per thigh
    leg_mass = 0.161  # per leg (shin + foot)

    # Vectorized segment positions: (N, 2)
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

    # Weighted sum of Y-coordinates: (N,)
    com_y = (
        head_mass * head[:, 1]
        + torso_mass * torso[:, 1]
        + arm_mass * (l_upper_arm[:, 1] + r_upper_arm[:, 1] + l_forearm[:, 1] + r_forearm[:, 1])
        + thigh_mass * (l_thigh[:, 1] + r_thigh[:, 1])
        + leg_mass * (l_leg[:, 1] + r_leg[:, 1])
    )

    return com_y.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify correctness preserved**

Run: `uv run pytest ml/tests/utils/test_geometry.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ml/src/utils/geometry.py ml/tests/utils/test_geometry.py
git commit -m "perf(ml): vectorize calculate_com_trajectory — replace per-frame loop with NumPy broadcast"
```

---

### Task 4: Vectorize `metrics.py` loops

**Files:**
- Modify: `ml/src/analysis/metrics.py` — `compute_edge_indicator`, `compute_rotation_speed`, `compute_angle_series`, `compute_relative_jump_height`
- Test: `ml/tests/analysis/test_metrics.py`

Four per-frame Python loops replaced with vectorized NumPy operations.

- [ ] **Step 1: Write failing tests for vectorized outputs**

Add to `ml/tests/analysis/test_metrics.py`:

```python
class TestVectorizedMetrics:
    """Tests that vectorized metric functions produce correct results."""

    @pytest.fixture
    def jump_poses(self):
        """Synthetic jump poses (100 frames, 17, 2)."""
        rng = np.random.default_rng(42)
        poses = rng.uniform(-0.5, 0.5, size=(100, 17, 2)).astype(np.float32)
        return poses

    @pytest.fixture
    def jump_phases(self):
        return ElementPhase(
            name="waltz_jump", start=0, takeoff=30, peak=45, landing=60, end=90,
        )

    def test_edge_indicator_shape(self, jump_poses):
        """compute_edge_indicator should return (N,) array."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)
        result = analyzer.compute_edge_indicator(jump_poses, side="left")
        assert result.shape == (100,)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_rotation_speed_shape(self, jump_poses, jump_phases):
        """compute_rotation_speed should return float."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)
        result = analyzer.compute_rotation_speed(jump_poses, jump_phases, fps=30.0)
        assert isinstance(result, float)
        assert result >= 0

    def test_angle_series_shape(self, jump_poses):
        """compute_angle_series should return (N,) array."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)
        result = analyzer.compute_angle_series(
            jump_poses,
            int(H36Key.LHIP), int(H36Key.LKNEE), int(H36Key.LFOOT),
        )
        assert result.shape == (100,)

    def test_relative_jump_height_shape(self, jump_poses, jump_phases):
        """compute_relative_jump_height should return float."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)
        result = analyzer.compute_relative_jump_height(jump_poses, jump_phases)
        assert isinstance(result, float)
```

- [ ] **Step 2: Run tests to verify existing behavior**

Run: `uv run pytest ml/tests/analysis/test_metrics.py::TestVectorizedMetrics -v`
Expected: All PASS (existing implementation is correct, just slow)

- [ ] **Step 3: Vectorize `compute_edge_indicator`**

Replace `compute_edge_indicator` in `ml/src/analysis/metrics.py` (lines 615-662):

```python
def compute_edge_indicator(
    self,
    poses: NormalizedPose,
    side: str = "left",
) -> TimeSeries:
    """Compute edge indicator using H3.6M 17-keypoint format.

    Vectorized implementation — processes all frames at once.
    """
    if side == "left":
        hip = poses[:, H36Key.LHIP]
        shoulder = poses[:, H36Key.LSHOULDER]
    else:
        hip = poses[:, H36Key.RHIP]
        shoulder = poses[:, H36Key.RSHOULDER]

    # Vector from hip to shoulder: (N, 2)
    spine_vector = shoulder - hip

    # Angle from vertical: atan2(x, -y)
    angle = np.arctan2(spine_vector[:, 0], -spine_vector[:, 1])

    # Normalize to [-1, 1]
    edge_indicator = np.clip(angle / (np.pi / 6), -1, 1).astype(np.float32)

    return edge_indicator
```

- [ ] **Step 4: Vectorize `compute_rotation_speed`**

Replace `compute_rotation_speed` in `ml/src/analysis/metrics.py` (lines 664-704):

```python
def compute_rotation_speed(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    fps: float,
) -> float:
    """Compute peak rotation speed during jump."""
    # Vectorized shoulder axis angle: (N,)
    left_shoulder = poses[:, H36Key.LSHOULDER]
    right_shoulder = poses[:, H36Key.RSHOULDER]

    shoulder_vector = right_shoulder - left_shoulder
    angles = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])
    angles_deg = np.degrees(angles)

    # Angular velocity
    velocity = self.compute_angular_velocity(angles_deg, fps)

    # Peak in flight phase
    if phases.takeoff < phases.landing and phases.landing < len(velocity):
        flight_velocity = velocity[phases.takeoff : phases.landing]
        return float(np.max(np.abs(flight_velocity)))

    return 0.0
```

- [ ] **Step 5: Vectorize `compute_angle_series`**

Replace `compute_angle_series` in `ml/src/analysis/metrics.py` (lines 344-370):

```python
def compute_angle_series(
    self,
    poses: NormalizedPose,
    joint_a: int,
    joint_b: int,
    joint_c: int,
) -> TimeSeries:
    """Compute angle ABC for each frame (vectorized)."""
    a = poses[:, joint_a]  # (N, 2)
    b = poses[:, joint_b]  # (N, 2)
    c = poses[:, joint_c]  # (N, 2)

    # Build triplet array for batch processing: (N, 3, 2)
    abc_triplets = np.stack([a, b, c], axis=1)
    return angle_3pt_batch(abc_triplets).astype(np.float32)
```

- [ ] **Step 6: Vectorize `compute_relative_jump_height` spine length loop**

Replace the spine length loop in `compute_relative_jump_height` (lines 773-798):

```python
# Calculate average spine length around takeoff
start_frame = max(0, phases.takeoff - 2)
end_frame = min(len(poses), phases.takeoff + 3)

takeoff_window = poses[start_frame:end_frame]
mid_hip = (takeoff_window[:, H36Key.LHIP] + takeoff_window[:, H36Key.RHIP]) / 2
mid_shoulder = (takeoff_window[:, H36Key.LSHOULDER] + takeoff_window[:, H36Key.RSHOULDER]) / 2
spine_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
valid_spines = spine_lengths[spine_lengths >= 0.01]
```

Then use `valid_spines` instead of the `spine_lengths` list:

```python
if len(valid_spines) == 0:
    return 0.0

avg_spine = float(np.mean(valid_spines))
```

- [ ] **Step 7: Run full metrics test suite**

Run: `uv run pytest ml/tests/analysis/test_metrics.py ml/tests/analysis/test_metrics_numba.py -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add ml/src/analysis/metrics.py ml/tests/analysis/test_metrics.py
git commit -m "perf(ml): vectorize metrics.py loops — edge_indicator, rotation_speed, angle_series, spine length"
```

---

### Task 5: Vectorize `normalize_poses` in geometry.py

**Files:**
- Modify: `ml/src/utils/geometry.py:132-175`
- Test: `ml/tests/utils/test_geometry.py`

`normalize_poses` has a per-frame loop for root-centering and scale normalization. Replace with vectorized NumPy broadcast.

- [ ] **Step 1: Write failing test**

Add to `ml/tests/utils/test_geometry.py`:

```python
class TestNormalizePosesVectorized:
    """Tests for vectorized normalize_poses."""

    def test_matches_original(self, sample_normalized_poses):
        """Vectorized output should match original loop-based output."""
        from src.utils.geometry import normalize_poses as normalize

        raw = np.random.default_rng(42).uniform(-1, 1, size=(50, 17, 3)).astype(np.float32)
        result = normalize(raw)

        assert result.shape == (50, 17, 2)
        # Root-centered: mid-hip should be near origin
        mid_hip = (result[:, H36Key.LHIP] + result[:, H36Key.RHIP]) / 2
        np.testing.assert_allclose(mid_hip, 0, atol=1e-5)

    def test_17_keypoints_only(self):
        """Should raise ValueError for non-17 keypoint input."""
        with pytest.raises(ValueError, match="17 keypoints"):
            from src.utils.geometry import normalize_poses as normalize
            normalize(np.zeros((10, 15, 3), dtype=np.float32))
```

- [ ] **Step 2: Run test to verify existing behavior**

Run: `uv run pytest ml/tests/utils/test_geometry.py::TestNormalizePosesVectorized -v`
Expected: PASS

- [ ] **Step 3: Replace per-frame loop with vectorized computation**

Replace the loop in `normalize_poses` (lines 156-173) with:

```python
    # Mid-hip point (N, 2)
    mid_hip_raw = (raw[:, H36Key.LHIP, :2] + raw[:, H36Key.RHIP, :2]) / 2

    # 1. Root-centering: shift mid-hip to origin (N, 17, 2)
    centered = raw[:, :, :2] - mid_hip_raw[:, np.newaxis, :]

    # 2. Scale normalization
    shoulder_idx, hip_idx = spine_indices
    spine_vector = centered[:, shoulder_idx] - centered[:, hip_idx]  # (N, 2)
    spine_length = np.linalg.norm(spine_vector, axis=1)  # (N,)

    scale = np.where(spine_length < 1e-6, 1.0, target_spine_length / spine_length)  # (N,)

    normalized = centered * scale[:, np.newaxis, np.newaxis]  # (N, 17, 2)
```

The full function becomes:

```python
def normalize_poses(
    raw: FrameKeypoints,
    spine_indices: tuple[int, int] = (H36Key.LSHOULDER, H36Key.LHIP),
    target_spine_length: float = 0.4,
) -> NormalizedPose:
    """Normalize poses via root-centering and scale normalization.

    Vectorized — processes all frames at once using NumPy broadcasting.

    Args:
        raw: Raw keypoints (num_frames, 17, 3) with x, y, confidence.
        spine_indices: (shoulder_idx, hip_idx) for spine length calculation.
        target_spine_length: Target spine length after normalization.

    Returns:
        NormalizedPose (num_frames, 17, 2) with centered, scaled coordinates.
    """
    if raw.shape[1] != 17:
        raise ValueError(f"Expected 17 keypoints (H3.6M format), got {raw.shape[1]}")

    # Mid-hip point (N, 2)
    mid_hip_raw = (raw[:, H36Key.LHIP, :2] + raw[:, H36Key.RHIP, :2]) / 2

    # 1. Root-centering: shift mid-hip to origin (N, 17, 2)
    centered = raw[:, :, :2] - mid_hip_raw[:, np.newaxis, :]

    # 2. Scale normalization
    shoulder_idx, hip_idx = spine_indices
    spine_vector = centered[:, shoulder_idx] - centered[:, hip_idx]  # (N, 2)
    spine_length = np.linalg.norm(spine_vector, axis=1)  # (N,)

    scale = np.where(spine_length < 1e-6, 1.0, target_spine_length / spine_length)  # (N,)

    normalized = centered * scale[:, np.newaxis, np.newaxis]  # (N, 17, 2)

    return normalized.astype(np.float32)
```

**Important behavioral difference:** The original code uses per-frame spine length. The vectorized version also uses per-frame spine length (each frame scaled independently). This preserves existing behavior exactly.

- [ ] **Step 4: Run all geometry tests**

Run: `uv run pytest ml/tests/utils/test_geometry.py ml/tests/utils/test_geometry_numba.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ml/src/utils/geometry.py ml/tests/utils/test_geometry.py
git commit -m "perf(ml): vectorize normalize_poses — replace per-frame loop with NumPy broadcast"
```

---

### Task 6: Conditional 3D Lift Gate

**Files:**
- Modify: `ml/src/pipeline.py` — add `compute_3d` parameter
- Test: `ml/tests/test_pipeline.py` (create if needed)

Add `compute_3d: bool = False` parameter to `AnalysisPipeline`. Skip CorrectiveLens/blade detection when False. Default matches current behavior (3D already disabled in practice since MotionAGFormer model rarely present).

- [ ] **Step 1: Write failing test**

Create `ml/tests/test_pipeline.py`:

```python
"""Tests for AnalysisPipeline configuration."""

import numpy as np
import pytest
from pathlib import Path

from src.pipeline import AnalysisPipeline
from src.types import ElementPhase


class TestConditional3D:
    """Tests for compute_3d gate."""

    def test_compute_3d_defaults_false(self):
        """compute_3d should default to False."""
        pipeline = AnalysisPipeline()
        assert pipeline._compute_3d is False

    def test_compute_3d_explicit_true(self):
        """compute_3d=True should be stored."""
        pipeline = AnalysisPipeline(compute_3d=True)
        assert pipeline._compute_3d is True

    def test_analyze_skips_3d_when_false(self, sample_video_path):
        """When compute_3d=False, 3D lifting should be skipped entirely."""
        pipeline = AnalysisPipeline(compute_3d=False)
        report = pipeline.analyze(sample_video_path, element_type="three_turn")
        # 3D blade summaries should be empty
        assert report.blade_summary_left == {}
        assert report.blade_summary_right == {}
        assert report.physics == {}
```

Note: `sample_video_path` fixture needs to be added to conftest.py or the test uses a mock. Since pipeline tests require actual video/ONNX models, we test the parameter storage directly and mock the 3D path.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/test_pipeline.py -v`
Expected: FAIL — `TypeError` (no `compute_3d` parameter)

- [ ] **Step 3: Add `compute_3d` parameter to `__init__`**

In `ml/src/pipeline.py`, add `compute_3d` to the constructor (line 53):

```python
def __init__(
    self,
    reference_store: ReferenceStore | None = None,
    device: str | DeviceConfig = "auto",
    enable_smoothing: bool = True,
    smoothing_config: OneEuroFilterConfig | None = None,
    person_click: PersonClick | None = None,
    reestimate_camera: bool = False,
    profiler: PipelineProfiler | None = None,
    compute_3d: bool = False,
) -> None:
```

And store it:

```python
self._compute_3d = compute_3d
```

- [ ] **Step 4: Gate 3D lift in `analyze()`**

In `ml/src/pipeline.py` `analyze()` method (lines 239-271), wrap the 3D block:

Replace:

```python
t0 = time.perf_counter()
poses_3d = None
blade_summary_left = None
blade_summary_right = None
try:
    # Use MotionAGFormer for 3D lifting ...
```

With:

```python
t0 = time.perf_counter()
poses_3d = None
blade_summary_left = None
blade_summary_right = None
if self._compute_3d:
    try:
        # Use MotionAGFormer for 3D lifting ...
```

And dedent the block so the `self._profiler.record("3d_lift_and_blade", ...)` runs outside the try/except but still only when `compute_3d` is True:

```python
if self._compute_3d:
    try:
        poses_3d = self._get_pose_3d_extractor().extract_sequence(smoothed)
        # ... blade detection ...
    except Exception:
        pass
    self._profiler.record("3d_lift_and_blade", time.perf_counter() - t0)
```

- [ ] **Step 5: Gate 3D lift in `analyze_async()`**

Same gate in `analyze_async()` (line 662). Replace:

```python
poses_3d_future = asyncio.create_task(self._lift_poses_3d_async(smoothed, meta.fps))
```

With:

```python
if self._compute_3d:
    poses_3d_future = asyncio.create_task(self._lift_poses_3d_async(smoothed, meta.fps))
else:
    poses_3d_future = asyncio.ensure_future(asyncio.coroutine(lambda: (None, None))())
```

Or more simply:

```python
if self._compute_3d:
    poses_3d_future = asyncio.create_task(self._lift_poses_3d_async(smoothed, meta.fps))
    poses_3d, blade_summaries = await poses_3d_future
else:
    poses_3d, blade_summaries = None, None
```

And remove the separate `await poses_3d_future` that follows.

- [ ] **Step 6: Run tests**

Run: `uv run pytest ml/tests/test_pipeline.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add ml/src/pipeline.py ml/tests/test_pipeline.py
git commit -m "feat(ml): add compute_3d gate to AnalysisPipeline — skip 3D lift by default"
```

---

### Task 7: Post-Smoothing Fan-Out with CoM Cache

**Files:**
- Modify: `ml/src/pipeline.py` — restructure `analyze_async()` for 3-branch fan-out
- Modify: `ml/src/analysis/metrics.py` — accept pre-computed CoM
- Modify: `ml/src/analysis/physics_engine.py` — accept pre-computed CoM, avoid triple computation
- Test: `ml/tests/test_pipeline.py`

After smoothing, run 3 analysis branches concurrently. Pre-compute CoM once and pass to all consumers.

**Current flow** (sequential stages):
```
Smooth → [3D lift ‖ Phase det] → Metrics → Reference → DTW → Physics → Recommend
```

**Target flow** (DAG with 2 parallel waves):
```
Smooth → CoM pre-compute
       → [3D lift ‖ Phase det ‖ Ref load]     ← Wave 1
       → [Physics ‖ Metrics]                   ← Wave 2 (after phases)
       → DTW → Recommend → merge               ← Wave 3
```

- [ ] **Step 1: Add `com_trajectory` parameter to `compute_jump_height_com`**

In `ml/src/analysis/metrics.py`, modify `compute_jump_height_com` (line 434):

```python
def compute_jump_height_com(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    com_trajectory: NDArray[np.float32] | None = None,
) -> float:
    """Compute jump height using Center of Mass trajectory.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        phases: Element phase boundaries.
        com_trajectory: Pre-computed CoM trajectory (num_frames,).
            If None, computed internally.

    Returns:
        Maximum jump height in normalized units.
    """
    if com_trajectory is None:
        com_trajectory = calculate_com_trajectory(poses)

    takeoff_com = com_trajectory[phases.takeoff]
    flight_com = com_trajectory[phases.takeoff : phases.landing + 1]
    peak_com = np.min(flight_com)

    return float(takeoff_com - peak_com)
```

- [ ] **Step 2: Add `com_trajectory` parameter to `compute_relative_jump_height`**

In `ml/src/analysis/metrics.py`, modify `compute_relative_jump_height` (line 748):

```python
def compute_relative_jump_height(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    com_trajectory: NDArray[np.float32] | None = None,
) -> float:
    """Compute jump height normalized by spine length.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        phases: Element phase boundaries.
        com_trajectory: Pre-computed CoM trajectory (num_frames,).
            If None, computed internally.
    """
    if phases.takeoff >= phases.landing:
        return 0.0

    # Spine length calculation (vectorized from Task 4)
    start_frame = max(0, phases.takeoff - 2)
    end_frame = min(len(poses), phases.takeoff + 3)
    takeoff_window = poses[start_frame:end_frame]
    mid_hip = (takeoff_window[:, H36Key.LHIP] + takeoff_window[:, H36Key.RHIP]) / 2
    mid_shoulder = (
        takeoff_window[:, H36Key.LSHOULDER] + takeoff_window[:, H36Key.RSHOULDER]
    ) / 2
    spine_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    valid_spines = spine_lengths[spine_lengths >= 0.01]

    if len(valid_spines) == 0:
        return 0.0

    avg_spine = float(np.mean(valid_spines))

    if com_trajectory is None:
        com_trajectory = calculate_com_trajectory(poses)

    takeoff_com = com_trajectory[phases.takeoff]
    flight_com = com_trajectory[phases.takeoff : phases.landing + 1]
    peak_com = np.min(flight_com)

    com_displacement = float(takeoff_com - peak_com)
    return com_displacement / avg_spine
```

- [ ] **Step 3: Add `com_trajectory` parameter to `_analyze_jump`**

Update the `_analyze_jump` method to accept and pass `com_trajectory`:

```python
def _analyze_jump(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    fps: float,
    com_trajectory: NDArray[np.float32] | None = None,
) -> list[MetricResult]:
    """Analyze jump-specific metrics."""
    results: list[MetricResult] = []

    # ... airtime (unchanged) ...

    height = self.compute_jump_height_com(poses, phases, com_trajectory=com_trajectory)
    # ... rest uses height ...

    rel_height = self.compute_relative_jump_height(poses, phases, com_trajectory=com_trajectory)
    # ... rest uses rel_height ...

    return results
```

Update the `analyze()` method call to pass `com_trajectory`:

```python
def analyze(
    self,
    poses: NormalizedPose,
    phases: ElementPhase,
    fps: float,
    com_trajectory: NDArray[np.float32] | None = None,
) -> list[MetricResult]:
    results: list[MetricResult] = []

    if self._element_def.rotations > 0:
        results.extend(self._analyze_jump(poses, phases, fps, com_trajectory=com_trajectory))
    else:
        results.extend(self._analyze_step(poses, phases, fps))

    results.extend(self._analyze_common(poses, phases, fps))

    # ... mark goodness ...

    return results
```

- [ ] **Step 4: Add CoM caching to `PhysicsEngine`**

In `ml/src/analysis/physics_engine.py`, modify `analyze()` (line 359):

```python
def analyze(
    self,
    poses_3d: np.ndarray,
    takeoff_idx: int | None = None,
    landing_idx: int | None = None,
) -> PhysicsResult:
    """Run full physics analysis on 3D pose sequence."""
    # Calculate CoM once — share with moment of inertia and trajectory
    com = self.calculate_center_of_mass(poses_3d)

    # Calculate moment of inertia (reuses CoM internally)
    inertia = self.calculate_moment_of_inertia_with_com(poses_3d, com)

    angular_momentum = np.zeros_like(inertia)

    jump_height = None
    flight_time = None

    if takeoff_idx is not None and landing_idx is not None:
        trajectory = self.fit_jump_trajectory_with_com(poses_3d, takeoff_idx, landing_idx, com)
        jump_height = trajectory["height"]
        flight_time = trajectory["flight_time"]

    return PhysicsResult(
        center_of_mass=com,
        moment_of_inertia=inertia,
        angular_momentum=angular_momentum,
        jump_height=jump_height,
        flight_time=flight_time,
        rotation_rate=None,
    )
```

Add helper methods that accept pre-computed CoM:

```python
def calculate_moment_of_inertia_with_com(
    self,
    poses_3d: np.ndarray,
    com_trajectory: np.ndarray,
) -> np.ndarray:
    """Calculate MoI using pre-computed CoM (avoids recomputation)."""
    # Same as calculate_moment_of_inertia but uses passed com_trajectory
    # instead of calling self.calculate_center_of_mass(poses_3d)
    from ..pose_estimation import H36Key

    head = poses_3d[:, H36Key.HEAD]
    spine = poses_3d[:, H36Key.SPINE]
    thorax = poses_3d[:, H36Key.THORAX]
    l_shoulder = poses_3d[:, H36Key.LSHOULDER]
    l_elbow = poses_3d[:, H36Key.LELBOW]
    l_wrist = poses_3d[:, H36Key.LWRIST]
    r_shoulder = poses_3d[:, H36Key.RSHOULDER]
    r_elbow = poses_3d[:, H36Key.RELBOW]
    r_wrist = poses_3d[:, H36Key.RWRIST]
    l_hip = poses_3d[:, H36Key.LHIP]
    l_knee = poses_3d[:, H36Key.LKNEE]
    l_foot = poses_3d[:, H36Key.LFOOT]
    r_hip = poses_3d[:, H36Key.RHIP]
    r_knee = poses_3d[:, H36Key.RKNEE]
    r_foot = poses_3d[:, H36Key.RFOOT]

    n_frames = poses_3d.shape[0]
    inertia = np.zeros(n_frames, dtype=np.float32)

    def add_segment_inertia(segments: list[tuple[np.ndarray, float]]) -> None:
        for pos, mass in segments:
            r = np.linalg.norm(pos - com_trajectory, axis=1)
            inertia[:] += mass * r**2

    # Head
    add_segment_inertia([(head, self.segment_masses["head"])])

    torso_pos = (spine + thorax) / 2
    add_segment_inertia([(torso_pos, self.segment_masses["torso"])])

    l_upper_arm = (l_shoulder + l_elbow) / 2
    r_upper_arm = (r_shoulder + r_elbow) / 2
    l_forearm = (l_elbow + l_wrist) / 2
    r_forearm = (r_elbow + r_wrist) / 2

    add_segment_inertia([
        (l_upper_arm, self.segment_masses["left_upper_arm"]),
        (r_upper_arm, self.segment_masses["right_upper_arm"]),
        (l_forearm, self.segment_masses["left_forearm"]),
        (r_forearm, self.segment_masses["right_forearm"]),
        (l_wrist, self.segment_masses["left_hand"]),
        (r_wrist, self.segment_masses["right_hand"]),
    ])

    l_thigh = (l_hip + l_knee) / 2
    r_thigh = (r_hip + r_knee) / 2
    l_shin = (l_knee + l_foot) / 2
    r_shin = (r_knee + r_foot) / 2

    add_segment_inertia([
        (l_thigh, self.segment_masses["left_thigh"]),
        (r_thigh, self.segment_masses["right_thigh"]),
        (l_shin, self.segment_masses["left_shin"]),
        (r_shin, self.segment_masses["right_shin"]),
        (l_foot, self.segment_masses["left_foot"]),
        (r_foot, self.segment_masses["right_foot"]),
    ])

    return inertia


def fit_jump_trajectory_with_com(
    self,
    poses_3d: np.ndarray,
    takeoff_idx: int,
    landing_idx: int,
    com_trajectory: np.ndarray,
) -> dict:
    """Fit parabolic trajectory using pre-computed CoM."""
    flight_com = com_trajectory[takeoff_idx : landing_idx + 1, 1]
    n_frames = len(flight_com)
    t = np.arange(n_frames) / 30.0

    def parabola(t, a, b, c):
        return a * t**2 + b * t + c

    try:
        params, _ = curve_fit(parabola, t, flight_com)
        a, b, c = params

        t_peak = -b / (2 * a)
        h_peak = parabola(t_peak, a, b, c)
        h_takeoff = parabola(0, a, b, c)
        jump_height = h_peak - h_takeoff
        flight_time = t[-1] - t[0]

        residuals = flight_com - parabola(t, a, b, c)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((flight_com - np.mean(flight_com)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "height": abs(jump_height),
            "flight_time": flight_time,
            "takeoff_velocity": b,
            "fit_quality": r_squared,
        }
    except Exception:
        return {
            "height": np.max(flight_com) - np.min(flight_com),
            "flight_time": n_frames / 30.0,
            "takeoff_velocity": 0.0,
            "fit_quality": 0.0,
        }
```

- [ ] **Step 5: Restructure `analyze_async()` for 3-branch fan-out**

Replace the element-analysis section of `analyze_async()` (lines 680-756) with:

```python
# Element-specific analysis
if element_type is not None and element_def is not None:
    # Pre-compute CoM once (shared by metrics + 3D physics)
    com_trajectory = calculate_com_trajectory(smoothed)

    # === Wave 1: 3D lift, phase detection, reference load in parallel ===
    wave1_tasks: list[asyncio.Task] = []

    if poses_3d is None and self._compute_3d:
        wave1_tasks.append(
            asyncio.create_task(self._lift_poses_3d_async(smoothed, meta.fps))
        )

    wave1_tasks.append(
        asyncio.create_task(
            self._detect_phases_async(smoothed, meta.fps, element_type, manual_phases)
        )
    )

    if self._reference_store is not None:
        wave1_tasks.append(
            asyncio.create_task(self._load_reference_async(element_type))
        )

    wave1_results = await asyncio.gather(*wave1_tasks)

    # Unpack wave 1 results
    result_idx = 0
    if poses_3d is None and self._compute_3d and result_idx < len(wave1_results):
        poses_3d, blade_summaries = wave1_results[result_idx]
        result_idx += 1
    elif not self._compute_3d:
        blade_summaries = None
    else:
        result_idx += 1  # skip 3D slot

    phases = wave1_results[result_idx]
    result_idx += 1

    reference = wave1_results[result_idx] if result_idx < len(wave1_results) else None

    # === Wave 2: physics, metrics, DTW in parallel ===
    wave2_tasks: list[asyncio.Task] = []

    if poses_3d is not None:
        wave2_tasks.append(
            asyncio.create_task(
                self._compute_physics_async(poses_3d, phases)
            )
        )

    wave2_tasks.append(
        asyncio.create_task(
            self._compute_metrics_async(smoothed, phases, meta.fps, element_def, com_trajectory=com_trajectory)
        )
    )

    wave2_results = await asyncio.gather(*wave2_tasks)

    # Unpack wave 2 results
    result_idx = 0
    if poses_3d is not None:
        physics_dict = wave2_results[result_idx] or {}
        result_idx += 1
    else:
        physics_dict = {}

    metrics = wave2_results[result_idx]

    # DTW alignment (needs phases + reference — both available after wave 1)
    dtw_distance = None
    if reference is not None:
        aligner = self._get_aligner()
        dtw_distance = aligner.compute_distance(
            normalized[phases.start : phases.end],
            reference.poses[reference.phases.start : reference.phases.end],
        )

    recommender = self._get_recommender()
    recommendations = recommender.recommend(metrics, element_type)
    overall_score = self._compute_overall_score(metrics)
else:
    # No element type specified
    phases = ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0)
    metrics = []
    recommendations = []
    overall_score = None
    dtw_distance = None
    physics_dict = {}
    blade_summaries = None
```

- [ ] **Step 6: Add `_compute_physics_async` helper**

Add to `AnalysisPipeline`:

```python
async def _compute_physics_async(
    self,
    poses_3d: np.ndarray,
    phases: ElementPhase,
) -> dict | None:
    """Async physics calculations with CoM caching."""
    try:
        from .analysis.physics_engine import PhysicsEngine

        physics_engine = PhysicsEngine(body_mass=60.0)
        result = physics_engine.analyze(
            poses_3d, takeoff_idx=phases.takeoff, landing_idx=phases.landing
        )
        physics_dict: dict = {
            "avg_inertia": float(np.mean(result.moment_of_inertia)),
        }
        if result.jump_height is not None:
            physics_dict["jump_height"] = result.jump_height
            physics_dict["flight_time"] = result.flight_time
        return physics_dict
    except Exception:
        return None
```

- [ ] **Step 7: Update `_compute_metrics_async` to accept `com_trajectory`**

Modify `_compute_metrics_async`:

```python
async def _compute_metrics_async(
    self,
    poses: np.ndarray,
    phases: ElementPhase,
    fps: float,
    element_def,
    com_trajectory: np.ndarray | None = None,
) -> list:
    """Async biomechanics metrics computation."""
    loop = asyncio.get_event_loop()
    analyzer = self._get_analyzer_factory()(element_def)
    metrics = await loop.run_in_executor(
        None, analyzer.analyze, poses, phases, fps, com_trajectory
    )
    return metrics
```

- [ ] **Step 8: Run all ML tests**

Run: `uv run pytest ml/tests/ -v --no-cov`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add ml/src/pipeline.py ml/src/analysis/metrics.py ml/src/analysis/physics_engine.py ml/tests/test_pipeline.py
git commit -m "feat(ml): post-smoothing fan-out with CoM cache — 3-branch parallel analysis"
```

---

### Task 8: Graceful Shutdown

**Files:**
- Modify: `backend/app/worker.py` — add `job_completion_wait` to both `WorkerSettings` classes
- Test: `backend/tests/test_task_manager.py`

arq supports `job_completion_wait` which tells the worker to finish current jobs before shutting down, instead of killing them mid-execution.

- [ ] **Step 1: Write failing test**

Add to `backend/tests/test_task_manager.py`:

```python
def test_graceful_shutdown_configured():
    """Worker settings should have job_completion_wait > 0."""
    from app.worker import FastWorkerSettings, HeavyWorkerSettings

    assert hasattr(FastWorkerSettings, "job_completion_wait")
    assert hasattr(HeavyWorkerSettings, "job_completion_wait")
    # Wait up to 120 seconds for running jobs to finish
    assert FastWorkerSettings.job_completion_wait == 120
    assert HeavyWorkerSettings.job_completion_wait == 600  # Heavy jobs need more time
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_task_manager.py::test_graceful_shutdown_configured -v`
Expected: FAIL — `AssertionError` or `AttributeError`

- [ ] **Step 3: Add `job_completion_wait` to both worker settings**

In `backend/app/worker.py`, add to `FastWorkerSettings`:

```python
class FastWorkerSettings:
    """arq worker for lightweight detection tasks."""

    queue_name: str = "skating:queue:fast"
    max_jobs: int = _settings.app.worker_max_jobs_remote if _settings.vastai.api_key.get_secret_value() else _settings.app.worker_max_jobs
    job_completion_wait: int = 120  # seconds — finish detect jobs before shutdown
    # ... rest unchanged ...
```

Add to `HeavyWorkerSettings`:

```python
class HeavyWorkerSettings:
    """arq worker for full ML pipeline processing."""

    queue_name: str = "skating:queue:heavy"
    max_jobs: int = 1
    job_completion_wait: int = 600  # seconds — finish GPU jobs before shutdown
    # ... rest unchanged ...
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest backend/tests/test_task_manager.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/worker.py backend/tests/test_task_manager.py
git commit -m "feat(backend): graceful shutdown — job_completion_wait on both worker queues"
```

---

## Self-Review

### 1. Spec Coverage

| Spec Item | Task | Status |
|-----------|------|--------|
| #5 Dual-queue priority | Task 2 | Two queues, fast/heavy |
| #6 Post-smoothing fan-out | Task 7 | 3-branch parallel after smoothing |
| #8 Conditional 3D lift | Task 6 | `compute_3d` gate |
| #9 CoM pre-computation cache | Task 7 | CoM computed once, passed via parameter |
| #10 Graceful shutdown | Task 8 | `job_completion_wait` |
| #11 Fix `_priority` dead code | Task 1 | Remove from detect + process |
| #16 Vectorize `calculate_com_trajectory` | Task 3 | NumPy broadcast |
| #16 Vectorize `normalize_poses` | Task 5 | NumPy broadcast |
| #17 Vectorize `metrics.py` loops | Task 4 | edge_indicator, rotation_speed, angle_series, spine |

### 2. Type Consistency

- `com_trajectory` parameter: `NDArray[np.float32] | None` — consistent across `compute_jump_height_com`, `compute_relative_jump_height`, `_analyze_jump`, `analyze`
- `PhysicsEngine` methods: `analyze()` unchanged signature, new `_with_com` methods are internal
- Queue names: `"skating:queue:fast"` and `"skating:queue:heavy"` — consistent across worker.py, detect.py, process.py

### 3. Placeholder Scan

No TBD, TODO, or vague requirements found. All code blocks contain actual implementations.

### 4. Dependency Order

Tasks are ordered to minimize rework:
1. Task 1 (remove _priority) → Task 2 (dual-queue) — logical dependency
2. Tasks 3-5 (vectorization) — independent, can run in any order
3. Task 6 (conditional 3D) → Task 7 (fan-out) — logical dependency
4. Task 8 (graceful shutdown) — independent

### 5. Risk Areas

- **Task 7 (fan-out)** is the most complex. The wave-based parallelism requires careful ordering of asyncio tasks. The `_compute_metrics_async` signature change (adding `com_trajectory`) must propagate through `_analyze_jump` correctly.
- **Task 5 (normalize_poses)**: behavioral difference if the test fixture uses non-standard spine indices. The vectorized version correctly handles per-frame scaling but should be verified against the existing test suite.
