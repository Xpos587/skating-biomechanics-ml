# Parabolic Flight Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace velocity-threshold phase detection with parabolic CoM fitting that won't trigger on preparation movements (leg swings, crouches).

**Architecture:** A new `_detect_jump_phases_parabolic()` method in `PhaseDetector` finds contiguous CoM elevation segments, fits parabolas to each, and selects the best one by R² × excursion score. Falls back to the existing velocity method if no parabola passes quality checks.

**Tech Stack:** numpy (polyfit), scipy.ndimage (median_filter), existing `calculate_com_trajectory()` from `src/utils/geometry.py`

---

## Context

### The Problem

The current `_detect_jump_phases_com_improved()` in `src/analysis/phase_detector.py:72-205` uses CoM velocity peaks above 2σ/3σ thresholds to find takeoff/landing. This triggers on **preparation movements** — leg swings, crouch-and-extend sequences, cross rolls — because these produce large velocity spikes even though CoM stays flat.

**Real example:** On a 2S attempt (out_poses.npy, 334 frames), the detector found takeoff at f239 (a leg swing with lean=-86°) instead of the real takeoff at f278. CoM at f239 was 471.7 — barely moved from baseline 471.1. The real flight f278-f289 had CoM excursion 463→454→467 (a clear parabola).

### Why Parabolic Fitting Works

During true flight, the only force is gravity → CoM follows `y(t) = ½gt² + v₀t + y₀`. A parabola. During preparation (leg swing, crouch), CoM stays near baseline — any attempt to fit a parabola gets terrible R². This is the discriminating signal.

### Files Involved

| File | Role |
|------|------|
| `src/analysis/phase_detector.py` | **Modify** — add `_detect_jump_phases_parabolic()`, wire into `detect_jump_phases()` |
| `src/utils/geometry.py` | **Reuse** — `calculate_com_trajectory()` (line 214), `smooth_signal()` (line 97) |
| `tests/analysis/test_phase_detector.py` | **Modify** — add parabolic-specific tests |
| `out_poses.npy` / `out_biomechanics.csv` | **Validation** — real 2S data, expect takeoff≈f278, peak≈f283, landing≈f289 |

### Existing Code to Reuse

- `calculate_com_trajectory(poses)` → `src/utils/geometry.py:214-229` — returns (N,) float32 array of CoM Y
- `smooth_signal(signal, window)` → `src/utils/geometry.py:97-120` — moving average
- `PhaseDetectionResult(phases, confidence)` → `src/analysis/metrics.py:581-588` — result dataclass
- `ElementPhase(name, start, takeoff, peak, landing, end)` → `src/types.py:438` — phase boundaries
- `scipy.optimize.curve_fit` parabola fit in `src/analysis/physics_engine.py:283-284` — same `parabola(t, a, b, c)` model

---

## File Structure

```
src/analysis/phase_detector.py    # Add parabolic method, rewire detect_jump_phases
tests/analysis/test_phase_detector.py  # Add 4 new tests
```

No new files. One module change, one test file change.

---

### Task 1: Add parabolic detection method

**Files:**
- Modify: `src/analysis/phase_detector.py` — add method after line 205 (after `_detect_jump_phases_com_improved`)
- Test: `tests/analysis/test_phase_detector.py`

- [ ] **Step 1: Write the failing tests**

Add these 4 tests to `tests/analysis/test_phase_detector.py`, after the existing `TestPhaseDetectionResult` class:

```python
class TestParabolicFlightDetector:
    """Test parabolic CoM fitting for flight detection."""

    def _make_jump_poses(self, n_frames=80, takeoff=25, peak=40, landing=55):
        """Create poses with a clean parabolic CoM trajectory."""
        poses = np.zeros((n_frames, 17, 2), dtype=np.float32)
        baseline_y = 0.3
        for i in range(n_frames):
            # Set all keypoints to same Y for simple CoM
            for kp in range(17):
                poses[i, kp, 1] = baseline_y

            if takeoff <= i <= landing:
                # Parabolic arc: CoM drops in image coords (lower Y = higher)
                t = (i - takeoff) / (landing - takeoff)  # 0..1
                height = -0.15 * 4 * t * (1 - t)  # parabola, max -0.15 at t=0.5
                for kp in [H36Key.PELVIS, H36Key.LHIP, H36Key.RHIP, H36Key.THORAX]:
                    poses[i, kp, 1] = baseline_y + height
        return poses

    def test_parabolic_detects_clean_jump(self):
        """Should detect a clean parabolic jump."""
        detector = PhaseDetector()
        poses = self._make_jump_poses()

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.takeoff < result.phases.peak < result.phases.landing
        assert result.confidence > 0.5

    def test_parabolic_ignores_prep_movement(self):
        """Should not trigger on preparation movement with flat CoM."""
        detector = PhaseDetector()
        n = 80
        poses = np.zeros((n, 17, 2), dtype=np.float32)
        baseline_y = 0.3

        # Frames 10-25: big lean changes but CoM stays flat (leg swing)
        for i in range(n):
            for kp in range(17):
                poses[i, kp, 1] = baseline_y
            # Simulate lean by moving shoulders but keeping CoM stable
            if 10 <= i <= 25:
                poses[i, H36Key.LSHOULDER, 1] = baseline_y + 0.05 * np.sin(i * 0.5)
                poses[i, H36Key.RSHOULDER, 1] = baseline_y - 0.05 * np.sin(i * 0.5)
            # Real jump at frames 40-55
            if 40 <= i <= 55:
                t = (i - 40) / 15
                height = -0.15 * 4 * t * (1 - t)
                for kp in [H36Key.PELVIS, H36Key.LHIP, H36Key.RHIP, H36Key.THORAX]:
                    poses[i, kp, 1] = baseline_y + height

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        # Should detect jump around frames 40-55, NOT 10-25
        assert result.phases.takeoff >= 30  # well past the prep movement
        assert result.phases.peak > result.phases.takeoff
        assert result.phases.landing > result.phases.peak

    def test_parabolic_fallback_on_no_jump(self):
        """Should fallback to velocity method when no parabola found."""
        detector = PhaseDetector()
        # Flat poses — no jump at all
        poses = np.full((50, 17, 2), 0.3, dtype=np.float32)

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)

    def test_parabolic_short_sequence(self):
        """Should handle very short sequences gracefully."""
        detector = PhaseDetector()
        poses = np.zeros((10, 17, 2), dtype=np.float32)

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_phase_detector.py::TestParabolicFlightDetector -v`
Expected: FAIL — `_detect_jump_phases_parabolic` does not exist yet.

- [ ] **Step 3: Add the new import and method**

Add `from scipy.ndimage import median_filter` to the imports at the top of `src/analysis/phase_detector.py` (after line 8, `from scipy.signal import find_peaks`).

Add the following method to the `PhaseDetector` class, after `_detect_jump_phases_com_improved` (after line 205, before `detect_three_turn_phases`):

```python
    def _detect_jump_phases_parabolic(
        self, poses: NormalizedPose, fps: float
    ) -> PhaseDetectionResult:
        """Detect jump phases using parabolic CoM fitting.

        Finds segments where CoM drops below baseline (person rises),
        fits a parabola to each, and picks the best fit. During true
        flight, only gravity acts → CoM is parabolic. Preparation
        movements (leg swings, crouches) have flat CoM → poor fit.

        Falls back to _detect_jump_phases_com_improved if no good
        parabola is found.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        from scipy.ndimage import median_filter

        N = len(poses)
        if N < 10:
            return self._detect_jump_phases_com_improved(poses, fps)

        # 1. Compute and smooth CoM
        com_y = calculate_com_trajectory(poses)
        com_smooth = median_filter(com_y.astype(np.float64), size=5)

        # 2. Baseline: large-window median (represents "on ice" level)
        bl_window = min(61, max(21, N // 3))
        baseline = median_filter(com_smooth, size=bl_window)

        # 3. Find elevation threshold
        diff = com_smooth - baseline
        std = np.std(diff)
        if std < 1e-6:
            return self._detect_jump_phases_com_improved(poses, fps)
        threshold = 1.0 * std

        # 4. Find contiguous elevated segments (com_y < baseline - threshold)
        # In image coords: lower Y = person higher
        elevated = com_smooth < (baseline - threshold)

        # Merge close segments (gap < 3 frames → same event)
        segments = []
        seg_start = None
        for i in range(N):
            if elevated[i]:
                if seg_start is None:
                    seg_start = i
            else:
                if seg_start is not None:
                    # Check if gap to next elevated is small
                    look_ahead = min(i + 3, N)
                    if np.any(elevated[i:look_ahead]):
                        continue  # skip, gap too small to split
                    segments.append((seg_start, i - 1))
                    seg_start = None
        if seg_start is not None:
            segments.append((seg_start, N - 1))

        # Filter by minimum duration (0.2s)
        min_frames = max(5, int(0.2 * fps))
        segments = [(s, e) for s, e in segments if (e - s) >= min_frames]

        if not segments:
            return self._detect_jump_phases_com_improved(poses, fps)

        # 5. Fit parabola to each segment, pick best
        best = None
        best_score = -1.0

        for seg_start, seg_end in segments:
            # Extend segment by 3 frames each side for better fit
            ext_start = max(0, seg_start - 3)
            ext_end = min(N, seg_end + 4)
            segment = com_smooth[ext_start:ext_end]
            n_seg = len(segment)
            t = np.arange(n_seg, dtype=np.float64)

            # Fit parabola y(t) = a*t^2 + b*t + c
            try:
                coeffs = np.polyfit(t, segment, 2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            a, b, c = coeffs

            # In image coords: flight → Y drops → parabola opens UP (a > 0)
            if a <= 1e-8:
                continue

            # Peak must be inside the segment
            t_peak = -b / (2 * a)
            peak_local = int(round(t_peak))
            if peak_local < 1 or peak_local >= n_seg - 1:
                continue

            # R² quality check
            fitted = np.polyval(coeffs, t)
            ss_res = np.sum((segment - fitted) ** 2)
            ss_tot = np.sum((segment - np.mean(segment)) ** 2)
            r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            if r_sq < 0.80:
                continue

            # Excursion: how far CoM dropped below baseline at peak
            peak_global = ext_start + peak_local
            excursion = baseline[peak_global] - segment[peak_local]

            # Physical sanity: excursion should be meaningful
            if excursion < 0.5 * threshold:
                continue

            # Score: fit quality × magnitude
            score = r_sq * excursion

            if score > best_score:
                best_score = score

                # Find takeoff: backward from peak, where CoM crosses near baseline
                takeoff_idx = ext_start
                for j in range(peak_local, -1, -1):
                    if segment[j] >= baseline[ext_start + j] - threshold * 0.3:
                        takeoff_idx = ext_start + j
                        break

                # Find landing: forward from peak, where CoM crosses near baseline
                landing_idx = ext_start + n_seg - 1
                for j in range(peak_local, n_seg):
                    if segment[j] >= baseline[ext_start + j] - threshold * 0.3:
                        landing_idx = ext_start + j
                        break

                best = {
                    "takeoff": takeoff_idx,
                    "peak": peak_global,
                    "landing": landing_idx,
                    "r_sq": r_sq,
                    "excursion": excursion,
                }

        if best is None:
            return self._detect_jump_phases_com_improved(poses, fps)

        # Validate airtime
        airtime = (best["landing"] - best["takeoff"]) / fps
        if airtime < 0.2:
            return self._detect_jump_phases_com_improved(poses, fps)

        # Order validation
        takeoff_idx = best["takeoff"]
        peak_idx = best["peak"]
        landing_idx = best["landing"]

        if takeoff_idx >= peak_idx:
            takeoff_idx = max(0, peak_idx - 3)
        if landing_idx <= peak_idx:
            landing_idx = min(N - 1, peak_idx + 3)

        # Build result
        start_idx = max(0, takeoff_idx - 10)
        end_idx = min(N - 1, landing_idx + 10)

        phases = ElementPhase(
            name="jump",
            start=start_idx,
            takeoff=takeoff_idx,
            peak=peak_idx,
            landing=landing_idx,
            end=end_idx,
        )

        # Confidence: R² × normalized excursion
        prominence = best["excursion"]
        confidence = min(1.0, best["r_sq"] * min(1.0, prominence / (2 * threshold)))

        return PhaseDetectionResult(phases=phases, confidence=confidence)
```

- [ ] **Step 4: Wire parabolic detector into detect_jump_phases**

Change `detect_jump_phases` (line 56-70) to call the parabolic method:

```python
    def detect_jump_phases(self, poses: NormalizedPose, fps: float) -> PhaseDetectionResult:
        """Detect jump phases: takeoff, peak, landing.

        Uses parabolic CoM fitting as primary method. During true flight,
        only gravity acts → CoM follows a parabola. Preparation movements
        (leg swings, crouches) have flat CoM → poor parabolic fit.

        Falls back to velocity-based detection if no parabola found.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        return self._detect_jump_phases_parabolic(poses, fps)
```

Also add `"salchow"`, `"loop"`, `"lutz"`, `"axel"` to the jump dispatch in `detect_phases` (line 38):

```python
        if element_type in ("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_phase_detector.py -v`
Expected: ALL PASS — both old and new tests.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/phase_detector.py tests/analysis/test_phase_detector.py
git commit -m "feat(phase-detect): parabolic CoM fitting replaces velocity thresholds

Sliding-window parabolic fit on CoM trajectory finds true flight phases.
Velocity thresholds triggered on preparation movements (leg swings, crouches).
Parabola with R²>0.80 discriminates flight from prep — CoM is flat during prep.

Fallback to old velocity method when no parabola passes quality checks."
```

---

### Task 2: Validate on real 2S data

**Files:**
- No changes — validation only

- [ ] **Step 1: Run detector on out_poses.npy and verify correct phases**

```bash
uv run python -c "
import numpy as np
from src.analysis.phase_detector import PhaseDetector

poses_raw = np.load('out_poses.npy')  # (334, 17, 3)
# Strip confidence, keep x,y for NormalizedPose
poses = poses_raw[:, :, :2].astype(np.float32)

detector = PhaseDetector()
result = detector.detect_jump_phases(poses, fps=29.9)

print(f'Takeoff: f{result.phases.takeoff}')
print(f'Peak:    f{result.phases.peak}')
print(f'Landing: f{result.phases.landing}')
print(f'Confidence: {result.confidence:.3f}')
print(f'Airtime: {(result.phases.landing - result.phases.takeoff) / 29.9:.3f}s')
print()
print('Expected: takeoff≈f278, peak≈f283, landing≈f289')
"
```

Expected output:
```
Takeoff: f278 (or within ±5 frames)
Peak:    f283 (or within ±3 frames)
Landing: f289 (or within ±5 frames)
Confidence: > 0.5
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All existing tests still pass. New parabolic tests pass.

- [ ] **Step 3: Commit validation results (optional)**

No code changes — validation only.

---

## Verification Checklist

- [ ] `test_parabolic_detects_clean_jump` passes
- [ ] `test_parabolic_ignores_prep_movement` passes — prep at f10-25 does NOT become takeoff
- [ ] `test_parabolic_fallback_on_no_jump` passes — returns result (from fallback)
- [ ] `test_parabolic_short_sequence` passes — no crash
- [ ] Real 2S data: takeoff within ±5 frames of f278
- [ ] All existing tests in `test_phase_detector.py` still pass
- [ ] Full `uv run pytest tests/` passes (279+ tests)
