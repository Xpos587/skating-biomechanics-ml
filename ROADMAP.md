# Figure Skating Biomechanics ML - Roadmap

**Status:** MVP ~90% complete | Last updated: 2026-03-28

> **This is the SINGLE SOURCE OF TRUTH for project status.** All implementation decisions and priority changes must be reflected here first.

---

## Vision

AI-тренер по фигурному катанию который анализирует видео и даёт рекомендации на русском языке.

**Target Users:** Figure skaters and coaches looking for technical feedback
**Input:** Video recordings (mp4, webm, etc.)
**Output:** Biomechanics analysis + Russian recommendations

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Video     │ -> │  Detection   │ -> │  Pose 2D    │ -> │ Normalized   │
│   Input     │    │  (YOLOv11n)   │    │ (BlazePose)  │    │   Poses       │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  v
┌──────────────────────────────────────────────────────────────────────┐
│                        Analysis Pipeline                            │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Phase Detect │->│  Metrics    │->│  DTW Align  │->│ Recommend │ │
│  │   (TODO)     │  │  (Done)      │  │  (Fixing)   │  │ (Done)    │ │
│  └──────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                                                  v
                                                      ┌─────────────────────┐
                                                      │ Russian Text Report │
                                                      └─────────────────────┘
```

---

## Implementation Phases

### Phase 0: Foundation ✅ 100%
**Status:** Complete

- [x] Project structure
- [x] Type definitions (BKey, FrameKeypoints, etc.)
- [x] Video utilities (extract_frames, get_video_meta)
- [x] Geometry utilities (angles, distances)
- [x] Quality tooling (ruff, mypy, pytest)

**Files:** `types.py`, `utils/`, `pyproject.toml`

---

### Phase 1: Person Detection ✅ 100%
**Status:** Complete

- [x] YOLOv11n integration
- [x] PersonDetector class
- [x] Single-frame detection
- [x] Full-video detection
- [x] BoundingBox type

**Files:** `detection/person_detector.py`
**Tests:** `tests/detection/test_person_detector.py`

---

### Phase 2: 2D Pose Estimation ✅ 100%
**Status:** Complete

- [x] BlazePose integration (33 keypoints)
- [x] BlazePoseExtractor class
- [x] Pixel coordinates extraction
- [x] Confidence values
- [x] Alternative: YOLO-Pose (17 keypoints)

**Files:** `pose_2d/blazepose_extractor.py`, `pose_2d/pose_extractor.py`
**Tests:** `tests/pose_2d/test_blazepose_extractor.py`

---

### Phase 3: Pose Normalization ✅ 100%
**Status:** Complete

- [x] Root-centering (mid-hip → origin)
- [x] Scale normalization (spine length → 0.4)
- [x] PoseNormalizer class
- [x] Coordinate type system (PixelPose vs NormalizedPose)
- [x] Runtime validation (assert_pose_format)

**Files:** `pose_2d/normalizer.py`, `types.py` (coordinate types)
**Tests:** `tests/pose_2d/test_normalizer.py`

---

### Phase 4: Temporal Smoothing ✅ 100%
**Status:** Complete

- [x] One-Euro Filter implementation
- [x] PoseSmoother class
- [x] Skating-optimized config
- [x] 29% jitter reduction achieved
- [x] Normalized-space smoothing

**Files:** `utils/smoothing.py`
**Tests:** `tests/utils/test_smoothing.py`

---

### Phase 5: Biomechanics Metrics ✅ 100%
**Status:** Complete

- [x] Airtime calculation
- [x] Jump height (hip trajectory)
- [x] Knee angles (hip-knee-ankle)
- [x] Arm position
- [x] Edge detection (inside/outside/flat)
- [x] Rotation speed
- [x] BiomechanicsAnalyzer class

**Files:** `analysis/metrics.py`
**Tests:** `tests/analysis/test_metrics.py`

---

### Phase 6: Phase Detection ⚠️ 50%
**Status:** MANUAL ONLY - auto-detection NOT working

- [x] ElementPhase data structure
- [x] Manual phase specification via CLI
- [ ] **Auto takeoff detection** (height threshold)
- [ ] **Auto peak detection** (min hip y)
- [ ] **Auto landing detection** (impact detection)
- [ ] Phase transition smoothing

**Issue:** PhaseDetector always returns takeoff=0, landing=end
**Priority:** HIGH - blocks fully automated analysis

**Files:** `analysis/phase_detector.py`
**Tests:** `tests/analysis/test_phase_detector.py`

---

### Phase 7: DTW Motion Alignment ⚠️ 70%
**Status:** Code exists, tests failing

- [x] DTW implementation (dtw-python)
- [x] Sakoe-Chiba window
- [x] MotionAligner class
- [ ] **Fix tests** - expects 17 keypoints, BlazePose has 33
- [ ] Multi-segment alignment
- [ ] Alignment quality metrics

**Issue:** Test suite uses old 17-keypoint format, need update for 33-keypoint BlazePose
**Priority:** MEDIUM - manual analysis works without this

**Files:** `alignment/aligner.py`, `alignment/motion_dtw.py`
**Tests:** `tests/alignment/test_aligner.py` (7 failing)

---

### Phase 8: Rule-Based Recommender ✅ 100%
**Status:** Complete

- [x] Rule engine for each element
- [x] MetricResult validation
- [x] Russian text generation
- [x] jump_rules.py (waltz_jump, toe_loop, flip, salchow, loop, lutz, axel)
- [x] three_turn_rules.py
- [x] Recommender class

**Files:** `analysis/recommender.py`, `analysis/rules/`
**Tests:** `tests/analysis/test_recommender.py`

---

### Phase 9: Reference System ✅ 100%
**Status:** Complete

- [x] ReferenceData type
- [x] Save/Load .npz files
- [x] ReferenceBuilder CLI
- [x] Element definitions (ideal ranges)
- [x] Reference directory structure

**Files:** `references/element_defs.py`, `references/reference_builder.py`, `references/reference_store.py`

**Usage:**
```bash
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 \
    --element waltz_jump --takeoff 1.0 --peak 1.2 --landing 1.4
```

---

### Phase 10: Automatic Segmentation ✅ 90%
**Status:** Mostly working

- [x] ElementSegmenter class
- [x] Motion-based segmentation
- [x] Element type classification
- [x] JSON export
- [x] Segment visualization
- [ ] Refine segment boundaries (includes preparation/recovery)

**Issue:** Segments include too much context (preparation, recovery)
**Priority:** LOW - core functionality works

**Files:** `segmentation/element_segmenter.py`, `scripts/visualize_segmentation.py`

---

### Phase 11: Visualization ✅ 100%
**Status:** Complete

- [x] draw_skeleton() - 33-keypoint overlay
- [x] draw_velocity_vectors() - speed visualization
- [x] draw_trails() - motion history
- [x] draw_edge_indicators() - inside/outside/flat
- [x] draw_debug_hud() - telemetry overlay
- [x] Layered HUD system (0-3)
- [x] Cyrillic text support (Pillow)
- [x] Frame-perfect synchronization

**Files:** `utils/visualization.py`, `scripts/visualize_with_skeleton.py`
**Tests:** `tests/utils/test_visualization.py` (19 passing)

---

### Phase 12: CLI & Pipeline ✅ 100%
**Status:** Complete

- [x] argparse CLI (analyze, build-ref, segment)
- [x] AnalysisPipeline orchestrator
- [x] Russian output
- [x] Help text and examples

**Commands:**
```bash
# Analyze video
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element waltz_jump

# Build reference
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 --element three_turn

# Segment video
uv run python -m skating_biomechanics_ml.cli segment video.mp4

# Visualize with debug overlay
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3
```

**Files:** `cli.py`, `pipeline.py`

---

### Phase 13: Blade Edge Detection ✅ 100%
**Status:** Complete (NEW!)

**Based on Research:**
- "Automated Blade Type Discrimination Algorithm for Figure Skating Based on MediaPipe" (2025)
- "Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation" (2023)

- [x] BladeType enum (INSIDE, OUTSIDE, FLAT, TOE_PICK, UNKNOWN)
- [x] BladeEdgeDetector class with BDA Algorithm
- [x] Foot angle calculation relative to motion direction
- [x] Ankle flexion angle calculation
- [x] Vertical acceleration for toe pick detection
- [x] Temporal smoothing (majority voting)
- [x] Takeoff/landing detection from blade state sequence
- [x] 19 unit tests (all passing)
- [x] Blade summary statistics

**Key Insights from Research:**
1. **AthletePose3D** (2025) - New dataset with figure skating 3D poses!
2. **FS-Jump3D** - Public dataset for figure skating jumps
3. **BDA Algorithm** - 4 angular thresholds for blade transitions:
   - Strong inside: angle < -20°
   - Weak inside: -20° to -10°
   - Flat: -10° to 10°
   - Weak outside: 10° to 20°
   - Strong outside: angle > 20°
4. **ISU + Omega** (2026) - Multi-camera system achieving millimeter accuracy

**Files:** `utils/blade_edge_detector.py`, `types.py` (BladeType enum)
**Tests:** `tests/utils/test_blade_edge_detector.py` (19 passing)

**Usage:**
```python
from skating_biomechanics_ml.utils import BladeEdgeDetector

detector = BladeEdgeDetector(
    inside_threshold=-15.0,
    outside_threshold=15.0,
    smoothing_window=3
)
states = detector.detect_sequence(poses, fps=30.0, foot="left")
summary = detector.get_blade_summary(states)
```

---

## Current Blockers

### HIGH Priority
1. **Auto phase detection** - Manual specification required
   - Impact: Cannot fully automate analysis
   - Solution: Implement height-based takeoff, peak, landing detection

### MEDIUM Priority
2. **DTW alignment tests** - Test suite outdated
   - Impact: Cannot verify alignment correctness
   - Solution: Update tests for 33-keypoint format

### LOW Priority
3. **Segment boundaries** - Too broad, includes preparation
   - Impact: Segments not precise
   - Solution: Trim to element core motion

---

## Next Steps (Priority Order)

1. **Fix Auto Phase Detection** ⚠️ HIGH
   - Implement height threshold for takeoff
   - Find minimum hip y for peak
   - Detect landing impact
   - Estimated: 2-3 hours

2. **Fix DTW Tests** ⚠️ MEDIUM
   - Update test data for 33 keypoints
   - Fix shape mismatches
   - Estimated: 1 hour

3. **Integrate Blade Detection into Analysis Pipeline** 🆕 MEDIUM
   - Add blade state to MetricResult
   - Update rules to use edge information
   - Add edge visualization to HUD
   - Estimated: 1-2 hours

4. **Improve Segmentation** 📝 LOW
   - Trim segment boundaries
   - Remove preparation/recovery
   - Estimated: 2-3 hours

---

## Future Enhancements (Research Findings)

### Physics-Based Improvements
1. **Physics-Informed Pose Validation**
   - Bone length consistency checks (reduces MPJPE by 10%)
   - Biomechanically realistic pose filtering
   - Reference: "Physics Informed Human Posture Estimation" (2025)

2. **3D Pose Estimation for Occlusion**
   - **PoseMamba** - State Space Model with linear complexity
   - **STRIDE** - Temporally continuous occlusion-robust 3D pose
   - **Di²Pose** - Discrete diffusion for occluded poses
   - **AthletePose3D** - Dataset with 1.3M frames including figure skating!

3. **Physical Parameter Estimation**
   - **A2B Model** - Anthropometric measurements → SMPL beta (MPJPE -30mm)
   - **SMPLest-X** - SOTA pose + shape estimation (TPAMI 2025)
   - Height/weight estimation from skeleton proportions

4. **Multi-Person Tracking**
   - **Deep HM-SORT** - 80.1 HOTA on SportsMOT
   - **Basketball-SORT** - Handles 3+ person occlusions
   - Re-identification during brief occlusions

5. **Hierarchical Element Classification**
   - **FS-Jump3D** - Public 3D pose dataset for jumps
   - **MMFS** - 11,672 clips, 256 categories with skeleton data
   - **MCFS** - Motion-Centered Figure Skating dataset
   - **VIFSS** - View-Invariant Figure Skating-Specific representation

### Data Resources
| Dataset | Content | Link |
|---------|---------|------|
| FS-Jump3D | 3D pose jumps with optical markerless mocap | github.com/ryota-skating/FS-Jump3D |
| MMFS | 11,672 clips, 256 categories, skeleton | Multi-modality dataset |
| MCFS | Temporal action segmentation | shenglanliu.github.io/mcfs-dataset |
| AthletePose3D | 1.3M frames, 12 sports including skating | Nagoya University |

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 0.1 | 2026-03-27 | MVP 85% | Core pipeline working, visualization complete |
| 0.2 | 2026-03-28 | MVP 90% | Blade edge detection (BDA algorithm), research findings |

---

## Reference

- Original architecture: `research/RESEARCH.md`
- Visualization research: `research/VISUALIZATION_RESEARCH_PROMPT.md`
- **Physics/Blade detection research:** `research/PHYSICS_DETECTION_RESEARCH.md` 🆕
- API documentation: See individual module docstrings
