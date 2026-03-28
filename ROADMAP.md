# Figure Skating Biomechanics ML - Roadmap

**Status:** MVP ~96% complete | Last updated: 2026-03-28

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

### Phase 14: 3D Pose & Physics Engine ✅ 100%
**Status:** Complete

- [x] BlazePose 33 → H3.6M 17 keypoint mapping
- [x] PhysicsEngine (CoM, Moment of Inertia, Angular Momentum)
- [x] Parabolic trajectory fitting for jump height
- [x] AthletePose3DExtractor with MotionAGFormer integration
- [x] Load AthletePose3D fine-tuned models
- [x] 3D skeleton visualization in main HUD
- [x] CoM trajectory visualization

**Key Features:**
1. **blazepose_to_h36m()** - Convert 33kp BlazePose to 17kp H3.6M format
2. **PhysicsEngine** - Calculate CoM, I, L from 3D poses
   - Dempster anthropometric tables (segment masses)
   - Parabolic trajectory fit for accurate jump height
   - Solves the 60% flight time error problem!
3. **AthletePose3DExtractor** - Monocular 3D pose estimation
   - 81-frame temporal windows
   - MotionAgFormer-S (59MB) for real-time
   - State dict prefix stripping for compatibility
4. **3D Visualization** - Depth color-coded skeleton in HUD
   - Layer 0: 3D skeleton overlay
   - Layer 1: CoM trajectory visualization

**Files:** `pose_3d/blazepose_to_h36m.py`, `analysis/physics_engine.py`, `pose_3d/athletepose_extractor.py`, `models/motionagformer/`
**Tests:** `tests/pose_3d/` (11 passing), `tests/analysis/test_physics_engine.py` (18 passing)

**Usage:**
```bash
# Visualize with 3D skeleton
uv run python scripts/visualize_with_skeleton.py video.mp4 --3d --layer 3

# With specific model
uv run python scripts/visualize_with_skeleton.py video.mp4 --3d \
    --model-3d data/models/motionagformer-s-ap3d.pth.tr
```

**Python API:**
```python
from src.pose_3d import blazepose_to_h36m, AthletePose3DExtractor
from src.analysis import PhysicsEngine

# Convert BlazePose to H3.6M
poses_h36m = blazepose_to_h36m(blazepose_poses)  # (N, 33, 2) -> (N, 17, 2)

# Extract 3D poses with MotionAGFormer
extractor = AthletePose3DExtractor(
    model_path="data/models/motionagformer-s-ap3d.pth.tr",
    model_type="motionagformer-s"
)
poses_3d = extractor.extract_sequence(poses_h36m)  # (N, 17, 3)

# Calculate physics
engine = PhysicsEngine(body_mass=60.0)
com = engine.calculate_center_of_mass(poses_3d)
result = engine.fit_jump_trajectory(poses_3d, takeoff_idx, landing_idx)
# result["height"] - accurate jump height from parabolic fit
```

**Model Files:**
- Download location: AthletePose3D repo (Nagoya University)
- Recommended: motionagformer-s-ap3d.pth.tr (59MB)
- Alternative: TCPFormer_ap3d_81.pth.tr (422MB) - Higher accuracy
- Fallback: Biomechanics3DEstimator (no model required)

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

### Phase A: Physics-Informed Improvements ✅ COMPLETE (2026-03-28)

1. **Replace Flight Time with CoM Trajectory** ✅ DONE
   - `compute_jump_height_com()` uses parabolic CoM trajectory
   - Deprecation warning added to hip-only method (60% error)
   - **Files:** `src/metrics.py`

2. **Physics-Informed Pose Validator** ✅ DONE
   - Hampel filter for outlier rejection (MAD-based)
   - Enhanced Kalman filter with 6-state model [x, vx, ax, y, vy, ay]
   - RTS smoother for bidirectional filtering
   - **Files:** `src/pose_filtering.py`

3. **Fix Auto Phase Detection** ✅ DONE
   - Improved CoM-based detection with adaptive sigma thresholds
   - 2-sigma for takeoff, 3-sigma for landing
   - Physical plausibility validation (min 0.3s airtime)
   - **Files:** `src/phase_detector.py`

### Phase B: Multi-Person Tracking ✅ COMPLETE (2026-03-28)

4. **OC-SORT + Pose Biometrics** ✅ DONE
   - PoseTracker class with constant acceleration Kalman filter
   - Anatomical ratio Re-ID (shoulder/torso, femur/tibia, arm_span/height)
   - Solves skeleton switching with identical clothing
   - **Files:** `src/detection/pose_tracker.py`, `src/skeletons.py`
   - **Tests:** 14 tests for tracker, 14 tests for skeleton hierarchy

5. **Integrate Blade Detection into Pipeline** PENDING
   - Add blade state to MetricResult
   - Update rules to use edge information
   - Add edge visualization to HUD
   - **Estimated:** 1-2 hours

6. **Fix DTW Tests** PENDING
   - Update test data for 33 keypoints
   - Fix shape mismatches
   - **Estimated:** 1 hour

### Phase C: Advanced Features

7. **Pose3DM-S for Occlusion** 📝 LOW (Optional)
   - 0.5M params, real-time on RTX 3050 Ti
   - Evaluate vs BlazePose + Physics optimizer
   - **Estimated:** 6-8 hours (includes evaluation)

8. **Improve Segmentation** LOW
   - Trim segment boundaries
   - Remove preparation/recovery
   - **Estimated:** 2-3 hours

### Phase D: Future Enhancements

9. **GCN Element Classifier** 📝 RESEARCH
   - Collect YouTube dataset (BIOES-tagged)
   - Train SAFSAR-style few-shot model
   - Hierarchical rules for complex elements
   - **Estimated:** 1-2 weeks (data collection + training)

10. **Two-Stream Blade Detection** 📝 RESEARCH
    - Train MobileNetV3 on 64x64 patches
    - Merge with kinematic heuristics
    - Target: 80% accuracy
    - **Estimated:** 1-2 weeks (data + training)

---

## Future Enhancements (Research Findings)

📚 **Comprehensive Summary:** See `research/RESEARCH_SUMMARY_2026-03-28.md` for Exa + Gemini Deep Research (41 papers)

### Physics-Based Improvements (Gemini Recommendations)

1. **Physics-Informed Pose Validator** ⚠️ HIGH PRIORITY
   - **Paper:** Leuthold et al. (December 2025) "Physics Informed Human Posture Estimation"
   - **Results:** -10.2% MPJPE, -16.6% joint angle error
   - **Method:** Kalman filter + bone length constraints (post-processing only)
   - **Compute:** Minimal (NumPy + scipy)
   - **Impact:** Eliminates occlusion artifacts without heavy models

2. **Replace Flight Time with CoM Trajectory** ⚠️ CRITICAL
   - **Gemini Finding:** Flight time has 60% error for low jumps!
   - **Solution:** CoM parabolic trajectory (physically accurate)
   - **Formula:** `CoM(t) = (1/M) × Σ(mᵢ × pᵢ)`
   - **Impact:** Eliminates landing pose bias

3. **3D Pose for Occlusion**
   - **Pose3DM-S** (NEW!) - 0.5M params, 2.1G MACs, real-time on RTX 3050 Ti
   - Mamba architecture (State Space Model) - linear O(N) complexity
   - Handles 243-frame temporal windows
   - Alternative: BlazePose + physics optimizer (above)

4. **Physical Parameter Estimation**
   - **User input approach:** Height/weight entered once at registration
   - **Dempster/Zatsiorsky tables:** Segment masses from total weight
   - **SMPLest-X** (TPAMI 2025) - SOTA pose + shape if needed
   - **Sensitivity:** Moment of inertia I depends on r² (height critical!)

5. **Multi-Person Tracking**
   - **OC-SORT** - Handles nonlinear skating trajectories
   - **Pose Biometrics** - Anatomical ratios as unique ID (solves black clothing!)
   - **Alternative:** Deep HM-SORT (80.1 HOTA on SportsMOT)
   - **MOTE** (ICML 2025) - Optical flow + splatting (too compute-heavy)

6. **Hierarchical Element Classification**
   - **FSBench** (CVPR 2025) - 783 videos, 76+ hours, 3D kinematics
   - **YourSkatingCoach** (2024) - BIOES-tagging for precise boundaries
   - **SAFSAR** - Few-shot learning for rare elements
   - **GCN architectures** - SkelFormer, HI-GCN (work on skeleton data)

### Blade Detection Enhancements
- **Current:** BDA Algorithm (79-83% accuracy) ✅ Implemented
- **Gemini Recommendation:** Two-stream hybrid
  - Stream 1: Kinematic heuristics (already done)
  - Stream 2: MobileNetV3-Small on 64x64 patches (critical frames only)
  - Expected: 80% accuracy with minimal compute
- **Future:** Audio-visual fusion (skate sound vs toe pick)

### Data Resources
| Dataset | Content | Link | Status |
|---------|---------|------|--------|
| FS-Jump3D | 3D pose jumps, markerless mocap | github.com/ryota-skating/FS-Jump3D | ✅ Public |
| FSBench | 783 videos, 76h+, 3D+audio+text | arXiv:2504.19514 | ✅ CVPR 2025 |
| YourSkatingCoach | BIOES-tagged elements | arXiv:2410.20427 | ✅ 2024 |
| MMFS | 11,672 clips, 256 categories | Multi-modality | Request |
| AthletePose3D | 1.3M frames, 12 sports | Nagoya University | New! |

---

## GitHub Projects Found (2026-03-28)

### Figure Skating Projects

| Project | Stars | Description | Link | What to Use |
|---------|-------|-------------|------|-------------|
| **JudgeAI-LutzEdge** | 4 | 3D pose + IMU for blade edge judgment | github.com/ryota-skating/JudgeAI-LutzEdge | ✅ Already integrated (BladeEdgeDetector) |
| **Figure-Skating-Quality-Assessment** | 6 | Multi-modal framework (CVPRW 2024) | github.com/ycwfs/Figure-Skating-Quality-Assessment | Multi-modal assessment |
| **awesome.skating.ai** | 33 | Curated list of skating AI projects | github.com/na018/awesome.skating.ai | Project catalog |
| **Figure-Skating-Action-Quality-Assessment** | 4 | Mamba Pyramid (ACM MM 2025) | github.com/ycwfs/Figure-Skating-Action-Quality-Assessment | Action quality models |

### Biomechanics & 3D Pose Projects

| Project | Stars | Description | Link | Integration Potential |
|---------|-------|-------------|------|----------------------|
| **Pose2Sim** ⭐ | 592 | 2D→3D→OpenSim pipeline | github.com/perfanalytics/pose2sim | 🔥 **HIGH** - Full pipeline |
| **HSMR** | 608 | CVPR25 Oral biomechanically accurate 3D | github.com/IsshikiHugh/HSMR | SOTA reconstruction |
| **SKEL** | 335 | SMPL→OpenSim (Siggraph Asia) | github.com/MarilynKeller/SKEL | SMPL integration |
| **Sports2D** | 200 | 2D pose + joint angles | github.com/davidpagnon/Sports2D | Ready joint angle formulas |
| **kineticstoolkit** | 110 | Biomechanics research toolkit | github.com/kineticstoolkit/kineticstoolkit | Analysis utilities |

### Pose2Sim Highlights (v0.10+)

**Features:**
- ✅ Full pipeline: 2D pose → 3D → OpenSim kinematics
- ✅ BlazePose support (same as our project!)
- ✅ Multi-person tracking and association
- ✅ Camera calibration and synchronization
- ✅ Batch processing
- ✅ OpenSim integration for biomechanics
- 🆕 v0.11: Monocular 3D pose estimation (in development)

**Modules to Explore:**
- `Pose2Sim/poseEstimation.py` - 2D pose estimation
- `Pose2Sim/triangulation.py` - 3D reconstruction
- `Pose2Sim/filtering.py` - 3D coordinate filtering
- `Pose2Sim/kinematics.py` - Joint angle computation
- `Pose2Sim/personAssociation.py` - Multi-person tracking

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 0.1 | 2026-03-27 | MVP 85% | Core pipeline working, visualization complete |
| 0.2 | 2026-03-28 | MVP 90% | Blade edge detection (BDA algorithm), research findings |
| 0.3 | 2026-03-28 | MVP 92% | 3D pose infrastructure, PhysicsEngine, keypoint mapping |
| 0.4 | 2026-03-28 | MVP 95% | Phase 14 complete: MotionAGFormer integration, 3D viz |
| 0.5 | 2026-03-28 | MVP 96% | Phase A+B complete: Pose filtering + multi-person tracking |

---

## Reference

### Project Documentation
- Original architecture: `research/RESEARCH.md`
- Visualization research: `research/VISUALIZATION_RESEARCH_PROMPT.md`
- **Physics/Blade detection research:** `research/PHYSICS_DETECTION_RESEARCH.md`
- **🆕 Comprehensive Research Summary:** `research/RESEARCH_SUMMARY_2026-03-28.md`
  - Exa Web Search findings (5 themes)
  - Gemini Deep Research (41 cited papers)
  - Actionable recommendations with priorities

### Key Papers (Gemini Research)
1. **Leuthold et al. (2025)** - Physics Informed Human Posture Estimation
   - arXiv:2512.06783
   - Kalman filter, bone constraints, -10% MPJPE
2. **Pose3DM (2025)** - Bidirectional Mamba-Enhanced 3D HPE
   - MDPI 2504-3110/9/9/603
   - Linear complexity, 0.5M params (Small)
3. **FSBench (CVPR 2025)** - Figure Skating Benchmark
   - arXiv:2504.19514
   - 783 videos, FSBench/FSAnno datasets
4. **YourSkatingCoach (2024)** - Fine-Grained Element Analysis
   - arXiv:2410.20427
   - BIOES-tagging, precise boundaries

### API Documentation
See individual module docstrings
