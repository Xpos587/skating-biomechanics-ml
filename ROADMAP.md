# Figure Skating Biomechanics ML - Roadmap

**Status:** MVP 100% complete | Last updated: 2026-04-11

> **This is the SINGLE SOURCE OF TRUTH for project status.** All implementation decisions and priority changes must be reflected here first.

---

## Vision

AI-тренер по фигурному катанию который анализирует видео и даёт рекомендации на русском языке.

**Target Users:** Figure skaters and coaches looking for technical feedback
**Input:** Video recordings (mp4, webm, etc.)
**Output:** Biomechanics analysis + Russian recommendations

---

## 🎉 Major Milestone: 3D-Only Migration Complete (2026-03-29)

**Migration:** BlazePose 2D (33 keypoints) → H3.6M 3D (17 keypoints)

The system has been migrated to use H3.6M 17-keypoint 3D format as the primary pose representation:
- ✅ Type system: H36Key enum with backward compatibility
- ✅ Metrics: All biomechanics calculations updated for 17kp
- ✅ Visualization: Skeleton, velocity, trails all updated
- ✅ Pipeline: 3D-first architecture
- ✅ Tests: 279+ passing

## 🎉 Major Milestone: RTMPose + GPU Pipeline (2026-04-01)

**Migration:** YOLO26-Pose → RTMPose via rtmlib (default), GPU acceleration enabled

- ✅ RTMPoseExtractor: HALPE26 (26kp with foot keypoints), 100% frame coverage
- ✅ 3D-corrected 2D overlay (CorrectiveLens): kinematic constraints + anchor projection
- ✅ Multi-person tracking with anatomical biometric Re-ID
- ✅ GapFiller: linear interpolation + velocity extrapolation
- ✅ CUDA acceleration: 7.1x speedup (5.6s vs 39.4s for 364 frames)
- ✅ Comparison tool (side-by-side, overlay, selectable overlays)
- ✅ Performance: ~12s for 14.5s video (GPU, frame_skip=8)
- ✅ Tests: 279+ passing

## 🎉 Strategic Pivot: OOFSkate-Approach (2026-04-11)

**Decision:** Body kinematics proxy features instead of direct blade edge detection.

Direct blade edge detection from single-camera video is an unsolved problem in open-source:
- Omega (Olympics 2026): 14 specialized rink cameras, closed-source
- JudgeAI-LutzEdge: requires IMU sensors on boots (private data)
- BDA Algorithm (79%): needs reliable foot keypoints — unreliable on ice
- No open-source solution achieves acceptable accuracy from phone video

**Chosen approach (inspired by MIT OOFSkate):**
- Infer element quality from body kinematics, not blade edge
- Proxy features: CoM trajectory, torso lean, approach arc, landing deceleration
- Works with H3.6M 17kp (no foot keypoints needed)
- Validated at 2026 Winter Olympics (MIT → NBC Sports)

**Implications:**
- Blade edge detection → deprioritized (requires specialized hardware)
- Focus → improve body kinematics analysis quality
- Pose backbone → RTMPose confirmed (rtmlib: RTMPose/DWPose/RTMO/RTMW available, none solve edge detection on ice)
- Foot keypoints (HALPE26 extra 9kp) → kept for future use but not critical

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Video     │ -> │  RTMPose     │ -> │  Corrective │ -> │ Normalized   │
│   Input     │    │  (rtmlib)    │    │  Lens (3D)   │    │   Poses       │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  v
┌──────────────────────────────────────────────────────────────────────┐
│                        Analysis Pipeline                            │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Phase Detect │->│  Metrics    │->│  DTW Align  │->│ Recommend │ │
│  │   (Done)     │  │  (Done)      │  │  (Done)     │  │ (Done)    │ │
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

### Phase 2: 3D Pose Estimation ✅ 100%
**Status:** Complete (Updated 2026-03-29)

- [x] **MIGRATION COMPLETE:** BlazePose 33kp → H3.6M 17kp 3D format
- [x] H36Key enum (17 keypoints) with backward compatibility aliases
- [x] BlazePoseExtractor class (2D → 3D conversion)
- [x] AthletePose3DExtractor with MotionAGFormer integration
- [x] 3D normalizer (`pose_3d/normalizer_3d.py`)
- [x] All metrics updated for 17kp format

**Files:** `pose_3d/`, `src/types.py`, `src/normalizer.py`
**Tests:** 263 tests passing, 59% coverage

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

### Phase 6: Phase Detection ✅ 90%
**Status:** Auto-detection working (CoM-based)

- [x] ElementPhase data structure
- [x] Manual phase specification via CLI
- [x] **Auto takeoff detection** (CoM velocity with adaptive sigma)
- [x] **Auto peak detection** (min CoM y during flight)
- [x] **Auto landing detection** (3-sigma threshold)
- [x] Physical plausibility validation (min 0.3s airtime)
- [ ] Phase transition smoothing

**Improvements (2026-03-28):**
- Improved CoM-based detection with adaptive sigma thresholds
- 2-sigma for takeoff, 3-sigma for landing
- Falls back to blade detection if CoM confidence low

**Files:** `analysis/phase_detector.py`
**Tests:** `tests/analysis/test_phase_detector.py`

---

### Phase 7: DTW Motion Alignment ✅ 100%
**Status:** Complete (Updated 2026-03-29)

- [x] DTW implementation (dtw-python)
- [x] Sakoe-Chiba window
- [x] MotionAligner class
- [x] All tests passing (21/21) for H3.6M 17-keypoint format
- [ ] Multi-segment alignment (future enhancement)
- [ ] Alignment quality metrics (future enhancement)

**Files:** `alignment/aligner.py`, `alignment/motion_dtw.py`
**Tests:** `tests/alignment/test_aligner.py`, `tests/alignment/test_motion_dtw.py`

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

- [x] draw_skeleton() - H3.6M 17-keypoint overlay (supports 2D and 3D)
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

- [x] H36MExtractor (H3.6M 17-keypoint format, MediaPipe backend)
- [x] PhysicsEngine (CoM, Moment of Inertia, Angular Momentum)
- [x] Parabolic trajectory fitting for jump height
- [x] AthletePose3DExtractor with MotionAGFormer/TCPFormer integration
- [x] Load AthletePose3D fine-tuned models
- [x] 3D skeleton visualization in main HUD
- [x] CoM trajectory visualization

**Key Features:**
1. **H36MExtractor** - Direct H3.6M 17kp extraction (integrated BlazePose→H3.6M conversion)
2. **PhysicsEngine** - Calculate CoM, I, L from 3D poses
   - Dempster anthropometric tables (segment masses)
   - Parabolic trajectory fit for accurate jump height
   - Solves the 60% flight time error problem!
3. **AthletePose3DExtractor** - Monocular 3D pose estimation
   - 81-frame temporal windows
   - MotionAGFormer-S (59MB) for real-time
   - TCPFormer (422MB) for high accuracy
   - State dict prefix stripping for compatibility
4. **3D Visualization** - Depth color-coded skeleton in HUD
   - Layer 0: 3D skeleton overlay
   - Layer 1: CoM trajectory visualization

**Files:** `src/pose_estimation/h36m_extractor.py`, `src/analysis/physics_engine.py`, `src/pose_3d/athletepose_extractor.py`, `src/models/motionagformer/`, `src/models/tcpformer/`
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
from src.pose_estimation import H36MExtractor
from src.pose_3d import AthletePose3DExtractor
from src.analysis import PhysicsEngine

# Extract H3.6M 17-keypoint poses (YOLO26-Pose backend)
extractor_2d = H36MExtractor(conf_threshold=0.5, output_format="normalized")
poses_2d = extractor_2d.extract_video(video_path)  # (N, 17, 2)

# Extract 3D poses with MotionAGFormer
extractor_3d = AthletePose3DExtractor(
    model_path="data/models/motionagformer-s-ap3d.pth.tr",
    model_type="motionagformer-s"
)
poses_3d = extractor_3d.extract_sequence(poses_2d)  # (N, 17, 3)

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

### LOW Priority
1. **Segment boundaries** - Too broad, includes preparation
   - Impact: Segments not precise
   - Solution: Trim to element core motion

### DEPRIORITIZED
2. **Direct blade edge detection from video** — unsolved in open-source
   - Omega uses 14 rink cameras + proprietary AI (not replicable)
   - IMU-based approaches need sensors on boots
   - Foot keypoints unreliable on ice even with HALPE26
   - See "Strategic Pivot: OOFSkate-Approach" above for chosen direction

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

5. **Integrate Blade Detection into Pipeline** ❌ DEPRIORITIZED
   - Direct blade edge detection from single camera is unsolved in open-source
   - Foot keypoints unreliable on ice even with HALPE26
   - Replaced by OOFSkate proxy-feature approach (Phase I below)
   - Note: blade_edge_detector_3d.py kept for reference but not wired into pipeline

6. **Fix DTW Tests** ✅ DONE
   - All DTW tests passing (H3.6M 17kp format)

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
   - Train on Figure-Skating-Classification dataset (5168 seq, 64 classes, COCO 17kp → H3.6M mapping needed)
   - SAFSAR-style few-shot for rare elements
   - Hierarchical rules for complex combinations
   - **Data available:** MMFS (26198 seq), Figure-Skating-Classification (5168 seq), AthletePose3D (71GB)
   - **Estimated:** 1-2 weeks (data prep + training + evaluation)

10. **Two-Stream Blade Detection** ❌ DEPRIORITIZED
    - Direct blade edge detection from video is unsolved in open-source
    - Replaced by OOFSkate proxy-feature approach (Phase I)
    - Audio-visual fusion (skate sound) remains a possible future direction

### Phase E: Pose Estimation Upgrade (2026-03-31)

📚 **Detailed Research:** See `research/RESEARCH_POSE_TOOLS_2026-03-31.md`

11. **YOLO26-Pose Upgrade** ✅ DONE (2026-04-01)
    - Migrated from YOLOv8/YOLO11 to YOLO26-Pose across all modules
    - NMS-free inference, better occlusion handling
    - All files standardized on `yolo26{size}-pose.pt`

12. **Standardize YOLO Model** ✅ DONE (2026-04-01)
    - Both extractors now use `yolo26{size}-pose.pt`

13. **rtmlib Integration** ✅ DONE (2026-04-01)
    - RTMPoseExtractor: HALPE26 (26kp with foot keypoints)
    - ONNX Runtime backend (fast on CPU, CUDA on GPU)
    - Built-in tracking via PoseTracker
    - Default pose backend (--pose-backend rtmlib)
    - Files: `src/pose_estimation/rtmlib_extractor.py`, `src/pose_estimation/halpe26.py`

14. **Sports2D Angle Integration** ✅ DONE (2026-04-01)
    - Foot angle functions: segment_angle, foot_angle, ankle_dorsiflexion
    - Used in blade edge detection pipeline
    - Files: `src/pose_estimation/halpe26.py`

15. **Monitor Pose3DM-L** 📝 WATCH
    - New SOTA 3D lifter: 37.9mm MPJPE (vs MotionAGFormer 38.4mm)
    - **Code not released yet** (github.com/Reus3237/Pose3DM returns 404)

16. **FS-Jump3D Fine-tuning** 📝 LOW
    - Real skating 3D pose data (4 skaters x 7 jumps, H3.6M 17kp)
    - **Estimated:** 2-3 days (download + fine-tune + evaluate)

### Phase F: Tracking & Robustness (2026-04-01) ✅ DONE

17. **Multi-Person Tracked Extraction** ✅ DONE
    - extract_video_tracked() replaces extract_video()
    - PersonClick + TrackedExtraction types
    - Interactive person selection (--select-person)
    - Track migration via anatomical biometrics
    - Files: `src/pose_estimation/h36m_extractor.py`, `src/types.py`

18. **GapFiller** ✅ DONE
    - 3-tier gap filling: linear interp, velocity extrapolation, split+warn
    - Phase-aware (splits at phase boundaries)
    - Files: `src/utils/gap_filling.py`

19. **Per-Frame Spatial Reference** ✅ DONE
    - Adaptive camera pose estimation every 30 frames
    - Files: `src/detection/spatial_reference.py`

20. **Pipeline Restructure** ✅ DONE
    - _extract_and_track() replaces direct extract_video() calls
    - Files: `src/pipeline.py`, `src/cli.py`, `src/references/reference_builder.py`

### Phase G: 3D-Corrected Overlay (2026-04-01) ✅ DONE

21. **Kinematic Constraints** ✅ DONE
    - Bone length enforcement, joint angle limits
    - 3 iterations, kinematic chain order
    - Files: `src/pose_3d/kinematic_constraints.py`

22. **Anchor-Based Projection** ✅ DONE
    - 3D→2D projection with per-frame torso scale
    - Confidence blending (trust corrected at low confidence)
    - Files: `src/pose_3d/anchor_projection.py`

23. **CorrectiveLens Pipeline** ✅ DONE
    - Orchestrates: 3D lift → constraints → project → blend
    - Falls back to Biomechanics3DEstimator
    - Files: `src/pose_3d/corrective_pipeline.py`

24. **3D Overlay in Visualization** ✅ DONE
    - --3d flag uses CorrectiveLens instead of PIP window
    - PIP window deprecated
    - Files: `scripts/visualize_with_skeleton.py`, `src/visualization/skeleton/drawer.py`

### Phase H: Performance & GPU (2026-04-01) ✅ DONE

25. **CUDA Acceleration** ✅ DONE
    - onnxruntime-gpu with CUDA 12 compat libs on CUDA 13.x system
    - 7.1x speedup (5.6s vs 39.4s for 364 frames)
    - setup_cuda_compat.sh for persistence
    - Files: `scripts/setup_cuda_compat.sh`

26. **Frame Skip & Render Scale** ✅ DONE
    - frame_skip=8: extract every 8th frame, interpolate rest
    - render-scale 0.5/0.33: downscale rendering
    - det_frequency=8: detect person every 8 frames
    - Files: `scripts/visualize_with_skeleton.py`

27. **Comparison Tool** ✅ DONE
    - Side-by-side and overlay modes
    - Selectable overlays (skeleton, angles, timer, axis)
    - Pose caching for instant re-render
    - Files: `src/visualization/comparison.py`

28. **Interactive Person Selection** ✅ DONE
    - --select-person flag with numbered preview
    - --person-click X Y for scripted use
    - Files: `src/cli.py`, `src/pose_estimation/h36m_extractor.py`

### Phase I: OOFSkate-Approach — Body Kinematics Quality Analysis (2026-04-11)

**Strategy:** Infer element quality from body kinematics proxy features instead of direct blade edge detection. Inspired by MIT OOFSkate (deployed at 2026 Winter Olympics with NBC Sports).

**Why:** Direct blade edge detection requires 14 specialized rink cameras (Omega) or IMU sensors (JudgeAI). No open-source solution works from single-camera phone video. Body kinematics approach works with H3.6M 17kp and validated at Olympic level.

29. **Landing Quality Score** 📝 NEXT
    - Smooth deceleration metric (CoM velocity at landing)
    - Hard landing detection (velocity spike > threshold)
    - Landing stability (ankle/knee angle consistency post-landing)
    - Clean edge vs toe assist proxy: sudden velocity change = likely toe pick
    - **Files:** `src/analysis/metrics.py`
    - **Estimated:** 2-3 hours

30. **Torso Lean & Approach Arc** 📝 PLANNED
    - Torso lean angle relative to vertical (spine→neck vector)
    - Approach trajectory curvature (CoM x-z path)
    - Proxy for edge type: lutz (lean back, long outside arc) vs flip (lean forward, inside arc)
    - **Files:** `src/analysis/metrics.py`, `src/analysis/physics_engine.py`
    - **Estimated:** 3-4 hours

31. **Element Quality Scoring (GOE proxy)** 📝 PLANNED
    - Numerical quality score per element (inspired by OOFSkate's GOE estimation)
    - Based on: height, rotation, landing quality, airtime, torso control
    - Comparison against reference database averages
    - Russian text output: "Оценка качества: +1.2 GOE"
    - **Files:** `src/analysis/metrics.py`, `src/analysis/recommender.py`
    - **Estimated:** 4-6 hours

32. **Reference Database Expansion** 📝 PLANNED
    - Build reference library from competition videos (YouTube)
    - Per-element average metrics from elite skaters
    - Store in `data/references/` as .npz with metadata
    - Enable automatic comparison: "Your jump height is 85% of elite average"
    - **Data available:** AthletePose3D (71GB, 5154 videos), MMFS, Figure-Skating-Classification
    - **Estimated:** 1-2 days (extraction + normalization + storage)

33. **GCN Element Classifier** 📝 RESEARCH
    - See Phase D item 9 above
    - Training data ready: Figure-Skating-Classification (5168 seq, 64 classes)
    - COCO 17kp → H3.6M 17kp mapping required (different skeleton definitions)
    - **Estimated:** 1-2 weeks

---

## Kinovea-like Comparison Tool (2026-03-31) ✅ DONE

**Status:** Fully implemented, GPU-accelerated

17. **Dual-Video Comparison** ✅ DONE
    - `src/visualization/comparison.py` — ComparisonRenderer module
    - Side-by-side and overlay modes
    - Configurable overlays: skeleton, axis, angles, timer
    - Pose caching for instant re-render

18. **Vertical Axis Layer** ✅ DONE
    - `src/visualization/layers/vertical_axis_layer.py`

19. **Joint Angle Layer** ✅ DONE
    - `src/visualization/layers/joint_angle_layer.py`

20. **Timer Layer** ✅ DONE
    - `src/visualization/layers/timer_layer.py`

21. **Pose Validation** ✅ DONE
    - Gap-based filling, spread threshold

---

## Future Enhancements (Research Findings)

📚 **Comprehensive Summary:** See `research/RESEARCH_SUMMARY_2026-03-28.md`
📚 **Pose Tools Research:** See `research/RESEARCH_POSE_TOOLS_2026-03-31.md` for Exa + Gemini Deep Research (41 papers)

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

### Blade Detection Strategy (Updated 2026-04-11)
- **Current:** BDA Algorithm (79-83% accuracy) ✅ Implemented but not wired into pipeline
- **Problem:** Foot keypoints unreliable on ice; no open-source solution for single-camera video
- **Omega (2026 Olympics):** 14 rink cameras, proprietary — not replicable
- **JudgeAI-LutzEdge:** IMU sensors required — private data
- **Chosen direction:** OOFSkate proxy features (body kinematics) — Phase I above
- **Future possibility:** Audio-visual fusion (skate sound classification) for mobile app

### Data Resources
| Dataset | Content | Link | Status |
|---------|---------|------|--------|
| AthletePose3D | 1.3M frames, 12 sports, 71GB | github.com/calvinyeungck/AthletePose3D | ✅ Downloaded |
| MMFS | 26198 skeleton seq, 256 categories, 1.7GB | github.com/dingyn-Reno/MMFS | ✅ Downloaded |
| Figure-Skating-Classification | 5168 seq, 64 classes, 340MB | huggingface.co/datasets/Mercity/... | ✅ Downloaded |
| FS-Jump3D | Subset of AthletePose3D | github.com/ryota-skating/FS-Jump3D | ❌ Duplicate (skip) |
| FSBench | 783 videos, 76h+, 3D+audio+text | arXiv:2504.19514 | 📝 Temporarily closed |
| YourSkatingCoach | BIOES-tagged elements | arXiv:2410.20427 | 📝 Supplementary only |
| FineFS | 1167 samples, scores+boundaries | github.com/yanliji/FineFS-dataset | 📝 Google Drive link dead |

**See `data/DATASETS.md` for detailed registry and relationships.**

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
| 0.6 | 2026-04-01 | MVP 100% | RTMPose + GPU pipeline, Nike design system, SaaS frontend |
| 0.7 | 2026-04-11 | Strategic pivot | OOFSkate approach (proxy features over blade edge), datasets collected |

---

## Reference

### Project Documentation
- Research memory bank: `research/RESEARCH.md`
- Comprehensive summary: `research/RESEARCH_SUMMARY_2026-03-28.md`
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

### Industry References
5. **OOFSkate (MIT, 2026)** - Optical tracking system for figure skating
   - Jerry Lu MFin '24, MIT Sports Lab
   - Mobile app: video → physics metrics → GOE score estimate
   - Deployed with NBC Sports at 2026 Winter Olympics
   - Approach: body kinematics (no direct blade edge detection)
   - https://news.mit.edu/2026/3-questions-ai-olympic-skaters-0213
6. **Omega (2026 Olympics)** - Computer vision blade angle detection
   - 14 specialized cameras around rink
   - Jump height, rotation, blade angle — real-time
   - Closed-source, proprietary
   - Blade detection ready but not yet used for judging

### API Documentation
See individual module docstrings
