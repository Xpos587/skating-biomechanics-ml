# CLAUDE.md

> **⚠️ PROJECT ROADMAP:** See @ROADMAP.md for the SINGLE SOURCE OF TRUTH on implementation status, phases, and blockers.
> **📚 RESEARCH SUMMARY:** See @research/RESEARCH_SUMMARY_2026-03-28.md for comprehensive Exa + Gemini findings (41 papers)

> This file provides project context, development conventions, and workflow guidelines.

---

## Project Overview

ML-based personal AI coach for figure skating using computer vision. Analyzes skating technique from video and provides specific recommendations in Russian.

**Vision:** AI-тренер по фигурному катанию который анализирует видео и даёт рекомендации на русском языке.

**Target Users:** Figure skaters and coaches looking for technical feedback

## Tech Stack (MVP)

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| **Language**        | Python 3.11+                             |
| **Package Manager** | `uv`                                     |
| **Detection**       | YOLOv11n (Ultralytics)                   |
| **2D Pose**         | MediaPipe BlazePose (33 keypoints)       |
| **Normalization**   | Root-centering + scale normalization     |
| **Alignment**       | DTW (dtw-python) with Sakoe-Chiba window |
| **Analysis**        | Custom biomechanics metrics              |
| **Recommendations** | Rule-based engine (Russian output)       |
| **Testing**         | Pytest + pytest-cov                      |

## MVP Architecture (2D-only)

```
Video Input → YOLOv11n (detect) → BlazePose (2D keypoints) → Normalization
    ↓
Phase Detection → Biomechanics Metrics → DTW (vs reference)
    ↓
Rule-based Recommender → Text Report (Russian)
```

**Key Decision:** MVP uses 2D normalized poses instead of 3D lifting. This simplifies the pipeline while providing sufficient information for basic biomechanics analysis. 3D lifting can be added later as an enhancement.

---

## 🔄 Git Workflow (CRITICAL)

### Commit Discipline

**MANDATORY:** Commit frequently after completing logical units of work. Never leave uncommitted changes overnight.

```bash
# Check status before starting work
git status

# After completing a feature/unit:
git add <files>
git commit -m "<type>: <description>"

# Logical commit types:
feat:     New feature
fix:      Bug fix
docs:     Documentation changes
refactor: Code refactoring (no behavior change)
test:     Adding/updating tests
chore:    Maintenance, tooling, dependencies
```

### Commit Message Format

```
<type>: <short description>

<detailed explanation if needed>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Examples:**
```
feat(blade-detection): add BDA algorithm for skate blade edge detection

- BladeType enum (INSIDE, OUTSIDE, FLAT, TOE_PICK, UNKNOWN)
- BladeEdgeDetector class with 4 angular thresholds
- 19 unit tests (all passing)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

### Branch Strategy

- `master` - Main development branch
- Feature branches for larger work (optional for this project)
- Always pull before starting work
- Push after significant commits

### Pre-Commit Checklist

Before committing:
1. [ ] Tests pass: `uv run pytest tests/ -v -m "not slow"`
2. [ ] Code formatted: `uv run ruff format .`
3. [ ] No lint errors: `uv run ruff check .`
4. [ ] Type check passes: `uv run mypy src/`

---

## Project Structure

```
src/skating_biomechanics_ml/
├── types.py              # Shared data types (BKey, FrameKeypoints, BladeType, etc.)
├── pipeline.py           # Main AnalysisPipeline orchestrator
├── cli.py                # argparse CLI (analyze, build-ref, segment commands)
├── detection/
│   └── person_detector.py    # YOLOv11n wrapper
├── pose_2d/
│   ├── blazepose_extractor.py  # BlazePose wrapper (33 keypoints)
│   ├── pose_extractor.py       # Abstract pose extractor interface
│   └── normalizer.py            # Root-centering, scale normalization
├── analysis/
│   ├── metrics.py             # BiomechanicsAnalyzer (airtime, angles, etc.)
│   ├── phase_detector.py      # Auto-detect takeoff/peak/landing (⚠️ 50% working)
│   ├── recommender.py         # Rule-based recommendation engine
│   └── rules/
│       ├── jump_rules.py      # Rules for all jump types
│       └── three_turn_rules.py # Rules for three_turn
├── alignment/
│   ├── aligner.py             # DTW motion alignment
│   └── motion_dtw.py          # DTW utilities
├── references/
│   ├── element_defs.py        # Element definitions & ideal metrics
│   ├── reference_builder.py   # Build reference from expert video
│   └── reference_store.py     # Store/load .npz reference files
├── segmentation/
│   └── element_segmenter.py   # Automatic motion segmentation
└── utils/
    ├── blade_edge_detector.py # BDA algorithm for blade edge detection ✨ NEW!
    ├── video.py               # cv2 video utilities
    ├── geometry.py            # Angles, distances, smoothing
    ├── smoothing.py           # One-Euro Filter for pose smoothing
    ├── visualization.py       # Skeleton, kinematics, HUD drawing
    └── subtitles.py           # VTT subtitle parser for coach commentary

scripts/
├── check_all.py               # Run all quality checks
├── build_references.py        # CLI to build references from video
├── download_models.py         # Download YOLOv11n weights
├── visualize_with_skeleton.py # Enhanced debug visualization with layered HUD
├── visualize_segmentation.py  # Visualize automatic segmentation
└── organize_dataset.py        # Dataset organization utilities

tests/
├── conftest.py            # Shared fixtures
├── test_types.py          # Type tests
├── test_pipeline.py       # Integration tests
├── detection/             # PersonDetector tests
├── pose_2d/               # BlazePose tests
├── analysis/              # Metrics, phase detector, recommender tests
├── alignment/             # DTW aligner tests
├── segmentation/          # Element segmenter tests
└── utils/                 # Utility tests (blade, geometry, smoothing, viz)

research/
├── RESEARCH.md                        # Original architecture research
├── VISUALIZATION_RESEARCH_PROMPT.md   # Visualization design research
├── PHYSICS_DETECTION_RESEARCH.md      # Exa web search research prompt
└── RESEARCH_SUMMARY_2026-03-28.md     # 📚 Comprehensive Exa + Gemini findings (41 papers)

data/
└── references/            # Expert reference .npz files (not in git)
    ├── three_turn/
    ├── waltz_jump/
    ├── toe_loop/
    └── flip/
```

---

## Development Workflow

### Quality Checks

```bash
# Run all checks
uv run python scripts/check_all.py

# Individual checks
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy src/             # Type check
uv run vulture src/ tests/   # Dead code
uv run pytest tests/ -v -m "not slow"  # Tests (exclude slow ML tests)
```

### CLI Usage

```bash
# Analyze a skating video
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element three_turn

# Build reference from expert video
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 --element waltz_jump \
    --takeoff 1.0 --peak 1.2 --landing 1.4

# Segment video automatically
uv run python -m skating_biomechanics_ml.cli segment video.mp4

# With reference directory
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element waltz_jump \
    --reference-dir data/references --output report.txt

# Visualize with debug overlay
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3 --output video_debug.mp4
```

### Supported Elements

| Element      | Type | Key Metrics                                           |
| ------------ | ---- | ----------------------------------------------------- |
| `three_turn` | Step | trunk_lean, edge_change_smoothness, knee_angle        |
| `waltz_jump` | Jump | airtime, max_height, landing_knee_angle, arm_position |
| `toe_loop`   | Jump | airtime, rotation_speed, toe_pick_timing              |
| `flip`       | Jump | airtime, pick_quality, air_position                   |
| `salchow`    | Jump | airtime, rotation_speed, edge_quality                 |
| `loop`       | Jump | airtime, rotation_speed, height                       |
| `lutz`       | Jump | airtime, toe_pick_quality, rotation                  |
| `axel`       | Jump | airtime, height, rotation                            |

---

## Key Concepts

### Data Types

- **FrameKeypoints**: `(N, 33, 3)` — x, y, confidence from BlazePose (pixel coords)
- **NormalizedPose**: `(N, 33, 2)` — x, y in [0,1] normalized coordinates
- **PixelPose**: `(N, 33, 2)` — x, y in pixel coordinates
- **BladeType**: Enum (INSIDE, OUTSIDE, FLAT, TOE_PICK, UNKNOWN)
- **ElementPhase**: start, takeoff, peak, landing, end frame indices
- **MetricResult**: name, value, unit, is_good, reference_range
- **BladeState**: blade_type, foot_angle, ankle_angle, vertical_accel, confidence

### Coordinate System Convention (CRITICAL)

**Always clarify coordinate system in variable names and function signatures:**

```python
# Naming convention
poses_norm = ...  # Normalized [0,1]
poses_px = ...    # Pixel coordinates

# Use validation from types.py
from skating_biomechanics_ml.types import assert_pose_format
assert_pose_format(poses, "normalized", context="my_function")

# Convert between formats
from skating_biomechanics_ml.types import normalize_pixel_poses, pixelize_normalized_poses
poses_norm = normalize_pixel_poses(poses_px, width=1920, height=1080)
poses_px = pixelize_normalized_poses(poses_norm, width=1920, height=1080)
```

**Visualization functions expect NORMALIZED coordinates:**

- `draw_velocity_vectors()` → normalized [0,1]
- `draw_trails()` → normalized [0,1]
- `draw_blade_indicator_hud()` → uses BladeState from BladeEdgeDetector (recommended)
- `draw_skeleton()` → both (handles conversion internally)

**Common bugs to avoid:**

1. Passing pixel coords to functions expecting normalized → wrong calculations
2. Passing normalized coords to functions expecting pixels → skeleton misaligned
3. Smoothing in wrong coordinate space → inconsistent results

### Normalization

1. **Root-centering**: mid-hip → origin (0, 0)
2. **Scale normalization**: spine length → 0.4 (typical adult athlete)

### Biomechanics Metrics

- **Airtime**: `(landing - takeoff) / fps` seconds
- **Jump height**: `hip_y[landing] - min(hip_y[takeoff:landing])` ⚠️ **WARNING: Use CoM trajectory instead!**
- **Knee angle**: Angle at hip-knee-ankle joint
- **Arm position**: Distance from wrist to shoulder (0 = close, 1 = extended)
- **Edge indicator**: +1 (inside edge), -1 (outside edge), 0 (flat)

### Blade Edge Detection (NEW!)

**BDA Algorithm** (Blade Discrimination Algorithm) uses 4 angular thresholds:
- Strong inside: angle < -20°
- Weak inside: -20° to -10°
- Flat: -10° to 10°
- Weak outside: 10° to 20°
- Strong outside: angle > 20°

**Toe pick detection:** Vertical acceleration spike during takeoff

```python
from skating_biomechanics_ml.utils import BladeEdgeDetector

detector = BladeEdgeDetector(
    inside_threshold=-15.0,
    outside_threshold=15.0,
    smoothing_window=3
)
states = detector.detect_sequence(poses, fps=30.0, foot="left")
```

### Visualization System

Enhanced debug visualization with layered HUD architecture:

**Layers:**

- **Layer 0 (Raw)**: Skeleton only
- **Layer 1 (Kinematics)**: + velocity vectors + motion trails
- **Layer 2 (Technical)**: + edge indicators + joint angles
- **Layer 3 (Coaching)**: + subtitles + full HUD

**Usage:**

```bash
# Generate debug visualization
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3 --output video_debug.mp4

# With pre-computed poses
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 1 --poses poses.npz
```

**Color scheme (skeleton):**

- Left side (arm/leg): Blue
- Right side (arm/leg): Red
- Center (torso/head): Green
- Joints: White

**Smoothing:**

- One-Euro Filter applied in normalized coordinate space
- Reduces jitter by ~30% while preserving fast motions

---

## 📚 Research Insights (2026-03-28)

See @research/RESEARCH_SUMMARY_2026-03-28.md for comprehensive findings from Exa + Gemini Deep Research (41 cited papers).

### Critical Findings

1. **Flight Time Method Has 60% Error** for low jumps! Use CoM parabolic trajectory instead.
2. **Physics-Informed Optimizer** (Leuthold 2025) reduces MPJPE by 10.2% using Kalman filter + bone constraints.
3. **Pose3DM-S** (0.5M params) enables real-time 3D pose on RTX 3050 Ti for occlusion handling.
4. **OC-SORT + Pose Biometrics** solves tracking with identical black clothing.
5. **FSBench** (CVPR 2025) provides 783 videos with 3D kinematics for training.

### Future Enhancements (Priority Order)

**Phase A: Critical (1-2 days)**
1. Replace flight time with CoM trajectory
2. Physics-informed pose validator (Kalman + bone constraints)
3. Fix auto phase detection

**Phase B: Tracking (4-5 days)**
4. OC-SORT + pose biometrics for multi-person tracking
5. Integrate blade detection into analysis pipeline

**Phase C: Advanced (1-2 weeks)**
6. Pose3DM-S for complex occlusions
7. GCN element classifier with BIOES-tagging

---

## Environment

- **OS**: Artix Linux (Ryzen 7 5800H / RTX 3050 Ti 4GB VRAM)
- **Python**: 3.11+ via `uv`
- **VRAM Budget**: <200MB for current pipeline, ~520MB with all enhancements

---

## Implementation Status

✅ **Overall: MVP ~90% complete**

**See @ROADMAP.md for detailed phase-by-phase status**

**Complete (100%):**
- Phase 0: Foundation (types, utils)
- Phase 1: Person Detection (YOLOv11n)
- Phase 2: Pose Estimation (BlazePose 33kp)
- Phase 3: Normalization (root-centering + scale)
- Phase 4: Smoothing (One-Euro Filter, 29% jitter reduction)
- Phase 5: Metrics (airtime, height, angles, edge)
- Phase 8: Recommender (rule-based, Russian output)
- Phase 9: Reference System (save/load .npz)
- Phase 11: Visualization (layered HUD, skeleton, kinematics)
- Phase 12: CLI & Pipeline (analyze, build-ref, segment)
- **Phase 13: Blade Edge Detection** ✨ (BDA algorithm, 19 tests passing)

**Partial (50-90%):**
- Phase 6: Phase Detection ⚠️ 50% - MANUAL ONLY, auto-detection NOT working
- Phase 7: DTW Alignment ⚠️ 70% - code exists, tests failing (17 vs 33 keypoints)
- Phase 10: Segmentation ✅ 90% - working, but boundaries too broad

---

## Recent Improvements (2026-03)

### Blade Edge Detection (Phase 13) ✨ NEW!

- Implemented BDA Algorithm based on Chen et al. (2025) and Tanaka et al. (2023)
- BladeType enum (INSIDE, OUTSIDE, FLAT, TOE_PICK, UNKNOWN)
- BladeEdgeDetector with foot angle, ankle angle, vertical acceleration
- Temporal smoothing via majority voting
- 19 unit tests (all passing)
- Takeoff/landing detection from blade state sequence

### Coordinate System Architecture

- Added explicit `PixelPose` and `NormalizedPose` type aliases
- Runtime validation with `assert_pose_format()` catches coordinate bugs early
- Helper functions: `normalize_pixel_poses()`, `pixelize_normalized_poses()`
- Documented convention in CLAUDE.md to prevent future confusion

### Enhanced Visualization

- Frame-perfect synchronization between poses and video frames
- One-Euro Filter smoothing in normalized coordinate space (~30% jitter reduction)
- Support for VTT subtitles with Russian/Cyrillic text (via Pillow)
- Layered HUD system for focused debugging

---

## Known Issues & Workarounds

### BlazePose Frame Skipping

BlazePose may skip frames where person detection confidence is low. This causes:

- Fewer extracted poses than video frames
- Potential synchronization issues

**Solution:** The visualization script tracks frame indices and only draws skeleton when pose data exists for that frame.

### Coordinate System Confusion

Historically, mixing pixel and normalized coordinates caused visualization bugs.

**Solution:** Always use variable name suffixes (`_px`, `_norm`) and validate with `assert_pose_format()`.

### Flight Time Jump Height Error

**CRITICAL:** Flight time method overestimates low jumps by up to 60%!

**Solution:** Use parabolic trajectory of Center of Mass (CoM) instead. See research summary for details.

---

## References

- **Project roadmap:** @ROADMAP.md (SINGLE SOURCE OF TRUTH)
- **Research summary:** @research/RESEARCH_SUMMARY_2026-03-28.md (Exa + Gemini, 41 papers)
- **Original architecture:** @research/RESEARCH.md
- **Visualization research:** @research/VISUALIZATION_RESEARCH_PROMPT.md
- **BlazePose keypoints:** <https://google.github.io/mediapipe/solutions/pose.html>
- **DTW in Python:** <https://dynamictimewarping.github.io/>

---

## Quick Reference

### Most Used Commands

```bash
# Quality check
uv run python scripts/check_all.py

# Analyze video
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element waltz_jump

# Visualize with skeleton
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3

# Run tests
uv run pytest tests/ -v -m "not slow"
```

### Key Files to Know

- `types.py` - All data types and validation
- `pipeline.py` - Main orchestrator
- `cli.py` - Command-line interface
- `utils/blade_edge_detector.py` - Blade edge detection (NEW!)
- `utils/visualization.py` - All visualization functions
- `ROADMAP.md` - Project status and next steps
