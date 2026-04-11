# AthletePose3D Integration Plan

**Date:** 2026-03-28
**Status:** ✅ COMPLETE (Implemented 2026-03-28)

> **Note:** This document describes the original integration plan. The implementation is now complete and uses H3.6M 17-keypoint format directly via H36MExtractor. The `blazepose_to_h36m` conversion function mentioned below is deprecated (see `src/pose_estimation/h36m_extractor.py`).

---

## Overview

Integrate AthletePose3D fine-tuned models for monocular 3D pose estimation in figure skating videos.

**Key Benefits:**
- Pre-trained on 12 sports including figure skating
- MPJPE: 214mm → 65mm (-69% with fine-tuning)
- Real-time inference with MotionAgFormer-S (59MB model)
- Temporal modeling (81-frame windows)

---

## Data Format Analysis

### Input/Output Structure

```python
# Input: 2D pose sequence
data_input: (81, 17, 3)  # frames × keypoints × (x, y, confidence)

# Output: 3D pose sequence
data_label: (81, 17, 3)  # frames × keypoints × (x, y, z)
```

### H3.6M 17 Keypoint Format

| Index | Name          | BlazePose Equivalent |
|-------|---------------|---------------------|
| 0     | hip (center)  | mid_hip (0.5 * (L_HIP + R_HIP)) |
| 1     | rhip          | R_HIP (24) |
| 2     | rknee         | R_KNEE (26) |
| 3     | rfoot         | R_ANKLE (28) |
| 4     | lhip          | L_HIP (23) |
| 5     | lknee         | L_KNEE (25) |
| 6     | lfoot         | L_ANKLE (27) |
| 7     | spine         | mid_shoulder * 0.5 + mid_hip * 0.5 |
| 8     | thorax        | mid_shoulder |
| 9     | neck          | NOSE (0) approximation |
| 10    | head          | NOSE (0) |
| 11    | lshoulder     | L_SHOULDER (11) |
| 12    | lelbow        | L_ELBOW (13) |
| 13    | lwrist        | L_WRIST (15) |
| 14    | rshoulder     | R_SHOULDER (12) |
| 15    | relbow        | R_ELBOW (14) |
| 16    | rwrist        | R_WRIST (16) |

### Coordinate System

- **2D Input**: Normalized [0,1] with some negative margins
- **3D Output**: X (horizontal), Y (vertical), Z (depth)
- **Z=0**: Hip plane reference
- **Positive Z**: Toward camera
- **Negative Z**: Away from camera

---

## Model Architecture

### Available Models

| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| MotionAgFormer-S | 59MB | ~200MB | Fast | Best for RTX 3050 Ti |
| TCPFormer | 422MB | ~500MB | Medium | Higher accuracy |
| Moganet-B (2D) | 570MB | ~300MB | Fast | 2D detection |

**Recommended:** MotionAgFormer-S for real-time performance.

### Temporal Window

- **81 frames** at 30fps = 2.7 seconds of context
- Model processes entire window at once
- Output frame is typically the middle frame (frame 40)

---

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │ -> │  BlazePose 33kp  │ -> │  Keypoint Map   │
│   (mp4)         │    │  (2D detection)  │    │  33 → 17        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         v
┌─────────────────────────────────────────────────────────────────┐
│                    AthletePose3D Pipeline                      │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐ │
│  │ Temporal     │->│ MotionAgFormer│->│ 3D Pose (17kp)     │ │
│  │ Buffer (81)  │  │   -S Model    │  │ (x, y, z)          │ │
│  └──────────────┘  └───────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                                          v
┌─────────────────────────────────────────────────────────────────┐
│                    Physics Engine                               │
│  - Center of Mass (CoM) trajectory                             │
│  - Angular momentum L = I × ω                                   │
│  - Moment of inertia I = Σ(m × r²)                              │
│  - Jump height from parabolic fit                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Keypoint Mapping (1-2 hours)

**File:** `src/pose_2d/blazepose_to_h36m.py`

```python
def blazepose_to_h36m(blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints.

    Args:
        blazepose_pose: (33, 2) or (33, 3) array

    Returns:
        h36m_pose: (17, 2) or (17, 3) array
    """
```

### Phase 2: Model Loader (1 hour)

**File:** `src/pose_3d/athletepose_extractor.py`

```python
class AthletePose3DExtractor:
    """Monocular 3D pose estimation using AthletePose3D models."""

    def __init__(self, model_path: str, model_type: str = "motionagformer-s"):
        self.model = load_model(model_path)
        self.temporal_buffer = deque(maxlen=81)

    def extract_frame(self, pose_2d: np.ndarray) -> np.ndarray | None:
        """Extract 3D pose from single frame (with temporal context)."""

    def extract_sequence(self, poses_2d: np.ndarray) -> np.ndarray:
        """Extract 3D poses from sequence."""
```

### Phase 3: Physics Engine (2-3 hours)

**File:** `src/analysis/physics_engine.py`

```python
class PhysicsEngine:
    """Physics calculations from 3D poses."""

    def calculate_center_of_mass(self, pose_3d: np.ndarray) -> np.ndarray:
        """Calculate CoM using Dempster anthropometric tables."""

    def calculate_moment_of_inertia(self, pose_3d: np.ndarray) -> float:
        """Calculate moment of inertia I = Σ(m × r²)."""

    def calculate_angular_momentum(self, pose_3d: np.ndarray,
                                    angular_velocity: float) -> float:
        """Calculate L = I × ω."""

    def fit_jump_trajectory(self, poses_3d: np.ndarray,
                            takeoff: int, landing: int) -> dict:
        """Fit parabolic trajectory to CoM during flight."""
```

### Phase 4: Pipeline Integration (1 hour)

**File:** `src/pipeline.py` (modify `AnalysisPipeline`)

```python
# Add 3D pose extraction stage
poses_3d = self._get_3d_extractor().extract_sequence(poses_2d)

# Add physics metrics
physics_metrics = self._get_physics_engine().analyze(poses_3d, phases)
```

---

## File Structure

```
src/
├── pose_3d/                          # NEW
│   ├── __init__.py
│   ├── athletepose_extractor.py      # Main 3D extractor
│   └── blazepose_to_h36m.py          # Keypoint mapping
├── analysis/
│   └── physics_engine.py             # NEW: Physics calculations
├── models/                           # NEW
│   └── athletepose3d/                # Model checkpoints
│       ├── motionagformer-s-ap3d.pth.tr
│       └── config.json
└── types.py                          # Add 3D pose types

tests/
├── pose_3d/                          # NEW
│   ├── test_athletepose_extractor.py
│   └── test_blazepose_to_h36m.py
└── analysis/
    └── test_physics_engine.py        # NEW
```

---

## Dependencies to Add

```toml
[tool.poetry.dependencies]
torch = ">=2.0.0"          # Already in project
scipy = ">=1.10.0"         # Already in project
# No new dependencies needed!
```

---

## Verification Tests

```bash
# Unit tests
uv run pytest tests/pose_3d/ -v
uv run pytest tests/analysis/test_physics_engine.py -v

# Integration test
uv run python -m src.cli analyze test_video.mp4 \
    --element waltz_jump --use-3d

# Visual verification
uv run python scripts/visualize_3d_trajectory.py test_video.mp4 \
    --output 3d_trajectory.mp4
```

---

## Performance Budget

| Component | VRAM | Time per frame |
|-----------|------|----------------|
| BlazePose | ~150MB | ~20ms |
| AthletePose3D-S | ~200MB | ~30ms |
| Physics Engine | ~0MB | ~5ms |
| **Total** | **~350MB** | **~55ms** |

**Result:** Fits within RTX 3050 Ti (4GB VRAM) and <100ms constraint ✅

---

## Next Steps

1. ✅ Data format analyzed
2. ✅ H36MExtractor implemented (YOLOv11-Pose backend, direct H3.6M output)
3. ✅ AthletePose3DExtractor implemented
4. ✅ PhysicsEngine implemented
5. ✅ AnalysisPipeline updated

**Implementation Status:** Complete - See Phase 14 in ROADMAP.md
6. ⏳ Add tests
7. ⏳ Verify with real video

---

## References

- **AthletePose3D Paper:** "AthletePose3D: A Large-Scale 3D Sports Pose Dataset"
- **H3.6M Dataset:** Human3.6M 17-keypoint format
- **MotionAgFormer:** Transformer-based 3D pose estimation
- **Camera Parameters:** /home/michael/Downloads/cam_param.json
- **Model Weights:** /home/michael/Downloads/model_params-*.zip
