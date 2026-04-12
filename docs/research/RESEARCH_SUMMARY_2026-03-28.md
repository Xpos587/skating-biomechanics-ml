# Research Summary: Figure Skating ML System Enhancements
**Date:** 2026-03-28
**Sources:** Exa Web Search + Gemini Deep Research

---

## Executive Summary

Two parallel research efforts identified key improvements for the figure skating biomechanics ML system:
1. **Exa MCP Web Search** - Fast literature review (5 themes)
2. **Gemini Deep Research** - Comprehensive Russian technical report (41 cited papers)

**Critical Finding:** Flight Time method for jump height has **60% error** for low jumps - must use CoM parabolic trajectory instead!

---

## Theme 1: Blade Edge Detection

### Current Status
- ✅ **IMPLEMENTED:** BDA Algorithm in `utils/blade_edge_detector.py`
- 19 unit tests passing
- Based on Chen et al. (2025) and Tanaka et al. (2023)

### Research Findings

| Source | Accuracy | Method |
|--------|----------|--------|
| Chen 2025 (Exa) | 79.09% | MediaPipe BDA |
| Tanaka 2023 (Exa) | 83% | 3D pose + IMU |
| Gemini | Recommends 75-80% | Two-stream hybrid |

### Gemini Recommendations

**Two-Stream Hybrid Approach:**
```
Stream 1: Kinematic heuristics (3D keypoints)
├── Foot angle relative to motion
├── Ankle flexion
└── Compute: Near-zero GPU usage

Stream 2: Local CNN patches (critical frames only)
├── 64x64 pixel patches
├── MobileNetV3-Small
└── Triggers: 10 frames before takeoff/landing
```

**Visual signals CNN can learn:**
- Ice chip patterns (sparks from edge braking)
- Blade reflections/shine
- Snow/ice dust patterns

**Future Enhancement:**
- Audio-visual fusion (skating sound vs toe pick scrape)
- Low-latency audio transformers

---

## Theme 2: 2D vs 3D Pose Estimation for Occlusion

### Problem
- BlazePose flips left/right during rotations
- Bone lengths change unnaturally during occlusion
- Arms compressed to chest create joint confusion

### Research Findings

| Model | Architecture | MPJPE | Params | MACs | VRAM Fit |
|-------|--------------|-------|--------|------|----------|
| MotionBERT | Transformer | 39.2mm | 42.5M | 717M | ❌ No |
| MotionAGFormer-L | Transformer+GCN | 38.4mm | 25.0M | 322M | ⚠️ Maybe |
| **Pose3DM-L** | **Mamba (SSM)** | **37.9mm** | **7.43M** | **127M** | ✅ Yes |
| **Pose3DM-S** | **Mamba (SSM)** | **42.1mm** | **0.50M** | **2.1G** | ✅ **Real-time** |
| BlazePose + Physics | CNN + Kalman | -10% MPJPE | <5M | Minimal | ✅ **Mobile** |

### Key Paper: Leuthold et al. (December 2025)
**"Physics Informed Human Posture Estimation Based on 3D Landmarks"**

- **Results:** -10.2% MPJPE, -16.6% joint angle error
- **Method:** Kalman filter iteratively refines bone lengths
- **Cost:** Minimal computation (post-processing only)
- **Code:** Can be implemented with NumPy + scipy

### Gemini Recommendation: Cascade Approach

```
BlazePose (2D keypoints)
    ↓
Physics-Informed Optimizer (Leuthold 2025)
    ├── Kalman filter for bone length consistency
    ├── Line of Sight vectors
    └── Anatomical constraints
    ↓
Optional: Pose3DM-S for complex sequences
    └── 0.5M params, handles 243-frame windows
```

**Angular Momentum Conservation (Future):**
```
L = I·ω = constant during flight
If takeoff ω₀ and I₀ known, reject poses with impossible ω(t)
```

---

## Theme 3: Multi-Person Tracking

### Problem
- Black clothing eliminates color-based Re-ID
- White ice background creates poor contrast
- Skeleton jumps to background skaters

### Research Findings

| Tracker | Method | Occlusion (>5s) | Same Clothing | Compute |
|---------|--------|-----------------|---------------|---------|
| SORT/DeepSORT | IoU + Kalman + Color | ❌ Low | ❌ Low | Low |
| ByteTrack | IoU (all boxes) | ⚠️ Medium | ❌ Low | Low |
| Deep OC-SORT | Nonlinear kinematics + Re-ID | ✅ High | ⚠️ Medium | Medium |
| **MOTE (ICML 2025)** | Optical flow + Splatting | ✅✅ Max | ✅ High | **Extreme** |
| **Recommended** | **OC-SORT + Pose Bio** | **✅ High** | **✅✅ Max** | **Low** |

### Gemini Recommendation: Pose-based Tracking

**Biometric Signature from Skeleton:**
```python
# Compute scale-invariant anatomical ratios
shoulder_width / torso_length
femur_length / tibia_length
arm_span / height

# Use as multi-dimensional feature vector
# Cosine distance for association (with IoU)
```

**Advantages:**
- ✅ Works with identical clothing
- ✅ Reuses already-computed keypoints
- ✅ No additional neural networks
- ✅ Unique per skater

**Future: Gait-based Re-ID**
- ST-GCN (Spatio-Temporal GCN) for skating style embedding
- Unique rhythm of pushes, swing patterns
- Dynamic biometric that can't be faked

---

## Theme 4: Physical Parameter Estimation

### Critical Finding: Flight Time is Wrong!

**Gemini:** "Using FT without considering landing posture leads to jump height overestimation of 18% for medium jumps and up to 60% for low jumps!"

**Why:** Skaters land with bent knees (dorsiflexion), artificially increasing flight time.

### Solution: CoM Trajectory

```python
# Center of Mass = weighted sum of joint positions
CoM(t) = (1/M) × Σ(mᵢ × pᵢ)

# During flight: parabolic trajectory (only gravity)
h_max = CoM_y_peak - CoM_y_takeoff
# This is PHYSICALLY ACCURATE, independent of landing pose
```

### Parameter Estimation Methods

| Parameter | Current | Gemini Recommendation |
|-----------|---------|---------------------|
| Height | ❌ Not estimated | **User input** (one-time registration) |
| Weight | ❌ Not estimated | **User input** + Dempster/Zatsiorsky tables |
| Bone lengths | ❌ Not estimated | **Physics-informed optimizer** (relative ratios) |
| Segment masses | ❌ Not estimated | **Biomechanics tables** from total weight |

### Sensitivity Analysis

| Metric | ±10cm Height Error | ±10kg Weight Error |
|--------|-------------------|-------------------|
| Jump height (CoM) | Moderate (scale) | None |
| Jump height (flight time) | None | None |
| Angular velocity ω | **High** (r² in I) | None |
| Moment of inertia I | **EXTREME** (quadratic) | Linear |
| Landing impact force | Moderate | **Critical** |

### Open Problem: Mass Estimation from Video
**Gemini:** "Mathematical calculation of absolute mass exclusively from monocular video remains an unsolved physics problem."

**Possible approaches:**
- Angular momentum comparison (arms out vs tucked)
- Blade deflection (micro-bends from weight)
- Ice imprint depth analysis

---

## Theme 5: Hierarchical Element Classification

### Current Status
- ✅ Rule-based system implemented
- ❌ Cannot scale to complex combinations
- ❌ Requires manual pattern encoding

### Research Findings: New Datasets!

| Dataset | Content | Size | Features |
|---------|---------|------|----------|
| **FSBench** (CVPR 2025) | Competition programs | 783 videos, 76h+ | 3D kinematics + audio + text |
| **FSAnno** | Fine-grained annotations | - | Temporal segmentation |
| **YourSkatingCoach** (2024) | Micro-element analysis | - | BIOES-tagging |
| **MMFS** | Multi-modality | 11,672 clips | 256 categories, skeleton |
| **FS-Jump3D** | 3D pose jumps | - | Optical markerless mocap |

### BIOES-Tagging (YourSkatingCoach)

**B**egin - **I**nside - **O**ut - **E**nd - **S**ingle

Converts classification to sequential labeling problem:
- Enables precise takeoff/landing frame detection
- Natural fit for temporal action segmentation

### Architecture Recommendations

| Architecture | Input | Compute | Few-Shot | Best For |
|--------------|-------|---------|----------|----------|
| 3D CNN (SlowFast) | RGB pixels | Extreme | Low | Atmosphere analysis |
| Video Transformers | RGB patches | Very High | Medium | Equipment detection |
| **GCN (SkelFormer)** | **3D skeleton** | **Low (ms)** | **High (SAFSAR)** | **Steps/Jumps** |

### Gemini Recommendation: Hierarchical GCN

```
Level 1: Basic Elements (Few-Shot GCN)
├── Three-turn, Bracket, Rocker, Counter
├── Mohawk, Choctaw
├── Forward/Backward crossovers
└── Training: 10-20 examples per class (YouTube)

Level 2: Complex Elements (Rule-based)
├── Salchow = Back crossover → Three-turn → Jump → Landing
├── Lutz = Back crossover → Outside edge → Toe pick → Jump
└── Logic: Temporal sequence of Level 1 classifications
```

**SAFSAR (Semantic-Aware Few-Shot Action Recognition):**
- Direct 3D feature extraction
- Cosine similarity for semantic alignment
- No heavy transformers needed
- >85% accuracy on basic steps

**Open Problem:** Rocker vs Counter differentiation
- Identical upper body dynamics
- Difference: whether skater left previous arc
- **Solution:** Integrate blade state + CoM projection + optical flow

---

## Hardware Constraints Analysis

**Target:** RTX 3050 Ti (4GB VRAM), <100ms per frame

### Compute Budget per Module

| Module | Current | Recommended | VRAM | Time |
|--------|---------|-------------|------|------|
| Detection (YOLOv11n) | ~100MB | ✅ Keep | ~100MB | ~10ms |
| Pose (BlazePose) | ~150MB | ✅ Keep | ~150MB | ~20ms |
| Physics Optimizer | ❌ None | ✅ **Add** | ~0MB | ~5ms |
| 3D Pose (Pose3DM-S) | ❌ None | Optional | ~200MB | ~30ms |
| Blade CNN | ❌ None | Conditional | ~50MB | ~10ms |
| OC-SORT | ❌ None | ✅ **Add** | ~20MB | ~5ms |
| GCN Classifier | ❌ None | Future | ~100MB | ~10ms |
| **TOTAL** | **~250MB** | **~520MB** | **4GB** | **~50-90ms** |

**Conclusion:** All enhancements fit within budget!

---

## Priority Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Physics-Informed Post-Processor**
   - Implement Leuthold 2025 algorithm
   - Kalman filter + bone length constraints
   - Expected: -10% MPJPE
   - Files: `utils/physics_optimizer.py`

2. **Replace Flight Time with CoM**
   - Compute CoM from joint weights
   - Parabolic trajectory fitting
   - Eliminates 60% error for low jumps
   - Files: `analysis/metrics.py` (update)

### Phase 2: Tracking (1-2 days)
3. **OC-SORT + Pose Biometrics**
   - Implement OC-SORT algorithm
   - Add anatomical ratio Re-ID
   - Prevents skeleton switching
   - Files: `detection/pose_tracker.py`

4. **Integrate Blade Detection**
   - Add to AnalysisPipeline
   - Update rules to use edge info
   - Visualize in HUD
   - Files: `pipeline.py`, `utils/visualization.py`

### Phase 3: Advanced (3-5 days)
5. **Pose3DM-S for Occlusion**
   - Evaluate vs BlazePose + Physics
   - Use for complex rotations
   - Files: `pose_2d/pose3dm_extractor.py`

6. **GCN Element Classifier**
   - Collect YouTube dataset (BIOES-tagged)
   - Train SAFSAR-style model
   - Implement hierarchical rules
   - Files: `analysis/element_classifier.py`

### Phase 4: Future Enhancements
7. **Two-Stream Blade Detection**
   - Train MobileNetV3 on 64x64 patches
   - Merge with kinematic heuristics
   - Target: 80% accuracy

8. **Audio-Visual Fusion**
   - Skate sound classification
   - Toe pick scrape detection
   - Multimodal confirmation

---

## Key References

### Papers (Must Read)

1. **Leuthold et al. (2025)** - "Physics Informed Human Posture Estimation"
   - arXiv:2512.06783
   - Kalman filter, bone constraints, -10% MPJPE

2. **Pose3DM** (2025) - "Bidirectional Mamba-Enhanced 3D HPE"
   - MDPI 2504-3110/9/9/603
   - Linear complexity, 0.5M params (Small)

3. **Chen et al. (2025)** - "Automated Blade Type Discrimination"
   - ResearchGate 393670132
   - BDA Algorithm, 79.09% accuracy

4. **FSBench (CVPR 2025)** - "Figure Skating Benchmark"
   - arXiv:2504.19514
   - 783 videos, FSBench/FSAnno datasets

5. **YourSkatingCoach (2024)** - "Fine-Grained Element Analysis"
   - arXiv:2410.20427
   - BIOES-tagging, precise boundaries

### Datasets (Download for Training)

| Dataset | Download | License |
|---------|----------|----------|
| FS-Jump3D | github.com/ryota-skating/FS-Jump3D | Academic |
| MMFS | Available on request | Research |
| FSBench | CVPR 2025 Open Access | Research |
| YourSkatingCoach | arXiv supplementary | Research |

---

## Open Research Questions

1. **Blade Visual Ambiguity**
   - Ice dust, reflections, white-on-white
   - Can multimodal (audio+visual) solve this?

2. **Mass from Video**
   - Impossible without reference force?
   - Blade deflection patterns?

3. **Motion Blur at 4-5 rev/s**
   - Can angular momentum constraints help?
   - Temporal interpolation?

4. **Rocker vs Counter**
   - Requires arc history + blade state
   - Underspecified problem?

---

## Conclusion

The research provides a clear path forward:

**Immediate (high ROI):**
- Physics-informed optimizer (-10% error, minimal compute)
- CoM-based jump height (eliminate 60% error)
- OC-SORT + pose biometrics (solve tracking)

**Short-term (medium ROI):**
- Pose3DM-S for occlusion handling
- Blade detection integration
- GCN basic element classifier

**Long-term (research):**
- Two-stream blade detection with CNN
- Audio-visual fusion
- Angular momentum physics constraints

All enhancements fit within RTX 3050 Ti (4GB VRAM) and <100ms per frame constraint.
