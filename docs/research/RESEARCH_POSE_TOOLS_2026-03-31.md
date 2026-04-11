# Research: Pose Estimation Tools for Figure Skating (2026-03-31)

**Sources:** Web search + GitHub repos + documentation
**Context:** Evaluation of Sports2D, RTMPose, YOLO11/26 Pose, Kinovea alternatives for figure skating biomechanics analysis

---

## Executive Summary

Research conducted to evaluate Grok's recommendations on Sports2D, YOLO11/26, and other pose estimation tools. Key findings:

1. **Sports2D** — real tool, uses rtmlib (ONNX-based, no mmpose needed), but outputs HALPE26 format (26kp), not H3.6M (17kp). Has ready-made joint angles. Heavy Pose2Sim dependency.
2. **YOLO26-Pose** — released January 2026, best Ultralytics pose model (57.2 mAP50-95 for nano). NMS-free, 43% faster CPU. Worth upgrading from YOLOv8.
3. **Pose3DM-L** — new SOTA for 3D lifting (37.9mm vs our MotionAGFormer 38.4mm), 60% less compute, but **code not released yet** (repo 404).
4. **Kinovea alternatives** — Pose2Sim (592 stars) is the closest "Kinovea with AI", full biomechanics pipeline. Sports2D (202 stars) has ready angles.
5. **MediaPipe** — no sports improvements, still 33kp, limb-flipping issue persists. Our migration was correct.

---

## 1. Sports2D + RTMPose

### Sports2D (github.com/davidpagnon/Sports2D)

- **Version:** v0.8.28 (2026-03-06), 202 stars, JOSS published
- **What it does:** 2D joint positions + joint angles + segment angles from video
- **Pipeline:** rtmlib (RTMPose ONNX) -> pixel-to-meter -> angle computation -> optional OpenSim IK
- **Limitation:** 2D only, only valid in 2D plane (sagittal/frontal). Not for 3D biomechanics.

### Installation

- `pip install sports2d` — depends on Pose2Sim (heavy, many scientific deps)
- Pose estimation uses **rtmlib** directly (ONNX runtime, no mmpose/mmcv needed)
- GPU: requires `onnxruntime-gpu` + PyTorch with CUDA (~6GB)

### Keypoint Formats

| Model | Keypoints | Format |
|-------|-----------|--------|
| `body_with_feet` (default) | **26** | HALPE26 (COCO 17 + 6 foot: BigToe, SmallToe, Heel x2) |
| `body` | 17 | Standard COCO |
| `whole_body` | 133 | Full body (face + hands) |

**Incompatibility with our project:** Sports2D/rtmlib use HALPE26 or COCO format. Our project uses H3.6M 17kp with different keypoint ordering. **Mapping layer required.**

### Joint Angles (Sports2D computes automatically)

**Joint angles (12):** Ankle dorsiflexion, Knee flexion, Hip flexion, Shoulder flexion, Elbow flexion, Wrist flexion

**Segment angles (15):** Foot, Shank, Thigh, Pelvis, Trunk, Shoulders, Head, Arm, Forearm

**Post-processing:** Hampel outlier rejection, interpolation, configurable filtering (Butterworth 6Hz, Kalman, GCV spline, Gaussian, LOESS, Median)

### rtmlib (github.com/Tau-J/rtmlib)

- **Version:** v0.0.15, standalone inference library
- **Dependencies:** numpy, opencv-python, opencv-contrib-python, onnxruntime (no mmpose!)
- **Modes:** lightweight (RTMPose-t), balanced (RTMPose-m), performance (RTMPose-x)
- **Python API:**
```python
from rtmlib import Body_with_feet, PoseTracker
pose_tracker = PoseTracker(Body_with_feet, mode='balanced', backend='onnxruntime')
keypoints, scores = pose_tracker(frame)  # (N_persons, 26, 2)
```

### What Sports2D gives us that we don't have

- Ready-made biomechanics-grade joint angles (with proper zero references)
- Foot keypoints (BigToe, SmallToe, Heel) for blade angle estimation
- Well-tested post-processing (Hampel, Butterworth, Kalman)
- Pixel-to-meter conversion with perspective correction

### What Sports2D does NOT solve

- 3D pose (we already have this via MotionAGFormer/TCPFormer)
- Figure-skating-specific analysis
- H3.6M format output (requires mapping)

### Integration complexity: MEDIUM

- Need HALPE26→H3.6M keypoint mapping
- Pose2Sim dependency may conflict with our stack
- Can use rtmlib directly (lighter) but still need format conversion

---

## 2. YOLO11/26 Pose

### YOLO11 Pose

- **Exists:** Yes, confirmed on HuggingFace and Ultralytics Platform
- **Models:** yolo11n/s/m/l/x-pose.pt (5 sizes, 17 COCO keypoints)
- **vs YOLOv8:** ~2% higher mAP, ~22% fewer params, faster CPU inference
- **API:** Identical to YOLOv8: `YOLO("yolo11n-pose.pt")`

**Codebase issue found:** `h36m_extractor.py` uses `yolov8{size}-pose.pt`, `yolo_extractor.py` uses `yolov11{size}-pose.pt` — inconsistent.

### YOLO26 Pose (January 2026)

**COCO Keypoints benchmarks (640px):**

| Model | mAP50-95 | Params | Speed CPU (ms) | Speed T4 (ms) |
|-------|----------|--------|----------------|---------------|
| YOLO26n-pose | 57.2 | 2.9M | 40.3 | 1.8 |
| YOLO26s-pose | 63.0 | 10.4M | 85.3 | 2.7 |
| YOLO26m-pose | 68.8 | 21.5M | 218.0 | 5.0 |
| YOLO26l-pose | 70.4 | 25.9M | 275.4 | 6.5 |
| YOLO26x-pose | 71.6 | 57.6M | 565.4 | 12.2 |

**Key innovations:** NMS-free, ProgLoss+STAL loss, MuSGD optimizer, 43% faster CPU

### Recommendation

- **Short-term:** Standardize on YOLOv8-Pose (current, works) or upgrade to YOLO11n-Pose (drop-in replacement)
- **Medium-term:** YOLO26n-Pose for best accuracy/efficiency tradeoff (1.8ms on T4 vs current ~10ms)
- All output **17 COCO keypoints** — compatible with our H3.6M mapping

---

## 3. 3D Lifting Models

### Current Stack

| Model | MPJPE (P1) | Params | Status |
|-------|-----------|--------|--------|
| MotionAGFormer-S | ~41mm | ~5M | Integrated (60MB) |
| TCPFormer | ~38mm | varies | Integrated (423MB) |

### New Contenders

| Model | MPJPE | Params | MACs | VRAM | Code Available |
|-------|-------|--------|------|------|----------------|
| **Pose3DM-L** | **37.9mm** | 7.4M | 127M | ~200MB | **NO (repo 404)** |
| Pose3DM-S | 42.1mm | 0.5M | 9M | ~50MB | NO |
| PoseMamba-L | 38.1mm | 6.7M | 115M | ~180MB | Likely yes |
| MotionBERT | 39.2mm | 42.3M | 719M | ~600MB | Yes, but too heavy |

**Pose3DM-L** would beat our MotionAGFormer-L by 0.5mm MPJPE with 60% less compute — but code isn't released yet. **Monitor this.**

**Pose3DM technique worth adopting:** FTV (Fractional-order Total Variation) regularization for temporal smoothing. Better than Butterworth/Kalman at preserving motion discontinuities.

---

## 4. Kinovea Alternatives with AI

### Pose2Sim (592 stars) — BEST overall

- Full markerless biomechanics pipeline: 2D pose → 3D reconstruction → OpenSim kinematics
- Supports BlazePose, OpenPose, MMPose backends
- Multi-camera calibration, triangulation, filtering, person association
- **Limitation:** Designed for multi-camera; monocular mode in development
- **Relevance:** If we ever add musculoskeletal modeling (joint forces, muscle activations)

### Sports2D (202 stars) — Best for angles

- Ready-made joint/segment angle computation
- Foot keypoints for blade analysis
- **Limitation:** 2D only, no 3D, no skating-specific features

### SprintLab (32 stars) — Sprint-specific

- YOLO-based, sprint kinematic analysis
- Not applicable to figure skating

---

## 5. Figure Skating Repos (2025-2026)

### FS-Jump3D (github.com/ryota-skating/FS-Jump3D) — MOST USEFUL

- ACM MM 2024 workshop
- First figure skating jump dataset with 3D pose (12-viewpoint, markerless mocap)
- 4 skaters x 7 jump types, outputs H3.6M 17kp format
- **Integration:** Direct — same format as our project
- **Use:** Training data for fine-tuning our 3D lifter on skating

### VIFSS (August 2025, Tanaka et al.)

- View-invariant embeddings for temporal action segmentation
- 92%+ F1@50 on element-level segmentation
- Uses custom 3D pose extraction (not YOLO-based)

### JudgeAI-LutzEdge (github.com/ryota-skating/JudgeAI-LutzEdge)

- Binary Lutz edge error detection
- HRNet → Strided Transformer → Logistic Regression
- **Limited:** Only Lutz jumps, pretrained only
- **Relevance:** Low — our BDA algorithm already does generic edge detection

### Figure-Skating-Quality-Assessment (ACM MM 2025)

- Multi-modal (audio+video) quality assessment
- Outputs ISU TES/PCS scores
- **Limitation:** Requires pre-computed features (I3D + VGGish), labeled ISU data
- **Relevance:** Medium — AGL (Audio-Guided Localization) concept interesting for segmentation

---

## 6. MediaPipe Status (2026)

- **No sports improvements.** Still 33 landmarks, same limb-flipping issues
- **Version:** v0.10.32 (January 2026)
- **API:** Transitioning to Tasks API (PoseLandmarker)
- **Our migration away from MediaPipe was correct.**

---

## Recommendations (Priority Order)

### Phase A: Quick Wins (1-2 days)

1. **Standardize YOLO model** — fix inconsistency between h36m_extractor.py (v8) and yolo_extractor.py (v11)
2. **Upgrade to YOLO26n-Pose** — drop-in replacement, best accuracy/efficiency
3. **Fix false positive validation** — already done (spread threshold)

### Phase B: Pose Estimation Upgrade (1-2 weeks)

4. **Try rtmlib (standalone)** — no Pose2Sim dependency, just ONNX inference
   - Test RTMPose with `body_with_feet` (26kp HALPE26)
   - Evaluate foot keypoints for blade angle estimation
   - Compare with current YOLOv8-Pose
5. **Sports2D angle computation** — borrow joint angle formulas (even without full integration)
6. **Monitor Pose3DM-L** — if code released, switch from MotionAGFormer

### Phase C: Future (research phase)

7. **FS-Jump3D data** — fine-tune 3D lifter on skating-specific poses
8. **FTV temporal smoothing** — implement fractional-order TV regularization
9. **Pose2Sim integration** — for full biomechanics pipeline if needed
10. **Kinovea-style interactive viewer** — web-based with real-time toggle controls

---

## References

- Sports2D: github.com/davidpagnon/Sports2D (v0.8.28)
- rtmlib: github.com/Tau-J/rtmlib (v0.0.15)
- Pose2Sim: github.com/perfanalytics/pose2sim (v0.10.40)
- YOLO26: docs.ultralytics.com/tasks/pose/ (Jan 2026)
- YOLO11: huggingface.co/Ultralytics/YOLO11
- Pose3DM: MDPI 2504-3110/9/9/603 (code not available)
- FS-Jump3D: github.com/ryota-skating/FS-Jump3D (ACM MM 2024)
- VIFSS: arXiv:2508.10281 (Aug 2025)
- JudgeAI-LutzEdge: github.com/ryota-skating/JudgeAI-LutzEdge
- Figure-Skating-Quality-Assessment: github.com/ycwfs (ACM MM 2025)
- SprintLab: github.com/mvch1ne/sprintlab (32 stars)
