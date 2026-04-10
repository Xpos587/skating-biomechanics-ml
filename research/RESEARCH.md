# Research Memory Bank

> Index of all research materials. Documents are kept for historical context — decisions made, alternatives evaluated, and dead ends explored.

## Current System (2026-04-09)

| Component | Chosen | Rejected |
|-----------|--------|----------|
| **2D Pose** | RTMPose via rtmlib (HALPE26, 26kp) — sole backend | YOLO26-Pose (removed), BlazePose (33kp, deprecated) |
| **3D Lifting** | MotionAGFormer-S (38.4mm MPJPE) | Pose3DM (code not released), VIBE (too heavy) |
| **Tracking** | OC-SORT + anatomical biometrics | DeepSORT (color-based, fails on black clothing) |
| **Jump Height** | CoM parabolic trajectory | Flight time (60% error for low jumps) |
| **Inference** | ONNX Runtime + CUDA 12 compat | PyTorch CUDA (broken on CUDA 13.2) |

---

## Documents

### Active References

| File | Date | Content |
|------|------|---------|
| [RESEARCH_SUMMARY_2026-03-28.md](RESEARCH_SUMMARY_2026-03-28.md) | 2026-03-28 | **Comprehensive findings** from Exa + Gemini Deep Research (41 papers). Critical: flight time 60% error, physics-informed optimizer, OC-SORT biometrics, FSBench dataset |
| [RESEARCH_POSE_TOOLS_2026-03-31.md](RESEARCH_POSE_TOOLS_2026-03-31.md) | 2026-03-31 | Sports2D, RTMPose, YOLO26-Pose, Kinovea alternatives evaluation. Led to rtmlib integration |

### Integration Plans (Completed)

| File | Date | Content |
|------|------|---------|
| [ATHLETEPOSE3D_INTEGRATION.md](ATHLETEPOSE3D_INTEGRATION.md) | 2026-03-28 | AthletePose3D (MotionAGFormer/TCPFormer) integration plan. Status: **DONE** |

### Research Prompts (Historical)

These are the prompts given to Gemini Deep Research. The results informed implementation decisions.

| File | Date | Topic | Outcome |
|------|------|-------|---------|
| [PHYSICS_DETECTION_RESEARCH.md](PHYSICS_DETECTION_RESEARCH.md) | 2026-03-27 | Physics-based detection, occlusion, blade edge, CoM | → PhysicsEngine, BladeEdgeDetector3D, PoseFiltering |
| [SPATIAL_REFERENCE_RESEARCH_PROMPT.md](SPATIAL_REFERENCE_RESEARCH_PROMPT.md) | 2026-03-28 | Camera calibration, ice rink detection | → SpatialReferenceDetector (per-frame horizon) |
| [SPATIAL_REFERENCE_EXA_SUMMARY.md](SPATIAL_REFERENCE_EXA_SUMMARY.md) | 2026-03-28 | Exa search results for spatial reference | → Same as above |
| [VISUALIZATION_RESEARCH_PROMPT.md](VISUALIZATION_RESEARCH_PROMPT.md) | 2026-03-28 | Skeleton overlay, kinematics viz, HUD design | → Layered HUD system (0-3) |

### Evaluated & Rejected Alternatives

| File | Date | Topic | Why Rejected |
|------|------|-------|---------------|
| [MOGANET_RESEARCH.md](MOGANET_RESEARCH.md) | 2026-03-29 | MogaNet-B as pose estimator | Requires mmcv compilation, not standalone |
| [MOGANET_SUMMARY.md](MOGANET_SUMMARY.md) | 2026-03-29 | MogaNet summary + YOLOv8 alternative | Duplicate of above. YOLOv8 superseded by YOLO26+rtmlib |

### Original Research

| File | Date | Content |
|------|------|---------|
| [RESEARCH_ORIGINAL.md](RESEARCH_ORIGINAL.md) | 2026-03-27 | First architecture research: YOLOv8 vs YOLOv11, BlazePose vs MoveNet, MotionBERT vs Pose3DM, AthletePose3D, FS-Jump3D datasets, MotionDTW, KISMAM, multimodal RAG |

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-27 | BlazePose → H3.6M 17kp | Simpler, sufficient for biomechanics, 3D-first |
| 2026-03-28 | Flight time → CoM trajectory | 60% error on low jumps (Gemini finding) |
| 2026-03-28 | Color Re-ID → anatomical biometrics | Black clothing on ice eliminates color |
| 2026-03-29 | MMPose/MogaNet → YOLO26-Pose | mmcv compilation fails on CUDA 13 |
| 2026-03-31 | YOLO26-Pose → RTMPose (rtmlib) | Better tracking, foot keypoints, ONNX (CPU+GPU) |
| 2026-04-01 | PyTorch CUDA → onnxruntime-gpu | CUDA 13 incompatibility solved with standalone CUDA 12 libs |
| 2026-04-01 | Raw 2D skeleton → CorrectiveLens 3D→2D | Kinematic constraints fix occlusion artifacts |
| 2026-04-09 | Remove YOLO26-Pose backend entirely | RTMPose is sole backend; removed ultralytics dep, h36m_extractor.py, yolo_extractor.py, action segmentation code |

## Open Questions

1. **Pose3DM-L** (Mamba, 37.9mm MPJPE) — code not released. Monitor github.com/Reus3237/Pose3DM
2. **Mass from video** — mathematically unsolved without reference force
3. **Rocker vs Counter** — requires arc history + blade state, underspecified
4. **Motion blur at 4-5 rev/s** — can angular momentum constraints help?
