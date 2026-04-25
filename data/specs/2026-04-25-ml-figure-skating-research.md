# ML in Figure Skating: Comprehensive Research Report

> **Date:** 2026-04-25
> **Method:** 5 parallel research agents (Opus-level), each with independent web search
> **Status:** Final synthesis (updated with R3 YOLO26 architecture deep-dive)

---

## 1. Pose Estimation Landscape (2025-2026)

### COCO 17kp Leaderboard (val2017)

| Model | Architecture | AP | Params | GPU Speed | Notes |
|-------|-------------|-----|--------|-----------|-------|
| PoseSynViT-XL | Top-down ViT | **84.3** | ~1B | Slow | CVPR 2025W, new SOTA |
| ViTPose-H | Top-down ViT | 81.1 | Huge | Slow | TPAMI 2024 |
| HRNet-W48+Dark | Top-down CNN | 78.9 | Large | Slow | |
| **RTMPose-l** | Top-down SimCC | **76.0** | ~49M | 4.6ms GTX1660Ti | **Our production backbone** |
| **RTMPose-m** | Top-down SimCC | **75.8** | ~34M | 4.3ms GTX1660Ti | 73.2 pipeline AP |
| RTMO-l | One-stage SimCC | 74.8 | ~35M | 141 FPS V100 | CVPR 2024 |
| DETRPose-L | Transformer | 72.5 | ~40M | 32.5ms | ICCV 2025, no bbox output |
| YOLO26x-pose | One-stage RLE | 71.6 | 57.6M | 12.2ms T4 | Sep 2025 |
| **YOLO26s-pose** | One-stage RLE | **63.0** | **11.9M** | 2.7ms T4 | **Our student model** |
| YOLO26n-pose | One-stage RLE | 57.2 | 2.9M | 1.8ms T4 | Mobile candidate |

### Key Architectural Trends

**Three paradigms coexist — each optimal for its architecture:**

1. **Heatmap-based** (ViTPose, HRNet) — highest accuracy, heaviest
2. **SimCC / coordinate classification** (RTMPose, RTMO) — best speed/accuracy tradeoff
3. **RLE / direct regression** (YOLO26) — best for single-stage, NMS-free

**YOLO26 innovations over YOLOv8/YOLO11:**
- **RLE** replaces L1 regression → better keypoint localization
- **NMS-free inference** → constant latency, simpler deployment
- **DFL removed** → simpler architecture
- **MuSGD optimizer** → 5-10x faster convergence
- **ProgLoss** → dynamic loss balancing
- **STAL** → better small-target (keypoint) supervision

**RTMPose has NOT been updated** since v1.3.0 (Jan 2024). MMPose project semi-dormant.

### YOLO26 Pose26 Head Architecture (Deep Dive)

**Pose26 vs YOLO11 Pose head:**

| Component | YOLO11 Pose | YOLO26 Pose26 |
|-----------|-------------|---------------|
| Keypoint decoding | `(pred * 2.0 + (anchor - 0.5)) * stride` | `(pred + anchor) * stride` (simpler) |
| Uncertainty model | None | **RealNVP flow model** (6 coupling layers) |
| Sigma prediction | None | Per-keypoint σ_x, σ_y (training only) |
| Classification head | Standard Conv | DWConv-based (lighter) |
| DFL | Distribution Focal Loss | **Removed** (reg_max=1 → Identity) |
| NMS | Required | **Optional** (end2end dual head) |

**RealNVP flow model details:**
- 6 coupling layers with alternating binary masks `[[0,1],[1,0]] × 3`
- Each layer: Linear(2,64) → SiLU → Linear(64,64) → SiLU → Linear(64,2) → Tanh
- Prior: `MultivariateNormal(loc=zeros(2), cov=eye(2))`
- Models keypoint error *distribution*, not just point estimates
- Fused away at inference (not needed for prediction)

**End2end dual head:**
- Training: `E2ELoss` on both one2many + one2one heads simultaneously
  - o2m weight: 0.8 → 0.1 (decays over training)
  - o2o weight: 0.2 → 0.9 (increases)
- Inference: only one2one head used, one2many fused away
- NCNN/RKNN/TFLite do NOT support end2end (auto-disabled)

**Known issues (from GitHub):**
1. **rle_loss goes negative** (issue #23233): Fixed with `rle_loss.clamp(min=0)`. Monitor CSV!
2. **FP16 ONNX export** (issue #23645): End2end+FP16 output may remain FP32
3. **COCO val discrepancy** (discussion #23343): Must use `coco-pose.yaml` not `coco.yaml`
4. **TensorRT int8** (issue #23756): End2end build issues with TRT ≤ 10.3.0

**YOLO26 model sizes (official):**

| Size | Depth | Width | Params | GFLOPs |
|------|-------|-------|--------|--------|
| n | 0.50 | 0.25 | 3.7M | 10.7 |
| **s** | **0.50** | **0.50** | **11.9M** | **29.6** |
| m | 0.50 | 1.00 | 24.3M | 85.9 |
| l | 1.00 | 1.00 | 28.7M | 104.3 |
| x | 1.00 | 1.50 | 62.9M | 226.3 |

**RLE loss weights (hardcoded for COCO 17kp):**
`[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5]`
Higher weights (1.2, 1.5) on wrists, ankles, knees — harder keypoints.

---

## 2. Model Comparison for Our Project

### RTMPose-m vs YOLO26s-Pose vs MogaNet-B

| Criterion | RTMPose-m | YOLO26s-Pose | MogaNet-B (AP3D) |
|-----------|-----------|-------------|-----------------|
| **COCO AP** | **75.8** | 63.0 | ~73 (est.) |
| **On-ice AP** | ~72% (measured) | ~65% (target) | **~95%** (teacher) |
| **Params** | ~34M | **~12M** | ~44M |
| **FLOPs** | 1.93G | 29.6G | ~3.5G |
| **GPU latency** | **4.3ms** | 15ms | N/A (teacher) |
| **CPU latency** | **27ms** | ~45ms | N/A |
| **NMS** | Required | **None** | Required |
| **Keypoint head** | SimCC | RLE | Heatmap |
| **Export** | ONNX (rtmlib) | ONNX, TRT, NCNN | PyTorch |
| **Role** | **Production** | Student (KD) | Teacher (pseudo-labels) |

### Bottom Line

- **RTMPose-m** = best accuracy/speed for production (75.8 AP, 4.3ms GPU)
- **YOLO26s-Pose** = best deployment (11.9M params, NMS-free, single-stage, RealNVP uncertainty)
- **MogaNet-B** = best accuracy (95% on skating, teacher-only)
- **KD strategy correct**: MogaNet-B → pseudo-labels → YOLO26s → approximate RTMPose accuracy at single-stage speed

---

## 3. Knowledge Distillation for Pose Estimation

### DistilPose (CVPR 2023) — The Reference

- Heatmap→regression KD via Token-distilling Encoder (TDE) + Simulated Heatmaps
- DistilPose-L: **74.4 AP** (SOTA among regression models at time)
- Bridges accuracy gap between heatmap teacher and regression student
- Code: github.com/yshMars/DistilPose

### Our Approach: Offline Pseudo-Labeling

Instead of online KD (DistilPose), we use offline pseudo-labels:
1. MogaNet-B teacher predicts keypoints on 291K skating images
2. Teacher coords converted to YOLO format (crop → original → normalized)
3. YOLO26s-Pose fine-tuned on pseudo-labels + 10% COCO real labels
4. No KD loss modifications needed — standard YOLO training

**Advantages over online KD:**
- No custom trainer code needed
- Standard Ultralytics pipeline
- Can mix multiple data sources
- Simpler debugging

**Limitation:** No explicit teacher-student feature alignment (TDE), relies on label quality.

### Confirmed by Research

- Suzuki et al. (CVPR 2024): pseudo-labeling effective for pose when teacher >> student
- DistilPose (CVPR 2023): heatmap→regression transfer is lossless with TDE
- RTMPose training recipe: stage 1 on large dataset, stage 2 fine-tune on target

---

## 4. Figure Skating ML Research (2024-2026)

### Datasets

| Dataset | Content | Classes | Status |
|---------|---------|---------|--------|
| FSC | 5168 seq | 64 | Downloaded |
| MCFS | 2668 seg | 129 | Downloaded |
| SkatingVerse | 28K videos | 28 | Downloaded |
| AthletePose3D | 1.3M frames | 12 sports | Downloaded |
| FSBench | 783 videos | 76+ hours | Temporarily closed |
| FineFS | 1167 samples | scores+boundaries | Link dead |
| FSD-10 | 10 classes | 10 | Available |

### Action Recognition on Skating Data

**GCN on FSD-10 (10 classes):**
- Extended skeleton graph + partitioning: **94.7%** (spins), **91.2%** (jumps)
- Standard ST-GCN: ~80-90%
- Our ST-GCN on MCFS (129 classes): 28.5% (over-smoothing on 17-node graph)

**GCN on MCFS (129 classes):**
- Our BiGRU: **63.9%** (dominates GCN by 2.3x)
- Why BiGRU wins: over-smoothing, small data, variable length, temporal > spatial for skating

### Key Papers (2024-2026)

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| VIFSS | arXiv 2025 | View-invariant features, 92.56% F1@50 on element-level TAS |
| PoseSynViT | CVPR 2025W | SOTA 84.3 AP, scalable ViT 10M-1B |
| DETRPose | ICCV 2025 | First real-time transformer MPPE, 72.5 AP |
| ProbPose | CVPR 2025 | Probabilistic keypoints, handles out-of-image |
| SHARDeg | 2025 | Accuracy drops >40% depending on degradation type at same FPS |
| PCCTR-GCN | CMC 2025 | Pose correction + channel topology refinement for GCN |
| Mamba Pyramid | ACM MM 2025 | Action quality assessment for skating |
| Figure Skating QA | CVPR 2024W | Multi-modal quality assessment |

### VIFSS (View-Invariant Figure Skating-Specific)

- **Paper:** arXiv:2508.10281 (Aug 2025)
- **Result:** 92.56% F1@50 on element-level Temporal Action Segmentation
- **Key:** Learns camera-invariant features from 2D skeletons — no 3D needed
- **Code:** Public (Apache-2.0)
- **Compatibility:** H3.6M 17kp plug-and-play
- **Blocking:** SkatingVerse not public, but FSC viable as substitute
- **For us:** Potential solution for cross-athlete comparison without 3D poses

### SHARDeg (Degradation Robustness)

- **Finding:** At same effective FPS, model accuracy varies by **>40%** depending on degradation type
- **Relevance:** Our pipeline uses frame_skip=8 → real-world degraded data
- **Mitigation:** Bilinear interpolation during training and inference improves resilience by up to 40%
- **LogSigRNN** outperforms DeGCN at 3 FPS by avg 6%, despite trailing by 11-12% at 30 FPS

---

## 5. Downstream Quality: Pose Quality → Action Recognition

### Why Pose Quality Matters for InfoGCN

- **17kp vs 26kp**: Extended keypoints (HALPE26) provide foot angle info → +3-5% on skating action recognition (estimated)
- **Pose noise impact**: GCNs are more sensitive to spatial noise than temporal models
- **Frame rate**: 3 FPS → accuracy drop varies from 5% to 40%+ depending on degradation type
- **Interpolation helps**: Bilinear interpolation during training improves degradation resilience

### PCCTR-GCN (Pose Correction for GCN)

- Integrates pose correction into GCN architecture
- Channel topology refinement captures inter-joint relationships
- Relevant for skating where joint angles are discriminative features

### Our Specific Context

- **Current pose backbone:** RTMPose (rtmlib, HALPE26) → 26kp with feet
- **Downstream model:** BiGRU at 63.9% on MCFS (129 classes)
- **Bottleneck:** Not pose quality — it's data quantity and class granularity
- **v36b training target:** If YOLO26s reaches ~70 AP on skating, it could replace RTMPose for mobile deployment while maintaining acceptable downstream quality

---

## 6. Recommendations

### Immediate (v36b training running)

1. **Let v36b complete** — 200 epochs with corrected hyperparams (cos_lr, scale=0.3, etc.)
2. **Monitor at epoch 10, 20, 50** — check mAP50(P) convergence
3. **Export best model to ONNX** — test inference speed on RTX 3050 Ti

### Short-term (after v36b)

4. **Evaluate YOLO26s-finetuned on downstream BiGRU** — does pose quality transfer?
5. **Consider RTMO** — one-stage SimCC (74.8 AP) could be better than YOLO26s RLE (63.0 AP) for our use case
6. **640px fine-tuning stage** — 15-20 epochs after main training for full-resolution accuracy

### Medium-term

7. **VIFSS pre-training** — view-invariant features for cross-athlete comparison
8. **Pseudo-label on SkatingVerse** — 28K real competition videos → expand training data
9. **Per-keypoint filtering** — re-extract heatmaps with confidence > 0.5 threshold

### Long-term

10. **DETRPose** — watch for maturity (currently no bbox output, no visibility prediction)
11. **PoseSynViT** — if accuracy ceiling matters more than speed
12. **MotionBERT pretrain → finetune** — strongest research direction for element classification

---

## Sources

- **Agent R1 (SOTA Pose):** 73 web searches, PoseSynViT, DETRPose, YOLO26 benchmark tables, RTMPose family, regression vs heatmap debate
- **Agent R2 (Figure Skating ML):** FSBench, VIFSS, MCFS, FSD-10, skating-specific GCN results, Mamba Pyramid, YourSkatingCoach
- **Agent R3 (YOLO26-Pose):** YOLO26 architecture (Pose26 head, RealNVP flow model, RLE, MuSGD, STAL, ProgLoss), DFL removal, training recipe, DistilPose KD, export issues, known bugs (rle_loss negative, NCNN end2end)
- **Agent R4 (InfoGCN & Downstream):** SHARDeg degradation benchmark, PCCTR-GCN pose correction, 17kp vs 26kp, frame rate impact, LogSigRNN
- **Agent R5 (Production):** RTMPose-m vs YOLO26s comparison, deployment considerations, ONNX export
