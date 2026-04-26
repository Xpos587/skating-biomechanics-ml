# Post-v35d Strategy: 5-Agent Brainstorm Synthesis

**Date:** 2026-04-25
**Scope:** Post-v35d planning — scenarios, alternatives, production path, data scaling, budget
**Agents:** Results Analyst, Alternative Strategies, Production Integration, Data & Scaling, Budget & Timeline

---

## Executive Summary

v35d is running with 3 critical bug fixes (KD activation, coordinate mismatch, per-image normalization). While we wait for results, this report maps the full decision tree for every possible outcome.

**Key cross-agent consensus:**
1. **RTMPose Fine-Tuning is the #1 fallback** — 75.8% COCO AP baseline (vs 57-60% YOLO26s), zero pipeline changes, $15-25
2. **Cross-architecture KD is fundamentally risky** — heatmap-to-regression paradigm mismatch, 36 bugs found, more may lurk
3. **Pseudo-Labeling SkatingVerse is the highest-upside scaling opportunity** — 45K → 300K+ data, $10-15
4. **Budget is sufficient** — $119-128 remaining, risk-adjusted total ~$100 (67% of $150)
5. **Production integration is straightforward** — ONNX export easy, 3-5 days with feature flag

---

## 1. v35d Monitoring Protocol

### Go/No-Go Thresholds (Results Analyst)

| Epoch | RED (abort) | YELLOW (monitor) | GREEN (proceed) |
|-------|-------------|-------------------|-----------------|
| 10 | kd_coord_loss=0, cls_loss>50 | kd_coord_loss flat, cls_loss 5-20 | kd_coord_loss>0 decreasing, cls_loss<5 |
| 20 | Pose mAP50<0.05 | Pose mAP50 0.05-0.15 | Pose mAP50>0.15 |
| 50 | Pose mAP50<0.15 | Pose mAP50 0.15-0.30 | Pose mAP50>0.30 |
| 100 | Pose mAP50<0.30 | Pose mAP50 0.30-0.50 | Pose mAP50>0.50 |
| 210 | Pose mAP50<0.40 | Pose mAP50 0.40-0.60 | Pose mAP50>0.60 |

### Bug Fix Verification (v35d vs v35c)

| Fix | What to Check | v35c Symptom | v35d Expected |
|-----|---------------|--------------|---------------|
| KD activation | `[KD] Epoch N` logs | kd_weight=0 all epochs | kd_weight>0 from epoch 3 |
| Coordinate mismatch | val/cls_loss | Spikes 142+ at ep 4-5 | Stay <10 |
| Per-image normalization | Loss curve smoothness | Batch-to-batch jumps | Smooth descent |

### Required Baselines ($35 total)

| Baseline | Cost | Priority | When |
|----------|------|----------|------|
| YOLO26s-pose zero-shot on skating val | $0.20 | P0 | Now |
| RTMPose/RTMO on same val set | $0.20 | P0 | Now |
| Direct fine-tuning no-KD (50 epochs) | $7 | P0 | After v35d ep 20 |
| Full fine-tuning no-KD (210 epochs) | $15 | P1 | If v35d mediocre |

---

## 2. Decision Tree: If v35d Shows X, Do Y

```
v35d completes (epoch 210 or early stop)
│
├─ Pose mAP50 > 0.60 (GOOD, ~15-25% probability)
│   ├─ COCO AP > 0.55 → DEPLOY via ONNX export
│   │   └─ Integration: 3-5 days, feature flag, parallel to RTMO
│   └─ COCO AP < 0.55 → Add COCO 15-20%, re-train ($10)
│
├─ 0.30 < Pose mAP50 < 0.60 (MEDIOCRE, ~50-60% probability)
│   ├─ kd_coord_loss converged → Architecture mismatch is bottleneck
│   │   → RTMPose Fine-Tuning ($15-25) [recommended]
│   │   → OR Same-Family KD YOLO26l→n ($25-35)
│   ├─ kd_coord_loss NOT converged → Bug in coordinate transform
│   │   → Debug first, then Direct Fine-Tune ($10-20) as fallback
│   └─ Loss unstable → Remaining bugs from audit
│       → Direct Fine-Tune ($10-20) or RTMPose Fine-Tune ($15-25)
│
└─ Pose mAP50 < 0.30 (POOR, ~20-30% probability)
    ├─ GT loss also poor → Data problem
    │   → Pseudo-Labeling SkatingVerse ($10-15) + retrain
    └─ GT loss OK but KD hurts → KD actively harmful
        → Abandon KD → Direct Fine-Tune ($10-20)
```

---

## 3. Alternative Approaches Ranked

### Comparison Matrix (Alternative Strategies Agent)

| Rank | Approach | Accuracy | Risk | Cost | Time | Production | Skating |
|------|----------|----------|------|------|------|------------|---------|
| **1** | **RTMPose Fine-Tuning** | 5/5 | 3/5 | 4/5 | 3/5 | 5/5 | 4/5 |
| 2 | Pseudo-Labeling SkatingVerse | 5/5 | 3/5 | 4/5 | 2/5 | 4/5 | 5/5 |
| 3 | RTMO Direct Fine-Tuning | 4/5 | 3/5 | 5/5 | 3/5 | 5/5 | 4/5 |
| 4 | Direct Fine-Tuning YOLO26s | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 3/5 |
| 5 | Same-Family KD YOLO26 | 4/5 | 4/5 | 4/5 | 3/5 | 5/5 | 3/5 |

### Why RTMPose Fine-Tuning Wins

1. **18+ percentage points higher baseline** — RTMPose-m 75.8% vs YOLO26s-pose 57-60% COCO AP
2. **SimCC sub-pixel precision** — fundamentally better than YOLO's direct regression for keypoint localization
3. **Zero pipeline changes** — already in production via rtmlib, ONNX proven
4. **Well-documented** — mmpose fine-tuning tutorial, TensorRT support via mmdeploy
5. **$15-25 cost** — well within budget

### Key Insight from Research

The cross-architecture KD (heatmap→regression) faces fundamental challenges:
- Coordinate system mismatch (288x384 crops vs 640x640 full-frame)
- Feature space incompatibility (no learned projectors, raw coordinate MSE)
- Inherent precision gap between heatmap and regression paradigms

ECCV 2024 paper "Cross-Representation Distillation" directly addresses this problem with projector-based alignment — worth studying if we continue the cross-arch KD path.

---

## 4. Production Integration (if v35d succeeds)

### Complexity: MEDIUM (3-5 days)

### ONNX Export: EASY
```python
from ultralytics import YOLO
model = YOLO('yolo26s-pose-kd.pt')
model.export(format='onnx', half=True, imgsz=640, simplify=True)
```

### Key Architecture Decision: Parallel Extractor (not swap)

Create `YOLO26PoseExtractor` alongside existing `PoseExtractor`. Both return `TrackedExtraction` with H3.6M 17kp poses. Feature flag selects backend.

### Files Requiring Changes

| File | Change | Complexity |
|------|--------|------------|
| `ml/src/pose_estimation/yolo26_extractor.py` | NEW — ONNX extractor | Medium |
| `ml/src/visualization/pipeline.py` | Add `pose_backend` param | Easy |
| `ml/src/web_helpers.py` | Pass through | Easy |
| `ml/gpu_server/Containerfile` | Add ONNX model (~20MB) | Easy |
| `backend/app/config.py` | Add `pose_backend` setting | Easy |

### What Does NOT Change
- `coco_to_h36m()` — shared conversion, works for both
- `PoseTracker` — backend-agnostic, receives H3.6M poses
- GapFiller, Smoothing, PhaseDetector, Metrics, DTW — all operate on `TrackedExtraction`
- GPU server image size — ONNX only, no torch needed

### Rollback: Instant
```python
# backend/app/config.py
pose_backend: str = "rtmo"  # one-line change
```

### Speed Estimate
- Current RTMO-m: ~12s for 14.5s video (frame_skip=8)
- Expected YOLO26s-pose KD: ~4-5s (3x faster, 9M vs ~34M params)

---

## 5. Data Strategy

### Current Data (~50K images)

| Dataset | Train | Element Types | Quality |
|---------|-------|---------------|---------|
| FineFS | 8,904 | Jumps 44%, Spins 34%, Steps 15% | GOE scores 0-6.66 |
| AP3D-FS | 35,705 | Jumps only (7 types) | Human-verified 3D→2D |
| COCO 10% | ~5,659 | General poses | Standard |

### Data Gaps

| Gap | Severity | Solution |
|-----|----------|----------|
| No spins in AP3D-FS | Medium | FineFS has 34% spins (covers this) |
| No step sequences in AP3D-FS | Low | FineFS has 15% steps |
| No pairs elements | Low | Deprioritize — single-skater first |
| Validation jump-biased | Medium | Build stratified val set |

### Pseudo-Labeling SkatingVerse (highest-upside data opportunity)

- 28,579 videos, 28 element classes including spins and quads
- ~200K frames at 2fps sampling → after filtering ~60K high-quality
- **Cost: ~$0.10 inference + $5-10 training**
- Expected improvement: +0.03-0.08 AP

### GOE-Weighted Training
FineFS quality scores provide natural sample weighting:
```python
weight = max(GOE + 2, 0.1) / 3  # Higher GOE = more influence
```
Keep low-GOE samples — valuable for recognizing errors.

### Teacher Heatmap Quality Note
Confidence values exceed 1.0 (raw DeconvHead logits). Current `clamp(min=0.1)` works but is uncalibrated. Recommendation: use softmax probability at predicted location as confidence (naturally [0,1]).

---

## 6. Budget & Timeline

### Spend Audit

| Category | Amount | Notes |
|----------|--------|-------|
| v1-v35 iterations | ~$22-31 | Including buggy runs |
| Wasted on bugs | ~$7-8 | 23-26% waste ratio |
| **Remaining** | **$119-128** | |

### Scenario-Based Allocation

| Scenario | Probability | Additional Cost | Total Project | Budget OK? |
|----------|-------------|----------------|---------------|------------|
| A: v35d succeeds (AP>0.60) | 25% | $15-25 (ONNX + testing) | $52-66 | Yes (44%) |
| B: v35d mediocre (AP 0.30-0.60) | 45% | $30-50 (RTMPose + tuning) | $67-91 | Yes (61%) |
| C: v35d fails (AP<0.30) | 20% | $25-45 (pivot + alternatives) | $62-86 | Yes (57%) |
| D: Needs more data | 10% | $15-30 (pseudo-labeling) | $52-71 | Yes (47%) |

### Risk-Adjusted Expected Total: ~$100 (67% of budget)

### Calendar Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| v35d training | ~35h GPU (1.5 days) | Running now |
| Baseline comparisons | 1-2 days | Parallel with v35d |
| Evaluation + decision | 0.5 day | After v35d epoch 50+ |
| Pivot execution (if needed) | 2-4 days | After decision |
| Production integration | 3-5 days | After model selected |
| **Total (best case)** | **5-7 days** | v35d succeeds |
| **Total (worst case)** | **10-14 days** | Pivot + retraining |

### Hard Budget Gates

1. **Stop at $75 spent** if AP < 0.50 — cut losses
2. **No spot instances** — 35h training jobs have high interruption risk
3. **Run direct fine-tune baseline in parallel** ($12 extra, saves decision time)

---

## 7. Recommended Action Plan

### Immediate (while v35d runs, $0)

1. Run zero-shot baselines: YOLO26s-pose + RTMPose on skating val ($0.20)
2. Prepare RTMPose fine-tuning data (YOLO→mmpose JSON conversion, CPU only)
3. Develop pseudo-labeling pipeline for SkatingVerse (CPU only)
4. Study Cross-Representation Distillation paper (ECCV 2024)

### After v35d Epoch 20 ($7)

5. Run direct fine-tuning baseline (50 epochs, no KD)
6. Compare v35d vs baseline vs zero-shot

### After v35d Completes (decision point)

**If AP > 0.50:**
7. Continue to epoch 210 or early stop
8. ONNX export + production integration

**If AP < 0.50:**
7. Stop v35d
8. Start RTMPose Fine-Tuning ($15-25) — recommended
9. OR start pseudo-labeling + retrain ($10-15)
10. Direct fine-tune as cheap comparison ($10-20)

### Parallel Tracks (can run simultaneously)

| Track | Cost | GPU Time | Status |
|-------|------|----------|--------|
| v35d KD | $25-35 | 35h | Running |
| Direct fine-tune baseline | $7-15 | 20-40h | Queue after v35d ep 20 |
| RTMPose data prep | $0 | CPU | Can start now |
| Pseudo-label pipeline | $0 | CPU | Can start now |

---

## 8. Minimum Viable Path to Production

**Cheapest: Direct Fine-Tuning ($10-20)**
- Fine-tune YOLO26s-pose on skating data, no KD
- Expected: ~0.55-0.65 AP on skating (limited by regression paradigm)
- Risk: lowest, standard Ultralytics workflow

**Best accuracy: RTMPose Fine-Tuning ($15-25)**
- Fine-tune RTMPose-m on skating data
- Expected: ~0.76-0.79 COCO AP, +5-8% skating improvement
- Zero pipeline changes, already in production

**Highest upside: Pseudo-Labeling + RTMPose ($25-40)**
- RTMPose fine-tuning + SkatingVerse pseudo-labels
- Expected: ~0.80-0.84 skating AP
- Best long-term trajectory

**If v35d works: KD path ($25-35)**
- YOLO26s-pose with skating KD
- 3x faster inference than RTMO
- Requires new extractor class but minimal downstream changes

---

## Agent Reports (Source Documents)

| Agent | Report | Location |
|-------|--------|----------|
| Results Analyst | v35d Metric Interpretation & Decision Tree | `data/plans/2026-04-25-kd-v35d-results-analysis.md` |
| Alternative Strategies | 7 Alternatives with Ranking | Inline (this synthesis) |
| Production Integration | ONNX, Pipeline, Rollback | Inline (this synthesis) |
| Data & Scaling | Data Gaps, Pseudo-Labeling, Augmentation | Inline (this synthesis) |
| Budget & Timeline | Scenarios, Calendar, Gates | `data/plans/2026-04-25-kd-budget-timeline.md` |

---

## Correction to Agent 4 Finding

**Agent 4 flagged FineFS keypoint ordering mismatch as P0 data corruption.** This was investigated in the previous session and confirmed as **FALSE POSITIVE**. Z-coordinates from FineFS NPZ files perfectly match H3.6M anatomical order. The server-side v2 converter includes correct remapping. FineFS data is clean.
