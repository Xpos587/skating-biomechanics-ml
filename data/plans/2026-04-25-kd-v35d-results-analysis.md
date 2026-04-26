# v35d Results Analysis Framework

**Date:** 2026-04-25
**Author:** Agent 1 (Results Analyst)
**Scope:** Metric interpretation, go/no-go thresholds, decision trees, KD diagnostics for v35d training
**Training status:** v35d JUST STARTED (epoch 0), 3 P0 bugs fixed from v35c

---

## 1. v35d Metric Interpretation Framework

### 1.1 Loss Components

The v35d trainer logs 8 loss values per epoch (6 standard + 2 KD). Understanding what each means is critical.

| Loss Name | Source | Computation | Expected Range (v35d) | Interpretation |
|-----------|--------|-------------|----------------------|----------------|
| `train/box_loss` | PoseLoss26 | GIoU + DFL on detected person bboxes | 1.0-2.5 early, 0.8-1.5 late | Person detection quality. Should decrease steadily. |
| `train/pose_loss` | PoseLoss26 | OKS-based loss on GT keypoints (with sigma) | 1.5-3.0 early, 1.0-2.0 late | GT keypoint accuracy. The primary task loss. |
| `train/kpt_loss` | PoseLoss26 | Keypoint-specific component | 1.0-2.5 early, 0.8-1.5 late | Keypoint confidence calibration. |
| `train/cls_loss` | PoseLoss26 | Classification loss (person vs background) | 0.5-2.0 | **CANARY IN COAL MINE** -- spikes indicate coordinate mismatch. |
| `train/dfl_loss` | PoseLoss26 | Distribution focal loss for bbox regression | 0.8-1.5 | Bbox boundary precision. |
| `train/kd_coord_loss` | DistilPoseTrainer | `MSE(student_lb_norm, teacher_lb_norm) * teacher_conf * kp_weights * B` | 0.0 (epochs 0-2), then 0.001-0.05 | KD coordinate loss in letterbox [0,1] space. **Should be > 0 starting epoch 3.** |
| `train/kd_weight` | DistilPoseTrainer | Progressive schedule: 0.0 (ep 0-2), ramp 0->1 (ep 3-17), 1.0 (ep 18+) | 0.0, then 0.067, 0.133, ... | KD schedule multiplier. **Diagnostic -- should follow schedule exactly.** |

**Key formula:**
```
total_loss = gt_loss + coord_alpha * w_kd * kd_coord_loss
where coord_alpha = 0.05, w_kd = schedule, kd_coord_loss = MSE * B
```

Effective KD contribution to total loss:
- At epoch 4 (w_kd=0.067): `0.05 * 0.067 * kd_coord_loss` = `0.0034 * kd_coord_loss`
- At epoch 18+ (w_kd=1.0): `0.05 * 1.0 * kd_coord_loss` = `0.05 * kd_coord_loss`
- If `kd_coord_loss` ~ 0.01, then KD adds 0.0005 (epoch 4) to 0.0005 (epoch 18) to total loss
- **This is very small** -- by design, KD provides a gentle regularization signal

### 1.2 Validation Metrics

| Metric | COCO Protocol | Expected Range (v35d) | What It Measures |
|--------|--------------|----------------------|------------------|
| `metrics/mAP50(B)` | Box mAP@IoU=0.5 | 0.0 (ep 0-1), 0.2-0.5 (ep 10), 0.5-0.8 (ep 50) | Person detection accuracy at loose threshold |
| `metrics/mAP50-95(B)` | Box mAP@IoU=0.5:0.95 | 0.0 (ep 0-1), 0.05-0.15 (ep 10), 0.15-0.4 (ep 50) | Strict person detection |
| `metrics/mAP50(P)` | Pose mAP@OKS=0.5 | 0.0 (ep 0-1), 0.05-0.2 (ep 10), 0.3-0.7 (ep 50) | **PRIMARY METRIC** -- pose accuracy at loose threshold |
| `metrics/mAP50-95(P)` | Pose mAP@OKS=0.5:0.95 | 0.0 (ep 0-1), 0.01-0.05 (ep 10), 0.05-0.2 (ep 50) | Strict pose accuracy |
| `metrics/P(P)` | Pose precision | 0.0 (ep 0), 0.1-0.3 (ep 10), 0.3-0.6 (ep 50) | How many predicted keypoints are correct |
| `metrics/R(P)` | Pose recall | 0.0 (ep 0), 0.1-0.4 (ep 10), 0.3-0.7 (ep 50) | How many GT keypoints are found |
| `val/box_loss`, `val/cls_loss`, etc. | Validation losses | Should track training losses | **CANARY -- val/cls_loss > 10 = coordinate mismatch** |

### 1.3 Leading vs Lagging Indicators

**Leading indicators** (predict future mAP, visible within 1-3 epochs):
1. `train/pose_loss` -- direct gradient signal for keypoints. Decreasing = learning.
2. `train/kd_coord_loss` -- if decreasing, student is aligning with teacher.
3. `train/cls_loss` -- if stable (< 5.0), coordinate space is correct. If spiking, bugs remain.
4. `kd_weight` -- should follow the schedule. If not, epoch tracking bug remains.

**Lagging indicators** (reflect accumulated learning, visible after 5-10 epochs):
1. `metrics/mAP50(P)` -- the gold standard but noisy early on.
2. `metrics/mAP50-95(P)` -- stricter version, more stable.
3. `metrics/P(P)` and `metrics/R(P)` -- precision/recall balance.

### 1.4 Expected Learning Curve Shape

v35d has a unique multi-phase training schedule. The curve will NOT be monotonic.

```
Phase 1: Head-only warmup (epochs 0-2)
  - kd_weight = 0.0 (no KD)
  - Backbone frozen, only detect/sigma head trains
  - Expected: pose_loss drops 30-50%, box mAP50 rises to 0.15-0.30
  - Pose mAP50: 0.000-0.010 (head barely initialized)

Phase 2: KD ramp + head-only (epochs 3-7)
  - kd_weight ramps 0.067 -> 0.333
  - Backbone still frozen
  - KD signal should be visible (kd_coord_loss > 0)
  - Expected: pose_loss continues dropping, cls_loss STABLE (< 5.0)
  - Pose mAP50: 0.01-0.10

Phase 3: Backbone unfreeze (epoch 8)
  - CRITICAL TRANSITION -- expect brief loss spike (1-2 epochs)
  - Backbone at 0.1x head LR, so disruption should be mild
  - kd_weight continues ramping: 0.4 (ep 8) -> 1.0 (ep 18)
  - Expected: box mAP50 jumps (backbone can now detect), pose mAP50 accelerates
  - Pose mAP50 at epoch 10: 0.05-0.15

Phase 4: Full training (epochs 18-210)
  - kd_weight = 1.0 (sustained)
  - All parameters training with cosine LR
  - Expected: steady improvement, plateau around epoch 100-150
  - Pose mAP50 at epoch 50: 0.30-0.70 (depends on data quality)
  - Pose mAP50 at epoch 100: 0.50-0.80
  - Pose mAP50 at epoch 210: 0.60-0.85

Phase 5: Cosine LR decay (epochs 150-210)
  - LR decays to near-zero
  - Fine-tuning / convergence phase
  - Diminishing returns -- patience=30 may trigger early stop
```

### 1.5 Distinguishing "KD is Working" from "GT Loss is Just Improving"

This is the hardest interpretive question. Three signals differentiate KD effectiveness from GT-only improvement:

1. **Coordinate loss trajectory:** If `kd_coord_loss` decreases from epoch 3 onward, the student is learning to match the teacher's coordinates. This is independent of GT loss.

2. **Relative gain vs no-KD baseline:** The only definitive proof. If no-KD baseline reaches pose mAP50=0.60 at epoch 100 and v35d reaches 0.65, the +0.05 is from KD. Without the baseline, this is unmeasurable.

3. **Per-keypoint AP on skating-specific keypoints:** KD should disproportionately improve keypoints that the GT labels are noisy on (e.g., foot keypoints during jumps, occluded limbs). If all keypoints improve equally, KD is not providing domain-specific signal.

**Proxy indicator (before baseline exists):**
- If `kd_coord_loss` decreases AND `train/pose_loss` decreases faster than expected for the data volume, KD is likely helping.
- If `kd_coord_loss` is flat/zero, KD is not contributing regardless of mAP improvement.

---

## 2. Go/No-Go Decision Tree

### 2.1 Epoch 10 Checkpoint: "KD is Active"

**Context:** KD ramp started at epoch 3, backbone unfreezes at epoch 8. By epoch 10, we should see the first real signal.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **GREEN** | `kd_coord_loss` > 0 AND decreasing; `val/cls_loss` < 5.0; Pose mAP50 > 0.05 | Proceed. KD pipeline is functional. |
| **YELLOW** | `kd_coord_loss` > 0 but not decreasing; OR `val/cls_loss` 5-20 | Monitor 5 more epochs. Coordinate mismatch may still exist but is less severe than v35c. |
| **RED** | `kd_coord_loss` = 0 for all epochs; OR `val/cls_loss` > 50; OR Pose mAP50 = 0.000 | **STOP.** KD activation bug or coordinate mismatch persists. Debug before continuing. |

**Diagnostic checks at epoch 10:**
```bash
# Check KD activation logs -- should see:
# [KD] Epoch 3, warmup=3, kd_weight=0.0667
# [KD] Epoch 4, warmup=3, kd_weight=0.1333
# ...
# [KD] Epoch 8, warmup=3, kd_weight=0.3333
# >>> Epoch 8: Unfreezing backbone with differential LR
# [KD] Epoch 10, warmup=3, kd_weight=0.4667

# Check teacher data loading:
# WARNING: teacher coords not found for ...  <-- should be RARE (< 1%)
```

### 2.2 Epoch 20 Checkpoint: "Is the Approach Viable?"

**Context:** KD at full ramp (w_kd = 0.8). Backbone has been training for 12 epochs. This is the primary viability checkpoint.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **GREEN** | Pose mAP50 > 0.15; `kd_coord_loss` decreasing; `val/cls_loss` < 5.0; box mAP50 > 0.40 | **Proceed.** Learning trajectory is healthy. |
| **YELLOW** | Pose mAP50 0.05-0.15; losses decreasing but slowly | **Monitor to epoch 50.** May need: (a) higher coord_alpha, (b) more epochs, (c) data quality check. |
| **RED** | Pose mAP50 < 0.05; OR losses not decreasing for 5+ epochs; OR `val/cls_loss` spiking > 20 | **STOP AND DIAGNOSE.** Something is fundamentally wrong. See Section 6. |

**v35c comparison baseline:**
- v35c at epoch 6 (bugs): Pose mAP50 = 0.069, val/cls_loss = 3.53 (after 142 spike)
- v35d at epoch 20 (bugs fixed): Should be significantly better than v35c epoch 6

**Why epoch 20 matters:** At 210 total epochs, epoch 20 = 9.5% of training. If pose mAP50 is still < 0.05 by now, the remaining 190 epochs are unlikely to rescue the run (based on typical YOLO fine-tuning convergence patterns: 80% of final mAP is achieved by 20% of training).

### 2.3 Epoch 50 Checkpoint: "Convergence Trajectory"

**Context:** Full KD active for 32 epochs. Backbone unfrozen for 42 epochs. Cosine LR still high. Should be past initial learning phase.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **GREEN** | Pose mAP50 > 0.30; steady improvement trajectory (mAP50 increased > 0.10 since epoch 20) | **Proceed to full training.** On track for production-quality model. |
| **YELLOW** | Pose mAP50 0.15-0.30; improving but slowly | **Consider tuning.** Options: (a) increase coord_alpha to 0.10, (b) check if learning rate is too high/low, (c) verify data balance. |
| **RED** | Pose mAP50 < 0.15; OR plateau (no improvement for 10+ epochs) | **EVALUATE.** Either: (a) data quality problem (FineFS labels bad), (b) architecture mismatch (31M -> 9M too aggressive), (c) KD hurting (coord_alpha too high). Run ablation: disable KD for 10 epochs and compare. |

**Realistic expectations grounded in literature:**

| Method | Data | Epochs | Pose AP (mAP50) |
|--------|------|--------|-----------------|
| DWPose stage 1 (RTMPose-l -> RTMPose-l) | COCO-WholeBody (UBody) | 300 | 65.3 -> 66.5 |
| DistilPose (HRNet-X -> token student) | COCO | 200 | Teacher: 72.5, Student: 71.6 |
| YOLOv8-pose fine-tune (custom data) | 50K images | 100 | 0.60-0.80 mAP50 (community reports) |
| YOLO26-pose pretrain | COCO | 300 | ~0.57 mAP50 (pretrained) |

Our setting is harder: fine-tuning on ~50K images (much less than COCO's 118K), with domain shift from generic person poses to figure skating. Expect lower absolute AP but meaningful KD contribution.

### 2.4 Epoch 100 Checkpoint: "Production Readiness"

**Context:** ~48% of training complete. Cosine LR starting to decay significantly. Model should be approaching final quality.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **GREEN** | Pose mAP50 > 0.50; kd_coord_loss stable/low | **Continue.** Model is production-viable. Monitor for early stopping. |
| **YELLOW** | Pose mAP50 0.30-0.50; still improving | **Continue with monitoring.** May need full 210 epochs. |
| **RED** | Pose mAP50 < 0.30; OR degradation from epoch 50 peak | **STOP.** Overfitting or catastrophic forgetting. Check: (a) COCO ratio, (b) LR schedule, (c) data leakage. |

### 2.5 Epoch 210 (Final): "Production Evaluation"

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **GREEN** | Pose mAP50 > 0.60; Pose mAP50-95 > 0.15; box mAP50 > 0.70 | **DEPLOY.** Evaluate on held-out test set (AP3D-FS test: 20,324 images). Compare with RTMO baseline. |
| **YELLOW** | Pose mAP50 0.40-0.60 | **Evaluate vs RTMO.** If competitive on skating data despite lower COCO mAP, may still be useful. Consider: (a) more training data, (b) DistilPose-style heatmap KD (v36). |
| **RED** | Pose mAP50 < 0.40 | **PIVOT.** Direct fine-tuning or RTMPose fine-tuning likely superior. KD from MogaNet-B did not transfer effectively to YOLO26 architecture. |

---

## 3. v35d vs v35c Comparison Framework

### 3.1 Bug Fixes and Expected Improvements

| Bug | v35c Symptom | v35d Fix | How to Verify |
|-----|-------------|----------|---------------|
| **P0-1: KD activation** | `kd_weight=0` for ALL 6 epochs | Fallback to `trainer_ref.epoch` in `kd_loss()` | Check logs: `[KD] Epoch N, warmup=3, kd_weight=X.XXXX` -- should show non-zero from epoch 3 |
| **P0-2: Coordinate space mismatch** | `val/cls_loss` spikes 142+ at epochs 4-5 | Teacher coords transformed crop -> orig [0,1] -> letterbox [0,1] | `val/cls_loss` should stay < 10.0 for all epochs |
| **P0-3: Per-image normalization** | Used first image dims for entire batch | Normalization uses letterbox `lb_h, lb_w` from `batch["img"].shape[2:]` | Consistent loss values across batches with different aspect ratios |

### 3.2 Side-by-Side Comparison Table

| Metric | v35c Epoch 6 (buggy) | v35d Expected Epoch 6 | Verification |
|--------|----------------------|----------------------|--------------|
| `kd_weight` | 0.0 (BUG) | 0.2 (schedule: (6-3)/15 = 0.2) | Log check |
| `kd_coord_loss` | 0.0 (BUG) | > 0 (KD active) | Log check |
| `val/cls_loss` | 3.53 (after 142 spike) | < 5.0 (stable) | Results table |
| Pose mAP50 | 0.069 | 0.01-0.05 (head-only, 3 epochs of KD) | Results table |
| Box mAP50 | 0.355 | 0.15-0.30 (backbone still frozen) | Results table |

**Note:** v35d epoch 6 has backbone frozen (unfreeze at epoch 8), so box mAP50 may be LOWER than v35c epoch 6 because v35c had backbone unfrozen from the start. This is expected and correct behavior.

### 3.3 Verification Checklist per Bug Fix

**Bug 1 (KD activation) -- verify at epoch 3+:
```python
# In training log, should see:
# [KD] Epoch 3, warmup=3, kd_weight=0.0667
# train_kd_coord_loss column should be > 0
# train_kd_weight column should match schedule
```

**Bug 2 (Coordinate mismatch) -- verify at epochs 3-6:
```python
# val_cls_loss should be in range [0.5, 5.0]
# v35c had: epoch 4 = 142.9, epoch 5 = 147.0
# v35d should have: ALL epochs < 10.0
# If spike appears: coordinate transform still broken
```

**Bug 3 (Per-image normalization) -- verify across all epochs:
```python
# No single batch should have dramatically different loss
# train_pose_loss should be smooth (no erratic jumps)
# If batch-to-batch variance is high: normalization issue remains
```

---

## 4. KD-Specific Diagnostics

### 4.1 Additional Logging Recommendations

The current trainer logs `kd_coord_loss` and `kd_weight`. To fully diagnose KD health, add these:

| Metric | Computation | Purpose |
|--------|-------------|---------|
| `kd_coord_loss_per_kp` | Per-keypoint MSE (17 values) | Identify which keypoints student struggles with |
| `kd_teacher_match_rate` | Fraction of batch where teacher data was found | Detect HDF5 index misses |
| `kd_coord_loss_weighted` | `coord_alpha * w_kd * kd_coord_loss` | Actual KD contribution to total loss |
| `gt_loss_component` | Total loss minus KD contribution | Monitor GT vs KD balance |
| `kd/student_teacher_dist` | Mean Euclidean distance student->teacher (in pixels) | Direct measure of alignment quality |

**Implementation** (minimal change to `kd_loss()`):
```python
# After computing coord_loss, add:
if self._last_logged_epoch != self._current_epoch:
    # Log per-keypoint loss (already computed as per_kp_loss)
    mean_per_kp = per_kp_loss.mean(dim=0)  # (K,)
    print(f"[KD] Per-kp loss: {mean_per_kp.detach().cpu().numpy().round(4)}")

    # Log teacher match rate
    match_rate = valid_cp.float().mean().item()
    print(f"[KD] Teacher match rate: {match_rate:.3f}")

    # Log effective KD weight
    effective = self.coord_alpha * w_kd * coord_loss.item()
    print(f"[KD] Effective KD loss: {effective:.6f} (alpha={self.coord_alpha}, w={w_kd:.4f})")
```

### 4.2 Detecting Failure Modes

#### Mode Collapse
**Symptoms:** `kd_coord_loss` drops to near-zero while `train/pose_loss` stays high or increases.
**Meaning:** Student is optimizing for teacher coordinates but ignoring GT keypoints entirely.
**Cause:** `coord_alpha` too high relative to GT loss.
**Fix:** Reduce `coord_alpha` from 0.05 to 0.01-0.02.

**Detection threshold:** If `coord_alpha * w_kd * kd_coord_loss > 0.5 * gt_loss` for 5+ epochs, KD is dominating.

#### Teacher Overfitting
**Symptoms:** `kd_coord_loss` approaches 0 while validation mAP plateaus or drops.
**Meaning:** Student memorizes teacher coordinates but doesn't generalize.
**Cause:** No augmentation mismatch between teacher (unaugmented crops) and student (augmented full images).
**Fix:** Add augmentation to teacher data, or reduce KD weight.

**Detection threshold:** If `kd_coord_loss` < 0.001 and val mAP hasn't improved in 10 epochs.

#### Catastrophic Forgetting
**Symptoms:** Pose mAP50 on validation drops after initial rise; `train/pose_loss` continues decreasing.
**Meaning:** Student overfits to training distribution (skating poses) and loses general person detection.
**Cause:** Insufficient COCO mix (currently ~12.8% = 5,659 / 44,609).
**Fix:** Increase COCO mix to 20-30%.

**Detection threshold:** If Pose mAP50 drops > 10% from peak for 5+ epochs while train loss decreases.

#### Coordinate Drift
**Symptoms:** `val/cls_loss` gradually increases (not spiking); Pose mAP50 plateaus early.
**Meaning:** Student and teacher coordinate spaces slowly diverge due to residual transform bug.
**Cause:** Letterbox transform not perfectly matching Ultralytics internal transform.
**Fix:** Debug-print actual student and teacher coords for 5 sample images, verify they align.

**Detection threshold:** If `val/cls_loss` trend is upward over 10+ epochs (even gradual).

### 4.3 Recommended Visualizations

1. **Loss curves** (train + val, all components):
   - Plot 1: `train/pose_loss`, `train/kd_coord_loss` (dual axis -- different scales)
   - Plot 2: `val/cls_loss` with threshold line at 10.0 (alarm)
   - Plot 3: `train/kd_weight` over epochs (should follow schedule exactly)

2. **mAP curves** (validation):
   - Plot 1: `Pose mAP50` and `Pose mAP50-95` over epochs
   - Plot 2: `Box mAP50` over epochs (should track ahead of pose)
   - Plot 3: `Pose P` and `Pose R` (precision-recall balance)

3. **v35d vs v35c overlay** (for the first 6 epochs):
   - Direct comparison of Pose mAP50 trajectory
   - val/cls_loss comparison (v35c should show spikes, v35d should be smooth)

4. **Per-keypoint AP** (if extractable from Ultralytics validation):
   - Bar chart at epochs 20, 50, 100, 210
   - Compare with teacher's per-keypoint AP
   - KD should help most on keypoints where GT labels are noisy (feet, hands)

5. **Teacher-Student coordinate comparison** (offline analysis):
   - Sample 100 validation images
   - Plot teacher coords vs student coords (scatter plot, x/y separately)
   - Perfect alignment = points on y=x diagonal
   - Systematic offset = coordinate transform bug

---

## 5. Comparison Baselines Needed

### 5.1 Essential Baselines

| Baseline | Purpose | How to Run | Expected Time |
|----------|---------|------------|---------------|
| **No-KD fine-tune** | Measure KD contribution | Same data.yaml, same hyperparams, `--teacher-coords` omitted | Same as v35d (210 epochs) |
| **YOLO26s-pose zero-shot** | Lower bound | `yolo val model=yolo26s-pose.pt data=data.yaml` | 10 min |
| **RTMPose/RTMO on same val** | Production comparison | `rtmlib` inference on val set | 30 min |
| **Teacher (MogaNet-B) on val** | Upper bound | Run teacher coords extraction on val set, compute AP | Already done (~95% on AP3D) |

### 5.2 Running Baselines Efficiently

**Strategy: Run no-KD baseline IN PARALLEL with v35d.**

Both use the same GPU instance. Since v35d is the primary run:
1. Start v35d training (primary, uses most GPU memory).
2. Queue the no-KD baseline to start after v35d epoch 20 checkpoint.
3. If v35d passes epoch 20 go/no-go, start no-KD baseline on same instance (or separate instance).

**Alternative: Use early-stop on no-KD baseline.**
- Run no-KD for 50 epochs only (not 210).
- If no-KD mAP50 at epoch 50 is close to v35d at epoch 50, KD is not helping.
- If no-KD is significantly lower, KD is adding value.

**GPU budget for baselines:**
| Baseline | GPU Hours | Cost (RTX 4090 @ $0.28/hr) |
|----------|-----------|---------------------------|
| Zero-shot val | 0.2h | $0.06 |
| RTMO val | 0.5h | $0.14 |
| No-KD 50 epochs | ~25h | $7.00 |
| No-KD 210 epochs | ~100h | $28.00 |
| **Total** | **~126h** | **~$35.20** |

**Recommendation:** Run zero-shot + RTMO val immediately ($0.20 total). Run no-KD 50 epochs only if v35d passes epoch 20 ($7.00). Skip no-KD 210 unless needed for paper/publication.

### 5.3 What Each Baseline Tells Us

| Comparison | Interpretation |
|------------|----------------|
| v35d > no-KD by > 0.03 mAP | KD is working. Worth the engineering effort. |
| v35d = no-KD (within 0.01 mAP) | KD is not helping. coord_alpha may need tuning, or teacher knowledge doesn't transfer. |
| v35d < no-KD | KD is hurting. coord_alpha too high, or coordinate transform bug remains. |
| v35d > RTMO on skating val | Production-ready. Switch from RTMPose to YOLO26s-pose. |
| v35d < RTMO but v35d is 3x faster | Deployment trade-off decision. Speed vs accuracy. |
| v35d > zero-shot by > 0.20 mAP | Fine-tuning is effective regardless of KD. |
| v35d close to teacher (~0.95 AP) | Near-perfect knowledge transfer. Unlikely but aspirational. |

---

## 6. Scenario Analysis

### 6.1 Scenario A: Great Results (Pose mAP50 > 0.70 at epoch 50)

**Probability estimate:** 15-25% (based on data quality uncertainty)

**Immediate actions:**
1. Continue training to epoch 100. If mAP50 > 0.75 at epoch 100, consider early stopping.
2. Run zero-shot and RTMO baselines on validation set.
3. If v35d > RTMO on skating data, begin production integration planning.
4. Export to ONNX, benchmark inference speed on RTX 3050 Ti.

**Production path:**
- Replace RTMPose/RTMO in `ml/src/pose_estimation/rtmlib_extractor.py` with YOLO26s-pose
- Update pipeline: `extract_video()` -> YOLO26s-pose -> H3.6M 17kp
- Verify tracking (PoseTracker) works with new pose format
- Benchmark end-to-end pipeline speed

**Risk:** High mAP on validation may not translate to production quality (distribution shift from competition videos to phone videos).

### 6.2 Scenario B: Mediocre Results (Pose mAP50 0.30-0.70 at epoch 50)

**Probability estimate:** 50-60% (most likely scenario)

**Diagnostic steps:**
1. Check `kd_coord_loss` trajectory -- is it decreasing? If yes, KD is helping but needs more time/tuning.
2. Compare per-keypoint AP -- which keypoints are weak? (Likely: feet, wrists during rotations)
3. Run no-KD baseline at epoch 50 to isolate KD contribution.
4. Check training data balance: FineFS vs AP3D-FS vs COCO ratios.

**Tuning options (in order of impact):**
1. **Increase coord_alpha** from 0.05 to 0.10-0.15 -- stronger KD signal
2. **Longer KD ramp** -- extend from 15 to 30 epochs for gentler introduction
3. **Add bone-length consistency loss** -- biomechanical constraint (from v35c review ARCH-3)
4. **DistilPose-style simulated heatmap KD (v36)** -- expected +1.0-2.0 AP per paper
5. **Increase COCO mix** to 20% -- prevent catastrophic forgetting
6. **Data augmentation** for skating: rotation, motion blur, ice glare

**Decision point at epoch 100:**
- If Pose mAP50 > 0.50: Continue to epoch 210, likely viable for production with compromises.
- If Pose mAP50 0.30-0.50: Consider switching to direct fine-tuning (simpler, possibly better).

### 6.3 Scenario C: Poor Results (Pose mAP50 < 0.30 at epoch 50)

**Probability estimate:** 20-30%

**Root cause analysis checklist (execute in order):**

1. **Is KD actually active?**
   - Check `kd_weight` column -- should be 1.0 by epoch 18
   - Check `kd_coord_loss` -- should be > 0
   - Check training logs for `[KD] Epoch N` messages
   - If KD is inactive: epoch tracking bug persists

2. **Is the data loading correctly?**
   - Check `kd_teacher_match_rate` (if implemented) -- should be > 0.90
   - Verify HDF5 index covers all training images
   - Check for path mismatches between `batch["im_file"]` and HDF5 index keys

3. **Is coordinate transform correct?**
   - Check `val/cls_loss` -- should be < 10.0 for ALL epochs
   - If cls_loss spikes: coordinate space mismatch
   - Debug: print student_xy_norm and teacher_lb for 5 samples, compare visually

4. **Is the data quality sufficient?**
   - FineFS: 8,904 images with potentially wrong keypoint ordering (P0-4 from audit)
   - AP3D-FS: 35,705 images from 3D->2D projection (should be good)
   - COCO: 5,659 images (generic poses, not skating)
   - If FineFS has wrong keypoint order, 20% of data is corrupted

5. **Is the architecture mismatch too large?**
   - Teacher: 31M params, heatmap-based, top-down
   - Student: 9M params, direct regression, bottom-up
   - Cross-family KD (MogaNet -> YOLO) is harder than same-family (YOLO26l -> YOLO26n)

**When to pivot vs debug:**
- **Debug** if: any of checks 1-4 above fail. Fix and restart.
- **Pivot** if: all checks pass but mAP is still < 0.30. The approach may be fundamentally limited.
- **Pivot options:**
  - Direct fine-tuning (no KD) -- simpler, possibly better for 9M param model
  - Same-family KD (YOLO26l-pose -> YOLO26s-pose) -- avoids architecture mismatch
  - RTMPose fine-tuning -- proven, but heavier model
  - Pseudo-labeling SkatingVerse (28K videos) -- more data may help more than KD

---

## 7. Summary: Critical Numbers to Watch

| Epoch | Pose mAP50 GREEN | Pose mAP50 YELLOW | Pose mAP50 RED | val/cls_loss Alarm |
|-------|-----------------|-------------------|---------------|-------------------|
| 6 | > 0.02 | 0.005-0.02 | < 0.005 | > 10.0 |
| 10 | > 0.05 | 0.02-0.05 | < 0.02 | > 10.0 |
| 20 | > 0.15 | 0.05-0.15 | < 0.05 | > 10.0 |
| 50 | > 0.30 | 0.15-0.30 | < 0.15 | > 10.0 |
| 100 | > 0.50 | 0.30-0.50 | < 0.30 | > 10.0 |
| 210 | > 0.60 | 0.40-0.60 | < 0.40 | > 10.0 |

| Loss Component | GREEN | YELLOW | RED |
|---------------|-------|--------|-----|
| `kd_coord_loss` | Decreasing, > 0 | Flat but > 0 | = 0 (KD inactive) |
| `kd_weight` | Follows schedule exactly | Close to schedule | Doesn't match schedule |
| `val/cls_loss` | < 5.0 always | 5.0-10.0 | > 10.0 any epoch |
| `train/pose_loss` | Steady decrease | Slow decrease | Increasing or flat |

---

## Appendix A: KD Schedule Reference

| Epoch | kd_weight | Backbone | Expected Behavior |
|-------|-----------|----------|-------------------|
| 0 | 0.0 | Frozen | Random predictions, GT loss only |
| 1 | 0.0 | Frozen | Head learning basics |
| 2 | 0.0 | Frozen | Head improving |
| 3 | 0.067 | Frozen | KD activates (first signal) |
| 4 | 0.133 | Frozen | KD growing |
| 5 | 0.200 | Frozen | KD growing |
| 6 | 0.267 | Frozen | KD growing |
| 7 | 0.333 | Frozen | KD growing |
| 8 | 0.400 | **UNFREEZE** | Backbone disruption possible |
| 9 | 0.467 | Unfreezing | Recovery from unfreeze |
| 10 | 0.533 | Unfreezing | Stabilizing |
| 11 | 0.600 | Training | Normal training |
| 12 | 0.667 | Training | Normal training |
| 13 | 0.733 | Training | Normal training |
| 14 | 0.800 | Training | Normal training |
| 15 | 0.867 | Training | Normal training |
| 16 | 0.933 | Training | Normal training |
| 17 | 1.000 | Training | Full KD reached |
| 18-210 | 1.000 | Training | Full KD sustained |

---

## Appendix B: Files to Monitor

| File | What to Check |
|------|---------------|
| Training logs (stdout) | `[KD] Epoch N` messages, `>>> Unfreezing backbone` |
| `results.csv` (Ultralytics output) | All metrics per epoch, loss components |
| `args.yaml` (Ultralytics output) | Verify hyperparams match config |
| GPU utilization | Should be > 80% during training, > 60% during validation |
