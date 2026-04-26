# KD Pipeline Full Audit Report

**Date:** 2026-04-25
**Scope:** 5-agent deep review of MogaNet-B → YOLO26-Pose KD pipeline
**Agents:** Loss Mechanics, Data Pipeline, Architecture & Training, Pose Domain, Strategic Roadmap

---

## Executive Summary

Audit found **5 P0 critical issues**, **11 P1 important issues**, and **15+ P2/P3** findings. Most critical: coordinate space mismatch affects every KD gradient, and FineFS keypoint ordering may corrupt 20% of training data. v35c training shows KD is completely inactive (kd_weight=0 for 6 epochs) and has val/cls_loss spikes (142+), indicating bugs in the KD schedule and/or coordinate handling.

**Teacher correction:** Agent 5 incorrectly stated MogaNet-B is COCO-only. In fact, it is `moganet_b_ap2d_384x288.pth` fine-tuned on AthletePose3D (includes FS-Jump3D skating subset) with ~95% accuracy. The teacher HAS skating-specific knowledge. KD from this teacher to YOLO26-Pose Small is a well-motivated approach.

**Immediate actions:**
1. Investigate why KD is not activating (kd_weight=0 for 6 epochs)
2. Fix val/cls_loss spikes (142+ at epochs 4-5)
3. Verify FineFS keypoint ordering (potential data corruption)
4. Fix coordinate space mismatch before restarting training

---

## v35c Training Status (Checked 2026-04-25)

**Progress:** Epoch 7/210, ~2.9 it/s, GPU 8.24G, batch=128, imgsz=384

### Validation Metrics

| Epoch | Box mAP50 | Pose mAP50 | Pose mAP50-95 | Pose P | Pose R | val/cls_loss |
|-------|-----------|------------|---------------|--------|--------|-------------|
| 1 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| 2 | 0.150 | 0.006 | 0.001 | 0.055 | 0.021 | 3.25 |
| 3 | 0.285 | 0.023 | 0.004 | 0.115 | 0.071 | 2.75 |
| 4 | 0.088 | 0.005 | 0.001 | 0.027 | 0.121 | **142.9** |
| 5 | 0.101 | 0.015 | 0.003 | 0.064 | 0.127 | **147.0** |
| 6 | 0.355 | 0.069 | 0.014 | 0.158 | 0.131 | 3.53 |

### Critical Anomalies

1. **KD coord_loss = 0 for ALL 6 epochs** — KD is completely inactive. The progressive schedule should start at epoch 4 (warmup_end=3) but kd_weight remains 0. Bug in schedule code or gating logic.

2. **val/cls_loss spikes at epochs 4-5**: 142.9 and 147.0 (normal range < 5.0). Likely caused by coordinate space mismatch (P0-1) corrupting loss computation after warmup phase.

3. **Pose mAP50 = 0.069 at epoch 6** — very low. Normal fine-tuning should reach 0.3+ by epoch 6 on similar data.

### Conclusion: v35c is NOT producing useful results. Training should be stopped and bugs fixed before restarting.

---

## P0 Findings (Critical — Correctness)

### P0-1: Coordinate Space Mismatch
**Sources:** Agents 1, 2, 3, 4, 5 (confirmed by all 5)

Student keypoints decoded to letterboxed pixel space, normalized to [0,1] using letterbox dimensions. Teacher coords in original image [0,1] space. For non-square images (16:9 phone video), KD loss compares coordinates in different coordinate systems.

- `batch["img_shape"]` key never exists in Ultralytics → fallback always uses letterbox dimensions (384×384)
- `_decode_student_kpts` applies stride multiplication producing pixel coords, then divides by letterbox `img_w`/`img_h`
- Teacher coords extracted from heatmaps in crop [0,1] space with separate crop_params

**Impact:** Systematic offset proportional to aspect ratio difference. Every KD gradient is wrong for non-square images.

### P0-2: `_decode_student_kpts` Stride Multiplication Uncertainty
**Source:** Agent 3

The decode formula `(pred + anchor) * stride` may double-count stride if `make_anchors(feats, stride, 0.5)` already returns pixel-space anchors. PoseLoss26.kpts_decode does NOT multiply by stride (operates in anchor-relative space). The KD trainer's formula matches the Pose26 head inference path, not the training loss path.

**Impact:** If stride is double-counted, student coords used for KD are wrong at all pyramid levels.

**Fix:** Debug-print actual anchor values during training to determine if stride multiplication is correct.

### P0-3: Per-Image Normalization Uses First Image's Dimensions
**Source:** Agent 4

```python
img_h = img_shapes[0, 0].item()  # First image only
img_w = img_shapes[0, 1].item()
student_xy_norm[..., 0] /= img_w  # Applied to ALL images
```

With `rect=True`, batches contain images of different aspect ratios. Using first image's dimensions for all introduces systematic errors.

**Impact:** Every image except the first in each batch gets wrong normalization.

### P0-4: FineFS Keypoint Ordering May Not Match COCO/H3.6M
**Source:** Agent 4

FineFS native order: `hip_center, left_hip, right_hip, spine, knee_l, knee_r, ...`
H36Key/COCO order: `HIP_CENTER, RHIP, RKNEE, RFOOT, LHIP, LKNEE, ...`

Converter scripts pass keypoints without remapping. Plan document (line 180) flagged this as blocker but task was never completed.

**Impact:** If mismatched, 8,904 FineFS images (20% of training data) teach wrong keypoint associations. Right knee GT is actually spine, left hip GT is actually right hip, etc.

**Fix:** Load a FineFS NPZ file, compare landmark positions against expected anatomy. If order differs, add permutation remap.

### P0-5: Heatmap Raw Values May Exceed [0,1]
**Source:** Agent 4

Generation script stores raw DeconvHead output (peak ~2-6, no sigmoid). Verification script checks `max_val <= 1.0` and would reject valid heatmaps. Confidence values exceed 1.0 but KD trainer clamps at `min=0.1` (fine).

**Impact:** Verification may falsely report failure. Confidence values not in expected range for downstream consumers.

---

## P1 Findings (Important)

### P1-1: `__setstate__` Doesn't Re-create TeacherCoordLoader
**Sources:** Agents 1, 2

After EMA deepcopy, `_coord_loader` is None. If `kd_loss` is called before re-initialization, teacher coords cannot be loaded.

### P1-2: Student Bbox Uses All Keypoints Including Invisible
**Sources:** Agents 1, 3

```python
sx = student_kpts[..., 0]  # All keypoints, including invisible
student_x1 = sx.min(dim=-1).values
```

Invisible keypoints may have arbitrary (x,y), dominating the bbox and causing wrong anchor selection.

**Fix:** Filter by visibility (sigmoid > 0.5) or use detection head bbox predictions.

### P1-3: Biomechanical Weight Left/Right Asymmetry
**Sources:** Agents 1, 4

| Keypoint | Left | Right | Expected |
|----------|------|-------|----------|
| SHOULDER | 1.0 | **0.5** | 1.0 |
| ELBOW | 0.8 | **0.5** | 0.8 |
| WRIST | 0.8 | **1.5** | 0.8 |

Right arm underweighted 50-83%. Most skaters rotate counter-clockwise (lead with right side), so right arm accuracy disproportionately affects jump analysis.

### P1-4: FineFS Black Placeholder Images
**Source:** Agent 4

Original converter creates 640×640 black images (FineFS has no video frames). If teacher heatmaps generated on black images, MogaNet-B produces garbage predictions.

**Fix:** Verify which converter version is active. If black images, re-run with `convert_finefs_video.py` or exclude FineFS.

### P1-5: HDF5 File Handle Not Fork-Safe
**Sources:** Agent 3, CLAUDE.md protocol

Single HDF5 handle shared across forked DataLoader workers. Can cause silent corruption or crashes.

**Fix:** Open HDF5 per-call with per-process cache, or switch to LMDB (per CLAUDE.md protocol).

### P1-6: Teacher Crop 20% Padding Clips Wide Skating Poses
**Source:** Agent 4

Spread eagle, Ina Bauer, camel push keypoints near image edges. 20% padding insufficient, crop clamping shifts coordinates.

**Fix:** Increase padding to 40% or use Ultralytics letterbox approach.

### P1-7: No Temporal Consistency in KD
**Source:** Agent 4

Frames from same video treated independently. Jitter corrupts downstream phase detection and DTW alignment.

**Fix:** Add temporal smoothness regularization: `L_temporal = MSE(kpts[t] - kpts[t-1])` with curriculum sampler keeping adjacent frames together.

### P1-8: EMA Deepcopy + Checkpoint Resume
**Source:** Agent 3

EMA works (never calls loss), but checkpoint resume may not re-patch `model.loss` to `kd_loss`.

### P1-9: RealNVP Flow Not Considered in Decode
**Source:** Agent 3

Currently safe (flow not applied during training), but fragile — future Ultralytics update could break it.

---

## P2 Findings (Minor)

| ID | Source | Finding |
|----|--------|---------|
| P2-1 | Agent 3 | Backbone unfreeze inside loss function — optimizer rebuilt mid-epoch |
| P2-2 | Agent 3 | KD schedule too weak during head-only phase (effective alpha 0.003-0.015) |
| P2-3 | Agent 3 | No KD-specific validation metric |
| P2-4 | Agent 3 | Teacher heatmaps only process first person in multi-person labels |
| P2-5 | Agent 3 | O(n*m) identity check in `_rebuild_optimizer` |
| P2-6 | Agent 4 | 72×96 heatmap resolution limits sub-pixel accuracy (~±4px for wrists) |
| P2-7 | Agent 4 | COCO 10% dilutes skating-specific learning (12.8% of data, no skating poses) |
| P2-8 | Agent 4 | AP3D-FS is 100% jumps — no spins, steps, lifts |
| P2-9 | Agent 4 | Visibility handling inconsistent across datasets |
| P2-10 | Agent 2 | Symlink path issues in dataset loader |
| P2-11 | Agent 1 | Student confidence not used in KD loss weighting |

---

## Strategic Analysis (Revised)

### Teacher Model: MogaNet-B on AthletePose3D (NOT COCO-only)

> **Correction to Agent 5's analysis:** MogaNet-B is `moganet_b_ap2d_384x288.pth`, fine-tuned on AthletePose3D (includes FS-Jump3D skating subset with 4 skaters × 7 jump types). ~95% accuracy on AP3D validation. The teacher has genuine skating-specific knowledge.

KD from a domain-specific teacher (MogaNet-B, 31M params, heatmap-based, ~95% on skating) to a lightweight student (YOLO26-Pose Small, 9M params, direct regression) is a well-motivated approach:
- **31M → 9M compression** (3.4× smaller)
- **Top-down heatmap → bottom-up regression** (different inference paradigm, potentially faster)
- **Domain transfer** (skating-specific knowledge preserved in lighter model)

### Remaining Strategic Concerns

| Concern | Status |
|---------|--------|
| Teacher accuracy on skating | Good (~95% on AP3D) |
| Architecture mismatch (top-down vs bottom-up) | Real — causes coordinate bugs |
| No baseline vs RTMO | Still missing — should measure |
| Production integration path | Non-trivial (tracking rework) |

### Revised Action Plan (Priority-Ranked)

1. **Fix P0 bugs** — coordinate space mismatch, KD schedule, stride decode
2. **Verify FineFS keypoint order** — potential data corruption for 20% of data
3. **Fix KD activation** — kd_weight=0 for 6 epochs is a bug, not a schedule choice
4. **Restart v35c** — with fixed bugs, monitor val/cls_loss stability
5. **Measure RTMO baseline** — needed for comparison regardless of KD outcome
6. **Direct fine-tuning baseline** — run in parallel to isolate KD contribution

### Alternative Approaches (Complementary, Not Replacement)

| Approach | Cost | Risk | Role |
|----------|------|------|------|
| Direct fine-tuning (no KD) | $15 | Low | Baseline for measuring KD contribution |
| RTMPose fine-tuning + TensorRT | $20 | Low | Alternative if YOLO26 underperforms |
| Pseudo-labeling SkatingVerse (28K videos) | $35 | Medium | Data augmentation for any approach |
| Same-family KD (YOLO26l → YOLO26n) | $30 | Medium | If cross-family KD proves too buggy |

---

## Deduplicated Finding Count

| Severity | Unique findings | Agents reporting |
|----------|----------------|-----------------|
| P0 | 5 | 2-5 agents each |
| P1 | 9 | 1-3 agents each |
| P2 | 11 | 1 agent each |
| P3 | 5+ | 1 agent each |
| Strategic | 6 recommendations | Agent 5 |
| **Total** | **36** | |

---

## Top 6 Actions by Impact

1. **Fix KD activation bug** — kd_weight=0 for 6 epochs, schedule code needs investigation
2. **Fix coordinate space mismatch (P0-1)** — systematic error in every KD gradient
3. **Verify FineFS keypoint ordering (P0-4)** — if wrong, 20% of training data is corrupted
4. **Investigate val/cls_loss spikes** — 142+ at epochs 4-5, likely related to P0 bugs
5. **Verify stride decode formula (P0-2)** — debug-print anchors during training
6. **Fix per-image normalization (P0-3)** — first-image-dimensions used for entire batch
