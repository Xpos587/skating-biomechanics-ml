# Implementation Plan: KD MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Design spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
**Created:** 2026-04-18
**Branch:** `experiments/yolo26-pose-kd`

---

## Task 0: Training Time Estimation

Calculate expected training time using published benchmarks.

**Reference:** Ultralytics docs — **2.8s compute per 1000 images on RTX 4090** (YOLO26n, baseline).

**Full pipeline cost (RTX 4090 $0.295/hr, RTX 5090 $0.305/hr, KD overhead 2.0x):**

| Dataset | Stage 2 (ablation) | Stage 2.5 (teacher) | Stage 3 (KD) | Stage 4 (sizes) | Total | Cost (5090) | Cost (4090) |
|---------|-------------------|--------------------|--------------------|--------------------|-------|-------------|-------------|
| ~343K (expected) | 32h | 8h | 53h | 123h | 216h | $66 | $64 |

**Conclusion:** $150 budget is NOT a limiting factor. Expected ~343K images × full pipeline = $64-66. Dataset size driven by domain coverage (FineFS + FSAnno), not budget.

**VRAM budget (teacher + student):**
- YOLO26n + MogaNet-B (eval): ~8-10GB → RTX 4090 24GB OK
- YOLO26s + MogaNet-B (eval): ~12-14GB → RTX 4090 24GB OK
- YOLO26m + MogaNet-B (eval): ~18-20GB → RTX 4090 24GB tight (batch=16)

**Actions:**
- [ ] After data conversion (Task 3-5): record actual N_images
- [ ] Recalculate using table above with real N_images
- [ ] Verify RTX 4090 24GB is sufficient (if using YOLO26m, plan batch=16)

**Validation:** Total cost < $150 (already confirmed for any realistic N).

---

## Task 1: Project Structure & Dependencies

Create experiment directory and install dependencies.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/` with subdirs: `scripts/`, `configs/`, `checkpoints/`, `results/`
- [ ] Add `checkpoints/` and `results/` to `.gitignore`
- [ ] Create `experiments/yolo26-pose-kd/requirements.txt` (ultralytics, mmpose, torch, torchvision)
- [ ] Verify `ultralytics` supports YOLO26-Pose (`pip show ultralytics` or `uv add`)

**Files:**
- `experiments/yolo26-pose-kd/.gitignore`
- `experiments/yolo26-pose-kd/requirements.txt`

**Validation:** `ls experiments/yolo26-pose-kd/` shows expected structure.

---

## Task 2: Explore FineFS Data Format

Before writing converters, understand FineFS data.

**Actions:**
- [ ] Extract `skeleton.zip` from `/home/michael/Downloads/FineFS/data/`
- [ ] Read 1 NPZ file — check shape, keypoint format, coordinate system
- [ ] Read 1 annotation JSON — check structure, timing, element labels
- [ ] Determine: 3D or 2D? Normalized or pixel? Which 17kp mapping?
- [ ] Document format in a comment at top of converter script

**Output:** Format spec for FineFS (shape, dtype, keypoint order, coordinate range).

**Blocker:** Tasks 3 and 4 depend on this.

---

## Task 3: FineFS → YOLO Converter

Convert FineFS dataset to YOLO pose format.

**Context:** FineFS has 1,167 videos, NPZ shape (4,350, 17, 3) per video = ~5M raw frames. Files: `skeleton.zip` (868MB), `video.zip` (40GB), `annotation.zip` (1.1MB) — all at `/home/michael/Downloads/FineFS/data/`. Videos already extracted (1167 MP4 files).

**Sampling strategy:** 2fps from video (assume 30fps source → every 15th frame) → ~290 frames/video → 338K raw. After filter (>= 5 visible keypoints, ~80% pass) → **~270K frames**. This is the primary dataset (75% of training data).

**Why 2fps:** Adjacent frames in skating video are nearly identical (30fps). 2fps captures diverse poses without redundancy. Research shows diminishing returns after ~100K images — our 270K is above sweet spot but necessary for domain coverage across 1167 different skaters/elements.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_finefs.py`
- [ ] Read annotation JSON — check structure, timing, element labels, visibility flags
- [ ] Map FineFS 17kp to COCO 17kp (if different order)
- [ ] 3D→2D projection (take x,y, discard z)
- [ ] Extract frames from videos at 2fps (OpenCV)
- [ ] Generate bounding boxes from keypoints (PCK-based padding, factor=0.2)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Split: 80% train / 20% val (video-level split, not frame-level — no leakage)
- [ ] Output: YOLO format (images/ + labels/*.txt per image)
- [ ] Spot-check: visual overlay of keypoints on 10 random frames
- [ ] Record actual frame count after sampling

**Input:** `/home/michael/Downloads/FineFS/data/skeleton/` + `video/` + `annotation/`
**Output:** `experiments/yolo26-pose-kd/data/finefs/train/` (~216K) and `val/` (~54K)

**Validation:** Count total frames, check label distribution, visual spot-check.

---

## Task 4: FSAnno → YOLO Converter

Convert FSAnno dataset to YOLO pose format.

**Context:** FSAnno has 3,700 clips with 4DHuman pose outputs. Available via `rclone` at `gdrive-advanced:FSAnno/`. NOT downloaded locally yet. 4DHuman format — need to map to COCO 17kp.

**Sampling:** ~20 frames/clip at 2fps → 74K raw. After filter → **~59K frames**.

**Actions:**
- [ ] Download FSAnno from GDrive (`rclone copy gdrive-advanced:FSAnno ...`)
- [ ] Explore PKL format in `4dhuman_outputs/` — determine keypoint count, format
- [ ] Map 4DHuman keypoints to COCO 17kp
- [ ] Extract frames from videos at 2fps (OpenCV)
- [ ] Generate bounding boxes from keypoints (PCK-based padding)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Split: 80% train / 20% val (clip-level)
- [ ] Output: YOLO format
- [ ] Spot-check: visual overlay on 10 random frames

**Input:** `gdrive-advanced:FSAnno/4dhuman_outputs/`
**Output:** `experiments/yolo26-pose-kd/data/fsanno/train/` (~47K) and `val/` (~12K)

**Validation:** Count total frames, spot-check.

---

## ~~Task 5: FSC + MCFS → EXCLUDED~~

**Status:** EXCLUDED — no original video frames available.

**Reason:** FSC (4168 sequences, 150 frames each, PKL) and MCFS (2668 segments, 141 frames each, PKL) contain only pose data without source video. YOLO training requires images for person detection + pose regression — skeleton-only data on black background is unsuitable. Generating synthetic images from poses would introduce domain gap.

**Data available (for reference only):**
- FSC: `data/datasets/figure-skating-classification/train_data.pkl` — shape (150, 17, 3), 4161 train + 1007 test sequences
- MCFS: `data/datasets/mcfs/segments.pkl` — shape (T, 17, 2), 2668 segments

**Impact:** Minor. FineFS (270K) + FSAnno (59K) = 329K skating frames already above sweet spot (~100K). FSC/MCFS would add redundancy, not diversity.
- [ ] Record actual frame count

---

## Task 6: Combine Datasets → data.yaml

Merge all datasets into single Ultralytics-compatible dataset.

**Expected data budget:**

| Dataset | Train | Val | Role |
|---------|-------|-----|------|
| FineFS (2fps, 80% filter) | ~216K | ~54K | Primary skating domain |
| FSAnno (2fps, 80% filter) | ~47K | ~12K | Skating domain supplement |
| AthletePose3D | 71K | — | Multi-sport generalization (train only) |
| COCO (15% mix) | ~8.5K | — | Prevents catastrophic forgetting |
| **TOTAL** | **~343K** | **~66K** | |

**Critical:** COCO mix in every batch. Domain-only fine-tuning killed model before (-38% AP, see POST-MORTEM). COCO = insurance against forgetting general poses.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/configs/data.yaml`
- [ ] Paths for train: finefs_train + fsanno_train + ap3d_train + coco_15pct
- [ ] Paths for val: finefs_val + fsanno_val (skating-only, primary quality metric)
- [ ] `kpt_shape: [17, 3]` (COCO 17kp + visibility)
- [ ] `names: ['person']`
- [ ] Verify: `yolo val model=yolo26n-pose.pt data=data.yaml` runs without errors
- [ ] Count actual train/val images (may differ from estimates)

**Output:** `data.yaml` with all dataset paths.

**Validation:** Ultralytics can load the dataset without errors.

---

## Task 7: Vast.ai Environment Setup

Prepare remote training environment on Vast.ai.

**GPU Selection:**
- Primary: RTX 4090 24GB (DLPerf≈55, ~$0.14-0.28/hr, best $/perf ratio)
- Fallback: A100 40GB ($0.26-0.52/hr) if VRAM insufficient for YOLO26m + teacher

**Rental Strategy — Unverified + Smoke Test:**
Unverified ≠ broken, just "not yet evaluated by platform". Verification is fully automated (reliability >= 90%, CUDA >= 12, 500+ Mbps). Real risks: provisioning failures (Docker/SSH), thermal throttling under load. Mitigated by 15-min smoke test. If machine survives 15 min at 100% GPU load → likely stable. Savings: 50-80% vs verified.

**Actions:**
- [ ] Search Vast.ai for RTX 4090 24GB, 200GB+ disk, on-demand
- [ ] Filter: reliability >= 95% (even unverified), CUDA >= 12
- [ ] Rent instance (unverified OK — cheaper, see strategy above)
- [ ] **Smoke test (15 min):** run `gpu_burn` or training on 100 images at 100% load
  - Check `nvidia-smi` — temp >85°C or clock drops → destroy instance, rent another
  - Check network — download test file, if <10 MB/s → destroy
  - If smoke test passes → proceed with setup
- [ ] Install: Python 3.11+, PyTorch with CUDA, ultralytics, mmpose
- [ ] Upload: all YOLO format datasets (rsync, compress)
- [ ] Upload: MogaNet-B weights (`moganet_b_ap2d_384x288.pth`)
- [ ] Upload: pretrained YOLO26 weights (`yolo26n/s/m-pose.pt`)
- [ ] Verify: MogaNet-B inference works on 1 test image
- [ ] Verify: YOLO26 validation works on skating val set
- [ ] Set up persistent tmux/screen session
- [ ] Set up checkpointing: best + every 10 epochs (required for interruptible recovery)

**Checkpointing Strategy (for unverified instances):**
- Save best model (by skating val AP)
- Save every 10 epochs to disk
- Optionally sync checkpoints to R2 (external storage)
- If instance dies: resume from last checkpoint on new instance

**Output:** Working remote environment with all data and models.

---

## Task 8: Stage 1 — Baseline Validation

Measure pretrained baselines on skating val. CRITICAL before any training.

**Actions:**
- [ ] Run `yolo val model=yolo26n-pose.pt data=skating.yaml` → record AP, AP50, per-joint AP
- [ ] Run `yolo val model=yolo26s-pose.pt data=skating.yaml` → same metrics
- [ ] Run `yolo val model=yolo26m-pose.pt data=skating.yaml` → same metrics
- [ ] Run MogaNet-B eval on skating val → AP, AP50
- [ ] Run YOLO26n on AP3D val (cross-domain check)
- [ ] Compile baseline comparison table
- [ ] Save to `results/baseline_comparison.csv`

**Output:** Baseline table with real numbers.

**Decision point:** If pretrained AP < 0.5 on skating val, data quality may be insufficient.

---

## Task 9: Stage 2 — Fine-tune Ablation (Grid Search)

Fine-tune YOLO26 on GT data to find best config.

**Actions:**
- [ ] Create fine-tune training script (extends existing `train_yolo26_pose.py`)
- [ ] Define grid: freeze=[0,10,20], lr=[0.0005,0.001], epochs=50
- [ ] Run experiments sequentially (single GPU)
- [ ] All configs: `val=true, save_period=10, patience=20`
- [ ] Track: train/val loss, AP per epoch
- [ ] After all runs: compile results table, find best config

**Output:** `results/stage2_finetune_results.csv`

**Decision point:**
- If best fine-tune > pretrained → use fine-tuned weights for Stage 3
- If pretrained wins all → use pretrained directly for Stage 3
- If all fine-tune < pretrained → STOP, debug data quality

---

## Task 10: Stage 2.5 — Teacher Domain Adaptation

Adapt MogaNet-B to skating domain.

**Actions:**
- [ ] Create MogaNet training script with frozen backbone
- [ ] Fine-tune deconv head only (3 layers, 256 channels)
- [ ] Config: lr=0.0001, epochs=10-15, data=skating train
- [ ] Validate on skating val before and after
- [ ] Save adapted weights: `moganet_b_skating_adapted.pth`

**Output:** MogaNet-B weights adapted to skating domain.

**Validation:** AP on skating val must improve (or stay same). If it drops, use original weights.

---

## Task 11: Simulated Heatmap Module

Implement MSRA encoding for DistilPose KD.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/simulate_heatmap.py`
- [ ] Function `keypoints_to_heatmap(keypoints, visibility, sigma=2.0, hm_shape=(17,48,64))`
- [ ] MSRA Gaussian encoding around each predicted keypoint
- [ ] Visibility masking: zero out invisible keypoints
- [ ] Unit tests: known keypoints → expected heatmap peaks
- [ ] Benchmark: verify output shape matches MogaNet heatmap resolution

**Output:** `simulate_heatmap.py` with tests.

---

## Task 12: DistilPoseTrainer Implementation

Implement custom Ultralytics trainer with KD loss.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/distill_trainer.py`
- [ ] Subclass `BaseTrainer` from Ultralytics
- [ ] Load teacher model (MogaNet-B) in `__init__`, set `eval()`
- [ ] Override `compute_loss()`:
  - Get GT loss from `super().compute_loss(batch)`
  - Extract student keypoints from current batch predictions
  - Convert to simulated heatmaps (Task 11)
  - Teacher forward pass with `torch.no_grad()`
  - Resize teacher heatmaps to match student resolution
  - Compute KL divergence with temperature scaling
  - Return `gt_loss + alpha * kd_loss`
- [ ] Alpha annealing: 0.5 → 0.3 linearly over training
- [ ] Warm-up: alpha=0 for first 10 epochs
- [ ] Fallback: offline teacher heatmaps if VRAM insufficient
- [ ] Test: run 1 epoch on 10 images, verify loss decreases

**Output:** `distill_trainer.py` with DistilPoseTrainer class.

---

## Task 13: Stage 3 — DistilPose Response KD

Run knowledge distillation training.

**Actions:**
- [ ] Create `configs/stage3_distill.yaml` with KD params
- [ ] Run training: `model.train(data=skating_full.yaml, trainer=DistilPoseTrainer, ...)`
- [ ] Monitor: GT loss, KD loss, total loss, val AP per epoch
- [ ] Check: KD loss should decrease, val AP should increase
- [ ] Early stop on val AP (patience=20)
- [ ] Save best model by skating val AP

**Output:** Trained YOLO26 model with distilled knowledge.

**Decision point:**
- If gap (teacher - student) < 0.08 AP → skip Stage 3.5
- If gap >= 0.08 AP → proceed to Stage 3.5

---

## Task 14: Stage 3.5 — Optional TDE + Feature KD [CONDITIONAL]

Only if Stage 3 gap >= 0.08 AP.

**Actions:**
- [ ] Research DistilPose TDE implementation from github.com/yshMars/DistilPose
- [ ] Implement Token-distilling Encoder (attention + tokenization)
- [ ] Add feature KD loss: `beta * MSE(student_feat, TDE(teacher_feat))`
- [ ] Re-run training with combined loss
- [ ] Compare: Stage 3 only vs Stage 3 + 3.5

**Output:** Potentially improved student model.

---

## Task 15: Stage 4 — Student Size Selection

Determine optimal student model size.

**Actions:**
- [ ] Run Stage 3 KD with best config for YOLO26n, YOLO26s, YOLO26m (parallel if multi-GPU)
- [ ] Compile final comparison table:
  - Model | AP (skating val) | AP (AP3D val) | Speed (FPS) | Params | Gap vs teacher
- [ ] Select smallest model meeting all success criteria
- [ ] Export selected model to ONNX for production use

**Output:** Final model + comparison table.

---

## Task 16: Results Documentation

Document all results.

**Actions:**
- [ ] Write `experiments/yolo26-pose-kd/results/README.md` with:
  - Baseline comparison table
  - Fine-tune ablation results
  - KD training curves summary
  - Final model performance
  - Lessons learned
- [ ] Update project ROADMAP.md with KD experiment status
- [ ] Commit all scripts and results (not checkpoints)

---

## Dependency Graph

```
Task 0 (time estimation) ──→ recalculate after Task 3-5 (actual N_images)

Task 1 (structure)
  ├── Task 2 (explore FineFS) ──→ Task 3 (FineFS converter)
  └── Task 4 (FSAnno converter)

Task 3 + Task 4 ──→ Task 6 (data.yaml) ──→ Task 7 (Vast.ai setup)
                                                         └──→ Task 8 (baseline)

Task 8 ──→ Task 9 (fine-tune ablation)
Task 8 ──→ Task 10 (teacher adaptation)

Task 11 (heatmap module) ──→ Task 12 (DistilPoseTrainer)

Task 9 + Task 10 + Task 12 ──→ Task 13 (KD training)
  └──→ Task 14 (TDE, conditional)
       └──→ Task 15 (student size selection)
            └──→ Task 16 (results)
```

## Parallelization Opportunities

| Group | Tasks | Can run in parallel? |
|-------|-------|---------------------|
| Structure + data explore | 0, 1, 2 | Yes |
| Data converters | 3, 4 | Yes (after Task 2) |
| KD modules | 11, 12 | Yes (sequential, 11→12) |
| Vast.ai setup | 7 | After Task 6 (needs data ready) |
| Baseline + teacher adaptation | 8, 10 | Partially (8 first, then 10) |
