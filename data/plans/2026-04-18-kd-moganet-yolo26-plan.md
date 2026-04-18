# Implementation Plan: KD MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Design spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
**Created:** 2026-04-18
**Branch:** `experiments/yolo26-pose-kd`

---

## Task 0: Local Benchmark (DLPerf Reference)

Measure local GPU performance to predict training time on Vast.ai BEFORE renting.

**Why:** $150 budget is hard constraint. Must know if training fits before spending money.

**Actions:**
- [ ] Verify `yolo26n-pose.pt` weights exist locally (download if missing)
- [ ] Run 1 epoch on 100 images, batch=16, imgsz=640 on local RTX 3050 Ti
- [ ] Record: `t_ref` = time for 100 images × 1 epoch (seconds)
- [ ] Record: `DLPerf_local` = DLPerf score for RTX 3050 Ti (~14)
- [ ] Calculate `time_per_iter` = `t_ref / (100 / batch)` = seconds per iteration
- [ ] Calculate training time for each target GPU:
  ```
  time_target = t_ref × (N_images / 100) × (DLPerf_local / DLPerf_target) × (batch_target / batch_local)
  time_real = time_target × 2.0  # KD overhead (teacher ×1.5, dataloader ×1.2, val ×1.1)
  cost = (time_real / 3600) × hourly_rate
  ```
- [ ] Verify: `cost < $150` for RTX 4090 ($0.28/hr, DLPerf≈55)
- [ ] If over budget: reduce N_images via sampling (design spec Section 2)

**Output:** Budget calculation with concrete numbers — hours and cost per GPU option.

**Blocker:** Task 3-5 (data conversion) needed for actual N_images count. Run with N=1000 placeholder first, recalculate after data prep.

**Validation:** `cost < $150` for primary GPU (RTX 4090).

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

**Context:** FineFS has 1,167 videos, NPZ shape (4,350, 17, 3) per video = ~5M raw frames. Need sampling strategy to reduce volume.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_finefs.py`
- [ ] Decide sampling strategy (see design spec "Sampling Strategy" section)
- [ ] Extract frames from videos at chosen fps (OpenCV)
- [ ] Map FineFS 17kp to COCO 17kp (if different order)
- [ ] 3D→2D projection if needed (take x,y, discard z)
- [ ] Generate bounding boxes from keypoints (PCK-based padding, factor=0.2)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Output: YOLO format (images/ + labels/*.txt per image)
- [ ] Spot-check: visual overlay of keypoints on 10 random frames
- [ ] Record actual frame count after sampling

**Input:** `/home/michael/Downloads/FineFS/data/skeleton.zip` + `video.zip`
**Output:** `experiments/yolo26-pose-kd/data/finefs/train/` and `val/` (80/20 split)

**Validation:** Count total frames, check label distribution, visual spot-check.

---

## Task 4: FSAnno → YOLO Converter

Convert FSAnno dataset to YOLO pose format.

**Actions:**
- [ ] Download FSAnno from GDrive (`rclone copy gdrive-advanced:FSAnno ...`)
- [ ] Explore PKL format in `4dhuman_outputs/` — determine keypoint count, format
- [ ] Map 4DHuman keypoints to COCO 17kp
- [ ] Extract frames from videos at 10fps (OpenCV)
- [ ] Generate bounding boxes from keypoints (PCK-based padding)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Output: YOLO format
- [ ] Spot-check: visual overlay on 10 random frames

**Input:** `gdrive-advanced:FSAnno/4dhuman_outputs/`
**Output:** `experiments/yolo26-pose-kd/data/fsanno/train/` and `val/`

**Validation:** Count total frames, spot-check.

---

## Task 5: FSC + MCFS → YOLO Converter

Convert existing unified numpy data to YOLO pose format.

**Important:** FSC/MCFS have pose sequences (T, 17, 2) but no original video frames. Need to determine if we have the source videos or need to work with skeleton-only data. Skeleton-only images (black bg, white skeleton) are not ideal for YOLO training.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_fsc_mcfs.py`
- [ ] Read unified format: `{split}_poses.npy` (shape, dtype, keypoint order)
- [ ] Measure actual frame counts per sequence
- [ ] Determine: do we have original video frames? Check source data.
- [ ] If no original frames: assess whether skeleton-only images are viable for YOLO
- [ ] If frames available: extract from source
- [ ] Generate bounding boxes from keypoints
- [ ] Filter frames with < 5 visible keypoints
- [ ] Output: YOLO format
- [ ] Spot-check: 10 random samples
- [ ] Record actual frame count

**Input:** `data/unified/fsc-64/`, `data/unified/mcfs-129/`
**Output:** `experiments/yolo26-pose-kd/data/fsc/train/`, `data/mcfs/train/`

**Decision point:** If no original frames, FSC/MCFS may need to be excluded from training.

---

## Task 6: Combine Datasets → data.yaml

Merge all datasets into single Ultralytics-compatible dataset.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/configs/data.yaml`
- [ ] Paths for train: finefs + fsanno + fsc + mcfs + ap3d + 15% COCO
- [ ] Paths for val: finefs_val + fsanno_val + fsc_val + mcfs_val (skating-only)
- [ ] `kpt_shape: [17, 3]` (COCO 17kp + visibility)
- [ ] `names: ['person']`
- [ ] Verify: `yolo val model=yolo26n-pose.pt data=data.yaml` runs without errors
- [ ] Count total train/val images

**Output:** `data.yaml` with all dataset paths.

**Validation:** Ultralytics can load the dataset without errors.

---

## Task 7: Vast.ai Environment Setup

Prepare remote training environment on Vast.ai.

**Prerequisite:** Task 0 benchmark complete, budget verified.

**GPU Selection (DLPerf-based, from Task 0):**
- Primary: RTX 4090 24GB (DLPerf≈55, $0.28/hr verified, best $/DLPerf-hr)
- Fallback: A100 40GB (DLPerf≈52, $0.52/hr) if 24GB VRAM insufficient for teacher+student
- Budget: $150 total, calculate max hours from Task 0 formula

**Rental Requirements:**
- Type: **On-Demand, Verified only** (unverified can be killed mid-training)
- Disk: 200GB+ (datasets + checkpoints)
- Image: CUDA 12.x + PyTorch compatible

**Actions:**
- [ ] Rent Vast.ai instance (RTX 4090 24GB verified, 200GB+ disk, on-demand)
- [ ] Verify DLPerf matches expected (~55 for 4090) — if significantly lower, recalculate budget
- [ ] Install: Python 3.11+, PyTorch with CUDA, ultralytics, mmpose
- [ ] Upload: all YOLO format datasets (rsync, compress)
- [ ] Upload: MogaNet-B weights (`moganet_b_ap2d_384x288.pth`)
- [ ] Upload: pretrained YOLO26 weights (`yolo26n/s/m-pose.pt`)
- [ ] Verify: MogaNet-B inference works on 1 test image
- [ ] Verify: YOLO26 validation works on skating val set
- [ ] Verify: Task 0 benchmark replicates on remote GPU (compare t_ref)
- [ ] Set up persistent tmux/screen session
- [ ] Set up checkpointing: best + every 10 epochs (required for interruptible recovery)

**Budget Guard:**
- Track cumulative cost after each stage
- If cost > 80% budget with stages remaining: reduce epochs, skip optional stages, or switch to smaller student
- If GPU gets killed: resume from last checkpoint (save_period=10)

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
Task 0 (local benchmark) ──→ Task 7 (Vast.ai setup, needs budget calc)

Task 1 (structure)
  ├── Task 2 (explore FineFS) ──→ Task 3 (FineFS converter)
  ├── Task 4 (FSAnno converter)
  └── Task 5 (FSC/MCFS converter)

Task 3 + Task 4 + Task 5 ──→ Task 6 (data.yaml) ──→ Task 8 (baseline)

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
| Benchmark + structure | 0, 1 | Yes |
| Data converters | 2, 3, 4, 5 | Yes (after Task 1) |
| Vast.ai setup | 7 | After Task 0 (needs budget calc) |
| Baseline + teacher adaptation | 8, 10 | Partially (8 first, then 10) |
| KD modules | 11, 12 | Yes (sequential, 11→12) |
