# Implementation Plan: KD MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Design spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
**Created:** 2026-04-18
**Branch:** `experiments/yolo26-pose-kd`

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

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_finefs.py`
- [ ] Extract frames from videos at 10fps (OpenCV)
- [ ] Map FineFS 17kp to COCO 17kp (if different order)
- [ ] 3D→2D projection if needed (take x,y, discard z)
- [ ] Generate bounding boxes from keypoints (PCK-based padding, factor=0.2)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Output: YOLO format (images/ + labels/*.txt per image)
- [ ] Spot-check: visual overlay of keypoints on 10 random frames

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

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_fsc_mcfs.py`
- [ ] Read unified format: `{split}_poses.npy` (shape, dtype, keypoint order)
- [ ] Render each frame as image from keypoints (black bg, white skeleton) OR
  - [ ] **Decision needed:** Do we have original video frames for FSC/MCFS?
- [ ] If no original frames: generate synthetic skeleton images (not ideal for training)
- [ ] If frames available: extract from source
- [ ] Generate bounding boxes from keypoints
- [ ] Filter frames with < 5 visible keypoints
- [ ] Output: YOLO format
- [ ] Spot-check: 10 random samples

**Input:** `data/unified/fsc-64/`, `data/unified/mcfs-129/`
**Output:** `experiments/yolo26-pose-kd/data/fsc/train/`, `data/mcfs/train/`

**Important question:** FSC/MCFS have pose sequences (T, 17, 2) but no original video frames. Need to determine if we have the source videos or need to work with skeleton-only data.

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

Prepare remote training environment.

**Actions:**
- [ ] Rent Vast.ai instance (RTX 4090 24GB, 200GB+ disk)
- [ ] Install: Python 3.11+, PyTorch with CUDA, ultralytics, mmpose
- [ ] Upload: all YOLO format datasets (rsync)
- [ ] Upload: MogaNet-B weights (`moganet_b_ap2d_384x288.pth`)
- [ ] Upload: pretrained YOLO26 weights (`yolo26n/s/m-pose.pt`)
- [ ] Verify: MogaNet-B inference works on 1 test image
- [ ] Verify: YOLO26 validation works on skating val set
- [ ] Set up persistent tmux/screen session

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
Task 1 (structure)
  ├── Task 2 (explore FineFS) ──→ Task 3 (FineFS converter)
  ├── Task 4 (FSAnno converter)
  ├── Task 5 (FSC/MCFS converter)
  └── Task 7 (Vast.ai setup)

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
| Data converters | 2, 3, 4, 5 | Yes (after Task 1) |
| Vast.ai setup | 7 | Yes (with data converters) |
| Baseline + teacher adapation | 8, 10 | Partially (8 first, then 10) |
| KD modules | 11, 12 | Yes (sequential, 11→12) |

## Estimated Timeline

| Phase | Tasks | Duration (est.) |
|-------|-------|----------------|
| Data prep | 1-6 | 2-3 days |
| Environment | 7 | 0.5 day |
| Baseline + ablation | 8-10 | 1-2 days |
| KD implementation | 11-12 | 1-2 days |
| KD training | 13-14 | 1-2 days |
| Selection + docs | 15-16 | 0.5 day |
| **Total** | | **6-10 days** |
