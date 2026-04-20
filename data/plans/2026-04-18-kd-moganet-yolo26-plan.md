# Implementation Plan: KD MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Design spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
**Created:** 2026-04-18
**Branch:** `experiments/yolo26-pose-kd`

---

## Task 0: Training Time & Risk Estimation

Calculate expected training time using published benchmarks.

**Reference:** Ultralytics docs — **2.8s compute per 1000 images on RTX 4090** (YOLO26n, baseline).

### Risk Mitigation Strategies

**1. Offline teacher heatmaps (biggest cost saving)**
Pre-compute MogaNet-B heatmaps ONCE for all training images, store as .npy alongside labels. Eliminates 1.5× teacher overhead during KD, eliminates VRAM concern. Stage 3 time drops ~33%.

**2. Skip Stage 2 ablation — POST-MORTEM already answered**
Previous experiment: freeze=20 was best (0.517 vs 0.406 at freeze=0). Start KD from pretrained + freeze=10-20. Fallback to ablation only if KD fails.

**3. Gate Stage 2.5 — skip if teacher already good on skating**
Run MogaNet-B eval on 100 skating val images first. If AP > 0.85: skip teacher adaptation. Expected: MogaNet-B AP=0.962 on AP3D, likely > 0.85 on skating.

**4. Gate Stage 3.5 — skip if gap is small**
After Stage 3 converges: check teacher-student gap. gap < 0.08 → DONE, skip TDE. 50% probability of saving 53h.

**5. Progressive sizing — n first, then s/m only if needed**
Train YOLO26n (100 epochs). If AP >= 0.85 → DONE. Saves ~50h if n is sufficient.

**6. Data validation locally — zero GPU cost**
Convert ALL datasets locally. Spot-check: visual overlay on 10 random frames per dataset. `yolo val` on 100 images to verify format. Fix bugs BEFORE renting GPU.

### Revised Pipeline (risk-mitigated)

| Stage | Eff epochs | 4090 h | 5090 h | Probability |
|-------|-----------|--------|--------|-------------|
| Data conversion (LOCAL, no GPU) | 0 | 0h | 0h | 100% |
| Teacher heatmap pre-compute | 30 | 8h | 4h | 100% |
| Stage 1: Baseline val | 5 | 1h | 1h | 100% |
| Stage 3: KD (offline heatmaps) | 260 | 69h | 33h | 100% |
| Stage 2.5: Teacher adap (gated) | 15 | 4h | 2h | 50% |
| Stage 3.5: TDE (gated) | 100 | 27h | 13h | 50% |
| Stage 4: YOLO26n (progressive) | 120 | 32h | 15h | 100% |
| Stage 4b: YOLO26s (if n fails) | 50 | 13h | 6h | 50% |
| **TOTAL** | | **155h** | **74h** | |

**Cost estimate (with contingency):**

| GPU | ETA | Cost | Remaining ($150) |
|-----|-----|------|-------------------|
| RTX 5090 ($0.305/hr) | 74h | **$23** | $127 (85%) |
| RTX 4090 ($0.295/hr) | 155h | **$46** | $104 (70%) |

**VRAM:** Not a concern with offline heatmaps. Teacher model NOT in GPU during training. Any 24GB GPU works for YOLO26n/s/m.

**Actions:**
- [ ] After data conversion (Task 3-4): record actual N_images
- [ ] Recalculate using table above with real N_images
- [ ] Pre-compute teacher heatmaps as first step on rented GPU (one-time, ~4-8h)

**Validation:** Total cost < $150 with >70% margin.

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

## Task 9: Stage 2 — Fine-tune Ablation [SKIP BY DEFAULT]

**Status:** SKIP — POST-MORTEM already answered. Previous experiment showed freeze=20 was best (0.517 vs 0.406 at freeze=0). Start KD from pretrained + freeze=10-20.

**Fallback trigger:** Only run if Stage 3 KD fails (student AP < 0.70 after convergence).

**Actions (only if triggered):**
- [ ] Create fine-tune training script
- [ ] Grid: freeze=[10,20], lr=[0.0005,0.001], epochs=50
- [ ] All configs: `val=true, save_period=10, patience=20`
- [ ] Use fine-tuned weights for Stage 3 retry

**Output:** `results/stage2_finetune_results.csv` (only if triggered)

---

## Task 10: Stage 2.5 — Teacher Domain Adaptation [GATED]

**Gate:** Skip if MogaNet-B AP > 0.85 on 100 skating val images (likely — AP=0.962 on AP3D).

**Actions:**
- [ ] Run MogaNet-B eval on 100 skating val images → record AP
- [ ] If AP > 0.85: **SKIP**, use original weights. Proceed to Task 11.
- [ ] If AP <= 0.85: fine-tune deconv head only (frozen backbone)
  - [ ] Config: lr=0.0001, epochs=10-15, data=skating train
  - [ ] Validate AP before and after
  - [ ] Save: `moganet_b_skating_adapted.pth`

**Output:** Decision (skip or adapted weights).

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

## Task 12: DistilPoseTrainer Implementation (Offline Heatmaps)

Implement custom Ultralytics trainer with KD loss using **pre-computed offline teacher heatmaps**.

**Key change:** Teacher model NOT loaded during training. Heatmaps pre-computed once (Task 13 step 0), stored as .npy alongside labels. Eliminates 1.5× overhead and VRAM concern.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/distill_trainer.py`
- [ ] Subclass `BaseTrainer` from Ultralytics
- [ ] Override `get_dataset()` to load offline heatmaps (.npy) alongside labels
- [ ] Override `compute_loss()`:
  - Get GT loss from `super().compute_loss(batch)`
  - Extract student keypoints from current batch predictions
  - Convert to simulated heatmaps (Task 11)
  - Load pre-computed teacher heatmaps for current batch (no teacher model needed)
  - Compute KL divergence with temperature scaling
  - Return `gt_loss + alpha * kd_loss`
- [ ] Alpha annealing: 0.5 → 0.3 linearly over training
- [ ] Warm-up: alpha=0 for first 10 epochs
- [ ] Test: run 1 epoch on 10 images, verify loss decreases

**Output:** `distill_trainer.py` with DistilPoseTrainer class.

---

## Task 13: Stage 3 — DistilPose Response KD (Progressive Sizing)

Run knowledge distillation training with offline teacher heatmaps.

**Step 0: Pre-compute teacher heatmaps (first thing on rented GPU)**
- [ ] Run MogaNet-B inference on all training images → save heatmaps as .npy
- [ ] File naming: `{image_id}_teacher_hm.npy`, shape (17, 48, 64)
- [ ] Verify: load 10 random heatmaps, visualize, check peaks match GT keypoints
- [ ] Estimated time: ~4h on RTX 5090, ~8h on RTX 4090

**Step 1: Train YOLO26n (primary)**
- [ ] Create `configs/stage3_distill.yaml` with KD params
- [ ] Run: `model.train(data=skating_full.yaml, trainer=DistilPoseTrainer, freeze=10, epochs=100)`
- [ ] Monitor: GT loss, KD loss, total loss, val AP per epoch
- [ ] Early stop on val AP (patience=20)
- [ ] Save best model by skating val AP

**Step 2: Check success criteria**
- [ ] If YOLO26n AP >= 0.85 on skating val → **DONE**, skip s/m
- [ ] If YOLO26n AP < 0.85 → train YOLO26s with same config (Step 3)
- [ ] If YOLO26s AP < 0.85 → train YOLO26m (Step 4, last resort)

**Output:** Trained YOLO26 model(s) with distilled knowledge.

**Decision point:**
- If gap (teacher - student) < 0.08 AP → skip Stage 3.5
- If gap >= 0.08 AP → proceed to Stage 3.5

---

## Task 14: Stage 3.5 — Optional TDE + Feature KD [GATED]

**Gate:** Only if Stage 3 gap >= 0.08 AP.

**Actions:**
- [ ] Research DistilPose TDE implementation from github.com/yshMars/DistilPose
- [ ] Implement Token-distilling Encoder (attention + tokenization)
- [ ] Add feature KD loss: `beta * MSE(student_feat, TDE(teacher_feat))`
- [ ] Re-run training with combined loss
- [ ] Compare: Stage 3 only vs Stage 3 + 3.5

**Output:** Potentially improved student model.

---

## Task 15: Stage 4 — Student Size Selection [PROGRESSIVE]

Already handled in Task 13 (progressive sizing). This task is for final evaluation.

**Actions:**
- [ ] Compile final comparison table:
  - Model | AP (skating val) | AP (AP3D val) | Speed (FPS) | Params | Gap vs teacher
- [ ] Select smallest model meeting all success criteria
- [ ] Export selected model to ONNX for production use
- [ ] If no model meets criteria → increase data or add Stage 3.5

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
Task 0 (time estimation) ──→ recalculate after Task 3-4 (actual N_images)

Task 1 (structure)
  ├── Task 2 (explore FineFS) ──→ Task 3 (FineFS converter)
  └── Task 4 (FSAnno converter)

Task 3 + Task 4 ──→ Task 6 (data.yaml) ──→ Task 7 (Vast.ai setup)
                                                         └──→ Task 8 (baseline)
                                                                └──→ Task 10 (teacher adap, GATED)
                                                                       └──→ Task 13 step 0 (pre-compute heatmaps)

Task 11 (heatmap module) ──→ Task 12 (DistilPoseTrainer)

Task 10 + Task 12 ──→ Task 13 (KD training + progressive sizing)
  └──→ Task 14 (TDE, GATED)
       └──→ Task 15 (final evaluation)
            └──→ Task 16 (results)

Task 9 (fine-tune ablation) — SKIP by default, only if KD fails
```

## Parallelization Opportunities

| Group | Tasks | Can run in parallel? |
|-------|-------|---------------------|
| Structure + data explore | 0, 1, 2 | Yes |
| Data converters | 3, 4 | Yes (after Task 2) |
| KD modules | 11, 12 | Yes (sequential, 11→12) |
| Vast.ai setup | 7 | After Task 6 (needs data ready) |
| Baseline → teacher eval → heatmaps | 8, 10, 13-step0 | Sequential (8→10→13) |
| Task 9 (ablation) | 9 | SKIP by default |
