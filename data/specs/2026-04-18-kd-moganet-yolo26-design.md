# Knowledge Distillation: MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Date:** 2026-04-18
**Status:** Approved
**Location:** `experiments/yolo26-pose-kd/`

---

## 1. Goal

Figure skating pose estimator: YOLO26-Pose model optimized for figure skating, trained via knowledge distillation from MogaNet-B teacher. Universal sports coverage (athletics + skating) with primary quality metric on skating data.

### Success Criteria

| Metric | Target | Failure |
|--------|--------|---------|
| AP on skating val set | >= 0.85 | < 0.75 |
| AP on AthletePose3D val | >= 0.70 | < 0.55 |
| Gap vs MogaNet teacher (skating) | < 0.11 | > 0.20 |
| Inference speed | > 100 FPS | < 30 FPS |
| Fine-tune > pretrained baseline | mAP improvement | mAP regression |

### Out of Scope

- Replacing RTMO in production pipeline (experiment first)
- Vast.ai setup details (implementation concern)
- Pseudo-labeling (GT-only strategy)

---

## 2. Data Pipeline

### Datasets (GT-only, no pseudo-labeling)

| Dataset | Pose GT | Format | Conversion Needed | Volume (est.) |
|---------|---------|--------|-------------------|---------------|
| FineFS | 17kp 3D (NPZ) | (frames, 17, 3) | 3D->2D projection + COCO JSON | 1,167 videos, NPZ shape (4350, 17, 3) per video |
| FSAnno | 4D pose (PKL) | 4DHuman format | Extract 17kp -> COCO JSON | 3,700 clips |
| FSC-64 | 17kp 2D (NPY) | (T, 17, 2) | Frame->image + bbox + COCO JSON | 5,031 seq |
| MCFS-129 | 17kp 2D (NPY) | (T, 17, 2) | Frame->image + bbox + COCO JSON | 2,617 seq |
| AthletePose3D | 17kp 2D (COCO JSON) | Already COCO | None (already done) | 71K frames |
| COCO train2017 | 17kp (COCO JSON) | Already COCO | None | ~15% mix |

**Total training data: TBD — measure actual frame counts after conversion (Task 2-5). FineFS alone is ~5M raw frames (1,167 videos × 4,350 frames). Sampling strategy will determine final volume.**

### Data Split

- Skating data (FineFS + FSAnno + FSC + MCFS): 80% train / 20% val
- AP3D + COCO: train only (domain diversity, no leakage into val)
- Val set = skating-only (primary quality metric)

### Sampling Strategy

FineFS alone has ~5M raw frames (1,167 × 4,350). Budget constraint ($150) limits training time.

**Sampling is budget-driven.** Calculate max frames from budget:

```
time_per_epoch = t_ref × (N_images / 100) × (DLPerf_ref / DLPerf_target) × (batch_ref / batch_target)
max_hours = $150 / hourly_rate
max_N_images = (max_hours × 3600 × batch) / (time_per_iter × epochs × overhead)
```

Where:
- `t_ref` = measured time for 100 images, 1 epoch on local GPU (RTX 3050 Ti, DLPerf≈6)
- `overhead` = 2.0× (KD teacher ×1.5, DataLoader ×1.2, validation ×1.1)
- `DLPerf` = Vast.ai DLPerf score (predicts iterations/sec for CNN training)
- `hourly_rate` = Vast.ai verified on-demand price for target GPU

**Decision process:**
1. Measure `t_ref` on local GPU (100 images, batch=16, YOLO26n-pose)
2. Calculate max_N for RTX 4090 ($0.28/hr, DLPerf≈55)
3. Sample datasets proportionally to fit max_N
4. Priority: FineFS > FSAnno > FSC > MCFS (by quality and diversity)

**Priority order for sampling (best quality first):**
- FineFS: highest quality (competition GT scores), 1,167 videos
- FSAnno: competition data with 4D pose, 3,700 clips
- FSC/MCFS: sequential data, no original video frames (may be excluded)
- AP3D: already in YOLO format, 71K frames (include all)
- COCO: 15% mix for domain diversity

### Preprocessing Steps

1. Measure actual frame counts per dataset (before conversion)
2. Decide sampling strategy based on total volume
3. Extract frames from videos (FineFS, FSAnno) at chosen fps
4. Convert all pose data to COCO JSON format (bbox + keypoints + visibility)
5. Generate bounding boxes from keypoints (PCK-based padding)
6. Filter: remove frames with < 5 visible keypoints
7. Convert to YOLO pose format: images/ + labels/ (txt per image)
8. Create data.yaml for Ultralytics

---

## 3. Distillation Architecture

### Teacher

- **Model:** MogaNet-B, fine-tuned on AthletePose3D
- **AP:** 0.962 on AP3D val, AP50 = 1.000
- **Params:** 47.4M
- **Output:** 2D heatmaps (48x64 or 64x48 resolution)
- **Format:** Top-down, heatmap-based

### Student

- **Model:** YOLO26-Pose (size TBD: n/s/m)
- **Pretrained AP:** 0.712 (n), 0.657 (s) on AP3D val
- **Output:** Keypoint regression (x, y, conf) + bounding boxes
- **Format:** One-stage detection + regression

### KD Method: DistilPose-Style Response KD

Source: DistilPose (CVPR 2023) - "Tokenized Pose Regression with Heatmap Distillation"

Core idea: Convert student regression output to simulated heatmaps, then apply KL divergence with teacher's real heatmaps.

#### Simulated Heatmap Encoding (MSRA)

```python
# Student predicts keypoints (x, y, conf) -> simulated heatmaps
heatmap[j, y, x] = exp(-(dx^2 + dy^2) / (2 * sigma^2))
sigma = 2.0  # standard for 64x48 heatmap resolution
# Mask invisible keypoints from GT (FineFS/FSAnno visibility flags)
```

#### Loss Function

```
L_total = L_yolo_pose(GT) + alpha * T^2 * KL_div(
    softmax(student_simulated_hm / T),
    softmax(teacher_hm / T)
)
```

Where:
- `L_yolo_pose` = standard Ultralytics pose loss (box + kpt + RLE)
- `alpha` = 0.5 (annealing to 0.3 over training)
- `T` = 4 (temperature, Hinton standard)
- `T^2` = standard Hinton KD normalization

#### Implementation: Custom Ultralytics Trainer

```python
class DistilPoseTrainer(BaseTrainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temp=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.alpha = alpha
        self.temp = temp

    def compute_loss(self, batch):
        gt_loss = super().compute_loss(batch)

        # Student keypoints -> simulated heatmaps (MSRA encoding)
        student_hm = keypoints_to_heatmap(student_kpts, sigma=2.0, visibility=gt_visibility)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_hm = self.teacher(batch['img'])  # MogaNet heatmaps
            teacher_hm = resize_to_match(teacher_hm, student_hm.shape)

        # KL divergence on heatmaps
        kd_loss = (self.temp ** 2) * F.kl_div(
            F.log_softmax(student_hm / self.temp, dim=1),
            F.softmax(teacher_hm / self.temp, dim=1),
            reduction='batchmean'
        )

        return gt_loss + self.alpha * kd_loss
```

---

## 4. Training Pipeline (Stages)

### Stage 0: Data Preparation

Convert all datasets to YOLO pose format.

**Files to create:**
- `scripts/convert_finefs.py` - FineFS NPZ -> COCO JSON -> YOLO format
- `scripts/convert_fsanno.py` - FSAnno PKL -> COCO JSON -> YOLO format
- `scripts/convert_fsc_mcfs.py` - FSC/MCFS NPY -> COCO JSON -> YOLO format
- `configs/data.yaml` - Dataset configuration for Ultralytics

**Validation:** Spot-check 10 random annotations per dataset, visual overlay on images.

### Stage 1: Baseline Validation

**CRITICAL: Must complete BEFORE any fine-tuning or KD.**

```bash
# Validate all pretrained baselines on skating val set
yolo val model=yolo26n-pose.pt data=skating.yaml device=0
yolo val model=yolo26s-pose.pt data=skating.yaml device=0
yolo val model=yolo26m-pose.pt data=skating.yaml device=0

# Validate MogaNet-B teacher on skating val set
python scripts/eval_moganet.py --model moganet_b_ap2d.pth --data skating_val.yaml

# Validate on AP3D val (cross-domain check)
yolo val model=yolo26n-pose.pt data=ap3d.yaml device=0
```

**Rule:** If pretrained > all fine-tuned configs, do NOT proceed to Stage 2 with those configs. Start KD directly on pretrained.

**Output:** Baseline comparison table with real numbers on skating val.

### Stage 2: Fine-tune Ablation

Fine-tune YOLO26n/s/m on GT data with validation every epoch.

**Hyperparameter grid:**
- freeze: [0, 5, 10, 20]
- lr: [0.0005, 0.001]
- epochs: [50, 100]
- COCO mix: [0%, 10%, 20%]
- patience: 20 (early stopping)

**Mandatory in all configs:**
```yaml
val: true
save_period: 10
patience: 20
```

**Stop condition:** If no fine-tune config beats pretrained on skating val, skip to Stage 3 with pretrained weights.

**Output:** Best fine-tune config (or confirmation that pretrained is optimal).

### Stage 2.5: Teacher Domain Adaptation

Fine-tune MogaNet-B head on skating GT to improve soft-target quality.

**Config:**
- Freeze backbone (MogaNet-B backbone)
- Train only deconv head (3 layers, 256 channels)
- 10-15 epochs
- LR: 0.0001 (conservative)
- Data: skating GT only (FineFS + FSAnno + FSC + MCFS train)

**Validation:** MogaNet AP on skating val (before and after).

### Stage 3: DistilPose Response KD

Apply DistilPose-style response KD.

**Training config:**
```yaml
model: yolo26{s}-pose.pt  # best from Stage 2
data: skating_full.yaml
epochs: 100
batch: 32  # depends on GPU VRAM
imgsz: 640
freeze: 10  # backbone warm-up

# KD params
trainer: DistilPoseTrainer
teacher: moganet_b_skating_adapted.pth
alpha: 0.5  # annealing to 0.3
temp: 4.0

# Augmentation
mosaic: 1.0
fliplr: 0.5
mixup: 0.15

# Validation
val: true
val_period: 5
save_period: 10
patience: 20
```

**Warm-up:** First 10 epochs with GT loss only (alpha=0), then enable KD.

**Files to create:**
- `scripts/distill_trainer.py` - DistilPoseTrainer implementation
- `scripts/simulate_heatmap.py` - MSRA heatmap encoding
- `configs/stage3_distill.yaml` - KD training config

### Stage 3.5: Optional TDE + Feature KD

**Trigger:** Only if gap > 0.08 AP after Stage 3 convergence.

**TDE (Token-distilling Encoder):**
- 1-2 linear layers + tokenization
- Bridges feature space gap between MogaNet (ConvNet) and YOLO26 (CSPNet)
- Based on DistilPose original implementation

**Feature KD:**
- Via YOLOv8-KD adapter pattern
- MSE on intermediate backbone features
- Additional loss term: `beta * MSE(student_feat, adapter(teacher_feat))`

### Stage 4: Student Size Selection

Parallel training of n/s/m with best KD config from Stage 3.

| Size | Params | Speed (est.) | Trade-off |
|------|--------|-------------|-----------|
| YOLO26n | ~3M | ~500 FPS | Fastest, risk of quality gap |
| YOLO26s | ~9M | ~200 FPS | Balanced |
| YOLO26m | ~20M | ~80 FPS | Highest capacity |

**Selection criteria:**
1. All must meet success criteria (AP >= 0.85 on skating val)
2. Pick smallest model that meets criteria
3. If none meet criteria, increase training data or add Stage 3.5

---

## 5. Infrastructure

### Budget Constraint

- **Budget:** $150 total
- **GPU selection:** Based on DLPerf/$/hr ratio (higher is better)
- **Rental type:** On-Demand, Verified only (unverified may be killed mid-training)

### GPU Selection

| GPU | DLPerf (est.) | $/hr (verified) | $/DLPerf-hr | Max hours on $150 |
|-----|---------------|-----------------|-------------|-------------------|
| RTX 4090 | ~55 | $0.28 | **0.005** | 536 hrs |
| A100 PCIE 40GB | ~52 | $0.52 | 0.010 | 288 hrs |
| A100 SXM 80GB | ~52 | $0.67 | 0.013 | 224 hrs |
| H100 SXM 80GB | ~165 | $1.35 | 0.008 | 111 hrs |

**Primary:** RTX 4090 (best $/DLPerf-hr). 24GB VRAM sufficient for YOLO26n/s + MogaNet-B teacher.
**Fallback:** A100 40GB if VRAM insufficient. H100 only if RTX 4090 proves too slow.

### DLPerf-based Time Estimation

Before renting, calculate training time without spending money:

1. **Benchmark locally:** Run 1 epoch on 100 images with local GPU (any GPU)
2. **Calculate target time:**
   ```
   time_target = t_local × (DLPerf_local / DLPerf_target) × (batch_target / batch_local)
   time_real = time_target × overhead  # overhead = 2.0 for KD training
   ```
3. **Verify budget fits:** `cost = time_real_hours × hourly_rate`

DLPerf predicts iterations/second for CNN training (ResNet50-style). YOLO training is CNN work — DLPerf is a good predictor. Less accurate for unusual compute patterns.

### Training Environment

- **Platform:** Vast.ai on-demand
- **Container:** Docker/Podman with CUDA, PyTorch, Ultralytics, MMPose
- **Persistent storage:** Dataset + checkpoints
- **Checkpointing:** Save best + every 10 epochs (required for interruptible recovery)

### Evaluation Protocol

- **Primary metric:** AP (COCO-style, IoU 0.5:0.95) on skating val set
- **Secondary metric:** AP on AthletePose3D val (domain transfer check)
- **Guard:** Pretrained baseline always in comparison table
- **Per-joint AP:** Track worst joint (wrist/elbow typically weakest)

### Checkpointing

- Save best model (by skating val AP)
- Save every 10 epochs
- Config JSON for reproducibility
- Training log CSV (per-epoch metrics)

### Project Structure

```
experiments/yolo26-pose-kd/
├── configs/
│   ├── data.yaml              # Dataset paths
│   ├── stage2_finetune.yaml   # Fine-tune configs
│   └── stage3_distill.yaml    # KD configs
├── scripts/
│   ├── convert_finefs.py      # FineFS -> YOLO format
│   ├── convert_fsanno.py      # FSAnno -> YOLO format
│   ├── convert_fsc_mcfs.py    # FSC/MCFS -> YOLO format
│   ├── simulate_heatmap.py    # Keypoint -> simulated heatmap (MSRA)
│   ├── distill_trainer.py     # DistilPoseTrainer (Ultralytics)
│   └── eval_moganet.py        # MogaNet evaluation script
├── checkpoints/               # .gitignore - no commits
├── results/                   # Training logs, evaluation tables
└── README.md                  # Experiment tracking
```

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Fine-tune kills pretrained (again) | Medium | High | Stage 1 baseline check; early stopping; patience=20 |
| MogaNet-B domain gap on skating | Medium | Medium | Stage 2.5 teacher adaptation; AP3D contains some skating |
| Data conversion bugs | High | Medium | Spot-check 10 samples per dataset; visual overlay validation |
| Insufficient VRAM for teacher+student | Low | High | Use RTX 4090 24GB; if insufficient, offline teacher heatmaps |
| Simulated heatmap resolution mismatch | Medium | Medium | Match teacher heatmap resolution exactly; resize both to same grid |
| Skating annotations quality varies | Medium | Medium | Filter frames with < 5 visible keypoints; confidence weighting |

---

## 7. References

- DistilPose (CVPR 2023): "Tokenized Pose Regression with Heatmap Distillation" - github.com/yshMars/DistilPose
- YOLOv8-KD: github.com/KefanZhan/YOLOv8-KD (integration pattern)
- AthletePose3D (CVPRW 2025): MogaNet-B teacher, AP=0.962
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network" - temperature scaling
- Project POST-MORTEM: experiments/yolo26-pose/notes/POST-MORTEM.md (lessons learned)
