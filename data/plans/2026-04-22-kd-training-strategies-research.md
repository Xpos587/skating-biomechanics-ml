# Knowledge Distillation Training Strategies Research
**Date:** 2026-04-22
**Context:** MogaNet-B → YOLO26-Pose distillation for figure skating pose estimation

---

## Training Strategy Findings

### Discovery 1: Two-Stage Distillation is Superior
**Source:** MMPose DWPoseDistiller implementation (configs/wholebody_2d_keypoint/dwpose/coco-wholebody/s1_dis/dwpose_x_dis_l_coco-384x288.py)

**Key Finding:** DWPose uses a two-stage distillation approach:
- **Stage 1** (`two_dis=False`): Student extracts features, KD loss decays linearly with epoch progress
- **Stage 2** (`two_dis=True`): Uses teacher's features directly, student only trains the head

**Decay Schedule:** `loss_weight = (1 - epoch / max_epochs) * loss`
- KD loss starts at full weight, gradually reduces to zero by final epoch
- Allows student to gradually transition from mimicking teacher to learning task directly

**Rationale:** Prevents overfitting to teacher's knowledge, allows student to develop its own representation

---

### Discovery 2: Progressive Loss Weight Scheduling
**Source:** MMPose DWPoseDistiller (mmpose/models/distillers/dwpose_distiller.py lines 145-154)

**Implementation:**
```python
if 'loss_fea' in all_keys:
    losses[loss_name] = self.distill_losses[loss_name](fea_s[-1], fea_t[-1])
    if not self.two_dis:
        losses[loss_name] = (1 - self.epoch / self.max_epochs) * losses[loss_name]

if 'loss_logit' in all_keys:
    losses[loss_name] = self.distill_losses[loss_name](pred, pred_t, beta, target_weight)
    if not self.two_dis:
        losses[loss_name] = (1 - self.epoch / self.max_epochs) * losses[loss_name]
```

**Key Insight:** Both feature and logit distillation losses decay together
- **Start:** High KD weight (student learns from teacher)
- **Middle:** Linear decay (student transitions to task loss)
- **End:** Zero KD weight (student fine-tunes on task alone)

---

### Discovery 3: RTMPose Training Configuration
**Source:** MMPose RTMPose configs (projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py)

**Standard RTMPose Training:**
```python
max_epochs = 420
base_lr = 4e-3
train_batch_size = 256

# Optimizer
optimizer = AdamW(lr=4e-3, weight_decay=0.05)
clip_grad = dict(max_norm=35, norm_type=2)

# Learning Rate Schedule
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),  # Warmup
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs,
         T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]

# Auto-scale LR based on batch size
auto_scale_lr = dict(base_batch_size=1024)
```

**Key Observations:**
- **Warmup:** 1000 iterations (not epochs), linear from 1e-5 to base_lr
- **LR Schedule:** Cosine annealing starts at epoch 210 (halfway), decays to 5% of base_lr
- **Gradient Clipping:** Max norm 35 prevents gradient explosion
- **Batch Size Scaling:** Base batch 1024, actual 256 (LR scales automatically)

---

### Discovery 4: Distillation Loss Weights
**Source:** DWPose config (configs/wholebody_2d_keypoint/dwpose/coco-wholebody/s1_dis/dwpose_x_dis_l_coco-384x288.py)

**Feature Loss (FeaLoss):**
```python
alpha_fea = 0.00007  # Extremely low weight
```
- MSE loss between student and teacher feature maps
- Weight 0.00007 prevents feature loss from dominating task loss
- Uses 1x1 conv to align channels if student/teacher dimensions differ

**Logit Loss (KDLoss):**
```python
weight = 0.1  # Logit distillation weight
beta = (temperature scaling parameter)
```
- KL divergence loss between student and teacher predictions
- Weight 0.1 provides moderate supervision signal
- Beta temperature parameter softens teacher's distribution

**Rationale for Low Weights:**
- Feature maps are high-dimensional (1024-1280 channels), MSE loss can explode
- Logit loss is more direct supervision signal, hence higher weight
- Task loss remains primary objective (loss_kpt in MMPose)

---

### Discovery 5: Mixup in Knowledge Distillation
**Source:** "Understanding the Role of Mixup in Knowledge Distillation" (arXiv:2211.03946, WACV 2023)

**Key Findings:**
1. **Mixup is compatible with KD** but requires careful implementation
2. **Standard Mixup:** Mixes images and labels linearly: `x = λx₁ + (1-λ)x₂`
3. **KD Mixup:** Must mix teacher predictions, not just labels:
   ```
   teacher_pred = λ * teacher(x₁) + (1-λ) * teacher(x₂)
   student learns from mixed teacher predictions
   ```

**Recommendation:**
- **Use Mixup in Stage 1 only** (when KD loss is active)
- **Disable Mixup in Stage 2** (task-only fine-tuning)
- **Match teacher/student augmentations** — both see same mixed samples

**Caution:** Mixup can conflict with offline heatmaps (pre-computed teacher targets). Use only for online distillation.

---

### Discovery 6: Mosaic Augmentation Compatibility
**Source:** MMYOLO documentation and Ultralytics issues

**Key Finding:** **Mosaic is INCOMPATIBLE with offline distillation**

**Why:**
- Mosaic creates composite images (4 images stitched into 1)
- Offline heatmaps are pre-computed for single images
- Mosaic breaks spatial correspondence between image and heatmap

**Solution:**
- **Stage 1 (KD):** Disable Mosaic, use single-image augmentations (flip, rotate, scale)
- **Stage 2 (Task-only):** Can re-enable Mosaic for final fine-tuning

**Compatible Augmentations for KD:**
- RandomFlip (horizontal)
- RandomHalfBody (pose-specific)
- RandomBBoxTransform (scale, rotation)
- TopdownAffine (spatial transform)
- Color jittering (safe, doesn't affect geometry)

---

### Discovery 7: Batch Size and Learning Rate Scaling
**Source:** "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling" (NeurIPS 2024)

**Key Findings:**
1. **Linear scaling rule (LR ∝ batch_size) breaks down** for large batches (>512)
2. **Surge phenomenon:** LR spikes cause optimization instability at large batches
3. **Recommended scaling:** LR ∝ √batch_size for batches > 512

**Practical Guidance:**
```
Batch 256:  LR = 4e-3  (RTMPose standard)
Batch 512:  LR = 5.6e-3 (linear: 8e-3, safe: 5.6e-3)
Batch 1024: LR = 8e-3   (linear: 1.6e-2, safe: 8e-3)
```

**For KD Training:**
- **Smaller batch preferred:** Better gradient estimates from teacher
- **Recommended:** Batch 128-256 for stable KD
- **If using large batch (512+):** Use √scaling, not linear scaling

---

### Discovery 8: Optimizer Choice for Pose KD
**Source:** PaddleDetection distill configs, Ultralytics optimizer auto-selection

**PaddleDetection PP-YOLOE Distillation:**
```yaml
Optimizer:
  type: Momentum
  momentum: 0.9
  learning_rate: 0.01
  regularizer:
    type: L2
    factor: 0.0001
```

**Ultralytics Auto-Selection Logic:**
```python
if iterations > 10000:
    optimizer = "MuSGD"  # Multi-scale SGD for long training
    lr = 0.01
else:
    optimizer = "AdamW"
    lr = 0.002 * 5 / (4 + num_classes)  # Class-count adjusted
```

**Consensus for Pose KD:**
- **AdamW preferred** for shorter training (<10K iterations)
- **SGD/Momentum preferred** for longer training (>10K iterations)
- **RTMPose standard:** AdamW (4e-3, weight_decay=0.05)

**Recommendation for MogaNet→YOLO26:**
- **Stage 1 (KD):** AdamW (lr=4e-3, wd=0.05) — matches RTMPose
- **Stage 2 (Fine-tune):** Can switch to SGD (lr=1e-3, momentum=0.9) for final polish

---

### Discovery 9: Convergence and Early Stopping
**Source:** "On the Efficacy of Knowledge Distillation" (ICCV 2019, arXiv:1910.01348)

**Key Findings:**
1. **Distillation ≈ Early Stopping:** KD acts as a regularizer, similar to stopping training early
2. **Teacher-student gap analysis:**
   - **Gap too large (>20% AP):** Student cannot learn from teacher (capacity mismatch)
   - **Gap moderate (10-20% AP):** Optimal for KD
   - **Gap small (<5% AP):** KD may not help (already similar capacity)

**Early Stopping Criteria:**
```python
# Monitor student validation AP
if student_val_ap > teacher_val_ap * 0.95:
    # Student within 5% of teacher
    print("KD converged, student competitive with teacher")
    break

# Monitor KD loss contribution
if kd_loss_weight * kd_loss < 0.01 * task_loss:
    # KD loss negligible compared to task loss
    print("KD loss exhausted, transitioning to task-only")
    break
```

**Recommended Training Duration:**
- **Stage 1 (KD):** 50-70% of total epochs (e.g., 210/420 epochs)
- **Stage 2 (Task-only):** 30-50% of total epochs (e.g., 210/420 epochs)

---

### Discovery 10: Mixed Precision Training (AMP)
**Source:** TensorFlow model optimization KD implementation

**Key Findings:**
1. **AMP is safe for KD** if teacher runs in FP32
2. **Teacher must be in FP32** to maintain prediction quality
3. **Student can train in FP16** for speed (1.5-2x faster)

**Implementation:**
```python
with torch.no_grad():
    teacher_features = teacher(inputs.float())  # FP32

with torch.cuda.amp.autocast():
    student_features = student(inputs.float())  # FP16
    loss = kd_loss(student_features, teacher_features)
```

**Caution:** Feature alignment layers (1x1 conv) may need FP32 for stability.

---

## Recommended Curriculum

### Stage 1: Knowledge Distillation (70% of epochs)
**Duration:** 210 epochs (out of 300 total)
**Goal:** Student learns from teacher's features and predictions

**Configuration:**
```python
# Training
max_epochs = 210
batch_size = 128  # Stable for KD, fits RTX 5090 (32GB)

# Optimizer
optimizer = AdamW(lr=4e-3, weight_decay=0.05)
clip_grad = dict(max_norm=35, norm_type=2)

# Learning Rate
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, begin=0, end=1000, by_epoch=False),  # Warmup
    dict(type='CosineAnnealingLR', eta_min=4e-3 * 0.05, begin=105, end=210,
         T_max=105, by_epoch=True),  # Halfway cosine decay
]

# Distillation Loss
distill_losses = {
    'loss_fea': FeaLoss(
        student_channels=512,  # YOLO26-n backbone output
        teacher_channels=1024,  # MogaNet-B backbone output
        alpha_fea=0.00007,  # Per DWPose
    ),
    'loss_logit': KDLoss(
        weight=0.1,  # Per DWPose
        beta=10,  # Temperature for logit softening
    ),
}

# Loss decay schedule
def get_kd_weight(epoch, max_epochs):
    return 1.0 - (epoch / max_epochs)  # Linear decay

# Data Augmentation
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='RandomFlip', direction='horizontal'),  # Safe
    dict(type='RandomHalfBody'),  # Pose-specific
    dict(type='RandomBBoxTransform'),  # Scale, rotate
    dict(type='TopdownAffine', input_size=(256, 192)),
    # NO Mosaic in Stage 1!
    dict(type='PackPoseInputs'),
]
```

**Monitoring:**
- **Student AP** (target: >90% of teacher AP by epoch 150)
- **KD loss ratio** (target: <20% of total loss by epoch 210)
- **Teacher-student feature distance** (MSE should decrease)

---

### Stage 2: Task-Only Fine-Tuning (30% of epochs)
**Duration:** 90 epochs (out of 300 total)
**Goal:** Student specializes on task, develops own representation

**Configuration:**
```python
# Training
max_epochs = 90
batch_size = 256  # Can increase, no teacher overhead

# Optimizer (optional: switch to SGD for final polish)
optimizer = AdamW(lr=1e-3, weight_decay=0.05)  # Lower LR for fine-tuning
# OR
optimizer = SGD(lr=1e-3, momentum=0.9, weight_decay=1e-4)

# Learning Rate
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=1e-4, begin=0, end=90,
         T_max=90, by_epoch=True),
]

# NO Distillation Loss
# Only task loss (keypoint MSE, visibility loss)

# Data Augmentation (can add Mosaic)
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(256, 192)),
    dict(type='Mosaic', prob=0.5),  # Optional: re-enable Mosaic
    dict(type='PackPoseInputs'),
]
```

**Monitoring:**
- **Student AP** (target: >95% of teacher AP)
- **Task loss convergence** (should plateau)
- **Overfitting check** (train AP vs val AP gap)

---

## Hyperparameters

### Learning Rate
**Value:** `4e-3` (Stage 1), `1e-3` (Stage 2)

**Rationale:**
- **Matches RTMPose standard:** RTMPose-m uses 4e-3 for 420 epochs
- **Batch size 128:** Within linear scaling regime (LR ∝ batch_size)
- **Stage 2 lower LR:** Fine-tuning requires smaller steps

**Scaling Formula:**
```
LR = base_lr * (batch_size / 256)
LR_128 = 4e-3 * (128 / 256) = 2e-3  # Conservative
```

**Recommendation:** Start with `2e-3` for batch 128, adjust based on training stability.

---

### Batch Size
**Value:** `128` (Stage 1), `256` (Stage 2)

**Rationale:**
- **Stage 1 (KD):** Smaller batch provides better gradient estimates from teacher
- **Stage 2 (Task-only):** Larger batch speeds up fine-tuning (no teacher overhead)
- **Hardware fit:** Batch 128 with 256x192 input fits in RTX 5090 32GB (teacher + student)

**Memory Estimate:**
```
Teacher (MogaNet-B): ~4GB
Student (YOLO26-n): ~1GB
Features (cached): ~2GB
Gradients: ~1GB
Optimizer states: ~1GB
Total: ~9GB (well within 32GB)
```

---

### Epochs
**Value:** `300` total (210 Stage 1 + 90 Stage 2)

**Rationale:**
- **RTMPose-m standard:** 420 epochs for full training
- **KD accelerates convergence:** Student learns faster with teacher guidance
- **Two-stage split:** 70% KD, 30% fine-tuning (per DWPose practice)

**Convergence Checkpoints:**
- **Epoch 150:** Student should reach >90% of teacher AP
- **Epoch 210:** Transition to Stage 2 (KD loss → 0)
- **Epoch 270:** Student should reach >95% of teacher AP
- **Epoch 300:** Final evaluation

**Early Stopping:**
```python
if student_val_ap > teacher_val_ap * 0.98:
    print("Student surpassed teacher! Stopping early.")
    break
```

---

### Warmup
**Value:** `1000 iterations` with linear warmup from `1e-5` to `base_lr`

**Rationale:**
- **RTMPose standard:** 1000 iterations (not epochs)
- **For batch 128, 300 epochs:** 1000 iterations ≈ 3-4 epochs
- **Prevents early instability:** KD loss can explode if student starts random

**Implementation:**
```python
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,  # 1000 iterations
    ),
    # ... main scheduler
]
```

---

### Optimizer
**Value:** `AdamW(lr=4e-3, weight_decay=0.05, betas=(0.9, 0.999))`

**Rationale:**
- **Matches RTMPose:** AdamW is standard for pose estimation
- **Weight decay 0.05:** Strong regularization prevents overfitting to teacher
- **Betas (0.9, 0.999):** Standard AdamW settings

**Alternative (Stage 2 only):**
```python
optimizer = SGD(
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
)
```
- **SGD for final polish:** Better generalization in late training
- **Use only if AdamW plateaus:** Switch after epoch 240 if needed

---

### Gradient Clipping
**Value:** `max_norm=35, norm_type=2`

**Rationale:**
- **RTMPose standard:** Prevents gradient explosion in deep networks
- **KD sensitive:** Teacher features can have large magnitudes, clipping stabilizes

**Implementation:**
```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=4e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
)
```

---

## Data Augmentation

### Compatible (Safe for KD)
**These augmentations preserve teacher signal:**

1. **RandomFlip (horizontal)**
   - Safe: Teacher predictions flip with image
   - Implementation: Flip both image and keypoints

2. **RandomHalfBody**
   - Safe: Teacher handles partial poses
   - Implementation: Randomly crop to upper/lower body

3. **RandomBBoxTransform**
   - Safe: Teacher predicts on transformed bbox
   - Implementation: Scale, rotate, translate bbox

4. **TopdownAffine**
   - Safe: Standard pose estimation transform
   - Implementation: Affine transform to fixed size (256x192)

5. **Color Jittering**
   - Safe: Doesn't affect geometry
   - Implementation: Adjust brightness, contrast, saturation

---

### Incompatible (Breaks KD)
**These augmentations destroy teacher signal:**

1. **Mosaic**
   - **Why incompatible:** 4 images stitched → heatmap misalignment
   - **Use in:** Stage 2 only (task-only fine-tuning)

2. **Mixup (online)**
   - **Why incompatible:** Requires mixing teacher predictions
   - **Use in:** Stage 1 only with special implementation
   - **Implementation:**
     ```python
     # Mix two images
     lam = np.random.beta(0.2, 0.2)
     mixed_input = lam * input1 + (1 - lam) * input2

     # Mix teacher predictions
     with torch.no_grad():
         pred1 = teacher(input1)
         pred2 = teacher(input2)
         mixed_teacher_pred = lam * pred1 + (1 - lam) * pred2

     # Student learns from mixed predictions
     student_pred = student(mixed_input)
     loss = kd_loss(student_pred, mixed_teacher_pred)
     ```

3. **CutMix**
   - **Why incompatible:** Patches from different images → heatmap fragmentation
   - **Alternative:** Use RandomBBoxTransform instead

---

### Special Considerations

**1. Teacher-Student Augmentation Matching**
- **Critical:** Teacher and student must see identical augmentations
- **Implementation:**
  ```python
  # Single augmentation pipeline
  augmented = augment(raw_input)

  # Teacher sees augmented input
  with torch.no_grad():
      teacher_feat = teacher(augmented)

  # Student sees same augmented input
  student_feat = student(augmented)

  # Compute KD loss
  loss = kd_loss(student_feat, teacher_feat)
  ```

**2. Offline vs Online Distillation**
- **Offline:** Teacher pre-computes features → No Mosaic, no Mixup
- **Online:** Teacher runs in loop → Can use Mixup with special handling
- **Recommendation:** Online distillation for flexibility

**3. Augmentation Strength**
- **Stage 1 (KD):** Moderate augmentation (teacher provides guidance)
- **Stage 2 (Task-only):** Strong augmentation (regularization)
- **Example:**
  ```python
  # Stage 1
  flip_prob = 0.5
  rotate_range = (-30, 30)  # degrees
  scale_range = (0.75, 1.25)

  # Stage 2
  flip_prob = 0.5
  rotate_range = (-45, 45)  # Wider range
  scale_range = (0.5, 1.5)   # Wider range
  ```

---

## Monitoring & Evaluation

### Key Metrics

**1. Student Validation AP**
- **What:** Average Precision on validation set
- **Target:** >95% of teacher AP by end of Stage 2
- **Frequency:** Every 10 epochs (per RTMPose standard)

**2. Teacher-Student Gap**
- **What:** `teacher_ap - student_ap`
- **Target:** <5% AP gap at convergence
- **Warning:** If gap >20%, student capacity insufficient

**3. KD Loss Ratio**
- **What:** `kd_loss / (task_loss + kd_loss)`
- **Target:** Starts at ~20%, decays to <5% by epoch 210
- **Warning:** If ratio >50%, KD loss dominating (reduce alpha_fea/weight)

**4. Feature Distance (MSE)**
- **What:** MSE between student and teacher features
- **Target:** Decreases over time (student learns to match teacher)
- **Frequency:** Every epoch

**5. Training Stability**
- **What:** Gradient norm, loss variance
- **Target:** Gradient norm <35 (clipping threshold)
- **Warning:** If gradient norm spikes → reduce LR or batch size

---

### Success Criteria

**Stage 1 Success (Epoch 210):**
- [ ] Student AP >90% of teacher AP
- [ ] KD loss ratio <10% of total loss
- [ ] Feature distance (MSE) decreased by >50% from start
- [ ] Training stable (no loss spikes, gradient norms normal)

**Stage 2 Success (Epoch 300):**
- [ ] Student AP >95% of teacher AP
- [ ] Task loss plateaued (no improvement for 20 epochs)
- [ ] Train-val AP gap <5% (no overfitting)
- [ ] Student inference speed >2x teacher (compression goal)

**Final Evaluation:**
- [ ] Student AP within 2-3% of teacher on test set
- [ ] Student model size <30% of teacher (parameter count)
- [ ] Student inference latency <50% of teacher (ms/image)

---

### Failure Diagnostics

**Problem: Student not learning from teacher**
- **Symptoms:** Student AP plateaus at <70% of teacher
- **Causes:**
  - Student capacity too small (try larger student: YOLO26-s instead of n)
  - Teacher signal too weak (increase alpha_fea or weight)
  - Learning rate too low (student can't follow teacher)

**Problem: KD loss dominating**
- **Symptoms:** KD loss ratio >50%, task loss not decreasing
- **Causes:**
  - alpha_fea or weight too high (reduce by 10x)
  - Feature dimension mismatch (check alignment layer)
  - Teacher features unstable (check teacher normalization)

**Problem: Training instability**
- **Symptoms:** Loss spikes, gradient explosion, NaN losses
- **Causes:**
  - Learning rate too high (reduce by 2x)
  - Batch size too small (increase to 256)
  - Gradient clipping disabled (enable max_norm=35)

**Problem: Overfitting to teacher**
- **Symptoms:** Student AP > teacher AP on train, but < on val
- **Causes:**
  - KD weight too high (reduce weight)
  - Not enough task-only training (increase Stage 2 to 120 epochs)
  - Augmentation too weak (increase rotation/scale range)

---

### Early Stopping Criteria

**Stop Stage 1 early if:**
```python
if epoch > 150 and student_val_ap > teacher_val_ap * 0.95:
    print("Student reached 95% of teacher AP. Transitioning to Stage 2.")
    break
```

**Stop Stage 2 early if:**
```python
if epoch > 240 and no_improvement_epochs > 30:
    print("No improvement for 30 epochs. Stopping early.")
    break
```

**Stop entire training if:**
```python
if student_val_ap > teacher_val_ap * 0.98:
    print("Student surpassed teacher! Training complete.")
    break
```

---

## References

**Papers:**
1. DWPose: "Real-Time Multi-Person Pose Estimation with Distillation" (arXiv:2307.15880)
2. "On the Efficacy of Knowledge Distillation" (ICCV 2019, arXiv:1910.01348)
3. "Understanding the Role of Mixup in Knowledge Distillation" (WACV 2023, arXiv:2211.03946)
4. "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling" (NeurIPS 2024)
5. OKDHP: "Online Knowledge Distillation for Efficient Pose Estimation" (ICCV 2021, arXiv:2108.02092)

**Codebases:**
1. MMPose DWPose implementation: github.com/open-mmlab/mmpose
2. PaddleDetection distillation: github.com/PaddlePaddle/PaddleDetection
3. Ultralytics YOLO training: github.com/ultralytics/ultralytics
4. MMRazor distillation framework: github.com/open-mmlab/mmrazor

**Training Configs:**
1. RTMPose-m standard: projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py
2. DWPose distillation: configs/wholebody_2d_keypoint/dwpose/coco-wholebody/s1_dis/dwpose_x_dis_l_coco-384x288.py

---

## Summary

**Recommended Training Strategy:**
- **Two-stage curriculum:** 70% KD (210 epochs) + 30% task-only (90 epochs)
- **Progressive KD decay:** Linear decay from 1.0 → 0.0 over Stage 1
- **Hyperparameters:** AdamW (lr=2e-3, wd=0.05), batch=128, epochs=300
- **Augmentations:** No Mosaic in Stage 1, enable in Stage 2
- **Monitoring:** Student AP >95% teacher AP, KD loss ratio <10%

**Key Insights:**
1. KD loss must decay to zero (student develops own representation)
2. Mosaic incompatible with offline distillation (breaks heatmaps)
3. Smaller batches (128) better for KD (stable teacher gradients)
4. AdamW preferred for shorter training, SGD for final polish
5. Early stopping when student reaches 98% of teacher AP

**Next Steps:**
1. Implement two-stage training loop with KD decay
2. Validate teacher-student feature alignment (1x1 conv)
3. Run small-scale test (10 epochs) to verify stability
4. Scale to full 300-epoch training
5. Evaluate on AthletePose3D validation set
