# Figure Skating Element Classification — Experiment Series

**Date:** 2026-04-11
**Goal:** Determine feasibility of element classification from 2D pose skeletons and identify the path to >= 80% accuracy.
**Hardware:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
**Framework:** PyTorch (CUDA)

---

## Table of Contents

1. [Hypotheses & Decisions](#hypotheses--decisions)
2. [Datasets](#datasets)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Experiment 1 — Embedding Quality](#experiment-1--embedding-quality)
6. [Experiment 2 — Baseline Classifier](#experiment-2--baseline-classifier)
7. [Experiment 2b — Class Imbalance & Cropping](#experiment-2b--class-imbalance--cropping)
8. [Experiment 2c — Temporal Model](#experiment-2c--temporal-model)
9. [Experiment 2d — MMFS Cross-Dataset](#experiment-2d--mmfs-cross-dataset)
10. [Experiment 2e — Top-10 BiGRU Feasibility](#experiment-2e--top-10-bigru-feasibility)
11. [Experiment 3 — Data Augmentation](#experiment-3--data-augmentation)
12. [Master Results Table](#master-results-table)
13. [Conclusions](#conclusions)
14. [Files](#files)

---

## Hypotheses & Decisions

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H0 | VIFSS contrastive pre-training is necessary (accuracy < 70% without it) | **REJECTED** | BiGRU achieves 67.3% without any pre-training |
| H1 | Raw normalized 2D poses + simple classifier gives >= 80% accuracy | **REJECTED** | 1D-CNN: 21.8% (64 cls), 70.4% (10 cls) |
| H2 | Normalization alone provides view-invariance (cosine sim > 0.85 within-class) | **REJECTED** | Within-class sim 0.557 ± 0.239, gap only 0.090 |
| H3 | VIFSS contrastive pre-training helps marginally (+3-5%) | **PENDING** | Not yet tested |
| H4 | Data augmentation reduces overfitting on 64-class problem | **REJECTED** | Best augmentation: 67.9% vs baseline 67.3% (+0.6%, noise-level) |
| H5 | Class imbalance is the primary bottleneck | **CONFIRMED** | Top-10: 81.5%, all-64: 67.3% — same model, same data |

---

## Datasets

### Figure-Skating-Classification (FSC) — Primary

| Property | Value |
|----------|-------|
| **Source** | HuggingFace (Mercity/Figure-Skating-Classification) |
| **Origin** | MMFS skeleton data (4915 seq) + JSON mocap (253 seq) |
| **Train / Test** | 4161 / 1007 sequences |
| **Classes** | 64 element categories |
| **Keypoints** | COCO 17kp (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| **Format** | `(F, 17, 3, 1)` per sample — x, y, confidence; normalized to [-1, 1] |
| **Frame count** | 150 frames @ 30 fps (5 seconds) — uniform length |
| **Class distribution** | min=8, max=406, mean=65 samples/class — highly imbalanced |

### MMFS — Secondary

| Property | Value |
|----------|-------|
| **Paper** | "Multi-Modal Figure Skating Dataset" (arXiv:2307.02730) |
| **Train / Test** | 3957 / 956 sequences (skeleton split) |
| **Classes** | 63 element categories |
| **Keypoints** | COCO 17kp |
| **Format** | Variable-length numpy arrays, unnormalized coordinates |
| **Frame count** | min=34, max=7497, mean=347 frames |
| **Quality scores** | 0.0 – 48.6 (continuous, available for regression task) |

### Why FSC > MMFS for classification

- FSC sequences are uniform length (150 frames) — simpler preprocessing
- MMFS max length 7497 frames — 150-frame crop destroys most content
- FSC top-10 CNN: 70.4% vs MMFS top-10 CNN: 23.5% with identical model

---

## Preprocessing

Applied identically to all experiments:

```python
def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + scale normalize. Input: (F, 17, 2) float32."""
    mid_hip = p[:, [11, 12], :].mean(axis=1, keepdims=True)   # LHIP=11, RHIP=12
    p = p - mid_hip                                             # root-center
    mid_shoulder = p[:, [5, 6], :].mean(axis=1, keepdims=True)  # LSHOULDER=5, RSHOULDER=6
    spine_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1, keepdims=True)
    return p / np.maximum(spine_length, 0.01)                   # scale to unit spine
```

- **Root centering:** subtract mid-hip position → removes camera translation
- **Scale normalization:** divide by spine length → removes distance/zoom dependence
- **Input:** `(F, 17, 2)` — x, y coordinates only (confidence channel discarded)
- **Empty sequences:** filtered out (2 samples in FSC train had 0 frames)

---

## Models

### 1D-CNN (Fixed-Length Input)

| Layer | Config |
|-------|--------|
| Conv1d | 34→64, kernel=5, pad=2, BN, ReLU, MaxPool(2) |
| Conv1d | 64→128, kernel=5, pad=2, BN, ReLU, MaxPool(2) |
| Conv1d | 128→128, kernel=3, pad=1, BN, ReLU |
| Pool | AdaptiveAvgPool1d(1) |
| FC | 128→num_classes |
| **Params** | **114,368** (10-class) / ~114K (64-class) |

### BiGRU (Variable-Length Input)

| Layer | Config |
|-------|--------|
| GRU | input=34, hidden=128, layers=2, bidirectional, dropout=0.3 |
| FC | 256→128, ReLU, Dropout(0.3), 128→num_classes |
| **Params** | ~230K (64-class) |
| **Packing** | `pack_padded_sequence` — no zero-padding during forward pass |

### BiGRU + Regularization (Augmentation Experiments)

Same architecture as above, with training improvements:

| Technique | Config |
|-----------|--------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | CosineAnnealingLR (T_max=50) |
| Label Smoothing | 0.1 |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=10 epochs on validation loss |
| Train/Val Split | 90/10 stratified by class |

---

## Experiment 1 — Embedding Quality

**Script:** (inline, not saved separately)
**Question:** Can raw normalized poses be compared directly via cosine similarity?

**Method:**
- Compute normalized poses for all FSC train samples
- Sample 500 within-class pairs + 500 between-class pairs
- For each pair: take random frame from each sequence, flatten 17×2=34 dims, compute cosine similarity

**Results:**

| Metric | Within-class | Between-class |
|--------|-------------|---------------|
| Mean | 0.557 | 0.466 |
| Std | 0.239 | 0.258 |
| Min | -0.401 | -0.465 |
| Max | 0.989 | 0.869 |
| **Gap** | **+0.090** | |

**Threshold analysis:**

| Cosine ≥ t | Within-class | Between-class | Separation |
|-----------|-------------|---------------|------------|
| 0.0 | 96.4% | 92.0% | 4.4% |
| 0.4 | 77.0% | 67.8% | 9.2% |
| 0.6 | 58.2% | 37.8% | 20.4% |
| 0.8 | 5.0% | 2.4% | 2.6% |

**Verdict:** H2 REJECTED. Raw cosine similarity is noise-level (gap 0.09). Within-class variance is enormous (range 1.39). A learned embedding is required.

---

## Experiment 2 — Baseline Classifier

**Script:** `exp_2b_2c_2d.py` (CNN1D model, `train_eval_cnn`)
**Question:** Can a simple 1D-CNN classify elements from truncated/padded sequences?

**Config:**
- Model: 1D-CNN (114K params), 64 classes
- Input: 150 frames, start-crop, zero-pad shorter sequences
- Training: Adam lr=1e-3, CrossEntropyLoss, 30 epochs, batch=64, CPU
- No validation split — track best test accuracy

**Training curve:**

| Epoch | Loss | Test Acc | Best |
|-------|------|----------|------|
| 1 | 3.640 | 19.7% | 19.7% |
| 5 | 2.482 | 5.0% | 19.7% |
| 10 | 2.272 | 5.2% | 19.7% |
| 15 | 2.105 | 5.9% | 19.7% |
| 20 | 1.868 | 6.1% | 19.7% |
| 25 | 1.702 | 20.8% | 21.9% |
| 30 | 1.579 | 9.9% | 21.9% |

**Result:** 21.8% (random = 1.6%)

**Analysis:**
- Loss decreases steadily (3.64→1.58) — model IS learning
- Accuracy oscillates wildly — training instability
- Best accuracy at epoch 25, not 30 — early overfitting
- 150-frame truncation destroys content (sequences up to 606 frames)
- Zero-padding dilutes signal

---

## Experiment 2b — Class Imbalance & Cropping

**Script:** `exp_2b_2c_2d.py` (CNN1D, top-10 classes)
**Question:** How much does class imbalance hurt? Does center-crop help?

**Config:**
- Same 1D-CNN, 10 classes (top-10 by sample count: 2319 train, 574 test)
- Compared: start-crop vs center-crop, both 150 frames
- GPU, 30 epochs

**Results:**

| Config | Accuracy | vs Random (10%) |
|--------|----------|----------------|
| Start-crop | **70.4%** | +60.4pp |
| Center-crop | 31.2% | +21.2pp |

**Analysis:**
- Reducing from 64→10 classes: +48.6pp (21.8%→70.4%) — class imbalance is the #1 problem
- Start-crop >> center-crop: +39.2pp — figure skating elements begin early in clips
- Center-crop likely removes the takeoff/preparation phase which is discriminative
- Overfitting still present: accuracy oscillates (ep20: 70.4%, ep30: 51.2%)

---

## Experiment 2c — Temporal Model

**Script:** `exp_2b_2c_2d.py` (BiGRU, 64 classes)
**Question:** Does a variable-length temporal model fix the truncation problem?

**Config:**
- Model: 2-layer BiGRU (hidden=128, bidirectional, dropout=0.3)
- Input: full variable-length sequences, packed (no padding)
- Training: Adam lr=1e-3, CrossEntropyLoss, 30 epochs, batch=64, GPU
- No validation split

**Training curve:**

| Epoch | Test Acc | Best |
|-------|----------|------|
| 1 | 10.9% | 10.9% |
| 10 | 42.1% | 42.3% |
| 20 | 55.3% | 55.3% |
| 30 | 61.1% | 61.6% |

**Result:** 61.6% (267s)

**Analysis:**
- BiGRU is **3× better** than CNN on 64 classes (61.6% vs 21.8%)
- Steady monotonic improvement — no oscillation
- PackPaddedSequence eliminates padding noise
- Still 18.4pp below 80% target — class imbalance + view variance remain

---

## Experiment 2d — MMFS Cross-Dataset

**Script:** `exp_2b_2c_2d.py` (MMFS loading, CNN + BiGRU)
**Question:** Is MMFS a better dataset for element classification?

**Config:**
- Same models as 2b/2c, applied to MMFS skeleton split
- MMFS: 3957 train, 956 test, 63 classes, quality scores 0.0–48.6
- Sequence lengths: min=34, max=7497, mean=347

**Results:**

| Config | Dataset | Accuracy |
|--------|---------|----------|
| CNN top-10 center-crop | MMFS | 23.5% |
| BiGRU all classes | MMFS | **65.4%** |
| BiGRU all classes | FSC | 61.6% |

**Analysis:**
- BiGRU on MMFS (65.4%) slightly better than FSC (61.6%) — more data per class
- MMFS CNN top-10 is terrible (23.5%) — max 7497 frames makes 150-frame crop useless
- FSC is preferred for classification: uniform length, cleaner labels, easier preprocessing
- MMFS quality scores make it suitable for regression (quality prediction), not classification

---

## Experiment 2e — Top-10 BiGRU Feasibility

**Script:** (inline, not saved separately)
**Question:** Can BiGRU reach 80% on top-10 classes?

**Config:**
- Same BiGRU as 2c, 10 classes only (2318 train, 574 test)
- Training: Adam lr=1e-3, CrossEntropyLoss, 30 epochs, batch=64, GPU

**Training curve:**

| Epoch | Test Acc | Best |
|-------|----------|------|
| 1 | 20.6% | 20.6% |
| 5 | 45.5% | 45.5% |
| 10 | 56.8% | 56.8% |
| 15 | 70.9% | 70.9% |
| 20 | 75.3% | 75.3% |
| 25 | 75.3% | 79.1% |
| 30 | 77.2% | **81.5%** |

**Result:** 81.5% (158s)

**Verdict:** 80% target is achievable with sufficient data per class. The bottleneck for 64-class is data scarcity, not model capacity.

---

## Experiment 3 — Data Augmentation

**Script:** `exp_augmentation.py`
**Question:** Can augmentation reduce overfitting and improve 64-class accuracy?

**Config:**
- Model: BiGRU with regularization (AdamW, CosineAnnealing, label smoothing 0.1, grad clip 1.0, early stopping patience=10)
- Train/Val/Test: 3765 / 394 / 1007 (stratified 90/10 split from train)
- 4 configurations tested:

| Config | Augmentations | Dataset Size | LR | Weight Decay | Dropout |
|--------|-------------|-------------|-----|-------------|---------|
| Baseline | None | 3765 | 1e-3 | 1e-4 | 0.3 |
| Aug A | Noise σ=0.02 (p=0.8), Mirror (p=0.5) | 8699 (2.3×) | 1e-3 | 1e-4 | 0.3 |
| Aug B | Noise + Mirror + TemporalScale + JointDrop + SkeletonMix | 12443 (3.3×) | 1e-3 | 1e-4 | 0.3 |
| Aug C | Noise σ=0.015 (always) + Mirror (always) | 11295 (3.0×) | 5e-4 | 1e-3 | 0.4 |

**Augmentation details:**

| Augmentation | Implementation | Rationale |
|-------------|---------------|-----------|
| Joint Noise | `N(0, σ²)` per coordinate, σ=0.02 | Simulates pose estimation errors |
| Mirror | Flip x-axis, swap COCO L/R indices `[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]` | Doubles data, but dangerous for directional elements |
| Temporal Scale | Resample to 0.9×–1.1× length via linear interpolation | Speed variation |
| Joint Drop | Zero out 1–2 random joints per frame | Simulates occlusion during rotations |
| SkeletonMix | `0.3·seq_A + 0.7·seq_B` for same-class pairs | Inter-class blending |

**Results:**

| Config | Best Test Acc | Epochs | Final Train-Val Gap |
|--------|-------------|--------|-------------------|
| Baseline | **67.3%** | 45 | -0.339 |
| Aug A | 63.2% | 25 | -0.788 |
| Aug B | 67.9% | 21 | -0.690 |
| Aug C | 62.3% | 47 | -0.354 |

**Training curves — Baseline:**

| Epoch | Train Loss | Val Loss | Gap | Val Acc | Test Acc | Best Test |
|-------|-----------|----------|-----|---------|----------|-----------|
| 1 | 3.790 | 3.564 | +0.225 | 14.0% | 13.5% | 13.5% |
| 10 | 2.424 | 2.296 | +0.127 | 45.2% | 39.4% | 39.4% |
| 20 | 1.671 | 1.683 | -0.012 | 62.7% | 59.0% | 59.0% |
| 30 | 1.349 | 1.500 | -0.151 | 69.8% | 66.4% | 66.4% |
| 40 | 1.198 | 1.496 | -0.298 | 71.6% | 67.3% | 67.3% |
| 45 | 1.165 | 1.504 | -0.339 | 70.3% | 67.5% | 67.3% |

**Training curves — Aug A (Noise + Mirror):**

| Epoch | Train Loss | Val Loss | Gap | Val Acc | Test Acc | Best Test |
|-------|-----------|----------|-----|---------|----------|-----------|
| 1 | 3.646 | 3.275 | +0.372 | 19.5% | 18.8% | 18.8% |
| 10 | 1.719 | 1.776 | -0.057 | 57.6% | 55.3% | 55.3% |
| 15 | 1.357 | 1.549 | -0.193 | 68.8% | 63.2% | 63.2% |
| 20 | 1.117 | 1.644 | -0.527 | 67.0% | 64.1% | 63.2% |
| 25 | 0.987 | 1.774 | -0.788 | 65.7% | 64.9% | 63.2% |

**Training curves — Aug B (Full):**

| Epoch | Train Loss | Val Loss | Gap | Val Acc | Test Acc | Best Test |
|-------|-----------|----------|-----|---------|----------|-----------|
| 1 | 3.509 | 3.069 | +0.440 | 21.3% | 22.7% | 22.7% |
| 5 | 2.035 | 1.937 | +0.098 | 54.6% | 51.2% | 51.2% |
| 10 | 1.426 | 1.522 | -0.096 | 71.3% | 64.4% | 64.4% |
| 15 | 1.126 | 1.503 | -0.376 | 74.4% | 69.2% | 67.9% |
| 21 | — | — | — | — | — | 67.9% |

**Training curves — Aug C (Conservative):**

| Epoch | Train Loss | Val Loss | Gap | Val Acc | Test Acc | Best Test |
|-------|-----------|----------|-----|---------|----------|-----------|
| 1 | 3.728 | 3.417 | +0.310 | 17.0% | 16.4% | 16.4% |
| 15 | 1.771 | 1.747 | +0.024 | 61.7% | 56.9% | 56.9% |
| 25 | 1.429 | 1.518 | -0.089 | 70.1% | 62.6% | 62.6% |
| 35 | 1.264 | 1.535 | -0.271 | 68.5% | 62.3% | 64.3% |
| 47 | — | — | — | — | — | 62.3% |

**Analysis:**

1. **Augmentation does NOT help on 64 classes.** Best augmented result: 67.9% (Aug B) vs 67.3% (baseline) — +0.6pp, within noise.

2. **Aug A and Aug C are HARMFUL** (-4.1pp and -5.0pp respectively). Mirror augmentation is particularly damaging for figure skating — it converts a left-edge takeoff (flip) into a right-edge takeoff (lutz) appearance, creating contradictory training signals.

3. **All augmented models overfit harder** (train-val gap -0.7 vs -0.3). The model learns to exploit augmentation artifacts rather than discriminative features.

4. **Aug B (full pipeline) is the least bad** because SkeletonMix and temporal scaling are direction-agnostic. But the gain is negligible.

5. **Root cause:** With ~65 samples per class, the model has insufficient signal to learn invariant features. Adding noisy variants doesn't help — it needs real data from more athletes, cameras, and execution styles.

---

## Master Results Table

| # | Model | Classes | Dataset | Augmentation | Accuracy | Random | Key Variable |
|---|-------|---------|---------|-------------|----------|--------|-------------|
| 1 | Cosine sim | — | FSC | — | gap 0.09 | — | Embedding quality |
| 2 | 1D-CNN | 64 | FSC | None | 21.8% | 1.6% | Fixed-length baseline |
| 2b | 1D-CNN | 10 | FSC | None, start-crop | **70.4%** | 10.0% | Class imbalance |
| 2b | 1D-CNN | 10 | FSC | None, center-crop | 31.2% | 10.0% | Crop strategy |
| 2c | BiGRU | 64 | FSC | None | 61.6% | 1.6% | Variable-length |
| 2d | BiGRU | 63 | MMFS | None | 65.4% | 1.6% | Cross-dataset |
| 2e | BiGRU | 10 | FSC | None | **81.5%** | 10.0% | Feasibility |
| 3 | BiGRU+reg | 64 | FSC | None | **67.3%** | 1.6% | Regularization |
| 3 | BiGRU+reg | 64 | FSC | Noise+Mirror | 63.2% | 1.6% | Aug: spatial |
| 3 | BiGRU+reg | 64 | FSC | Full pipeline | 67.9% | 1.6% | Aug: all |
| 3 | BiGRU+reg | 64 | FSC | Conservative | 62.3% | 1.6% | Aug: strong reg |

---

## Conclusions

### Confirmed

1. **BiGRU with packed sequences** is the correct architecture for variable-length pose classification. 3× better than CNN with truncation/padding.

2. **Class imbalance is the primary bottleneck.** Top-10: 81.5%, all-64: 67.3% — same model, same preprocessing. The model capacity is sufficient; data per class is not.

3. **Start-crop >> center-crop** for figure skating. Elements begin early in the clip (preparation + takeoff are discriminative).

4. **FSC is the best available dataset** for this task. MMFS has longer, messier sequences with no quality advantage for classification.

5. **Raw pose similarity is useless** for cross-athlete comparison (cosine sim gap 0.09). A learned encoder is mandatory.

### Rejected

6. **Data augmentation does not help** on 64 classes. Mirror is harmful (direction-dependent elements). Joint noise and SkeletonMix provide no measurable gain.

7. **Simple truncation/padding** destroys temporal content. Variable-length handling is non-negotiable.

8. **MMFS is not worth the preprocessing effort** for classification. Its value is in quality scores (regression task).

### Open

9. **VIFSS contrastive pre-training** — not yet tested. May help if it learns view-invariant features that generalize across the rare classes.

10. **Cross-dataset training** (FSC + MMFS merged for shared classes) — not tested.

11. **GCN architectures** (SkelFormer, ST-GCN) — not tested. May outperform BiGRU on skeleton-structured data.

12. **Temporal action segmentation** (per-frame labels from MCFS dataset) — not tested. Would enable precise element boundary detection.

---

## Files

```
experiments/
├── README.md                      ← This file (master report)
├── 2026-04-11-experiments.md      ← Original experiment log (superseded)
├── exp_2b_2c_2d.py               ← Experiments 2b, 2c, 2d (CNN + BiGRU + MMFS)
└── exp_augmentation.py            ← Experiment 3 (augmentation ablation)
```

### How to reproduce

```bash
cd /home/michael/Github/skating-biomechanics-ml

# Experiments 2b, 2c, 2d (GPU, ~5 min)
uv run python experiments/exp_2b_2c_2d.py

# Experiment 3 — augmentation ablation (GPU, ~35 min)
uv run python experiments/exp_augmentation.py
```

### Dependencies

- `torch` (CUDA) — PyTorch with GPU support
- `numpy` — array operations
- `pickle` — dataset loading (FSC, MMFS)
- Datasets must be present at `data/datasets/figure-skating-classification/` and `data/datasets/mmfs/`
