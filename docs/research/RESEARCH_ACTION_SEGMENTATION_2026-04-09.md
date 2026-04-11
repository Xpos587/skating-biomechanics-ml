# Figure Skating Action Segmentation — Research
**Date:** 2026-04-09
**Source:** https://github.com/mayupei/figure-skating-action-segmentation

---

## Overview

Automated action segmentation of figure skating competition videos (World Championships 2017-2019). Identifies start/end times of technical elements (jumps, spins, sequences) to automate replay operator work.

**Key result:** Two-stage LSTM-CNN achieves 0.89 F1@50 — massive improvement over LSTM alone (0.31 F1@50).

---

## Architecture

```
Skeleton (COCO 17kp, 3fps)
    → Normalize (center at origin, scale by vertical distance)
    → Sliding window (20 frames, step 10)
    → Stage 1: LSTM (64 units) → Dense → softmax (Jump/Spin/None)
        → 0.88 accuracy, 0.31 F1@50 (heavy over-segmentation)
    → Stage 2: Embedding → Conv1D(64, k=10) → Dropout → Conv1D → Dense
        → 0.92 accuracy, 0.89 F1@50 (temporal refinement fixes boundaries)
```

### Why two stages?
- LSTM alone produces noisy, over-segmented predictions (many short spurious segments)
- 1D-CNN learns temporal patterns of valid label sequences (e.g., jumps have minimum duration)
- The CNN essentially learns "what a realistic element boundary looks like"

---

## Data

| Dataset | Videos | Annotations | Skeleton Quality |
|---------|--------|-------------|------------------|
| **MCFS** | 271 | Frame-level labels (perfect) | Poor: 56% frames have missing joints, BODY_25 |
| **MMFS** | 1176 | Routine-level only | Clean: COCO 17kp, no missing data |
| **Fused** | 222 shared | MCFS labels + MMFS skeletons | Best of both |

### Label schema
- `"NONE"` = 0 (background, step sequences, transitions)
- `"Jump"` = 1 (avg 158 frames ≈ 5.3s at 30fps)
- `"Spin"` = 2 (avg 414 frames ≈ 13.8s at 30fps)

### Key stats
- Typical SP: 3 jumps + 3 spins
- Typical LP: 7-8 jumps + 3 spins
- Downsampled to 3fps (every 10th frame) — sufficient for segmentation

---

## Key Techniques to Borrow

### 1. Two-Stage Refinement Pattern
**Most important for our project.** Our `phase_detector.py` has the same over-segmentation problem. The LSTM→CNN pattern is directly applicable:

- **Stage 1 (LSTM):** Frame-wise phase prediction (takeoff/flight/landing/idle)
- **Stage 2 (1D-CNN):** Temporal refinement — learns valid phase sequences and minimum durations

### 2. F1@50 Evaluation Metric
`OverlapF1` class computes segment-wise F1 with IoU ≥ 0.5 threshold. This is the correct metric for temporal segmentation — our current phase detection has no proper evaluation framework.

### 3. Skeleton Normalization
```python
SELECTED_BONES = [
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (13, 15), (14, 16) # Legs
]
# Center at origin, scale by vertical distance
```
They focus on 6 key bone connections rather than all keypoints. Could adapt for biomechanically relevant connections.

### 4. Sliding Window for Variable-Length Input
```python
def sliding_window(skeleton_seq, label_seq, window_length=20, window_interval=10):
    # Creates fixed-size windows from variable-length routines
    # Overlap ensures no boundary is missed
```
Directly applicable to our DTW alignment — normalize element duration to fixed window size.

### 5. 3fps Downsampling
30fps → 3fps (every 10th frame). For action segmentation this is sufficient. Could speed up our pipeline significantly for segmentation tasks.

---

## Comparison with Our Phase Detection

| Aspect | Their Approach | Our `phase_detector.py` |
|--------|---------------|------------------------|
| **Method** | ML (LSTM + CNN) | Rule-based (CoM velocity + adaptive sigma) |
| **Input** | Skeleton keypoints (17kp) | CoM trajectory from 3D poses |
| **Over-segmentation** | Solved by Stage 2 CNN | Known issue (Phase 10 at 90%) |
| **Evaluation** | F1@50 with OverlapF1 class | Manual / visual inspection |
| **Granularity** | Jump/Spin/None (3 classes) | takeoff/flight/landing (per-jump) |
| **Generalization** | Trained on 222 competition videos | Works on any video with CoM |

### Verdict
Their ML approach is more robust and generalizable. Our rule-based approach works but struggles with edge cases. **Hybrid approach would be ideal:** ML segmentation for element boundaries, rule-based biomechanics for within-element analysis.

---

## Potential Integration Path

1. **Short term:** Adopt F1@50 metric and OverlapF1 class for evaluating our phase detection
2. **Medium term:** Train LSTM-CNN on our existing pose data (RTMPose → H3.6M) for element segmentation (Jump/Spin/Step/Idle)
3. **Long term:** Replace rule-based phase_detector with ML model, keep rule-based recommender for biomechanics feedback

---

## Code Components of Interest

| File | What to borrow |
|------|---------------|
| `data_prep.py` | `sliding_window()`, skeleton normalization, label encoding |
| `models.py` | Two-stage LSTM-CNN architecture (adaptable to Keras/PyTorch) |
| `metrics.py` | `OverlapF1` class for temporal segmentation evaluation |
| `visualize.py` | Skeleton visualization with element boundaries overlaid |
