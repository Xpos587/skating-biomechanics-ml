# Pre-flight Check Results

**Date:** 2026-04-21
**Purpose:** Validation before Vast.ai rental

---

## 1. FineFS Quality Check ✅

**Agent:** FineFS Quality Check
**Files analyzed:** 10 random NPZ from `/home/michael/Downloads/FineFS/data/skeleton/`

### Findings:

| Metric | Result |
|--------|--------|
| Shape | (T, 17, 3) |
| Frame count | 4225-6450 per video |
| Coordinate range | x=[-0.89, 0.90], y=[-0.82, 0.98], z=[0.00, 1.92] |
| Coordinate system | Camera-relative (y-down, z+ depth) |
| NaN/Inf | **0** (perfect) |
| Keypoint visibility | **94.1%** (5.9% occluded) |
| H3.6M compatibility | ✅ Exact match |
| Temporal stability | ✅ Verified |

### Per-file details:

| File | Frames | Avg visible KP | Frames <5 KP |
|------|--------|----------------|--------------|
| 0.npz | 4350 | 16.0/17 | 0% |
| 1.npz | 4450 | 16.0/17 | 0% |
| 10.npz | 4300 | 16.0/17 | 0% |
| 100.npz | 4225 | 16.0/17 | 0% |
| 1000.npz | 6350 | 16.0/17 | 0% |

### Verdict: **USE AS-IS**

Data Strategist был wrong — не "Tier 2 pseudo-labels", а отличные H3.6M-совместимые данные.

---

## 2. Sigma Head POC ✅

**Agent:** Sigma Head POC

### НЕОЖИДАННАЯ НАХОДКА:

**YOLO26-Pose УЖЕ ИМЕЕТ встроенный sigma head!**

```
Структура Pose26:
├── cv4: keypoints (51 channels → 17kp × 3)
└── cv4_sigma: sigma (34 channels → 17kp × 2)
    ├── 12 тензоров (multi-scale)
    ├── Предобученные веса (mean=0.12, std=0.41)
    └── Forward возвращает kpts_sigma во время training
```

### Technical Auditor был wrong:

Separate cv5 head НЕ нужна — она уже существует.

### Implementation:

```python
# Разморозить sigma head для обучения
for name, param in model.model.named_parameters():
    if 'sigma' in name:
        param.requires_grad = True

# Использовать pretrained sigma как baseline
# Fine-tune на skating данных
```

### Verdict: **USE BUILT-IN**

Не нужно модифицировать архитектуру.

---

## 3. HDF5 Benchmark ✅

**Agent:** HDF5 Benchmark

### Results:

| Операция | Throughput |
|----------|------------|
| Random read | 4179.8 heatmaps/sec |
| libver='latest' | **4330.3 heatmaps/sec** |
| Batch read (bs=16) | 3106.7 heatmaps/sec |
| Sequential read | 5379.6 heatmaps/sec |

### Target: >100 heatmaps/sec
### Achieved: **43× faster** ✅

### File size:
- 5K heatmaps (17, 72, 96) float16 = **1.1 GB**
- Extrapolated: 350K heatmaps ≈ **77 GB**

### Verdict: **SINGLE FILE SUFFICIENT**

HDF5 не bottleneck. Budget Analyst был right.

---

## 4. FSAnno YouTube Check ✅

**Agent:** FSAnno YouTube Check

### Test: 13 unique YouTube URLs

| Status | Count | % |
|--------|-------|---|
| Available | 5 | 38.5% |
| NOT Available | 8 | **61.5%** |

### Sample unavailable:
```
https://www.youtube.com/watch?v=D7lI_Hm-Qs8 ✗
https://www.youtube.com/watch?v=FFYH_ZcIir4 ✗
https://www.youtube.com/watch?v=DnjevK6KSTM ✗
... (8 total)
```

### Verdict: **SKIP FSAnno**

Data Strategist был right — YouTube videos недоступы. Dataset unusable без source videos.

---

## Updated Data Budget

### Original plan (with FSAnno):
```
FineFS:     ~272K train
FSAnno:     ~17K train
AP3D:       71K train
COCO:       ~12K train (15%)
TOTAL:      ~372K
```

### Actual (after validation):
```
FineFS:     ~272K train ✅ (USE AS-IS)
FSAnno:     0 ❌ (SKIP - unavailable)
AP3D:       71K train ❓ (UNDER REVIEW)
COCO:       ~10K train (10% dynamic)
TOTAL:      ~284K-355K (depending on AP3D)
```

### Remaining question: **AthletePose3D**

Data Strategist recommends DROP.
User question: Does athletic poses help domain transfer?

**Open issue for deep dive.**

---

## Files Created

- `experiments/benchmark_hdf5.py` — HDF5 benchmark script
- `/tmp/test_heatmaps.h5` — Test data (can be deleted)

---

## Next Steps

**Resolved:**
- ✅ FineFS quality confirmed
- ✅ Sigma head exists (no architecture changes)
- ✅ HDF5 performance confirmed
- ✅ FSAnno skipped

**Open:**
- ❓ AthletePose3D: keep or drop?
- ❓ Final dataset composition
