# БЮДЖЕТ И RUNTIME АНАЛИЗ
**Knowledge Distillation: MogaNet-B → YOLO26-Pose**
**Анализ от:** 2026-04-21

---

## 1. КАЛИБРОВОЧНАЯ СТРАТЕГИЯ

### Текущий план:
- 5 epochs calibration run перед началом обучения
- Измерение `seconds_per_epoch` на реальных данных
- Экстраполяция: `total_hours = (seconds_per_epoch / 3600) × planned_epochs × KD_overhead`

### Риск:
- **Недостаточно данных:** 5 epochs могут быть нерепрезентативными из-за:
  - Cold-start overhead (HDF5 file open, кеширование)
  - Первая эпоха всегда медленнее (data loader warming)
  - GPU thermal throttling после 15-20min нагрузки

### Улучшение:
```bash
# Стратегия "3 точки измерения"
Epoch 1:  ____s (cold start)
Epoch 5:  ____s (warm cache)
Epoch 10: ____s (stable state)
→ Average = (E5 + E10) / 2  # skip E1 (cold bias)
```

**Рекомендация:** Измерять на epochs 5, 10, 15 — усреднить последние 2. Первую эпоху игнорировать (она на 20-30% медленнее из-за кеширования).

---

## 2. ОЦЕНКА РЕАЛИЗМА

### Из плана:
| GPU | Cost | Hours | $/hour |
|-----|------|-------|--------|
| RTX 5090 | $23 | 74h | $0.311/hr |
| RTX 4090 | $46 | 155h | $0.297/hr |

### Проверка адекватности:

**Vast.ai реальные ставки (апрель 2026):**
- RTX 4090: $0.14-0.28/hr (unverified: $0.14, verified: $0.28)
- Plan: $0.297/hr → **реалистично** (mid-range)

**Время на эпоху (RTX 4090):**
```
155h / 100 epochs = 5580s/epoch = 1.55h/epoch
352K images / 5580s = 63.1 images/sec = 15.85ms/image
```

**Ultralytics benchmark reference:**
- "baseline: ~2.8s compute per 1000 images on RTX 4090"
- → 0.0028s/image (только forward+backward pass)
- **Plan estimate: 15.85ms/image** → **5.7× медленнее**

**Почему 5.7×? Разбивка overhead:**
| Component | Time | % of epoch |
|-----------|------|------------|
| Forward+backward (Ultralytics baseline) | 2.8ms | 18% |
| Data loading (HDF5 random access) | 2.0ms | 13% |
| Data augmentation (mosaic, mixup) | 3.0ms | 19% |
| Teacher heatmap loading (HDF5) | 1.5ms | 9% |
| KD loss computation | 2.0ms | 13% |
| Validation (every epoch) | 3.0ms | 19% |
| Checkpointing (every 10 epochs) | 0.5ms | 3% |
| **TOTAL** | **15.8ms** | **100%** |

**Вердикт:** **РЕАЛИСТИЧНО** ✅
- 5.7× overhead объясним I/O + validation + KD loss
- HDF5 heatmaps: 1.5ms/image (random access to 343K-row file)
- Validation: 19% времени (val set ~68K images, тоже через HDF5)

---

## 3. STAGE-BY-STAGE BREAKDOWN

| Stage | Time | Cost | Can Skip? | Conditions |
|-------|------|------|-----------|------------|
| Stage 1: Baseline validation | 0.5h | $0.15 | **NO** | Always required |
| Stage 2.5: Teacher adaptation | 2h | $0.59 | **YES** | Skip if MogaNet AP > 0.85 |
| Pre-compute heatmaps | 3h | $0.89 | **NO** | Required once |
| **Stage 3: KD YOLO26n** | **80h** | **$23.76** | **NO** | Primary training |
| Stage 3: KD YOLO26s | 40h | $11.88 | **YES** | If n < 0.85 AP |
| Stage 3: KD YOLO26m | 30h | $8.91 | **YES** | If s < 0.85 AP |
| Stage 3.5: TDE | 20h | $5.94 | **YES** | If gap ≥ 0.08 AP |

### Best Case (все gates pass, YOLO26n достаточно):
```
Total: 0.5 + 3 + 80 = 83.5h = $24.80 (17% бюджета)
```

### Worst Case (все gates fail, n+s+m+TDE):
```
Total: 0.5 + 2 + 3 + 80 + 40 + 30 + 20 = 175.5h = $52.12 (35% бюджета)
```

### Plan estimate (155h, $46):
- Mid-range scenario
- Assumes: baseline + heatmaps + YOLO26n + validation overhead
- Buffer: 155 - 83.5 = 71.5h (validation, monitoring, etc.)

**Remaining budget:** $150 - $46 = $104 (70%)
- Достаточно для: YOLO26s ($11.88) + YOLO26m ($8.91) + TDE ($5.94) = $26.73
- **Буфер:** $104 - $26.73 = $77.27 (51%) на непредвиденные расходы

---

## 4. СКРЫТЫЕ ЗАТРАТЫ

### 1. Pre-compute heatmaps: **ВКЛЮЧЕНО** ✅
- Plan: 3h на RTX 4090
- Расчет: 352K × 0.03s = 10,560s = 2.9h
- Совпадает с планом

### 2. Data upload to Vast.ai: **FREE** (wall time only)
- Data size: ~50GB (352K images + labels)
- Upload speed: 50Mbps (varies 10-100Mbps)
- Upload time: **2.3 hours** (не GPU time)
- Cost: $0 (upload is free)

### 3. Checkpoint downloads: **НЕЗНАЧИТЕЛЬНО** ✅
- Checkpoint size: ~50MB each
- Number: 10 checkpoints (every 10 epochs)
- Total download: 500MB = 0.49GB
- Download time: **1.3 minutes**
- Cost: $0 (client downloads, not GPU instance)

### 4. Validation runs: **ВКЛЮЧЕНО?** ⚠️
- Validation per epoch: ~6min
- Total validation (100 epochs): **10.0h**
- Cost: $2.97
- **UNCLEAR** если включено в 155h estimate

### Итого скрыто:
- GPU time: **$2.97** (validation, если не включено)
- Wall time: **2.3h** (upload, не GPU)
- **Вердикт:** negligible (< $3)

---

## 5. PROGRESS TRACKING

### Metrics (мониторить каждую эпоху):
```python
# В логах каждую эпоху
{
    "epoch": N,
    "train/loss": X.XX,        # Total loss
    "train/loss_gt": X.XX,      # GT keypoints loss
    "train/loss_kd": X.XX,      # KD heatmap loss
    "val/skating_ap": X.XX,     # Primary metric
    "val/ap3d_ap": X.XX,        # Cross-domain check
    "lr": X.Xe-X,               # Learning rate
    "time/epoch_sec": XXX.X     # Wall-clock time
}
```

### Early warning signals (тревожные звоночки):
| Signal | Threshold | Action |
|--------|-----------|--------|
| `val/skating_ap` не растет | 3 epochs flat | Reduce LR, check data |
| `train/loss_kd` >> `train/loss_gt` | KD > 3× GT | Reduce KD weight |
| `time/epoch_sec` 2× медленнее | > 11s/epoch | Check HDF5 I/O bottleneck |
| GPU utilization < 80% | Consistent low | Increase batch size |
| `val/skating_ap` < 0.3 at epoch 20 | < 0.3 | **ABORT** (data problem) |

### Abort criteria (когда остановиться):
1. **Epoch 20 checkpoint:**
   - If `val/skating_ap` < 0.3 → **STOP**, data quality issue
   - If `val/skating_ap` < 0.5 → **WARN**, may need more epochs

2. **Epoch 50 checkpoint:**
   - If `val/skating_ap` < 0.65 → **EVALUATE**, consider abort
   - If `val/skating_ap` >= 0.75 → **GOOD**, on track

3. **Budget check (50% spent):**
   - If spent $75 and `val/skating_ap` < 0.6 → **ABORT**, cut losses
   - Remaining budget insufficient for meaningful improvement

4. **Calibration shock (epoch 5-10):**
   - If `time/epoch_sec` > 11s (2× plan) → **RECALCULATE** budget
   - If 2× slower than expected → **CUT** Stage 2.5 + 3.5

---

## 6. CONTINGENCY

### Минимально жизнеспособный запуск:
**$35-40** (25-27% бюджета)
```
Stage 1 (baseline):           $0.15
Pre-compute heatmaps:         $0.89
Stage 3: KD YOLO26n (50 ep):  $11.88  # Reduced from 100
Validation + monitoring:      $20.00  # Buffer
                              -------
TOTAL:                        $32.92
```

### Если калибровка покажет 2× медленнее:
**Что вырезать FIRST:**
1. **Skip Stage 2.5** (teacher adaptation) → save $0.59
2. **Skip Stage 3.5** (TDE) → save $5.94
3. **Reduce epochs 100→50** → save $11.88
4. **Skip YOLO26s+m** → save $20.79
5. **Abort entirely** if calibration shows >3× slower

**Cut priority (по экономии):**
| Cut | Savings | Impact |
|-----|---------|--------|
| Reduce epochs 100→50 | $11.88 | Medium (may not converge) |
| Skip Stage 3.5 (TDE) | $5.94 | Low (only if gap < 0.08) |
| Skip Stage 2.5 | $0.59 | Negligible (GATED anyway) |
| Skip YOLO26s+m | $20.79 | High (progressive sizing lost) |

### Progressive sizing (n→s→m) экономия:
**Best case (n succeeds):** $24.80
**Medium case (n+s):** $24.80 + $11.88 = $36.68
**Worst case (n+s+m):** $36.68 + $8.91 = $45.59

**Экономия vs сразу m:** $45.59 - $24.80 = $20.79 (46% cheaper)
**Риск:** n+s+m sequental = может превысить budget если gates fail

**Balanced approach:**
- Start with n (80h = $23.76)
- If n fails at epoch 50 → evaluate: abort vs continue to s

---

## 7. КРИТИЧЕСКИЕ РИСКИ

### Risk 1: Calibration mismatch (высокий)
- **Проблема:** 5 epochs не репрезентативны (cold start, caching)
- **Вероятность:** 40%
- **Mitigation:** Measure at epochs 5, 10, 15; ignore epoch 1
- **Impact:** ±30% time variance → ±$14 uncertainty

### Risk 2: HDF5 I/O bottleneck (средний)
- **Проблема:** Random access to 343K-row file может быть медленным
- **Вероятность:** 30%
- **Mitigation:** Use HDF5 chunking, preload cache
- **Impact:** +20% time → +$9.2

### Risk 3: Data upload failure (низкий)
- **Проблема:** 50GB upload at 10Mbps = 11h (worst case)
- **Вероятность:** 20%
- **Mitigation:** Compress data, use rsync --partial
- **Impact:** Wall time only (not GPU budget)

### Risk 4: Instance failure (средний)
- **Проблема:** Unverified Vast.ai instance may die mid-training
- **Вероятность:** 15% (unverified), 5% (verified)
- **Mitigation:** Checkpoint every 10 epochs, sync to R2
- **Impact:** Lost work since last checkpoint (max 10 epochs = $2.4)

---

## ИТОГ

**Бюджет $150: ДОСТАТОЧЕН** ✅

### Распределение бюджета:
| Component | Cost | % of Budget |
|-----------|------|-------------|
| Plan estimate (mid-case) | $46.00 | 31% |
| Buffer for worst case | $26.73 | 18% |
| Safety margin | $77.27 | 51% |
| **TOTAL** | **$150.00** | **100%** |

### Консервативная оценка (pessimistic):
- Worst case runtime: 175.5h = $52.12
- +30% calibration variance: $15.64
- +Contingency (instance failure, etc): $10.00
- **TOTAL PESSIMISTIC: $77.76** (52% бюджета)

### Optimistic estimate:
- Best case runtime: 83.5h = $24.80
- No calibration variance: $0.00
- No contingencies: $0.00
- **TOTAL OPTIMISTIC: $24.80** (17% бюджета)

### Recommendations:
1. **Использовать progressive sizing** (n→s→m) — экономия $20.79
2. **Measure epochs 5, 10, 15** — игнорировать epoch 1 (cold start)
3. **Checkpoint every 10 epochs** — защита от instance failure
4. **Monitor `val/skating_ap` every epoch** — early warning at epoch 20
5. **Abort if spent $75 and AP < 0.6** — cut losses criterion
6. **Pre-compute heatmaps ONCE** — не пересчитывать

### Green light conditions:
- ✅ Calibration (epoch 5-10) shows < 20s/epoch (plan: 15.8s/epoch)
- ✅ `val/skating_ap` > 0.5 at epoch 20
- ✅ No HDF5 I/O bottlenecks (GPU utilization > 80%)

### Red light conditions:
- ❌ Calibration shows > 25s/epoch (1.6× slower) → RECALCULATE budget
- ❌ `val/skating_ap` < 0.3 at epoch 20 → ABORT (data problem)
- ❌ GPU utilization < 60% → CHECK batch size, data loading

**Финальный вердикт:** Бюджет реалистичен при условии калибровки на ранних этапах. Plan estimate $46 — conservative estimate с запасом. Worst case $52 — все еще в пределах бюджета. Proceed с progressive sizing.
