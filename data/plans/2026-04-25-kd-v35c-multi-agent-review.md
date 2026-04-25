# Комплексный отчёт глубокого ревью KD: MogaNet-B -> YOLO26-Pose (v35c)

**Дата:** 2026-04-25
**Версия:** v35c (исправленная)
**Исходный код:** `experiments/yolo26-pose-kd/scripts/distill_trainer.py`
**Участники:** A1 (Loss & Coordinate Math), A2 (Data Pipeline), A3 (Training Strategy), A4 (SOTA KD Research), A5 (Figure Skating Domain)
**Статус:** КРИТИЧЕСКИЕ БАГИ ОБНАРУЖЕНЫ -- обучение невозможно до исправления

---

## 1. Executive Summary

Многоагентное ревью выявило **2 критических бага (P0)**, которые делают KD-обучение полностью нерабочим: (1) `_decode_student_kpts()` использует формулу v8PoseLoss (`raw * 2.0 + anchor - 0.5`) вместо PoseLoss26 (`raw + anchor`), что направляет все KD-градиенты в неверном направлении; (2) расписание KD-весов инвертировано -- максимум KD на старте, когда student-предсказания случайный шум. Дополнительные баги: опечатки в anchor selection (строка 358/361), несовпадение путей HDF5-index (100% потеря KD-данных), отсутствие cosine LR. Также обнаружены 3 ошибки проектирования (letterboxing, inverse affine при edge clamp, confidence без clamp). SOTA-анализ показывает, что текущий coordinate-only KD даёт зазор 2-5% AP; переход на DistilPose-style heatmap KD сокращает зазор до 0.9% AP. Рекомендация: исправить P0-баги, внедрить progressive KD schedule с cosine LR, добавить bone-length consistency loss как quick win.

---

## 2. Critical Bugs (Must Fix Before Training)

### BUG-1 [P0]: Неверная формула декодирования student keypoints

**Файл:** `distill_trainer.py`, строки 247-253
**Агенты:** A1 (primary), A2 (подтверждение)
**Верифицировано:** чтением исходного кода Ultralytics 8.4.41

**Суть:** `_decode_student_kpts()` использует формулу из v8PoseLoss:
```python
decoded[..., 0] = (decoded[..., 0] * 2.0 + anchor_points[:, [0]] - 0.5) * stride_tensor[:, [0]]
```

Но YOLO26 использует PoseLoss26 с совершенно другой формулой (RealNVP flow заменил старое масштабирование):
```python
# PoseLoss26.kpts_decode() -- ultralytics/utils/loss.py:855-860
y[..., 0] += anchor_points[:, [0]]
y[..., 1] += anchor_points[:, [1]]
# НЕТ умножения на 2.0, НЕТ вычитания 0.5, НЕТ умножения на stride
```

**Последствия:**
- Все student-координаты декодируются неверно (смещение + масштабирование)
- KD loss сравнивает координаты в разных пространствах
- Все KD-градиенты направлены в неверном направлении
- PoseLoss26.calculate_keypoints_loss() делит GT на stride (строка 933), поэтому decoded -- anchor-relative координаты, НЕ пиксельные

**Fix:**
```python
def _decode_student_kpts(self, preds, B, K):
    # ... (boilerplate unchanged) ...
    decoded = kpts.clone()
    # PoseLoss26 decode: raw + anchor (no *2.0, no -0.5, no stride)
    decoded[..., 0] = decoded[..., 0] + anchor_points[:, [0]]
    decoded[..., 1] = decoded[..., 1] + anchor_points[:, [1]]
    decoded[..., 2] = decoded[..., 2].sigmoid()
    return decoded, stride_tensor
```

**Важно:** В `kd_loss()` на строках 454-456 координаты нормализуются к [0,1] через деление на img_w/img_h. PoseLoss26.decode возвращает anchor-relative координаты (без stride), которые при делении на imgsz дадут нормализованные [0,1]. Нужно верифицировать, что imgsz в loss совпадает с размером входного изображения.

---

### BUG-2 [P0]: Инвертированное расписание KD-весов

**Файл:** `distill_trainer.py`, строки 174-178
**Агенты:** A3 (primary), A4 (подтверждение)

**Суть:** Текущая формула `w_kd = 1.0 - (epoch-1)/max_epochs` даёт:
- Epoch 1: w_kd = 1.0 (максимум KD)
- Epoch 210: w_kd = 0.05 (минимум KD)

Это **инвертировано**: на старте student-предсказания -- случайный шум, KD-loss двигает шум к teacher, конфликтует с GT-learning. KD должен усиливаться по мере того, как student учится базовые координаты.

**Литература:**
- DWPose: фиксированный alpha=0.00005 (без decay)
- DistilPose: reg2hm_loss=0 для первых 5 эпох, затем фиксированный weight=1.0
- CrossKD: progressive growth после warmup

**Fix (progressive schedule):**
```python
def compute_kd_weight(self) -> float:
    # Warmup: 0 KD в первые warmup_epochs (handled in kd_loss)
    # Growth: линейный рост от 0 до 1 за 15 эпох после warmup
    # Sustain: w_kd = 1.0 до конца
    growth_epochs = 15
    post_warmup = self._current_epoch - self.warmup_epochs
    if post_warmup <= 0:
        return 0.0
    if post_warmup >= growth_epochs:
        return 1.0
    return post_warmup / growth_epochs
```

---

### BUG-3 [P0]: Отсутствие cosine LR

**Файл:** `distill_trainer.py`, строка 681 (overrides dict)
**Агенты:** A3 (primary), A4 (подтверждение)

**Суть:** В overrides нет `cos_lr=True`. Ultralytics по умолчанию использует линейный decay. Все pose-модели SOTA (DWPose, RTMPose, GAME-YOLO) используют cosine annealing. Без cosine LR модель теряет способность к fine-tuning в последних 50% обучения.

**Fix:**
```python
overrides = {
    ...
    "cos_lr": True,  # P0: cosine annealing
    ...
}
```

---

### BUG-4 [P0]: 100% потеря KD-данных из-за несовпадения путей

**Файл:** `generate_teacher_heatmaps.py`, строка 532 + `distill_trainer.py`, строки 87-99
**Агент:** A2 (primary)

**Суть:** Три уровня несовпадения путей между генерацией и загрузкой:

1. `generate_teacher_heatmaps.py` строка 532: `entries.append((str(f), ...))` -- сохраняет путь как `str(Path)`, который может быть относительным или абсолютным в зависимости от cwd
2. `TeacherCoordLoader.load()` строка 88: получает `batch["im_file"]` от Ultralytics -- всегда абсолютный, через `Path.resolve()`
3. Fallback на строке 90: `Path(img_path).name` -- берёт только имя файла, вызывает коллизии между датасетами
4. Fallback на строке 95: поиск "data" в `p.parts` -- хрупкий, ломается на других структурах

**Последствия:** Если пути не совпадают, teacher_coords = zeros, teacher_conf = zeros, KD loss = 0. Тренировка идёт без дистилляции.

**Fix (два варианта):**

Вариант A -- нормализовать пути при генерации:
```python
# generate_teacher_heatmaps.py
entries.append((str(f.resolve()), str(lbl.resolve()), dataset_name))
```

Вариант B -- нормализовать пути при загрузке (robuster):
```python
# TeacherCoordLoader.load()
for img_path in im_files:
    # Try absolute resolved path first
    resolved = str(Path(img_path).resolve())
    idx = self.index.get(resolved)
    if idx is None:
        # Try as-is
        idx = self.index.get(img_path)
    if idx is None:
        # Try relative from "data/" component
        p = Path(img_path)
        parts = p.parts
        try:
            data_idx = parts.index("data")
            rel = str(Path(*parts[data_idx:]))
            idx = self.index.get(rel)
        except ValueError:
            pass
```

**Рекомендация:** вариант A + добавить warning при miss:
```python
if idx is None:
    print(f"WARNING: teacher coords not found for {img_path}")
```

---

### BUG-5 [P1]: Опечатки в `_select_best_anchor()`

**Файл:** `distill_trainer.py`, строки 358 и 361
**Агент:** A1

**Суть:** Когда нет валидного GT (fallback ветка):
- Строка 358: `center_x = student_kpts.shape[-1] / 2` -- это K/2 = 8.5 (количество keypoints / 2), а не img_w/2
- Строка 361: `(cy - center_x).abs()` -- Y-расстояние сравнивается с X-центром (должно быть `center_y`)

**Fix:**
```python
if gt_bboxes[img_i, 0] < 0:
    # Используем медиану всех anchor-центров как proxy для центра изображения
    cx = (student_x1[img_i] + student_x2[img_i]) / 2
    cy = (student_y1[img_i] + student_y2[img_i]) / 2
    median_cx = cx.median()
    median_cy = cy.median()
    dist = (cx - median_cx).abs() + (cy - median_cy).abs()
    best = dist.argmin()
    selected[img_i] = student_kpts[img_i, best]
    continue
```

---

### BUG-6 [P1]: Teacher confidence без clamp(min=0)

**Файл:** `distill_trainer.py`, строка 492
**Агенты:** A2, A5

**Суть:** `teacher_conf` -- raw Gaussian values из soft_argmax_heatmap(). После sigmoid+softmax они в [0,1], но если generate_teacher_heatmaps.py использовал clamp вместо sigmoid (баг, уже исправлен в v5), значения могут быть отрицательными. Даже после исправления, для отсутствующих keypoints confidence может быть близка к 0.

`weight = teacher_conf * vis_mask * valid_cp.unsqueeze(1).float()` -- если teacher_conf отрицательный, weight инвертирует loss (максимизирует ошибку вместо минимизации).

**Fix:**
```python
weight = teacher_conf.clamp(min=0.0) * vis_mask * valid_cp.unsqueeze(1).float()
# Domain fix (A5): минимум 0.1 для signal при вращениях
weight = weight.clamp(min=0.1) * vis_mask  # для skating domain
```

---

### BUG-7 [P1]: Только первый человек из YOLO-label

**Файл:** `generate_teacher_heatmaps.py`, строки 544-545
**Агент:** A2

**Суть:** `parse_yolo_label()` читает `f.readline()` -- только первую строку. Для multi-person COCO-кадров это может быть class 0 (background) или не тот человек.

**Fix:**
```python
def parse_yolo_label(label_path, img_w, img_h):
    """Parse first PERSON (class 0) from YOLO pose label."""
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != 0:  # Skip non-person classes
                continue
            cx_n, cy_n, w_n, h_n = parts[1:5]
            cx = float(cx_n) * img_w
            cy = float(cy_n) * img_h
            w = float(w_n) * img_w
            h = float(h_n) * img_h
            return cx, cy, w, h
    return None
```

---

## 3. Architecture Issues (Should Fix)

### ARCH-1 [P1]: Letterboxing mismatch

**Агенты:** A1, A2

Student-координаты в `kd_loss()` нормализуются через `img_w, img_h = batch["img"].shape[2:]` (размер тензора после letterboxing/padding). Teacher-координаты нормализуются через `crop_params[:, 4:6]` (оригинальный размер изображения). Ultralytics добавляет letterbox padding когда `rect=False` (по умолчанию).

**Решение:** Получить оригинальные размеры из batch или использовать `rect=True` в training overrides. Ultralytics кладёт `batch["img_shape"]` с оригинальными размерами.

```python
# В kd_loss():
if "img_shape" in batch:
    img_h, img_w = batch["img_shape"][0].tolist()  # оригинальные размеры
else:
    img_h, img_w = batch["img"].shape[2], batch["img"].shape[3]
```

Или принудительно: `"rect": True` в overrides (но это может сломать batch size uniformity).

---

### ARCH-2 [P1]: Inverse affine asymmetry при edge clamp

**Агент:** A1

Когда crop выходит за край изображения, `crop_and_resize()` зажимает x1/y1 (строки 584-587 generate_teacher_heatmaps.py). Но `_inverse_affine_transform()` использует clamped-координаты как origin. Это корректно для x1_clamped/y1_clamped, но создаёт ассиметрию: если человек на левом краю и x1 был -40, зажатый до 0, то центр crop сдвигается на 40px вправо относительно ожидаемого.

**Текущий статус:** crop_params сохраняют фактические clamped значения, что уже корректно -- inverse transform использует их напрямую. Проблема возникает только если padding ratio 1.4x недостаточен (человек частично обрезан). Текущий padding=0.2 (bw*1.4) -- обычно достаточно.

**Решение:** Увеличить padding до 0.3 (bw*1.6) для людей на краях:
```python
pad = 0.3  # было 0.2
```

---

### ARCH-3 [P2]: Weight imbalance в KD loss

**Файл:** `distill_trainer.py`, строки 496-498
**Агент:** A1

`weight_sum.clamp(min=1.0)` + `coord_loss * B` создаёт непропорциональный loss для изображений с малым количеством видимых keypoints. Если у одного изображения 2 видимых keypoints, а у другого 15 -- первый вносит ~7.5x больший вклад в loss на keypoint.

**Fix:** Использовать mean вместо weighted sum:
```python
# Вариант A: mean по видимым keypoints
visible_count = vis_mask.sum(dim=1).clamp(min=1.0)  # (B,)
per_img_loss = (weight * per_kp_loss).sum(dim=1) / visible_count  # (B,)
coord_loss = per_img_loss.mean() * B

# Вариант B: weighted mean с per-keypoint weights (рекомендуется A5)
kp_weights = torch.tensor([3.0, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.8, 0.8, 0.5, 0.5, 1.5], device=gt_loss.device)
weighted_per_kp = per_kp_loss * kp_weights.unsqueeze(0) * vis_mask
coord_loss = weighted_per_kp.sum() / (kp_weights.unsqueeze(0) * vis_mask).sum().clamp(min=1.0)
coord_loss = coord_loss * B
```

---

### ARCH-4 [P2]: _rebuild_optimizer без weight/bias/bn split

**Файл:** `distill_trainer.py`, строки 511-541
**Агент:** A1

Ultralytics' `build_optimizer()` разделяет параметры на 3 группы: weight (с decay), bias (без decay), BatchNorm (без decay). При unfreeze backbone, `_rebuild_optimizer()` добавляет все backbone params в одну группу с единым weight_decay, что может ухудшить обучение.

**Fix:** Копировать логику разделения из Ultralytics:
```python
def _rebuild_optimizer(self, model):
    trainer = self._trainer_ref
    if trainer is None:
        return
    base_lr = getattr(self, "_base_lr", trainer.optimizer.param_groups[0]["lr"])
    backbone_lr = base_lr * 0.1

    backbone_params_wd = []
    backbone_params_no_wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        already = any(param is p for pg in trainer.optimizer.param_groups for p in pg["params"])
        if already:
            continue
        if param.ndim == 1 or "bn" in name.lower() or "norm" in name.lower():
            backbone_params_no_wd.append(param)
        else:
            backbone_params_wd.append(param)

    wd = trainer.optimizer.param_groups[0].get("weight_decay", 1e-5)
    for params, decay in [(backbone_params_wd, wd), (backbone_params_no_wd, 0.0)]:
        if params:
            trainer.optimizer.add_param_group({
                "params": params,
                "lr": backbone_lr,
                "weight_decay": decay,
            })
```

---

### ARCH-5 [P2]: Non-vectorized soft_argmax

**Файл:** `extract_teacher_coords.py`, строки 97-106
**Агент:** A2

Per-image loop с вызовом `soft_argmax_heatmap()` для каждого изображения. При 264K heatmaps это ~264K отдельных вызовов F.softmax на CPU.

**Fix:** Batch processing:
```python
def soft_argmax_heatmap_batch(hm_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized soft argmax for batch of heatmaps.
    Args: hm_batch: (B, K, H, W) float16
    Returns: coords (B, K, 2), confidence (B, K)
    """
    hm = torch.from_numpy(hm_batch.astype(np.float32))  # (B, K, H, W)
    K, H, W = hm.shape[1], hm.shape[2], hm.shape[3]
    flat = hm.reshape(-1, K, H * W)  # (B, K, H*W)
    probs = F.softmax(flat, dim=-1).reshape(-1, K, H, W)

    gx = torch.linspace(0, 1, W, device=hm.device)
    gy = torch.linspace(0, 1, H, device=hm.device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")

    x_coords = (probs * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=[2, 3])  # (B, K)
    y_coords = (probs * grid_y.unsqueeze(0).unsqueeze(0)).sum(dim=[2, 3])

    confidence = hm.flatten(2).max(dim=2).values  # (B, K)
    return x_coords.numpy(), y_coords.numpy(), confidence.numpy()
```

---

## 4. SOTA Comparison

### Текущий подход vs Литература

| Аспект | Наш v35b | SOTA Best | Gap |
|--------|----------|-----------|-----|
| KD type | Coordinate-only | Coordinate + Heatmap (DistilPose) | High |
| Schedule | Linear decay (inverted!) | Progressive growth | Critical |
| LR | Linear decay | Cosine annealing | Medium |
| Temperature | None | Annealing 3->1 | Low |
| Feature KD | Removed (spatial mismatch) | CrossKD (student->teacher) | Medium |
| Self-KD | None | Head-aware (DWPose) | Medium |
| Expected AP gap | 2-5% | 0.9% (DistilPose) / student > teacher (DWPose) | -- |

### Key Papers

1. **DistilPose (CVPR 2023)** -- THE most relevant. Exact same problem: heatmap-teacher -> coordinate-student. +15.6% AP. Simulated Heatmaps (student predicts mu, sigma -> virtual Gaussian vs teacher heatmap). GitHub: https://github.com/yshMars/DistilPose

2. **DWPose (ICCV 2023 Workshop)** -- Two-stage KD. Feature + logit KD with weight-decay schedule. Head-aware self-KD: +1.0 AP за 20% extra training time. Student SURPASSES teacher.

3. **CrossKD (CVPR 2024)** -- Student features -> teacher head (reversed direction). Solves contradictory supervision. +3.5 AP on GFL.

4. **Delta-Distillation (arXiv 2025)** -- Dynamic alpha + temperature scheduling.

### Expected Gap Analysis

| Approach | Expected Gap vs Teacher |
|----------|------------------------|
| Coordinate-only KD (current v35b) | 2-5% AP |
| Coordinate + Simulated Heatmap KD (DistilPose) | 0.9% AP |
| Feature + Logit KD (DWPose) | Student SURPASSES teacher (+1.2% AP) |

---

## 5. Quick Wins (Sorted by Impact/Effort)

### 30 минут

| # | Fix | Impact | File:Line |
|---|-----|--------|-----------|
| 1 | `cos_lr=True` в overrides | +0.5-1.0 AP | distill_trainer.py:681 |
| 2 | `teacher_conf.clamp(min=0.0)` | Prevent inverted loss | distill_trainer.py:492 |
| 3 | Fix `_select_best_anchor` typos (center_x -> median) | Correct anchor selection | distill_trainer.py:358,361 |
| 4 | Temperature scaling `tau=2.0` for KD loss | +0.2-0.5 AP | distill_trainer.py:493 |

### 1 час

| # | Fix | Impact | File:Line |
|---|-----|--------|-----------|
| 5 | Progressive KD schedule (growth over 15 epochs) | +0.5-1.0 AP | distill_trainer.py:174-178 |
| 6 | Fix decode formula (PoseLoss26 style) | CRITICAL -- enables KD | distill_trainer.py:247-253 |
| 7 | Bone-length consistency loss | +0.3-0.5 AP | distill_trainer.py (new) |
| 8 | Per-keypoint weights from A5 | +0.3-0.5 AP (domain) | distill_trainer.py:492 |
| 9 | Path normalization in TeacherCoordLoader | CRITICAL -- enables KD data | distill_trainer.py:87-99 + generate_teacher_heatmaps.py:532 |

### 1 день

| # | Fix | Impact | File |
|---|-----|--------|------|
| 10 | Simulated Heatmap KD (sigma prediction) | +1.0-2.0 AP | simulate_heatmap.py (exists!) + distill_trainer.py |
| 11 | Head-aware self-KD | +0.8-1.0 AP | distill_trainer.py (new) |
| 12 | Invisible KP distillation (remove vis_mask from KD) | +0.5-1.0 AP | distill_trainer.py:492 |
| 13 | COCO mix increase to 15% | Prevent catastrophic forgetting | data.yaml |

---

## 6. Recommended v35c Configuration

### Hyperparameters

```python
overrides = {
    "model": "yolo26s-pose.yaml",
    "data": "data.yaml",
    "epochs": 210,
    "batch": 128,
    "imgsz": 384,
    "mosaic": 0.0,          # Keep: multi-person labels break with mosaic
    "mixup": 0.1,           # NEW: light regularization, preserves spatial layout
    "lr0": 0.001,           # CHANGED: conservative (was 0.002)
    "cos_lr": True,         # NEW: cosine annealing (P0 fix)
    "warmup_epochs": 3,     # CHANGED: shorter (was 5)
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    "patience": 30,         # NEW: early stopping
    "rect": True,           # NEW: avoid letterboxing mismatch
}

kd_config = {
    "coord_alpha": 0.05,
    "warmup_epochs": 3,     # Match trainer warmup
    "unfreeze_epoch": 8,    # CHANGED: earlier (was 11)
}
```

### KD Schedule (progressive growth)

```python
def compute_kd_weight(self) -> float:
    if self._max_epochs <= 1:
        return 1.0
    warmup = self.warmup_epochs
    growth = 15  # epochs to reach full KD
    post_warmup = self._current_epoch - warmup
    if post_warmup <= 0:
        return 0.0
    if post_warmup >= growth:
        return 1.0
    return post_warmup / growth
```

**Timeline:**
| Epoch | w_kd | What happens |
|-------|------|-------------|
| 1-3 | 0.0 | Warmup, head-only training, GT-only |
| 4-18 | 0.0->1.0 | Progressive KD growth |
| 19-210 | 1.0 | Full KD (GT + teacher) |
| Epoch 8 | -- | Backbone unfreeze with differential LR |

### Loss Formula (v35c)

```
L_total = L_gt + coord_alpha * w_kd * L_coord * B

L_coord = MSE(student_kpts_norm, teacher_kpts_global) * teacher_conf * vis_mask * kp_weights
```

**С coordinate-only KD (текущий подход):**
- coord_alpha = 0.05
- kp_weights: per-keypoint from A5 biomechanical importance
- teacher_conf: clamp(min=0.1) для domain signal при вращениях

**С Simulated Heatmap KD (DistilPose-style, future):**
```
L_total = L_gt + 1.0 * L_reg2hm + 0.01 * L_score
```

### Validation Protocol

| Checkpoint | Epoch | What to check |
|-----------|-------|---------------|
| Warmup end | 5 | GT loss decreasing, no KD yet |
| Unfreeze | 8 | LR schedule, no loss spike |
| KD=1.0 | 19 | KD loss meaningful (not zero) |
| Early convergence | 50 | Pose AP > 0.5 (skating val) |
| COCO retention | 150 | COCO AP not dropped >5% |
| Final | 210 | Target AP > 0.65 (skating val) |

**Metrics:**
- Pose AP (COCO eval protocol) на skating-val
- OKS@0.5: relaxed metric для domain
- Per-keypoint AP: выявить слабые keypoints
- gt_loss / total_loss ratio: отслеживать баланс GT vs KD
- COCO AP retention: катастрофическое забывание

---

## 7. Future Roadmap (v36+)

### v36: DistilPose-style Heatmap KD

Использовать существующий `simulate_heatmap.py` (уже протестирован, 12/12 tests pass) для Simulated Heatmap KD:

1. Student предсказывает (x, y) + sigma из YOLO26 cv4_sigma head
2. Генерируется virtual Gaussian heatmap: `exp(-0.5 * (dx^2/sigma_x^2 + dy^2/sigma_y^2))`
3. MSE loss против teacher heatmap (pre-computed)
4. Score loss: teacher confidence at predicted location

**Ожидаемый эффект:** gap teacher-student сокращается с 2-5% до 0.9% AP.
**Effort:** 2-3 часа (simulate_heatmap.py уже готов).
**Зависимость:** BUG-1 fix (correct decode) обязательна.

### v37: CrossKD Feature Alignment

1. Student backbone features -> teacher deconv head (reversed direction от standard feature KD)
2. Решает проблему contradictory supervision (student features don't match teacher expectations)
3. +3.5 AP по результатам CrossKD paper

### v38: Head-Aware Self-KD

1. DWPose-style: head предсказывает для нескольких anchors, loss между ними
2. +1.0 AP за 20% extra training time
3. Student может SURPASS teacher

### v39: Domain-Specific Enhancements

1. Rotation-augmented KD: teacher heatmaps для rotated crops
2. Spin detection: отдельный KD для вращений (confidence clamp min=0.1)
3. Bone-length consistency: `L_bone = |pred_bone_len - gt_bone_len|^2`
4. Left/Right swap penalty: decrease swap rate below 5%

---

## 8. Cross-Agent Agreement Matrix

| Issue | A1 | A2 | A3 | A4 | A5 | Priority |
|-------|----|----|----|----|-----|----------|
| Decode formula (PoseLoss26 vs v8) | **FIND** | confirm | -- | -- | -- | **P0** |
| KD schedule inverted | -- | -- | **FIND** | confirm | -- | **P0** |
| Missing cos_lr | -- | -- | **FIND** | confirm | -- | **P0** |
| Path mismatch (100% data loss) | -- | **FIND** | -- | -- | -- | **P0** |
| Anchor selection typos | **FIND** | -- | -- | -- | -- | P1 |
| Confidence without clamp | -- | **FIND** | -- | -- | **FIND** | P1 |
| Only first person parsed | -- | **FIND** | -- | -- | -- | P1 |
| Letterboxing mismatch | **FIND** | confirm | -- | -- | -- | P1 |
| Inverse affine asymmetry | **FIND** | -- | -- | -- | -- | P1 |
| Weight imbalance | **FIND** | -- | -- | -- | -- | P2 |
| Backbone warmup at unfreeze | -- | -- | **FIND** | -- | -- | P1 |
| COCO mix only 1.9% | -- | -- | **FIND** | -- | -- | P1 |
| Non-vectorized soft_argmax | -- | **FIND** | -- | -- | -- | P2 |
| Per-keypoint weights | -- | -- | -- | -- | **FIND** | P2 |
| KD signal disappears in rotations | -- | -- | -- | -- | **FIND** | P1 |
| Spins underrepresented in data | -- | -- | -- | -- | **FIND** | P2 |
| Simulated Heatmap as best KD | -- | -- | -- | **FIND** | -- | v36 |
| Head-aware self-KD | -- | -- | -- | **FIND** | -- | v38 |
| Progressive KD schedule | -- | -- | **FIND** | confirm | -- | P0 |

**Legend:**
- **FIND** = первичное обнаружение
- confirm = независимое подтверждение
- P0 = блокирует обучение
- P1 = значительно ухудшает качество
- P2 = ухудшает качество, но не блокирует

**Статистика согласия:**
- 3+ агента: 3 issue (decode, schedule, cos_lr) -- все P0
- 2 агента: 7 issues
- 1 агент: 7 issues
- Всего: 17 уникальных issues, 4 P0, 5 P1, 5 P2, 3 roadmap

---

## Appendix A: Verification Checklist

Перед запуском v35c training:

- [ ] BUG-1: Decode formula соответствует PoseLoss26 (no *2.0, no -0.5)
- [ ] BUG-2: KD schedule -- progressive growth (0 -> 1 за 15 epochs)
- [ ] BUG-3: `cos_lr=True` в overrides
- [ ] BUG-4: Пути в HDF5 index совпадают с Ultralytics batch["im_file"]
- [ ] BUG-5: `_select_best_anchor` fallback использует медиану, не K/2
- [ ] BUG-6: teacher_conf.clamp(min=0.0) или clamp(min=0.1)
- [ ] BUG-7: parse_yolo_label читает первую строку с class=0
- [ ] ARCH-1: letterboxing учтён (rect=True или img_shape из batch)
- [ ] Тест: прогнать 1 epoch на 10 images, проверить KD loss != 0
- [ ] Тест: проверить teacher_coords != zeros для всех images в batch
- [ ] Calibration: 5-epoch run для замера wall-clock time

---

## Appendix B: Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `distill_trainer.py` | BUG-1 (decode), BUG-2 (schedule), BUG-5 (anchor), BUG-6 (clamp), ARCH-3 (weights), ARCH-4 (optimizer) | P0 |
| `distill_trainer.py` CLI overrides | BUG-3 (cos_lr), ARCH-1 (rect), hyperparams | P0 |
| `generate_teacher_heatmaps.py` | BUG-4 (path normalization), BUG-7 (multi-person) | P0 |
| `extract_teacher_coords.py` | ARCH-5 (vectorized soft_argmax) | P2 |
| `simulate_heatmap.py` | Нет изменений (уже готов для v36) | -- |
