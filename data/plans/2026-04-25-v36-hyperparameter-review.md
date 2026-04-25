# v36 Teacher-Labels: Hyperparameter Review & Recommendations

> **Дата:** 2026-04-25
> **Метод:** 4 специализированных агента (Opus) — параллельное исследование
> **Статус текущего training:** epoch 2/100, mAP50(B)=0.81, mAP50(P)=0.29

---

## Консенсус (все 4 агента сходятся)

1. **100 epochs мало** — 211K weight updates, это 29% от Ultralytics default для COCO
2. **cos_lr=true** — cosine annealing лучше для fine-tuning (подтверждено RTMPose, DWPose, DistilPose)
3. **Pseudo-labeling был правильным выбором** — подтверждено Suzuki et al. (CVPR 2024) и DistilPose

---

## Рекомендации по приоритету

### P0: Перезапустить с исправленными параметрами

| Параметр | Сейчас | Рекомендация | Источник | Почему |
|----------|--------|-------------|----------|--------|
| **epochs** | 100 | **200** | Агент 1 | RTMPose на 150K = 420 epochs. 270K / 150K × 420 ≈ 233 → 200 |
| **patience** | 20 | **30** | Агент 1 | 15% от total epochs |
| **cos_lr** | false | **true** | Агенты 1,2,4 | Consensus. RTMPose, DWPose, DistilPose используют cosine |
| **scale** | 0.5 | **0.3** | Агент 2 | 50-150% resize разрушает spatial structure keypoints |
| **hsv_s** | 0.7 | **0.3** | Агент 2 | Лёд имеет ограниченную цветовую вариацию |
| **hsv_v** | 0.4 | **0.2** | Агент 2 | То же |
| **warmup_epochs** | 3.0 | **5.0** | Агент 1 | 2114 batches/epoch → больше warmup iterations нужно |
| **close_mosaic** | 10 | **20** | Агент 1 | 10% от 200 epochs, match RTMPose stage2 |

### P1: Рекомендовано (низкий риск)

| Параметр | Сейчас | Рекомендация | Источник | Почему |
|----------|--------|-------------|----------|--------|
| **cache** | false | **ram** (если A100) | Агент 4 | -10-15% epoch time, ~100-150GB RAM для 270K |
| **multi_scale** | 0.0 | 0.0 (оставить) | Агент 4 | Учитывая imgsz=384, малые размеры вредят keypoints |
| **freeze** | null | **10** (опционально) | Агенты 2,3 | Head-only первые 10 epochs, потом full model |

### P2: Без изменений

| Параметр | Значение | Почему |
|----------|----------|--------|
| **batch** | 128 | Оптимально для GPU memory |
| **imgsz** | 384 | Teacher heatmaps на 384, не менять |
| **lr0** | 0.01 | MuSGD auto для >10K iterations — корректно |
| **optimizer** | auto (MuSGD) | Правильный выбор Ultralytics для длинного training |
| **pose/box/cls/dfl/kobj** | defaults | pose=12.0 откалиброван для 17kp OKS loss |
| **fliplr** | 0.5 | Валидно для фигурного катания (латеральная симметрия) |
| **mosaic** | 1.0 | OK для pseudo-labeling (нет teacher alignment issue) |
| **dropout** | 0.0 | Не влияет на pose (только classification head) |
| **nbs** | 64 | Корректный, но batch=128 → эффективный LR x2 (нормально) |

### P3: Будущие улучшения (не для v36)

- **640px fine-tuning stage**: 15-20 epochs после основного обучения для восстановления full-resolution accuracy
- **Per-keypoint filtering by heatmap max > 0.5**: требует ре-экстракции heatmaps (confidence = NaN)
- **FineFS-only val split**: AP3D-FS valid = synthetic 3D→2D проекции, может раздувать метрики

---

## Время обучения

~20 min/epoch (измерено на сервере):

| Epochs | Время | Weight Updates | Примечание |
|--------|-------|---------------|------------|
| 100 (сейчас) | ~33ч | 211,400 | Недостаточно |
| **200** | **~67ч** | **422,800** | Рекомендовано |
| 300 | ~100ч | 634,200 | Diminishing returns |

С patience=30 и early stopping, реальное время может быть меньше.

---

## Рекомендуемая команда

```bash
yolo train \
    model=yolo26s-pose.yaml \
    data=configs/data_teacher.yaml \
    epochs=200 \
    batch=128 \
    imgsz=384 \
    name=v36b-teacher-labels \
    exist_ok=true \
    patience=30 \
    save_period=10 \
    cos_lr=true \
    scale=0.3 \
    hsv_s=0.3 \
    hsv_v=0.2 \
    warmup_epochs=5 \
    close_mosaic=20
```

---

## Источники

- **Агент 1 (Epoch count):** RTMPose configs (open-mmlab/mmpose), DWPose (ICCV 2023), Ultralytics default.yaml
- **Агент 2 (Hyperparams):** DistilPose config, Ultralytics loss.py, pose augmentation literature
- **Агент 3 (Pseudo-labeling):** Suzuki et al. CVPR 2024, SSPCM Wang et al. CVPR 2023, DualPose Wei et al. CVPR 2022, DistilPose
- **Агент 4 (Ultralytics):** ultralytics/ultralytics GitHub source (trainer.py, detect/train.py, base.py, torch_utils.py)
