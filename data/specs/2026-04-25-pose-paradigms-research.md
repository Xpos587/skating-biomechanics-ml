# Beyond Heatmap/SimCC/RLE: SOTA Pose Estimation Research

> **Date:** 2026-04-25
> **Method:** 3 parallel research agents (Opus-level) with independent web search
> **Status:** Final synthesis

---

## 1. Executive Summary

**Есть ли парадигмы лучше Heatmap/SimCC/RLE?** Да, но ни одна не является "убийцей" SimCC для нашего use case. Ключевые находки:

1. **RLE (YOLO26) фундаментально ограничен** — gap 8+ AP vs SimCC при сопоставимых размерах структурный, не тренировочный
2. **DETRPose (ICCV 2025)** — первый real-time DETR pose (72.5 AP, 11.9M), но нет ONNX export, экосистема сырая
3. **ER-Pose (2025)** — новая парадигма: pose без bounding box (keypoint-driven regression)
4. **Dark Pose + UDP** — FREE +1.5-2.5 AP на heatmap teacher без переобучения
5. **Mamba/Flow Matching** — доминируют в 3D lifting, не в 2D pose detection

**Рекомендация:** продолжить v36b (YOLO26s + KD). Если skating AP < 70 — переключить student на RTMPose-s (SimCC) с DWPose-style distillation от MogaNet-B.

---

## 2. Новые модели (2025-2026)

### 2.1 DETRPose (ICCV 2025) — NMS-free Transformer Pose

Первый real-time DETR-based multi-person pose estimation.

| Модель | COCO AP | Params | GFLOPs | Ключевая инновация |
|--------|---------|--------|--------|-------------------|
| DETRPose-N | 70.3 | 4.1M | 8.8 | Real-time DETR pose |
| DETRPose-S | 72.5 | 11.9M | 17.3 | NMS-free bipartite matching |
| DETRPose-M | 74.5 | 24.7M | 33.2 | Pose Denoising |
| DETRPose-L | 75.5 | 45.1M | 57.4 | FDR + KSVF Loss |

**Архитектура:** PVT-v2 backbone + DETR decoder + FDR heads (Feature Distribution Regression — регрессия распределения offsets, не точечных координат).

**Плюсы:** NMS-free (как YOLO26), global attention, competitive AP/params ratio.
**Минусы:** Нет стабильного ONNX export, нет Ultralytics интеграции, академический код.

**RT-DETR-Pose НЕ существует.** RT-DETR v2 (Baidu/Ultralytics) — только detection. DETRPose — независимая работа.

### 2.2 ER-Pose (arXiv 2025) — Pose без Bounding Box

Новая парадигма: keypoint-driven regression вместо box-driven. Модель напрямую предсказывает offsets от anchor points к ключевым точкам, полностью удаляя bbox detection branch.

**Потенциал:** Если подтвердится — четвертая парадигма для single-stage pose. Пока arXiv preprint, недостаточно данных для оценки.

### 2.3 PoseSynViT (CVPR 2025W) — ViT Scaling SOTA

84.3 AP на COCO — новый SOTA для ViT-based pose. Подтверждает что plain ViT масштабируется для pose estimation, но top-down (нужен detector) и не real-time.

### 2.4 PoseBH (CVPR 2026) — Multi-Skeleton Prototypes

ViT + nonparametric keypoint prototypes для unified body/hand/animal pose. Интересный подход к multi-dataset training, но не pose estimation модель per se.

### 2.5 FMPose3D (CVPR 2026) — Flow Matching для 3D Pose

Заменяет diffusion на flow matching (оптимальный транспорт) для 3D pose из 2D. 2-5x быстрее diffusion при сопоставимой точности. Релевантно для CorrectiveLens (3D lifting pipeline).

### 2.6 CIGPose (CVPR 2026) — Causal Intervention на RTMPose

RTMPose backbone + Causal Intervention Module + Hierarchical GNN. +1.7 AP на whole-body, +3.1 AP на foot keypoints. ONNX-ready. Уже исследован в проекте.

### 2.7 Mamba/SSM модели (2025)

| Модель | Задача | Инновация |
|--------|--------|-----------|
| PoseMamba (AAAI 2025) | 2D→3D lifting | Bidirectional Global-Local SSM |
| SasMamba (2025) | 2D→3D lifting | Structure-Aware Stride SSM |
| GLSMamba (2025) | Video temporal | High-res spatiotemporal modelling |
| DF-Mamba (CVPR 2025) | 3D hand pose | Deformable SSM |

**Вердикт:** Mamba доминирует в 3D lifting и temporal modelling, не в single-frame 2D keypoint detection.

---

## 3. Полная таксономия Keypoint Head парадигм

```
Keypoint Head Paradigms
├── A. Spatial Regression
│   ├── A1. Direct FC Regression (устарело, ~60 AP)
│   ├── A2. RLE — Residual Log-Likelihood (YOLO26, 63 AP)
│   ├── A3. Integral Regression / Soft-Argmax (~75 AP)
│   ├── A4. DSNT — Diff. Spatial to Numerical (суперсидено SimCC)
│   ├── A5. DEKR — Disentangled Keypoint (bottom-up, 70.4 AP)
│   ├── A6. DAR — Distribution-Aware (пост-процессинг, +1.4 AP)
│   └── A7. ER-Pose — Keypoint-Driven (без bbox, 2025)
│
├── B. Classification
│   ├── B1. SimCC — Simple Coordinate Classification (RTMPose, 75.8 AP)
│   ├── B1a. FastCC — Lightweight SimCC + shared FC Transformer
│   ├── B1b. FlatPose — Upsampling-free SimCC
│   ├── B2. TokenPose — Keypoint Tokens + Attention (76.1 AP, медленный)
│   └── B3. TransPose — Transformer Regression (73 AP, суперсидено)
│
├── C. Heatmap
│   ├── C1. Standard + Argmax (HRNet, ViTPose, MogaNet)
│   ├── C2. Dark Pose — Taylor Expansion Offset (+1.0-1.7 AP)
│   ├── C3. UDP — Unbiased Data Processing (+1.4-1.7 AP)
│   └── C4. ProbPose — Probabilistic OKS Heatmap (CVPR 2025, +1.0 AP)
│
├── D. Multi-person / Bottom-up
│   ├── D1. Associative Embedding (HigherHRNet, 74.6 AP)
│   └── D2. DETRPose — DETR Matching (ICCV 2025, 72.5 AP)
│
├── E. Generative
│   ├── E1. DiffPose — Diffusion Heatmap (AAAI 2024, медленный)
│   └── E2. DiffusionPose — Markov-Optimized (AAAI 2025, медленный)
│
├── F. GNN Refinement
│   └── F1. Graph-based keypoint refinement (+1.0 AP, marginal)
│
├── G. Hybrid / Distillation
│   ├── G1. DWPose Distiller — SimCC→SimCC (RTMPose tiny→large)
│   ├── G2. DistilPose — Heatmap→Regression via TDE (71.0 AP student)
│   └── G3. RLE + RealNVP Flow (YOLO26, текущий)
│
└── H. New (2025-2026)
    ├── H1. FDR — Feature Distribution Regression (DETRPose)
    ├── H2. Flow Matching (FMPose3D, 3D)
    └── H3. Prototypical Embedding (PoseBH)
```

---

## 4. Сравнительная таблица парадигм

| Парадигма | COCO AP (rep.) | Overhead | Sub-pixel | ONNX | Real-time RTX3050Ti | Наш use case |
|-----------|---------------|----------|-----------|------|---------------------|-------------|
| Heatmap + Argmax | 74.3 | 0 | Нет | Да | Да | Teacher |
| Heatmap + Dark Pose | 75.7 | 0 (post) | Да | Да* | Да | Teacher (free +1.4) |
| Heatmap + UDP | 76.5 | 0 (post) | Да | Да | Да | Teacher (free +1.7) |
| **SimCC** | **75.8** | ~0.5M | Да | Да | Да | **Student (рекомендуется)** |
| RLE (YOLO26) | 63.0 | ~0.1M | Да | Да | Да | Текущий student |
| Integral Regression | ~75 | 0 | Да | Да | Да | Альтернатива SimCC |
| ProbPose (CVPR 2025) | ~75.5 | 0 | Да | Да | Да | Marginal gain |
| FDR (DETRPose) | 72.5 | query heads | Да | Нет | Да (TRT) | Будущее |
| ER-Pose (без bbox) | TBD | 0 | Да | Да | Да | Мониторить |
| Diffusion | ~77 | +10M | Да | Нет | Нет | Нет |
| TokenPose | 76.1 | +tokens | Через HM | Да | Нет | Нет |
| DistilPose student | 71.0 | TDE ~1M | Да | Да | Да | Альтернатива |

\* Dark Pose обычно numpy post-processing, не в ONNX graph.

---

## 5. Почему RLE ограничен (детальный анализ)

### Структурный gap

RLE моделирует *распределение ошибок* между предсказанием и GT. SimCC моделирует *классификацию по пространственным бинам*. Разница фундаментальная:

| Аспект | RLE | SimCC |
|--------|-----|-------|
| Выход | Точечные координаты + σ | Распределение по бинам |
| Градиент | От ошибки (1D) | От cross-entropy (rich) |
| Пространственная структура | Нет (x, y независимы) | Неявная (через bins) |
| Expressiveness | Низкий (offset от anchor) | Высокий (классификация) |

### Доказательства gap

| Размер модели | RLE (YOLO26) | SimCC (RTMPose) | Gap |
|--------------|-------------|-----------------|-----|
| Small | 63.0 (s, 11.9M) | 71.5 (s, ~15M) | **8.5 pp** |
| Medium | 67.2 (m, 21.5M) | 75.8 (m, ~34M) | **8.6 pp** |

Gap стабильно ~8.5 pp при разных размерах → структурное ограничение RLE, не вопрос обучения.

### Что RLE делает хорошо

- **Single-stage**: detect + pose за один pass
- **Edge deployment**: минимальный overhead
- **NMS-free**: deterministic latency
- **Uncertainty**: sigma для downstream filtering

RLE решает другую задачу (speed-first mobile deployment), не нашу (accuracy-first desktop analysis).

---

## 6. Transformer vs CNN vs Mamba (2025-2026)

| Архитектура | Accuracy Leader | Real-time Leader | Best For |
|-------------|----------------|-----------------|----------|
| **ViT** | PoseSynViT 84.3 AP | — | Accuracy-at-any-cost |
| **CNN** | RTMPose-m 75.8 AP | YOLO26s 2.7ms T4 | Production deployment |
| **Transformer (DETR)** | DETRPose-L 75.5 AP | DETRPose-N real-time | NMS-free multi-person |
| **Mamba/SSM** | PoseMamba SOTA 3D | — | 3D lifting, temporal |

**Тренд:** CNN остается production-стандартом для real-time pose. DETR bridge между accuracy и speed. Mamba доминирует в 3D lifting (релевантно для CorrectiveLens).

**Scaling laws для pose:** Pose estimation ещё не достигла LLM-like scaling. Diminishing returns после ~300M params. Архитектурная эффективность важнее сырой масштабности.

---

## 7. Практические рекомендации

### 7.1 Короткосрочные (текущий v36b)

1. **Дать v36b завершиться** (200 epochs, target: skating AP ≥ 70)
2. **Добавить Dark Pose + UDP на MogaNet-B teacher** — FREE +1.5-2.5 AP
3. **Мониторить per-keypoint AP** — особенно ankle, wrist, knee

### 7.2 Если v36b достигнет AP ≥ 70 на skating

- YOLO26s + KD достаточно для production
- Single-stage speed — преимущество для real-time preview
- Сохранить RLE для mobile/deployment scenarios

### 7.3 Если v36b не достигнет AP ≥ 70

**Переключить student на RTMPose-s (SimCC):**

| Критерий | YOLO26s (RLE) | RTMPose-s (SimCC) |
|----------|--------------|-------------------|
| COCO AP | 63.0 | 71.5 |
| Skating AP (est. с KD) | 65-70 | 72-78 |
| Params | 11.9M | ~15M |
| Speed T4 | 2.7ms | 4.2ms |
| ONNX | Да (Ultralytics) | Да (rtmlib) |
| Keypoint head | RLE (ограничен) | SimCC (лучший ceiling) |

**Distillation pipeline:**
1. MogaNet-B (heatmap teacher) → pseudo-labels → RTMPose-s (SimCC student)
2. Или DWPose-style: feature + logit distillation от MogaNet к RTMPose-s
3. Оба подхода совместимы с rtmlib (уже в проекте)

### 7.4 Мониторить (не действовать сейчас)

- **ER-Pose** — если подтвердится, новая парадигма без bbox
- **DETRPose** — когда ONNX export стабилизируется
- **FMPose3D** — для CorrectiveLens (3D lifting через flow matching)
- **PoseMamba** — для temporal 3D refinement

---

## 8. Источники

- **Agent R1 (SOTA Models):** DETRPose (ICCV 2025), ER-Pose, PoseSynViT, PoseBH, FMPose3D, CIGPose, Mamba/SSM survey, RT-DETR status
- **Agent R2 (Keypoint Heads):** Complete paradigm taxonomy (20+ variants), Dark Pose, UDP, ProbPose, SimCC variants, DistilPose TDE, integral regression, DSNT, TokenPose, DEKR, DAR, NerPE
- **Agent R3 (Practical):** Accuracy vs downstream threshold, single-stage vs top-down for skating, DETRPose feasibility, RTX 3050 Ti deployment, KD effectiveness prediction
