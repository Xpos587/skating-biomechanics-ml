# Research: ML-анализ биомеханики фигурного катания

> Комплексное исследование архитектур, алгоритмов и аппаратных решений

## 1. 2D Human Pose Estimation

### 1.1 Обнаружение объектов: YOLOv8 vs YOLOv11

| Характеристика | YOLOv8 | YOLOv11 |
|----------------|--------|---------|
| Архитектура | C2f modules | C3k2 + FPN + Spatial Attention |
| Precision | ~0.850 | 0.877 |
| Recall | — | 0.842 |
| FPS | 20.73 | 16.84 |
| Параметры (m) | Стандарт | -22% |

**Выбор**: YOLOv11m для серверного GPU, YOLOv8n для мобильных.

### 1.2 Keypoint Detection

| Модель | Точек | FPS (mobile) | Примечание |
|--------|-------|--------------|------------|
| BlazePose | 33 | 10-40 | Детальная стопа |
| MoveNet | 17 | — | Недостаточно |
| ViTPose | — | — | SOTA, тяжёлый |

**Выбор**: BlazePose для real-time, ViTPose для offline.

---

## 2. 3D Pose Lifting (2D→3D)

### 2.1 MotionBERT (Transformer)

- Двухпоточный DSTformer
- $\mathcal{O}(N^2)$ сложность
- Отлично при длительных перекрытиях
- Высокое потребление памяти

### 2.2 PoseMamba / Pose3DM (State-Space Models)

| Модель | Сложность | MPJPE | Параметры |
|--------|-----------|-------|-----------|
| MotionBERT | $\mathcal{O}(N^2)$ | ~40.0 мм | — |
| PoseMamba-L | $\mathcal{O}(N)$ | 38.1 мм | -64.7% память |
| Pose3DM-L | $\mathcal{O}(N)$ | 37.9 мм | -82.5% params |

**Pose3DM-L** с FTV-регуляризацией:
- Линейная сложность
- Интеллектуальное сглаживание джиттера
- Идеален для edge deployment

### 2.3 VIBE (SMPL Mesh)

- GRU + SMPL parameters
- Восстанавливает mesh тела
- Более тяжеловесный

---

## 3. Специфичные датасеты

### AthletePose3D
- ~1.3M кадров, 165K поз
- 12 видов спорта
- 73K кадров фигурного катания
- MPJPE: 214→65 мм (-69% при fine-tuning)

### FS-Jump3D
- 12 высокоскоростных камер
- Семантическая аннотация фаз: entry/flight/landing
- VIFSS: view-invariant embeddings

### Нормализация
1. Компенсация движения камеры (гомография)
2. Root centering (таз → origin)
3. Масштабирование (позвоночник = 0.4)

---

## 4. Сравнение движений

### 4.1 Dynamic Time Warping (DTW)

Библиотеки: `dtw-python`, `tslearn`, `dtaidistance`

### 4.2 MotionDTW

- Двухэтапное выравнивание
- Ключевые кадры: take-off, peak, landing
- Superior IoU для сегментов

### 4.3 KISMAM

Переводит дельты углов → семантические метрики:
- "недостаточная группировка"
- "чрезмерный наклон корпуса"

### 4.4 Физические метрики

- **Время полета**: $H = 4.905 \times (t/2)^2$
- **Угловая скорость**: до 1500°/с (top-level)
- **Классификация прыжка**: Bi-LSTM, 99% accuracy
- **Детекция ребра**: анализ голеностопа (зубец vs rocker)

---

## 5. Multimodal RAG

### 5.1 Pipeline

| Компонент | Технология |
|-----------|------------|
| Извлечение | yt-dlp, FFmpeg, Whisper |
| Метрики | MotionDTW + KISMAM |
| Векторизация | Pinecone / pgvector |
| Графы | TTGNN (таблицы ISU) |
| Генерация | Qwen3 / GPT-4o |

### 5.2 Cross-modal RAG

1. KISMAM → семантический диагноз
2. Embedding → поиск в QA базе
3. TTGNN → логика таблиц GOE
4. LLM + контекст → рекомендация

### 5.3 Trustworthy AI

- **L2A** (Learning-to-Abstain): отказ при низкой уверенности
- **TTGNN**: точная работа с таблицами

---

## 6. Итоговая архитектура

```
Video → YOLOv11 → BlazePose → Pose3DM-L → MotionDTW → KISMAM → RAG → Coach Feedback
```

**Ключевые решения:**
- YOLOv11m + BlazePose (2D)
- Pose3DM-L с FTV (3D)
- MotionDTW + KISMAM (анализ)
- Cross-modal RAG + TTGNN + L2A (генерация)
