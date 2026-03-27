# Skating Biomechanics ML

Комплексная система ML-анализа биомеханики фигурного катания на основе компьютерного зрения.

## Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  2D Pose Est.   │───▶│ 3D Lifting SSM  │
│  (monocular)    │    │ (YOLO + Blaze)  │    │ (Pose3DM)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐             │
                       │ MotionDTW Align │◀────────────┘
                       │   + KISMAM      │
                       └─────────────────┘
                                  │
                       ┌─────────────────┐
                       │ Multimodal RAG  │
                       │  (Qwen/GPT-4o)  │
                       └─────────────────┘
                                  │
                       ┌─────────────────┐
                       │ Coach Feedback  │
                       │   (Natural)     │
                       └─────────────────┘
```

## Компоненты

| Модуль | Технология | Назначение |
|--------|------------|------------|
| **Детекция** | YOLOv11m | Локализация фигуриста в кадре |
| **2D Pose** | BlazePose/HRNet | 33 keypoints включая стопу |
| **3D Lifting** | Pose3DM-L | SSM с FTV-регуляризацией |
| **Выравнивание** | MotionDTW | Нелинейная синхронизация |
| **Анализ** | KISMAM | Семантические диагнозы |
| **RAG** | Qwen3/GPT-4o | Генерация рекомендаций |

## Установка

```bash
uv sync
```

## Структура проекта

```
skating-biomechanics-ml/
├── src/
│   ├── detection/      # YOLOv11 object detection
│   ├── pose_2d/        # BlazePose keypoints
│   ├── pose_3d/        # Pose3DM lifting
│   ├── alignment/      # MotionDTW
│   ├── analysis/       # KISMAM biomechanics
│   └── rag/            # Multimodal RAG
├── research/           # Исследовательские материалы
├── data/               # Датасеты (AthletePose3D, FS-Jump3D)
└── tests/              # Тесты
```

## Research

См. [`research/RESEARCH.md`](research/RESEARCH.md) — полное исследование архитектур, алгоритмов и готовых решений.

## License

MIT
