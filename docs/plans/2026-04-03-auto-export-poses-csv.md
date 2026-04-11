# Авто-экспорт данных при визуализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Автоматически сохранять NPY (poses) и CSV (таймер, ось, углы) рядом с выходным видео при каждом рендере.

**Architecture:** Собираем per-frame данные в списки во время рендер-цикла, после окончания сохраняем NPY и CSV рядом с `--output` путём. CSV содержит: frame, timestamp, floor_angle, trunk_tilt, и 12 joint angles. NPY содержит (N, 17, 3) raw poses.

**Tech Stack:** numpy, csv (stdlib), pathlib

---

## Файлы

| Файл | Действие | Назначение |
|------|----------|------------|
| `scripts/visualize_with_skeleton.py` | Modify | Добавить сбор данных + экспорт |

---

## Контекст: доступные данные

Во время рендер-цикла (строки ~430-580) уже вычисляются:

1. **`poses[current_pose_idx]`** — (17, 2) нормализованные ключевые точки
2. **`floor_angle`** — угол наклона льда (degrees, строка ~545)
3. **`joint_angles`** — dict[str, float] 12 углов (строка ~488, только при `layer >= 2`)
4. **`frame_idx`**, **`meta.fps`** — для таймера

**Функция `compute_joint_angles()`** (`src/analysis/angles.py:61`):
```python
def compute_joint_angles(pose: NDArray[np.float32]) -> dict[str, float]:
    # Возвращает 12 углов: R/L Ankle, R/L Knee, R/L Hip, R/L Shoulder, R/L Elbow, R/L Wrist
```

**Вызов в рендер-цикле** (строка 488):
```python
joint_angles = compute_joint_angles(poses_viz[current_pose_idx])
```

Проблема: `joint_angles` вычисляется только при `args.layer >= 2`. Для экспорта нужно вычислять всегда.

---

## Task 1: Добавить флаг `--export` и сбор данных

**Файл:** `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Добавить аргумент `--export`**

После строки 91 (`--output`), добавить:

```python
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export poses (NPY) and biomechanics data (CSV) alongside output video",
    )
```

- [ ] **Step 2: Инициализировать списки для сбора данных**

После строки 362 (`output_path = args.output or ...`), добавить:

```python
    # Data export buffers
    export_frames: list[int] = []
    export_timestamps: list[float] = []
    export_floor_angles: list[float] = []
    export_joint_angles: list[dict[str, float]] = []
    export_poses: list[np.ndarray] = []
```

- [ ] **Step 3: Добавить импорт `compute_joint_angles` в начале файла**

Уже есть импорты из `src.analysis`. Проверить что `compute_joint_angles` импортирован. Если нет — добавить в блок импортов анализа:

```python
from src.analysis.angles import compute_joint_angles
```

- [ ] **Step 4: Собирать данные per-frame в рендер-цикле**

После строки 545 (`floor_angle = estimate_floor_angle(...)`), в блоке `if current_pose_idx is not None:`, добавить сбор (не зависит от `args.layer`):

```python
                # Export data collection
                if args.export and current_pose_idx is not None:
                    export_frames.append(frame_idx)
                    export_timestamps.append(round(frame_idx / meta.fps, 3))
                    export_floor_angles.append(round(floor_angle, 2))
                    ja = compute_joint_angles(poses_viz[current_pose_idx])
                    export_joint_angles.append(ja)
                    export_poses.append(poses[current_pose_idx].copy())
```

---

## Task 2: Сохранение NPY и CSV после рендера

**Файл:** `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Добавить блок экспорта перед финальным `print`**

Перед строкой `print(f"Saved to: {output_path}")` (строка ~595), добавить:

```python
        # Export NPY + CSV
        if args.export and export_poses:
            import csv as _csv

            out_dir = output_path.parent
            stem = output_path.stem

            # NPY: (N, 17, 3) raw poses
            npy_path = out_dir / f"{stem}_poses.npy"
            np.save(str(npy_path), np.array(export_poses))
            print(f"Poses saved: {npy_path}")

            # CSV: frame, timestamp, floor_angle, 12 joint angles
            csv_path = out_dir / f"{stem}_biomechanics.csv"
            angle_keys = [
                "R Ankle", "L Ankle", "R Knee", "L Knee",
                "R Hip", "L Hip", "R Shoulder", "L Shoulder",
                "R Elbow", "L Elbow", "R Wrist", "L Wrist",
            ]
            header = ["frame", "timestamp_s", "floor_angle_deg"] + angle_keys
            with open(csv_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(header)
                for idx in range(len(export_frames)):
                    ja = export_joint_angles[idx]
                    row = [
                        export_frames[idx],
                        export_timestamps[idx],
                        export_floor_angles[idx],
                    ]
                    row += [round(ja.get(k, float("nan")), 1) for k in angle_keys]
                    writer.writerow(row)
            print(f"Biomechanics saved: {csv_path}")
```

---

## Task 3: Тест и коммит

- [ ] **Step 1: Запустить с `--export`**

```bash
uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/кораблик.MOV --tracking sports2d --person-click 450 941 --layer 1 --compress --crf 18 --export --output /tmp/export_test.mp4
```

Проверить:
- Файлы `/tmp/export_test_poses.npy` и `/tmp/export_test_biomechanics.csv` созданы
- NPY shape: (365, 17, 3)
- CSV имеет 365 строк + заголовок, 15 колонок

- [ ] **Step 2: Коммит**

```bash
git add scripts/visualize_with_skeleton.py
git commit -m "feat(viz): add --export flag for auto-saving poses (NPY) and biomechanics (CSV)"
```
