# Улучшение --select-person UX

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить нечитаемый превью с 30+ наслаивающимися bbox на сетку миниатюр (contact sheet) — один человек = одна ячейка с уникальным цветом, скелетом и номером.

**Architecture:** Вместо одного кадра со всеми bbox, генерируем композитное изображение-сетку. Каждый человек рисуется в своей ячейке: фон из лучшего кадра этого человека, скелет, номер крупным шрифтом. Пользователь видит каждого человека крупно и выбирает по номеру.

**Tech Stack:** OpenCV, numpy

---

## Файлы

| Файл | Действие | Назначение |
|------|----------|------------|
| `src/pose_estimation/rtmlib_extractor.py:620-648` | Modify | Заменить рисование превью |
| `scripts/visualize_with_skeleton.py:220-251` | Modify | Упростить CLI (меньше вывода) |

---

## Контекст: доступные данные на человека

`person_data[tid]` содержит:
- `best_kps`: ndarray (17, 3) — H3.6M нормализованные ключевые точки (x, y, conf)
- `best_conf`: float — средняя уверенность лучшего кадра
- `best_frame`: int — индекс кадра
- `hits`: int — количество кадров
- `first_frame`: int — первый кадр

Существующие утилиты рисования:
- `src/visualization/core/text.py` → `draw_text_box()` — текст с полупрозрачным фоном
- `src/visualization/skeleton/drawer.py` → `draw_skeleton()` — рисование скелета H3.6M

**Подпись `draw_skeleton()`** (`src/visualization/skeleton/drawer.py:46`):
```python
def draw_skeleton(
    frame: Frame,          # (H, W, 3) BGR
    pose: Pose2D,          # (17, 2) или (17, 3) — нормализованные или пиксели
    height: int,           # высота кадра
    width: int,            # ширина кадра
    confidence_threshold: float = 0.5,
    line_width: int = 2,
    joint_radius: int = 4,
    normalized: bool | None = None,  # авто-определение если None
    confidences: np.ndarray | None = None,
    foot_keypoints: np.ndarray | None = None,
) -> Frame
```

---

## Task 1: Создать функцию `_build_person_grid()` в rtmlib_extractor.py

**Файл:** `src/pose_estimation/rtmlib_extractor.py`

Заменить блок рисования превью (строки 620-648) на вызов новой функции `_build_person_grid()`.

- [ ] **Step 1: Добавить статический метод `_build_person_grid()`**

Добавить перед `preview_persons()` (перед строкой 500):

```python
@staticmethod
def _build_person_grid(
    best_frame: np.ndarray,
    persons: list[dict],
    cell_width: int = 320,
    cell_height: int = 240,
    padding: int = 10,
) -> str:
    """Создать сетку миниатюр для выбора человека.

    Каждый человек — своя ячейка с фрагментом кадра, скелетом и номером.
    Решает проблему: 30+ наслаивающихся bbox, нечитаемый текст.

    Args:
        best_frame: Кадр (H, W, 3) BGR.
        persons: Список dict с ключами:
            - best_kps: (17, 3) нормализованные H3.6M ключевые точки
            - hits: int
            - best_conf: float
        cell_width: Ширина ячейки в пикселях.
        cell_height: Высота ячейки в пикселях.
        padding: Отступ между ячейками.

    Returns:
        Путь к сохранённому изображению.
    """
    if not persons:
        return ""

    import tempfile

    n = len(persons)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    # Цветовая палитра (BGR) — контрастные цвета
    palette = [
        (0, 255, 0),    # зелёный
        (255, 60, 60),  # красный
        (255, 200, 0),  # жёлтый
        (255, 100, 200),# розовый
        (0, 200, 255),  # голубой
        (200, 100, 255),# фиолетовый
        (255, 160, 50), # оранжевый
        (100, 255, 200),# бирюзовый
        (255, 255, 255),# белый
        (200, 200, 0),  # оливковый
    ]

    frame_h, frame_w = best_frame.shape[:2]
    grid_w = cols * cell_width + (cols - 1) * padding
    grid_h = rows * cell_height + (rows - 1) * padding
    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)  # тёмный фон

    for i, person in enumerate(persons):
        row = i // cols
        col = i % cols
        x0 = col * (cell_width + padding)
        y0 = row * (cell_height + padding)
        color = palette[i % len(palette)]

        # --- Фрагмент кадра (центрированный на человеке) ---
        kps = person["best_kps"]
        valid = kps[kps[:, 2] > 0.1]
        if len(valid) >= 3:
            cx = float(np.mean(valid[:, 0]))
            cy = float(np.mean(valid[:, 1]))
            bx1 = int(np.min(valid[:, 0]) * frame_w)
            by1 = int(np.min(valid[:, 1]) * frame_h)
            bx2 = int(np.max(valid[:, 0]) * frame_w)
            by2 = int(np.max(valid[:, 1]) * frame_h)
            bw = bx2 - bx1
            bh = by2 - by1
            # Масштабировать bbox чтобы человек заполнял ~80% ячейки
            scale = min(cell_width * 0.8 / max(bw, 1), cell_height * 0.8 / max(bh, 1), 2.0)
            crop_w = int(bw * scale)
            crop_h = int(bh * scale)
            crop_cx = int(cx * frame_w)
            crop_cy = int(cy * frame_h)
            sx1 = max(0, crop_cx - crop_w // 2)
            sy1 = max(0, crop_cy - crop_h // 2)
            sx2 = min(frame_w, crop_cx + crop_w // 2)
            sy2 = min(frame_h, crop_cy + crop_h // 2)
            crop = best_frame[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                resized = cv2.resize(crop, (cell_width, cell_height), interpolation=cv2.INTER_LINEAR)
                grid[y0:y0 + cell_height, x0:x0 + cell_width] = resized

        # --- Рисуем скелет ---
        if kps is not None and len(valid) >= 3:
            cell_kps = kps.copy()
            cell_kps[:, 0] = (cell_kps[:, 0] - cx) / max(bw, 0.01) * cell_width * 0.4 + cell_width * 0.5
            cell_kps[:, 1] = (cell_kps[:, 1] - cy) / max(bh, 0.01) * cell_height * 0.4 + cell_height * 0.5
            cell_crop = grid[y0:y0 + cell_height, x0:x0 + cell_width].copy()
            draw_skeleton(
                cell_crop,
                cell_kps[:, :2],
                cell_height,
                cell_width,
                confidence_threshold=0.3,
                confidences=cell_kps[:, 2],
            )
            grid[y0:y0 + cell_height, x0:x0 + cell_width] = cell_crop

        # --- Номер + hits ---
        label = f"#{i + 1}"
        hits_label = f"hits={person['hits']}"
        # Полупрозрачный фон для текста
        cv2.rectangle(grid, (x0 + 2, y0 + 2), (x0 + 90, y0 + 42), (0, 0, 0), -1)
        cv2.addWeighted(
            np.full((40, 88, 3), color, dtype=np.uint8),
            0.6,
            grid[y0 + 2:y0 + 42, x0 + 2:x0 + 90],
            0.4,
            0,
            dst=grid[y0 + 2:y0 + 42, x0 + 2:x0 + 90],
        )
        cv2.putText(grid, label, (x0 + 6, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(grid, hits_label, (x0 + 6, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Рамка ячеек
    for i in range(n):
        row = i // cols
        col = i % cols
        x0 = col * (cell_width + padding)
        y0 = row * (cell_height + padding)
        color = palette[i % len(palette)]
        cv2.rectangle(grid, (x0, y0), (x0 + cell_width, y0 + cell_height), color, 2)

    preview_path = str(Path(tempfile.mktemp(suffix=".jpg")).with_name("person_preview.jpg"))
    cv2.imwrite(preview_path, grid)
    return preview_path
```

- [ ] **Step 2: Добавить импорт `draw_skeleton` вверху `preview_persons()`**

Внутри `preview_persons()`, после строки с `from ..tracking.sports2d import Sports2DTracker` (~строка 539), добавить:

```python
from ..visualization.skeleton.drawer import draw_skeleton
```

- [ ] **Step 3: Заменить блок рисования превью (строки 620-648)**

Заменить весь блок от `# Build visual preview with numbered bboxes` до `cv2.imwrite(preview_path, preview_img)` на:

```python
        # Build person grid preview
        preview_path: str | None = None
        if best_frame is not None and person_data:
            persons_for_grid = []
            for tid, data in sorted(
                person_data.items(), key=lambda kv: kv[1]["hits"], reverse=True
            ):
                if data["best_kps"] is not None:
                    valid = data["best_kps"][data["best_kps"][:, 2] > 0.1]
                    if len(valid) >= 3:
                        persons_for_grid.append(data)
            if persons_for_grid:
                preview_path = RTMPoseExtractor._build_person_grid(
                    best_frame, persons_for_grid
                )
```

---

## Task 2: Упростить CLI вывод

**Файл:** `scripts/visualize_with_skeleton.py`

- [ ] **Step 1: Упростить вывод списка людей (строки 227-236)**

Заменить verbose вывод:

```python
                print(f"\nDetected {len(persons)} persons:\n")
                for i, p in enumerate(persons, 1):
                    x1, y1, x2, y2 = p["bbox"]
                    print(
                        f"  #{i}: track_id={p['track_id']}, "
                        f"bbox=({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f}), "
                        f"hits={p['hits']}, first_frame={p['first_frame']}"
                    )
                if preview_path:
                    print(f"\n  Preview: {preview_path}")
                print()
```

На компактный:

```python
                print(f"\nОбнаружено {len(persons)} человек. Смотри превью.")
                if preview_path:
                    print(f"  {preview_path}")
                print()
```

---

## Task 3: Тест и коммит

- [ ] **Step 1: Запустить на тестовом видео**

```bash
uv run python scripts/visualize_with_skeleton.py /home/michael/Downloads/кораблик.MOV --tracking sports2d --select-person --layer 1 --compress --crf 18 --output /tmp/test_select.mp4
```

Проверить:
- Открывается превью-сетка (не один кадр с 30+ bbox)
- Каждый человек в своей ячейке
- Скелет нарисован
- Номера читаемые, цвета разные
- Выбор по номеру работает

- [ ] **Step 2: Коммит**

```bash
git add src/pose_estimation/rtmlib_extractor.py scripts/visualize_with_skeleton.py
git commit -m "feat(ui): contact-sheet preview for --select-person (fix unreadable 30+ bbox overlay)"
```
