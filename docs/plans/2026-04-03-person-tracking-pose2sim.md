# Интеграция трекинга Pose2Sim

> **Для agentic-работников:** ОБЯЗАТЕЛЬНАЯ SUB-SKILL: Используй superpowers:subagent-driven-development (рекомендуется) или superpowers:executing-plans для реализации плана задача-за-задачей. Шаги используют чекбоксы (`- [ ]`) для отслеживания.

**Goal:** Заменить сломанный трекинг rtmlib на проверенные алгоритмы Pose2Sim — Sports2D (Венгерский) и DeepSORT.

**Architecture:** Пост-обработка: rtmlib детектирует людей → конвертируем в H3.6M → Sports2D/DeepSORT назначают стабильные ID → выбор цели по PersonClick. Новый модуль `src/tracking/` с двумя классами, общим интерфейсом `update() → list[int]`.

**Tech Stack:** scipy (Hungarian), numpy, deep-sort-realtime (опционально)

---

## Контекст

RTMPoseExtractor использует встроенный трекер rtmlib (OC-SORT), но метод `_assign_track_ids()` (строки 503-546 в `rtmlib_extractor.py`) — заглушка: он назначает **новые** ID каждый кадр (строка 543: `new_id = next_id + p`). Вся реальная ассоциация ложится на эвристику биометрической миграции, которая ломается на видео с несколькими людьми — скелет прыгает с одного человека на другого.

Pose2Sim (`/home/michael/Github/Pose2Sim/Pose2Sim/common.py`) предоставляет два проверенных метода:
1. **`sort_people_sports2d()`** (строка 1100) — матрица расстояний ключевых точек + Венгерский алгоритм (`scipy.optimize.linear_sum_assignment`)
2. **`sort_people_deepsort()`** (строка 1230) — DeepSORT с ReID-эмбеддингами через `deep-sort-realtime`

---

## Файлы

### Новые файлы
| Файл | Ответственность |
|------|-----------------|
| `src/tracking/__init__.py` | Экспорт `Sports2DTracker`, `DeepSORTTracker` |
| `src/tracking/sports2d.py` | `Sports2DTracker` — Венгерский алгоритм на расстояниях ключевых точек |
| `src/tracking/deepsort_tracker.py` | `DeepSORTTracker` — обёртка над DeepSORT (lazy-import) |
| `tests/tracking/__init__.py` | Пустой |
| `tests/tracking/test_sports2d.py` | 9 unit-тестов |
| `tests/tracking/test_deepsort_tracker.py` | 5 unit-тестов (skipif нет пакета) |

### Изменяемые файлы
| Файл | Изменение |
|------|-----------|
| `src/pose_estimation/rtmlib_extractor.py` | Параметр `tracking_mode`, замена вызова `_assign_track_ids` |
| `scripts/visualize_with_skeleton.py` | Передача `tracking_mode` в RTMPoseExtractor |
| `pyproject.toml` | Опциональная зависимость `[deepsort]` |

### Справочные файлы (только чтение)
| Файл | Что адаптируем |
|------|---------------|
| `/home/michael/Github/Pose2Sim/Pose2Sim/common.py:1100-1310` | `sort_people_sports2d`, `sort_people_deepsort`, `pad_shape`, `bbox_ltwh_compute` |

---

## Ключевые индексы H3.6M

```python
# src/types.py — H36Key
HIP_CENTER = 0   RHIP = 1   RKNEE = 2   RFOOT = 3
LHIP = 4         LKNEE = 5  LFOOT = 6
SPINE = 7        THORAX = 8  NECK = 9    HEAD = 10
LSHOULDER = 11   LELBOW = 12  LWRIST = 13
RSHOULDER = 14   RELBOW = 15  RWRIST = 16
```

---

## Task 1: Модуль `src/tracking/` + `Sports2DTracker`

**Files:**
- Create: `src/tracking/__init__.py`
- Create: `src/tracking/sports2d.py`
- Create: `tests/tracking/__init__.py`
- Create: `tests/tracking/test_sports2d.py`

- [ ] **Step 1: Создать `src/tracking/__init__.py`**

```python
"""Алгоритмы трекинга людей для мульти-персональной ассоциации поз.

Предоставляет пос frame-to-frame реидентификацию:
- Sports2D: Венгерский алгоритм по расстояниям ключевых точек (scipy)
- DeepSORT: Appearance-based ReID (требуется deep-sort-realtime)
"""

from .sports2d import Sports2DTracker
from .deepsort_tracker import DeepSORTTracker

__all__ = ["Sports2DTracker", "DeepSORTTracker"]
```

- [ ] **Step 2: Создать `src/tracking/sports2d.py`**

Адаптация Pose2Sim `sort_people_sports2d()` (Pose2Sim/common.py:1100-1199).
Ключевые отличия: обёрнут в stateful-класс, хранит `_prev_keypoints`/`_prev_scores`,
возвращает `list[int]` стабильных ID, работает с H3.6M (17kp).

```python
"""Трекинг людей Sports2D через Венгерский алгоритм.

Адаптация Pose2Sim/common.py sort_people_sports2d().
Попарные расстояния ключевых точек + scipy.optimize.linear_sum_assignment
для оптимального однозначного сопоставления между кадрами.

Reference:
    - Pose2Sim: https://github.com/Pose2Sim/Pose2Sim
    - scipy.optimize.linear_sum_assignment (Венгерский алгоритм)
"""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class Sports2DTracker:
    """Попарный трекер на Венгерском алгоритме.

    Хранит ключевые точки предыдущего кадра и назначает стабильные ID,
    находя оптимальное однозначное сопоставление, минимизирующее суммарное
    попарное расстояние ключевых точек.

    Args:
        max_dist: Максимальное допустимое расстояние для ассоциации
            (нормализованные координаты). Если None, автоматически вычисляется
            как 1.5 * средняя диагональ bbox.
        max_disappeared: Кадров без детекции перед удалением трека.
    """

    def __init__(
        self,
        max_dist: float | None = None,
        max_disappeared: int = 30,
    ) -> None:
        self._max_dist = max_dist
        self._max_disappeared = max_disappeared

        # Состояние
        self._prev_keypoints: np.ndarray | None = None  # (P_prev, 17, 2)
        self._prev_scores: np.ndarray | None = None      # (P_prev, 17)
        self._prev_track_ids: list[int] = []
        self._track_last_seen: dict[int, int] = {}
        self._frame_count: int = 0
        self._next_id: int = 0

    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Обновить трекер детекциями текущего кадра.

        Args:
            keypoints: (P, 17, 2) ключевые точки H3.6M (только xy).
            scores: (P, 17) confidence для каждого ключа.

        Returns:
            Список track ID, по одному на каждого обнаруженного человека.
        """
        n_curr = len(keypoints)

        # Пустой кадр
        if n_curr == 0:
            self._prev_keypoints = None
            self._prev_scores = None
            self._prev_track_ids = []
            self._frame_count += 1
            self._track_last_seen = {
                tid: last for tid, last in self._track_last_seen.items()
                if self._frame_count - last <= self._max_disappeared
            }
            return []

        # Первый кадр — новые ID
        if self._prev_keypoints is None or len(self._prev_keypoints) == 0:
            track_ids = list(range(self._next_id, self._next_id + n_curr))
            self._next_id += n_curr
            self._prev_keypoints = keypoints.copy()
            self._prev_scores = scores.copy()
            self._prev_track_ids = track_ids.copy()
            for tid in track_ids:
                self._track_last_seen[tid] = self._frame_count
            self._frame_count += 1
            return track_ids

        # Матрица расстояний: (n_prev, n_curr)
        prev_expanded = self._prev_keypoints[:, np.newaxis, :, :]  # (n_prev, 1, 17, 2)
        curr_expanded = keypoints[np.newaxis, :, :, :]             # (1, n_curr, 17, 2)
        diff = curr_expanded - prev_expanded
        distances_per_kp = np.sqrt(np.nansum(diff ** 2, axis=3))   # (n_prev, n_curr, 17)
        dist_matrix = np.nanmean(distances_per_kp, axis=2)         # (n_prev, n_curr)
        dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10)

        # Авто max_dist из bbox
        max_dist = self._max_dist
        if max_dist is None:
            all_kps = np.concatenate([self._prev_keypoints, keypoints], axis=0)
            x_min = np.nanmin(all_kps[:, :, 0], axis=1)
            x_max = np.nanmax(all_kps[:, :, 0], axis=1)
            y_min = np.nanmin(all_kps[:, :, 1], axis=1)
            y_max = np.nanmax(all_kps[:, :, 1], axis=1)
            widths = x_max - x_min
            heights = y_max - y_min
            diagonals = np.sqrt(widths ** 2 + heights ** 2)
            valid_diags = diagonals[diagonals > 0]
            if len(valid_diags) > 0:
                max_dist = float(1.5 * np.mean(valid_diags))
            else:
                max_dist = 1.0

        # Венгерский алгоритм
        pre_ids, curr_ids = linear_sum_assignment(dist_matrix)

        # Фильтр по порогу
        valid_associations: list[tuple[int, int]] = []
        for pre_id, curr_id in zip(pre_ids, curr_ids):
            if dist_matrix[pre_id, curr_id] <= max_dist:
                valid_associations.append((pre_id, curr_id))

        # Построить результат
        associated_curr = {curr_id for _, curr_id in valid_associations}
        unassociated_curr = [i for i in range(n_curr) if i not in associated_curr]

        track_ids: list[int] = [0] * n_curr
        for prev_idx, curr_idx in valid_associations:
            track_ids[curr_idx] = self._prev_track_ids[prev_idx]

        for curr_idx in unassociated_curr:
            track_ids[curr_idx] = self._next_id
            self._next_id += 1

        # Обновить состояние
        self._prev_keypoints = keypoints.copy()
        self._prev_scores = scores.copy()
        self._prev_track_ids = track_ids.copy()

        for tid in track_ids:
            self._track_last_seen[tid] = self._frame_count

        # Удалить старые треки
        self._track_last_seen = {
            tid: last for tid, last in self._track_last_seen.items()
            if self._frame_count - last <= self._max_disappeared
        }

        self._frame_count += 1
        return track_ids

    def reset(self) -> None:
        """Сбросить состояние трекера."""
        self._prev_keypoints = None
        self._prev_scores = None
        self._prev_track_ids = []
        self._track_last_seen = {}
        self._frame_count = 0
        self._next_id = 0
```

- [ ] **Step 3: Создать `tests/tracking/__init__.py`** (пустой файл)

- [ ] **Step 4: Создать `tests/tracking/test_sports2d.py`**

```python
"""Тесты Sports2DTracker."""

import numpy as np
import pytest

from src.tracking.sports2d import Sports2DTracker
from src.types import H36Key


def _make_person_pose(
    cx: float, cy: float, scale: float = 0.1
) -> np.ndarray:
    """Создать простую позу стоящего человека по центру (cx, cy)."""
    pose = np.zeros((17, 2), dtype=np.float32)
    s = scale / 0.1  # нормализация масштаба
    pose[H36Key.HIP_CENTER] = [cx, cy]
    pose[H36Key.RHIP] = [cx - 0.04 * s, cy]
    pose[H36Key.LHIP] = [cx + 0.04 * s, cy]
    pose[H36Key.RKNEE] = [cx - 0.04 * s, cy + 0.20 * s]
    pose[H36Key.LKNEE] = [cx + 0.04 * s, cy + 0.20 * s]
    pose[H36Key.RFOOT] = [cx - 0.04 * s, cy + 0.40 * s]
    pose[H36Key.LFOOT] = [cx + 0.04 * s, cy + 0.40 * s]
    pose[H36Key.SPINE] = [cx, cy - 0.15 * s]
    pose[H36Key.THORAX] = [cx, cy - 0.25 * s]
    pose[H36Key.NECK] = [cx, cy - 0.30 * s]
    pose[H36Key.HEAD] = [cx, cy - 0.35 * s]
    pose[H36Key.LSHOULDER] = [cx + 0.08 * s, cy - 0.25 * s]
    pose[H36Key.RSHOULDER] = [cx - 0.08 * s, cy - 0.25 * s]
    pose[H36Key.LELBOW] = [cx + 0.12 * s, cy - 0.15 * s]
    pose[H36Key.RELBOW] = [cx - 0.12 * s, cy - 0.15 * s]
    pose[H36Key.LWRIST] = [cx + 0.14 * s, cy - 0.05 * s]
    pose[H36Key.RWRIST] = [cx - 0.14 * s, cy - 0.05 * s]
    return pose


def _make_scores(n_persons: int, base: float = 0.8) -> np.ndarray:
    """Создать confidence scores."""
    return np.full((n_persons, 17), base, dtype=np.float32)


class TestFirstFrame:
    def test_first_frame_assigns_sequential_ids(self):
        """Первый кадр получает ID [0, 1, ...]."""
        tracker = Sports2DTracker()
        kps = np.array([_make_person_pose(0.3, 0.5),
                         _make_person_pose(0.7, 0.5)])
        scores = _make_scores(2)

        ids = tracker.update(kps, scores)

        assert ids == [0, 1]


class TestStableTracking:
    def test_stable_ids_small_movement(self):
        """Тот же человек с небольшим смещением получает тот же ID."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(
            np.array([person_a, _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )

        # Кадр 2 — оба чуть сдвинулись
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.69, 0.49)]),
            _make_scores(2),
        )

        assert ids1 == ids2 == [0, 1]

    def test_two_people_swap_order(self):
        """Люди меняются порядком в списке — ID остаются стабильными."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: A первым
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: B первым (порядок меняется)
        ids2 = tracker.update(np.array([person_b, person_a]), _make_scores(2))

        # A должен быть ID=0, B — ID=1. Во втором кадре B (индекс 0) = ID 1,
        # A (индекс 1) = ID 0.
        assert ids2[0] == 1  # B
        assert ids2[1] == 0  # A


class TestNewPerson:
    def test_extra_person_gets_new_id(self):
        """Новый человек на кадре 2 получает новый ID."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1: 1 человек
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 2: 2 человека
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )

        assert ids1 == [0]
        # ID 0 = первый человек (сопоставлен), ID 1 = новый
        assert 0 in ids2
        assert 1 in ids2


class TestPersonLeaves:
    def test_person_disappears_id_persists(self):
        """Если человек пропал на кадр, его ID сохраняется при возвращении."""
        tracker = Sports2DTracker(max_disappeared=30)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: 2 человека
        ids1 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: только 1 человек
        ids2 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 3: снова 2 человека
        ids3 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        assert ids1[0] == 0  # A
        assert ids1[1] == 1  # B
        assert ids2 == [0]    # A остался
        assert ids3[0] == 0  # A
        assert ids3[1] == 1  # B вернулся с тем же ID


class TestNaNHandling:
    def test_nan_keypoints_no_crash(self):
        """NaN в ключевых точках не вызывает краш."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1 — нормальный
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2 — с NaN
        person_b_nan = person_b.copy()
        person_b_nan[3, :] = np.nan  # RFOOT = NaN
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51), person_b_nan]),
            _make_scores(2),
        )

        assert len(ids2) == 2
        assert all(isinstance(x, int) for x in ids2)


class TestEmptyFrame:
    def test_empty_then_normal(self):
        """Пустой кадр, затем нормальный — не крашится."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1: пустой
        ids1 = tracker.update(np.zeros((0, 17, 2)), np.zeros((0, 17)))
        assert ids1 == []

        # Кадр 2: нормальный
        ids2 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids2 == [0]


class TestMaxDist:
    def test_large_jump_creates_new_track(self):
        """Слишком большое перемещение → новый трек."""
        tracker = Sports2DTracker(max_dist=0.01)  # очень маленький порог
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 2: человек далеко (0.3 → 0.8)
        ids2 = tracker.update(
            np.array([_make_person_pose(0.8, 0.5)]),
            _make_scores(1),
        )

        # Слишком далеко для max_dist=0.01 → новый ID
        assert ids1 == [0]
        assert ids2 == [1]


class TestAutoMaxDist:
    def test_auto_max_dist_works(self):
        """Авто-вычисленный max_dist из диагонали bbox."""
        tracker = Sports2DTracker(max_dist=None)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: небольшое смещение (должно сопоставиться)
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.69, 0.49)]),
            _make_scores(2),
        )

        assert ids2[0] == 0
        assert ids2[1] == 1


class TestTrackPurge:
    def test_old_tracks_purged(self):
        """Старые треки удаляются после max_disappeared."""
        tracker = Sports2DTracker(max_disappeared=2)
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids1 == [0]

        # 3 пустых кадра (превышает max_disappeared=2)
        for _ in range(3):
            tracker.update(np.zeros((0, 17, 2)), np.zeros((0, 17)))

        # Новый человек → новый ID (старый 0 удалён)
        ids5 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids5 == [1]
```

- [ ] **Step 5: Запустить тесты**

Run: `uv run pytest tests/tracking/test_sports2d.py -x --tb=short -v`
Expected: все 9 тестов PASS

- [ ] **Step 6: Коммит**

```bash
git add src/tracking/__init__.py src/tracking/sports2d.py tests/tracking/__init__.py tests/tracking/test_sports2d.py
git commit -m "feat(tracking): add Sports2DTracker with Hungarian algorithm"
```

---

## Task 2: `DeepSORTTracker`

**Files:**
- Create: `src/tracking/deepsort_tracker.py`
- Create: `tests/tracking/test_deepsort_tracker.py`

- [ ] **Step 1: Создать `src/tracking/deepsort_tracker.py`**

Адаптация Pose2Sim `sort_people_deepsort()` (Pose2Sim/common.py:1230-1278).
Lazy-import `deep-sort-realtime`. Тот же интерфейс `update() → list[int]`.

```python
"""Трекинг людей DeepSORT с appearance ReID.

Адаптация Pose2Sim/common.py sort_people_deepsort().
Bounding boxes из ключевых точек + DeepSORT (Kalman filter + appearance model)
для робастного трекинга в переполненных сценах.

Требуется: deep-sort-realtime (опциональная зависимость)

Reference:
    - Pose2Sim: https://github.com/Pose2Sim/Pose2Sim
    - deep-sort-realtime: https://github.com/levan92/deep_sort_realtime
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DeepSORTTracker:
    """Трекер на DeepSORT с appearance ReID.

    Обёртка над deep-sort-realtime DeepSort, вычисляет bounding boxes
    из H3.6M ключевых точек и маппит DeepSORT track IDs на стабильные
    внутренние ID.

    Args:
        max_age: Максимум пропущенных кадров перед удалением трека.
        n_init: Детекций подряд до подтверждения трека.
        max_cosine_distance: Порог косинусного расстояния для матчинга.
        nn_budget: Максимальный размер галереи appearance-дескрипторов.
        embedder_gpu: Использовать GPU для ReID-эмбеддингов.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 200,
        embedder_gpu: bool = True,
    ) -> None:
        self._max_age = max_age
        self._n_init = n_init
        self._max_cosine_distance = max_cosine_distance
        self._nn_budget = nn_budget
        self._embedder_gpu = embedder_gpu

        self._tracker = None
        self._deepsort_id_to_internal: dict[int, int] = {}
        self._next_internal_id: int = 0
        self._frame_count: int = 0

    def _ensure_tracker(self) -> None:
        """Ленивая инициализация DeepSORT."""
        if self._tracker is not None:
            return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(
                max_age=self._max_age,
                n_init=self._n_init,
                max_cosine_distance=self._max_cosine_distance,
                nn_budget=self._nn_budget,
                embedder_gpu=self._embedder_gpu,
            )
        except ImportError:
            raise ImportError(
                "deep-sort-realtime нужен для DeepSORT-трекинга. "
                "Установите: uv add deep-sort-realtime"
            )

    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame: np.ndarray | None = None,
    ) -> list[int]:
        """Обновить трекер детекциями текущего кадра.

        Args:
            keypoints: (P, 17, 2) ключевые точки H3.6M (только xy).
            scores: (P, 17) confidence для каждого ключа.
            frame: BGR-изображение (нужно для appearance features).
                Если None, работает только по bbox.

        Returns:
            Список внутренних track ID, по одному на каждого человека.
        """
        self._ensure_tracker()
        n_curr = len(keypoints)

        if n_curr == 0:
            self._frame_count += 1
            return []

        # Bounding boxes из ключевых точек
        x_min = np.nanmin(keypoints[:, :, 0], axis=1)
        x_max = np.nanmax(keypoints[:, :, 0], axis=1)
        y_min = np.nanmin(keypoints[:, :, 1], axis=1)
        y_max = np.nanmax(keypoints[:, :, 1], axis=1)

        padding = 20.0
        width = x_max - x_min
        height = y_max - y_min
        x_min_pad = x_min - width * padding / 100
        y_min_pad = y_min - height * padding / 200
        width_pad = width + 2 * width * padding / 100
        height_pad = height + height * padding / 100

        bboxes_ltwh = np.stack(
            (x_min_pad, y_min_pad, width_pad, height_pad), axis=1
        )
        bbox_scores = np.nanmean(scores, axis=1)

        detections = list(
            zip(
                bboxes_ltwh.tolist(),
                bbox_scores.tolist(),
                ["person"] * n_curr,
            )
        )
        det_ids = list(range(n_curr))

        if frame is not None:
            tracks = self._tracker.update_tracks(
                detections, frame=frame, others=det_ids
            )
        else:
            tracks = self._tracker.update_tracks(detections, others=det_ids)

        # Маппинг DeepSORT ID → наш внутренний ID
        track_ids = [-1] * n_curr
        for track in tracks:
            if not track.is_confirmed():
                continue
            ds_id = int(track.track_id) - 1  # DeepSORT IDs 1-based
            det_idx = track.get_det_supplementary()
            if ds_id not in self._deepsort_id_to_internal:
                self._deepsort_id_to_internal[ds_id] = self._next_internal_id
                self._next_internal_id += 1
            internal_id = self._deepsort_id_to_internal[ds_id]
            if det_idx is not None and 0 <= det_idx < n_curr:
                track_ids[det_idx] = internal_id

        # Несопоставлённые детекции — новые ID
        for i in range(n_curr):
            if track_ids[i] == -1:
                track_ids[i] = self._next_internal_id
                self._next_internal_id += 1

        self._frame_count += 1
        return track_ids

    def reset(self) -> None:
        """Сбросить состояние трекера."""
        self._tracker = None
        self._deepsort_id_to_internal = {}
        self._next_internal_id = 0
        self._frame_count = 0
```

- [ ] **Step 2: Создать `tests/tracking/test_deepsort_tracker.py`**

```python
"""Тесты DeepSORTTracker."""

import numpy as np
import pytest

from src.types import H36Key


def _make_person_pose(cx: float, cy: float, scale: float = 0.1) -> np.ndarray:
    """Создать простую позу H3.6M по центру (cx, cy)."""
    pose = np.zeros((17, 2), dtype=np.float32)
    s = scale / 0.1
    pose[H36Key.HIP_CENTER] = [cx, cy]
    pose[H36Key.RHIP] = [cx - 0.04 * s, cy]
    pose[H36Key.LHIP] = [cx + 0.04 * s, cy]
    pose[H36Key.RKNEE] = [cx - 0.04 * s, cy + 0.20 * s]
    pose[H36Key.LKNEE] = [cx + 0.04 * s, cy + 0.20 * s]
    pose[H36Key.RFOOT] = [cx - 0.04 * s, cy + 0.40 * s]
    pose[H36Key.LFOOT] = [cx + 0.04 * s, cy + 0.40 * s]
    pose[H36Key.SPINE] = [cx, cy - 0.15 * s]
    pose[H36Key.THORAX] = [cx, cy - 0.25 * s]
    pose[H36Key.NECK] = [cx, cy - 0.30 * s]
    pose[H36Key.HEAD] = [cx, cy - 0.35 * s]
    pose[H36Key.LSHOULDER] = [cx + 0.08 * s, cy - 0.25 * s]
    pose[H36Key.RSHOULDER] = [cx - 0.08 * s, cy - 0.25 * s]
    pose[H36Key.LELBOW] = [cx + 0.12 * s, cy - 0.15 * s]
    pose[H36Key.RELBOW] = [cx - 0.12 * s, cy - 0.15 * s]
    pose[H36Key.LWRIST] = [cx + 0.14 * s, cy - 0.05 * s]
    pose[H36Key.RWRIST] = [cx - 0.14 * s, cy - 0.05 * s]
    return pose


deepsort_available = pytest.mark.skipif(
    True,  # Запустить с: uv run pytest ... -k "not deepsort" или поставить пакет
    reason="deep-sort-realtime не установлен (uv add deep-sort-realtime)",
)


@deepsort_available
class TestDeepSORTBasic:
    def test_first_frame_assigns_ids(self):
        """Первый кадр — все получают ID."""
        from src.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5),
                         _make_person_pose(0.7, 0.5)])
        scores = np.full((2, 17), 0.8, dtype=np.float32)

        ids = tracker.update(kps, scores)

        assert len(ids) == 2
        assert all(i >= 0 for i in ids)

    def test_stable_ids_across_frames(self):
        """Тот же человек на нескольких кадрах получает тот же ID."""
        from src.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)
        scores = np.full((2, 17), 0.8, dtype=np.float32)

        # Кадр 1
        ids1 = tracker.update(np.array([person_a, person_b]), scores)
        # Кадр 2 — чуть сдвинулись
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.69, 0.49)]),
            scores,
        )

        assert ids1 == ids2

    def test_no_frame_graceful(self):
        """frame=None (без изображения) не крашит."""
        from src.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5)])
        scores = np.full((1, 17), 0.8, dtype=np.float32)

        ids = tracker.update(kps, scores, frame=None)

        assert len(ids) == 1

    def test_import_error_message(self):
        """Без пакета — понятное ImportError."""
        import importlib
        import sys

        # Временно скрываем пакет
        saved = sys.modules.get("deep_sort_realtime")
        sys.modules["deep_sort_realtime"] = None  # type: ignore[assignment]

        try:
            import importlib
            from src.tracking.deepsort_tracker import DeepSORTTracker

            tracker = DeepSORTTracker(embedder_gpu=False)
            with pytest.raises(ImportError, match="deep-sort-realtime"):
                tracker.update(
                    np.array([_make_person_pose(0.3, 0.5)]),
                    np.full((1, 17), 0.8, dtype=np.float32),
                )
        finally:
            if saved is not None:
                sys.modules["deep_sort_realtime"] = saved
            elif "deep_sort_realtime" in sys.modules:
                del sys.modules["deep_sort_realtime"]

    def test_reset_clears_state(self):
        """reset() сбрасывает всё состояние."""
        from src.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5)])
        scores = np.full((1, 17), 0.8, dtype=np.float32)

        tracker.update(kps, scores)
        tracker.reset()

        # После reset — первый кадр снова как новый
        ids = tracker.update(kps, scores)
        assert ids[0] == 0
```

- [ ] **Step 3: Запустить тесты**

Run: `uv run pytest tests/tracking/test_deepsort_tracker.py -x --tb=short -v`
Expected: все тесты skip (пакет не установлен) или PASS (если установлен)

- [ ] **Step 4: Коммит**

```bash
git add src/tracking/deepsort_tracker.py tests/tracking/test_deepsort_tracker.py
git commit -m "feat(tracking): add DeepSORTTracker wrapper"
```

---

## Task 3: Подключение к RTMPoseExtractor

**Files:**
- Modify: `src/pose_estimation/rtmlib_extractor.py:68-88,145-153,253-264,404-453`

- [ ] **Step 1: Добавить параметр `tracking_mode` в конструктор**

В файле `src/pose_estimation/rtmlib_extractor.py`, строка 68-88.
Добавить `tracking_mode: str = "auto"` в `__init__`:

```python
def __init__(
    self,
    mode: str = "balanced",
    tracking_backend: str = "rtmlib",
    tracking_mode: str = "auto",
    conf_threshold: float = 0.3,
    output_format: str = "normalized",
    det_frequency: int = 1,
    frame_skip: int = 1,
    device: str = "cpu",
    backend: str = "onnxruntime",
) -> None:
    ...
    self._tracking_mode = tracking_mode
```

- [ ] **Step 2: Добавить метод `_resolve_tracking_mode()`**

Добавить перед `_assign_track_ids()` (строка ~500):

```python
def _resolve_tracking_mode(self) -> str:
    """Разрешить 'auto' в конкретный режим трекинга."""
    if self._tracking_mode != "auto":
        return self._tracking_mode
    try:
        import deep_sort_realtime  # noqa: F401
        logger.info("Авто-выбор: DeepSORT (deep-sort-realtime доступен)")
        return "deepsort"
    except ImportError:
        logger.info("Авто-выбор: Sports2D (Венгерский алгоритм)")
        return "sports2d"
```

- [ ] **Step 3: Инстанцировать трекер в `extract_video_tracked()`**

В файле `src/pose_estimation/rtmlib_extractor.py`, после строки 152 (после `custom_tracker = None`),
добавить инициализацию трекера:

```python
# Новый трекинг (Sports2D / DeepSORT)
resolved_mode = self._resolve_tracking_mode()
sports2d_tracker = None
deepsort_tracker = None
if resolved_mode == "sports2d":
    from ..tracking.sports2d import Sports2DTracker
    sports2d_tracker = Sports2DTracker(max_disappeared=30)
elif resolved_mode == "deepsort":
    from ..tracking.deepsort_tracker import DeepSORTTracker
    deepsort_tracker = DeepSORTTracker(max_age=30, embedder_gpu=True)
```

- [ ] **Step 4: Заменить блок трек-ассоциации (строки 253-264)**

Заменить:
```python
# --- Track association ---
if self._tracking_backend == "custom":
    track_ids = custom_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
else:
    track_ids = self._assign_track_ids(h36m_poses, rtmlib_id_map, next_internal_id)
    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1
```

На:
```python
# --- Track association ---
if sports2d_tracker is not None:
    track_ids = sports2d_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
elif deepsort_tracker is not None:
    track_ids = deepsort_tracker.update(
        h36m_poses[:, :, :2], h36m_poses[:, :, 2], frame=frame
    )
elif self._tracking_backend == "custom":
    track_ids = custom_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
else:
    track_ids = self._assign_track_ids(h36m_poses, rtmlib_id_map, next_internal_id)
    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1
```

- [ ] **Step 5: Та же замена в `preview_persons()` (строки 448-453)**

Заменить:
```python
if tracker is not None:
    track_ids = tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
else:
    track_ids = self._assign_track_ids(h36m_poses, rtmlib_id_map, next_internal_id)
    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1
```

На:
```python
if sports2d_tracker is not None:
    track_ids = sports2d_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
elif deepsort_tracker is not None:
    track_ids = deepsort_tracker.update(
        h36m_poses[:, :, :2], h36m_poses[:, :, 2], frame=frame
    )
elif tracker is not None:
    track_ids = tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
else:
    track_ids = self._assign_track_ids(h36m_poses, rtmlib_id_map, next_internal_id)
    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1
```

Для этого нужно добавить такую же инициализацию трекера в начале `preview_persons()` (после строки 407):

```python
resolved_mode = self._resolve_tracking_mode()
sports2d_tracker = None
deepsort_tracker = None
if resolved_mode == "sports2d":
    from ..tracking.sports2d import Sports2DTracker
    sports2d_tracker = Sports2DTracker(max_disappeared=30)
elif resolved_mode == "deepsort":
    from ..tracking.deepsort_tracker import DeepSORTTracker
    deepsort_tracker = DeepSORTTracker(max_age=30, embedder_gpu=True)
```

- [ ] **Step 6: Запустить существующие тесты**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: все тесты PASS (старый код "rtmlib" и "custom" не сломан)

- [ ] **Step 7: Коммит**

```bash
git add src/pose_estimation/rtmlib_extractor.py
git commit -m "feat(tracking): wire Sports2D/DeepSORT into RTMPoseExtractor"
```

---

## Task 4: CLI + viz скрипт + pyproject.toml

**Files:**
- Modify: `scripts/visualize_with_skeleton.py:175-185`
- Modify: `src/cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Добавить `--tracking` в `visualize_with_skeleton.py`**

Найти где добавляется argparse-аргументы. Добавить:

```python
parser.add_argument(
    "--tracking",
    choices=["auto", "sports2d", "deepsort"],
    default="auto",
    help="Режим трекинга (default: auto). "
         "auto=sports2d, если deep-sort-realtime не установлен",
)
```

Передать в конструктор RTMPoseExtractor (строка 180):

```python
extractor = RTMPoseExtractor(
    output_format="normalized",
    conf_threshold=0.3,
    det_frequency=1,
    device="cuda",
    tracking_mode=args.tracking,
)
```

- [ ] **Step 2: Добавить `--tracking` в `src/cli.py`**

Добавить в `_get_extractor()`:

```python
def _get_extractor(pose_backend: str, **kwargs):
    if pose_backend == "rtmlib":
        from .pose_estimation.rtmlib_extractor import RTMPoseExtractor
        return RTMPoseExtractor(**kwargs)
    else:
        from .pose_estimation import H36MExtractor
        return H36MExtractor(**kwargs)
```

Добавить аргумент в subparsers:

```python
parser.add_argument(
    "--tracking",
    choices=["auto", "sports2d", "deepsort"],
    default="auto",
    help="Режим трекинга людей (default: auto)",
)
```

Передать `tracking_mode=args.tracking` в `_get_extractor()`.

- [ ] **Step 3: Добавить опциональную зависимость в `pyproject.toml`**

```toml
[project.optional-dependencies]
deepsort = ["deep-sort-realtime>=1.3.2"]
```

- [ ] **Step 4: Запустить все тесты**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add scripts/visualize_with_skeleton.py src/cli.py pyproject.toml
git commit -m "feat(tracking): add --tracking CLI flag and deepsort optional dep"
```

---

## Task 5: Ручная проверка

- [ ] **Step 1: Запуск с Sports2D на видео с несколькими людьми**

```bash
uv run python scripts/visualize_with_skeleton.py \
    /path/to/VOVA.MOV \
    --tracking sports2d --layer 1 --compress --crf 18
```

Проверить визуально: скелет не прыгает с одного человека на другого.

- [ ] **Step 2: Запуск с DeepSORT (если установлен)**

```bash
uv run python scripts/visualize_with_skeleton.py \
    /path/to/VOVA.MOV \
    --tracking deepsort --layer 1 --compress --crf 18
```

- [ ] **Step 3: Сравнить с `--tracking auto`**

```bash
uv run python scripts/visualize_with_skeleton.py \
    /path/to/VOVA.MOV \
    --tracking auto --layer 1 --compress --crf 18
```

---

## Риски

1. **Быстрое движение на льду** → большое межкадровое смещение. Авто `max_dist = 1.5 * bbox_diagonal` может быть слишком жёстким. Устранение: параметр `max_dist` выведен наружу, можно перенастроить.

2. **Переменное количество людей** → Венгерский алгоритм естественно работает с разным P. DeepSORT — через Kalman filter.

3. **DeepSORT ReID на льду** — похожая чёрная одежда может спутать appearance features. Sports2D (только ключевые точки) может быть лучше для фигурного катания. Поэтому `auto` по умолчанию → `sports2d`.

4. **Обратная совместимость** — режимы `"rtmlib"` и `"custom"` сохранены. Без `tracking_mode` — по умолчанию `"auto"` → `sports2d`.
