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
            ) from None

    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame: np.ndarray | None = None,
        frame_width: int = 1,
        frame_height: int = 1,
    ) -> list[int]:
        """Обновить трекер детекциями текущего кадра.

        Args:
            keypoints: (P, 17, 2) ключевые точки H3.6M (только xy).
            scores: (P, 17) confidence для каждого ключа.
            frame: BGR-изображение (нужно для appearance features).
                Если None, работает только по bbox.
            frame_width: Ширина кадра в пикселях (для конвертации bbox).
            frame_height: Высота кадра в пикселях (для конвертации bbox).

        Returns:
            Список внутренних track ID, по одному на каждого человека.
        """
        self._ensure_tracker()
        n_curr = len(keypoints)

        if n_curr == 0:
            self._frame_count += 1
            return []

        # Bounding boxes из ключевых точек (Pose2Sim bbox_ltwh_compute)
        x_min = np.nanmin(keypoints[:, :, 0], axis=1)
        x_max = np.nanmax(keypoints[:, :, 0], axis=1)
        y_min = np.nanmin(keypoints[:, :, 1], axis=1)
        y_max = np.nanmax(keypoints[:, :, 1], axis=1)

        padding = 20.0
        width = x_max - x_min
        height = y_max - y_min
        x_min_pad = x_min - width * padding / 100
        y_min_pad = y_min - height * padding / 200  # padding/2% vertical
        width_pad = width + 2 * width * padding / 100
        height_pad = height + height * padding / 100

        bboxes_ltwh = np.stack((x_min_pad, y_min_pad, width_pad, height_pad), axis=1)

        # DeepSORT needs pixel coordinates for frame cropping
        bboxes_ltwh[:, 0] *= frame_width  # x_min
        bboxes_ltwh[:, 1] *= frame_height  # y_min
        bboxes_ltwh[:, 2] *= frame_width  # width
        bboxes_ltwh[:, 3] *= frame_height  # height
        bbox_scores = np.nanmean(scores, axis=1)

        detections = list(
            zip(
                bboxes_ltwh.tolist(),
                bbox_scores.tolist(),
                ["person"] * n_curr,
                strict=False,
            )
        )
        det_ids = list(range(n_curr))

        if frame is not None and self._tracker is not None:
            tracks = self._tracker.update_tracks(detections, frame=frame, others=det_ids)
        elif self._tracker is not None:
            tracks = self._tracker.update_tracks(detections, others=det_ids)
        else:
            tracks = []

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
