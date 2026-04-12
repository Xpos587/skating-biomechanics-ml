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
    """Попарный трекер на Венгерском алгоритме с фильтром Калмана.

    Хранит ключевые точки предыдущего кадра и назначает стабильные ID,
    находя оптимальное однозначное сопоставление, минимизирующее суммарное
    попарное расстояние ключевых точек.

    Для сопоставления используется 4D фильтр Калмана (постоянная скорость):
    state = [cx, cy, vx, vy]. Расстояние вычисляется между предсказанным
    центроидом (Kalman predict) и текущим центроидом детекции.

    При окклюзии (человек пропал на N кадров) хранит его последние keypoints
    и пробует восстановить ID при повторном появлении.

    Args:
        max_dist: Максимальное допустимое расстояние для ассоциации
            (нормализованные координаты). Если None, автоматически вычисляется
            как 1.5 * средняя диагональ bbox.
        max_disappeared: Кадров без детекции перед удалением трека.
        fps: Частота кадров для параметризации фильтра Калмана.
    """

    def __init__(
        self,
        max_dist: float | None = None,
        max_disappeared: int = 30,
        fps: float = 30.0,
    ) -> None:
        self._max_dist = max_dist
        self._max_disappeared = max_disappeared
        self._fps = fps

        # 4D constant-velocity Kalman: state = [cx, cy, vx, vy]
        # dt=1 (frame-based): vx is in normalized-coords-per-frame.
        # This converges faster than dt=1/fps for inter-frame association.
        self._F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        # Q: position noise low, velocity noise moderate (skaters change speed)
        self._Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        # R: trust measurements in normalized [0,1] coordinate space
        self._R = np.eye(2) * 1e-3
        self._P0 = np.eye(4) * 1.0

        # Per-track Kalman state: track_id -> (state (4,1), cov (4,4))
        self._kalman_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Состояние
        self._prev_keypoints: np.ndarray | None = None  # (P_prev, 17, 2)
        self._prev_scores: np.ndarray | None = None  # (P_prev, 17)
        self._prev_track_ids: list[int] = []
        self._track_last_seen: dict[int, int] = {}
        # Последние keypoints для треков, пропавших на текущем кадре
        self._lost_keypoints: dict[int, np.ndarray] = {}
        self._frame_count: int = 0
        self._next_id: int = 0

    @staticmethod
    def _centroid(keypoints: np.ndarray) -> np.ndarray:
        """(17,2) -> (2,) centroid [cx, cy]."""
        cx = float(np.nanmean(keypoints[:, 0]))
        cy = float(np.nanmean(keypoints[:, 1]))
        return np.array([cx, cy])

    def _kalman_predict(self, state: np.ndarray, cov: np.ndarray):
        x_pred = self._F @ state
        P_pred = self._F @ cov @ self._F.T + self._Q
        return x_pred, P_pred

    def _kalman_update(self, state: np.ndarray, cov: np.ndarray, z: np.ndarray):
        y = z - self._H @ state
        S = self._H @ cov @ self._H.T + self._R
        K = cov @ self._H.T @ np.linalg.inv(S)
        x_upd = state + K @ y
        I_KH = np.eye(4) - K @ self._H
        P_upd = I_KH @ cov @ I_KH.T + K @ self._R @ K.T
        return x_upd, P_upd

    def _predict_centroids(self) -> np.ndarray:
        """Predict centroid for each prev track. Returns (n_prev, 2)."""
        predicted = np.zeros((len(self._prev_track_ids), 2), dtype=np.float64)
        for i, tid in enumerate(self._prev_track_ids):
            if tid in self._kalman_states:
                state, cov = self._kalman_states[tid]
                state_pred, _ = self._kalman_predict(state, cov)
                predicted[i] = [state_pred[0, 0], state_pred[1, 0]]
            elif self._prev_keypoints is not None:
                predicted[i] = self._centroid(self._prev_keypoints[i])
        return predicted

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

        # Пустой кадр — сохранить текущие треки как "потерянные"
        if n_curr == 0:
            if self._prev_keypoints is not None and len(self._prev_keypoints) > 0:
                for i, tid in enumerate(self._prev_track_ids):
                    if tid not in self._lost_keypoints:
                        self._lost_keypoints[tid] = self._prev_keypoints[i].copy()
            self._prev_keypoints = None
            self._prev_scores = None
            self._prev_track_ids = []
            self._frame_count += 1
            self._purge_old_tracks()
            return []

        # Первый кадр — новые ID
        if self._prev_keypoints is None or len(self._prev_keypoints) == 0:
            track_ids = list(range(self._next_id, self._next_id + n_curr))
            self._next_id += n_curr
            # Initialize Kalman for each new track
            for i, tid in enumerate(track_ids):
                c = self._centroid(keypoints[i])
                state = np.array([[c[0]], [c[1]], [0.0], [0.0]])
                self._kalman_states[tid] = (state, self._P0.copy())
            self._prev_keypoints = keypoints.copy()
            self._prev_scores = scores.copy()
            self._prev_track_ids = track_ids.copy()
            for tid in track_ids:
                self._track_last_seen[tid] = self._frame_count
            self._frame_count += 1
            return track_ids

        # Predicted centroid distance matrix: (n_prev, n_curr)
        pred_centroids = self._predict_centroids()  # (n_prev, 2)
        curr_centroids = np.zeros((n_curr, 2), dtype=np.float64)
        for j in range(n_curr):
            curr_centroids[j] = self._centroid(keypoints[j])
        diff = pred_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)  # (n_prev, n_curr)
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
            diagonals = np.sqrt(widths**2 + heights**2)
            valid_diags = diagonals[diagonals > 0]
            if len(valid_diags) > 0:
                max_dist = float(1.5 * np.mean(valid_diags))
            else:
                max_dist = 1.0

        # Венгерский алгоритм
        pre_ids, curr_ids = linear_sum_assignment(dist_matrix)

        # Фильтр по порогу
        valid_associations: list[tuple[int, int]] = []
        for pre_id, curr_id in zip(pre_ids, curr_ids, strict=False):
            if dist_matrix[pre_id, curr_id] <= max_dist:
                valid_associations.append((pre_id, curr_id))

        # Построить результат
        associated_curr = {curr_id for _, curr_id in valid_associations}
        associated_prev = {prev_id for prev_id, _ in valid_associations}
        unassociated_curr = [i for i in range(n_curr) if i not in associated_curr]

        track_ids: list[int] = [0] * n_curr
        for prev_idx, curr_idx in valid_associations:
            track_ids[curr_idx] = self._prev_track_ids[prev_idx]

        # Update Kalman state for matched tracks
        for prev_idx, curr_idx in valid_associations:
            tid = self._prev_track_ids[prev_idx]
            c = self._centroid(keypoints[curr_idx])
            z = np.array([[c[0]], [c[1]]])
            if tid in self._kalman_states:
                state, cov = self._kalman_states[tid]
                state_pred, cov_pred = self._kalman_predict(state, cov)
                state_upd, cov_upd = self._kalman_update(state_pred, cov_pred, z)
                self._kalman_states[tid] = (state_upd, cov_upd)
            else:
                state = np.array([[c[0]], [c[1]], [0.0], [0.0]])
                self._kalman_states[tid] = (state, self._P0.copy())

        # Сохранить непревзошедшие треки как "потерянные"
        for prev_idx in range(len(self._prev_keypoints)):
            if prev_idx not in associated_prev:
                tid = self._prev_track_ids[prev_idx]
                # Predict-forward Kalman state (drift during occlusion)
                if tid in self._kalman_states:
                    state, cov = self._kalman_states[tid]
                    state_pred, cov_pred = self._kalman_predict(state, cov)
                    self._kalman_states[tid] = (state_pred, cov_pred)
                if tid not in self._lost_keypoints:
                    self._lost_keypoints[tid] = self._prev_keypoints[prev_idx].copy()

        # Re-ассоциация: попытка сопоставить несопоставленных с пропавшими треками
        if self._lost_keypoints and unassociated_curr:
            lost_ids = [
                tid
                for tid in self._lost_keypoints
                if self._frame_count - self._track_last_seen.get(tid, 0) <= self._max_disappeared
            ]
            if lost_ids:
                lost_kps = np.array([self._lost_keypoints[tid] for tid in lost_ids])
                unassoc_kps = np.array([keypoints[i] for i in unassociated_curr])

                disp_exp = lost_kps[:, np.newaxis, :, :]  # (n_lost, 1, 17, 2)
                unassoc_exp = unassoc_kps[np.newaxis, :, :, :]  # (1, n_unassoc, 17, 2)
                d = unassoc_exp - disp_exp
                dists = np.sqrt(np.nansum(d**2, axis=3))
                lost_matrix = np.nanmean(dists, axis=2)
                lost_matrix = np.nan_to_num(lost_matrix, nan=1e10, posinf=1e10)

                lost_ids_idx, unassoc_ids_idx = linear_sum_assignment(lost_matrix)

                matched_unassoc: set[int] = set()
                for l_idx, u_idx in zip(lost_ids_idx, unassoc_ids_idx, strict=False):
                    if lost_matrix[l_idx, u_idx] <= max_dist:
                        real_curr_idx = unassociated_curr[u_idx]
                        track_ids[real_curr_idx] = lost_ids[l_idx]
                        matched_unassoc.add(u_idx)
                        # Update Kalman for re-associated lost track
                        tid = lost_ids[l_idx]
                        c = self._centroid(keypoints[real_curr_idx])
                        z = np.array([[c[0]], [c[1]]])
                        if tid in self._kalman_states:
                            state, cov = self._kalman_states[tid]
                            state_upd, cov_upd = self._kalman_update(state, cov, z)
                            self._kalman_states[tid] = (state_upd, cov_upd)
                        # Трек восстановлен — убрать из потерянных
                        self._lost_keypoints.pop(lost_ids[l_idx], None)

                # Обновить unassociated_curr, убрав сопоставленные
                unassociated_curr = [
                    unassociated_curr[i]
                    for i in range(len(unassociated_curr))
                    if i not in matched_unassoc
                ]

        # Новые ID для оставшихся
        for curr_idx in unassociated_curr:
            track_ids[curr_idx] = self._next_id
            c = self._centroid(keypoints[curr_idx])
            state = np.array([[c[0]], [c[1]], [0.0], [0.0]])
            self._kalman_states[self._next_id] = (state, self._P0.copy())
            self._next_id += 1

        # Обновить состояние
        self._prev_keypoints = keypoints.copy()
        self._prev_scores = scores.copy()
        self._prev_track_ids = track_ids.copy()

        for tid in track_ids:
            self._track_last_seen[tid] = self._frame_count
            # Восстановленные треки уже удалены из _lost_keypoints выше

        # Удалить старые потерянные треки
        self._lost_keypoints = {
            tid: kps
            for tid, kps in self._lost_keypoints.items()
            if self._frame_count - self._track_last_seen.get(tid, 0) <= self._max_disappeared
        }
        self._purge_old_tracks()

        self._frame_count += 1
        return track_ids

    def _purge_old_tracks(self) -> None:
        """Удалить старые треки из всех структур."""
        self._track_last_seen = {
            tid: last
            for tid, last in self._track_last_seen.items()
            if self._frame_count - last <= self._max_disappeared
        }
        self._lost_keypoints = {
            tid: kps for tid, kps in self._lost_keypoints.items() if tid in self._track_last_seen
        }
        self._kalman_states = {
            tid: state for tid, state in self._kalman_states.items() if tid in self._track_last_seen
        }

    def reset(self) -> None:
        """Сбросить состояние трекера."""
        self._prev_keypoints = None
        self._prev_scores = None
        self._prev_track_ids = []
        self._track_last_seen = {}
        self._lost_keypoints = {}
        self._kalman_states = {}
        self._frame_count = 0
        self._next_id = 0
