"""Композитор слоёв для streamlit-overlay.

Layer composer for generating overlay masks.
"""

from collections import deque

import numpy as np

from src.pose_estimation import H36M_SKELETON_EDGES, H36Key
from src.ui.types import LayerSettings, ProcessedPoses
from src.visualization import (
    draw_blade_state_3d_hud,
    draw_skeleton,
    draw_skeleton_3d_pip,
    draw_trails,
    draw_velocity_vectors,
)


class LayerComposer:
    """Генерирует mask слоёв для overlay (НЕ модифицирует оригинальный кадр).

    Generates layer masks for overlay without modifying original frame.
    """

    def __init__(self) -> None:
        """Инициализация композитора."""
        self._trail_history_left: deque = deque(maxlen=50)
        self._trail_history_right: deque = deque(maxlen=50)

    def compose_mask(
        self,
        frame_shape: tuple[int, int, int],
        frame_idx: int,
        poses: ProcessedPoses,
        settings: LayerSettings,
    ) -> np.ndarray:
        """Создать mask слоёв для overlay.

        Args:
            frame_shape: Форма кадра (H, W, 3).
            frame_idx: Индекс кадра в видео.
            poses: Обработанные позы.
            settings: Настройки слоёв.

        Returns:
            Mask массив (H, W, 3) с визуализацией слоёв.
            Прозрачные пиксели = (0, 0, 0).
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Find corresponding pose index
        pose_idx = self._find_pose_index(frame_idx, poses.pose_frame_indices)
        if pose_idx is None:
            return mask

        # Layer 0: Skeleton
        if settings.skeleton:
            mask = self._draw_skeleton_layer(
                mask, pose_idx, poses, settings, h, w
            )

        # Layer 1: Kinematics
        if settings.velocity:
            mask = self._draw_velocity_layer(
                mask, pose_idx, poses, h, w
            )

        if settings.trails:
            mask = self._draw_trails_layer(
                mask, pose_idx, poses, settings, h, w
            )

        # Layer 2: Technical
        if settings.edge_indicators:
            mask = self._draw_edge_layer(
                mask, pose_idx, poses, h, w
            )

        return mask

    def _draw_skeleton_layer(
        self,
        mask: np.ndarray,
        pose_idx: int,
        poses: ProcessedPoses,
        settings: LayerSettings,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Отрисовать скелет на mask.

        Args:
            mask: Текущий mask.
            pose_idx: Индекс позы.
            poses: Обработанные позы.
            settings: Настройки.
            h: Высота.
            w: Ширина.

        Returns:
            Mask со скелетом.
        """
        if settings.enable_3d and poses.has_3d:
            pose_3d = poses.poses_3d[pose_idx]
            # Create transparent background for 3D skeleton
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg = draw_skeleton_3d_pip(
                bg,
                pose_3d,
                H36M_SKELETON_EDGES,
                h,
                w,
                camera_z=settings.d_3d_scale,
                auto_scale=not settings.no_3d_autoscale,
            )
            # Add non-black pixels to mask
            non_black = np.any(bg > 0, axis=2)
            mask[non_black] = bg[non_black]
        else:
            pose_h36m = poses.poses_h36m[pose_idx]
            pose_h36m_px = pose_h36m * np.array([w, h])
            mask = draw_skeleton(mask, pose_h36m_px, h, w)

        return mask

    def _draw_velocity_layer(
        self,
        mask: np.ndarray,
        pose_idx: int,
        poses: ProcessedPoses,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Отрисовать вектора скорости на mask."""
        if pose_idx < len(poses.poses_h36m):
            mask = draw_velocity_vectors(
                mask,
                poses.poses_h36m,
                pose_idx,
                poses.fps,
                h,
                w,
            )
        return mask

    def _draw_trails_layer(
        self,
        mask: np.ndarray,
        pose_idx: int,
        poses: ProcessedPoses,
        settings: LayerSettings,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Отрисовать траектории на mask."""
        if pose_idx >= len(poses.poses_h36m):
            return mask

        current_pose_h36m = poses.poses_h36m[pose_idx]
        self._trail_history_left.append(current_pose_h36m.copy())
        self._trail_history_right.append(current_pose_h36m.copy())

        while len(self._trail_history_left) > settings.trail_length:
            self._trail_history_left.popleft()
        while len(self._trail_history_right) > settings.trail_length:
            self._trail_history_right.popleft()

        if len(self._trail_history_left) > 1:
            mask = draw_trails(mask, self._trail_history_left, H36Key.LFOOT, h, w)
        if len(self._trail_history_right) > 1:
            mask = draw_trails(mask, self._trail_history_right, H36Key.RFOOT, h, w)

        return mask

    def _draw_edge_layer(
        self,
        mask: np.ndarray,
        pose_idx: int,
        poses: ProcessedPoses,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Отрисовать индикаторы ребра на mask."""
        if not poses.has_blade_states:
            return mask
        if not poses.blade_states_left or pose_idx >= len(poses.blade_states_left):
            return mask

        state_left = poses.blade_states_left[pose_idx]
        state_right = (
            poses.blade_states_right[pose_idx]
            if poses.blade_states_right
            else None
        )
        mask = draw_blade_state_3d_hud(mask, state_left, state_right, h, w)
        return mask

    def _find_pose_index(
        self,
        frame_idx: int,
        pose_frame_indices: np.ndarray | None,
    ) -> int | None:
        """Найти индекс позы для кадра."""
        if pose_frame_indices is None:
            return frame_idx

        for i, pose_frame_idx in enumerate(pose_frame_indices):
            if pose_frame_idx == frame_idx:
                return i
            if pose_frame_idx > frame_idx:
                return i - 1 if i > 0 else 0

        return len(pose_frame_indices) - 1 if len(pose_frame_indices) > 0 else None

    def clear_trails(self) -> None:
        """Очистить историю траекторий."""
        self._trail_history_left.clear()
        self._trail_history_right.clear()
