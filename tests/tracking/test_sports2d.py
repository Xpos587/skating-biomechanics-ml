"""Тесты Sports2DTracker."""

import numpy as np

from src.tracking.sports2d import Sports2DTracker
from src.types import H36Key


def _make_person_pose(cx: float, cy: float, scale: float = 0.1) -> np.ndarray:
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
        kps = np.array([_make_person_pose(0.3, 0.5), _make_person_pose(0.7, 0.5)])
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
            np.array([_make_person_pose(0.31, 0.51), _make_person_pose(0.69, 0.49)]),
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

        # A = ID 0, B = ID 1. Во втором кадре B (индекс 0) = ID 1,
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
            np.array([_make_person_pose(0.31, 0.51), _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )

        assert ids1 == [0]
        # ID 0 = первый человек (сопоставлен), ID 1 = новый
        assert 0 in ids2
        assert 1 in ids2


class TestPersonLeaves:
    def test_person_leaves_a_remains(self):
        """Если человек пропал на кадр, оставшийся сохраняет ID."""
        tracker = Sports2DTracker(max_disappeared=30)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: 2 человека
        ids1 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: только 1 человек — A сохраняет ID
        ids2 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 3: снова 2 человека — A сохраняет ID
        ids3 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        assert ids1[0] == 0  # A
        assert ids1[1] == 1  # B
        assert ids2 == [0]  # A остался
        assert ids3[0] == 0  # A всё ещё ID 0

    def test_person_reappears_after_occlusion(self):
        """Человек вернулся после окклюзии — сохраняет тот же ID."""
        tracker = Sports2DTracker(max_disappeared=30)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: 2 человека
        ids1 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадры 2-3: только occluder (B) — A пропал
        tracker.update(np.array([person_b]), _make_scores(1))
        tracker.update(np.array([person_b]), _make_scores(1))

        # Кадр 4: A снова виден вместе с B
        ids4 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        assert ids1[0] == 0  # A
        assert ids1[1] == 1  # B
        assert ids4[0] == 0  # A восстановил свой ID после окклюзии


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
            np.array([_make_person_pose(0.31, 0.51), _make_person_pose(0.69, 0.49)]),
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


class TestKalmanPrediction:
    """Tests for Kalman filter velocity prediction in Sports2DTracker."""

    def test_occluder_does_not_steal_track(self):
        """After velocity builds, occluder at target's old position does not steal ID."""
        tracker = Sports2DTracker(max_dist=None, fps=30.0)

        # Frames 1-5: A moves right steadily (+0.03/frame), B stays at 0.7.
        # Kalman converges velocity estimate to ~0.03/frame.
        for i in range(5):
            a = _make_person_pose(0.3 + 0.03 * i, 0.5)
            b = _make_person_pose(0.7, 0.5)
            ids = tracker.update(np.array([a, b]), _make_scores(2))
            assert ids[0] == 0, f"Frame {i + 1}: expected A=0, got {ids}"

        # Frame 6: A continues to 0.45, B jumps to A's previous position (0.42).
        # Kalman predicts A at ~0.45 (velocity converged). B at 0.42 is behind.
        a_continues = _make_person_pose(0.45, 0.5)
        b_at_a_old = _make_person_pose(0.42, 0.5)
        ids6 = tracker.update(np.array([a_continues, b_at_a_old]), _make_scores(2))

        # ID 0 stays on index 0 (A), not stolen by B
        assert ids6[0] == 0

    def test_stable_tracking_with_kalman(self):
        """Kalman does not break normal small movements."""
        tracker = Sports2DTracker(fps=30.0)
        prev_ids = None
        for frame in range(10):
            a = _make_person_pose(0.3 + 0.001 * frame, 0.5)
            b = _make_person_pose(0.7 - 0.001 * frame, 0.5)
            ids = tracker.update(np.array([a, b]), _make_scores(2))
            if prev_ids is not None:
                assert ids == prev_ids, f"IDs changed at frame {frame}"
            prev_ids = ids
        assert prev_ids == [0, 1]

    def test_kalman_velocity_builds_over_frames(self):
        """After several frames, velocity estimate supports association."""
        tracker = Sports2DTracker(fps=30.0)
        for i in range(5):
            a = _make_person_pose(0.3 + 0.01 * i, 0.5)
            tracker.update(np.array([a]), _make_scores(1))
        a = _make_person_pose(0.35, 0.5)
        ids = tracker.update(np.array([a]), _make_scores(1))
        assert ids == [0]

    def test_reappear_after_occlusion_with_kalman(self):
        """Lost track recovery still works."""
        tracker = Sports2DTracker(max_disappeared=30, fps=30.0)
        ids1 = tracker.update(
            np.array([_make_person_pose(0.3, 0.5), _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )
        assert ids1 == [0, 1]
        tracker.update(np.array([_make_person_pose(0.7, 0.5)]), _make_scores(1))
        tracker.update(np.array([_make_person_pose(0.7, 0.5)]), _make_scores(1))
        ids4 = tracker.update(
            np.array([_make_person_pose(0.3, 0.5), _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )
        assert 0 in ids4

    def test_empty_then_normal_with_kalman(self):
        """Empty frame (all people gone) clears previous state — new IDs on return."""
        tracker = Sports2DTracker(fps=30.0)
        tracker.update(np.array([_make_person_pose(0.3, 0.5)]), _make_scores(1))
        tracker.update(np.zeros((0, 17, 2)), np.zeros((0, 17)))
        ids3 = tracker.update(np.array([_make_person_pose(0.3, 0.5)]), _make_scores(1))
        # After full empty frame, _prev_keypoints is None → first-frame branch → new ID
        assert ids3 == [1]

    def test_swap_order_with_kalman(self):
        """Order swap — IDs stay stable."""
        tracker = Sports2DTracker(fps=30.0)
        a, b = _make_person_pose(0.3, 0.5), _make_person_pose(0.7, 0.5)
        tracker.update(np.array([a, b]), _make_scores(2))
        ids2 = tracker.update(np.array([b, a]), _make_scores(2))
        assert ids2[0] == 1
        assert ids2[1] == 0

    def test_kalman_reset_clears_state(self):
        """reset() clears Kalman state."""
        tracker = Sports2DTracker(fps=30.0)
        tracker.update(np.array([_make_person_pose(0.3, 0.5)]), _make_scores(1))
        assert len(tracker._kalman_states) == 1
        tracker.reset()
        assert len(tracker._kalman_states) == 0
        assert tracker._next_id == 0
