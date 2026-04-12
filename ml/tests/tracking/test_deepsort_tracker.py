"""Тесты DeepSORTTracker."""

import numpy as np
import pytest

from skating_ml.types import H36Key

# All tests skip because deep-sort-realtime is not installed
deepsort_available = pytest.mark.skipif(
    True,  # Packages not installed — all tests skip until installed
    reason="deep-sort-realtime не установлен",
)


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


@deepsort_available
class TestDeepSORTBasic:
    def test_first_frame_assigns_ids(self):
        """Первый кадр — все получают ID."""
        from skating_ml.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5), _make_person_pose(0.7, 0.5)])
        scores = np.full((2, 17), 0.8, dtype=np.float32)

        ids = tracker.update(kps, scores)

        assert len(ids) == 2
        assert all(i >= 0 for i in ids)

    def test_stable_ids_across_frames(self):
        """Тот же человек на нескольких кадрах получает тот же ID."""
        from skating_ml.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)
        scores = np.full((2, 17), 0.8, dtype=np.float32)

        ids1 = tracker.update(np.array([person_a, person_b]), scores)
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51), _make_person_pose(0.69, 0.49)]),
            scores,
        )

        assert ids1 == ids2

    def test_no_frame_graceful(self):
        """frame=None (без изображения) не крашит."""
        from skating_ml.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5)])
        scores = np.full((1, 17), 0.8, dtype=np.float32)

        ids = tracker.update(kps, scores, frame=None)

        assert len(ids) == 1

    def test_import_error_message(self):
        """Без пакета — понятное ImportError."""
        from skating_ml.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        with pytest.raises(ImportError, match="deep-sort-realtime"):
            tracker.update(
                np.array([_make_person_pose(0.3, 0.5)]),
                np.full((1, 17), 0.8, dtype=np.float32),
            )

    def test_reset_clears_state(self):
        """reset() сбрасывает всё состояние."""
        from skating_ml.tracking.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker(embedder_gpu=False)
        kps = np.array([_make_person_pose(0.3, 0.5)])
        scores = np.full((1, 17), 0.8, dtype=np.float32)

        tracker.update(kps, scores)
        tracker.reset()

        ids = tracker.update(kps, scores)
        assert ids[0] == 0
