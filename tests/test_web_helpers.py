"""Tests for gradio_helpers — pure functions used by the Gradio UI."""

import numpy as np

from src.gradio_helpers import (
    match_click_to_person,
    render_person_preview,
)


class TestMatchClickToPerson:
    """Tests for matching a click coordinate to detected person bboxes."""

    def test_click_inside_single_person(self):
        persons = [
            {"track_id": 0, "hits": 30, "bbox": (0.3, 0.2, 0.7, 0.8), "mid_hip": (0.5, 0.5)},
        ]
        result = match_click_to_person(persons, x=0.5, y=0.5)
        assert result is not None
        assert result["track_id"] == 0

    def test_click_outside_all_persons(self):
        persons = [
            {"track_id": 0, "hits": 30, "bbox": (0.3, 0.2, 0.7, 0.8), "mid_hip": (0.5, 0.5)},
        ]
        result = match_click_to_person(persons, x=0.1, y=0.1)
        assert result is None

    def test_click_between_two_persons_selects_closest_midhip(self):
        persons = [
            {"track_id": 0, "hits": 30, "bbox": (0.1, 0.2, 0.4, 0.8), "mid_hip": (0.25, 0.5)},
            {"track_id": 1, "hits": 20, "bbox": (0.6, 0.2, 0.9, 0.8), "mid_hip": (0.75, 0.5)},
        ]
        result = match_click_to_person(persons, x=0.3, y=0.5)
        assert result is not None
        assert result["track_id"] == 0

    def test_empty_persons_list(self):
        result = match_click_to_person([], x=0.5, y=0.5)
        assert result is None


class TestRenderPersonPreview:
    """Tests for rendering person bounding boxes on a frame."""

    def test_renders_numbered_boxes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = [
            {"track_id": 0, "hits": 30, "bbox": (0.1, 0.2, 0.4, 0.8), "mid_hip": (0.25, 0.5)},
            {"track_id": 1, "hits": 20, "bbox": (0.6, 0.2, 0.9, 0.8), "mid_hip": (0.75, 0.5)},
        ]
        result = render_person_preview(frame, persons)
        assert result.shape == frame.shape
        assert result.sum() > 0

    def test_no_persons_returns_original_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = render_person_preview(frame, [])
        np.testing.assert_array_equal(result, frame)

    def test_selected_person_gets_green_border(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = [
            {"track_id": 0, "hits": 30, "bbox": (0.1, 0.2, 0.4, 0.8), "mid_hip": (0.25, 0.5)},
        ]
        result = render_person_preview(frame, persons, selected_idx=0)
        h, w = frame.shape[:2]
        x1 = int(0.1 * w)
        y1 = int(0.2 * h)
        green_val = result[y1 + 2, x1 + 2, 1]
        assert green_val > 0
