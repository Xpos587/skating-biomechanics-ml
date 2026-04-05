"""Tests for unified put_text API."""

import numpy as np

from src.visualization.core.text import measure_text_size_fast, put_text


def _make_frame(h=200, w=400):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_put_text_cyrillic_no_full_frame_change():
    """Only the text region should change, rest stays black."""
    frame = _make_frame()
    put_text(frame, "Привет", (10, 10), font_size=20)
    # Top-left corner should have non-zero pixels
    assert frame[:50, :100].sum() > 0
    # Bottom-right corner should be untouched
    assert frame[150:, 300:].sum() == 0


def test_put_text_ascii():
    """ASCII text should render."""
    frame = _make_frame()
    put_text(frame, "Hello", (10, 10), font_size=16)
    assert frame[:30, :80].sum() > 0


def test_put_text_returns_frame():
    frame = _make_frame()
    result = put_text(frame, "Test", (10, 10))
    assert result is frame


def test_put_text_with_background():
    frame = _make_frame()
    put_text(frame, "Test", (10, 10), bg_color=(0, 0, 0), bg_alpha=0.7)
    # Background region should exist
    assert frame[:30, :80].sum() > 0


def test_put_text_empty_string_no_crash():
    frame = _make_frame()
    original = frame.copy()
    put_text(frame, "", (10, 10))
    np.testing.assert_array_equal(frame, original)


def test_measure_text_size_fast_returns_positive():
    w, h = measure_text_size_fast("Привет мир", font_size=16)
    assert w > 0
    assert h > 0


def test_measure_text_size_fast_ascii():
    w, h = measure_text_size_fast("Hello", font_size=16)
    assert w > 0
    assert h > 0
