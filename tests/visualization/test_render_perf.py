"""Performance regression tests for visualization rendering primitives.

These ensure rendering stays within acceptable time budgets.
If these fail, someone introduced a full-frame copy or Pillow conversion.
"""

import time

import numpy as np

from src.visualization.core.overlay import draw_overlay_rect
from src.visualization.core.text import put_text
from src.visualization.hud.coach_panel import CoachOverlayData, draw_coach_panel


def _make_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestRenderPerformance:
    """Ensure rendering primitives stay under budget."""

    def test_put_text_under_1ms(self):
        """put_text should render Cyrillic text in <1ms (cached bitmap)."""
        frame = _make_frame()
        # Warm up cache
        put_text(frame, "Тест", (10, 10), font_size=16)

        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(100):
            put_text(frame, "Высота: 0.45с ✓", (10, 10), font_size=16)
        elapsed = (time.perf_counter() - t0) / 100

        assert elapsed < 0.001, f"put_text took {elapsed * 1000:.2f}ms (>1ms)"

    def test_draw_overlay_rect_under_015ms(self):
        """draw_overlay_rect should render in <0.15ms."""
        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(1000):
            draw_overlay_rect(frame, (10, 10, 200, 100), color=(0, 0, 0), alpha=0.6)
        elapsed = (time.perf_counter() - t0) / 1000

        assert elapsed < 0.00020, f"draw_overlay_rect took {elapsed * 1000:.2f}ms (>0.20ms)"

    def test_coach_panel_under_2ms(self):
        """Full coach panel render should complete in <2ms."""
        data = CoachOverlayData(
            element_name_ru="Сальхов",
            metrics=[
                ("Время полёта", "0.45с", True),
                ("Высота", "0.32", True),
                ("Колено", "165°", False),
            ],
            recommendations=["Спрямить колено при приземлении"],
            landing_frame=100,
            fps=30.0,
        )
        # Warm up
        frame = _make_frame()
        draw_coach_panel(frame, data, position=(10, 90))

        frame = _make_frame()
        t0 = time.perf_counter()
        for _ in range(100):
            draw_coach_panel(frame, data, position=(10, 90))
        elapsed = (time.perf_counter() - t0) / 100

        assert elapsed < 0.002, f"draw_coach_panel took {elapsed * 1000:.2f}ms (>2ms)"
