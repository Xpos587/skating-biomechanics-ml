"""Tests for AI coach overlay panel."""

import numpy as np

from src.types import ElementPhase, H36Key, MetricResult
from src.visualization.hud.coach_panel import CoachOverlayData, compute_coach_overlays


class TestCoachOverlayData:
    """Test CoachOverlayData dataclass."""

    def test_creation(self):
        data = CoachOverlayData(
            element_name_ru="двойной сальхов",
            metrics=[
                ("Время полёта", "0.37с", True),
                ("Высота", "0.12", True),
                ("Ось", "наклон 25°", False),
            ],
            recommendations=["Работай над осью"],
            landing_frame=100,
            fps=30.0,
            display_duration=4.0,
        )
        assert data.element_name_ru == "двойной сальхов"
        assert len(data.metrics) == 3
        assert data.landing_frame == 100

    def test_is_visible_at_frame(self):
        data = CoachOverlayData(
            element_name_ru="тест",
            metrics=[("м1", "1.0", True)],
            recommendations=[],
            landing_frame=100,
            fps=30.0,
            display_duration=4.0,
        )
        # Not visible before landing
        assert not data.is_visible_at(99)
        # Visible at landing
        assert data.is_visible_at(100)
        # Visible during display window (100 + 4.0*30 = 220)
        assert data.is_visible_at(219)
        # Not visible after display duration
        assert not data.is_visible_at(221)


class TestComputeCoachOverlays:
    """Test analysis pre-computation."""

    def test_compute_from_jump_phases(self):
        """Should produce overlay data from jump phases and metrics."""
        phases = ElementPhase(
            name="jump",
            start=250,
            takeoff=278,
            peak=283,
            landing=289,
            end=300,
        )
        metrics = [
            MetricResult(name="airtime", value=0.37, unit="s", is_good=True, reference_range=(0.3, 0.7)),
            MetricResult(name="max_height", value=0.12, unit="norm", is_good=False, reference_range=(0.2, 0.5)),
        ]
        recommendations = ["Недостаточная высота прыжка"]

        overlays = compute_coach_overlays(
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            element_type="waltz_jump",
            fps=29.9,
        )

        assert len(overlays) == 1
        assert overlays[0].landing_frame == 289
        assert overlays[0].element_name_ru == "вальсовый прыжок"
        assert len(overlays[0].metrics) == 2
        assert overlays[0].recommendations == recommendations

    def test_no_overlay_for_step(self):
        """Should produce no overlay when phases have no takeoff/landing (steps)."""
        phases = ElementPhase(
            name="three_turn",
            start=10,
            takeoff=0,
            peak=20,
            landing=0,
            end=30,
        )
        overlays = compute_coach_overlays(
            phases=phases,
            metrics=[],
            recommendations=[],
            element_type="three_turn",
            fps=30.0,
        )
        assert len(overlays) == 0
