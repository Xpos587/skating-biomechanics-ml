"""Tests for diagnostics engine."""

import pytest

from src.backend.services.diagnostics import (
    Finding,
    check_consistently_below_range,
    check_declining_trend,
    check_high_variability,
    check_new_pr,
    check_stagnation,
)


def test_consistently_below_range_triggers():
    """Warning when >60% of values are out of range."""
    values = [False, False, False, False, True]  # 4/5 below range
    finding = check_consistently_below_range(
        element="lutz", metric="landing_knee_stability",
        in_range_flags=values, metric_label="Стабильность приземления",
        ref_range=(0.5, 1.0),
    )
    assert finding is not None
    assert finding.severity == "warning"
    assert "4 из 5" in finding.message


def test_consistently_below_range_ok():
    """No warning when most values are in range."""
    values = [True, True, True, True, False]
    finding = check_consistently_below_range(
        element="lutz", metric="landing_knee_stability",
        in_range_flags=values, metric_label="Стабильность приземления",
        ref_range=(0.5, 1.0),
    )
    assert finding is None


def test_declining_trend():
    """Warning when slope is negative with good R²."""
    values = [0.50, 0.48, 0.45, 0.43, 0.40]
    finding = check_declining_trend(
        element="lutz", metric="landing_knee_stability",
        values=values, metric_label="Стабильность приземления",
    )
    assert finding is not None
    assert finding.severity == "warning"


def test_stagnation():
    """Info when values barely change."""
    values = [0.50, 0.51, 0.50, 0.51, 0.50]
    finding = check_stagnation(
        element="lutz", metric="airtime",
        values=values, metric_label="Время полёта",
    )
    assert finding is not None
    assert finding.severity == "info"


def test_new_pr():
    """Info when most recent is a PR."""
    finding = check_new_pr(
        element="lutz", metric="max_height",
        is_latest_pr=True, metric_label="Высота прыжка",
        latest_value=0.42, prev_best=0.38,
    )
    assert finding is not None
    assert finding.severity == "info"
    assert "PR" in finding.message


def test_high_variability():
    """Warning when coefficient of variation > 20%."""
    values = [0.30, 0.60, 0.25, 0.55, 0.35]
    finding = check_high_variability(
        element="lutz", metric="airtime",
        values=values, metric_label="Время полёта",
    )
    assert finding is not None
    assert finding.severity == "warning"
