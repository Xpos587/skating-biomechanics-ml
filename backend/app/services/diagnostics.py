"""Automatic diagnostic rules engine.

Runs simple statistical checks on session_metrics to surface patterns
for coaches: declining trends, stagnation, instability, PRs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    severity: str  # "warning" or "info"
    element: str
    metric: str
    message: str
    detail: str


def _linear_regression(values: list[float]) -> tuple[float, float]:
    """Return (slope, r_squared) for a simple linear regression."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n

    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    ss_yy = sum((yi - y_mean) ** 2 for yi in values)

    if ss_xx == 0:
        return 0.0, 0.0

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0
    return slope, r_squared


def check_consistently_below_range(
    *,
    element: str,
    metric: str,
    in_range_flags: list[bool],
    metric_label: str,
    ref_range: tuple[float, float],
) -> Finding | None:
    """Warning when >60% of values are out of ideal range."""
    if len(in_range_flags) < 3:
        return None
    below_count = sum(1 for f in in_range_flags if not f)
    total = len(in_range_flags)
    if below_count / total > 0.6:
        return Finding(
            severity="warning",
            element=element,
            metric=metric,
            message=f"{metric_label}: ниже нормы в {below_count} из {total} последних сессий",
            detail=f"Норма: {ref_range[0]}–{ref_range[1]}",
        )
    return None


def check_declining_trend(
    *,
    element: str,
    metric: str,
    values: list[float],
    metric_label: str,
) -> Finding | None:
    """Warning when linear regression shows decline with R² > 0.5."""
    if len(values) < 5:
        return None
    slope, r_squared = _linear_regression(values)
    if slope < 0 and r_squared > 0.5:
        return Finding(
            severity="warning",
            element=element,
            metric=metric,
            message=f"{metric_label}: ухудшается",
            detail=f"Тренд: declining (R²={r_squared:.2f})",
        )
    return None


def check_stagnation(
    *,
    element: str,
    metric: str,
    values: list[float],
    metric_label: str,
) -> Finding | None:
    """Info when standard deviation < 5% of mean."""
    if len(values) < 5:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    cv = std / abs(mean)
    if cv < 0.05:
        return Finding(
            severity="info",
            element=element,
            metric=metric,
            message=f"{metric_label}: нет улучшений за {len(values)} сессий",
            detail=f"Среднее: {mean:.3f}, CV: {cv:.1%}",
        )
    return None


def check_new_pr(
    *,
    element: str,
    metric: str,
    is_latest_pr: bool,
    metric_label: str,
    latest_value: float,
    prev_best: float | None,
) -> Finding | None:
    """Info when the most recent session is a PR."""
    if not is_latest_pr:
        return None
    prev_str = f"{prev_best:.3f}" if prev_best is not None else "—"
    return Finding(
        severity="info",
        element=element,
        metric=metric,
        message=f"Новый PR по {metric_label}!",
        detail=f"{latest_value:.3f} (предыдущий: {prev_str})",
    )


def check_high_variability(
    *,
    element: str,
    metric: str,
    values: list[float],
    metric_label: str,
) -> Finding | None:
    """Warning when coefficient of variation > 20%."""
    if len(values) < 5:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    cv = std / abs(mean)
    if cv > 0.20:
        return Finding(
            severity="warning",
            element=element,
            metric=metric,
            message=f"{metric_label}: сильно колеблется",
            detail=f"CV: {cv:.1%}, среднее: {mean:.3f}",
        )
    return None
