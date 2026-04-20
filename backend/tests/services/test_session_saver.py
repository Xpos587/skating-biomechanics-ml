"""Tests for app.services.session_saver — save_analysis_results."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class FakeMetricResult:
    """Minimal stand-in for MetricResult from BiomechanicsAnalyzer."""

    name: str
    value: float


def _make_session(user_id: str = "user-1", element_type: str = "waltz_jump"):
    session = MagicMock()
    session.user_id = user_id
    session.element_type = element_type
    return session


def _make_metric_result(name: str, value: float):
    return FakeMetricResult(name=name, value=value)


# ---------------------------------------------------------------------------
# save_analysis_results — session not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_returns_early_when_session_not_found():
    """If get_by_id returns None, function returns without further DB calls."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()

    with (
        patch("app.services.session_saver.get_by_id", return_value=None) as mock_get,
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="nonexistent",
            metrics=[_make_metric_result("airtime", 0.5)],
            phases=None,
            recommendations=["test"],
        )

    mock_get.assert_called_once_with(db, "nonexistent")
    mock_bulk.assert_not_called()
    mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# save_analysis_results — happy path with metrics in range
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_happy_path():
    """All metrics in ideal range, no previous PRs."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    metrics = [
        _make_metric_result("airtime", 0.5),
        _make_metric_result("max_height", 0.3),
    ]

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="sess-1",
            metrics=metrics,
            phases=MagicMock(),
            recommendations=["Keep back straight"],
        )

    # bulk_create called once
    mock_bulk.assert_called_once()
    rows = mock_bulk.call_args[0][1]
    assert len(rows) == 2

    # Check first metric row
    row_airtime = rows[0]
    assert row_airtime["session_id"] == "sess-1"
    assert row_airtime["metric_name"] == "airtime"
    assert row_airtime["metric_value"] == 0.5
    assert row_airtime["is_pr"] is True  # No previous best -> is PR
    assert row_airtime["prev_best"] is None
    # airtime ideal_range=(0.3, 0.7), value=0.5 -> in range
    assert row_airtime["is_in_range"] is True
    assert row_airtime["reference_value"] == 0.3

    # Check second metric row
    row_height = rows[1]
    assert row_height["metric_name"] == "max_height"
    assert row_height["metric_value"] == 0.3
    assert row_height["is_pr"] is True
    assert row_height["reference_value"] == 0.2

    # update called with status, overall_score, recommendations
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs["status"] == "done"
    assert call_kwargs["recommendations"] == ["Keep back straight"]
    # Both in range -> score = 1.0
    assert call_kwargs["overall_score"] == 1.0


# ---------------------------------------------------------------------------
# save_analysis_results — metric outside range
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_metric_out_of_range():
    """Metric outside ideal range sets is_in_range=False."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    # rotation_speed ideal_range=(300, 550), value=200 -> outside range
    metrics = [_make_metric_result("rotation_speed", 200.0)]

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="sess-2",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    rows = mock_bulk.call_args[0][1]
    assert len(rows) == 1
    assert rows[0]["is_in_range"] is False
    assert rows[0]["is_pr"] is True  # No previous best

    # 0 out of 1 in range -> score = 0.0
    assert mock_update.call_args[1]["overall_score"] == 0.0


# ---------------------------------------------------------------------------
# save_analysis_results — PR detection with existing best
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_pr_detection_higher_direction():
    """'higher' direction metric: new value > best -> is PR."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    # airtime direction="higher", ideal_range=(0.3, 0.7)
    # current best=0.4, new value=0.6 -> PR (higher is better)
    metrics = [_make_metric_result("airtime", 0.6)]
    bests = {"airtime": 0.4}

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value=bests),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update"),
    ):
        await save_analysis_results(
            db,
            session_id="sess-3",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    rows = mock_bulk.call_args[0][1]
    assert rows[0]["is_pr"] is True
    assert rows[0]["prev_best"] == 0.4


@pytest.mark.asyncio
async def test_save_analysis_pr_detection_lower_direction():
    """'lower' direction metric: new value < best -> is PR."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    # landing_knee_angle direction="lower", ideal_range=(90, 130)
    # current best=110, new value=100 -> PR (lower is better)
    metrics = [_make_metric_result("landing_knee_angle", 100.0)]
    bests = {"landing_knee_angle": 110.0}

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value=bests),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update"),
    ):
        await save_analysis_results(
            db,
            session_id="sess-4",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    rows = mock_bulk.call_args[0][1]
    assert rows[0]["is_pr"] is True
    assert rows[0]["prev_best"] == 110.0


@pytest.mark.asyncio
async def test_save_analysis_no_pr_when_not_beating_best():
    """'higher' direction metric: new value <= best -> not PR."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    # airtime direction="higher", current best=0.6, new value=0.4 -> not PR
    metrics = [_make_metric_result("airtime", 0.4)]
    bests = {"airtime": 0.6}

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value=bests),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update"),
    ):
        await save_analysis_results(
            db,
            session_id="sess-5",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    rows = mock_bulk.call_args[0][1]
    assert rows[0]["is_pr"] is False
    assert rows[0]["prev_best"] is None


# ---------------------------------------------------------------------------
# save_analysis_results — unknown metric (not in registry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_unknown_metric_no_registry_entry():
    """Metric not in METRIC_REGISTRY: no range check, no reference_value."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    metrics = [_make_metric_result("unknown_metric", 42.0)]

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="sess-6",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    rows = mock_bulk.call_args[0][1]
    assert rows[0]["reference_value"] is None
    assert rows[0]["is_in_range"] is None
    # No registry entry -> direction defaults to "higher", no previous best -> PR
    assert rows[0]["is_pr"] is True

    # None is_in_range doesn't count toward score
    assert mock_update.call_args[1]["overall_score"] == 0.0


# ---------------------------------------------------------------------------
# save_analysis_results — empty metrics list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_empty_metrics():
    """No metrics -> overall_score=None, bulk_create still called with empty list."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}) as mock_batch,
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="sess-7",
            metrics=[],
            phases=None,
            recommendations=[],
        )

    mock_batch.assert_called_once_with(
        db, user_id="user-1", element_type="waltz_jump", metric_names=[]
    )
    mock_bulk.assert_called_once_with(db, [])
    assert mock_update.call_args[1]["overall_score"] is None


# ---------------------------------------------------------------------------
# save_analysis_results — mixed in-range / out-of-range
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_mixed_ranges_partial_score():
    """2 metrics: 1 in range, 1 out -> overall_score=0.5."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session()

    metrics = [
        _make_metric_result("airtime", 0.5),  # ideal_range=(0.3, 0.7) -> in range
        _make_metric_result("rotation_speed", 200.0),  # ideal_range=(300, 550) -> out of range
    ]

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}),
        patch("app.services.session_saver.bulk_create") as mock_bulk,
        patch("app.services.session_saver.update") as mock_update,
    ):
        await save_analysis_results(
            db,
            session_id="sess-8",
            metrics=metrics,
            phases=None,
            recommendations=["Improve rotation speed"],
        )

    rows = mock_bulk.call_args[0][1]
    assert rows[0]["is_in_range"] is True
    assert rows[1]["is_in_range"] is False

    # 1 out of 2 in range -> score = 0.5
    assert mock_update.call_args[1]["overall_score"] == 0.5
    assert mock_update.call_args[1]["recommendations"] == ["Improve rotation speed"]


# ---------------------------------------------------------------------------
# save_analysis_results — get_current_best_batch receives correct args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_analysis_batch_best_called_with_correct_args():
    """Verify batch query receives user_id, element_type, and metric_names."""
    from app.services.session_saver import save_analysis_results

    db = AsyncMock()
    session = _make_session(user_id="u-42", element_type="lutz")

    metrics = [
        _make_metric_result("airtime", 0.5),
        _make_metric_result("max_height", 0.3),
        _make_metric_result("symmetry", 0.8),
    ]

    with (
        patch("app.services.session_saver.get_by_id", return_value=session),
        patch("app.services.session_saver.get_current_best_batch", return_value={}) as mock_batch,
        patch("app.services.session_saver.bulk_create"),
        patch("app.services.session_saver.update"),
    ):
        await save_analysis_results(
            db,
            session_id="sess-9",
            metrics=metrics,
            phases=None,
            recommendations=[],
        )

    mock_batch.assert_called_once()
    call_kwargs = mock_batch.call_args[1]
    assert call_kwargs["user_id"] == "u-42"
    assert call_kwargs["element_type"] == "lutz"
    assert set(call_kwargs["metric_names"]) == {"airtime", "max_height", "symmetry"}
