"""Direct handler tests for metrics routes (coverage-tracked)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.metrics import get_diagnostics, get_prs, get_registry, get_trend
from fastapi import HTTPException
from sqlalchemy.engine.result import Result

# ---------------------------------------------------------------------------
# get_registry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_registry_returns_all_metrics():
    result = await get_registry()
    assert isinstance(result, dict)
    assert "airtime" in result
    assert "symmetry" in result
    entry = result["airtime"]
    assert entry["label_ru"] == "Время полёта"
    assert entry["unit"] == "s"
    assert entry["direction"] == "higher"
    assert entry["ideal_range"] == [0.3, 0.7]


# ---------------------------------------------------------------------------
# get_trend
# ---------------------------------------------------------------------------


def _mock_user(user_id: str = "user_1") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    return u


def _mock_row(
    created_at: datetime | None = None,
    session_id: str = "sess_1",
    metric_value: float = 0.5,
    is_pr: bool = False,
    element_type: str = "waltz_jump",
    metric_name: str = "airtime",
    is_in_range: bool | None = True,
    prev_best: float | None = None,
    reference_value: float | None = None,
) -> MagicMock:
    row = MagicMock()
    row.created_at = created_at or datetime.now(UTC)
    row.id = session_id
    row.element_type = element_type
    sm = MagicMock()
    sm.metric_value = metric_value
    sm.metric_name = metric_name
    sm.is_pr = is_pr
    sm.is_in_range = is_in_range
    sm.prev_best = prev_best
    sm.reference_value = reference_value
    row.SessionMetric = sm
    return row


def _mock_db_with_rows(rows: list) -> AsyncMock:
    """Create a mock db where db.execute(query) returns a result whose .all() yields rows."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    mock_db = AsyncMock()
    # db.execute is async, returns the sync mock_result
    mock_db.execute.return_value = mock_result
    return mock_db


@pytest.mark.anyio
async def test_get_trend_own_data():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_value=0.4, is_pr=True),
            _mock_row(metric_value=0.5, is_pr=False),
        ]
    )

    result = await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="airtime")

    assert result.metric_name == "airtime"
    assert result.element_type == "waltz_jump"
    assert len(result.data_points) == 2
    assert result.trend == "stable"
    assert result.current_pr == 0.4
    assert result.reference_range == {"min": 0.3, "max": 0.7}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "values,expected_trend",
    [
        pytest.param([0.30, 0.45, 0.60, 0.70], "improving", id="improving"),
        pytest.param([0.70, 0.50, 0.30], "declining", id="declining"),
    ],
)
async def test_get_trend_direction(values, expected_trend):
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows([_mock_row(metric_value=v) for v in values])

    result = await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="airtime")

    assert result.trend == expected_trend


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler,call_kwargs,assert_fn",
    [
        pytest.param(
            get_trend,
            {"user_id": "student_1", "element_type": "waltz_jump", "metric_name": "airtime"},
            lambda r: r.metric_name == "airtime",
            id="get_trend",
        ),
        pytest.param(
            get_prs,
            {"user_id": "student_1"},
            lambda r: "prs" in r,
            id="get_prs",
        ),
        pytest.param(
            get_diagnostics,
            {"user_id": "student_1"},
            lambda r: r.user_id == "student_1",
            id="get_diagnostics",
        ),
    ],
)
async def test_coach_access_allowed(handler, call_kwargs, assert_fn):
    mock_user = _mock_user("coach_1")
    mock_db = _mock_db_with_rows([])

    with patch("app.routes.metrics.is_connected_as", new_callable=AsyncMock, return_value=True):
        result = await handler(mock_user, mock_db, **call_kwargs)

    assert assert_fn(result)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler,call_kwargs",
    [
        pytest.param(
            get_trend,
            {"user_id": "student_1", "element_type": "waltz_jump", "metric_name": "airtime"},
            id="get_trend",
        ),
        pytest.param(
            get_prs,
            {"user_id": "student_1"},
            id="get_prs",
        ),
        pytest.param(
            get_diagnostics,
            {"user_id": "student_1"},
            id="get_diagnostics",
        ),
    ],
)
async def test_coach_access_denied(handler, call_kwargs):
    mock_user = _mock_user("coach_1")
    mock_db = AsyncMock()

    with patch("app.routes.metrics.is_connected_as", new_callable=AsyncMock, return_value=False):
        with pytest.raises(HTTPException, match="Not a coach"):
            await handler(mock_user, mock_db, **call_kwargs)


@pytest.mark.anyio
async def test_get_trend_unknown_metric():
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with pytest.raises(HTTPException, match="Unknown metric"):
        await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="nonexistent")


@pytest.mark.anyio
async def test_get_trend_empty_data():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows([])

    result = await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="airtime")

    assert result.data_points == []
    assert result.trend == "stable"
    assert result.current_pr is None


@pytest.mark.anyio
async def test_get_trend_pr_found_in_reverse():
    """Scans in reverse: the latest PR (closest to end) is returned as current_pr."""
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_value=0.6, is_pr=True),
            _mock_row(metric_value=0.4, is_pr=False),
            _mock_row(metric_value=0.5, is_pr=False),
        ]
    )

    result = await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="airtime")

    # Reverse scan finds PR at index 0 (the only PR)
    assert result.current_pr == 0.6


@pytest.mark.anyio
async def test_get_trend_no_pr_at_all():
    """No data points with is_pr=True -> current_pr is None."""
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_value=0.6, is_pr=False),
            _mock_row(metric_value=0.4, is_pr=False),
            _mock_row(metric_value=0.5, is_pr=False),
        ]
    )

    result = await get_trend(mock_user, mock_db, element_type="waltz_jump", metric_name="airtime")

    assert result.current_pr is None


@pytest.mark.anyio
async def test_get_trend_period_filtering():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows([])

    result = await get_trend(
        mock_user, mock_db, element_type="waltz_jump", metric_name="airtime", period="90d"
    )

    mock_db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# get_prs
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_prs_own():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                metric_value=0.7, is_pr=True, metric_name="airtime", element_type="waltz_jump"
            ),
            _mock_row(
                metric_value=0.8, is_pr=True, metric_name="max_height", element_type="toe_loop"
            ),
        ]
    )

    result = await get_prs(mock_user, mock_db)

    assert "prs" in result
    assert len(result["prs"]) == 2


@pytest.mark.anyio
async def test_get_prs_deduplication():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_value=0.5, metric_name="airtime", element_type="waltz_jump"),
            _mock_row(metric_value=0.7, metric_name="airtime", element_type="waltz_jump"),
        ]
    )

    result = await get_prs(mock_user, mock_db)

    assert len(result["prs"]) == 1
    assert result["prs"][0]["value"] == 0.5


@pytest.mark.anyio
async def test_get_prs_filter_element_type():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_name="airtime", element_type="waltz_jump"),
        ]
    )

    result = await get_prs(mock_user, mock_db, element_type="waltz_jump")

    assert len(result["prs"]) == 1


@pytest.mark.anyio
async def test_get_prs_empty():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows([])

    result = await get_prs(mock_user, mock_db)

    assert result["prs"] == []


# ---------------------------------------------------------------------------
# get_diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_diagnostics_own():
    mock_user = _mock_user()
    now = datetime.now(UTC)
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                created_at=now - timedelta(days=5),
                metric_value=0.2,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=3),
                metric_value=0.25,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=1),
                metric_value=0.28,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=True,
                prev_best=0.25,
            ),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    assert result.user_id == "user_1"
    assert isinstance(result.findings, list)


@pytest.mark.anyio
async def test_get_diagnostics_empty():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows([])

    result = await get_diagnostics(mock_user, mock_db)

    assert result.findings == []


@pytest.mark.anyio
async def test_get_diagnostics_unknown_metric_skipped():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(metric_name="nonexistent_metric", element_type="waltz_jump"),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    assert result.findings == []


@pytest.mark.anyio
async def test_get_diagnostics_stagnation_and_new_pr():
    """5+ nearly identical values triggers stagnation; latest PR triggers new_pr."""
    mock_user = _mock_user()
    now = datetime.now(UTC)
    # 5 very similar values -> stagnation (CV < 5%)
    # Latest is PR -> new PR
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                created_at=now - timedelta(days=10),
                metric_value=0.50,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.50,
            ),
            _mock_row(
                created_at=now - timedelta(days=8),
                metric_value=0.51,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.50,
            ),
            _mock_row(
                created_at=now - timedelta(days=6),
                metric_value=0.50,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.50,
            ),
            _mock_row(
                created_at=now - timedelta(days=4),
                metric_value=0.49,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.50,
            ),
            _mock_row(
                created_at=now - timedelta(days=2),
                metric_value=0.50,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=True,
                prev_best=0.49,
            ),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    assert result.user_id == "user_1"
    messages = [f.message for f in result.findings]
    assert any("PR" in m for m in messages), f"Expected PR finding, got: {messages}"
    assert any("нет улучшений" in m for m in messages), (
        f"Expected stagnation finding, got: {messages}"
    )


@pytest.mark.anyio
async def test_get_diagnostics_declining_trend():
    """5+ strictly declining values triggers declining_trend."""
    mock_user = _mock_user()
    now = datetime.now(UTC)
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                created_at=now - timedelta(days=10),
                metric_value=0.80,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.80,
            ),
            _mock_row(
                created_at=now - timedelta(days=8),
                metric_value=0.70,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.80,
            ),
            _mock_row(
                created_at=now - timedelta(days=6),
                metric_value=0.60,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.80,
            ),
            _mock_row(
                created_at=now - timedelta(days=4),
                metric_value=0.50,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.80,
            ),
            _mock_row(
                created_at=now - timedelta(days=2),
                metric_value=0.40,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.80,
            ),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    messages = [f.message for f in result.findings]
    assert any("ухудшается" in m for m in messages), f"Expected declining finding, got: {messages}"


@pytest.mark.anyio
async def test_get_diagnostics_high_variability():
    """5+ values with high CV triggers high_variability."""
    mock_user = _mock_user()
    now = datetime.now(UTC)
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                created_at=now - timedelta(days=10),
                metric_value=0.1,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=8),
                metric_value=0.9,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=6),
                metric_value=0.1,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=4),
                metric_value=0.95,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                created_at=now - timedelta(days=2),
                metric_value=0.5,
                metric_name="symmetry",
                element_type="waltz_jump",
                is_in_range=True,
                is_pr=False,
                prev_best=0.5,
            ),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    messages = [f.message for f in result.findings]
    assert any("колеблется" in m for m in messages), (
        f"Expected variability finding, got: {messages}"
    )


@pytest.mark.anyio
async def test_get_diagnostics_sorts_warnings_first():
    mock_user = _mock_user()
    mock_db = _mock_db_with_rows(
        [
            _mock_row(
                metric_value=0.1,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                metric_value=0.1,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=False,
                prev_best=0.5,
            ),
            _mock_row(
                metric_value=0.1,
                metric_name="airtime",
                element_type="waltz_jump",
                is_in_range=False,
                is_pr=False,
                prev_best=0.5,
            ),
        ]
    )

    result = await get_diagnostics(mock_user, mock_db)

    if result.findings:
        severities = [f.severity for f in result.findings]
        assert severities == sorted(severities, key=lambda s: 0 if s == "warning" else 1)
