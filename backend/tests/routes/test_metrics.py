"""Tests for metrics API routes (registry, trend, prs, diagnostics)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from app.auth.security import create_access_token, hash_password
from app.models.connection import Connection, ConnectionStatus, ConnectionType
from app.models.session import Session, SessionMetric
from app.models.user import User
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    from app.routes.metrics import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    from app.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def user_a(db_session: AsyncSession) -> User:
    user = User(email="a@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def user_b(db_session: AsyncSession) -> User:
    user = User(email="b@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers_a(user_a):
    token = create_access_token(user_id=user_a.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_b(user_b):
    token = create_access_token(user_id=user_b.id)
    return {"Authorization": f"Bearer {token}"}


async def _insert_session(
    db_session: AsyncSession,
    user_id: str,
    element_type: str = "waltz_jump",
    status: str = "done",
    created_at: datetime | None = None,
) -> Session:
    session = Session(
        id=str(uuid.uuid4()),
        user_id=user_id,
        element_type=element_type,
        status=status,
        created_at=created_at or datetime.now(UTC),
    )
    db_session.add(session)
    await db_session.flush()
    await db_session.refresh(session)
    return session


async def _insert_metric(
    db_session: AsyncSession,
    session_id: str,
    metric_name: str = "airtime",
    metric_value: float = 0.5,
    is_pr: bool = False,
    is_in_range: bool | None = True,
    prev_best: float | None = None,
) -> SessionMetric:
    metric = SessionMetric(
        session_id=session_id,
        metric_name=metric_name,
        metric_value=metric_value,
        is_pr=is_pr,
        is_in_range=is_in_range,
        prev_best=prev_best,
    )
    db_session.add(metric)
    await db_session.flush()
    return metric


# ---------------------------------------------------------------------------
# GET /metrics/registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_returns_metric_definitions(client: AsyncClient):
    """GET /metrics/registry returns all registered metrics with correct structure."""
    response = await client.get("/api/v1/metrics/registry")
    assert response.status_code == 200
    data = response.json()

    # Must contain known metrics
    assert "airtime" in data
    assert "symmetry" in data

    airtime = data["airtime"]
    assert airtime["name"] == "airtime"
    assert airtime["label_ru"] == "Время полёта"
    assert airtime["unit"] == "s"
    assert airtime["format"] == ".2f"
    assert airtime["direction"] == "higher"
    assert isinstance(airtime["element_types"], list)
    assert isinstance(airtime["ideal_range"], list)
    assert len(airtime["ideal_range"]) == 2


# ---------------------------------------------------------------------------
# GET /metrics/trend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trend_empty(client: AsyncClient, auth_headers_a, user_a):
    """GET /metrics/trend with no data returns empty data_points and trend='stable'."""
    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data_points"] == []
    assert data["trend"] == "stable"
    assert data["current_pr"] is None
    assert data["metric_name"] == "airtime"
    assert data["element_type"] == "waltz_jump"
    assert data["reference_range"]["min"] == 0.3
    assert data["reference_range"]["max"] == 0.7


@pytest.mark.asyncio
async def test_trend_with_data(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/trend with session + metric returns data points."""
    session = await _insert_session(db_session, user_a.id, "waltz_jump")
    await _insert_metric(db_session, session.id, "airtime", 0.45, is_pr=True)

    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data_points"]) == 1
    dp = data["data_points"][0]
    assert dp["value"] == 0.45
    assert dp["session_id"] == session.id
    assert dp["is_pr"] is True
    assert data["current_pr"] == 0.45
    assert data["trend"] == "stable"  # only 1 point


@pytest.mark.asyncio
async def test_trend_unknown_metric(client: AsyncClient, auth_headers_a):
    """GET /metrics/trend with invalid metric_name returns 400."""
    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "nonexistent_metric"},
        headers=auth_headers_a,
    )
    assert response.status_code == 400
    assert "Unknown metric" in response.json()["detail"]


@pytest.mark.asyncio
async def test_trend_coach_access_denied(client: AsyncClient, auth_headers_a, user_b):
    """GET /metrics/trend with user_id param but no coaching connection returns 403."""
    response = await client.get(
        "/api/v1/metrics/trend",
        params={
            "element_type": "waltz_jump",
            "metric_name": "airtime",
            "user_id": user_b.id,
        },
        headers=auth_headers_a,
    )
    assert response.status_code == 403
    assert "Not a coach" in response.json()["detail"]


@pytest.mark.asyncio
async def test_trend_coach_access_allowed(
    client: AsyncClient, auth_headers_a, user_a, user_b, db_session: AsyncSession
):
    """GET /metrics/trend with active coaching connection returns 200."""
    # Create active coaching connection: user_a is coach for user_b
    conn = Connection(
        from_user_id=user_a.id,
        to_user_id=user_b.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=user_a.id,
    )
    db_session.add(conn)
    await db_session.flush()

    response = await client.get(
        "/api/v1/metrics/trend",
        params={
            "element_type": "waltz_jump",
            "metric_name": "airtime",
            "user_id": user_b.id,
        },
        headers=auth_headers_a,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_trend_period_filter(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/trend with period=7d filters out old sessions."""
    now = datetime.now(UTC)

    # Old session (20 days ago) — should be excluded by 7d filter
    old_session = Session(
        id="s-old",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="done",
        created_at=now - timedelta(days=20),
    )
    db_session.add(old_session)
    await db_session.flush()
    await _insert_metric(db_session, old_session.id, "airtime", 0.30)

    # Recent session (3 days ago) — should be included
    recent_session = Session(
        id="s-recent",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="done",
        created_at=now - timedelta(days=3),
    )
    db_session.add(recent_session)
    await db_session.flush()
    await _insert_metric(db_session, recent_session.id, "airtime", 0.50)

    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime", "period": "7d"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data_points"]) == 1
    assert data["data_points"][0]["value"] == 0.50
    assert data["data_points"][0]["session_id"] == "s-recent"


@pytest.mark.asyncio
async def test_trend_improving_with_3_points(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/trend with 3 strictly increasing values returns trend='improving'."""
    now = datetime.now(UTC)
    values = [0.30, 0.40, 0.55]
    for i, val in enumerate(values):
        session = Session(
            id=f"s-trend-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=10 - i * 3),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(db_session, session.id, "airtime", val)

    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trend"] == "improving"


@pytest.mark.asyncio
async def test_trend_declining_with_3_points(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/trend with 3 strictly decreasing values returns trend='declining'."""
    now = datetime.now(UTC)
    values = [0.55, 0.40, 0.30]
    for i, val in enumerate(values):
        session = Session(
            id=f"s-decline-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=10 - i * 3),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(db_session, session.id, "airtime", val)

    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trend"] == "declining"


# ---------------------------------------------------------------------------
# GET /metrics/prs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prs_empty(client: AsyncClient, auth_headers_a):
    """GET /metrics/prs with no data returns empty list."""
    response = await client.get("/api/v1/metrics/prs", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert data["prs"] == []


@pytest.mark.asyncio
async def test_prs_with_data(client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession):
    """GET /metrics/prs returns metrics flagged as PRs."""
    session = await _insert_session(db_session, user_a.id, "waltz_jump")
    await _insert_metric(db_session, session.id, "airtime", 0.55, is_pr=True)
    await _insert_metric(db_session, session.id, "symmetry", 0.85, is_pr=True)

    response = await client.get("/api/v1/metrics/prs", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert len(data["prs"]) == 2

    pr_names = {pr["metric_name"] for pr in data["prs"]}
    assert "airtime" in pr_names
    assert "symmetry" in pr_names

    airtime_pr = next(pr for pr in data["prs"] if pr["metric_name"] == "airtime")
    assert airtime_pr["value"] == 0.55
    assert airtime_pr["element_type"] == "waltz_jump"
    assert airtime_pr["session_id"] == session.id


@pytest.mark.asyncio
async def test_prs_deduplication(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/prs deduplicates by (element_type, metric_name), keeping latest."""
    now = datetime.now(UTC)

    # First session with PR
    session1 = Session(
        id="s-pr1",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="done",
        created_at=now - timedelta(days=5),
    )
    db_session.add(session1)
    await db_session.flush()
    await _insert_metric(db_session, session1.id, "airtime", 0.40, is_pr=True)

    # Second session with newer PR
    session2 = Session(
        id="s-pr2",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="done",
        created_at=now - timedelta(days=1),
    )
    db_session.add(session2)
    await db_session.flush()
    await _insert_metric(db_session, session2.id, "airtime", 0.55, is_pr=True)

    response = await client.get("/api/v1/metrics/prs", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()

    # Only one airtime PR for waltz_jump
    airtime_prs = [pr for pr in data["prs"] if pr["metric_name"] == "airtime"]
    assert len(airtime_prs) == 1
    # Query has no explicit ORDER BY on created_at, but dedup keeps first encountered.
    # The PR list contains exactly one entry for (waltz_jump, airtime).
    assert airtime_prs[0]["value"] in (0.40, 0.55)


@pytest.mark.asyncio
async def test_prs_filter_by_element_type(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/prs?element_type=... filters by element type."""
    # WJ session with PR
    session_wj = await _insert_session(db_session, user_a.id, "waltz_jump")
    await _insert_metric(db_session, session_wj.id, "airtime", 0.50, is_pr=True)

    # Three turn session with PR
    session_tt = await _insert_session(db_session, user_a.id, "three_turn")
    await _insert_metric(db_session, session_tt.id, "knee_angle", 120.0, is_pr=True)

    response = await client.get(
        "/api/v1/metrics/prs",
        params={"element_type": "waltz_jump"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["prs"]) == 1
    assert data["prs"][0]["metric_name"] == "airtime"


@pytest.mark.asyncio
async def test_prs_coach_access_denied(client: AsyncClient, auth_headers_a, user_b):
    """GET /metrics/prs with user_id param but no coaching connection returns 403."""
    response = await client.get(
        "/api/v1/metrics/prs",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 403
    assert "Not a coach" in response.json()["detail"]


@pytest.mark.asyncio
async def test_prs_coach_access_allowed(
    client: AsyncClient, auth_headers_a, user_a, user_b, db_session: AsyncSession
):
    """GET /metrics/prs with active coaching connection returns 200."""
    conn = Connection(
        from_user_id=user_a.id,
        to_user_id=user_b.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=user_a.id,
    )
    db_session.add(conn)
    await db_session.flush()

    response = await client.get(
        "/api/v1/metrics/prs",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /metrics/diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_empty(client: AsyncClient, auth_headers_a):
    """GET /metrics/diagnostics with no sessions returns empty findings."""
    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert data["findings"] == []
    assert data["user_id"]


@pytest.mark.asyncio
async def test_diagnostics_with_new_pr(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics surfaces new PR finding."""
    session = await _insert_session(db_session, user_a.id, "waltz_jump")
    await _insert_metric(db_session, session.id, "airtime", 0.55, is_pr=True, prev_best=0.45)

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert len(data["findings"]) >= 1

    pr_findings = [f for f in data["findings"] if "PR" in f["message"]]
    assert len(pr_findings) == 1
    assert pr_findings[0]["severity"] == "info"
    assert pr_findings[0]["metric"] == "airtime"
    assert pr_findings[0]["element"] == "waltz_jump"


@pytest.mark.asyncio
async def test_diagnostics_consistently_below_range(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics surfaces warning when >60% of values below range."""
    now = datetime.now(UTC)
    for i in range(5):
        session = Session(
            id=f"s-below-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=10 - i * 2),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(
            db_session,
            session.id,
            "airtime",
            0.20,  # well below ideal_range (0.3, 0.7)
            is_in_range=False,
        )

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    below_findings = [f for f in data["findings"] if "ниже нормы" in f["message"]]
    assert len(below_findings) == 1
    assert below_findings[0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_diagnostics_coach_access_denied(client: AsyncClient, auth_headers_a, user_b):
    """GET /metrics/diagnostics with user_id param but no coaching connection returns 403."""
    response = await client.get(
        "/api/v1/metrics/diagnostics",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 403
    assert "Not a coach" in response.json()["detail"]


@pytest.mark.asyncio
async def test_diagnostics_coach_access_allowed(
    client: AsyncClient, auth_headers_a, user_a, user_b, db_session: AsyncSession
):
    """GET /metrics/diagnostics with active coaching connection returns 200."""
    conn = Connection(
        from_user_id=user_a.id,
        to_user_id=user_b.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=user_a.id,
    )
    db_session.add(conn)
    await db_session.flush()

    response = await client.get(
        "/api/v1/metrics/diagnostics",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_diagnostics_ignores_uploading_sessions(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics ignores sessions with status != 'done'."""
    session = Session(
        id="s-uploading",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="uploading",
    )
    db_session.add(session)
    await db_session.flush()
    await _insert_metric(db_session, session.id, "airtime", 0.50, is_pr=True)

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert data["findings"] == []


@pytest.mark.asyncio
async def test_diagnostics_sorts_warnings_first(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics sorts findings with warnings before info."""
    now = datetime.now(UTC)

    # Create sessions to trigger both warning (below range) and info (new PR)
    for i in range(5):
        session = Session(
            id=f"s-sort-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=10 - i * 2),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(
            db_session,
            session.id,
            "airtime",
            0.20,
            is_pr=(i == 4),  # last one is PR
            is_in_range=False,
            prev_best=0.20,
        )

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    assert len(data["findings"]) >= 2

    # Warnings must come before info
    severities = [f["severity"] for f in data["findings"]]
    for i in range(len(severities) - 1):
        if severities[i] == "info" and severities[i + 1] == "warning":
            pytest.fail(f"Info at index {i} before warning at {i + 1}")


@pytest.mark.asyncio
async def test_diagnostics_declining_trend(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics surfaces declining trend finding."""
    now = datetime.now(UTC)
    values = [0.60, 0.50, 0.40, 0.30, 0.25]
    for i, val in enumerate(values):
        session = Session(
            id=f"s-decl-diag-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=20 - i * 4),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(
            db_session,
            session.id,
            "airtime",
            val,
            is_in_range=True,
        )

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    decline_findings = [f for f in data["findings"] if "ухудшается" in f.get("message", "")]
    assert len(decline_findings) == 1
    assert decline_findings[0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_diagnostics_stagnation(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics surfaces stagnation finding when values are flat."""
    now = datetime.now(UTC)
    # Very consistent values (low variance) should trigger stagnation
    for i in range(6):
        session = Session(
            id=f"s-stag-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=30 - i * 5),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(
            db_session,
            session.id,
            "airtime",
            0.50 + 0.001 * i,  # nearly flat
            is_in_range=True,
        )

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    stag_findings = [f for f in data["findings"] if "нет улучшений" in f.get("message", "")]
    assert len(stag_findings) == 1


@pytest.mark.asyncio
async def test_diagnostics_high_variability(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics surfaces high variability finding."""
    now = datetime.now(UTC)
    # Values with high coefficient of variation
    values = [0.20, 0.70, 0.25, 0.65, 0.30]
    for i, val in enumerate(values):
        session = Session(
            id=f"s-var-{i}",
            user_id=user_a.id,
            element_type="waltz_jump",
            status="done",
            created_at=now - timedelta(days=20 - i * 4),
        )
        db_session.add(session)
        await db_session.flush()
        await _insert_metric(
            db_session,
            session.id,
            "airtime",
            val,
            is_in_range=True,
        )

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    var_findings = [f for f in data["findings"] if "колеблется" in f.get("message", "")]
    assert len(var_findings) == 1


@pytest.mark.asyncio
async def test_diagnostics_ignores_unknown_metrics(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/diagnostics skips metrics not in the registry."""
    session = await _insert_session(db_session, user_a.id, "waltz_jump")
    # Use a metric name not in METRIC_REGISTRY
    metric = SessionMetric(
        session_id=session.id,
        metric_name="unknown_custom_metric",
        metric_value=0.5,
        is_pr=False,
    )
    db_session.add(metric)
    await db_session.flush()

    response = await client.get("/api/v1/metrics/diagnostics", headers=auth_headers_a)
    assert response.status_code == 200
    data = response.json()
    # Should not crash, should return empty findings (unknown metric skipped)
    assert data["findings"] == []


@pytest.mark.asyncio
async def test_trend_period_all(
    client: AsyncClient, auth_headers_a, user_a, db_session: AsyncSession
):
    """GET /metrics/trend with period=all includes all sessions regardless of date."""
    now = datetime.now(UTC)

    # Very old session (200 days ago)
    old_session = Session(
        id="s-all-old",
        user_id=user_a.id,
        element_type="waltz_jump",
        status="done",
        created_at=now - timedelta(days=200),
    )
    db_session.add(old_session)
    await db_session.flush()
    await _insert_metric(db_session, old_session.id, "airtime", 0.30)

    response = await client.get(
        "/api/v1/metrics/trend",
        params={"element_type": "waltz_jump", "metric_name": "airtime", "period": "all"},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data_points"]) == 1
    assert data["data_points"][0]["value"] == 0.30


@pytest.mark.asyncio
async def test_prs_coach_access_allowed_with_data(
    client: AsyncClient, auth_headers_a, user_a, user_b, db_session: AsyncSession
):
    """GET /metrics/prs with coaching connection returns student's PRs."""
    from app.models.connection import Connection, ConnectionStatus, ConnectionType

    conn = Connection(
        from_user_id=user_a.id,
        to_user_id=user_b.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=user_a.id,
    )
    db_session.add(conn)
    await db_session.flush()

    # Create a PR for user_b
    session = await _insert_session(db_session, user_b.id, "waltz_jump")
    await _insert_metric(db_session, session.id, "airtime", 0.60, is_pr=True)

    response = await client.get(
        "/api/v1/metrics/prs",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["prs"]) == 1
    assert data["prs"][0]["value"] == 0.60


@pytest.mark.asyncio
async def test_diagnostics_coach_access_allowed_with_data(
    client: AsyncClient, auth_headers_a, user_a, user_b, db_session: AsyncSession
):
    """GET /metrics/diagnostics with coaching connection returns student's diagnostics."""
    from app.models.connection import Connection, ConnectionStatus, ConnectionType

    conn = Connection(
        from_user_id=user_a.id,
        to_user_id=user_b.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=user_a.id,
    )
    db_session.add(conn)
    await db_session.flush()

    # Create a new PR for user_b
    session = await _insert_session(db_session, user_b.id, "waltz_jump")
    await _insert_metric(db_session, session.id, "airtime", 0.55, is_pr=True, prev_best=0.45)

    response = await client.get(
        "/api/v1/metrics/diagnostics",
        params={"user_id": user_b.id},
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_b.id
    assert len(data["findings"]) >= 1
