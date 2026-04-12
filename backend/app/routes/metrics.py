"""Metrics, trend, PR, diagnostics, and registry API routes."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from backend.app.crud.relationship import is_coach_for_student
from backend.app.metrics_registry import METRIC_REGISTRY
from backend.app.models.session import Session, SessionMetric
from backend.app.schemas import (
    DiagnosticsFinding,
    DiagnosticsResponse,
    TrendDataPoint,
    TrendResponse,
)
from backend.app.services.diagnostics import (
    check_consistently_below_range,
    check_declining_trend,
    check_high_variability,
    check_new_pr,
    check_stagnation,
)

if TYPE_CHECKING:
    from backend.app.auth.deps import CurrentUser, DbDep

router = APIRouter(tags=["metrics"])


@router.get("/metrics/registry")
async def get_registry():
    """Static metric definitions for frontend."""
    return {
        name: {
            "name": m.name,
            "label_ru": m.label_ru,
            "unit": m.unit,
            "format": m.format,
            "direction": m.direction,
            "element_types": m.element_types,
            "ideal_range": list(m.ideal_range),
        }
        for name, m in METRIC_REGISTRY.items()
    }


@router.get("/metrics/trend", response_model=TrendResponse)
async def get_trend(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
    element_type: str = Query(..., min_length=1),
    metric_name: str = Query(..., min_length=1),
    period: str = Query("30d", pattern="^(7d|30d|90d|all)$"),
):
    target_user_id = user_id if user_id else user.id
    if (
        user_id
        and user_id != user.id
        and not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user"
        )

    mdef = METRIC_REGISTRY.get(metric_name)
    if not mdef:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown metric: {metric_name}"
        )

    # Calculate date filter
    now = datetime.now(UTC)
    period_map = {"7d": 7, "30d": 30, "90d": 90, "all": None}
    days = period_map.get(period)
    date_filter = Session.created_at >= (now - timedelta(days=days)) if days else True

    # Query data points
    query = (
        select(SessionMetric, Session.created_at, Session.id)
        .join(Session)
        .where(
            Session.user_id == target_user_id,
            Session.element_type == element_type,
            SessionMetric.metric_name == metric_name,
            Session.status == "done",
            date_filter,  # type: ignore[reportArgumentType]
        )
        .order_by(Session.created_at.asc())
    )
    result = await db.execute(query)
    rows = result.all()

    data_points = [
        TrendDataPoint(
            date=row.created_at.strftime("%Y-%m-%d"),
            value=row.SessionMetric.metric_value,
            session_id=row.id,
            is_pr=row.SessionMetric.is_pr,
        )
        for row in rows
    ]

    # Compute trend
    values = [dp.value for dp in data_points]
    trend = "stable"
    if len(values) >= 3:
        from backend.app.services.diagnostics import _linear_regression

        slope, r_sq = _linear_regression(values)
        if slope > 0 and r_sq > 0.3:
            trend = "improving"
        elif slope < 0 and r_sq > 0.3:
            trend = "declining"

    # Current PR
    pr_val = None
    for dp in reversed(data_points):
        if dp.is_pr:
            pr_val = dp.value
            break

    ref_range = {"min": mdef.ideal_range[0], "max": mdef.ideal_range[1]}

    return TrendResponse(
        metric_name=metric_name,
        element_type=element_type,
        data_points=data_points,
        trend=trend,
        current_pr=pr_val,
        reference_range=ref_range,
    )


@router.get("/metrics/prs")
async def get_prs(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
    element_type: str | None = None,
):
    """List all current personal records."""
    target_user_id = user_id if user_id else user.id
    if (
        user_id
        and user_id != user.id
        and not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user"
        )

    query = (
        select(SessionMetric, Session.element_type)
        .join(Session)
        .where(
            Session.user_id == target_user_id,
            Session.status == "done",
            SessionMetric.is_pr,
        )
    )
    if element_type:
        query = query.where(Session.element_type == element_type)

    result = await db.execute(query)
    rows = result.all()

    prs = []
    seen = set()
    for row in rows:
        key = (row.element_type, row.SessionMetric.metric_name)
        if key not in seen:
            seen.add(key)
            prs.append(
                {
                    "element_type": row.element_type,
                    "metric_name": row.SessionMetric.metric_name,
                    "value": row.SessionMetric.metric_value,
                    "session_id": row.SessionMetric.session_id,
                }
            )

    return {"prs": prs}


@router.get("/metrics/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
):
    """Run all diagnostic rules for a user."""
    target_user_id = user_id if user_id else user.id
    if (
        user_id
        and user_id != user.id
        and not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user"
        )

    findings: list[DiagnosticsFinding] = []

    # Group sessions by element type
    query = (
        select(SessionMetric, Session.element_type, Session.created_at, Session.id)
        .join(Session)
        .where(Session.user_id == target_user_id, Session.status == "done")
        .order_by(Session.element_type, Session.created_at.asc())
    )
    result = await db.execute(query)
    rows = result.all()

    by_element_metric: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        key = (row.element_type, row.SessionMetric.metric_name)
        by_element_metric[key].append(row)

    for (element, metric_name), metric_rows in by_element_metric.items():
        mdef = METRIC_REGISTRY.get(metric_name)
        if not mdef:
            continue

        values = [r.SessionMetric.metric_value for r in metric_rows]
        in_range_flags = [
            r.SessionMetric.is_in_range
            for r in metric_rows
            if r.SessionMetric.is_in_range is not None
        ]
        latest = metric_rows[-1]

        # Check rules
        f = check_consistently_below_range(
            element=element,
            metric=metric_name,
            in_range_flags=in_range_flags,
            metric_label=mdef.label_ru,
            ref_range=mdef.ideal_range,
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_declining_trend(
            element=element, metric=metric_name, values=values, metric_label=mdef.label_ru
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_stagnation(
            element=element, metric=metric_name, values=values, metric_label=mdef.label_ru
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_new_pr(
            element=element,
            metric=metric_name,
            is_latest_pr=latest.SessionMetric.is_pr,
            metric_label=mdef.label_ru,
            latest_value=latest.SessionMetric.metric_value,
            prev_best=latest.SessionMetric.prev_best,
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_high_variability(
            element=element, metric=metric_name, values=values, metric_label=mdef.label_ru
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

    # Sort: warnings first, then info
    findings.sort(key=lambda f: 0 if f.severity == "warning" else 1)

    return DiagnosticsResponse(user_id=target_user_id, findings=findings)
