"""SessionMetric CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models.session import Session, SessionMetric

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_current_best(
    db: AsyncSession,
    user_id: str,
    element_type: str,
    metric_name: str,
) -> float | None:
    """Get the current best value for a user+element+metric combination.

    Only considers non-deleted sessions. For "higher" direction metrics,
    this is the max value. For "lower", the min. The caller should pass
    the direction and handle accordingly.
    """
    query = (
        select(SessionMetric.metric_value)
        .join(Session)
        .where(
            Session.user_id == user_id,
            Session.element_type == element_type,
            SessionMetric.metric_name == metric_name,
            Session.status == "done",
        )
        .order_by(SessionMetric.metric_value.desc())
        .limit(1)
    )
    result = await db.execute(query)
    row = result.scalar_one_or_none()
    return row


async def get_current_best_batch(
    db: AsyncSession,
    user_id: str,
    element_type: str,
    metric_names: list[str],
) -> dict[str, float]:
    """Get current best values for multiple metrics in a single query.

    Returns dict mapping metric_name -> best_value (max for all metrics).
    Missing metrics (no data) are omitted from the dict.
    """
    if not metric_names:
        return {}

    from sqlalchemy import func

    subq = (
        select(
            SessionMetric.metric_name,
            func.max(SessionMetric.metric_value).label("best_value"),
        )
        .join(Session)
        .where(
            Session.user_id == user_id,
            Session.element_type == element_type,
            SessionMetric.metric_name.in_(metric_names),
            Session.status == "done",
        )
        .group_by(SessionMetric.metric_name)
    )
    result = await db.execute(subq)
    rows = result.all()
    return {row.metric_name: row.best_value for row in rows}


async def bulk_create(db: AsyncSession, metrics: list[dict]) -> None:
    """Insert multiple session metrics in one flush."""
    for m in metrics:
        db.add(SessionMetric(**m))
    await db.flush()
