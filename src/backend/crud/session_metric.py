"""SessionMetric CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from src.backend.models.session import SessionMetric

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_current_best(
    db: AsyncSession, user_id: str, element_type: str, metric_name: str,
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


async def bulk_create(db: AsyncSession, metrics: list[dict]) -> None:
    """Insert multiple session metrics in one flush."""
    for m in metrics:
        db.add(SessionMetric(**m))
    await db.flush()
