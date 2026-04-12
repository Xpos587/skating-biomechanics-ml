"""Save ML pipeline results to Postgres.

Called after successful video processing to persist sessions and metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.crud.session import get_by_id, update
from backend.app.crud.session_metric import bulk_create, get_current_best
from backend.app.metrics_registry import METRIC_REGISTRY
from backend.app.services.pr_tracker import check_pr

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def save_analysis_results(
    db: AsyncSession,
    session_id: str,
    metrics: list,  # list[MetricResult]
    phases,  # ElementPhase
    recommendations: list[str],
) -> None:
    """Save analysis results to Postgres.

    Args:
        db: Async database session.
        session_id: ID of the Session row (created at upload time).
        metrics: List of MetricResult from BiomechanicsAnalyzer.
        phases: Detected ElementPhase.
        recommendations: Russian recommendation strings.
    """
    session = await get_by_id(db, session_id)
    if not session:
        return

    # Build metric rows with PR tracking
    metric_rows = []
    for mr in metrics:
        mdef = METRIC_REGISTRY.get(mr.name)
        ref_value = mdef.ideal_range[0] if mdef else None
        ref_max = mdef.ideal_range[1] if mdef else None

        is_in_range = None
        if mdef and ref_value is not None and ref_max is not None:
            is_in_range = ref_value <= mr.value <= ref_max

        # Check PR
        current_best = await get_current_best(
            db,
            user_id=session.user_id,
            element_type=session.element_type,
            metric_name=mr.name,
        )
        direction = mdef.direction if mdef else "higher"
        is_pr, prev_best = check_pr(direction, current_best, mr.value)

        metric_rows.append(
            {
                "session_id": session_id,
                "metric_name": mr.name,
                "metric_value": mr.value,
                "is_pr": is_pr,
                "prev_best": prev_best,
                "reference_value": ref_value,
                "is_in_range": is_in_range,
            }
        )

    await bulk_create(db, metric_rows)

    # Compute overall_score
    in_range_count = sum(1 for m in metric_rows if m["is_in_range"])
    overall_score = in_range_count / len(metric_rows) if metric_rows else None

    # Update session
    await update(
        db, session, status="done", overall_score=overall_score, recommendations=recommendations
    )
